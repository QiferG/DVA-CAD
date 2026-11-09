import os
import json
from functools import partial
from argparse import ArgumentParser
import random
import numpy as np
import inspect
import torch
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from dvacad import DVACAD, collate
from dataset import CadEditDataset, CadEditDataset_query

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None


def align_model_tokenizer_config(model, processor):
    tokenizer = processor.tokenizer
    token_fields = ("pad_token_id", "eos_token_id", "bos_token_id")
    for field in token_fields:
        value = getattr(tokenizer, field, None)
        setattr(model.config, field, value)
        if getattr(model, "generation_config", None) is not None:
            setattr(model.generation_config, field, value)


def patch_accelerate_unwrap_model():
    if Accelerator is None:
        return
    signature = inspect.signature(Accelerator.unwrap_model)
    if 'keep_torch_compile' in signature.parameters:
        return

    original_unwrap_model = Accelerator.unwrap_model

    def unwrap_model_compat(self, model, keep_fp32_wrapper=True, keep_torch_compile=None):
        return original_unwrap_model(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

    Accelerator.unwrap_model = unwrap_model_compat

class PrintToFileCallback(TrainerCallback):
    def on_init_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            os.makedirs(args.logging_dir, exist_ok=True)

    def on_log(self, args, state, control, logs, **kwargs):
        if state.is_world_process_zero:
            with open(os.path.join(args.logging_dir, 'log.txt'), 'a') as f:
                f.write(str(logs) + '\n')


def prune_training_state_artifacts(save_dir):
    if not os.path.isdir(save_dir):
        return

    removable_files = {
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
    removable_prefixes = (
        "rng_state",
        "optimizer.bin",
        "scheduler.bin",
    )

    for name in os.listdir(save_dir):
        path = os.path.join(save_dir, name)
        if not os.path.isfile(path):
            continue
        if name in removable_files or name.startswith(removable_prefixes):
            os.remove(path)


class KeepInferenceFilesOnlyCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        prune_training_state_artifacts(checkpoint_dir)


class SeqEditorTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aux_log_buffer = {
            'lm_ce_loss': [],
            'ot_contrastive_loss': [],
            'weighted_ot_contrastive_loss': [],
        }

    def _append_aux_metric(self, name, value):
        if value is None:
            return
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() != 1:
                return
            value = float(value.item())
        else:
            value = float(value)
        self._aux_log_buffer[name].append(value)

    def _get_model_metric(self, model, name):
        candidates = []
        if model is not None:
            candidates.append(model)
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            try:
                candidates.append(accelerator.unwrap_model(model))
            except Exception:
                pass

        visited = set()
        while candidates:
            current = candidates.pop(0)
            if current is None or id(current) in visited:
                continue
            visited.add(id(current))

            value = getattr(current, name, None)
            if value is not None:
                return value

            for wrapper_attr in ("module", "_orig_mod"):
                wrapped = getattr(current, wrapper_attr, None)
                if wrapped is not None and id(wrapped) not in visited:
                    candidates.append(wrapped)

        return None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        self._append_aux_metric('lm_ce_loss', self._get_model_metric(model, 'last_lm_ce_loss'))
        self._append_aux_metric(
            'ot_contrastive_loss',
            self._get_model_metric(model, 'last_ot_contrastive_loss'),
        )
        self._append_aux_metric(
            'weighted_ot_contrastive_loss',
            self._get_model_metric(model, 'last_weighted_ot_contrastive_loss'),
        )
        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs, *args, **kwargs):
        for key, values in self._aux_log_buffer.items():
            if values:
                logs[key] = sum(values) / len(values)
        for key in self._aux_log_buffer:
            self._aux_log_buffer[key].clear()
        return super().log(logs, *args, **kwargs)

class TextSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        Args:
            data (list): List of dictionaries containing the training samples.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data sample.

        Returns:
            dict: A dictionary containing 'description' and 'answer'.
        """
        sample = self.data[idx]
        description = f"Instruction: {sample['instruction']} | Original Sequence: {sample['original_sequence']}"
        answer = sample['edited_sequence']
        return {"description": description, "answer": answer}


def train_model(
    data_path,
    log_path,
    ot_contrastive_weight=0.05,
    representation="command",
    init_checkpoint=None,
    max_steps=60000,
    learning_rate=1e-4,
    global_batch_size=32,
    per_device_batch_size=1,
    prediction_mesh_scale=None,
    model_path=None,
    processor_path=None,
    mesh_root=None,
):
    if not model_path:
        raise ValueError("Pass model_path (--model-path) for the base Qwen model")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    n_points = 2048

    dataset_cls = CadEditDataset_query if representation == "cadquery" else CadEditDataset
    train_dataset = dataset_cls(
        root_dir=data_path,
        mesh_root=mesh_root,
        split='train',
        n_points=n_points,
        normalize_std_pc=100,
        noise_scale_pc=None)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    batch_per_microstep = per_device_batch_size * world_size
    if global_batch_size % batch_per_microstep:
        raise ValueError("Global batch size must be divisible by per-device batch size times world size")
    accumulation_steps = global_batch_size // batch_per_microstep
    if accumulation_steps < 1:
        raise ValueError("Global batch size is smaller than one distributed microbatch")
    print(f"SFT effective batch={global_batch_size}: {per_device_batch_size} x {world_size} GPUs x {accumulation_steps} accumulation")

    processor = AutoProcessor.from_pretrained(
        processor_path or model_path,
        padding_side='left',
        local_files_only=True
    )

    if not processor.chat_template:
        processor.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n'}}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for block in message['content'] %}{% if block['type'] == 'text' %}{{ block['text'] }}{% endif %}{% endfor %}{% endif %}{{ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    model = DVACAD.from_pretrained(
        init_checkpoint or model_path,
        dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        local_files_only=True,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    align_model_tokenizer_config(model, processor)
    model.config.ot_contrastive_weight = ot_contrastive_weight
    model.ot_contrastive_weight = ot_contrastive_weight
    model.config.cad_representation = representation
    model.config.prediction_mesh_scale = (
        prediction_mesh_scale if prediction_mesh_scale is not None
        else 0.01 if representation == "cadquery" else 1.0
    )
    model.config.use_cache = False
    model = model.to(torch.device("cuda", local_rank))

    training_args_kwargs = dict(
        output_dir=log_path,
        per_device_train_batch_size=per_device_batch_size,
        dataloader_num_workers=8,
        max_steps=max_steps,
        bf16=True,
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        learning_rate=learning_rate,
        warmup_steps=2000,
        weight_decay=0.01,
        gradient_accumulation_steps=accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=True,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        logging_steps=10,
        logging_nan_inf_filter=False,
        save_total_limit=20,
        save_strategy='steps',
        save_steps=5000,
        report_to='none',
    )
    if "save_only_model" in inspect.signature(TrainingArguments.__init__).parameters:
        training_args_kwargs["save_only_model"] = True

    trainer = SeqEditorTrainer(
        model=model,
        args=TrainingArguments(**training_args_kwargs),
        train_dataset=train_dataset,
        data_collator=partial(
            collate,
            processor=processor,
            n_points=n_points,
        ),
        tokenizer=processor,
        callbacks=[PrintToFileCallback(), KeepInferenceFilesOnlyCallback()]
    )

    trainer.train()

    model_save_path = os.path.join(log_path, "seqeditor_model")
    trainer.save_model(model_save_path)
    processor.save_pretrained(model_save_path)
    prune_training_state_artifacts(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Base Qwen model directory")
    parser.add_argument("--processor-path", default=None, help="Processor directory; defaults to --model-path")
    parser.add_argument("--data-path", type=str, required=True, help="Training data directory")
    parser.add_argument("--mesh-root", default=None, help="Mesh directory; defaults to DATA_PATH/mesh")
    parser.add_argument("--log-path", type=str, default="saved/dvacad-sft", help="Output directory for logs and checkpoints")
    parser.add_argument("--ot-alignment-weight", "--ot-contrastive-weight", dest="ot_contrastive_weight", type=float, default=0.05)
    parser.add_argument("--representation", choices=["command", "cadquery"], default="command")
    parser.add_argument("--init-checkpoint", default=None, help="Optional compatible-weight warm start; starts a fresh SFT optimizer")
    parser.add_argument("--max-steps", type=int, default=60000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--prediction-mesh-scale", type=float, default=None,
                        help="Renderer units to dataset STL units; default 1 for commands, 0.01 for existing CadQuery data")
    args = parser.parse_args()
    patch_accelerate_unwrap_model()
    train_model(
        args.data_path,
        args.log_path,
        args.ot_contrastive_weight,
        args.representation,
        args.init_checkpoint,
        args.max_steps,
        args.learning_rate,
        args.global_batch_size,
        args.per_device_batch_size,
        args.prediction_mesh_scale,
        model_path=args.model_path,
        processor_path=args.processor_path,
        mesh_root=args.mesh_root,
    )
