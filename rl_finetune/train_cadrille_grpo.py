import os
import sys
from pathlib import Path
from typing import Optional

from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.distributed as dist
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import partial

import pyrallis
import wandb

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoConfig, AutoProcessor

from dvacad import DVACAD, collate
from rl_finetune.grpo_mm import train_with_grpo_mm
from rl_finetune.utils_dvacad import get_metrics_from_texts, localized_reward
from dataset import CadEditDataset, CadEditDataset_query

os.environ["PYGLET_HEADLESS"] = "True"

@dataclass
class TrainConfig:
    sft_path: str = ""
    data_path: str = ""
    processor_path: Optional[str] = None
    mesh_root: Optional[str] = None
    representation: str = "command"
    n_points: int = 2048
    prediction_mesh_scale: Optional[float] = None

    project: str = "CAD_test"
    group: str = "Dr-CCPO"
    name: str = "dr-ccpo"
    save_path: str = "saved/dvacad-rl"

    failure_reward: float = -3

    freeze_pc: bool = False
    train_epochs: int = 20
    batch_size: int = 4
    max_optimizer_steps: int = 2000
    save_mid_epoch: bool = True
    save_every_steps: int = 25
    save_every_optimizer_steps: int = 0

    epoch_save: int = 1

    temperature: float = 0.8
    do_sample: bool = True
    top_p: float = 0.95
    top_k: int = 50
    # GRPO params
    # how many outputs the old policy generates
    num_generations: int = 6
    # how many samples to choose for loss computation
    top_samples: int = 6
    max_completion_length: int = 2048
    learning_rate: float = 1e-5
    batch_updates: int = 1
    gradient_accumulation_steps: int = 1
    kl_coef: float = 0.3
    # clipping epsilon values
    epsilon_high: float = 0.1
    epsilon_low: float = 0.1

    pool_size : int = 4
    dataloader_workers : int = 4
    reward_timeout: int = 60
    find_unused_parameters: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True

    # Eqs. (10)-(18): occupancy/TSDF mask and point-cloud rewards.
    lambda_edit: float = 1.5
    lambda_stable: float = 1.0
    grid_resolution: int = 64
    tau_phi: float = 0.01


def get_reward_function(
    failure_reward,
    lambda_edit=1.5,
    lambda_stable=1.0,
    grid_resolution=64,
    tau_phi=0.01,
    max_workers: Optional[int] = None,
    timeout: int = 60,
    prediction_scale: float = 1.0,
    mesh_root: Optional[str] = None,
):
    def combined_reward(completions, answer, originals):
        rewards = []
        pred_metrics = get_metrics_from_texts(
            completions,
            answer,
            originals,
            max_workers=max_workers,
            timeout=timeout,
            grid_resolution=grid_resolution,
            tau_phi=tau_phi,
            prediction_scale=prediction_scale,
            mesh_root=mesh_root,
        )
        for m in pred_metrics:
            if m is not None and m.get("error_type") in ("reference", "dependency", "worker"):
                raise RuntimeError(f"LFM {m['error_type']} error: {m.get('error')}")
            edit_loss = m.get("reward_local") if m is not None else None
            stable_loss = m.get("reward_stable") if m is not None else None
            reward = localized_reward(edit_loss, stable_loss, lambda_edit, lambda_stable, failure_reward)
            rewards.append(reward)
        return rewards
    return combined_reward


def optimize_model_memory(model):
    """
    Optimizes the model to use less memory during training.
    """
    model.train()
    model.config.use_cache = False

    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Use the non-reentrant implementation once. Re-enabling checkpointing
    # later with different settings can register incompatible backward hooks.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    return model


def setup(world_size):
    """ Initialize the process group for distributed training """
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        timeout=timedelta(
            hours=5))


def cleanup():
    """ Destroy the process group """
    dist.destroy_process_group()


def disable_attention_dropout(module):
    """Keep attention consistent between rollouts and policy updates."""
    for submodule in module.modules():
        if hasattr(submodule, "attention_dropout"):
            submodule.attention_dropout = 0.0


@record
@pyrallis.wrap()
def main(config: TrainConfig):
    if not config.sft_path or not config.data_path:
        raise ValueError("Pass --sft_path and --data_path explicitly")
    config.mesh_root = config.mesh_root or os.path.join(config.data_path, "mesh")
    world_size = int(os.getenv("WORLD_SIZE"))
    setup(world_size)

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    print("RANK, WS:", rank, world_size, flush=True)
    checkpoint_config = AutoConfig.from_pretrained(config.sft_path, local_files_only=True)
    if getattr(checkpoint_config, "ggm_implementation", None) != "tip_dva_202611051433":
        raise ValueError("RL requires an SFT checkpoint trained with the paper GGM; warm-start it with seqeditor_train.py first")
    if config.representation not in ("command", "cadquery"):
        raise ValueError("representation must be command or cadquery")
    if config.prediction_mesh_scale is None:
        config.prediction_mesh_scale = getattr(checkpoint_config, "prediction_mesh_scale", 1.0)

    model = DVACAD.from_pretrained(
        config.sft_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=local_rank).train().to(local_rank)
    disable_attention_dropout(model)
    ref_model = DVACAD.from_pretrained(
        config.sft_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=local_rank,
    ).eval().to(local_rank)
    disable_attention_dropout(ref_model)
    ref_model.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(
        config.processor_path or config.sft_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side="left",
        local_files_only=True,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"
    ref_model.config.use_cache = False

    dataset_cls = CadEditDataset_query if config.representation == "cadquery" else CadEditDataset
    train_data = dataset_cls(
        root_dir=config.data_path,
        mesh_root=config.mesh_root,
        split='train',
        n_points=config.n_points,
        normalize_std_pc=100,
        noise_scale_pc=None)
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")

    model = optimize_model_memory(model)

    if config.freeze_pc:
        model.freeze_pc()

    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=config.find_unused_parameters,
    )
    dist.barrier()
    part_collate = partial(
        collate,
        processor=processor,
        n_points=config.n_points,
        # Rollouts need prompts without target answers.
        eval=True)

    print("\nStarting RL fine-tuning using GRPO...")
    training_config = {
        'train_epochs': config.train_epochs,
        'max_optimizer_steps': config.max_optimizer_steps,
        'batch_size': config.batch_size,
        'num_generations': config.num_generations,
        'top_samples': config.top_samples,
        'max_completion_length': config.max_completion_length,
        'learning_rate': config.learning_rate,
        'batch_updates': config.batch_updates,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'epsilon_high': config.epsilon_high,
        'epsilon_low': config.epsilon_low,
        'kl_coef': config.kl_coef,
        'temperature': config.temperature,
        'top_p': config.top_p,
        'do_sample': config.do_sample,
        'top_k': config.top_k,
        'dataloader_workers': config.dataloader_workers,
        'pin_memory': config.pin_memory,
        'persistent_workers': config.persistent_workers,
        'save_every_steps': config.save_every_steps,
        'save_every_optimizer_steps': config.save_every_optimizer_steps,
        'save_mid_epoch': config.save_mid_epoch,
        'epoch_save': config.epoch_save,
    }
    sampler = DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank)
    # Initialize Weights & Biases
    run_id = None
    if rank == 0:
        dict_config = asdict(config)
        wandb.init(
            project=config.project,
            group=config.group,
            name=config.name,
            reinit=True,
            config=dict_config)
        print("Weights & Biases initialized.")
        run_id = wandb.run.id

    model = train_with_grpo_mm(
        model=model,
        processor=processor,
        train_data=train_data,
        sampler=sampler,
        reward_function=get_reward_function(
            config.failure_reward,
            lambda_edit=config.lambda_edit,
            lambda_stable=config.lambda_stable,
            grid_resolution=config.grid_resolution,
            tau_phi=config.tau_phi,
            max_workers=config.pool_size,
            timeout=config.reward_timeout,
            prediction_scale=config.prediction_mesh_scale,
            mesh_root=config.mesh_root,
        ),
        ref_model=ref_model,
        collate_fn=part_collate,
        run_id=run_id,
        save_path=config.save_path,
        save_prefix=config.name,
        **training_config
    )
    if rank == 0:

        wandb.finish()
        print("Training completed and wandb run finished.")

        print("\nSaving GRPO fine-tuned model...")

        model.save_pretrained(f"{config.save_path}/{run_id}")
        processor.save_pretrained(f"{config.save_path}/{run_id}")
    cleanup()


if __name__ == "__main__":
    main()
