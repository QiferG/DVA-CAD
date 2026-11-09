import os
from tqdm import tqdm
from functools import partial
from argparse import ArgumentParser
import argparse
import torch
import json
from transformers import AutoProcessor
from torch.utils.data import DataLoader

from dvacad import DVACAD, collate
from dataset import CadEditDataset, CadEditDataset_query

def align_model_tokenizer_config(model, processor):
    tokenizer = processor.tokenizer
    token_fields = ("pad_token_id", "eos_token_id", "bos_token_id")
    for field in token_fields:
        value = getattr(tokenizer, field, None)
        setattr(model.config, field, value)
        if getattr(model, "generation_config", None) is not None:
            setattr(model.generation_config, field, value)


class RepeatedSampleDataset(torch.utils.data.Dataset):
    """Repeat each base sample consecutively `repeat_count` times."""

    def __init__(self, base_dataset, repeat_count):
        self.base_dataset = base_dataset
        self.repeat_count = repeat_count

    def __len__(self):
        return len(self.base_dataset) * self.repeat_count

    def __getitem__(self, idx):
        base_idx = idx // self.repeat_count
        return self.base_dataset[base_idx]


def generate_sequences(args):
    # should be no predicted codes from previous experiments
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    model = DVACAD.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        local_files_only=True,
        trust_remote_code=True,
        device_map='auto'
    )
    processor = AutoProcessor.from_pretrained(
        args.processor_path or args.model_path,
        padding_side='left',
        local_files_only=True
    )
    align_model_tokenizer_config(model, processor)
    model.eval()

    representation = args.representation or getattr(model.config, "cad_representation", "command")
    dataset_class = CadEditDataset_query if representation == "cadquery" else CadEditDataset
    val_dataset = dataset_class(
        root_dir=args.data_path,
        mesh_root=args.mesh_root,
        split='test',
        n_points=2048,
        normalize_std_pc=100,
        noise_scale_pc=None)
    repeated_val_dataset = RepeatedSampleDataset(val_dataset, args.num_samples)

    counter = 0
    dataloader = DataLoader(
        dataset=repeated_val_dataset,
        batch_size=args.batch_size,
        num_workers=16,
        collate_fn=partial(
            collate,
            processor=processor,
            n_points=2048,
            eval=True,
        ),
        shuffle=False)

    for batch in tqdm(dataloader):
        generated_ids = model.generate(
            do_sample=True,
            use_cache=True,
            temperature=args.temperature,
            top_p=args.top_p,
            input_ids=batch['input_ids'].to(model.device),
            attention_mask=batch['attention_mask'].to(model.device),
            point_clouds=batch['point_clouds'].to(model.device),
            sequences=batch['sequences'].to(model.device),
            sequence_attention_mask=batch['sequence_attention_mask'].to(model.device),
            is_pc=batch['is_pc'].to(model.device),
            is_img=batch['is_img'].to(model.device),
            pixel_values_videos=batch['pixel_values_videos'].to(model.device) if batch.get('pixel_values_videos', None) is not None else None,
            video_grid_thw=batch['video_grid_thw'].to(model.device) if batch.get('video_grid_thw', None) is not None else None,
            max_new_tokens=args.max_new_tokens)
        input_lengths = [input_ids.shape[0] for input_ids in batch['input_ids']]
        generated_ids_trimmed = [
            out_ids[input_len:] for input_len, out_ids in zip(input_lengths, generated_ids)
        ]
        decoded_sequences = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_index = 0
        with open(args.out_path, "a+", encoding='utf-8') as f:
            for item in batch['sample']:
                generated_sequence = decoded_sequences[output_index]
                
                output_item = {
                    "original_pic_name": item.get("original_pic_name", ""),
                    "edited_pic_name": item.get("edited_pic_name", ""),
                    "original_sequence": item.get("original_sequence", ""),
                    # keep ground-truth edited_sequence if present
                    "edited_sequence": item.get("edited_sequence", ""),
                    "type": item.get("type", ""),
                    "method": item.get("method", ""),
                    "instruction": item.get("instruction", ""),
                    # model output
                    "output": generated_sequence
                }
                f.write(json.dumps(output_item) + "\n")
                output_index += 1

    print(f"Generation complete. Results saved to {args.out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate edited CAD sequences using a fine-tuned model.")
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True, help="Fine-tuned model checkpoint directory")
    parser.add_argument("--processor-path", "--processor_path", dest="processor_path", default=None, help="Processor directory; defaults to --model-path")
    parser.add_argument("--representation", choices=["command", "cadquery"], default=None)
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True, help="Dataset root containing final/test.json")
    parser.add_argument("--mesh-root", "--mesh_root", dest="mesh_root", default=None, help="Mesh directory; defaults to DATA_PATH/mesh")
    parser.add_argument("--out-path", "--out_path", dest="out_path", required=True, help="Output JSONL file")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of edited sequences to generate for each input.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling.")
    args = parser.parse_args()

    generate_sequences(args)
