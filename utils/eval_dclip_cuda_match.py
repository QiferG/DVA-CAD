import torch  
from torchvision import transforms  
from PIL import Image  
from directional_clip_score import CLIPLoss
from torch.utils.data import Dataset, DataLoader
import json
import os
import math
from collections import defaultdict
import argparse
from tqdm import tqdm


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_pair_key(original_pic_name, edited_pic_name):
    return f"{original_pic_name}\t{edited_pic_name}"


def make_item_pair_key(item):
    return make_pair_key(
        str(item.get('original_pic_name', '')).strip(),
        str(item.get('edited_pic_name', '')).strip(),
    )


def parse_render_name(file_name):
    """Parse generated render names with or without a repetition index.

    Supported formats:
      00000_00000797_origInput_00000797_postAC3_002_final.png
      02829_00194208_postAC10_00194208_postAC0_final.png
    """
    stem, ext = os.path.splitext(file_name)
    if ext.lower() not in IMAGE_EXTENSIONS:
        return None

    parts = stem.split('_')
    # The current renderer emits six components and older outputs may include
    # an additional repetition component before ``final``.
    if len(parts) < 6 or parts[-1].lower() != 'final':
        return None

    return {
        'prefix': parts[0],
        'pair_key': make_pair_key('_'.join(parts[1:3]), '_'.join(parts[3:5])),
    }


def build_source_index(source_dir):
    source_by_pair = defaultdict(list)
    for file_name in sorted(os.listdir(source_dir)):
        parsed = parse_render_name(file_name)
        if parsed is None:
            continue
        source_by_pair[parsed['pair_key']].append(os.path.join(source_dir, file_name))
    return source_by_pair


def build_generated_index(edit_dir):
    generated_by_prefix = defaultdict(list)
    for file_name in sorted(os.listdir(edit_dir)):
        parsed = parse_render_name(file_name)
        if parsed is None:
            continue
        generated_by_prefix[parsed['prefix']].append({
            'path': os.path.join(edit_dir, file_name),
            'pair_key': parsed['pair_key'],
        })
    return generated_by_prefix


class DClipPairDataset(Dataset):
    def __init__(self, ins_data, source_dir, edit_dir, image_preprocess):
        self.ins_data = ins_data
        self.source_dir = source_dir
        self.edit_dir = edit_dir
        self.image_preprocess = image_preprocess
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        src_text = "This is a 3D shape. "
        source_by_pair = build_source_index(self.source_dir)
        generated_by_prefix = build_generated_index(self.edit_dir)
        missing_source = 0
        missing_generated = 0
        mismatched_generated_key = 0
        duplicate_source_keys = 0

        for idx, item in enumerate(self.ins_data):
            prefix = str(idx).zfill(5)
            pair_key = make_item_pair_key(item)
            source_paths = source_by_pair.get(pair_key, [])
            generated_items = generated_by_prefix.get(prefix, [])
            matching_generated_items = [
                generated_item for generated_item in generated_items
                if generated_item['pair_key'] == pair_key
            ]

            if not source_paths:
                missing_source += 1
                continue
            if not generated_items:
                missing_generated += 1
                continue
            if not matching_generated_items:
                mismatched_generated_key += 1
                continue
            if len(source_paths) > 1:
                duplicate_source_keys += 1

            instruction = item.get('instruction', '')
            target_text = src_text + instruction
            # Source images are rendered by CAD pair, while generated images are
            # rendered per JSONL row/repeated sample. Use the row-specific
            # generated image and one source image with the same CAD pair.
            source_path = source_paths[0]

            for generated_item in matching_generated_items:
                samples.append({
                    'item_idx': idx,
                    'source_path': source_path,
                    'matching_path': generated_item['path'],
                    'src_text': src_text,
                    'target_text': target_text,
                })

        print(
            "DCLIP matching summary: "
            f"samples={len(samples)}, "
            f"missing_source={missing_source}, "
            f"missing_generated={missing_generated}, "
            f"mismatched_generated_key={mismatched_generated_key}, "
            f"duplicate_source_keys={duplicate_source_keys}"
        )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            with Image.open(sample['source_path']) as source_image:
                src_tensor = self.image_preprocess(source_image)
            with Image.open(sample['matching_path']) as target_image:
                target_tensor = self.image_preprocess(target_image)
        except Exception as e:
            print(f"Error loading {sample['source_path']} -> {sample['matching_path']}: {e}")
            return None

        return {
            'item_idx': sample['item_idx'],
            'source_path': sample['source_path'],
            'matching_path': sample['matching_path'],
            'src_text': sample['src_text'],
            'target_text': sample['target_text'],
            'src_tensor': src_tensor,
            'target_tensor': target_tensor,
        }


def dclip_collate_fn(batch):
    # Drop broken samples so one bad image does not fail the whole run.
    valid_batch = [x for x in batch if x is not None]
    if not valid_batch:
        return None

    return {
        'item_idx': [x['item_idx'] for x in valid_batch],
        'source_path': [x['source_path'] for x in valid_batch],
        'matching_path': [x['matching_path'] for x in valid_batch],
        'src_text': [x['src_text'] for x in valid_batch],
        'target_text': [x['target_text'] for x in valid_batch],
        'src_tensor': torch.stack([x['src_tensor'] for x in valid_batch], dim=0),
        'target_tensor': torch.stack([x['target_tensor'] for x in valid_batch], dim=0),
    }

def cal_dclip(args):
    ensure_parent_dir(args.out_path)
    ensure_parent_dir(args.result_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_loss_module = CLIPLoss(device=device)
    
    with open(args.instruction_path, 'r') as file:  
        content = file.read().strip()
        try:
            data = json.loads(content)
            ins_data = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            ins_data =[]
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    ins_data.append(json.loads(line))
    
    # Collect all DCLIP scores for average calculation
    all_dclip_scores = []
    total_comparisons = 0

    dataset = DClipPairDataset(
        ins_data=ins_data,
        source_dir=args.source_dir,
        edit_dir=args.edit_dir,
        image_preprocess=clip_loss_module.clip_preprocess,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dclip_collate_fn,
    )
    
    with torch.no_grad():
        with open(args.out_path, 'a', encoding='utf-8') as out_file:
            for batch in tqdm(dataloader):
                if batch is None:
                    continue

                src_batch = batch['src_tensor'].to(device)
                target_batch = batch['target_tensor'].to(device)
                source_texts = batch['src_text']
                target_texts = batch['target_text']

                try:
                    batch_scores = clip_loss_module.batch_clip_directional_loss(
                        src_batch,
                        source_texts,
                        target_batch,
                        target_texts,
                    )
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue

                if isinstance(batch_scores, torch.Tensor):
                    batch_scores = batch_scores.detach().cpu().tolist()

                for item_idx, source_path, matching_path, score in zip(
                    batch['item_idx'],
                    batch['source_path'],
                    batch['matching_path'],
                    batch_scores,
                ):
                    score_value = float(score)
                    result_item = dict(ins_data[item_idx])
                    # include source/matching file info for traceability
                    result_item['source_path'] = source_path
                    result_item['matching_path'] = matching_path
                    result_item['dclip'] = score_value

                    if not math.isnan(score_value):
                        all_dclip_scores.append(score_value)
                        total_comparisons += 1

                    json.dump(result_item, out_file, ensure_ascii=False)
                    out_file.write(',\n')

    # Calculate and print statistics
    if all_dclip_scores:
        average_dclip = sum(all_dclip_scores) / len(all_dclip_scores)
        max_dclip = max(all_dclip_scores)
        min_dclip = min(all_dclip_scores)
        
        print(f"\n{'='*50}")
        print(f"DCLIP Score Statistics:")
        print(f"{'='*50}")
        print(f"Total comparisons: {total_comparisons}")
        print(f"Average DCLIP score: {average_dclip:.6f}")
        print(f"Maximum DCLIP score: {max_dclip:.6f}")
        print(f"Minimum DCLIP score: {min_dclip:.6f}")
        print(f"{'='*50}")
        result = f"Total comparisons: {total_comparisons}" + f"\nAverage DCLIP score: {average_dclip:.6f}" + f"\nMaximum DCLIP score: {max_dclip:.6f}" + f"\nMinimum DCLIP score: {min_dclip:.6f}"
        with open(args.result_path, 'a') as f:
            f.write(result)
            f.write('\n')
        
    else:
        print("No DCLIP scores were calculated!")
    
    return all_dclip_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--edit_dir", type=str, required=True)
    parser.add_argument("--instruction_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    
    cal_dclip(args)
