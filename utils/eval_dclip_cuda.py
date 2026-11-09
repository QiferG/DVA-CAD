import torch  
from torchvision import transforms  
from PIL import Image  
from directional_clip_score import CLIPLoss
from torch.utils.data import Dataset, DataLoader
import json
import os
import math
from collections import defaultdict
import heapq
import argparse
from tqdm import tqdm


def find_files_with_prefix_source(directory, prefix):
    files = sorted(os.listdir(directory))

    matching_files = [
        file_name for file_name in files 
        if file_name.startswith(prefix)
    ]
    return matching_files


def find_files_with_prefix(directory, prefix):
    files = sorted(os.listdir(directory))

    matching_files = [
        file_name for file_name in files 
        if file_name.startswith(prefix)
    ]
    return matching_files


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

        for idx, item in enumerate(self.ins_data):
            prefix_source = str(idx).zfill(5) + '_'
            prefix_edit = str(idx).zfill(5) + '_'

            source_files = find_files_with_prefix_source(self.source_dir, prefix_source)
            matching_files = find_files_with_prefix(self.edit_dir, prefix_edit)
            print(prefix_edit, matching_files)

            if not source_files or not matching_files:
                continue

            instruction = item.get('instruction', '')
            target_text = src_text + instruction

            for source_file in source_files:
                source_path = os.path.join(self.source_dir, source_file)
                for matching_file in matching_files:
                    matching_path = os.path.join(self.edit_dir, matching_file)
                    samples.append({
                        'item_idx': idx,
                        'source_path': source_path,
                        'matching_path': matching_path,
                        'src_text': src_text,
                        'target_text': target_text,
                    })

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
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)

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
