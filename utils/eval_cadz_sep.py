import argparse
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
import csv
import json
import sys
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_finetune.utils_cadrille import (
    _build_reference_field, normalize_mesh_triplet, point_cloud_losses,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", required=True, help="Generated pair-folder root")
    parser.add_argument("--real", required=True, help="Ground-truth pair-folder root")
    parser.add_argument("--original", required=True, help="Original-model pair-folder root")
    parser.add_argument("--output", help="Output summary JSON path")
    parser.add_argument("--n_points", type=int, default=2000)
    parser.add_argument("--grid_resolution", type=int, default=64)
    parser.add_argument("--tau_phi", type=float, default=0.01)
    parser.add_argument("--prediction_scale", type=float, default=0.01,
                        help="Scale exported predictions to dataset STL units; 1 for native-unit meshes")
    parser.add_argument("--num_pairs", type=int, default=100,
                        help="Randomly evaluate this many unique pairs; use 0 for all")
    parser.add_argument("--seed", type=int, default=32)
    args = parser.parse_args()
    np.random.seed(args.seed)
    if args.output is None:
        args.output = args.fake.rstrip("/") + "_paired_cd.json"

    generated = index_pair_folders(args.fake, keep_duplicates=True)
    targets = index_pair_folders(args.real)
    originals = index_pair_folders(args.original)
    generated_keys = set(generated)
    pair_keys = sorted(generated_keys & set(targets) & set(originals))
    missing_gt = len(generated_keys - set(targets))
    missing_original = len(generated_keys - set(originals))
    if args.num_pairs > 0 and len(pair_keys) > args.num_pairs:
        pair_keys = sorted(np.random.choice(pair_keys, size=args.num_pairs, replace=False).tolist())
    # Keep one generation per pair, choosing the lowest numeric sample prefix.
    generated_samples = [(key, generated[key][0]) for key in pair_keys]
    print(
        f"Generated samples={len(generated_samples)}, unique generated pairs={len(generated_keys)}, "
        f"gt pairs={len(targets)}, original pairs={len(originals)}, seed={args.seed}"
    )
    print(
        f"Matched samples={sum(1 for k, _ in generated_samples if k in pair_keys)}, "
        f"sampled unique pairs={len(pair_keys)}, missing_gt_pairs={missing_gt}, "
        f"missing_original_pairs={missing_original}, "
        f"duplicate generations ignored={len(generated_folders_count(generated)) - len(generated_keys)}"
    )

    rows = []
    for pair_key, pred_folder in tqdm(generated_samples, desc="Evaluating paired CAD models"):
        row = {
            "pair": pair_key,
            "generated_sample": pred_folder.name,
            "cd": None,
            "edited_cd": None,
            "stable_cd": None,
            "error": None,
        }
        try:
            if pair_key not in targets or pair_key not in originals:
                raise FileNotFoundError("No matching GT and/or original pair folder")
            pred_mesh = load_pair_mesh(pred_folder)
            gt_mesh = load_pair_mesh(targets[pair_key])
            ori_mesh = load_pair_mesh(originals[pair_key])
            pred_mesh, gt_mesh, ori_mesh = normalize_mesh_triplet(
                pred_mesh, gt_mesh, ori_mesh, prediction_scale=args.prediction_scale,
            )

            pred_points = sample_surface(pred_mesh, args.n_points)
            context = _build_reference_field(
                gt_mesh, ori_mesh, args.n_points, args.tau_phi, args.grid_resolution,
            )
            row["cd"] = symmetric_squared_cd(pred_points, context["gt_points"])
            row["edited_cd"], row["stable_cd"] = point_cloud_losses(pred_points, context)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    valid = [r for r in rows if all(r[k] is not None for k in ("cd", "edited_cd", "stable_cd"))]
    summary = {
        "num_generated_pairs_evaluated": len(generated_samples),
        "num_unique_generated_pairs": len(generated_keys),
        "num_pairs_requested": args.num_pairs,
        "random_seed": args.seed,
        "num_generated_folders": sum(len(folders) for folders in generated.values()),
        "num_duplicate_generations_ignored": sum(len(folders) - 1 for folders in generated.values()),
        "num_matched": sum(1 for k, _ in generated_samples if k in pair_keys),
        "num_matched_unique_pairs": len(pair_keys),
        "num_valid": len(valid),
        "num_failed": len(rows) - len(valid),
        "grid_resolution": args.grid_resolution,
        "tau_phi": args.tau_phi,
        "prediction_scale": args.prediction_scale,
        "n_points": args.n_points,
        "mean_cd": mean_metric(valid, "cd"),
        "mean_edited_cd": mean_metric(valid, "edited_cd"),
        "mean_stable_cd": mean_metric(valid, "stable_cd"),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2) + "\n")
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair", "generated_sample", "cd", "edited_cd", "stable_cd", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"Per-pair results: {csv_path}")


def pair_key_from_folder_name(folder_name):
    parts = folder_name.split("_", 1)
    if len(parts) != 2 or len(parts[0]) != 5 or not parts[0].isdigit():
        return folder_name
    return parts[1]


def generated_folders_count(indexed):
    return [folder for folders in indexed.values() for folder in folders]


def index_pair_folders(root, keep_duplicates=False):
    indexed = {}
    for folder in sorted(Path(p) for p in glob(os.path.join(root, "*/")) if Path(p).is_dir()):
        key = pair_key_from_folder_name(folder.name)
        if keep_duplicates:
            indexed.setdefault(key, []).append(folder)
        else:
            indexed.setdefault(key, folder)
    return indexed


def load_pair_mesh(folder):
    stl_files = sorted(folder.rglob("*.stl"))
    if not stl_files:
        raise FileNotFoundError(f"No STL found under {folder}")
    final_files = [p for p in stl_files if "final" in p.stem.lower()]
    mesh = trimesh.load_mesh(str((final_files or stl_files)[-1]))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Could not load a non-empty triangle mesh from {folder}")
    return mesh


def sample_surface(mesh, n_points):
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(points, dtype=np.float32)


def symmetric_squared_cd(points_a, points_b):
    d_ab = cKDTree(points_b).query(points_a, k=1)[0]
    d_ba = cKDTree(points_a).query(points_b, k=1)[0]
    return float(np.mean(d_ab ** 2) + np.mean(d_ba ** 2))


def mean_metric(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(vals)) if vals else None


if __name__ == '__main__':
    main()
