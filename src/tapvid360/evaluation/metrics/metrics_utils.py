from collections import defaultdict
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np

from src.tapvid360.evaluation.metrics.tapvid360_metrics import compute_metrics


def gen_metrics_and_dump(pred_tracks, sample, out_path):
    for b in range(sample.trajectory.shape[0]):
        n_points_pred = pred_tracks[b].shape[1]
        metrics = compute_metrics(pred_tracks[b], sample.trajectory[b][:, :n_points_pred],
                                  sample.visibility[b][:, :n_points_pred])
        with open(out_path / f"{sample.seq_name[b].replace('/', '-')}_metrics.json", 'w') as f:
            json.dump({m: metrics[m].mean().item() for m in metrics}, f, indent=4)


def aggregate_metrics(metrics_dir: Path):
    """
    Loads all individual JSON metric files from a directory, aggregates them,
    and computes the final mean and standard deviation for each metric.

    Args:
        metrics_dir: The path to the directory containing the JSON files.
    """
    if not metrics_dir.is_dir():
        print(f"Error: Directory not found at '{metrics_dir}'")
        return

    # Use glob to find all metric files, including in subdirectories
    json_files = list(metrics_dir.glob('**/*_metrics.json'))

    if not json_files:
        print(f"Error: No '*_metrics.json' files found in '{metrics_dir}'")
        return

    print(f"Found {len(json_files)} metric files to process...")

    # Use a defaultdict to easily collect values for each metric key
    aggregated_data = defaultdict(list)

    # 1. Loop through each file and collect the data
    for file_path in tqdm(json_files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            for metric_name, value in data.items():
                if value is not None:
                    aggregated_data[metric_name].append(value)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse {file_path}. Skipping. Error: {e}")

    # 2. Calculate the final statistics for each metric
    final_results = {}
    print("\n--- Aggregated Metrics ---")

    for metric_name, values in aggregated_data.items():
        if not values:
            continue

        values_np = np.array(values)

        mean_of_means = np.nanmean(values_np)
        std_of_means = np.nanstd(values_np)

        final_results[metric_name] = {'mean': mean_of_means, 'std': std_of_means}

        # 3. Print the results
        print(f"\nMetric: **{metric_name}**")
        print(f"  - Average of video means: {mean_of_means:.4f}")
        print(f"  - Std Dev of video means: {std_of_means:.4f}")

    return final_results


def aggregate_metrics_by_camera_method(metrics_dir: Path, dataset_dir: Path):
    """Mean of each metric per camera method (as in Table 4 of the paper), read from each sample's gen_info.txt.

    Args:
        metrics_dir: Directory of `<clip>-<idx>_metrics.json` files.
        dataset_dir: The dataset/ directory of TAPVid360-10k (containing <clip>/<idx>/gen_info.txt).
    """
    per_method = defaultdict(lambda: defaultdict(list))
    for file_path in metrics_dir.glob('**/*_metrics.json'):
        clip, idx = file_path.name[:-len('_metrics.json')].rsplit('-', 1)
        gen_info = dataset_dir / clip / idx / 'gen_info.txt'
        if not gen_info.exists():
            continue
        method = gen_info.read_text().split(':', 1)[1].strip()
        with open(file_path) as f:
            for metric_name, value in json.load(f).items():
                if value is not None:
                    per_method[method][metric_name].append(value)

    print("\n--- Metrics per camera method ---")
    final_results = {}
    for method in sorted(per_method):
        final_results[method] = {m: float(np.nanmean(v)) for m, v in per_method[method].items()}
        n_samples = max(len(v) for v in per_method[method].values())
        values = ", ".join(f"{m}: {v:.4f}" for m, v in final_results[method].items())
        print(f"{method} ({n_samples} samples): {values}")
    return final_results
