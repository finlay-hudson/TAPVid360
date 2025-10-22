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
