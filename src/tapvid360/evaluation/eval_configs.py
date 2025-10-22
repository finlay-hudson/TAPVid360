from dataclasses import dataclass
from pathlib import Path


@dataclass
class TAPVid360EvalConfig:
    results_dir: Path = Path("results")  # Path to store run through model predictions
    metrics_out_root: Path = Path("metrics")  # Path to store computed metrics
    data_dir: Path = Path("data")  # Parent of the /dataset folder of tapvid360
    trained_model_path: Path = Path(
        "checkpoints/cotracker360.pth")  # Path to the checkpoint file, if using cotracker360


class CoTracker360EvalConfig:
    stride: int = 4
    corr_radius: int = 3
    window_len: int = 60
    model_resolution: tuple = (240, 320)
    batch_size: int = 2
    num_workers: int = 4
    num_iters: int = 1
