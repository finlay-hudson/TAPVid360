"""Settings for stage 2: segmenting and tracking moving objects across each 360° clip.

Values match the pipeline used to build TAPVid360-10k (TAPVid-360 paper, Sec. 3.2).
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BatchSizes:
    video: int = 2           # 360° clips loaded and processed together.
    langsam_persp: int = 4   # Perspective crops per Lang-SAM batch.
    # Object-centred videos per CoTracker call. Upstream CoTracker3 offline only supports 1 (a `.view` on an
    # expanded tensor fails for larger batches); results do not depend on it.
    cotracker: int = 1


@dataclass
class TrackGenConfig:
    sam_checkpoint: Path = Path("checkpoints") / "sam2" / "sam2_hiera_large.pt"
    n_threads: int = 8               # Threads for loading clips.

    # Frame sampling from each 360° clip.
    max_frames: int = 32
    downsample_rate: int = 4         # Ideal frame stride; lowered automatically for short clips.
    start_frame: int = 0

    # Perspective camera used for object-centred crops.
    fov_x: float = 75.18
    crop_width: int = 320
    crop_height: int = 240

    # Object discovery (Lang-SAM on frame 0) and video segmentation (SAM2).
    masks_per_img: int = 10                  # Keep the top-k Lang-SAM detections.
    sam_ignore_conf_below: float = 0.05      # Pixels at or below this confidence don't count towards a mask's mean.
    sam_min_mask_conf: float = 0.9           # Mean confidence a propagated mask needs.
    sam_pixel_conf: float = 0.9              # Pixel confidence for the binary mask.

    # Query points and tracking.
    points_per_mask: int = 256
    query_mask_erosion_kernel: int = 3       # Avoid sampling query points on the mask boundary...
    query_mask_erosion_iterations: int = 2   # ...by eroding it first.

    # Track filtering.
    min_motion_threshold: float = 1.0        # Mean per-frame point displacement (px) for an object to count as moving.
    min_visible_points: int = 20             # Every frame must have at least this many visible points.
    min_fraction_points_in_mask: float = 0.97  # Visible points that must stay inside the object's SAM2 mask.
    duplicate_track_chamfer: float = 10.0    # Tracks closer than this (mean Chamfer distance, px) are duplicates.

    batch_size: BatchSizes = field(default_factory=BatchSizes)
    skip_if_exists: bool = True
    offload_to_cpu: bool = False             # Move idle models to the CPU to save GPU memory (slower).
