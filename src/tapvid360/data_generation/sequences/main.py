"""Stage 3: resample each tracked object into new narrow-FoV videos with simulated camera motion, and export samples.

Benchmark mode (TAPVid360-10k): only object tracks accepted in manual review are used, each with one randomly chosen
camera method. Training mode: no review; every track is rendered with 5 different camera methods.

Example:
    PYTHONPATH=. python src/tapvid360/data_generation/sequences/main.py \
        --tracks_dirs outputs/tracks_360 --out_dir outputs/TAPVid360/dataset --mode benchmark

Writes `<out_dir>/<clip>/<idx>/` (benchmark) or `<out_dir>/<clip>_<method>/<idx>/` (training) samples in the
released format (see sequences/export.py).
"""
import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from src.tapvid360.common.conversions.mapper import Mappers
from src.tapvid360.data_generation.sequences.camera import RotationMethods, get_rotation_matrices
from src.tapvid360.data_generation.precision import enable_generation_precision
from src.tapvid360.data_generation.sequences.export import export_sample
from src.tapvid360.data_generation.tracks.config import TrackGenConfig
from src.tapvid360.data_generation.video_utils import decode_video_bytes_to_frames

logger = logging.getLogger(__name__)

REVIEW_DIR_SUFFIX = "_manual_checking"
ACCEPTED = "accepted"

# Training mode: camera methods and sampling weights (5 distinct methods per track).
TRAINING_METHODS = [RotationMethods.SPIN_X, RotationMethods.SPIN_Y, RotationMethods.SPIN_Z,
                    RotationMethods.PERSPECTIVE_SAMPLER, RotationMethods.RANDOM, RotationMethods.SPIRAL,
                    RotationMethods.STATIC, RotationMethods.REID]
TRAINING_METHOD_WEIGHTS = [0.05, 0.05, 0.05, 0.3, 0.25, 0.05, 0.1, 0.15]
TRAINING_METHODS_PER_TRACK = 5


def are_points_in_frame(points: torch.Tensor, frame_w: int, frame_h: int) -> torch.Tensor:
    """Whether each (x, y) point lies inside a frame_w x frame_h image."""
    x, y = points[..., 0], points[..., 1]
    return (x >= 0) & (x < frame_w) & (y >= 0) & (y < frame_h)


def review_file(review_dir: Path, clip_name: str) -> Path:
    return review_dir / f"{clip_name}_index_status.json"


def load_accepted_mask(review_dir: Path, clip_name: str) -> torch.Tensor | None:
    """Boolean mask of tracks marked "accepted" in review, or None if the clip has no review file."""
    path = review_file(review_dir, clip_name)
    if not path.exists():
        return None
    with open(path) as f:
        statuses = {int(idx): status for idx, status in json.load(f).items()}
    return torch.tensor([statuses[idx] == ACCEPTED for idx in sorted(statuses)], dtype=torch.bool)


def num_complete_samples(sample_dir: Path) -> int:
    """Number of fully exported samples (<idx>/gen_info.txt is written last) in a <clip> or <clip>_<method> dir."""
    return sum(1 for p in sample_dir.glob("*/gen_info.txt")) if sample_dir.is_dir() else 0


def is_clip_done(out_dir: Path, clip_name: str, review_dir: Path, training: bool) -> bool:
    """Whether a previous (possibly interrupted) run fully exported this clip.

    In training mode, a partially exported clip is deleted so it is regenerated with a fresh set of camera methods.
    """
    if training:
        method_dirs = [out_dir / f"{clip_name}_{m.value}" for m in TRAINING_METHODS
                       if (out_dir / f"{clip_name}_{m.value}").is_dir()]
        counts = {num_complete_samples(d) for d in method_dirs}
        if len(method_dirs) == TRAINING_METHODS_PER_TRACK and len(counts) == 1 and counts.pop() > 0:
            return True
        for d in method_dirs:
            shutil.rmtree(d)
        return False
    accepted = load_accepted_mask(review_dir, clip_name)
    n_done = num_complete_samples(out_dir / clip_name)
    return n_done > 0 and (accepted is None or n_done == int(accepted.sum()))


def render_sequences(data: dict, eq_frames: torch.Tensor, mapper: Mappers, method: RotationMethods | None,
                     config: TrackGenConfig) -> dict:
    """Render every object track of a clip with a new camera trajectory (`method`, or random per track if None)."""
    new_rot_mats = torch.empty_like(data["rot_matrices"])
    methods, new_persp_imgs = [], []
    for i, rot_mat in enumerate(data["rot_matrices"]):
        new_rot_mats[i], used_method = get_rotation_matrices(rot_mat, method)
        methods.append(used_method)
        new_persp_imgs.append(mapper.image.equirectangular_image_to_perspective(eq_frames, new_rot_mats[i]))

    new_vc = mapper.point.cr.to_vc(data["cr_points"], new_rot_mats)
    new_persp_points = mapper.point.vc.to_ij(new_vc)
    return {
        "new_persp_imgs": torch.stack(new_persp_imgs),
        "new_rot_mats": new_rot_mats,
        "new_persp_coords": new_vc,
        # Visibility = inside the narrow-FoV frame (occlusion is not used).
        "new_persp_coords_vis": are_points_in_frame(new_persp_points, config.crop_width, config.crop_height),
        "methods": [m.name for m in methods],
        "fov_x": config.fov_x,
        "eq_frames_h": data["eq_img_h"],
    }


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    enable_generation_precision()
    config = TrackGenConfig()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    training = args.mode == "training"

    for tracks_dir in args.tracks_dirs:
        review_dir = Path(str(tracks_dir) + REVIEW_DIR_SUFFIX) if args.review_dir is None else args.review_dir
        if training:
            clip_names = sorted(p.name for p in tracks_dir.iterdir() if p.is_dir())
        else:
            # Benchmark: only clips that have been reviewed.
            clip_names = sorted(p.name.split("_index")[0] for p in review_dir.glob("*_index_status.json"))
        logger.info(f"{tracks_dir}: {len(clip_names)} clips")

        for clip_name in tqdm(clip_names):
            if is_clip_done(args.out_dir, clip_name, review_dir, training):
                continue
            tracks_path = tracks_dir / clip_name / "tracks_360.pth"
            frames_path = tracks_dir / clip_name / f"frames_{config.max_frames}.pth"
            if not tracks_path.exists() or not frames_path.exists():
                logger.warning(f"Skipping {clip_name}: missing {tracks_path.name} or {frames_path.name}")
                continue
            data = torch.load(tracks_path, weights_only=False, map_location=device)
            eq_frames = torch.from_numpy(decode_video_bytes_to_frames(
                torch.load(frames_path, weights_only=False, map_location=device))).to(device)

            if not training:
                accepted = load_accepted_mask(review_dir, clip_name)
                if accepted is not None:
                    if not accepted.any():
                        continue
                    for key in ("cr_points", "cr_points_vis", "rot_matrices"):
                        data[key] = data[key][accepted]

            mapper = Mappers(config.crop_width, config.crop_height, equirectangular_height=data["eq_img_h"],
                             fov_x=config.fov_x)
            if training:
                sampled = torch.utils.data.WeightedRandomSampler(weights=TRAINING_METHOD_WEIGHTS,
                                                                 num_samples=TRAINING_METHODS_PER_TRACK,
                                                                 replacement=False)
                methods = [TRAINING_METHODS[i] for i in list(sampled)]
            else:
                methods = [None]  # One randomly selected camera method per track.

            for method in methods:
                results = render_sequences(data, eq_frames, mapper, method, config)
                sample_name = f"{clip_name}_{method.value}" if training else clip_name
                for track_idx in range(len(results["new_persp_imgs"])):
                    export_sample(results, track_idx, args.out_dir / sample_name / str(track_idx))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracks_dirs", type=Path, nargs="+", required=True,
                        help="Stage-2 output directories (of <clip>/tracks_360.pth and frames_32.pth).")
    parser.add_argument("--out_dir", type=Path, required=True, help="Where to write the dataset samples.")
    parser.add_argument("--mode", choices=["benchmark", "training"], default="benchmark")
    parser.add_argument("--review_dir", type=Path, default=None,
                        help=f"Review statuses (benchmark mode). Default: '<tracks_dir>{REVIEW_DIR_SUFFIX}'.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for camera-trajectory sampling.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
