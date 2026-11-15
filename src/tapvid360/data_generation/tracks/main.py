"""Stage 2: find moving objects in each 360° clip and track points on them across the full sphere.

For each clip: Lang-SAM finds candidate objects in frame 0 of the equirectangular video, SAM2 propagates their masks
through the clip, CoTracker3 tracks points on each object in an object-centred perspective video, and the tracks are
projected back to equirectangular coordinates. Tracks of static, poorly segmented or duplicate objects are dropped.

Example:
    PYTHONPATH=. python src/tapvid360/data_generation/tracks/main.py \
        --clips_dir outputs/download/<run>/filtered_clips --out_dir outputs/tracks_360

Writes `<out_dir>/<clip name>/tracks_360.pth` (tracks) and `frames_32.pth` (the 32 sampled equirectangular frames).
"""
import argparse
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.tapvid360.common.conversions.mapper import Mappers
from src.tapvid360.data_generation.tracks.config import TrackGenConfig
from src.tapvid360.data_generation.tracks.cotracker import CoTracker, scale_points
from src.tapvid360.data_generation.precision import enable_generation_precision
from src.tapvid360.data_generation.tracks.langsam import LangSamRunner
from src.tapvid360.data_generation.tracks.mask_utils import (compute_center_of_mass, erode_batch,
                                                             find_mask_idx_closest_to_the_centre_of_img,
                                                             find_points_in_masks, get_distributed_points)
from src.tapvid360.data_generation.tracks.sam2_video import get_sam2_predictor, run_through_sam_video
from src.tapvid360.data_generation.video_utils import (EQ_HEIGHT, EQ_WIDTH, EquirectangularVideoReader,
                                                       encode_frames_to_video_bytes)

logger = logging.getLogger(__name__)


# --- Loading clips ---------------------------------------------------------------------------------------------------

def best_downsample_rate(num_frames: int, ideal_rate: int, min_required_frames: int) -> int | None:
    """Largest stride <= `ideal_rate` that still yields `min_required_frames` frames."""
    for rate in range(ideal_rate, 0, -1):
        if num_frames // rate >= min_required_frames:
            return rate
    return None


def load_clip(clip_path: Path, config: TrackGenConfig):
    """Sample `config.max_frames` frames from a clip. Returns (clip_path, frames, frame_idxs) or None."""
    reader = EquirectangularVideoReader(clip_path)
    ds_rate = best_downsample_rate(reader.count_frames(), config.downsample_rate, config.max_frames)
    if ds_rate is None:
        return None
    frame_idxs = [(i * ds_rate) + config.start_frame for i in range(config.max_frames)]
    try:
        frames = reader.load_frames(frame_idxs)
    except Exception as e:
        logger.warning(f"Could not read {clip_path}: {e}")
        return None
    return clip_path, torch.tensor(frames), frame_idxs


def load_clip_batch(clip_paths: list[Path], config: TrackGenConfig, frames_buffer: torch.Tensor):
    """Load clips into `frames_buffer`. Returns (loaded clip paths, buffer indices used, frame indices per clip)."""
    if config.n_threads > 1:
        with ThreadPoolExecutor(max_workers=config.n_threads) as executor:
            results = list(executor.map(lambda p: load_clip(p, config), clip_paths))
    else:
        results = [load_clip(p, config) for p in clip_paths]

    loaded, buffer_idxs, all_frame_idxs = [], [], []
    for i, result in enumerate(results):
        if result is None:
            continue
        clip_path, frames, frame_idxs = result
        frames_buffer[i] = frames.to(frames_buffer.device)
        loaded.append(clip_path)
        buffer_idxs.append(i)
        all_frame_idxs.append(frame_idxs)
    return loaded, buffer_idxs, all_frame_idxs


# --- Per-object processing -------------------------------------------------------------------------------------------

def find_complete_vid_masks(vid_seg_masks: torch.Tensor) -> torch.Tensor:
    """Masks (N, T, H, W) that are non-empty in every frame."""
    return vid_seg_masks.any(dim=(2, 3)).all(dim=1)


def get_object_centred_persp_imgs(eq_vid_frames, vid_seg_masks, mapper, batchsize=16):
    """Perspective videos that keep each object's mask centre in the middle of the frame."""
    num_masks, num_frames = vid_seg_masks.shape[:2]
    rot_matrices = torch.empty(num_masks, num_frames, 3, 3, device=vid_seg_masks.device)
    for bs in range(0, len(vid_seg_masks), batchsize):
        centres = compute_center_of_mass(vid_seg_masks[bs: bs + batchsize].flatten(0, 1))
        rot_matrices[bs: bs + batchsize] = mapper.compute_rotation_matrix_centred_at_point(centres).view(
            -1, num_frames, 3, 3)
    persp_imgs = torch.stack([mapper.image.equirectangular_image_to_perspective(eq_vid_frames, rot_mat)
                              for rot_mat in rot_matrices], dim=0)
    return persp_imgs, rot_matrices


def get_better_quality_query_masks(batch_size, lang_sam_runner, query_persp_imgs, labels):
    """Re-run Lang-SAM on each object-centred frame 0 for a sharper query mask.

    Returns (masks, indices of the objects found again), or (None, None) if none were.
    """
    if query_persp_imgs.ndim == 3:
        query_persp_imgs = query_persp_imgs[None]
    results = []
    for bs in range(0, len(query_persp_imgs), batch_size):
        results.extend(lang_sam_runner.run_custom(query_persp_imgs[bs: bs + batch_size],
                                                  prompt=labels[bs: bs + batch_size]))
    query_masks, idxs_matched = [], []
    for idx, res in enumerate(results):
        if not len(res["masks"]):
            continue
        mask_idx = find_mask_idx_closest_to_the_centre_of_img(res["masks"], h=query_persp_imgs.shape[-3],
                                                              w=query_persp_imgs.shape[-2])
        query_masks.extend(res["masks"][mask_idx])
        idxs_matched.append(idx)
    if not len(query_masks):
        return None, None
    return torch.stack(query_masks), idxs_matched


def run_through_cotracker(co_tracker: CoTracker, batchsize: int, device, persp_imgs, query_points):
    """Track (x, y) query points through perspective videos. Returns tracks in `persp_imgs` pixels, visibility."""
    video_resize, queries = co_tracker.preprocess(persp_imgs, query_points=query_points, device=device)
    all_tracks, all_vis = [], []
    for bs in range(0, len(persp_imgs), batchsize):
        tracks, vis = co_tracker.run(video_resize[bs:bs + batchsize].contiguous(),
                                     query_points=queries[bs:bs + batchsize].contiguous())
        all_tracks.append(tracks)
        all_vis.append(vis)
    all_tracks = scale_points(torch.vstack(all_tracks), co_tracker.interp_shape, persp_imgs.shape[-3:-1])
    return all_tracks, torch.vstack(all_vis)


def is_dynamic(tracks: torch.Tensor, min_motion_threshold: float) -> torch.Tensor:
    """Objects (tracks (B, T, P, 2)) whose mean per-frame point displacement exceeds the threshold."""
    dists = torch.norm((tracks[:, 1:] - tracks[:, :-1]).float(), dim=-1)  # (B, T-1, P)
    motion_per_point_per_frame = dists.sum(dim=1) / tracks.shape[1]
    return motion_per_point_per_frame.mean(-1) > min_motion_threshold


def get_unique_tracks_mask(tracks: torch.Tensor, threshold: float) -> torch.Tensor:
    """Keep the first of any pair of tracks (N, T, P, 2) whose mean Chamfer distance is below `threshold`."""
    n_tracks = tracks.shape[0]
    if n_tracks <= 1:
        return torch.tensor([True], device=tracks.device)
    keep = torch.ones(n_tracks, dtype=torch.bool, device=tracks.device)
    for i in range(n_tracks):
        if not keep[i]:
            continue
        for j in range(i + 1, n_tracks):
            if not keep[j]:
                continue
            dist = torch.cdist(tracks[i], tracks[j])  # (T, P, P)
            chamfer_per_frame = dist.min(dim=2).values.mean(dim=1) + dist.min(dim=1).values.mean(dim=1)
            if chamfer_per_frame.mean() < threshold:
                keep[j] = False
    return keep


def process_clip(clip_name, eq_vid_frames, frame_idxs, lang_sam_result, models, config: TrackGenConfig, device):
    """Find, segment and track moving objects in one clip. Returns the tracks dict, or None if nothing survives."""
    sam_pred_video, co_tracker, lang_sam_runner = models
    _, eq_h, _, _ = eq_vid_frames.shape
    if not len(lang_sam_result["masks"]):
        logger.info(f"{clip_name}: Lang-SAM found no objects")
        return None
    top_k = lang_sam_runner.get_top_k_masks_from_result(lang_sam_result, top_k=config.masks_per_img)
    mapper = Mappers(config.crop_width, config.crop_height, equirectangular_height=eq_h, fov_x=config.fov_x)

    # 1. Propagate frame-0 masks with SAM2; keep confident masks present in every frame.
    sam_pred_video = sam_pred_video.to(device)
    _, vid_seg_confs = run_through_sam_video(sam_pred_video, eq_vid_frames, masks=top_k["masks"])
    confs_for_mean = vid_seg_confs.clone()
    confs_for_mean[confs_for_mean <= config.sam_ignore_conf_below] = torch.nan
    mean_confs = torch.nanmean(confs_for_mean, (1, 2, 3))
    confident = mean_confs > config.sam_min_mask_conf
    labels = np.array(top_k["labels"])[confident.cpu().numpy()]
    vid_seg_masks = vid_seg_confs[confident] > config.sam_pixel_conf
    if not vid_seg_masks.any():
        logger.info(f"{clip_name}: no high-confidence SAM2 masks")
        return None
    full_length = find_complete_vid_masks(vid_seg_masks)
    if not full_length.any():
        logger.info(f"{clip_name}: no SAM2 mask present in every frame")
        return None
    if config.offload_to_cpu:
        sam_pred_video.to("cpu")
    labels = labels[full_length.cpu().numpy()]
    vid_seg_masks = vid_seg_masks[full_length]

    # 2. Object-centred perspective videos, and sharper query masks from Lang-SAM on their first frame.
    persp_imgs, persp_rot_mats = get_object_centred_persp_imgs(eq_vid_frames, vid_seg_masks, mapper,
                                                               batchsize=config.batch_size.langsam_persp)
    if config.offload_to_cpu:
        eq_vid_frames = eq_vid_frames.to("cpu")
    lang_sam_runner.to_device(device)
    query_masks, idxs_matched = get_better_quality_query_masks(config.batch_size.langsam_persp, lang_sam_runner,
                                                               persp_imgs[:, 0], labels.tolist())
    if query_masks is None:
        logger.info(f"{clip_name}: Lang-SAM found no objects in the perspective crops")
        return None
    if config.offload_to_cpu:
        lang_sam_runner.to_device("cpu")
    persp_imgs, persp_rot_mats = persp_imgs[idxs_matched], persp_rot_mats[idxs_matched]

    # 3. Query points spread over each eroded mask (avoiding mask boundaries).
    eroded = torch.stack([erode_batch(m[None], kernel_size=config.query_mask_erosion_kernel,
                                      iterations=config.query_mask_erosion_iterations)[0][0] for m in query_masks])
    query_points, idxs_with_points = [], []
    for idx, mask in enumerate(eroded):
        if mask.sum() > 0:
            query_points.append(get_distributed_points(mask, num_points=config.points_per_mask))
            idxs_with_points.append(idx)
    if not len(query_points):
        logger.info(f"{clip_name}: eroded query masks are empty")
        return None
    query_points = torch.stack(query_points)
    persp_imgs, persp_rot_mats = persp_imgs[idxs_with_points], persp_rot_mats[idxs_with_points]

    # 4. Track with CoTracker3 and keep moving objects.
    co_tracker.model = co_tracker.model.to(device)
    tracks, vis = run_through_cotracker(co_tracker, config.batch_size.cotracker, device, persp_imgs, query_points)
    if config.offload_to_cpu:
        co_tracker.model = co_tracker.model.to("cpu")
    dynamic = is_dynamic(tracks, config.min_motion_threshold)
    tracks, vis, persp_rot_mats = tracks[dynamic], vis[dynamic], persp_rot_mats[dynamic]
    obj_masks = vid_seg_masks[idxs_matched][idxs_with_points][dynamic]
    if not len(tracks):
        logger.info(f"{clip_name}: no moving objects")
        return None

    # 5. Back to equirectangular coordinates; drop poorly tracked objects and duplicates.
    eq_points = mapper.point.ij.to_cr(tracks, persp_rot_mats)
    mostly_vis = (vis.sum(2) >= config.min_visible_points).all(1)
    eq_points, vis, persp_rot_mats = eq_points[mostly_vis], vis[mostly_vis], persp_rot_mats[mostly_vis]
    points_in_masks = find_points_in_masks(eq_points, obj_masks[mostly_vis])
    points_in_masks[vis < 0.5] = 1  # Only visible points count.
    well_tracked = points_in_masks.float().mean((1, 2)) > config.min_fraction_points_in_mask
    eq_points, vis, persp_rot_mats = eq_points[well_tracked], vis[well_tracked], persp_rot_mats[well_tracked]
    if not len(eq_points):
        logger.info(f"{clip_name}: no objects left after the points-in-mask check")
        return None
    unique = get_unique_tracks_mask(eq_points, threshold=config.duplicate_track_chamfer)

    return {
        "cr_points": eq_points[unique],          # (N, T, P, 2) equirectangular (x, y) pixel tracks
        "cr_points_vis": vis[unique],            # (N, T, P) CoTracker visibility
        "rot_matrices": persp_rot_mats[unique],  # (N, T, 3, 3) object-centred camera rotations
        "frame_idxs": frame_idxs,                # Frame indices sampled from the source clip
        "eq_img_h": eq_h,
    }


# --- Main ------------------------------------------------------------------------------------------------------------

def list_clips(clips_dir: Path, clip_list: Path | None, exclude_list: Path | None) -> list[Path]:
    """Clip paths relative to `clips_dir` (`<video_id>/<video_id>_clip_NNNN.mp4`)."""
    if clip_list is not None:
        with open(clip_list) as f:
            clips = [Path(c) for c in json.load(f)]
    else:
        clips = sorted(p.relative_to(clips_dir) for p in clips_dir.glob("*/*.mp4"))
    for clip in clips:
        if not (clips_dir / clip).exists():
            raise FileNotFoundError(f"Missing clip: {clips_dir / clip}")
    if exclude_list is not None:
        excluded = excluded_clip_names(exclude_list)
        clips = [c for c in clips if c.stem not in excluded]
    return clips


def excluded_clip_names(exclude: Path) -> set[str]:
    """Clip names to skip: from a TAPVid360 dataset root (containing dataset/<clip>/<idx>/) or a JSON list of
    "<clip>/<idx>" sample names. As for TAPVid360-10k, only those clips are excluded, not their whole videos."""
    if exclude.is_dir():
        return {p.name for p in (exclude / "dataset").iterdir() if p.is_dir()}
    with open(exclude) as f:
        return {Path(name).parent.name for name in json.load(f)}


def main(args):
    random.seed(42)
    enable_generation_precision()
    config = TrackGenConfig(sam_checkpoint=args.sam_checkpoint)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clips = list_clips(args.clips_dir, args.clip_list, args.exclude_list)
    if config.skip_if_exists:
        done = {p.name for p in args.out_dir.iterdir() if (p / "tracks_360.pth").exists()}
        clips = [c for c in clips if c.stem not in done]
    logger.info(f"Running on {len(clips)} clips")
    if not clips:
        return

    # Models start on the CPU and are moved to the GPU when used.
    sam_pred_video = get_sam2_predictor(single_image=False, checkpoint=config.sam_checkpoint, device=torch.device("cpu"))
    sam_pred_image = get_sam2_predictor(single_image=True, checkpoint=config.sam_checkpoint, device=torch.device("cpu"))
    co_tracker = CoTracker(torch.device("cpu"))
    lang_sam_runner = LangSamRunner(image_sam=sam_pred_image)
    models = (sam_pred_video, co_tracker, lang_sam_runner)

    start = time.time()
    frames_buffer = torch.empty(config.batch_size.video, config.max_frames, EQ_HEIGHT, EQ_WIDTH, 3,
                                dtype=torch.uint8, device=device)
    batches = [clips[i:i + config.batch_size.video] for i in range(0, len(clips), config.batch_size.video)]
    for batch in tqdm(batches):
        loaded, buffer_idxs, all_frame_idxs = load_clip_batch([args.clips_dir / c for c in batch], config,
                                                              frames_buffer)
        if not loaded:
            continue
        batch_frames = frames_buffer[buffer_idxs]
        lang_sam_runner.to_device(device)
        lang_sam_results = lang_sam_runner.run_custom(batch_frames[:, 0])
        if config.offload_to_cpu:
            lang_sam_runner.to_device("cpu")

        for eq_vid_frames, frame_idxs, lang_sam_result, clip_path in zip(batch_frames, all_frame_idxs,
                                                                          lang_sam_results, loaded):
            results = process_clip(clip_path.stem, eq_vid_frames, frame_idxs, lang_sam_result, models, config, device)
            if results is None:
                continue
            out_dir = args.out_dir / clip_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(results, out_dir / "tracks_360.pth")
            torch.save(encode_frames_to_video_bytes(eq_vid_frames.cpu().numpy()),
                       out_dir / f"frames_{config.max_frames}.pth")
            logger.info(f"{clip_path.stem}: saved {len(results['cr_points'])} object tracks")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total = time.time() - start
    logger.info(f"Took {total:.0f}s ({total / len(clips):.1f}s per clip)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips_dir", type=Path, required=True,
                        help="Stage-1 output: directory of <video_id>/<video_id>_clip_NNNN.mp4 clips.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--clip_list", type=Path, default=None,
                        help="Optional JSON list of clip paths (relative to --clips_dir) to process.")
    parser.add_argument("--exclude_list", type=Path, default=None,
                        help="Clips to skip when generating training data: a TAPVid360 dataset root such as "
                             "data/TAPVid360-10k, or a JSON list of '<clip>/<idx>' sample names.")
    parser.add_argument("--sam_checkpoint", type=Path, default=TrackGenConfig.sam_checkpoint)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
