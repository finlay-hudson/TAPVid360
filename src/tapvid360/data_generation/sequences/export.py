"""Write one sample in the released TAPVid360 format.

    <sample dir>/
        perspective_frames/0.jpg ... 31.jpg
        data.pt        # coords (T, N, 3), coords_visibility (T, N), rot_matrices (T, 3, 3), equi_h, fov_x
        gen_info.txt   # camera method used
"""
from pathlib import Path

import torch
from PIL import Image


def export_sample(results: dict, track_idx: int, out_dir: Path):
    """Export object track `track_idx` of a clip's resampled sequences (see sequences/main.py) to `out_dir`."""
    frames_dir = out_dir / "perspective_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, img in enumerate(results["new_persp_imgs"][track_idx]):
        Image.fromarray(img.cpu().numpy()).save(frames_dir / f"{frame_idx}.jpg")

    torch.save({
        "coords": results["new_persp_coords"][track_idx].cpu(),
        "coords_visibility": results["new_persp_coords_vis"][track_idx].cpu(),
        "rot_matrices": results["new_rot_mats"][track_idx].cpu(),
        "equi_h": results["eq_frames_h"],
        "fov_x": results["fov_x"],
    }, out_dir / "data.pt")

    with open(out_dir / "gen_info.txt", "w") as f:
        f.write("Rotation method: " + results["methods"][track_idx] + "\n")
