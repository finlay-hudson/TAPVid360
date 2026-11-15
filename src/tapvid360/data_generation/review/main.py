"""Manual review of stage-2 object tracks (the curation step used for TAPVid360-10k).

Shows each object track either on the equirectangular video or in an object-centred perspective view. Mark each track
accepted (A) or rejected (R). Decisions are saved straight away to `<review_dir>/<clip>_index_status.json`, which
stage 3 reads in benchmark mode.

Example:
    PYTHONPATH=. python src/tapvid360/data_generation/review/main.py --tracks_dir outputs/tracks_360

Keys: A accept, R reject (both move on to the next track), Left/Right previous/next track, Space play/pause,
V switch view.
"""
import argparse
import base64
import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
import torch

from src.tapvid360.common.conversions.mapper import Mappers
from src.tapvid360.data_generation.tracks.config import TrackGenConfig
from src.tapvid360.data_generation.video_utils import decode_video_bytes_to_frames

NOT_REVIEWED, ACCEPTED, REJECTED = "not_reviewed", "accepted", "rejected"
STATUS_COLOURS = {NOT_REVIEWED: "black", ACCEPTED: "green", REJECTED: "red"}
REVIEW_DIR_SUFFIX = "_manual_checking"
DISPLAY_WIDTH = 1000
FPS = 8


def status_file(review_dir: Path, clip_name: str) -> Path:
    return review_dir / f"{clip_name}_index_status.json"


def load_statuses(review_dir: Path, clip_name: str, num_tracks: int) -> dict[int, str]:
    statuses = {i: NOT_REVIEWED for i in range(num_tracks)}
    path = status_file(review_dir, clip_name)
    if path.exists():
        with open(path) as f:
            statuses.update({int(k): v for k, v in json.load(f).items()})
    return statuses


def is_fully_reviewed(review_dir: Path, clip_name: str) -> bool:
    path = status_file(review_dir, clip_name)
    if not path.exists():
        return False
    with open(path) as f:
        return NOT_REVIEWED not in json.load(f).values()


class ClipData:
    """Tracks and frames of one clip, plus object-centred perspective views of each track."""

    def __init__(self, clip_dir: Path, config: TrackGenConfig):
        tracks = torch.load(clip_dir / "tracks_360.pth", map_location="cpu", weights_only=False)
        self.eq_frames = torch.from_numpy(decode_video_bytes_to_frames(
            torch.load(clip_dir / f"frames_{config.max_frames}.pth", weights_only=False)))
        self.cr_points = tracks["cr_points"].float()  # (N, T, P, 2)
        self.cr_points_vis = tracks["cr_points_vis"]  # (N, T, P)
        mapper = Mappers(config.crop_width, config.crop_height, equirectangular_height=tracks["eq_img_h"],
                         fov_x=config.fov_x)
        # Camera centred on the mean of each track's points in every frame.
        self.persp_imgs, self.persp_points = [], []
        for points in self.cr_points:
            rot_matrices = mapper.compute_rotation_matrix_centred_at_point(points.mean(-2)[None])
            self.persp_points.append(mapper.point.cr.to_ij(points[None], rot_matrices)[0])
            self.persp_imgs.append(mapper.image.equirectangular_image_to_perspective(self.eq_frames, rot_matrices[0]))

    def __len__(self):
        return len(self.cr_points)

    def render(self, track_idx: int, frame_idx: int, view: str) -> np.ndarray:
        if view == "equirectangular":
            img, points, radius = self.eq_frames[frame_idx].numpy().copy(), self.cr_points[track_idx, frame_idx], 6
        else:
            img, points, radius = (self.persp_imgs[track_idx][frame_idx].numpy().copy(),
                                   self.persp_points[track_idx][frame_idx], 3)
        for (x, y), vis in zip(points.tolist(), self.cr_points_vis[track_idx, frame_idx].tolist()):
            cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), -1 if vis > 0.5 else 1)
        scale = DISPLAY_WIDTH / img.shape[1]
        return cv2.resize(img, (DISPLAY_WIDTH, int(img.shape[0] * scale)))


class ReviewApp:
    def __init__(self, master: tk.Tk, tracks_dir: Path, review_dir: Path):
        self.master, self.tracks_dir, self.review_dir = master, tracks_dir, review_dir
        self.config = TrackGenConfig()
        self.clip_names = sorted(p.name for p in tracks_dir.iterdir() if (p / "tracks_360.pth").exists())
        self.clip, self.clip_name, self.statuses = None, None, {}
        self.track_idx, self.frame_idx, self.playing = 0, 0, False
        self.view = tk.StringVar(value="perspective")
        self.info = tk.StringVar()
        self._build_ui()
        self._refresh_clip_list()
        if self.clip_names:
            self.clip_list.selection_set(0)
            self._open_clip(0)

    # --- UI -------------------------------------------------------------------------------------------------------

    def _build_ui(self):
        self.master.title(f"TAPVid360 track review: {self.tracks_dir}")
        left = ttk.Frame(self.master, padding=5)
        left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="Clips (green = fully reviewed)").pack(anchor="w")
        self.clip_list = tk.Listbox(left, width=40, exportselection=False)
        self.clip_list.pack(fill=tk.Y, expand=True)
        self.clip_list.bind("<<ListboxSelect>>", lambda _: self._open_clip(self.clip_list.curselection()[0]))
        ttk.Label(left, text="Tracks").pack(anchor="w")
        self.track_list = tk.Listbox(left, width=40, height=12, exportselection=False)
        self.track_list.pack(fill=tk.X)
        self.track_list.bind("<<ListboxSelect>>", lambda _: self._select_track(self.track_list.curselection()[0]))

        right = ttk.Frame(self.master, padding=5)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls = ttk.Frame(right)
        controls.pack(fill=tk.X)
        for value, text in (("perspective", "Object-centred view"), ("equirectangular", "Equirectangular")):
            ttk.Radiobutton(controls, text=text, variable=self.view, value=value, command=self._show).pack(side=tk.LEFT)
        ttk.Button(controls, text="Accept (A)", command=lambda: self._set_status(ACCEPTED)).pack(side=tk.RIGHT)
        ttk.Button(controls, text="Reject (R)", command=lambda: self._set_status(REJECTED)).pack(side=tk.RIGHT)
        self.image_label = ttk.Label(right)
        self.image_label.pack()
        self.slider = ttk.Scale(right, from_=0, to=self.config.max_frames - 1, orient=tk.HORIZONTAL,
                                command=lambda v: self._set_frame(int(float(v))))
        self.slider.pack(fill=tk.X)
        ttk.Label(right, textvariable=self.info).pack(anchor="w")

        self.master.bind("<KeyPress>", self._on_key)

    def _on_key(self, event):
        key = event.keysym.lower()
        if key == "a":
            self._set_status(ACCEPTED)
        elif key == "r":
            self._set_status(REJECTED)
        elif key == "right":
            self._step_track(1)
        elif key == "left":
            self._step_track(-1)
        elif key == "space":
            self.playing = not self.playing
            if self.playing:
                self._play()
        elif key == "v":
            self.view.set("equirectangular" if self.view.get() == "perspective" else "perspective")
            self._show()

    def _refresh_clip_list(self):
        self.clip_list.delete(0, tk.END)
        for i, name in enumerate(self.clip_names):
            self.clip_list.insert(tk.END, name)
            self.clip_list.itemconfig(i, fg="green" if is_fully_reviewed(self.review_dir, name) else "black")

    def _refresh_track_list(self):
        self.track_list.delete(0, tk.END)
        for i in range(len(self.clip)):
            self.track_list.insert(tk.END, f"Track {i}: {self.statuses[i]}")
            self.track_list.itemconfig(i, fg=STATUS_COLOURS[self.statuses[i]])
        self.track_list.selection_clear(0, tk.END)
        self.track_list.selection_set(self.track_idx)

    # --- State ----------------------------------------------------------------------------------------------------

    def _open_clip(self, clip_idx: int):
        self.clip_name = self.clip_names[clip_idx]
        self.info.set(f"Loading {self.clip_name}...")
        self.master.update_idletasks()
        self.clip = ClipData(self.tracks_dir / self.clip_name, self.config)
        self.statuses = load_statuses(self.review_dir, self.clip_name, len(self.clip))
        unreviewed = [i for i, s in self.statuses.items() if s == NOT_REVIEWED]
        self._select_track(unreviewed[0] if unreviewed else 0)

    def _select_track(self, track_idx: int):
        self.track_idx, self.frame_idx = track_idx, 0
        self._refresh_track_list()
        self._show()

    def _step_track(self, step: int):
        """Move to the previous/next track, crossing into the neighbouring clip at either end."""
        new_idx = self.track_idx + step
        if 0 <= new_idx < len(self.clip):
            self._select_track(new_idx)
            return
        clip_idx = self.clip_names.index(self.clip_name) + step
        if 0 <= clip_idx < len(self.clip_names):
            self.clip_list.selection_clear(0, tk.END)
            self.clip_list.selection_set(clip_idx)
            self.clip_list.see(clip_idx)
            self._open_clip(clip_idx)

    def _set_status(self, status: str):
        if self.clip is None:
            return
        self.statuses[self.track_idx] = status
        self.review_dir.mkdir(parents=True, exist_ok=True)
        with open(status_file(self.review_dir, self.clip_name), "w") as f:
            json.dump(self.statuses, f, indent=4)
        if NOT_REVIEWED not in self.statuses.values():
            self._refresh_clip_list()
        self._step_track(1)

    def _set_frame(self, frame_idx: int):
        if frame_idx != self.frame_idx:
            self.frame_idx = frame_idx
            self._show()

    def _play(self):
        if not self.playing or self.clip is None:
            return
        self.frame_idx = (self.frame_idx + 1) % self.config.max_frames
        self._show()
        self.master.after(1000 // FPS, self._play)

    def _show(self):
        if self.clip is None or not len(self.clip):
            return
        img = self.clip.render(self.track_idx, self.frame_idx, self.view.get())
        # PNG through Tk's own PhotoImage (avoids PIL.ImageTk, which breaks with some Python/Tk builds).
        png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()
        self.photo = tk.PhotoImage(data=base64.b64encode(png))  # Keep a reference so it is not garbage collected.
        self.image_label.configure(image=self.photo)
        self.slider.set(self.frame_idx)
        self.info.set(f"{self.clip_name}  |  track {self.track_idx + 1}/{len(self.clip)} "
                      f"({self.statuses[self.track_idx]})  |  frame {self.frame_idx + 1}/{self.config.max_frames}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracks_dir", type=Path, required=True, help="Stage-2 output directory.")
    parser.add_argument("--review_dir", type=Path, default=None,
                        help=f"Where decisions are saved. Default: '<tracks_dir>{REVIEW_DIR_SUFFIX}'.")
    args = parser.parse_args()
    review_dir = args.review_dir or Path(str(args.tracks_dir) + REVIEW_DIR_SUFFIX)
    root = tk.Tk()
    ReviewApp(root, args.tracks_dir, review_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
