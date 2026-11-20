"""Video helpers shared by the tracking (stage 2) and sequence (stage 3) steps."""
import os
import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
from decord import VideoReader, cpu

# Equirectangular resolution the tracking stage works at. Source clips are resized to this on load.
EQ_WIDTH, EQ_HEIGHT = 2048, 1024


class EquirectangularVideoReader:
    """Reads frames from an equirectangular video, resized to EQ_WIDTH x EQ_HEIGHT."""

    def __init__(self, video_path: Path | str, width: int = EQ_WIDTH, height: int = EQ_HEIGHT):
        self.vr = VideoReader(str(video_path), ctx=cpu(0), width=width, height=height)

    def count_frames(self) -> int:
        return len(self.vr)

    def load_frames(self, frame_indices: list[int]) -> np.ndarray:
        """Frames as a (T, H, W, 3) uint8 RGB array."""
        return self.vr.get_batch(frame_indices).asnumpy()


def count_frames_pyav(video_path: Path | str) -> int:
    """Number of video frames, counted with PyAV (fallback for clips decord cannot read)."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return stream.frames
        return sum(1 for packet in container.demux(stream) if packet.size)


def read_frames_pyav(video_path: Path | str, frame_indices: list[int], width: int = EQ_WIDTH,
                     height: int = EQ_HEIGHT) -> np.ndarray:
    """Frames at `frame_indices` as a (T, H, W, 3) uint8 RGB array, resized to width x height, decoded with PyAV.

    Fallback for clips decord cannot decode (decord 0.6.0 fails on YouTube's 4K VP9 streams, for example).
    """
    wanted = set(frame_indices)
    frames = {}
    with av.open(str(video_path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                frames[i] = cv2.resize(frame.to_ndarray(format="rgb24"), (width, height), interpolation=cv2.INTER_AREA)
            if i >= max(frame_indices):
                break
    missing = wanted - frames.keys()
    if missing:
        raise RuntimeError(f"{video_path}: could not decode frames {sorted(missing)[:5]}")
    return np.stack([frames[i] for i in frame_indices])


def encode_frames_to_video_bytes(frames: np.ndarray, fps: int = 30) -> bytes:
    """Compress (T, H, W, 3) RGB frames to MP4 (mp4v) bytes, so the frames can be stored next to their tracks."""
    if not len(frames):
        return b""
    height, width, _ = frames[0].shape
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_filename = tmp.name
    try:
        writer = cv2.VideoWriter(temp_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        with open(temp_filename, "rb") as f:
            return f.read()
    finally:
        os.remove(temp_filename)


def decode_video_bytes_to_frames(video_bytes: bytes) -> np.ndarray:
    """Inverse of `encode_frames_to_video_bytes`: all frames as a (T, H, W, 3) uint8 RGB array."""
    if not video_bytes:
        return np.empty((0,), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_filename = tmp.name
        tmp.write(video_bytes)
    try:
        vr = VideoReader(temp_filename, ctx=cpu(0))
        return vr.get_batch(list(range(len(vr)))).asnumpy()
    finally:
        os.remove(temp_filename)
