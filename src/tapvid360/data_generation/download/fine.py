"""Fine filtering of 10-second clips (TAPVid-360 paper, Sec. 3.1, "Fine Filtering").

Each video that passes coarse filtering is cut into 10-second clips. A clip is kept only if it has enough motion
(sparse optical flow), contains no scene change (PySceneDetect) and has no watermark (LAION watermark detector).
"""
import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image
from scenedetect import SceneManager
from scenedetect.detectors import AdaptiveDetector

from src.tapvid360.data_generation.download import config
from src.tapvid360.data_generation.download.video_io import ContainerVideoStream, open_video, sample_frames

logger = logging.getLogger(__name__)


def split_video_into_clips(video_path: Path, output_dir: Path, clip_duration: int = config.CLIP_DURATION_S):
    """Cut a video into `clip_duration`-second clips named `<stem>_clip_NNNN<ext>` (stream copy, no re-encode)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / f"{video_path.stem}_clip_%04d{video_path.suffix or '.mp4'}"
    subprocess.run(["ffmpeg", "-i", str(video_path), "-c", "copy", "-map", "0",
                    "-segment_time", str(clip_duration), "-f", "segment", "-reset_timestamps", "1",
                    str(output_pattern)], check=True, capture_output=True, text=True)


def check_low_motion_optical_flow(video_path: Path,
                                  sample_interval: int = config.OPTICAL_FLOW_SAMPLE_INTERVAL,
                                  downsample_factor: float = config.OPTICAL_FLOW_DOWNSAMPLE,
                                  motion_threshold: float = config.OPTICAL_FLOW_MOTION_THRESHOLD) -> tuple[bool, str]:
    """Mean Lucas-Kanade displacement between every `sample_interval`-th frame must exceed `motion_threshold`."""
    container = open_video(video_path)
    motion_scores = []
    prev_gray = None
    width = height = 0
    frame_count = 0  # Counts decoded frames only (empty decoder outputs are skipped without counting).
    try:
        for frame in container.decode(video=0):
            if frame is None:
                continue
            frame = np.array(frame.to_ndarray(format="rgb24"))
            if width == 0 and height == 0:
                height, width = frame.shape[:2]
                height, width = int(height * downsample_factor), int(width * downsample_factor)
            if frame_count % sample_interval == 0:
                gray = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
                    if features is not None:
                        next_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, features, None)
                        if next_features is not None:
                            magnitudes = np.linalg.norm(next_features - features, axis=2)
                            tracked = magnitudes[status == 1]
                            motion_scores.append(np.mean(tracked) if tracked.size > 0 else 0)
                prev_gray = gray
            frame_count += 1
    finally:
        container.close()

    if motion_scores and np.mean(motion_scores) > motion_threshold:
        return True, None
    return False, "Optical Flow"


def detect_scenes(video_path: Path) -> list:
    """Scene changes found by PySceneDetect's AdaptiveDetector (empty list = no cut)."""
    container = open_video(video_path)
    try:
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector())
        scene_manager.detect_scenes(video=ContainerVideoStream(container, container.streams.video[0]))
        return scene_manager.get_scene_list()
    finally:
        container.close()


class WatermarkDetector:
    """LAION watermark classifier (EfficientNet-B3), weights from `finetrainers/laion-watermark-detection`."""

    def __init__(self, device="cpu", model_path: str | None = None):
        self.device = device
        self.transforms = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=1536, out_features=625), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256), nn.ReLU(),
            nn.Linear(in_features=256, out_features=2),
        )
        if model_path is None:
            model_path = hf_hub_download("finetrainers/laion-watermark-detection", "watermark_model_v1.pt")
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval().to(device)

    @torch.no_grad()
    def watermark_probability(self, images: list[Image.Image]) -> torch.Tensor:
        pixel_values = torch.stack([self.transforms(image) for image in images]).to(self.device)
        return nn.functional.softmax(self.model(pixel_values), dim=1)[:, 0]

    def has_watermark(self, video_path: Path) -> bool:
        frames, _, _ = sample_frames(video_path, num_frames=config.WATERMARK_NUM_FRAMES)
        images = [Image.fromarray(frame) for frame in frames]
        return self.watermark_probability(images).mean().item() > config.WATERMARK_THRESHOLD


def run_fine_checks(video_id: str, clips_dir: Path, filtered_clips_dir: Path, failed_fine_dir: Path,
                    watermark_detector: WatermarkDetector) -> tuple[bool, int | str]:
    """Filter every clip of one video, moving survivors to `filtered_clips_dir/<video_id>/`.

    Returns:
        (True, number_of_kept_clips) or (False, failure_reason)
    """
    clips_sub_dir = clips_dir / video_id
    kept_dir = filtered_clips_dir / video_id
    failed_dir = failed_fine_dir / video_id
    kept_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    def reject(clip_path: Path):
        if config.STORE_FAILED:
            shutil.move(str(clip_path), str(failed_dir / clip_path.name))
        else:
            clip_path.unlink()

    for clip_path in sorted(clips_sub_dir.iterdir()):
        has_motion, _ = check_low_motion_optical_flow(clip_path)
        if not has_motion:
            reject(clip_path)
            continue
        if len(detect_scenes(clip_path)) != 0:
            reject(clip_path)
            continue
        if watermark_detector.has_watermark(clip_path):
            reject(clip_path)
            continue
        shutil.move(str(clip_path), str(kept_dir / clip_path.name))

    clips_sub_dir.rmdir()
    if failed_dir.exists() and not any(failed_dir.iterdir()):
        failed_dir.rmdir()
    num_kept = len(list(kept_dir.iterdir()))
    if num_kept == 0:
        kept_dir.rmdir()
        return False, "All Clips Failed Fine Processing"
    return True, num_kept
