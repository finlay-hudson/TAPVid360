"""Coarse filtering of whole 360° videos (TAPVid-360 paper, Sec. 3.1, "Coarse Filtering").

Checks, in order: 360° metadata and format (poster / up-down / seam), left-right and top-bottom similarity (LPIPS),
and scene dynamics. Top-and-bottom stereo videos are cropped to their top half and other videos rescaled to 2:1.
"""
import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from src.tapvid360.data_generation.download import config
from src.tapvid360.data_generation.download.video_io import get_metadata, sample_frames

logger = logging.getLogger(__name__)

FORMAT_EQUIRECTANGULAR = "Equirectangular Format"
FORMAT_TOP_AND_BOTTOM = "top and bottom"


def check_region_black(region, black_level):
    """True if a region is predominantly black."""
    if region is None or region.size == 0:
        return False
    return np.mean(region) < black_level


def detect_bordered_video(frame, content_area_thresh=0.95, border_margin_ratio=0.05, black_level=25):
    """Detect a perspective video/image inset on a black background ("poster").

    Returns:
        (is_bordered, border_type)
    """
    detection_result, detection_type = False, "No Borders"
    if frame is None:
        return detection_result, detection_type
    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return detection_result, detection_type

    total_area = height * width
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        _, thresh = cv2.threshold(gray, black_level, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return detection_result, detection_type

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    content_area = w * h
    if content_area < total_area * 0.01 or content_area > total_area * content_area_thresh:
        return detection_result, detection_type

    margin_x, margin_y = width * border_margin_ratio, height * border_margin_ratio
    is_left_black = check_region_black(gray[:, 0:max(0, x)] if x > 0 else None, black_level)
    is_right_black = check_region_black(gray[:, min(width, x + w):width] if (x + w) < width else None, black_level)
    is_top_black = check_region_black(gray[0:max(0, y), :] if y > 0 else None, black_level)
    is_bottom_black = check_region_black(gray[min(height, y + h):height, :] if (y + h) < height else None, black_level)

    left_border = x >= margin_x and is_left_black
    right_border = (x + w) <= (width - margin_x) and is_right_black
    top_border = y >= margin_y and is_top_black
    bottom_border = (y + h) <= (height - margin_y) and is_bottom_black

    touches_top, touches_bottom = y < margin_y, (y + h) > (height - margin_y)
    touches_left, touches_right = x < margin_x, (x + w) > (width - margin_x)

    if left_border and right_border and top_border and bottom_border:
        detection_result, detection_type = True, "Full Inset"
    elif left_border and right_border and touches_top and touches_bottom:
        if not (top_border and bottom_border):
            detection_result, detection_type = True, "Vertical Bars"
    elif top_border and bottom_border and touches_left and touches_right:
        if not (left_border and right_border):
            detection_result, detection_type = True, "Horizontal Bars"
    return detection_result, detection_type


def detect_up_down_video(frame, width, height):
    """Detect a long horizontal line near the middle, as in top-and-bottom stereo videos."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=width // 6, maxLineGap=20)
    if lines is not None:
        mid_y = height // 2
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            line_y_avg = (y1 + y2) / 2
            if abs(angle) < 10 and abs(line_y_avg - mid_y) < height * 0.2 and line_length > width * 0.3:
                return True
    return False


def detect_seam(frame, strip_width=config.SEAM_STRIP_WIDTH, similarity_threshold=config.SEAM_SIMILARITY_THRESHOLD):
    """True if the left and right edges don't match (an equirectangular video should wrap around seamlessly).

    Uses normalised cross-correlation between thin strips from the left and right edges.
    """
    similarity = cv2.matchTemplate(frame[:, :strip_width], frame[:, -strip_width:], cv2.TM_CCOEFF_NORMED)[0][0]
    return not similarity > similarity_threshold


def check_equirectangular_format(frames: list, width: int, height: int, metadata: dict) -> tuple[bool, str]:
    """Check the 360° metadata, then reject posters, up-down stereo and videos with a wrap-around seam.

    A video tagged "top and bottom" in its 360° metadata passes straight away; it is cropped later.
    """
    if not frames:
        return False, "Empty frames list"

    stream = metadata["streams"][0]
    if "side_data_list" not in stream:
        return False, "No side data list"

    side_data = stream["side_data_list"]
    if side_data and side_data[0].get("type") == FORMAT_TOP_AND_BOTTOM:
        return True, FORMAT_TOP_AND_BOTTOM

    frames_to_check = min(len(frames), 10)
    frames = [frames[i] for i in np.linspace(0, len(frames) - 1, frames_to_check).astype(int)]

    poster_count, up_down_count, seam_count = 0, 0, 0
    border_type = "No Borders"
    for frame in frames:
        has_black_borders, border_type = detect_bordered_video(frame)
        poster_count += int(has_black_borders)
        up_down_count += int(detect_up_down_video(frame, width, height))
        seam_count += int(detect_seam(frame))

    if poster_count >= frames_to_check / 2:
        return False, f"Poster Format {border_type}"
    if up_down_count >= frames_to_check / 2:
        return False, "Up-Down Format"
    if seam_count >= frames_to_check / 2:
        return False, "Seam Detection"
    return True, FORMAT_EQUIRECTANGULAR


def check_intra_frame_similarity(frames: list, width: int, height: int, lpips_model, device,
                                 similarity_threshold: float = config.LPIPS_SIMILARITY_THRESHOLD) -> tuple[bool, str]:
    """Reject videos whose left/right or top/bottom halves are too similar (LPIPS), e.g. 180° or mis-tagged videos.

    Note: frames are scaled as `uint8 * 2 - 1`, i.e. NOT normalised to [-1, 1] before LPIPS. This is exactly what the
    pipeline that built TAPVid360-10k did, and `similarity_threshold` was set with it, so it is kept for fidelity.
    """
    if not frames:
        return True, None
    frames = [cv2.resize(frame, (width // 4, height // 4)) for frame in frames]
    frames_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float()
    frames_tensor = (frames_tensor * 2.0 - 1.0).to(device)

    width, height = width // 4, height // 4
    mid_w, mid_h = width // 2, height // 2
    left_half, right_half = frames_tensor[:, :, :, :mid_w], frames_tensor[:, :, :, mid_w:]
    top_half, bottom_half = frames_tensor[:, :, :mid_h, :], frames_tensor[:, :, mid_h:, :]
    if width % 2 != 0:
        left_half = torch.nn.functional.pad(left_half, (0, 1, 0, 0), mode="replicate")
    if height % 2 != 0:
        top_half = torch.nn.functional.pad(top_half, (0, 0, 0, 1), mode="replicate")

    with torch.no_grad():
        num_frames = frames_tensor.shape[0]
        dist_lr = sum(lpips_model(left_half[i:i + 1], right_half[i:i + 1]).mean().item()
                      for i in range(num_frames)) / num_frames
        dist_tb = sum(lpips_model(top_half[i:i + 1], bottom_half[i:i + 1]).mean().item()
                      for i in range(num_frames)) / num_frames

    if dist_lr < similarity_threshold or dist_tb < similarity_threshold:
        return False, "Inter Frame Similarity"
    return True, None


def check_inter_frame_dynamics(frames: list, static_threshold: float = config.STATIC_THRESHOLD) -> tuple[bool, str]:
    """Reject static videos: mean absolute difference between consecutive sampled frames is too small."""
    if len(frames) < 2:
        return False, "Not enough frames"
    variances = []
    for frame_a, frame_b in zip(frames, frames[1:]):
        diff = cv2.absdiff(cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY))
        variances.append(np.mean(diff))
    if np.mean(variances) < static_threshold:
        return False, "Inter Frame Dynamics"
    return True, None


def _ffmpeg_rewrite_in_place(video_path: Path, video_filter: str):
    """Apply an ffmpeg video filter, replacing the original file."""
    original = video_path.with_name(video_path.stem + "_old" + video_path.suffix)
    video_path.rename(original)
    subprocess.run(["ffmpeg", "-i", str(original), "-vf", video_filter, "-threads", "4",
                    "-reset_timestamps", "1", str(video_path)], check=True, capture_output=True, text=True)
    original.unlink()


def run_coarse_checks(video_path: Path, lpips_model, device) -> tuple[bool, str | None]:
    """Run every coarse check on a downloaded video, fixing its format in place if it passes.

    Returns:
        (passed, failure_reason)
    """
    metadata = get_metadata(video_path)
    frames, width, height = sample_frames(video_path, num_frames=config.NUM_SAMPLED_FRAMES)
    if not frames:
        return False, "Could not sample frames"

    is_equirectangular, format_reason = check_equirectangular_format(frames, width, height, metadata)
    if not is_equirectangular:
        return False, f"Format Check Failed: {format_reason}"

    if format_reason == FORMAT_TOP_AND_BOTTOM:
        # Keep the top half (one eye) and scale back to 2:1.
        _ffmpeg_rewrite_in_place(video_path, f"crop=in_w:in_h/2:0:0,scale={width}:{width // 2}")
    frames, width, height = sample_frames(video_path, num_frames=config.NUM_SAMPLED_FRAMES)

    passed, reason = check_intra_frame_similarity(frames, width, height, lpips_model, device)
    if not passed:
        return False, f"Intra-frame Check Failed: {reason}"
    passed, reason = check_inter_frame_dynamics(frames)
    if not passed:
        return False, f"Inter-frame Check Failed: {reason}"

    if format_reason == FORMAT_EQUIRECTANGULAR and height != width // 2:
        _ffmpeg_rewrite_in_place(video_path, f"scale={width}:{width // 2}")
    return True, None
