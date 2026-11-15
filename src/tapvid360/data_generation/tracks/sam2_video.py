"""SAM2 helpers. The video predictor is patched to take frames from memory instead of a directory of JPEGs."""
import types
from collections import OrderedDict
from pathlib import Path

import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM2_MODEL_CFG = "sam2_hiera_l.yaml"


def get_sam2_predictor(single_image: bool, checkpoint: str | Path, device: torch.device):
    """SAM2 Hiera-L image predictor, or video predictor that accepts in-memory frames."""
    if single_image:
        return SAM2ImagePredictor(build_sam2(SAM2_MODEL_CFG, str(checkpoint), device=device))
    predictor = build_sam2_video_predictor(SAM2_MODEL_CFG, str(checkpoint), device=device)
    predictor.init_state = types.MethodType(_init_state_from_frames, predictor)
    return predictor


@torch.inference_mode()
def _init_state_from_frames(self, images, video_height, video_width, compute_device,
                            offload_video_to_cpu=False, offload_state_to_cpu=False):
    """Replacement for SAM2VideoPredictor.init_state taking preprocessed frames (T, 3, S, S) directly."""
    inference_state = {
        "images": images,
        "num_frames": len(images),
        "offload_video_to_cpu": offload_video_to_cpu,
        "offload_state_to_cpu": offload_state_to_cpu,
        "video_height": video_height,
        "video_width": video_width,
        "device": compute_device,
        "storage_device": torch.device("cpu") if offload_state_to_cpu else compute_device,
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
        "tracking_has_started": False,
        "frames_already_tracked": {},
        "frames_tracked_per_obj": {},
    }
    # Warm up the visual backbone and cache the features of frame 0.
    self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state


def sam_preprocessing(frames: torch.Tensor, masks: torch.Tensor, resize_to: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize and ImageNet-normalise (T, H, W, 3) uint8 frames, and resize (N, 1, H, W) masks to match."""
    if not len(frames):
        raise ValueError("No frames given")
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    frames_scaled = torch.nn.functional.interpolate(frames.permute(0, 3, 1, 2) / 255.0, size=resize_to,
                                                    mode="bilinear", align_corners=False)
    frames_scaled = (frames_scaled - img_mean.to(frames_scaled.device)) / img_std.to(frames_scaled.device)
    masks_scaled = torch.nn.functional.interpolate(masks.float(), size=resize_to, mode="bilinear",
                                                   align_corners=False)[:, 0]
    return frames_scaled, masks_scaled


def run_through_sam_video(sam_pred_video, frames: torch.Tensor, masks: torch.Tensor):
    """Propagate frame-0 masks through the video.

    Returns:
        (logits, confidences), each (num_masks, T, H, W).
    """
    sam_frames, sam_masks = sam_preprocessing(frames, masks, resize_to=(sam_pred_video.image_size,
                                                                        sam_pred_video.image_size))
    inference_state = sam_pred_video.init_state(sam_frames, frames[0].shape[0], frames[0].shape[1],
                                                sam_pred_video.device)
    for obj_id, mask in enumerate(sam_masks, start=1):
        sam_pred_video.add_new_mask(inference_state, 0, obj_id, mask)
    out_logits = [logits for _, _, logits in sam_pred_video.propagate_in_video(inference_state)]
    assert len(out_logits), "No masks generated"
    confidences = [torch.sigmoid(logits) for logits in out_logits]
    return (torch.stack(out_logits)[:, :, 0].transpose(1, 0),
            torch.stack(confidences)[:, :, 0].transpose(1, 0))
