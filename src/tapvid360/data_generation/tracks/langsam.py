"""Lang-SAM style detection (Grounding DINO boxes -> SAM2 masks) for finding moving objects by text prompt.

Based on Lang-SAM (https://github.com/luca-medeiros/lang-segment-anything, Apache-2.0), which TAPVid360-10k was
generated with; its Grounding DINO wrapper is reproduced here so the package (which pins old torch versions and
Python >= 3.11) is not needed.
"""
from typing import Optional

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

# Text prompt: object categories that typically move.
MOVING_OBJECTS = [
    "person", "bird", "fish", "insect", "dog", "cat", "horse", "snake", "animal",
    "car", "bike", "motorcycle", "train", "airplane", "boat", "ship", "helicopter", "submarine", "rocket",
    "bus", "truck", "robot", "drone", "conveyor belt", "wind turbine", "fan", "clock hands", "gears",
    "ball", "frisbee", "pendulum", "swing", "yo-yo", "kite", "shopping cart",
]
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.15


class GDINO:
    """Grounding DINO (base) through Hugging Face transformers, as used by Lang-SAM."""

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID)


class LangSamRunner:
    def __init__(self, image_sam: SAM2ImagePredictor):
        self.gdino = GDINO()
        self.gdino.model = self.gdino.model.to(image_sam.device)
        self.image_sam = image_sam

    def _run_gdino(self, images: torch.Tensor, text_prompt: list[str]):
        inputs = self.gdino.processor(images=images, text=text_prompt, return_tensors="pt").to(images.device)
        with torch.no_grad():
            outputs = self.gdino.model(**inputs)
        return self.gdino.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=torch.stack([torch.tensor([images.shape[1], images.shape[2]])] * images.shape[0]),
        )

    def run_custom(self, images: torch.Tensor, prompt: Optional[list[str]] = None) -> list[dict]:
        """Detect and segment objects in (B, H, W, 3) uint8 images.

        Returns one dict per image with Grounding DINO's boxes/scores/labels plus "masks" and "mask_scores"
        ({"masks": []} when nothing is found).
        """
        text_prompt = ". ".join(prompt if prompt is not None else MOVING_OBJECTS) + "."
        gdino_results = self._run_gdino(images, [text_prompt] * len(images))
        final_results = []
        for i, gdino_result in enumerate(gdino_results):
            if not len(gdino_result["boxes"]):
                final_results.append({"masks": []})
                continue
            masks, mask_scores = self._run_sam(gdino_result["boxes"], images[i:i + 1])[:2]
            gdino_result["masks"], gdino_result["mask_scores"] = masks, mask_scores
            assert len(masks) == len(gdino_result["boxes"])
            final_results.append(gdino_result)
        return final_results

    def _run_sam(self, bboxes: torch.Tensor, images: torch.Tensor):
        """Box-prompted SAM2 on one (1, H, W, 3) image, using SAM2ImagePredictor internals to stay on the GPU."""
        sam = self.image_sam
        sam.reset_predictor()
        sam._orig_hw = [images.shape[1:3]] * len(images)
        sam._is_batch = False
        input_image = sam._transforms.transforms(images.permute(0, 3, 1, 2) / 255.0)
        assert len(input_image.shape) == 4 and input_image.shape[1] == 3, \
            f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        backbone_out = sam.model.forward_image(input_image)
        _, vision_feats, _, _ = sam.model._prepare_backbone_features(backbone_out)
        if sam.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + sam.model.no_mem_embed
        feats = [feat.permute(1, 2, 0).view(1, -1, *feat_size)
                 for feat, feat_size in zip(vision_feats[::-1], sam._bb_feat_sizes[::-1])][::-1]
        sam._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        sam._is_image_set = True
        prep_bboxes = sam._prep_prompts(None, None, bboxes, None, True)[-1]
        return sam._predict(None, None, boxes=prep_bboxes, multimask_output=False)

    @staticmethod
    def get_top_k_masks_from_result(single_result: dict, top_k: int = 10) -> dict:
        """Keep the `top_k` highest-scoring detections."""
        top_idxs = torch.argsort(single_result["scores"], descending=True)[:top_k]
        return {k: v[top_idxs] if isinstance(v, torch.Tensor) else [single_result["labels"][idx] for idx in top_idxs]
                for k, v in single_result.items()}

    def to_device(self, device):
        self.gdino.model = self.gdino.model.to(device)
        self.image_sam.model = self.image_sam.model.to(device)
