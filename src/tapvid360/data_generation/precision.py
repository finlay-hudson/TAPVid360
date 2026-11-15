"""Numerical precision used when generating TAPVid360-10k (stages 2 and 3)."""
import torch


def enable_generation_precision():
    """bfloat16 autocast (+ TF32 on Ampere or newer GPUs) for the whole process.

    TAPVid360-10k was generated like this: importing Lang-SAM's Grounding DINO module turned it on as a side effect,
    so SAM2, CoTracker3 and the perspective re-rendering all ran under it. Enabled explicitly here to match.
    """
    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
