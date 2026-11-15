"""Mask and point helpers for stage 2."""
import torch


def find_mask_idx_closest_to_the_centre_of_img(masks: torch.Tensor, h: int = 256, w: int = 256) -> int:
    """Index of the mask (B, 1, H, W) with a pixel nearest the image centre."""
    B = masks.shape[0]
    device = masks.device
    center = torch.tensor([h // 2, w // 2], device=device)
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    distances = torch.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).unsqueeze(0).unsqueeze(0)
    masked_distances = torch.where(masks.bool(), distances, torch.tensor(float("inf"), device=device))
    min_distances, _ = masked_distances.view(B, -1).min(dim=1)
    return torch.argmin(min_distances).item()


def get_distributed_points(mask: torch.Tensor, num_points: int = 32, seed: int = 42) -> torch.Tensor:
    """`num_points` (x, y) points spread across a binary (H, W) mask's foreground.

    Note: seeds torch's global RNG with `seed` on every call, so sampling is deterministic.
    """
    if mask.dim() != 2:
        raise ValueError("Mask must be a 2D tensor of shape (H, W).")
    device = mask.device
    torch.manual_seed(seed)
    y, x = torch.where(mask > 0)
    if y.numel() == 0:
        raise ValueError("Mask has no foreground pixels!")
    indices = torch.randperm(y.numel(), device=device)
    selected = torch.linspace(0, y.numel() - 1, steps=num_points, dtype=torch.long, device=device)
    return torch.stack((x[indices[selected]], y[indices[selected]]), dim=1)


def compute_center_of_mass(mask: torch.Tensor) -> torch.Tensor:
    """(x, y) centre of mass of each mask in a (B, H, W) batch."""
    assert mask.ndim == 3, "Mask must be a 3D tensor (B, H, W)"
    B, h, w = mask.shape
    y_coords = torch.arange(h, device=mask.device, dtype=torch.float32).view(1, h, 1).expand(B, h, w)
    x_coords = torch.arange(w, device=mask.device, dtype=torch.float32).view(1, 1, w).expand(B, h, w)
    total_mass = mask.sum(dim=(1, 2), keepdim=True)
    x_com = (mask * x_coords).sum(dim=(1, 2)) / total_mass.squeeze()
    y_com = (mask * y_coords).sum(dim=(1, 2)) / total_mass.squeeze()
    return torch.stack((x_com, y_com), dim=1)


def erode_batch(masks: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """Erode (B, 1, H, W) or (B, H, W) binary masks with min-pooling; returns (B, 1, H, W)."""
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    masks = masks.float()
    pad = kernel_size // 2
    for _ in range(iterations):
        masks = -torch.nn.functional.max_pool2d(-masks, kernel_size=kernel_size, stride=1, padding=pad)
    return masks


def find_points_in_masks(points: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Whether each (x, y) point (N, T, P, 2) falls inside its frame's mask (N, T, H, W); returns (N, T, P)."""
    N, num_frames, num_points, _ = points.shape
    _, _, H, W = masks.shape
    masks_reshaped = masks.reshape(N * num_frames, 1, H, W).float()
    points_reshaped = points.reshape(N * num_frames, 1, num_points, 2)
    norm_factors = torch.tensor([2.0 / (W - 1), 2.0 / (H - 1)], device=points.device)
    points_normalised = points_reshaped * norm_factors - 1.0
    sampled = torch.nn.functional.grid_sample(masks_reshaped, points_normalised, mode="nearest",
                                              padding_mode="zeros", align_corners=False)
    return sampled.reshape(N, num_frames, num_points) > 0.5
