from dataclasses import dataclass, fields
from typing import Optional

import torch
from torchvision import transforms


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])


@dataclass(eq=False)
class DataSample:
    """
    Dataclass for storing video tracks data.
    """

    video: torch.Tensor  # B, S, C, H, W
    trajectory: torch.Tensor  # B, S, N, 2/3
    seq_name: list[str] | str = None
    # optional data
    visibility: Optional[torch.Tensor] = None  # B, S, N
    valid: Optional[torch.Tensor] = None  # B, S, N
    segmentation: Optional[torch.Tensor] = None  # B, S, 1, H, W
    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format
    equirectangular_height: Optional[int] = None
    fov_x: Optional[float] = None
    rotations: Optional[torch.Tensor] = None,  # B, S, 3, 3
    orig_image_shapes: Optional[list] = None

    def to_device(self, device: torch.device):
        """
        Moves all torch.Tensor attributes to the specified device.
        """
        # Iterate over all fields of the dataclass
        for field in fields(self):  # fields(self) gives access to the dataclass fields
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):  # If the field is a tensor
                setattr(self, field.name, value.to(device))  # Move tensor to device


def collate_fn(data: list[DataSample]):
    return DataSample(
        video=torch.stack([d.video for d in data]),
        trajectory=torch.stack([d.trajectory for d in data]),
        rotations=torch.stack([d.rotations for d in data]) if data[0].rotations is not None else None,
        seq_name=[d.seq_name for d in data],
        equirectangular_height=[d.equirectangular_height for d in data
                                ] if data[0].equirectangular_height is not None else None,
        fov_x=[d.fov_x for d in data] if data[0].fov_x is not None else None,
        segmentation=torch.stack([d.segmentation for d in data]) if data[0].segmentation is not None else None,
        visibility=torch.stack([d.visibility for d in data]) if data[0].visibility is not None else None,
    )
