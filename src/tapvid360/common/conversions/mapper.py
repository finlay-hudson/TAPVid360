from typing import Optional

import torch

from src.tapvid360.common.conversions.convert import Conversions
from src.tapvid360.common.conversions.mapper_atomics import compute_vertical_fov, get_pixel_focal_length
from src.tapvid360.common.conversions.mapper_config import MapperConfig
from src.tapvid360.common.conversions.mappers_image import MappersImage
from src.tapvid360.common.conversions.rotations import rot


class Mappers:
    def __init__(self, crop_width: int, crop_height: int, equirectangular_height: int, f_x: Optional[float] = None,
                 fov_x: Optional[float] = None, fov_y: Optional[float] = None):
        """
        Args:

            crop_width (int): The width of the perspective image.
            crop_height (int): The height of the perspective image.
            equirectangular_height (int): The height of the equirectangular image.
            f_x (Optional[float]): The horizontal focal length in pixels. If None, it will be calculated from fov_x.
            fox_x (Optional[float]): The horizontal field of view in degrees. If None, it will be calculated from f_x.
        """
        assert fov_x is not None or f_x is not None, "Either fov_x or f_x must be provided."
        if fov_x is None:
            fov_x = torch.rad2deg(2 * torch.atan(crop_width / (2 * torch.tensor(f_x, dtype=torch.float32))))
        if fov_y is None:
            fov_y = compute_vertical_fov(fov_x, crop_width / crop_height)
        f_x, f_y = get_pixel_focal_length(crop_width, crop_height, fov_x, fov_y=fov_y)

        self.cfg = MapperConfig(
            crop_width=crop_width,
            crop_height=crop_height,
            equirectangular_width=equirectangular_height * 2,  # Assuming equirectangular width is double the height
            equirectangular_height=equirectangular_height,
            f_x=f_x,
            f_y=f_y,
            fov_x=fov_x,
            fov_y=fov_y,
        )

        self.image = MappersImage(self.cfg)
        self.point = Conversions(self.cfg)

    def compute_rotation_matrix_centred_at_point(self, cr: torch.Tensor) -> torch.Tensor:
        """
        cr: shape B, 2
        Computes the rotation matrix to center the given equirectangular coordinate in a perspective crop.
        """
        # Convert equirectangular point to a world 3D direction
        vw = self.point.cr.to_vw(cr)

        # Compute the rotation matrix to align D_initial → D_target
        # R = rotation_matrix_from_vectors(D_initial, D_target)
        # Extract pitch (elevation) and yaw (azimuth) from target vector
        pitch = -torch.atan2(-vw[..., 2], torch.sqrt(vw[..., 0] ** 2 + vw[..., 1] ** 2))
        yaw = -torch.atan2(vw[..., 1], vw[..., 0])

        # Set roll (alpha) to 0
        roll = torch.zeros_like(yaw)

        # Compute rotation matrix to align the camera with the target
        rotation_matrix = rot(roll, pitch, yaw)
        return rotation_matrix.view(*roll.shape, 3, 3)
