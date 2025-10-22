from dataclasses import dataclass
from typing import Optional


@dataclass
class MapperConfig:
    crop_width: int
    crop_height: int
    equirectangular_width: int
    equirectangular_height: int
    f_x: float
    f_y: float
    fov_x: Optional[float]
    fov_y: Optional[float]
