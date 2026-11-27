import torch

from src.tapvid360.common.conversions.mapper import Mappers
from src.tapvid360.data_generation.sequences.main import are_points_in_frame

W, H = 320, 240


def test_point_behind_camera_is_not_in_frame():
    mapper = Mappers(W, H, equirectangular_height=512, fov_x=75.18)
    ahead = torch.tensor([[1.0, 0.1, -0.05]])
    behind = -ahead  # exactly opposite direction
    vc = torch.nn.functional.normalize(torch.cat([ahead, behind]), dim=-1)
    ij = mapper.point.vc.to_ij(vc)
    # The projection alone cannot tell them apart: both land inside the image.
    assert ((ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)).all()
    assert are_points_in_frame(vc, ij, W, H).tolist() == [True, False]


def test_points_outside_image_bounds_are_not_in_frame():
    vc = torch.nn.functional.normalize(torch.ones(4, 3), dim=-1)  # all in front of the camera
    ij = torch.tensor([[-1.0, 10.0], [10.0, -1.0], [W, 10.0], [10.0, H]])
    assert not are_points_in_frame(vc, ij, W, H).any()
