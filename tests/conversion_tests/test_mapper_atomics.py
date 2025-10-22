import torch

from src.tapvid360.common.conversions.mapper_atomics import get_pixel_focal_length, ij_to_vc, vc_to_vw


class TestGetPixelFocalLength:
    @staticmethod
    def test_focal_length_90_deg_square():
        crop = 1000
        fov = 90.0
        f_x, f_y = get_pixel_focal_length(crop, crop, fov, fov)

        assert torch.isclose(f_x, torch.tensor(500.0), atol=1e-5)
        assert torch.isclose(f_y, torch.tensor(500.0), atol=1e-5)

    @staticmethod
    def test_focal_length_decreases_with_fov():
        crop = 1000
        fov_small = 60.0
        fov_large = 120.0

        f_x_small, _ = get_pixel_focal_length(crop, crop, fov_small)
        f_x_large, _ = get_pixel_focal_length(crop, crop, fov_large)

        assert f_x_large < f_x_small

    @staticmethod
    def test_focal_length_large_for_small_fov():
        crop = 1000
        fov_small = 10.0

        f_x, _ = get_pixel_focal_length(crop, crop, fov_small)

        assert f_x > 1000  # definitely bigger than image size


class TestPerspPointToCamera3DCoords:
    @staticmethod
    def test_center_pixel_maps_to_forward():
        width, height = 200, 400
        f_x, f_y = 90, 90
        center_pixel = torch.tensor([[width / 2, height / 2]])
        expected = torch.tensor([[1.0, 0.0, 0.0]])  # +X direction

        result = ij_to_vc(center_pixel, f_x, f_y, width, height)

        assert torch.allclose(result, expected, atol=1e-5)


class TestCamera3DCoordsToWorld3DCoords:
    @staticmethod
    def test_center_pixel_maps_to_forward():
        center_pixel = torch.tensor([[1.0, 0.0, 0.0]])
        R_w2c = torch.eye(3).unsqueeze(0)
        expected = torch.tensor([[1.0, 0.0, 0.0]])  # +X direction

        result = vc_to_vw(center_pixel, R_w2c)

        assert torch.allclose(result, expected, atol=1e-5)
