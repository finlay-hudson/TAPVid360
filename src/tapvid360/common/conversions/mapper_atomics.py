import math
from typing import Optional

import torch


def ij_to_vc(ij, f_x, f_y, W_persp, H_persp):
    """
    Args:
        ij is perspective pixel coordinates of shape *, 2
        f_x: focal length in pixels (x-direction)
        f_y: focal length in pixels (y-direction)
        W_persp: width of the perspective image
        H_persp: height of the perspective image
    Returns:
        vc: unit vectors in camera coordinate system of shape *,3
    """
    # Calculate the 3D coordinates of each pixel in the perspective image
    y_3d_perspective = (ij[..., 0] - W_persp / 2) / f_x
    z_3d_perspective = -(ij[..., 1] - H_persp / 2) / f_y
    x_3d_perspective = torch.ones_like(y_3d_perspective)
    # Stack the 3D coordinates and reshape to match the rotation matrix dimensions
    xyzc = torch.stack([x_3d_perspective, y_3d_perspective, z_3d_perspective], dim=-1)  # *, 3
    vc = torch.nn.functional.normalize(xyzc, dim=-1)
    return vc


def vc_to_ij(vc, f_x, f_y, W_persp, H_persp):
    """
    Args:
        vc: unit vectors in camera coordinate system of shape *,3
        f_x: focal length in pixels (x-direction)
        f_y: focal length in pixels (y-direction)
        W_persp: width of the perspective image
        H_persp: height of the perspective image
    Returns:
        ij: pixel coordinates in the perspective image of shape *, 2
    """
    # Decompose vc into x, y, z
    x = vc[..., 0]
    y = vc[..., 1]
    z = vc[..., 2]

    # Avoid division by zero
    eps = 1e-8
    x = torch.where(torch.abs(x) < eps, eps * torch.sign(x), x)

    # Reconstruct i and j
    i = y / x * f_x + W_persp / 2
    j = -z / x * f_y + H_persp / 2

    return torch.stack([i, j], dim=-1)


def vc_to_vw(vc, R_w2c):
    """
    Args:
        vc is unit vectors in camera coordinate system of shape *,3
        R_w2c is rotation matrix from camera to world coordinate system of shape *,3,3
    """
    # Get the shapes of the matrix and vector tensors
    r_shape = R_w2c.shape
    v_shape = vc.shape

    # The matrix part is the last 2 dims (3, 3)
    # The vector part is the last dim (3)

    # Calculate how many extra dimensions the vector tensor has
    num_extra_dims = len(v_shape) - len(r_shape) + 1

    # Construct the new shape for the matrix tensor.
    # This inserts (1,) for each extra dimension.
    # e.g., (B, N, 3, 3) -> (B, N, 1, 1, 3, 3)
    new_r_shape = r_shape[:-2] + (1,) * num_extra_dims + r_shape[-2:]

    # Reshape the matrix tensor to be broadcastable
    R_w2c_reshaped = R_w2c.view(new_r_shape)

    # Perform the einsum, which now works correctly
    xyzc = torch.einsum("...ij,...j->...i", torch.linalg.inv(R_w2c_reshaped), vc)

    return torch.nn.functional.normalize(xyzc, dim=-1)


def vw_to_vc(vw, R_w2c):
    """
    Transforms vectors from world to camera coordinates with robust broadcasting.

    Args:
        vw (torch.Tensor): Unit vectors in world coordinate system of shape (*, 3).
        R_w2c (torch.Tensor): Rotation matrices from world to camera system of shape (*, 3, 3).
    """
    # Get the shapes of the matrix and vector tensors
    r_shape = R_w2c.shape
    v_shape = vw.shape

    # Calculate how many extra leading dimensions the vector tensor has
    # e.g., v_shape=(B, N, 3), r_shape=(B, 3, 3) -> len(v_shape)=3, len(r_shape)=3 -> num_extra_dims=1
    num_extra_dims = len(v_shape) - len(r_shape) + 1

    # Construct the new shape for the matrix tensor.
    # This inserts singleton dimensions to align with the vector's extra dims.
    # e.g., for R_w2c shape (B, 3, 3), new shape becomes (B, 1, 3, 3)
    # to broadcast against a vector of shape (B, N, 3).
    new_r_shape = r_shape[:-2] + (1,) * num_extra_dims + r_shape[-2:]

    # Reshape the matrix tensor to be broadcastable
    R_w2c_reshaped = R_w2c.view(new_r_shape)

    # Perform the einsum, which now broadcasts correctly.
    # This is the forward transformation, so we use R_w2c directly.
    xyzc = torch.einsum("...ij,...j->...i", R_w2c_reshaped, vw)

    return torch.nn.functional.normalize(xyzc, dim=-1)


def cr_to_uv(cr, H_equi):
    """
    Args:
        cr: pixel coordinates in the equirectangular image of shape *, 2
        H_equi is height of equirectangular image
    Returns:
        uv: normalised coordinates in equirectangular image of shape *, 2
    """
    u = (cr[..., 0] - 0.5) / H_equi
    v = (cr[..., 1] - 0.5) / H_equi
    uv = torch.stack([u, v], dim=-1)
    return uv


def uv_to_cr(uv, H_equi):
    """
    Args:
        uv: normalised coordinates in equirectangular image of shape *, 2
        H_equi: height of the equirectangular image
        W_equi: width of the equirectangular image
    Returns:
        cr: pixel coordinates in the equirectangular image of shape *, 2
    """
    c = uv[..., 0] * H_equi + 0.5
    r = uv[..., 1] * H_equi + 0.5
    return torch.stack([c, r], dim=-1)


def thetaphi_to_vw(thetaphi):
    """
    Converts spherical coordinates (theta, phi) to 3D unit direction vectors in world coordinates.

    Args:
        thetaphi: Tensor of shape (*, 2), where:
            - thetaphi[..., 0] = theta (azimuthal angle) in radians, ranging from [-π, π]
            - thetaphi[..., 1] = phi (elevation angle) in radians, ranging from [-π/2, π/2]

    Returns:
        vw: Tensor of shape (*, 3), unit 3D direction vectors in world coordinates.
    """
    theta = thetaphi[..., 0]
    phi = thetaphi[..., 1]
    xv = torch.cos(phi) * torch.cos(theta)
    yv = torch.cos(phi) * torch.sin(theta)
    zv = -torch.sin(phi)
    xyzv = torch.stack((xv, yv, zv), dim=-1)

    return torch.nn.functional.normalize(xyzv, dim=-1)


def vw_to_thetaphi(vw):
    """
    Args:
        vw: Tensor of shape (..., 3), unit direction vectors in world coordinates.
    Returns:
        theta: Tensor of shape (...,), azimuth angle in radians in range [-π, π].
               Measured counterclockwise from the x-axis in the x-y plane.
        phi:   Tensor of shape (...,), elevation angle in radians in range [-π/2, π/2].
               Zero at the horizon, positive above, negative below.
    """
    xw = vw[..., 0]
    yw = vw[..., 1]
    zw = vw[..., 2]
    theta = torch.atan2(yw, xw)
    phi = torch.atan2(-zw, torch.sqrt(xw ** 2 + yw ** 2))

    return torch.stack([theta, phi], dim=-1)


def thetaphi_to_uv(thetaphi):
    """
    Converts spherical coordinates (theta, phi) to normalized coordinates
    in an equirectangular projection.

    Args:
        thetaphi: Tensor of shape (..., 2), where:
            - theta: azimuth angle in radians, in range [-π, π]
            - phi: elevation angle in radians, in range [-π/2, π/2]

    Returns:
        uv: Tensor of shape (..., 2), normalized coordinates in [0, 1):
            - u corresponds to horizontal position (longitude)
            - v corresponds to vertical position (latitude)
    """
    theta = thetaphi[..., 0]
    phi = thetaphi[..., 1]

    u = (theta + torch.pi) / torch.pi
    v = (phi + torch.pi / 2) / torch.pi

    return torch.stack([u, v], dim=-1)


def uv_to_thetaphi(uv):
    """
    Converts normalized coordinates in an equirectangular image to spherical coordinates (theta, phi).

    Args:
        uv: Tensor of shape (..., 2), normalized coordinates in [0, 1)

    Returns:
        thetaphi: Tensor of shape (..., 2), spherical coordinates in radians:
            - theta ∈ [-π, π] (azimuth)
            - phi ∈ [-π/2, π/2] (elevation)
    """
    u = uv[..., 0]
    v = uv[..., 1]

    theta = u * torch.pi - torch.pi
    phi = v * torch.pi - torch.pi / 2

    return torch.stack([theta, phi], dim=-1)


def compute_vertical_fov(horizontal_fov_deg: float, aspect_ratio: float):
    fov_x_rad = math.radians(horizontal_fov_deg)
    fov_y_rad = 2 * math.atan(math.tan(fov_x_rad / 2) / aspect_ratio)
    return math.degrees(fov_y_rad)


def get_pixel_focal_length(crop_width: int, crop_height: int, fov_x: float, fov_y: Optional[float] = None):
    if fov_y is None:
        fov_y = compute_vertical_fov(fov_x, crop_width / crop_height)
        # fov_y = float(crop_height) / crop_width * fov_x
    # Convert FOV from degrees to radians
    fov_x_rad = torch.deg2rad(torch.tensor(fov_x))
    fov_y_rad = torch.deg2rad(torch.tensor(fov_y))
    # Calculate the focal lengths in pixels
    f_x = crop_width / (2 * torch.tan(fov_x_rad / 2))
    f_y = crop_height / (2 * torch.tan(fov_y_rad / 2))

    return f_x, f_y
