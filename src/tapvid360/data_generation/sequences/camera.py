"""Simulated camera trajectories used to resample each object track into a new narrow-FoV video (stage 3).

Each method takes the object-centred rotations (T, 3, 3) from stage 2 and returns new camera rotations (T, 3, 3),
always keeping frame 0 unchanged so the object is in view (and the query points are visible) in the first frame.
"""
import random
from enum import Enum

import torch

from src.tapvid360.common.conversions.rotations import update_rotation_matrix


class RotationMethods(Enum):
    NONE = "none"
    SPIN_X = "spin_x"
    SPIN_Y = "spin_y"
    SPIN_Z = "spin_z"
    PERSPECTIVE_SAMPLER = "persp_sampler"
    RANDOM = "random"
    SPIRAL = "spiral"
    STATIC = "static"
    REID = "REID"


# --- Samplers --------------------------------------------------------------------------------------------------------

def get_angle_steps(num_steps: int, max_noise_ratio: float = 0.1) -> torch.Tensor:
    """Per-step angles (degrees) summing to 360, with bounded zero-mean noise."""
    base_step = 360.0 / num_steps
    noise = (torch.rand(num_steps) * 2 - 1) * max_noise_ratio * base_step
    noise -= noise.mean()
    return base_step + noise


class SpinSequence:
    """Hold still for `delay_frames`, then spin a full 360° about one axis.

    Axes: "x" spins upwards (pitch), "y" sideways (roll), "z" round (yaw).
    """

    def __init__(self, axis: str = "y", delay_frames: int = 6):
        self.axis = axis
        self.add_noise = True
        self.delay_frames = delay_frames

    def get_rotations(self, num_frames, initial_rots):
        R = initial_rots[0].clone()[None]
        delay = min(self.delay_frames, num_frames)
        num_active = num_frames - delay
        if num_active > 0:
            active_steps = (get_angle_steps(num_active, max_noise_ratio=0.5) if self.add_noise
                            else torch.full((num_active,), 360 / num_active))
        else:
            active_steps = torch.tensor([])
        angle_steps = torch.cat((torch.zeros(delay), active_steps))

        rots = []
        zeros = torch.tensor([0.0])
        for i in range(num_frames):
            rots.append(R.clone())
            angle = torch.tensor([angle_steps[i]], dtype=torch.float32)
            if self.axis == "x":
                R = update_rotation_matrix(R, d_alpha=zeros, d_beta=angle, d_gamma=zeros, degrees=True)
            elif self.axis == "y":
                R = update_rotation_matrix(R, d_alpha=angle, d_beta=zeros, d_gamma=zeros, degrees=True)
            elif self.axis == "z":
                R = update_rotation_matrix(R, d_alpha=zeros, d_beta=zeros, d_gamma=angle, degrees=True)
        return torch.vstack(rots)


class CameraMotion:
    """Frame-to-frame camera motion starting from the first object-centred rotation.

    motion_type: "linear" (random pitch/yaw steps), "spiral" (steadily growing pitch/yaw), "random" (random
    pitch/yaw/roll steps) or "static". Steps are drawn from [min_angle, max_angle] degrees.
    """

    def __init__(self, motion_type: str = "linear", min_angle: int = -15, max_angle: int = 15):
        self.motion_type = motion_type
        self.min_angle = min_angle
        self.max_angle = max_angle

    def _random_step(self):
        return torch.randint(self.min_angle, self.max_angle + 1, (1,), dtype=torch.int64)

    def get_rotations(self, num_frames, initial_rots):
        R = initial_rots[0].clone()
        rots = []
        for i in range(num_frames):
            if self.motion_type == "linear":
                pitch, yaw, roll = self._random_step(), self._random_step(), 0
            elif self.motion_type == "spiral":
                step = (i * (self.max_angle - self.min_angle) / num_frames) % 360
                pitch, yaw, roll = step, step, 0
            elif self.motion_type == "random":
                pitch, yaw, roll = self._random_step(), self._random_step(), self._random_step()
            elif self.motion_type == "static":
                pitch, yaw, roll = 0, 0, 0
            else:
                raise ValueError(f"Unknown motion type: {self.motion_type}")
            R = update_rotation_matrix(R, d_alpha=pitch, d_beta=roll, d_gamma=yaw, degrees=True)
            rots.append(R)
        return torch.vstack(rots)


class PerspectiveSampler:
    """Hand-held style camera: per-frame noise + sinusoidal oscillation + linear drift in roll, pitch and yaw.

    Args (all angles in radians):
        sigma_noise: noise std for (roll, pitch, yaw).
        amp_osc: max oscillation amplitudes for (roll, pitch, yaw); sampled uniformly from [0, max].
        drift_rate: max drift rates per frame for (pitch, yaw); sampled uniformly from [-max, max].
        osc_freq_range: oscillation frequency range per frame.
        based_off_prev_rots: if True, perturb the object-centred rotation of every frame (all parameters x5);
            otherwise accumulate the motion from frame 0 alone.
        fov_range_deg: validated only (the camera FoV is fixed elsewhere).
    """

    def __init__(self, fov_range_deg=(30.0, 120.0), sigma_noise=(0.01, 0.01, 0.01), amp_osc=(0.02, 0.02, 0.04),
                 drift_rate=(0.005, 0.01), osc_freq_range=(0.05, 0.2), based_off_prev_rots: bool = False):
        self.fov_min_deg, self.fov_max_deg = fov_range_deg
        self.sigma_r, self.sigma_p, self.sigma_y = sigma_noise
        self.A_r_max, self.A_p_max, self.A_y_max = amp_osc
        self.D_p_max, self.D_y_max = drift_rate
        self.f_min, self.f_max = osc_freq_range
        self.based_off_prev_rots = based_off_prev_rots
        if based_off_prev_rots:
            self.sigma_r, self.sigma_p, self.sigma_y = self.sigma_r * 5, self.sigma_p * 5, self.sigma_y * 5
            self.A_r_max, self.A_p_max, self.A_y_max = self.A_r_max * 5, self.A_p_max * 5, self.A_y_max * 5
            self.D_p_max, self.D_y_max = self.D_p_max * 5, self.D_y_max * 5
            self.f_min, self.f_max = self.f_min * 5, self.f_max * 5

        assert 0 < self.fov_min_deg <= self.fov_max_deg < 180, "Invalid FoV range"
        assert all(s >= 0 for s in sigma_noise), "Noise sigmas must be non-negative"
        assert all(a >= 0 for a in amp_osc), "Oscillation amplitudes must be non-negative"
        assert all(d >= 0 for d in drift_rate), "Drift rates must be non-negative"
        assert 0 <= self.f_min <= self.f_max, "Invalid frequency range"

    def get_rotations(self, num_frames: int, init_rot) -> torch.Tensor:
        device = init_rot.device
        t = torch.arange(num_frames, device=device, dtype=torch.float32)

        noise_r = torch.randn(num_frames, device=device) * self.sigma_r
        noise_p = torch.randn(num_frames, device=device) * self.sigma_p
        noise_y = torch.randn(num_frames, device=device) * self.sigma_y

        A_r = torch.rand(1, device=device) * self.A_r_max
        A_p = torch.rand(1, device=device) * self.A_p_max
        A_y = torch.rand(1, device=device) * self.A_y_max
        freq = torch.rand(1, device=device) * (self.f_max - self.f_min) + self.f_min
        osc = torch.sin(freq * t)

        D_p = (torch.rand(1, device=device) * 2 - 1) * self.D_p_max
        D_y = (torch.rand(1, device=device) * 2 - 1) * self.D_y_max

        theta_roll = noise_r + A_r * osc
        theta_pitch = noise_p + A_p * osc + D_p * t
        theta_yaw = noise_y + A_y * osc + D_y * t

        rotation_matrices = init_rot.clone()
        prev_rot = init_rot[0]
        for i in range(1, num_frames):
            base = init_rot[i] if self.based_off_prev_rots else prev_rot
            rotation_matrices[i] = update_rotation_matrix(base, d_alpha=theta_pitch[i], d_beta=theta_roll[i],
                                                          d_gamma=theta_yaw[i], degrees=False)
            prev_rot = rotation_matrices[i]
        return rotation_matrices


class ReIdSampler:
    """Re-identification: mid-clip, the camera moves away from the object and then retraces its path back."""

    def __init__(self, motion_type: str = "random", min_angle: int = -15, max_angle: int = 15):
        self.motion_runner = CameraMotion(motion_type=motion_type, min_angle=min_angle, max_angle=max_angle)

    @staticmethod
    def random_even_choice(min_val, max_val):
        return random.choice(list(range(min_val + (min_val % 2 == 1), max_val + 1, 2)))

    def get_rotations(self, num_frames, initial_rots):
        new_rots = initial_rots.clone()
        min_either_side = 2
        num_frames_to_change = self.random_even_choice(num_frames // 2, num_frames - (min_either_side * 2))
        middle = num_frames // 2
        start_idx = middle - (num_frames_to_change // 2)
        end_idx = start_idx + num_frames_to_change

        fwd_rots = self.motion_runner.get_rotations(num_frames, initial_rots[start_idx:middle])[:middle - start_idx]
        rev_rots = torch.flip(fwd_rots[:-1], dims=[0])
        new_rots[start_idx:end_idx - 1] = torch.cat([fwd_rots, rev_rots], dim=0)
        return new_rots


# --- Method selection ------------------------------------------------------------------------------------------------

def select_method(random_val=None, spin_rand=None) -> RotationMethods:
    """Pick a camera method at random (used for the benchmark: one method per object track).

    Probabilities: spin 5% (z 50% / x 30% / y 20% of those), perspective sampler 30%, random 25%, spiral 2%,
    re-identification 36%, static 2%.
    """
    if random_val is None:
        random_val = random.random()
    if random_val < 0.05:
        if spin_rand is None:
            spin_rand = random.random()
        axis = "z" if spin_rand < 0.5 else ("x" if spin_rand < 0.8 else "y")
        return RotationMethods(f"spin_{axis}")
    if random_val < 0.35:
        return RotationMethods.PERSPECTIVE_SAMPLER
    if random_val < 0.60:
        return RotationMethods.RANDOM
    if random_val < 0.62:
        return RotationMethods.SPIRAL
    if random_val < 0.98:
        return RotationMethods.REID
    return RotationMethods.STATIC


def get_sampler(method: RotationMethods):
    if method == RotationMethods.NONE:
        return None
    if method in (RotationMethods.SPIN_X, RotationMethods.SPIN_Y, RotationMethods.SPIN_Z):
        return SpinSequence(axis=method.value[-1])
    if method == RotationMethods.PERSPECTIVE_SAMPLER:
        return PerspectiveSampler(fov_range_deg=(45.0, 100.0), sigma_noise=(0.01, 0.01, 0.02),
                                  amp_osc=(0.015, 0.015, 0.03), drift_rate=(0.003, 0.008),
                                  osc_freq_range=(0.08, 0.15), based_off_prev_rots=random.random() > 0.3)
    if method == RotationMethods.RANDOM:
        return CameraMotion(motion_type="random", min_angle=-10, max_angle=10)
    if method == RotationMethods.SPIRAL:
        return CameraMotion(motion_type="spiral")
    if method == RotationMethods.STATIC:
        return CameraMotion(motion_type="static")
    if method == RotationMethods.REID:
        return ReIdSampler(motion_type="random")
    raise ValueError(f"Unknown method: {method}")


def get_rotation_matrices(rot_matrices: torch.Tensor, method: RotationMethods | None = None):
    """New camera rotations (T, 3, 3) for one object track. Returns (rotations, method used)."""
    if method is None:
        method = select_method()
    sampler = get_sampler(method)
    if sampler is None:
        return rot_matrices, method
    first_rot = rot_matrices[0].clone()
    new_rot_mats = sampler.get_rotations(len(rot_matrices), rot_matrices)
    new_rot_mats[0] = first_rot  # Frame 0 always keeps the original, object-centred rotation.
    return new_rot_mats, method
