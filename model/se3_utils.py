from __future__ import annotations

import torch


def _skew(vec: torch.Tensor) -> torch.Tensor:
    x, y, z = vec.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    axis_angle = axis_angle.to(torch.float32)
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    safe_theta = theta.clamp_min(1e-8)
    axis = axis_angle / safe_theta
    k = _skew(axis)

    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.expand(axis_angle.shape[:-1] + (3, 3))

    theta_expand = theta.unsqueeze(-1)
    sin_term = torch.sin(theta_expand)
    cos_term = 1.0 - torch.cos(theta_expand)
    kk = k @ k
    return eye + sin_term * k + cos_term * kk


def world_to_local_vector(vec: torch.Tensor, axis_angle: torch.Tensor) -> torch.Tensor:
    rot = axis_angle_to_matrix(axis_angle)
    return torch.matmul(rot.transpose(-1, -2), vec.unsqueeze(-1)).squeeze(-1)


def local_to_world_vector(vec: torch.Tensor, axis_angle: torch.Tensor) -> torch.Tensor:
    rot = axis_angle_to_matrix(axis_angle)
    return torch.matmul(rot, vec.unsqueeze(-1)).squeeze(-1)


def world_to_local_motion(motion: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    motion_local = motion.clone()
    axis_angle = state[..., 3:6]
    motion_local[..., 0:3] = world_to_local_vector(motion[..., 0:3], axis_angle)
    if motion.shape[-1] >= 6:
        motion_local[..., 3:6] = world_to_local_vector(motion[..., 3:6], axis_angle)
    return motion_local


def local_to_world_motion(motion: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    motion_world = motion.clone()
    axis_angle = state[..., 3:6]
    motion_world[..., 0:3] = local_to_world_vector(motion[..., 0:3], axis_angle)
    if motion.shape[-1] >= 6:
        motion_world[..., 3:6] = local_to_world_vector(motion[..., 3:6], axis_angle)
    return motion_world
