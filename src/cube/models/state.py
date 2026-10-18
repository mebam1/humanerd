"""Differentiable MuJoCo free-root/hinge state algebra (wxyz quaternions)."""
from __future__ import annotations

import torch
from torch import Tensor


def quat_normalize(q: Tensor) -> Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)


def quat_conjugate(q: Tensor) -> Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_multiply(a: Tensor, b: Tensor) -> Tensor:
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv)), -1)


def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    q = quat_normalize(q)
    xyz = q[..., 1:]
    uv = torch.linalg.cross(xyz.expand_as(v), v)
    return v + 2 * (q[..., :1] * uv + torch.linalg.cross(xyz.expand_as(v), uv))


def rotvec_to_quat(r: Tensor) -> Tensor:
    angle = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
    # sinc is analytic at zero; unlike a raw r / ||r|| expression it has finite gradients.
    scale = 0.5 * torch.sinc(angle / (2 * torch.pi))
    return quat_normalize(torch.cat((torch.cos(angle * 0.5), r * scale), -1))


def quat_to_rotvec(q: Tensor) -> Tensor:
    q = quat_normalize(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    norm = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    scale = 2 * torch.atan2(norm, q[..., :1].clamp_min(1e-8)) / norm.clamp_min(1e-8)
    return q[..., 1:] * torch.where(norm < 1e-7, 2 * torch.ones_like(scale), scale)


def state_dimensions(state: Tensor) -> tuple[int, int]:
    dim = state.shape[-1]
    if dim < 13 or (dim - 13) % 2:
        raise ValueError(f"Expected [qpos,qvel] free-root/hinge state 13+2J, got {dim}")
    joints = (dim - 13) // 2
    return 7 + joints, joints


def state_delta(state: Tensor, following: Tensor) -> Tensor:
    """Return [dx_local, rotation_vector, dv_local, dw_local, dqj, dvj].

    Linear and angular increments refer to the CURRENT root frame. MuJoCo's
    free angular velocity is stored in the body frame at its own timestep.
    Consequently we rotate the following angular velocity into the current
    frame before subtracting it. Quaternions q and -q yield identical targets.
    """
    nq, joints = state_dimensions(state)
    q = quat_normalize(state[..., 3:7])
    qn = quat_normalize(following[..., 3:7])
    inv = quat_conjugate(q)
    relative = quat_multiply(inv, qn)
    parts = [quat_rotate(inv, following[..., :3] - state[..., :3]),
             quat_to_rotvec(relative),
             quat_rotate(inv, following[..., nq:nq+3] - state[..., nq:nq+3]),
             quat_rotate(relative, following[..., nq+3:nq+6]) - state[..., nq+3:nq+6]]
    if joints:
        parts += [following[..., 7:nq] - state[..., 7:nq],
                  following[..., nq+6:] - state[..., nq+6:]]
    return torch.cat(parts, -1)


def apply_delta(state: Tensor, delta: Tensor) -> Tensor:
    """Integrate a learned generalized delta without detaching the rollout graph."""
    nq, joints = state_dimensions(state)
    if delta.shape[-1] != 12 + 2 * joints:
        raise ValueError("Delta width does not match state")
    q = quat_normalize(state[..., 3:7])
    relative = rotvec_to_quat(delta[..., 3:6])
    qn = quat_normalize(quat_multiply(q, relative))
    position = state[..., :3] + quat_rotate(q, delta[..., :3])
    velocity = state[..., nq:nq+3] + quat_rotate(q, delta[..., 6:9])
    angular = quat_rotate(quat_conjugate(relative), state[..., nq+3:nq+6] + delta[..., 9:12])
    return torch.cat((position, qn, state[..., 7:nq] + delta[..., 12:12+joints],
                      velocity, angular, state[..., nq+6:] + delta[..., 12+joints:]), -1)


def rotation_error(predicted: Tensor, target: Tensor) -> Tensor:
    """Shortest SO(3) geodesic angle in radians, invariant to quaternion sign."""
    return torch.linalg.vector_norm(quat_to_rotvec(quat_multiply(
        quat_conjugate(predicted[..., 3:7]), target[..., 3:7])), dim=-1)
