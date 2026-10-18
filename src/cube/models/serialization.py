"""Integer 3D Hilbert serialization and auditable local-attention coverage."""
from __future__ import annotations

import torch
from torch import Tensor


def quantize_points(points: Tensor, bits: int = 8) -> Tensor:
    """One isotropic grid for dynamic AND static surfaces in each batch member."""
    if not 1 <= bits <= 20:
        raise ValueError("bits must be between 1 and 20 (int64 3D keys)")
    lower = points.amin(dim=-2, keepdim=True)
    span = (points.amax(dim=-2, keepdim=True) - lower).amax(-1, keepdim=True).clamp_min(1e-6)
    return ((points - lower) / span * ((1 << bits) - 1)).floor().long().clamp(0, (1 << bits)-1)


def hilbert_keys(coordinates: Tensor, bits: int = 8) -> Tensor:
    """Skilling transpose algorithm: actual Hilbert distance, not Morton renamed.

    Integer sorting is piecewise constant; selected feature values and their
    physical coordinates retain their autograd graph after gathering.
    """
    x = coordinates.clone().long()
    top = 1 << (bits - 1)
    q = top
    while q > 1:
        p = q - 1
        for axis in range(3):
            mask = (x[..., axis] & q) != 0
            exchange = (x[..., 0] ^ x[..., axis]) & p
            first = torch.where(mask, x[..., 0] ^ p, x[..., 0] ^ exchange)
            other = torch.where(mask, x[..., axis], x[..., axis] ^ exchange)
            if axis == 0:
                x[..., 0] = first
            else:
                x[..., 0], x[..., axis] = first, other
        q >>= 1
    x[..., 1] ^= x[..., 0]
    x[..., 2] ^= x[..., 1]
    t = torch.zeros_like(x[..., 0])
    q = top
    while q > 1:
        t ^= torch.where((x[..., 2] & q) != 0, q - 1, 0)
        q >>= 1
    x ^= t.unsqueeze(-1)
    return morton_keys(x, bits)


def morton_keys(coordinates: Tensor, bits: int = 8) -> Tensor:
    key = torch.zeros_like(coordinates[..., 0], dtype=torch.long)
    for bit in range(bits):
        for axis in range(3):
            key |= ((coordinates[..., axis] >> bit) & 1) << (3 * bit + 2 - axis)
    return key


def serialization_order(points: Tensor, kind: str = "hilbert", bits: int = 8) -> Tensor:
    grid = quantize_points(points.detach(), bits)
    if kind == "hilbert":
        keys = hilbert_keys(grid, bits)
    elif kind == "morton":
        keys = morton_keys(grid, bits)
    else:
        raise ValueError(f"Unknown serialization: {kind}")
    return torch.argsort(keys, dim=-1, stable=True)


def window_labels(order: Tensor, window_size: int, shift: int = 0) -> Tensor:
    """Non-cyclic shifted partitions; no spurious first/last wraparound edge."""
    rank = torch.argsort(order, dim=-1)
    return (rank + shift) // window_size


@torch.no_grad()
def local_pair_coverage(points: Tensor, dynamic: Tensor, window_size: int = 32,
                        layers: int = 2, threshold: float = 0.08,
                        kind: str = "hilbert") -> dict[str, float]:
    """Fraction of geometrically close object/environment pairs sharing a layer window.

    This deliberately reports DIRECT local-attention coverage, not transitive
    communication through deep layers. NaN marks a scene with no close pair.
    """
    if points.ndim == 2:
        points = points.unsqueeze(0)
    order = serialization_order(points, kind)
    if dynamic.ndim == 1:
        dynamic = dynamic.unsqueeze(0).expand(points.shape[0], -1)
    pair = (torch.cdist(points, points) <= threshold) & dynamic[:, :, None] & ~dynamic[:, None, :]
    covered = torch.zeros_like(pair)
    for layer in range(layers):
        labels = window_labels(order, window_size, (layer % 2) * (window_size // 2))
        covered |= labels[:, :, None] == labels[:, None, :]
    total = int(pair.sum().item())
    seen = int((pair & covered).sum().item())
    return {"near_pairs": total, "covered_pairs": seen,
            "coverage": seen / total if total else float("nan")}
