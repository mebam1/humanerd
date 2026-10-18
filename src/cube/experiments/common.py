"""Shared, explicit experiment I/O and simulation adapters."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_config(config: dict | str | Path | None) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    with Path(config).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, value: Any) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(item):
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        if dataclasses.is_dataclass(item):
            return dataclasses.asdict(item)
        raise TypeError(type(item).__name__)
    def finite_json(item):
        if isinstance(item, dict):
            return {key: finite_json(part) for key, part in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite_json(part) for part in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return None  # No nearby contact has infinite separation; JSON represents this as null.
        if isinstance(item, np.ndarray):
            return finite_json(item.tolist())
        return item
    path.write_text(json.dumps(finite_json(value), ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding="utf-8")
    return value


def make_sim(spec: dict, backend: str = "mujoco", device: str = "cpu"):
    from cube.sim import SceneSpec, Simulator
    return Simulator(SceneSpec(**spec), backend=backend, device=device)


def sim_state(sim) -> np.ndarray:
    value = sim.state() if callable(sim.state) else sim.state
    return np.asarray(value, dtype=np.float64).copy()


def model_state(model, state, action, history=None):
    """All inference paths use the same interface, including NeRD-like history."""
    return model(state, action, history=history)


def load_manifest(dataset: str | Path) -> tuple[Path, dict]:
    path = Path(dataset)
    path = path / "manifest.json" if path.is_dir() else path
    from .storage import expand_manifest
    return path.parent, expand_manifest(path.parent, read_config(path))


def load_trajectory(root: Path, entry: dict, keys=None) -> dict:
    """Read selected arrays from legacy NPZ files or memory-mapped NPY shards."""
    from .storage import read_trajectory
    return read_trajectory(Path(root), entry, keys=keys)


def state_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred, truth = np.asarray(pred), np.asarray(truth)
    nq = (pred.shape[-1] + 1) // 2
    dot = np.sum(pred[..., 3:7] * truth[..., 3:7], axis=-1)
    norm = np.linalg.norm(pred[..., 3:7], axis=-1) * np.linalg.norm(truth[..., 3:7], axis=-1)
    angle = 2 * np.arccos(np.clip(np.abs(dot) / np.maximum(norm, 1e-12), 0, 1))
    def world_omega(state):
        quat = state[..., 3:7]
        quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
        vector = state[..., nq+3:nq+6]
        cross = np.cross(quat[..., 1:], vector)
        return vector + 2 * (quat[..., :1] * cross + np.cross(quat[..., 1:], cross))
    return {
        "position_rmse": float(np.sqrt(np.mean((pred[..., :3] - truth[..., :3]) ** 2))),
        "rotation_geodesic_rad": float(np.mean(angle)),
        "linear_velocity_rmse": float(np.sqrt(np.mean((pred[..., nq:nq+3] - truth[..., nq:nq+3]) ** 2))),
        "angular_velocity_rmse": float(np.sqrt(np.mean((world_omega(pred) - world_omega(truth)) ** 2))),
        "joint_position_rmse": float(np.sqrt(np.mean((pred[..., 7:nq] - truth[..., 7:nq]) ** 2))) if nq > 7 else 0.0,
        "joint_velocity_rmse": float(np.sqrt(np.mean((pred[..., nq+6:] - truth[..., nq+6:]) ** 2))) if nq > 7 else 0.0,
    }


def upright(state: np.ndarray) -> np.ndarray:
    q = np.asarray(state)[..., 3:7]
    return 1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2)


def history_window(states, index: int, length: int):
    indices = np.maximum(np.arange(index - length + 1, index + 1), 0)
    return states[indices]
