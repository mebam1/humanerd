"""Verify saved toss data against the hierarchical model's numerical contracts.

This is a CPU-only compatibility check: no training, optimizer, W&B run, or
accuracy claim. Run with Docker Compose and the same config used for training.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from cube.experiments.common import load_manifest, load_trajectory, read_config, write_json
from cube.experiments.training_runtime import TrajectoryCache, fit_normalization
from cube.models import apply_delta, build_model, state_delta
from cube.sim import Simulator


@torch.no_grad()
def verify(dataset, config, output, trajectories=40):
    if trajectories < 1:
        raise ValueError("trajectories must be positive")
    cfg = read_config(config)
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    torch.manual_seed(0)
    root, manifest = load_manifest(dataset)
    entries = manifest["trajectories"]
    train = sorted((entry for entry in entries if entry["split"] == "train" and entry["scenario"] == "toss"),
                   key=lambda entry: entry["id"])
    if not train:
        raise ValueError("Dataset must contain toss training trajectories")
    stride = max(1, len(train) // trajectories)
    selected = train[::stride][:trajectories]
    ids = [entry["id"] for entry in selected]
    seeds = [entry["seed"] for entry in entries]
    seed_unique = len(seeds) == len(set(seeds))
    assert seed_unique, "Repeated episode seed across dataset splits"

    floor = float(cfg.get("normalization_floor", .02))
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("normalization_floor must be finite and positive")
    normalization = fit_normalization(selected, TrajectoryCache(root), floor,
                                      int(cfg.get("normalization_chunk_size", 16384)), {"enabled": False})
    scale = np.array(normalization["delta_std"], dtype=np.float32)
    assert np.isfinite(scale).all() and np.all(scale >= np.float32(floor)), "Invalid/clamp-violating delta scale"
    assert normalization["fit_split"] == "train" and normalization["trajectory_ids"] == ids

    arrays = [load_trajectory(root, entry, keys=["states", "actions"]) for entry in selected]
    maximum_action = max(float(np.abs(array["actions"]).max()) for array in arrays)
    assert all(np.isfinite(array["actions"]).all() and np.count_nonzero(array["actions"]) == 0
               for array in arrays), "Toss contains nonzero or nonfinite external actions"
    current = torch.from_numpy(np.concatenate([np.array(array["states"][:-1]) for array in arrays])).float()
    following = torch.from_numpy(np.concatenate([np.array(array["states"][1:]) for array in arrays])).float()
    assert current.shape[1] == following.shape[1] == 13, "Expected free Cube state width 13"
    assert torch.isfinite(current).all() and torch.isfinite(following).all(), "Nonfinite saved state"
    target = state_delta(current, following)
    restored = apply_delta(current, target)
    position_error = float((restored[:, :3] - following[:, :3]).abs().max())
    velocity_error = float((restored[:, 7:] - following[:, 7:]).abs().max())
    quaternion_dot = float((restored[:, 3:7] * following[:, 3:7]).sum(-1).abs().min())
    assert torch.isfinite(target).all() and torch.isfinite(restored).all(), "Nonfinite generalized delta"
    assert position_error < 5e-6, "Position delta inverse disagrees with saved next state"
    assert velocity_error < 1e-4, "Linear/angular delta inverse disagrees with saved next state"
    assert abs(1 - quaternion_dot) < 1e-5, "Quaternion target/inverse mismatch (sign-invariant)"

    sim = Simulator(selected[0]["scene_spec"])
    try:
        model = build_model(sim, {**cfg.get("model", {}), "name": "hierarchical"}).eval()
        geometry = model.features
        states = current[::200].clone()
        shift = states.new_tensor([30., -40.])
        states[:, :2] += shift
        features = geometry(states, torch.zeros(len(states), 6))
        floor_points = features["world_points"][:, geometry.plane_ids >= 0]
        assert floor_points.shape[1] > 0, "Toss scene is missing sampled floor geometry"
        floor_center_error = float((floor_points[:, :, :2].mean(1) - states[:, :2]).abs().max())
        floor_z = float(floor_points[:, :, 2].abs().max())
        floor_velocity = float(features["velocities"][:, geometry.plane_ids >= 0].abs().max())
        assert floor_center_error < 1e-4, "Floor samples do not follow the current root projection"
        assert floor_z < 1e-6, "Resampled floor points left the z=0 plane"
        assert floor_velocity == 0., "Resampling introduced a fictitious floor velocity"
        model_config = dict(model.config)
        point_counts = {"dynamic": int(geometry.dynamic.sum()), "floor": int((geometry.plane_ids >= 0).sum())}
    finally:
        sim.close()

    report = {
        "passed": True,
        "scope": "Numerical data/model compatibility only; no training, W&B run, or learned accuracy evaluation",
        "dataset": str(dataset),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "generator_version": manifest.get("generator_version"),
        "dataset_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "config": str(config),
        "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        "model_config": model_config,
        "device": "cpu",
        "torch_version": torch.__version__,
        "dataset_trajectories": len(entries),
        "split_counts": manifest["splits"],
        "all_episode_seeds_unique": seed_unique,
        "unique_episode_seed_count": len(set(seeds)),
        "selection": {"split": "train", "scenario": "toss", "stride": stride, "trajectory_ids": ids},
        "sample_trajectories": len(selected),
        "sample_transitions": len(current),
        "maximum_absolute_action": maximum_action,
        "normalization": normalization,
        "normalization_floor": floor,
        "action_normalization": "None; only output delta standard deviation scales the loss",
        "roundtrip": {"position_max_absolute_error_m": position_error,
                      "velocity_max_absolute_error": velocity_error,
                      "quaternion_min_absolute_dot": quaternion_dot},
        "floor_resampling": {"checked_states": len(states), "root_xy_translation_m": shift.tolist(),
                             "xy_center_max_absolute_error_m": floor_center_error,
                             "world_z_max_absolute_value_m": floor_z,
                             "world_velocity_max_absolute_value": floor_velocity,
                             "point_counts": point_counts},
        "tolerances": {"position_max_absolute_error_m": 5e-6, "velocity_max_absolute_error": 1e-4,
                       "quaternion_absolute_dot_error": 1e-5, "floor_xy_center_error_m": 1e-4,
                       "floor_world_z_error_m": 1e-6},
        "limitations": ["Only the selected training trajectories are checked for zero actions and delta roundtrip.",
                        "Episode seed uniqueness is checked across the complete dataset metadata.",
                        "No rollout accuracy or convergence claim follows from this compatibility check."],
    }
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/cube-toss")
    parser.add_argument("--config", default="configs/train_gpu.json")
    parser.add_argument("--output", default="docs/results/cube_toss_model_contract.json")
    parser.add_argument("--trajectories", type=int, default=40)
    args = parser.parse_args()
    result = verify(args.dataset, args.config, args.output, args.trajectories)
    print({key: result[key] for key in ("passed", "sample_trajectories", "sample_transitions",
                                      "all_episode_seeds_unique", "dataset_fingerprint")})
