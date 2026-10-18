"""Dataset selection and autonomous inference for visual inspection.

These helpers do not run training, initialize W&B, or substitute simulator states
for failed model predictions. Both legacy NPZ and sharded NPY datasets use the
same manifest and trajectory readers as the experiments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cube.experiments.common import history_window, load_manifest, make_sim, model_state
from cube.experiments.learning import _tensor, load_checkpoint
from cube.experiments.tracking import progress
from cube.experiments.training_runtime import resolve_device
from cube.models import apply_delta


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _matching_entries(dataset, split, scenario, trajectory_id, default_split):
    root, manifest = load_manifest(dataset)
    entries = manifest.get("trajectories", [])
    if trajectory_id is not None:
        entries = [entry for entry in entries if str(entry["id"]) == str(trajectory_id)]
        if not entries:
            raise ValueError(f"Trajectory id {trajectory_id!r} does not exist in {dataset}")
        # An explicit ID resolves globally unless the caller also adds a filter.
        selected_split = split
    else:
        selected_split = split if split is not None else default_split
    if selected_split is not None:
        entries = [entry for entry in entries if entry["split"] == selected_split]
    if scenario is not None:
        entries = [entry for entry in entries if entry["scenario"] == scenario]
    if not entries:
        raise ValueError(
            f"No trajectories match id={trajectory_id!r}, split={selected_split!r}, scenario={scenario!r}"
        )
    # Generated IDs are integers; also support externally supplied string IDs.
    def order(entry):
        identity = entry["id"]
        return (0, int(identity)) if str(identity).isdigit() else (1, str(identity))
    return root, manifest, sorted(entries, key=order)


def select_trajectories(dataset, split=None, scenario=None, trajectory_id=None,
                        limit=4, default_split="train"):
    """Return a deterministic subset, without loading trajectory array payloads."""
    limit = _positive_integer(limit, "limit")
    root, manifest, entries = _matching_entries(dataset, split, scenario, trajectory_id, default_split)
    return root, manifest, entries[:limit]


def list_trajectories(dataset, split=None, scenario=None, trajectory_id=None,
                      limit=10, default_split="train") -> dict:
    """List compact selection metadata; ``total_matches`` is before truncation."""
    limit = _positive_integer(limit, "limit")
    root, manifest, entries = _matching_entries(dataset, split, scenario, trajectory_id, default_split)
    return {
        "dataset": str(root),
        "backend": manifest.get("backend", "unknown"),
        "kind": manifest.get("kind", entries[0].get("scene_spec", {}).get("kind", "unknown")),
        "total_matches": len(entries),
        "trajectories": [
            {key: entry.get(key) for key in ("id", "split", "scenario", "steps", "seed")}
            for entry in entries[:limit]
        ],
    }


def predict_trajectory(checkpoint, scene_spec, truth_states, actions, horizon=100,
                       device="cpu", progress_config=None, loaded_checkpoint=None) -> dict:
    """Roll out from GT state zero, replaying recorded actions without teacher forcing.

    State history contains only the initial GT state and the model's predictions.
    Missing past states/actions are padded with the initial state/first action.
    On numerical divergence, return the valid prefix including its initial state;
    the rejected state is neither rendered nor replaced by ground truth.
    """
    horizon = _positive_integer(horizon, "horizon")
    truth_states, actions = np.asarray(truth_states), np.asarray(actions)
    if truth_states.ndim != 2 or actions.ndim != 2:
        raise ValueError("truth_states and actions must be two-dimensional arrays")
    if len(truth_states) != len(actions) + 1 or not len(actions):
        raise ValueError("Expected at least one action and exactly one more truth state than actions")
    if not np.isfinite(truth_states).all() or not np.isfinite(actions).all():
        raise ValueError("truth_states and actions must contain only finite values")
    selected_device = resolve_device(device)
    model, payload = (load_checkpoint(str(checkpoint), str(selected_device))
                      if loaded_checkpoint is None else loaded_checkpoint)
    model.to(selected_device).float().eval()
    sim = make_sim(scene_spec)
    previous_threads = torch.get_num_threads()
    try:
        scene_dims = (int(sim.model.nq), int(sim.model.nv), int(sim.action_dim))
        model_dims = (int(model.nq), int(model.nv), int(model.action_dim))
        if scene_dims != model_dims:
            raise ValueError(
                "Checkpoint morphology does not match trajectory scene: "
                f"checkpoint (nq, nv, action_dim)={model_dims}, scene={scene_dims}. "
                "Use a checkpoint trained for this robot (e.g. Cube versus Ant)."
            )
        if truth_states.shape[1] != sum(scene_dims[:2]) or actions.shape[1] != scene_dims[2]:
            raise ValueError(
                f"Trajectory dimensions do not match scene: expected state={sum(scene_dims[:2])}, "
                f"action={scene_dims[2]}, got state={truth_states.shape[1]}, action={actions.shape[1]}"
            )
        if np.linalg.norm(truth_states[0, 3:7]) < 1e-12:
            raise ValueError("Initial state root quaternion must be nonzero")
        model.set_scene(sim)
        threads = _positive_integer(payload.get("training_config", {}).get("torch_threads", 1), "torch_threads")
        torch.set_num_threads(threads)
        history_length = int(model.config.get("history", 1))
        available = len(actions)
        steps = min(horizon, available)
        predicted = [np.asarray(truth_states[0], dtype=np.float32).copy()]
        reason = None
        with torch.no_grad():
            for index in progress(range(steps), config=progress_config, desc="Model rollout", unit="step"):
                current = _tensor(np.asarray(predicted[-1:]), selected_device)
                history = {
                    "states": _tensor(history_window(np.asarray(predicted), index, history_length)[None], selected_device),
                    "actions": _tensor(history_window(actions, index, history_length)[None], selected_device),
                }
                delta = model_state(model, current, _tensor(actions[index:index + 1], selected_device), history)
                next_state = apply_delta(current, delta).cpu().numpy()[0]
                if not np.isfinite(next_state).all():
                    reason = f"Non-finite predicted state at step {index + 1}"
                    break
                if np.max(np.abs(next_state)) > 1e4:
                    reason = f"Predicted state magnitude exceeded 10000 at step {index + 1}"
                    break
                predicted.append(next_state.astype(np.float32, copy=False))
        completed = len(predicted) - 1
        checkpoint_meta = {key: payload[key] for key in (
            "seed", "epoch", "next_epoch", "batch_cursor", "global_step", "val_loss", "best_val_loss",
            "dataset_fingerprint", "dataset_manifest", "description",
        ) if key in payload}
        checkpoint_meta.update(path=str(Path(checkpoint)), model_config=dict(model.config))
        return {
            "predicted": np.asarray(predicted, dtype=np.float32),
            "truth": np.asarray(truth_states[:completed + 1], dtype=np.float32).copy(),
            "actions": np.asarray(actions[:completed], dtype=np.float32).copy(),
            "requested_horizon": horizon,
            "available_steps": available,
            "completed_steps": completed,
            "diverged": reason is not None,
            "reason": reason,
            "model_name": model.config.get("name", type(model).__name__),
            "checkpoint_meta": checkpoint_meta,
        }
    finally:
        sim.close()
        torch.set_num_threads(previous_threads)
