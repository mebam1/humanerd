"""Train and evaluate dynamics without trajectory or future-contact leakage."""
from __future__ import annotations

from pathlib import Path
import random
import json
import math
import time

import numpy as np
import torch

from .common import (history_window, load_manifest, load_trajectory, model_state,
                     state_metrics, write_json)


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tensor(values, device):
    array = np.asarray(values)
    if not array.flags.writeable:
        array = array.copy()
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _prediction_loss(predicted_delta, truth_delta, current, truth_next, scale):
    from cube.models import apply_delta
    normalized = ((predicted_delta - truth_delta) / scale).square().mean()
    predicted_next = apply_delta(current, predicted_delta)
    # Relative-quaternion sine gives a stable rotation geodesic including q/-q equivalence.
    qp, qt = predicted_next[..., 3:7], truth_next[..., 3:7]
    qp = torch.nn.functional.normalize(qp, dim=-1)
    qt = torch.nn.functional.normalize(qt, dim=-1)
    dot = (qp * qt).sum(-1).abs().clamp(max=1.0)
    angle = 2 * torch.atan2(torch.sqrt((1 - dot.square()).clamp_min(1e-8)), dot.clamp_min(1e-8))
    return normalized + angle.square().mean(), predicted_next


def load_checkpoint(checkpoint: str, device: str = "cpu"):
    from cube.models import build_model
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(payload["metadata"], payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _transfer_weights(model, checkpoint: str | None, mode: str) -> dict:
    aliases = {"point-frozen": "point", "point_frozen": "point", "aggregation-frozen": "aggregation", "aggregation_frozen": "aggregation", "full": "none", "scratch": "none"}
    if mode not in aliases:
        raise ValueError(f"Unknown transfer mode {mode}; use scratch, full, point-frozen, aggregation-frozen")
    report = {"mode": mode, "loaded_parameters": [], "skipped_parameters": []}
    new_joint_head = False
    if mode != "scratch":
        if checkpoint is None:
            raise ValueError("A source checkpoint is required for transfer")
        source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = source_payload["state_dict"]
        new_joint_head = int(source_payload["metadata"]["nq"]) == 7 and model.nq > 7
        destination = model.state_dict()
        matched = {key: value for key, value in source.items() if key in destination and destination[key].shape == value.shape
                   and not (new_joint_head and key.startswith("aggregation.joint_decoder."))}
        model.load_state_dict(matched, strict=False)
        report["loaded_parameters"] = list(matched)
        report["skipped_parameters"] = [key for key in source if key not in matched]
    model.freeze_transfer(aliases[mode])
    if new_joint_head and aliases[mode] == "aggregation":
        for name, parameter in model.named_parameters():
            if name.startswith("aggregation.joint_decoder."):
                parameter.requires_grad_(True)
    report["novel_joint_head"] = new_joint_head
    report["novel_head_policy"] = "Cube has no trained joint output. Target joint head is newly initialized and remains trainable even when source aggregation is frozen." if new_joint_head else "All available source heads follow the selected freeze policy."
    report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["total_parameters"] = sum(p.numel() for p in model.parameters())
    return report


def train(dataset: str, output: str, config: dict | str | None = None, model_name: str = "hierarchical",
          seed: int = 0, epochs: int | None = None, transfer_checkpoint: str | None = None,
          transfer_mode: str = "scratch", resume: str | None = None) -> dict:
    from .training_runtime import run_training
    return run_training(dataset, output, config, model_name, seed, epochs,
                        transfer_checkpoint, transfer_mode, resume)


def evaluate(checkpoint: str, dataset: str, output: str, split: str = "test", rollout_horizon: int = 32,
             device: str | None = None, batch_size: int | None = None, max_trajectories: int | None = None,
             save_rollouts: bool | None = None) -> dict:
    from cube.models import apply_delta
    from .tracking import progress
    from .training_runtime import SceneCache, fixed_subset, resolve_device, scene_key, _sync
    if rollout_horizon < 1:
        raise ValueError("rollout_horizon must be positive")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    training_cfg = payload.get("training_config", {})
    cfg = training_cfg.get("evaluation", {})
    selected_device = resolve_device(device or cfg.get("device", training_cfg.get("device", "cpu")))
    batch_size = int(batch_size if batch_size is not None else cfg.get("batch_size", 32))
    if batch_size < 1:
        raise ValueError("evaluation batch_size must be positive")
    limit = max_trajectories if max_trajectories is not None else cfg.get("max_trajectories")
    save_rollouts = save_rollouts if save_rollouts is not None else cfg.get("save_rollouts", True)
    root, manifest = load_manifest(dataset)
    entries = [entry for entry in manifest["trajectories"] if entry["split"] == split]
    available = len(entries)
    if not entries:
        raise ValueError(f"No trajectories in split {split}")
    entries = fixed_subset(entries, limit, int(cfg.get("seed", 2026)))
    model, payload = load_checkpoint(checkpoint, str(selected_device))
    torch.set_num_threads(int(training_cfg.get("torch_threads", 1)))
    history_length = int(model.config.get("history", 1))
    scenes = SceneCache(training_cfg.get("scene_cache_size", 16))
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports, sums = [], {}
    component_sums = {name: {} for name in ("rollout", "impact", "persistent_contact")}
    component_counts = {name: 0 for name in component_sums}
    diagnostic_sums = {name: 0. for name in ("penetration_mean", "penetration_max", "contact_disagreement",
                                            "contact_onset_error_seconds", "missed_contact_onsets", "false_contact_onsets")}
    diagnostic_counts = {name: 0 for name in diagnostic_sums}
    retain = cfg.get("retain_trajectory_metrics", len(entries) <= 1024)
    latency_total, divergence_count = 0., 0
    warmed = set()
    metrics_path = output_dir / "trajectories.jsonl"
    with torch.no_grad(), metrics_path.open("w", encoding="utf-8") as metrics_file:
        for entry in progress(entries, config=training_cfg.get("progress"), desc=f"Evaluate {split}", unit="trajectory"):
            trajectory = load_trajectory(root, entry)
            states, actions = trajectory["states"], trajectory["actions"]
            sim = scenes.activate(model, entry)
            one_step, latency, one_step_batch_ms = [], [], []
            for start in range(0, len(actions), batch_size):
                indices = np.arange(start, min(start+batch_size, len(actions)))
                hist = np.maximum(indices[:, None]+np.arange(1-history_length, 1), 0)
                # CUDA synchronization includes host transfers, geometry, serialization,
                # model and state integration; warmup is excluded from timing.
                key = scene_key(entry)
                if key not in warmed:
                    for _ in range(int(cfg.get("warmup_batches", 2))):
                        x, u = _tensor(states[indices], selected_device), _tensor(actions[indices], selected_device)
                        h = {"states": _tensor(states[hist], selected_device), "actions": _tensor(actions[hist], selected_device)}
                        apply_delta(x, model_state(model, x, u, h))
                    _sync(selected_device)
                    warmed.add(key)
                _sync(selected_device)
                started = time.perf_counter()
                current = _tensor(states[indices], selected_device)
                act = _tensor(actions[indices], selected_device)
                history = {"states": _tensor(states[hist], selected_device), "actions": _tensor(actions[hist], selected_device)}
                pred = apply_delta(current, model_state(model, current, act, history)).cpu().numpy()
                _sync(selected_device)
                one_step_batch_ms.append((time.perf_counter()-started)*1000/len(indices))
                one_step.append(pred)
            one_step = np.concatenate(one_step)
            if not np.isfinite(one_step).all():
                raise RuntimeError(f"Non-finite one-step predictions on trajectory {entry['id']}; evaluation cannot report finite errors")
            event_contact = trajectory["contact_count"][1:] > 0
            impact_mask = trajectory.get("impact_event", event_contact & (trajectory["contact_count"][:-1] == 0)).astype(bool)
            persistent_mask = trajectory.get("persistent_contact", event_contact & (trajectory["contact_count"][:-1] > 0)).astype(bool)
            horizon = min(rollout_horizon, len(actions))
            predicted = [states[0]]
            penetration, contact_pred = [], []
            divergent = False
            for i in range(horizon):
                _sync(selected_device)
                started = time.perf_counter()
                current = _tensor(np.asarray(predicted[-1:]), selected_device)
                history = {"states": _tensor(history_window(np.asarray(predicted), i, history_length)[None], selected_device),
                           "actions": _tensor(history_window(actions, i, history_length)[None], selected_device)}
                next_state = apply_delta(current, model_state(model, current, _tensor(actions[i:i+1], selected_device), history)).cpu().numpy()[0]
                _sync(selected_device)
                latency.append(time.perf_counter()-started)
                if not np.isfinite(next_state).all() or np.max(np.abs(next_state)) > 1e4:
                    divergent = True
                    break
                predicted.append(next_state)
                # Simulator contacts are diagnostics only; never inputs to prediction.
                sim.reset(next_state)
                diagnostics = sim.diagnostics()
                penetration.append(float(diagnostics["penetration"]))
                contact_pred.append(int(diagnostics["contact_count"]) > 0)
            predicted = np.asarray(predicted)
            realized = len(predicted)-1
            true_contact = trajectory["contact_count"][1:realized+1] > 0
            pred_contact = np.asarray(contact_pred, dtype=bool)
            true_onsets = np.flatnonzero(true_contact & ~np.r_[False, true_contact[:-1]]) if realized else np.array([], dtype=int)
            pred_onsets = np.flatnonzero(pred_contact & ~np.r_[False, pred_contact[:-1]]) if realized else np.array([], dtype=int)
            onset_error = None
            if len(true_onsets) and len(pred_onsets):
                onset_error = float(np.mean([np.min(np.abs(pred_onsets-onset)) * sim.dt for onset in true_onsets]))
            missed_onsets = sum(not len(pred_onsets) or np.min(np.abs(pred_onsets-onset)) > 2 for onset in true_onsets)
            false_onsets = sum(not len(true_onsets) or np.min(np.abs(true_onsets-onset)) > 2 for onset in pred_onsets)
            stationary = np.linalg.norm(states[0, sim.model.nq:sim.model.nq+6]) < 0.1
            record = {"trajectory_id": entry["id"], "scenario": entry["scenario"], "split": split,
                      "one_step": state_metrics(one_step, states[1:]),
                      "impact": state_metrics(one_step[impact_mask], states[1:][impact_mask]) if impact_mask.any() else None,
                      "persistent_contact": state_metrics(one_step[persistent_mask], states[1:][persistent_mask]) if persistent_mask.any() else None,
                      "impact_frames": int(impact_mask.sum()), "persistent_frames": int(persistent_mask.sum()),
                      "event_label_source": "substep contacts and approach velocity" if "impact_event" in trajectory else "legacy control endpoints",
                      "rollout": state_metrics(predicted[1:], states[1:realized+1]) if realized else None,
                      "rollout_steps": realized, "diverged": divergent,
                      "penetration_max": max(penetration, default=0.0),
                      "penetration_mean": float(np.mean(penetration)) if penetration else 0.0,
                      "contact_disagreement": float(np.mean(pred_contact != true_contact)) if realized else None,
                      "contact_onset_error_seconds": onset_error,
                      "contact_onset_resolution_seconds": sim.dt,
                      "ground_truth_substep_penetration_max": float(np.max(trajectory.get("substep_penetration", trajectory["penetration"]))),
                      "missed_contact_onsets": int(missed_onsets), "false_contact_onsets": int(false_onsets),
                      "onset_match_tolerance_steps": 2,
                      "rest_position_drift": float(np.linalg.norm(predicted[-1,:3]-states[0,:3])) if stationary else None,
                      "rest_rotation_drift_rad": state_metrics(predicted[-1:], states[:1])["rotation_geodesic_rad"] if stationary else None,
                      "latency_end_to_end_ms": float(np.mean(latency)*1000),
                      "latency_p95_ms": float(np.quantile(latency,0.95)*1000),
                      "throughput_single_rollouts_per_second": float(1/np.mean(latency)),
                      "one_step_batch_size": batch_size, "one_step_batched_ms_per_sample": float(np.mean(one_step_batch_ms))}
            record["local_pair_coverage_initial"] = model.coverage(_tensor(states[:1], selected_device), _tensor(actions[:1], selected_device))
            if not math.isfinite(record["local_pair_coverage_initial"]["coverage"]):
                record["local_pair_coverage_initial"]["coverage"] = None
            for key, value in record["one_step"].items():
                sums[key] = sums.get(key, 0.) + value
            for component, totals in component_sums.items():
                if record[component] is not None:
                    component_counts[component] += 1
                    for key, value in record[component].items():
                        totals[key] = totals.get(key, 0.) + value
            for key in diagnostic_sums:
                if record[key] is not None:
                    diagnostic_sums[key] += record[key]
                    diagnostic_counts[key] += 1
            latency_total += record["latency_end_to_end_ms"]
            divergence_count += int(divergent)
            metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
            if retain:
                reports.append(record)
            if save_rollouts:
                np.savez_compressed(output_dir / f"rollout_{entry['id']:05d}.npz", predicted=predicted, truth=states[:realized+1],
                                    actions=actions[:realized], one_step=one_step, penetration=penetration)
    report = {"checkpoint": checkpoint, "dataset": dataset, "split": split, "device": str(selected_device),
              "trajectories": reports, "trajectory_metrics_file": str(metrics_path), "evaluated_trajectories": len(entries),
              "available_trajectories": available, "trajectory_ids": [entry["id"] for entry in entries],
              "subset_seed": int(cfg.get("seed", 2026)), "save_rollouts": save_rollouts,
              "mean_one_step": {key: value/len(entries) for key, value in sums.items()},
              **{f"mean_{name}": {key: value/component_counts[name] for key, value in totals.items()}
                 for name, totals in component_sums.items()},
              "metric_trajectory_counts": component_counts,
              "mean_diagnostics": {key: diagnostic_sums[key]/diagnostic_counts[key] if diagnostic_counts[key] else None
                                   for key in diagnostic_sums},
              "divergence_rate": divergence_count/len(entries), "mean_latency_end_to_end_ms": latency_total/len(entries),
              "evaluation_policy": "fixed scenario-stratified subset; batched one-step; autonomous rollouts; synchronized warmed GPU timings include host transfers, geometry, serialization, network and state update; CPU diagnostics excluded",
              "baseline_notice": "nerd_like is a controlled local baseline, not official NeRD/FIGNet/RigidFormer"}
    return write_json(output_dir / "evaluation.json", report)
