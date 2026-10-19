"""Bounded-memory, reproducible GPU runtime for trajectory dynamics models."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .common import load_manifest, load_trajectory, make_sim, model_state, read_config, write_json
from .tracking import RunTracker, progress


def resolve_device(value):
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but PyTorch cannot access a GPU. Use the Compose cuda service.")
    if device.type == "cuda":
        torch.cuda.get_device_properties(device)  # Also validate an explicit GPU index.
    return device


def scene_key(entry):
    # SceneSpec.seed changes initialization only, never MJCF geometry/materials.
    return json.dumps({key: value for key, value in entry["scene_spec"].items() if key != "seed"}, sort_keys=True)


class TrajectoryCache:
    def __init__(self, root, size=128):
        self.root, self.size, self.values = root, max(1, int(size)), OrderedDict()

    def get(self, entry):
        key = entry["id"]
        if key not in self.values:
            self.values[key] = load_trajectory(self.root, entry, keys=["states", "actions"])
            if len(self.values) > self.size:
                self.values.popitem(last=False)
        self.values.move_to_end(key)
        return self.values[key]


class SceneCache:
    """Cache simulators and static feature buffers, not per-example computation graphs."""
    def __init__(self, size=16):
        self.size, self.values = max(1, int(size)), OrderedDict()

    def get(self, entry):
        key = scene_key(entry)
        if key not in self.values:
            self.values[key] = [make_sim(entry["scene_spec"]), None]
            if len(self.values) > self.size:
                self.values.popitem(last=False)
        self.values.move_to_end(key)
        return self.values[key]

    def activate(self, model, entry):
        item = self.get(entry)
        if item[1] is None:
            model.set_scene(item[0])
            item[1] = model.features
        else:
            model.features = item[1]
            model.metadata = item[1].metadata
        return item[0]


def fixed_subset(entries, limit, seed=2026):
    if limit is None:
        return entries
    if int(limit) < 1:
        raise ValueError("Trajectory subset limits must be positive")
    if len(entries) <= int(limit):
        return entries
    # Round robin shuffled strata so rare scenarios are represented across baselines.
    groups = {}
    for entry in entries:
        groups.setdefault(entry.get("scenario", "unknown"), []).append(entry)
    rng = np.random.default_rng(seed)
    groups = [list(rng.permutation(group)) for _, group in sorted(groups.items())]
    result = []
    while len(result) < int(limit):
        for group in groups:
            if group:
                result.append(group.pop())
                if len(result) == int(limit):
                    break
    return result


class BatchPlan:
    """Shuffle trajectories and transitions, then pack same-scene cross-trajectory batches.

    The plan stores O(trajectories) metadata and O(batch size + one trajectory)
    indices. It never builds a 10-million-transition Python object list.
    """
    def __init__(self, entries, cache, batch_size, rollout_steps=1):
        self.batch_size, self.groups, self.counts = batch_size, {}, {}
        self.transitions = 0
        for entry in entries:
            count = int(entry["steps"]) if "steps" in entry else len(cache.get(entry)["actions"])
            count -= rollout_steps - 1
            if count < 1:
                raise ValueError("rollout_steps exceeds available trajectory length")
            self.counts[entry["id"]] = count
            self.groups.setdefault(scene_key(entry), []).append(entry)
            self.transitions += count
        self.batches = sum(math.ceil(sum(self.counts[e["id"]] for e in group) / batch_size)
                           for group in self.groups.values())

    def _group_batches(self, group, rng, shuffle):
        order = rng.permutation(len(group)) if shuffle else range(len(group))
        pending, length = [], 0
        for index in order:
            entry = group[int(index)]
            count = self.counts[entry["id"]]
            indices = rng.permutation(count) if shuffle else np.arange(count)
            offset = 0
            while offset < count:
                take = min(self.batch_size - length, count - offset)
                pending.append((entry, indices[offset:offset+take]))
                length += take
                offset += take
                if length == self.batch_size:
                    yield pending
                    pending, length = [], 0
        if pending:
            yield pending

    def iterate(self, seed, epoch=0, shuffle=True):
        groups = list(self.groups.values())
        seeds = np.random.SeedSequence([seed, epoch, 5781]).spawn(len(groups)+1)
        chooser = np.random.default_rng(seeds[0])
        streams = [self._group_batches(group, np.random.default_rng(seeds[index+1]), shuffle)
                   for index, group in enumerate(groups)]
        remaining = np.array([math.ceil(sum(self.counts[entry["id"]] for entry in group) / self.batch_size)
                              for group in groups], dtype=np.int64)
        total = int(remaining.sum())
        while total:
            if shuffle:
                # Interleave scene-homogeneous batches in proportion to remaining
                # batches, avoiding long single-scenario blocks and implicit curricula.
                draw = chooser.integers(total)
                group_index = int(np.searchsorted(np.cumsum(remaining), draw, side="right"))
            else:
                group_index = int(np.flatnonzero(remaining)[0])
            yield next(streams[group_index])
            remaining[group_index] -= 1
            total -= 1


def materialize(plan, cache, history, horizon, device):
    parts = {"current": [], "future": [], "actions": [], "history_states": [], "history_actions": []}
    for entry, indices in plan:
        value = cache.get(entry)
        states, actions = value["states"], value["actions"]
        hist = np.maximum(indices[:, None] + np.arange(1-history, 1), 0)
        future = indices[:, None] + np.arange(1, horizon+1)
        parts["current"].append(states[indices])
        parts["future"].append(states[future])
        parts["actions"].append(actions[future-1])
        parts["history_states"].append(states[hist])
        parts["history_actions"].append(actions[hist])
    return {key: torch.from_numpy(np.concatenate(values).astype(np.float32, copy=False)).to(device)
            for key, values in parts.items()}


def fit_normalization(entries, cache, floor, chunk_size, progress_config=None):
    """Train-only float64 parallel Welford statistics; deltas computed in bounded CPU chunks."""
    from cube.models import state_delta
    count, mean, m2 = 0, None, None
    currents, futures, pending = [], [], 0

    def consume():
        nonlocal count, mean, m2, currents, futures, pending
        x = torch.from_numpy(np.concatenate(currents).astype(np.float32, copy=False))
        y = torch.from_numpy(np.concatenate(futures).astype(np.float32, copy=False))
        delta = state_delta(x, y).double()
        if not torch.isfinite(delta).all():
            raise ValueError("Non-finite training delta found during normalization")
        n = len(delta)
        avg = delta.mean(0)
        variance_sum = (delta-avg).square().sum(0)
        if count == 0:
            mean, m2 = avg, variance_sum
        else:
            difference = avg - mean
            m2 = m2 + variance_sum + difference.square() * count * n / (count+n)
            mean = mean + difference * n / (count+n)
        count += n
        currents, futures, pending = [], [], 0

    for entry in progress(entries, config=progress_config, desc="Train normalization", unit="trajectory"):
        states = cache.get(entry)["states"]
        offset = 0
        while offset < len(states)-1:
            take = min(chunk_size-pending, len(states)-1-offset)
            # Copy rows to ensure only at most chunk_size samples keep mapped shards alive.
            currents.append(np.array(states[offset:offset+take], copy=True))
            futures.append(np.array(states[offset+1:offset+take+1], copy=True))
            pending += take
            offset += take
            if pending == chunk_size:
                consume()
    if pending:
        consume()
    return {"delta_mean": mean.float().tolist(), "delta_std": (m2/count).sqrt().clamp_min(floor).float().tolist(),
            "fit_split": "train", "trajectory_ids": [entry["id"] for entry in entries], "transitions": count,
            "method": "streaming float64 Welford; deltas float32 on CPU"}


def fit_input_normalization(model, entries, cache, scenes, batch_size, epsilon=1e-5,
                            progress_config=None):
    """Fit contact input statistics using only current states/actions in the train split.

    Collision queries are recomputed from each current pose. The loader used here
    exposes only states/actions, so saved contact labels and future states cannot
    become feature inputs. Scene-homogeneous chunks bound both memory and queries.
    """
    if not entries or any(entry.get("split") != "train" for entry in entries):
        raise ValueError("Input normalization requires nonempty train-only trajectories")
    if int(batch_size) < 1 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Input normalization batch size and epsilon must be positive")
    device = next(model.parameters()).device
    plan = BatchPlan(entries, cache, int(batch_size), rollout_steps=1)
    count, mean, m2 = 0, None, None
    with torch.no_grad():
        for batch in progress(plan.iterate(0, shuffle=False), config=progress_config,
                              total=plan.batches, desc="Train contact input normalization", unit="batch"):
            scenes.activate(model, batch[0][0])
            states, actions = [], []
            for entry, indices in batch:
                value = cache.get(entry)
                states.append(value["states"][indices])
                actions.append(value["actions"][indices])
            current = torch.from_numpy(np.concatenate(states).astype(np.float32, copy=False)).to(device)
            action = torch.from_numpy(np.concatenate(actions).astype(np.float32, copy=False)).to(device)
            values = model.raw_input(current, action).detach().to(device="cpu", dtype=torch.float64)
            if values.ndim != 2 or not torch.isfinite(values).all():
                raise ValueError("Contact input features must be a finite two-dimensional tensor")
            n, average = len(values), values.mean(0)
            variance_sum = (values-average).square().sum(0)
            if count == 0:
                mean, m2 = average, variance_sum
            else:
                difference = average-mean
                m2 = m2 + variance_sum + difference.square()*count*n/(count+n)
                mean = mean + difference*n/(count+n)
            count += n
    return {"mean": mean.float().tolist(), "scale": (m2/count+epsilon).sqrt().float().tolist(),
            "variance": (m2/count).tolist(), "epsilon": epsilon, "fit_split": "train",
            "trajectory_ids": [entry["id"] for entry in entries], "transitions": count,
            "feature_dimension": len(mean), "batch_size": int(batch_size),
            "feature_source": "current state/action and current-pose collision query; no stored contact labels or future state",
            "method": "streaming float64 Welford; population scale=sqrt(variance+epsilon)"}


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda"]])


def _atomic_checkpoint(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _amp_context(device, precision):
    return nullcontext() if precision == "none" else torch.autocast(device_type=device.type,
             dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def _contract(cfg, model_name, seed, num_epochs):
    # Operational limits/logging may change on resume; optimization and sampling must not.
    keys = ("model", "trajectory_budget", "batch_size", "gradient_accumulation_steps", "rollout_steps",
            "rollout_weight", "learning_rate", "weight_decay", "gradient_clip", "normalization_floor",
            "mixed_precision", "scheduler", "validation_max_trajectories", "validation_seed")
    return {"settings": {key: cfg.get(key) for key in keys}, "model_name": model_name,
            "seed": seed, "epochs": num_epochs, "sampler_version": 2}


def run_training(dataset, output, config, model_name, seed, epochs, transfer_checkpoint, transfer_mode, resume):
    from cube.models import build_model, state_delta
    from .learning import _prediction_loss, _seed, _transfer_weights
    cfg = read_config(config)
    _seed(seed)
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    device = resolve_device(cfg.get("device", "cpu"))
    precision = cfg.get("mixed_precision", "none")
    if precision not in {"none", "bf16", "fp16"}:
        raise ValueError("mixed_precision must be none, bf16, or fp16")
    if precision != "none" and device.type != "cuda":
        raise ValueError("Mixed precision is supported on CUDA; use none for CPU runs")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; configure fp16 or none")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    num_epochs = int(epochs if epochs is not None else cfg.get("epochs", 10))
    batch_size = int(cfg.get("batch_size", 32))
    accumulation = int(cfg.get("gradient_accumulation_steps", 1))
    horizon = int(cfg.get("rollout_steps", 1))
    chunk_size = int(cfg.get("normalization_chunk_size", 16384))
    max_steps = cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    if min(num_epochs, batch_size, accumulation, horizon, chunk_size) < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("epochs, batch sizes, horizon, chunk size and max_steps must be positive")
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = resume or cfg.get("resume")
    if not resume and any((output_dir / name).exists() for name in ("last.pt", "best.pt", "training.json")):
        raise ValueError("This run directory already contains training results. Use --resume with last.pt or a new output directory.")
    root, manifest = load_manifest(dataset)
    manifest_path = Path(dataset) / "manifest.json" if Path(dataset).is_dir() else Path(dataset)
    train_entries = [entry for entry in manifest["trajectories"] if entry["split"] == "train"]
    val_entries = [entry for entry in manifest["trajectories"] if entry["split"] == "val"]
    budget = cfg.get("trajectory_budget")
    if budget is not None:
        if int(budget) < 1 or int(budget) > len(train_entries):
            raise ValueError(f"trajectory_budget={budget} requires that many training trajectories; available={len(train_entries)}")
        order = np.random.default_rng(seed).permutation(len(train_entries))[:int(budget)]
        train_entries = [train_entries[int(index)] for index in order]
    if not train_entries or not val_entries:
        raise ValueError("Training and validation trajectories are required")
    val_entries = fixed_subset(val_entries, cfg.get("validation_max_trajectories"), int(cfg.get("validation_seed", 2026)))
    data = TrajectoryCache(root, cfg.get("data_cache_size", 128))
    scenes = SceneCache(cfg.get("scene_cache_size", 16))
    train_plan = BatchPlan(train_entries, data, batch_size, horizon)
    val_plan = BatchPlan(val_entries, data, batch_size)
    fingerprint = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    contract = _contract(cfg, model_name, seed, num_epochs)
    previous = None
    if resume:
        if Path(resume).resolve().parent != output_dir.resolve():
            raise ValueError("Resume into the existing run directory to preserve its best.pt and W&B identity")
        previous = torch.load(resume, map_location="cpu", weights_only=False)
        if previous.get("dataset_fingerprint") != fingerprint or previous.get("training_contract") != contract:
            raise ValueError("Resume dataset manifest, model, seed, or optimization/sampling settings changed")
        if "optimizer_state_dict" not in previous:
            raise ValueError("Resume requires a runtime last.pt checkpoint with optimizer and RNG state")
    model_cfg = {**cfg.get("model", {}), "name": model_name}
    model = build_model(scenes.get(train_entries[0])[0], model_cfg).to(device)
    if previous:
        model.load_state_dict(previous["state_dict"])
        transfer_report = previous["transfer"]
        frozen = {"point-frozen": "point", "point_frozen": "point", "aggregation-frozen": "aggregation",
                  "aggregation_frozen": "aggregation", "full": "none", "scratch": "none"}[transfer_report["mode"]]
        model.freeze_transfer(frozen)
        if transfer_report.get("novel_joint_head") and frozen == "aggregation":
            for name, parameter in model.named_parameters():
                if name.startswith("aggregation.joint_decoder."):
                    parameter.requires_grad_(True)
    else:
        transfer_report = _transfer_weights(model, transfer_checkpoint, transfer_mode)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(cfg.get("learning_rate", 3e-4)),
                                  weight_decay=float(cfg.get("weight_decay", 1e-5)))
    scheduler_config = cfg.get("scheduler", {})
    scheduler_name = scheduler_config.get("name", "none")
    if scheduler_name not in {"none", "cosine", "linear"}:
        raise ValueError("scheduler.name must be none, cosine, or linear")
    updates_per_epoch = math.ceil(train_plan.batches / accumulation)
    total_updates = num_epochs * updates_per_epoch
    warmup = int(float(scheduler_config.get("warmup_epochs", 0)) * updates_per_epoch)
    min_ratio = float(scheduler_config.get("min_lr_ratio", 0.1))
    if not 0 <= min_ratio <= 1 or warmup < 0 or warmup > total_updates:
        raise ValueError("Invalid scheduler warmup_epochs or min_lr_ratio")
    def lr_multiplier(step):
        if scheduler_name == "none":
            return 1.0
        if step < warmup:
            return (step+1)/max(warmup, 1)
        fraction = min(1., max(0., (step-warmup)/max(1, total_updates-warmup)))
        if scheduler_name == "linear":
            return 1. - (1-min_ratio)*fraction
        return min_ratio + (1-min_ratio)*0.5*(1+math.cos(math.pi*fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    tracker_config = dict(cfg.get("wandb", {}))
    tracker_config.setdefault("name", f"{model_name}-seed{seed}")
    online_config = {"architecture": model.config, "seed": seed, "device": str(device),
                     "epochs": num_epochs, "batch_size": batch_size, "effective_batch_size": batch_size*accumulation,
                     "learning_rate": float(cfg.get("learning_rate", 3e-4)), "mixed_precision": precision,
                     "train_trajectories": len(train_entries), "train_transitions": train_plan.transitions,
                     "validation_trajectories": len(val_entries), "dataset_manifest_sha256": fingerprint,
                     "rollout_steps": horizon, "scheduler": scheduler_config,
                     "baseline_notice": "Controlled local implementations, not official NeRD reproductions"}
    tracker = RunTracker(tracker_config, online_config, output_dir,
                         previous.get("wandb", {}).get("id") if previous else None)
    write_json(output_dir / "wandb_run.json", {"mode": tracker.mode, "id": tracker.id, "url": tracker.url})
    started = time.perf_counter()
    best, records, global_step = float("inf"), [], 0
    start_epoch, resume_cursor, epoch_sum, epoch_samples = 0, 0, 0., 0
    failed = True
    try:
        normalization = previous["normalization"] if previous else fit_normalization(
            train_entries, data, float(cfg.get("normalization_floor", .02)), chunk_size, cfg.get("progress"))
        scale = torch.tensor(normalization["delta_std"], dtype=torch.float32, device=device)
        contact_model = model.kind == "contact_mlp"
        input_normalization = None
        if contact_model:
            if previous:
                # Normalization buffers have already been restored by load_state_dict.
                # Never refit on resume or overwrite checkpoint precision/values.
                input_normalization = previous["input_normalization"]
            else:
                if model.config.get("normalize_input", True):
                    input_normalization = fit_input_normalization(
                        model, train_entries, data, scenes,
                        int(cfg.get("input_normalization_batch_size", batch_size)),
                        float(model.config.get("input_normalization_epsilon", 1e-5)), cfg.get("progress"))
                    input_normalization["dataset_manifest_sha256"] = fingerprint
                    model.set_input_normalization(input_normalization["mean"], input_normalization["scale"])
                else:
                    input_normalization = {"enabled": False, "fit_split": None, "trajectory_ids": [], "transitions": 0}
                if model.config.get("normalize_output", True):
                    model.set_output_normalization(normalization["delta_mean"], normalization["delta_std"])
        if previous:
            optimizer.load_state_dict(previous["optimizer_state_dict"])
            scheduler.load_state_dict(previous["scheduler_state_dict"])
            scaler.load_state_dict(previous["scaler_state_dict"])
            best, records, global_step = previous["best_val_loss"], previous["records"], previous["global_step"]
            start_epoch, resume_cursor = previous["next_epoch"], previous["batch_cursor"]
            epoch_sum, epoch_samples = previous["epoch_loss_sum"], previous["epoch_samples"]
            _restore_rng(previous["rng_state"])
        history_length = int(model_cfg.get("history", 1))
        checkpoint_interval = int(cfg.get("checkpoint_every_steps", 1000))
        log_interval = max(1, int(cfg.get("log_every_steps", 10)))
        rollout_weight = float(cfg.get("rollout_weight", .25))
        previous_elapsed = float(previous.get("elapsed_seconds", 0.)) if previous else 0.

        def payload(next_epoch, cursor):
            return {"state_dict": model.state_dict(), "model_config": model.config, "metadata": model.metadata,
                    "scene_spec": train_entries[0]["scene_spec"], "normalization": normalization,
                    **({"input_normalization": input_normalization} if contact_model else {}),
                    "training_config": cfg, "training_contract": contract, "seed": seed,
                    "epoch": next_epoch + (cursor > 0), "next_epoch": next_epoch, "batch_cursor": cursor,
                    "global_step": global_step, "epoch_loss_sum": epoch_sum, "epoch_samples": epoch_samples,
                    "val_loss": best, "best_val_loss": best, "records": records, "transfer": transfer_report,
                    "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(), "rng_state": _rng_state(),
                    "dataset_manifest": str(manifest_path), "dataset_fingerprint": fingerprint,
                    "wandb": {"mode": tracker.mode, "id": tracker.id, "url": tracker.url},
                    "elapsed_seconds": previous_elapsed + time.perf_counter()-started,
                    "description": "Research implementation; NeRD-like baseline is not an official reproduction"}

        exhausted = max_steps is not None and global_step >= max_steps
        epoch_progress = progress(range(start_epoch, num_epochs), config=cfg.get("progress"), desc="Epochs", unit="epoch")
        for epoch in epoch_progress:
            if exhausted:
                break
            model.train()
            _sync(device)
            epoch_started, new_samples = time.perf_counter(), 0
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples, accumulated_loss, accumulated_batches = 0, 0., 0
            last_cursor = resume_cursor
            batches = progress(enumerate(train_plan.iterate(seed, epoch)), config=cfg.get("progress"),
                               total=train_plan.batches, desc=f"Train {epoch+1}/{num_epochs}", leave=False, unit="batch")
            for batch_index, plan in batches:
                if batch_index < resume_cursor:
                    continue
                scenes.activate(model, plan[0][0])
                batch = materialize(plan, data, history_length, horizon, device)
                current = batch["current"]
                history = {"states": batch["history_states"], "actions": batch["history_actions"]}
                total_loss = 0.
                for step in range(horizon):
                    target = state_delta(current, batch["future"][:, step])
                    with _amp_context(device, precision):
                        prediction = model_state(model, current, batch["actions"][:, step], history)
                    # Keep quaternion algebra and normalized loss in float32 under AMP.
                    loss, current = _prediction_loss(prediction.float(), target, current, batch["future"][:, step], scale)
                    total_loss = total_loss + (1. if step == 0 else rollout_weight)*loss
                    history["states"] = torch.cat((history["states"][:, 1:], current[:, None]), 1)
                    if step+1 < horizon:
                        history["actions"] = torch.cat((history["actions"][:, 1:], batch["actions"][:, step+1, None]), 1)
                if not torch.isfinite(total_loss):
                    raise RuntimeError("Non-finite training loss; inspect data, precision and learning rate")
                count = len(current)
                scaler.scale(total_loss * count).backward()
                measured_loss = float(total_loss.detach())
                accumulated_loss += measured_loss*count
                accumulated_samples += count
                accumulated_batches += 1
                epoch_sum += measured_loss*count
                epoch_samples += count
                new_samples += count
                last_cursor = batch_index+1
                final_batch = last_cursor == train_plan.batches
                if accumulated_batches == accumulation or final_batch:
                    scaler.unscale_(optimizer)
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated_samples)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float(cfg.get("gradient_clip", 1.)))
                    learning_rate = optimizer.param_groups[0]["lr"]
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped = scaler.get_scale() < old_scale
                    if not skipped:
                        scheduler.step()
                        global_step += 1
                    optimizer.zero_grad(set_to_none=True)
                    if global_step % log_interval == 0 or final_batch:
                        tracker.log({"optimizer_step": global_step, "train/loss": accumulated_loss/accumulated_samples,
                                     "train/learning_rate": learning_rate, "train/gradient_norm": float(gradient_norm),
                                     "train/effective_samples": accumulated_samples, "train/amp_update_skipped": int(skipped),
                                     "train/samples_per_second": new_samples/max(time.perf_counter()-epoch_started, 1e-9),
                                     "train/gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device)/2**20
                                     if device.type == "cuda" else 0.})
                        batches.set_postfix(loss=f"{measured_loss:.4g}", lr=f"{learning_rate:.2g}", step=global_step)
                    accumulated_samples, accumulated_loss, accumulated_batches = 0, 0., 0
                    if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                        _atomic_checkpoint(payload(epoch, last_cursor), output_dir / "last.pt")
                    exhausted = max_steps is not None and global_step >= max_steps
                    if exhausted:
                        break
            batches.close()
            _sync(device)
            train_seconds = time.perf_counter()-epoch_started
            model.eval()
            val_sum, val_samples = 0., 0
            with torch.no_grad():
                for plan in progress(val_plan.iterate(seed, shuffle=False), config=cfg.get("progress"),
                                     total=val_plan.batches, desc="Validation", leave=False, unit="batch"):
                    scenes.activate(model, plan[0][0])
                    batch = materialize(plan, data, history_length, 1, device)
                    current, truth = batch["current"], batch["future"][:, 0]
                    with _amp_context(device, precision):
                        prediction = model_state(model, current, batch["actions"][:, 0],
                                   {"states": batch["history_states"], "actions": batch["history_actions"]})
                    loss, _ = _prediction_loss(prediction.float(), state_delta(current, truth), current, truth, scale)
                    val_sum += float(loss)*len(current)
                    val_samples += len(current)
            completed_epoch = last_cursor == train_plan.batches
            record = {"epoch": epoch+1, "epoch_complete": completed_epoch, "optimizer_step": global_step,
                      "train_loss": epoch_sum/max(1, epoch_samples), "val_loss": val_sum/val_samples,
                      "train_samples": epoch_samples, "validation_samples": val_samples,
                      "train_seconds": train_seconds, "samples_per_second": new_samples/max(train_seconds, 1e-9),
                      "learning_rate": optimizer.param_groups[0]["lr"],
                      "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0.}
            if not math.isfinite(record["val_loss"]):
                raise RuntimeError("Non-finite validation loss")
            records.append(record)
            tracker.log({"epoch": epoch+1, "optimizer_step": global_step,
                         **{f"epoch/{key}": value for key, value in record.items() if key not in {"epoch", "optimizer_step"}}})
            epoch_progress.set_postfix(val=f"{record['val_loss']:.4g}", step=global_step)
            improved = record["val_loss"] < best
            if improved:
                best = record["val_loss"]
            if completed_epoch:
                epoch_sum, epoch_samples = 0., 0
            checkpoint_payload = payload(epoch+1 if completed_epoch else epoch, 0 if completed_epoch else last_cursor)
            if improved:
                _atomic_checkpoint(checkpoint_payload, output_dir / "best.pt")
            _atomic_checkpoint(checkpoint_payload, output_dir / "last.pt")
            resume_cursor = 0
            if exhausted:
                break
        epoch_progress.close()
        report = {"checkpoint": str(output_dir / "best.pt"), "last_checkpoint": str(output_dir / "last.pt"),
                  "model": model_name, "seed": seed, "device": str(device), "mixed_precision": precision,
                  "train_trajectories": len(train_entries), "val_trajectories": len(val_entries),
                  "validation_trajectory_ids": [entry["id"] for entry in val_entries],
                  "train_transitions_per_epoch": train_plan.transitions, "optimizer_steps": global_step,
                  "effective_batch_size": batch_size*accumulation, "epochs": records, "best_val_loss": best,
                  "normalization": normalization, "transfer": transfer_report, "dataset_fingerprint": fingerprint,
                  **({"input_normalization": input_normalization} if contact_model else {}),
                  "stop_reason": "max_steps" if exhausted else "epochs", "wandb": {"mode": tracker.mode, "id": tracker.id, "url": tracker.url},
                  "elapsed_seconds": previous_elapsed+time.perf_counter()-started}
        write_json(output_dir / "training.json", report)
        tracker.finish({"best_val_loss": best, "optimizer_steps": global_step, "checkpoint": str(output_dir / "best.pt"),
                        "last_checkpoint": str(output_dir / "last.pt"), "stop_reason": report["stop_reason"]})
        failed = False
        return report
    finally:
        if failed:
            tracker.finish({"status": "failed"}, failed=True)
