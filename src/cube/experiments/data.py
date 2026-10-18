"""Trajectory-grouped datasets with deliberately held-out contact combinations."""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import OrderedDict
import itertools
import multiprocessing
import os
import platform
import time
import uuid

import numpy as np
import mujoco

from .common import make_sim, read_config, sim_state
from .storage import array_schema, atomic_json, content_hash, validate_shard

DEFAULT_SCENARIOS = ["free_fall", "impact", "rebound", "settle", "rest", "slide", "edge", "wall_only"]


def substep_events(sim, snapshots: list[dict]) -> dict:
    """Label events from observed substeps; labels never enter model features."""
    approaching_threshold = -0.05
    impact, persistent = False, False
    penetration = 0.0
    event_times = []
    previous = {}
    previous_state = None
    scratch = getattr(sim, "_event_scratch", None)
    if scratch is None:
        scratch = sim._event_scratch = mujoco.MjData(sim.model)
    for index, snapshot in enumerate(snapshots):
        current = {}
        for contact in snapshot["contacts"]:
            penetration = max(penetration, -float(contact["distance"]))
            if float(contact["distance"]) <= float(contact.get("inclusion_margin", sim.spec.contact_margin)):
                key = (int(contact["geom1"]), int(contact["geom2"]))
                # Multiple points on one pair remain represented by the most approaching point.
                if key not in current or contact["relative_normal_velocity"] < current[key]["relative_normal_velocity"]:
                    current[key] = contact
        if index == 0:
            if any(c["relative_normal_velocity"] < approaching_threshold for c in current.values()):
                impact = True
                event_times.append(float(snapshot["time"]))
        else:
            persistent = persistent or bool(set(current) & set(previous))
            for key, contact in current.items():
                speed = float(contact["relative_normal_velocity"])
                if previous_state is not None:
                    scratch.qpos[:] = previous_state[:sim.model.nq]
                    scratch.qvel[:] = previous_state[sim.model.nq:]
                    mujoco.mj_forward(sim.model, scratch)
                    velocity = []
                    for body_id in (contact["body1"], contact["body2"]):
                        jp, jr = np.zeros((3, sim.model.nv)), np.zeros((3, sim.model.nv))
                        mujoco.mj_jac(sim.model, scratch, jp, jr, np.asarray(contact["position"]), body_id)
                        velocity.append(jp @ scratch.qvel)
                    speed = float(np.dot(velocity[1]-velocity[0], contact["normal"]))
                if key not in previous and speed < approaching_threshold:
                    impact = True
                    event_times.append(float(snapshot["time"]))
        previous = current
        previous_state = np.asarray(snapshot["state"]) if "state" in snapshot else None
    return {"impact_event": impact, "persistent_contact": persistent,
            "substep_penetration": max(penetration, 0.0), "first_impact_time": min(event_times) if event_times else -1.0}


def _positive_integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _settings(config: dict | str | None, backend: str, device: str) -> dict:
    cfg = read_config(config)
    if backend not in {"mujoco", "warp"}:
        raise ValueError("backend must be mujoco or warp")
    scene = dict(cfg.get("scene", {}))
    kind = scene.setdefault("kind", cfg.get("kind", "cube"))
    scene.setdefault("dt", .02)
    scene.setdefault("substeps", 4)
    rigid = kind in {"cube", "cuboid"}
    scenarios = list(cfg.get("scenarios", DEFAULT_SCENARIOS if rigid else ["walk", "rest", "free_fall"]))
    ood = list(cfg.get("ood_scenarios", ["floor_wall"] if rigid else []))
    if not scenarios or len(set(scenarios)) != len(scenarios) or len(set(ood)) != len(ood):
        raise ValueError("Training scenarios must be nonempty and scenario lists cannot contain duplicates")
    if set(scenarios) & set(ood):
        raise ValueError("OOD environments cannot also be training environments")
    per_split = cfg.get("trajectories_per_split", {"train": 4, "val": 1, "test": 1})
    if set(per_split) != {"train", "val", "test"}:
        raise ValueError("trajectories_per_split must contain exactly train, val, test")
    per_split = {key: _positive_integer(per_split[key], key) for key in ("train", "val", "test")}
    steps = _positive_integer(cfg.get("steps", 64), "steps", 2)
    ood_count = _positive_integer(cfg.get("ood_trajectories", 2), "ood_trajectories", 0)
    if ood and not ood_count:
        raise ValueError("OOD scenarios require positive ood_trajectories")
    storage = dict(cfg.get("storage", {}))
    storage_format = storage.get("format", "npz")
    if storage_format not in {"npz", "sharded_npy"}:
        raise ValueError("storage.format must be npz or sharded_npy")
    dtype = storage.get("dtype", "float32" if storage_format == "sharded_npy" else "float64")
    if dtype not in {"float32", "float64"}:
        raise ValueError("storage.dtype must be float32 or float64")
    generation = dict(cfg.get("generation", {}))
    workers = _positive_integer(generation.get("workers", 1), "generation.workers")
    if backend == "warp" and workers != 1:
        raise ValueError("Warp generation uses one world/worker; set generation.workers=1 in the dataset config")
    if storage_format == "npz" and workers != 1:
        raise ValueError("Parallel generation requires storage.format=sharded_npy")
    _positive_integer(cfg.get("action_hold", 4), "action_hold")
    if not np.isfinite(float(cfg.get("action_scale", .15 if kind == "ant" else .2))):
        raise ValueError("action_scale must be finite")
    # Validate the scene without compiling a MuJoCo model for a dry-run plan.
    from cube.sim import SceneSpec
    for scenario in scenarios + ood:
        SceneSpec(**{**scene, "scenario": scenario})
    state_dim, action_dim = {"cube": (13, 6), "cuboid": (13, 6), "two_link": (15, 1), "ant": (29, 8)}[kind]
    schema = array_schema(steps, state_dim, action_dim, dtype)
    return {"config": cfg, "scene": scene, "kind": kind, "seed": int(cfg.get("seed", 0)),
            "scenarios": scenarios, "ood": ood, "per_split": per_split, "ood_count": ood_count,
            "steps": steps, "format": storage_format, "dtype": dtype, "schema": schema,
            "shard_size": _positive_integer(storage.get("trajectories_per_shard", 256), "trajectories_per_shard"),
            "workers": workers, "progress": bool(generation.get("progress", True)),
            "record_substeps": bool(generation.get("record_substeps", True)), "backend": backend, "device": device}


def dataset_plan(config: dict | str | None, backend: str = "mujoco", device: str = "cpu") -> dict:
    """Return exact trajectory/transition counts and payload size without generating data."""
    settings = _settings(config, backend, device)
    splits = {key: count * len(settings["scenarios"]) for key, count in settings["per_split"].items()}
    splits["ood"] = settings["ood_count"] * len(settings["ood"])
    trajectories = sum(splits.values())
    per_trajectory = sum(int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize
                         for spec in settings["schema"].values())
    return {"kind": settings["kind"], "backend": backend, "device": device, "storage_format": settings["format"],
            "trajectories": trajectories, "steps_per_trajectory": settings["steps"],
            "transitions": trajectories * settings["steps"], "splits": splits,
            "transitions_per_split": {key: count * settings["steps"] for key, count in splits.items()},
            "estimated_uncompressed_bytes": trajectories * per_trajectory,
            "size_estimate_note": "Array payload only; JSON/NPY headers and temporary unfinished shards add overhead; NPZ compresses payload.",
            "shards": (trajectories + settings["shard_size"] - 1) // settings["shard_size"] if settings["format"] == "sharded_npy" else 0,
            "workers": settings["workers"], "record_substeps": settings["record_substeps"],
            "event_temporal_resolution": "dt/substeps" if settings["record_substeps"] else "dt (endpoint observations only)"}


def _entries(settings: dict):
    trajectory_id = 0
    for scenario in settings["scenarios"] + settings["ood"]:
        counts = {"ood": settings["ood_count"]} if scenario in settings["ood"] else settings["per_split"]
        for split, count in counts.items():
            for _ in range(count):
                local_seed = settings["seed"] + 104729 * (trajectory_id + 1)
                entry = {"id": trajectory_id, "split": split, "scenario": scenario, "seed": local_seed,
                         "steps": settings["steps"], "scene_spec": {**settings["scene"], "scenario": scenario, "seed": local_seed}}
                if settings["format"] == "sharded_npy":
                    entry.update(shard=f"shards/{trajectory_id // settings['shard_size']:06d}",
                                 row=trajectory_id % settings["shard_size"], array_keys=list(settings["schema"]))
                else:
                    entry["file"] = f"trajectory_{trajectory_id:05d}.npz"
                yield entry
                trajectory_id += 1


_SIM_CACHE = OrderedDict()


def _episode_sim(entry: dict, settings: dict):
    key = content_hash({"scene": {key: value for key, value in entry["scene_spec"].items() if key != "seed"},
                        "backend": settings["backend"], "device": settings["device"]})
    if key in _SIM_CACHE:
        sim = _SIM_CACHE.pop(key)
        sim.reset_episode(entry["seed"])
    else:
        sim = make_sim(entry["scene_spec"], settings["backend"], settings["device"])
    _SIM_CACHE[key] = sim
    while len(_SIM_CACHE) > 16:
        _SIM_CACHE.popitem(last=False)[1].close()
    return sim


def _snapshot(sim) -> dict:
    return {"time": float(sim.data.time), "state": sim.state().tolist(), "contacts": sim.contact_info()}


def _simulate(entry: dict, settings: dict) -> dict:
    sim = _episode_sim(entry, settings)
    cfg, kind, scenario, steps = settings["config"], settings["kind"], entry["scenario"], settings["steps"]
    rng = np.random.default_rng(entry["seed"])
    result = {key: np.empty(spec["shape"], dtype=spec["dtype"]) for key, spec in settings["schema"].items()}
    result["states"][0] = sim_state(sim)
    diagnostic = sim.diagnostics()
    for key in ("contact_count", "penetration", "min_distance"):
        result[key][0] = diagnostic[key]
    random_action = np.zeros(sim.action_dim)
    for index in range(steps):
        if index % int(cfg.get("action_hold", 4)) == 0:
            random_action = rng.normal(0, float(cfg.get("action_scale", .15 if kind == "ant" else .2)), sim.action_dim)
        if kind == "ant":
            from .control import ant_gait_action
            mix = entry["id"] % 3
            # Control uses simulator float64 state even when storage uses float32.
            nominal = ant_gait_action(sim_state(sim), index * sim.dt)
            action = nominal if mix == 0 else nominal + random_action if mix == 1 else random_action
            action = np.clip(action, -1, 1)
        elif kind == "two_link":
            action = np.clip(random_action, -1, 1)
        else:
            action = np.zeros(sim.action_dim) if scenario in {"rest", "settle", "free_fall"} else random_action
        initial = None if settings["record_substeps"] else _snapshot(sim)
        next_state = sim.step(action, record_substeps=settings["record_substeps"])
        snapshots = sim.last_substep_contacts if initial is None else [initial, _snapshot(sim)]
        labels = substep_events(sim, snapshots)
        if not np.isfinite(next_state).all():
            raise RuntimeError(f"Non-finite simulator state in {scenario}, seed {entry['seed']}")
        result["states"][index + 1], result["actions"][index] = next_state, action
        for key, value in labels.items():
            result[key][index] = value
        diagnostic = sim.diagnostics()
        for key in ("contact_count", "penetration", "min_distance"):
            result[key][index + 1] = diagnostic[key]
    return result


def _write_shard(output: str, entries: list[dict], settings: dict, fingerprint: str) -> dict:
    root = Path(output)
    final = root / entries[0]["shard"]
    temporary = final.with_name(f".{final.name}.{uuid.uuid4().hex}.partial")
    temporary.mkdir(parents=True)
    arrays = {}
    try:
        for key, spec in settings["schema"].items():
            arrays[key] = np.lib.format.open_memmap(temporary / f"{key}.npy", mode="w+", dtype=spec["dtype"],
                                                   shape=(len(entries), *spec["shape"]))
        for row, entry in enumerate(entries):
            data = _simulate(entry, settings)
            for key in arrays:
                arrays[key][row] = data[key]
        for array in arrays.values():
            array.flush()
        del array
        arrays.clear()  # Release every map before rename (required on Windows).
        for key in settings["schema"]:
            with (temporary / f"{key}.npy").open("rb") as stream:
                os.fsync(stream.fileno())
        atomic_json(temporary / "metadata.json", {"dataset_fingerprint": fingerprint,
                    "schema_hash": content_hash(settings["schema"]), "trajectories": entries})
        os.replace(temporary, final)
        return validate_shard(root, entries[0]["shard"], fingerprint, settings["schema"], entries)
    except BaseException:
        # Incomplete directories stay explicitly hidden; resume never accepts them.
        arrays.clear()
        raise


def _fingerprint(settings: dict) -> str:
    # Worker count/progress/provenance are operational/documentary; data semantics
    # (including scene, solver, precision, layout, and labels) are immutable.
    return content_hash({"generator_version": 2, "mujoco_version": mujoco.__version__,
                         **{key: settings[key] for key in ("scene", "kind", "seed", "scenarios", "ood", "per_split",
                             "ood_count", "steps", "format", "dtype", "schema", "shard_size", "record_substeps", "backend", "device")},
                         "action_hold": settings["config"].get("action_hold", 4),
                         "action_scale": settings["config"].get("action_scale", .15 if settings["kind"] == "ant" else .2)})


def _generate_shards(root: Path, settings: dict, manifest: dict, progress) -> None:
    entries = iter(_entries(settings))
    fingerprint = manifest["dataset_fingerprint"]
    complete = {}

    def pending():
        while batch := list(itertools.islice(entries, settings["shard_size"])):
            shard = batch[0]["shard"]
            if (root / shard).exists():
                descriptor = validate_shard(root, shard, fingerprint, settings["schema"], batch)
                complete[shard] = descriptor
                progress.update(len(batch))
            else:
                yield batch

    def commit(descriptor):
        complete[descriptor["path"]] = descriptor
        manifest["shards"] = [complete[key] for key in sorted(complete)]
        atomic_json(root / "manifest.json", manifest)
        progress.update(descriptor["trajectories"])

    if settings["workers"] == 1:
        for batch in pending():
            commit(_write_shard(str(root), batch, settings, fingerprint))
    else:
        # Bounded submitted work avoids retaining 100K tasks/futures and prevents
        # serializing simulation arrays through process pipes.
        batches = iter(pending())
        with ProcessPoolExecutor(max_workers=settings["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = set()
            exhausted = False
            while futures or not exhausted:
                while len(futures) < settings["workers"] * 2 and not exhausted:
                    batch = next(batches, None)
                    if batch is None:
                        exhausted = True
                    else:
                        futures.add(pool.submit(_write_shard, str(root), batch, settings, fingerprint))
                if futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        commit(future.result())
    manifest["shards"] = [complete[key] for key in sorted(complete)]


def _generate_npz(root: Path, settings: dict, manifest: dict, progress) -> None:
    completed = manifest.get("trajectories", [])
    expected = iter(_entries(settings))
    for entry in completed:
        if entry != next(expected):
            raise ValueError("Existing NPZ trajectory metadata differs from generation plan")
        with np.load(root / entry["file"], allow_pickle=False) as data:
            for key, schema in settings["schema"].items():
                if data[key].shape != tuple(schema["shape"]) or data[key].dtype != np.dtype(schema["dtype"]):
                    raise ValueError(f"Invalid saved trajectory {entry['file']}")
        progress.update(1)
    for entry in expected:
        temporary = root / f".{entry['file']}.partial"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **_simulate(entry, settings))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / entry["file"])
        completed.append(entry)
        manifest["trajectories"] = completed
        atomic_json(root / "manifest.json", manifest)
        progress.update(1)


def generate_dataset(config: dict | str | None, output: str, backend: str = "mujoco", device: str = "cpu",
                     *, resume: bool = False) -> dict:
    """Generate bounded-memory shards; resume only exactly matching dataset semantics."""
    from .tracking import progress as progress_bar
    settings = _settings(config, backend, device)
    plan = dataset_plan(config, backend, device)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(settings)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Dataset already exists at {root}; pass --resume or choose a new output directory")
        manifest = read_config(manifest_path)
        if manifest.get("dataset_fingerprint") != fingerprint or manifest.get("schema_hash") != content_hash(settings["schema"]):
            raise ValueError("Existing dataset has an incompatible config/backend/seed/schema; use a new output directory")
        # A completed resume verifies files without changing the manifest identity
        # recorded by training checkpoints and suite plans.
        if manifest.get("status") == "complete":
            if settings["format"] == "sharded_npy":
                entries = iter(_entries(settings))
                while batch := list(itertools.islice(entries, settings["shard_size"])):
                    validate_shard(root, batch[0]["shard"], fingerprint, settings["schema"], batch)
            else:
                expected = list(_entries(settings))
                if manifest.get("trajectories") != expected:
                    raise ValueError("Existing NPZ trajectory metadata differs from generation plan")
                for entry in expected:
                    with np.load(root / entry["file"], allow_pickle=False) as data:
                        for key, schema in settings["schema"].items():
                            if data[key].shape != tuple(schema["shape"]) or data[key].dtype != np.dtype(schema["dtype"]):
                                raise ValueError(f"Invalid saved trajectory {entry['file']}")
            return {"manifest": str(manifest_path), "trajectories": plan["trajectories"], "transitions": plan["transitions"],
                    "splits": plan["splits"], "elapsed_seconds": manifest["elapsed_seconds"],
                    "dataset_fingerprint": fingerprint, "storage_format": settings["format"], "resumed_complete": True}
    elif any(root.iterdir()):
        raise FileExistsError(f"Nonempty output directory has no dataset manifest: {root}")
    else:
        manifest = {"format_version": 2 if settings["format"] == "sharded_npy" else 1,
                    "status": "incomplete", "kind": settings["kind"], "backend": backend, "device": device,
                    "seed": settings["seed"], "dataset_fingerprint": fingerprint,
                    "schema": settings["schema"], "schema_hash": content_hash(settings["schema"]),
                    "config": settings["config"], "scene_spec": settings["scene"],
                    "state_convention": "qpos(root xyz,wxyz,joints),qvel(root linear-world,angular-local,joints)",
                    "expected_trajectories": plan["trajectories"], "splits": plan["splits"], "shards": [], "trajectories": [],
                    "split_policy": "whole independent trajectories; OOD scenarios excluded entirely from train/val/test",
                    "normalization": "not fitted during generation; fitted from training trajectories only",
                    "contact_labels": "Contact activation + prior observed approach speed <-0.05 m/s; persistent means active at adjacent observations. Labels only, never model inputs.",
                    "record_substeps": settings["record_substeps"],
                    "event_temporal_resolution": plan["event_temporal_resolution"],
                    "elapsed_seconds": 0.0, "python": platform.python_version(), "mujoco": mujoco.__version__}
        atomic_json(manifest_path, manifest)
    started = time.perf_counter()
    manifest["status"] = "incomplete"
    atomic_json(manifest_path, manifest)
    with progress_bar(total=plan["trajectories"], desc=f"Generate {settings['kind']} ({backend})", unit="trajectory",
                      config={"enabled": settings["progress"]}) as progress:
        if settings["format"] == "sharded_npy":
            _generate_shards(root, settings, manifest, progress)
        else:
            _generate_npz(root, settings, manifest, progress)
    manifest["status"] = "complete"
    manifest["elapsed_seconds"] += time.perf_counter() - started
    atomic_json(manifest_path, manifest)
    return {"manifest": str(manifest_path), "trajectories": plan["trajectories"], "transitions": plan["transitions"],
            "splits": plan["splits"], "elapsed_seconds": manifest["elapsed_seconds"],
            "dataset_fingerprint": fingerprint, "storage_format": settings["format"]}



