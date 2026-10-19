"""Physics checks, controlled contact interventions and matched experiment grids."""
from __future__ import annotations

import hashlib
import copy
import re
from pathlib import Path

import mujoco
import numpy as np
import torch

from .common import load_manifest, make_sim, read_config, sim_state, write_json
from .data import generate_dataset
from .learning import evaluate, load_checkpoint, train
from .storage import atomic_json, content_hash
from .tracking import RunTracker, progress


def _rotation(quat):
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=np.float64))
    return matrix.reshape(3, 3)


def _world_twist(state):
    nq = (len(state)+1)//2
    return np.r_[state[nq:nq+3], _rotation(state[3:7]) @ state[nq+3:nq+6]]


def validate(output: str, backend: str = "mujoco", device: str = "cpu") -> dict:
    results = []
    for kind in ("cube", "two_link", "ant"):
        sim = make_sim({"kind": kind, "scenario": "free_fall", "seed": 17}, backend, device)
        initial = sim_state(sim)
        action = np.linspace(-0.1, 0.1, sim.action_dim)
        repeats = []
        for _ in range(3):
            sim.reset(initial)
            repeats.append(sim.step(action))
        maximum = float(np.max(np.abs(np.asarray(repeats)-repeats[0])))
        results.append({"kind": kind, "state_dimension": len(initial), "action_dimension": sim.action_dim,
                        "repeat_transition_max_abs": maximum, "deterministic": maximum < (1e-6 if backend == "warp" else 1e-12),
                        "quaternion_norm_error": float(abs(np.linalg.norm(repeats[0][3:7])-1)),
                        "dt": sim.dt, "solver_dt": float(sim.model.opt.timestep), "substeps": sim.spec.substeps,
                        "friction": sim.model.geom_friction.tolist(), "contact_margin": sim.model.geom_margin.tolist(),
                        "mjcf_sha256": hashlib.sha256(sim.xml.encode()).hexdigest()})
    sim = make_sim({"kind": "cube", "scenario": "free_fall", "gravity": [0, 0, 0]})
    state = sim_state(sim)
    state[3:7] = np.array([0.9238795325, 0.10227645, 0.30682935, 0.2045529])
    state[3:7] /= np.linalg.norm(state[3:7])
    state[7:13] = [0.3, -0.2, 0.1, 0.5, -0.4, 0.8]
    sim.reset(state)
    local_point = np.array([0.18, -0.11, 0.17])
    before = sim.data.xpos[sim.root_body_id] + sim.data.xmat[sim.root_body_id].reshape(3,3) @ local_point
    r = sim.data.xmat[sim.root_body_id].reshape(3,3)
    expected_velocity = state[7:10] + np.cross(r @ state[10:13], before-state[:3])
    epsilon = 1e-7
    mujoco.mj_integratePos(sim.model, sim.data.qpos, sim.data.qvel, epsilon)
    mujoco.mj_forward(sim.model, sim.data)
    after = sim.data.xpos[sim.root_body_id] + sim.data.xmat[sim.root_body_id].reshape(3,3) @ local_point
    velocity_error = float(np.max(np.abs((after-before)/epsilon-expected_velocity)))
    report = {"backend": backend, "device": device, "checks": results,
              "point_velocity_finite_difference_error": velocity_error,
              "point_velocity_convention_pass": velocity_error < 1e-5,
              "passed": all(r["deterministic"] and r["quaternion_norm_error"] < 1e-5 for r in results) and velocity_error < 1e-5,
              "conventions": {"quaternion": "wxyz", "free_linear_velocity": "world", "free_angular_velocity": "root local",
                              "surface_velocity": "root COM v + (R omega_local) cross r", "rigid_action": "world COM wrench N,Nm",
                              "articulated_action": "normalized motor control times MJCF actuator gear", "cache": "reset integration state and warmstart for repeated transitions"}}
    return write_json(Path(output) / "validation.json", report)


def coupling(output: str, checkpoint: str | None = None, backend: str = "mujoco", device: str = "cpu",
             seeds: list[int] | None = None) -> dict:
    from cube.models import apply_delta
    model, _ = load_checkpoint(checkpoint) if checkpoint else (None, None)
    records = []
    for layout in ("same_face", "cross_face"):
        for seed in (seeds if seeds is not None else [0, 1, 2, 3]):
            base = {"kind": "cube", "scenario": "intervention", "support_layout": layout, "seed": seed}
            reference = make_sim({**base, "support": "AB"}, backend, device)
            initial = sim_state(reference)
            outcomes, predicted, conditions = {}, {}, {}
            for support in ("none", "A", "B", "AB"):
                sim = make_sim({**base, "support": support}, backend, device)
                sim.reset(initial)
                action = np.zeros(6)
                current_contact = sim.contact_info()
                next_state = sim.step(action)
                outcomes[support] = _world_twist(next_state)-_world_twist(initial)
                conditions[support] = {"initial_state": initial, "next_state": next_state,
                                       "current_contacts": current_contact, "next_contacts": sim.contact_info()}
                if model is not None:
                    model.set_scene(sim)
                    with torch.no_grad():
                        x, u = torch.tensor(initial[None], dtype=torch.float32), torch.zeros(1, 6)
                        history = {"states": x[:, None].repeat(1, int(model.config.get("history", 1)), 1),
                                   "actions": u[:, None].repeat(1, int(model.config.get("history", 1)), 1)}
                        prediction = apply_delta(x, model(x, u, history=history)).numpy()[0]
                    predicted[support] = _world_twist(prediction)-_world_twist(initial)
            truth = outcomes["AB"]-outcomes["A"]-outcomes["B"]+outcomes["none"]
            estimated = predicted["AB"]-predicted["A"]-predicted["B"]+predicted["none"] if predicted else None
            records.append({"layout": layout, "seed": seed, "ground_truth_C_AB": truth, "ground_truth_norm": float(np.linalg.norm(truth)),
                            "predicted_C_AB": estimated, "coupling_error_norm": float(np.linalg.norm(estimated-truth)) if estimated is not None else None,
                            "conditions": conditions, "velocity_deltas": outcomes})
    report = {"backend": backend, "checkpoint": checkpoint, "records": records,
              "mean_ground_truth_coupling_norm": float(np.mean([r["ground_truth_norm"] for r in records])),
              "mean_coupling_error_norm": float(np.mean([r["coupling_error_norm"] for r in records])) if model is not None else None,
              "protocol": "Within each seed/layout: identical qpos, qvel, material, action and solver; only support A/B existence changes. Twists are expressed in the same world frame."}
    return write_json(Path(output) / "coupling.json", report)


def _aggregate(records):
    groups = {}
    for record in records:
        key = (record["model"], record["budget"])
        groups.setdefault(key, []).append(record)
    summary = []
    for (name, budget), group in groups.items():
        metrics = {metric: {"mean": float(np.mean([r["metrics"][metric] for r in group])),
                            "std": float(np.std([r["metrics"][metric] for r in group])),
                            "seeds": len(group)} for metric in group[0]["metrics"]}
        aggregate = {"model": name, "budget": budget, "metrics": metrics}
        for section in ("ood_metrics", "evaluation_metrics"):
            rows = [row[section] for row in group if row.get(section)]
            if rows:
                common_keys = set.intersection(*(set(row) for row in rows))
                aggregate[section] = {key: {"mean": float(np.mean([row[key] for row in rows])),
                                          "std": float(np.std([row[key] for row in rows])), "seeds": len(rows)}
                                      for key in sorted(common_keys)}
        summary.append(aggregate)
    return summary


def _merge(base, changes):
    result = copy.deepcopy(base)
    for key, value in changes.items():
        result[key] = _merge(result.get(key, {}), value) if isinstance(value, dict) else copy.deepcopy(value)
    return result


def _evaluation_scalars(report, prefix):
    values = {f"{prefix}/one_step/{key}": value for key, value in report["mean_one_step"].items()}
    values.update({f"{prefix}/{key}": report[key] for key in
                   ("divergence_rate", "mean_latency_end_to_end_ms", "evaluated_trajectories")})
    for section in ("rollout", "impact", "persistent_contact"):
        if report.get(f"mean_{section}"):
            values.update({f"{prefix}/{section}/{key}": value for key, value in report[f"mean_{section}"].items()})
            continue
        rows = [row[section] for row in report.get("trajectories", []) if row.get(section)]
        if rows:
            values.update({f"{prefix}/{section}/{key}": float(np.mean([row[key] for row in rows])) for key in rows[0]})
    values.update({f"{prefix}/diagnostics/{key}": value for key, value in report.get("mean_diagnostics", {}).items() if value is not None})
    return values


def run_suite(config: dict | str, output: str, backend: str = "mujoco", device: str = "cpu",
              resume: bool = False, model_filter: list[str] | None = None) -> dict:
    cfg = read_config(config)
    output_dir = Path(output)
    dataset = cfg.get("dataset")
    if dataset is None:
        dataset = str(output_dir / "dataset")
        generate_dataset(cfg.get("data", {}), dataset, backend, device, resume=resume)
    dataset_root, manifest = load_manifest(dataset)
    ood_dataset = cfg.get("ood_dataset")
    ood_root, ood_manifest = load_manifest(ood_dataset) if ood_dataset is not None else (dataset_root, manifest)
    has_ood = any(entry["split"] == "ood" for entry in ood_manifest["trajectories"])
    if ood_dataset is not None:
        if not has_ood:
            raise ValueError("The configured OOD dataset must contain trajectories in the ood split")
        if ood_manifest["kind"] != manifest["kind"]:
            raise ValueError("The OOD dataset must have the same body kind as the training dataset")
    else:
        ood_dataset = dataset
    available = sum(e["split"] == "train" for e in manifest["trajectories"])
    budgets = cfg.get("budgets", [available])
    variants = []
    for value in cfg.get("models", ["state_mlp", "deepsets", "hierarchical"]):
        variant = {"name": value} if isinstance(value, str) else copy.deepcopy(value)
        variant.setdefault("label", variant["name"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", variant["label"]):
            raise ValueError("Suite model labels must use only letters, digits, underscores and hyphens")
        variants.append(variant)
    labels = [variant["label"] for variant in variants]
    if len(labels) != len(set(labels)):
        raise ValueError("Suite model labels must be unique")
    if model_filter and set(model_filter)-set(labels):
        raise ValueError(f"Unknown model labels: {sorted(set(model_filter)-set(labels))}")
    seeds = cfg.get("seeds", [0, 1, 2])
    if not seeds or not variants or not budgets or len(seeds) != len(set(seeds)) or len(budgets) != len(set(budgets)):
        raise ValueError("Suite requires nonempty, unique seeds and budgets and at least one model")
    if any(int(budget) != budget or budget < 1 or budget > available for budget in budgets):
        raise ValueError(f"Suite budget exceeds available training trajectories ({available}) or is invalid")
    base_training = _merge(read_config(cfg.get("training_config")), cfg.get("training", {}))
    evaluation_config = cfg.get("evaluation", {})
    base_training["evaluation"] = _merge(base_training.get("evaluation", {}), evaluation_config)
    signature = {"dataset_sha256": hashlib.sha256((dataset_root / "manifest.json").read_bytes()).hexdigest(),
                 "budgets": budgets, "seeds": seeds, "models": variants, "training": base_training,
                 "evaluation": evaluation_config, "rollout_horizon": int(cfg.get("rollout_horizon", 32))}
    # Omit the optional identity for existing suites so their resume plans remain
    # compatible. Switching OOD sources requires a new suite output directory.
    if cfg.get("ood_dataset") is not None:
        signature["ood_dataset_sha256"] = hashlib.sha256((ood_root / "manifest.json").read_bytes()).hexdigest()
    signature_hash = content_hash(signature)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path, plan_path = output_dir / "suite.json", output_dir / "suite_plan.json"
    records = []
    if plan_path.exists():
        previous_plan = read_config(plan_path)
        if not resume:
            raise ValueError("Suite output exists. Use --resume or choose a new output directory")
        if previous_plan.get("signature") != signature_hash:
            raise ValueError("Suite configuration or dataset changed; use a new output directory")
        if state_path.exists():
            records = read_config(state_path)["runs"]
    elif state_path.exists():
        raise ValueError("Legacy suite output has no resume plan; use a new output directory")
    atomic_json(plan_path, {"signature": signature_hash, "configuration": signature})
    completed_keys = {(row["model"], row["budget"], row["seed"]) for row in records}
    total = len(seeds)*len(budgets)*len(variants)
    jobs = [(seed, budget, variant) for seed in seeds for budget in budgets for variant in variants
            if not model_filter or variant["label"] in model_filter]

    def save():
        report = {"runs": records, "summary": _aggregate(records), "completed": len(records) == total,
                  "planned_runs": total, "completed_runs": len(records), "signature": signature_hash,
                  "fairness": "Same trajectory IDs per seed/budget, epoch passes, loss and fixed evaluation subsets. Effective batch and per-model overrides are recorded; parameter count and wall time are measured.",
                  "baseline_notice": "NeRD-like is a controlled local baseline, not official NeRD/FIGNet/RigidFormer."}
        if cfg.get("ood_dataset") is not None:
            report["ood_dataset"] = {"path": str(ood_root), "manifest_sha256": signature["ood_dataset_sha256"],
                                     "dataset_fingerprint": ood_manifest.get("dataset_fingerprint")}
        atomic_json(state_path, report)
        return report

    save()
    for seed, budget, source_variant in progress(jobs, config=base_training.get("progress"), desc="Model comparison", unit="run"):
        variant = copy.deepcopy(source_variant)
        name, label = variant.pop("name"), variant.pop("label")
        run_dir = output_dir / f"{label}_budget{budget}_seed{seed}"
        if (label, budget, seed) in completed_keys:
            if (not (run_dir / "best.pt").exists() or not (run_dir / "test/evaluation.json").exists()
                    or (has_ood and not (run_dir / "ood/evaluation.json").exists())):
                raise ValueError(f"Completed run artifacts are missing: {run_dir}")
            continue
        training_config = _merge(base_training, variant.pop("training_overrides", {}))
        training_config["trajectory_budget"] = budget
        training_config["model"] = _merge(training_config.get("model", {}), variant)
        training_config.setdefault("wandb", {}).setdefault("name", f"{label}-budget{budget}-seed{seed}")
        last = run_dir / "last.pt"
        training_path = run_dir / "training.json"
        if resume and training_path.exists():
            training = read_config(training_path)
        else:
            training = train(dataset, str(run_dir), training_config, name, seed,
                             resume=str(last) if resume and last.exists() else None)
        evaluation = evaluate(training["checkpoint"], dataset, str(run_dir / "test"),
                              rollout_horizon=int(cfg.get("rollout_horizon", 32)), **evaluation_config)
        ood_report = None
        if has_ood:
            ood_report = evaluate(training["checkpoint"], ood_dataset, str(run_dir / "ood"), "ood",
                                  int(cfg.get("rollout_horizon", 32)), **evaluation_config)
        scalars = _evaluation_scalars(evaluation, "test")
        if ood_report:
            scalars.update(_evaluation_scalars(ood_report, "ood"))
        tracker = RunTracker(training_config["wandb"], {}, run_dir, run_id=training.get("wandb", {}).get("id"))
        try:
            tracker.log(scalars)
            tracker.finish(scalars)
        except Exception:
            tracker.finish(failed=True)
            raise
        records.append({"model": label, "budget": budget, "seed": seed, "checkpoint": training["checkpoint"],
                        "metrics": evaluation["mean_one_step"], "ood_metrics": ood_report["mean_one_step"] if ood_report else None,
                        "evaluation_metrics": scalars, "train_seconds": training["elapsed_seconds"],
                        "latency_ms": evaluation["mean_latency_end_to_end_ms"],
                        "effective_batch_size": training["effective_batch_size"], "optimizer_steps": training["optimizer_steps"],
                        "wandb": training["wandb"], "parameters": training["transfer"]["total_parameters"]})
        completed_keys.add((label, budget, seed))
        save()
    return save()


def transfer(source_checkpoint: str, target_dataset: str, output: str, config: dict | str | None = None) -> dict:
    cfg = read_config(config)
    output_dir = Path(output)
    modes = cfg.get("modes", ["scratch", "full", "point-frozen", "aggregation-frozen"])
    budgets, seeds = cfg.get("budgets", [4, 16, 64]), cfg.get("seeds", [0, 1, 2])
    _, source = load_checkpoint(source_checkpoint)
    records = []
    for seed in seeds:
        for budget in budgets:
            for mode in modes:
                run_dir = output_dir / f"{mode}_budget{budget}_seed{seed}"
                training_config = {**cfg.get("training", {}), "trajectory_budget": budget}
                training_config["model"] = {**source["model_config"], **training_config.get("model", {})}
                training = train(target_dataset, str(run_dir), training_config, source["model_config"].get("name", "hierarchical"),
                                 seed=seed, transfer_checkpoint=source_checkpoint, transfer_mode=mode)
                evaluation = evaluate(training["checkpoint"], target_dataset, str(run_dir / "test"), rollout_horizon=int(cfg.get("rollout_horizon", 32)))
                records.append({"model": mode, "budget": budget, "seed": seed, "checkpoint": training["checkpoint"],
                                "metrics": evaluation["mean_one_step"], "train_seconds": training["elapsed_seconds"],
                                "weight_transfer": training["transfer"]})
                write_json(output_dir / "transfer.json", {"runs": records, "summary": _aggregate(records), "completed": False})
    return write_json(output_dir / "transfer.json", {"source": source_checkpoint, "target": target_dataset, "runs": records,
                      "summary": _aggregate(records), "completed": True,
                      "protocol": "Shape-compatible weights transferred; changed-DOF decoders initialized for target. Source architecture shared across all four modes; only designated parameter groups frozen."})
