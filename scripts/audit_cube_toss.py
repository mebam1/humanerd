"""Audit saved unactuated toss trajectories and plot their initial-state coverage."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from cube.experiments.common import load_manifest, load_trajectory, write_json
from cube.experiments.tracking import progress


def bounds(values):
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist()}


def audit(dataset, output, figure=None):
    root, manifest = load_manifest(dataset)
    initial, paths, first_contacts, penetration = [], [], [], []
    nonzero_actions = initial_contacts = nonfinite = 0
    maximum_action = 0.0
    splits = Counter()
    for entry in progress(manifest["trajectories"], desc="Audit toss", unit="trajectory"):
        arrays = load_trajectory(root, entry, keys=["states", "actions", "contact_count", "substep_penetration"])
        states, actions = arrays["states"], arrays["actions"]
        nonzero_actions += int(np.count_nonzero(actions))
        maximum_action = max(maximum_action, float(np.abs(actions).max()))
        nonfinite += int(not np.isfinite(states).all())
        splits[entry["split"]] += 1
        if entry["scenario"] != "toss":
            continue
        initial.append(np.array(states[0], dtype=np.float64))
        initial_contacts += int(arrays["contact_count"][0] > 0)
        hits = np.flatnonzero(arrays["contact_count"] > 0)
        first_contacts.append(float(hits[0] * entry["scene_spec"].get("dt", .02)) if len(hits) else None)
        penetration.append(float(np.max(arrays["substep_penetration"])))
        if len(paths) < 6:
            paths.append((entry["id"], np.array(states[:, :3])))
    if not initial:
        raise ValueError("Dataset contains no toss trajectories")
    initial = np.stack(initial)
    quaternions = initial[:, 3:7]
    norms = np.linalg.norm(quaternions, axis=1)
    contacts = np.array([t for t in first_contacts if t is not None])
    report = {"dataset": str(dataset), "dataset_fingerprint": manifest.get("dataset_fingerprint"),
              "backend": manifest["backend"], "generator_version": manifest.get("generator_version"),
              "splits": dict(splits), "total_trajectories": sum(splits.values()),
              "toss_trajectories": len(initial), "nonzero_action_values": nonzero_actions,
              "maximum_absolute_action": maximum_action, "nonfinite_trajectories": nonfinite,
              "toss_initial_contact_trajectories": initial_contacts,
              "toss_position_world_m": bounds(initial[:, :3]),
              "toss_linear_velocity_world_m_s": bounds(initial[:, 7:10]),
              "toss_angular_velocity_local_rad_s": bounds(initial[:, 10:13]),
              "toss_initial_linear_speed_m_s": bounds(np.linalg.norm(initial[:, 7:10], axis=1)),
              "toss_initial_angular_speed_rad_s": bounds(np.linalg.norm(initial[:, 10:13], axis=1)),
              "quaternion_norm_max_error": float(np.max(np.abs(norms-1))),
              "quaternion_second_moment": (quaternions.T @ quaternions / len(quaternions)).tolist(),
              "quaternion_second_moment_reference": "Uniform SO(3): diagonal 0.25, off-diagonal 0 in expectation",
              "toss_unique_initial_states": len(np.unique(initial, axis=0)),
              "toss_trajectories_with_ground_contact": len(contacts),
              "first_endpoint_contact_seconds": bounds(contacts) if len(contacts) else None,
              "toss_max_substep_penetration_m": float(max(penetration)),
              "example_ids": [identity for identity, _ in paths]}
    report["passed"] = bool(nonzero_actions == 0 and nonfinite == 0 and initial_contacts == 0 and np.max(np.abs(norms-1)) < 1e-6)
    write_json(output, report)
    if figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(13, 9), constrained_layout=True)
        axis = fig.add_subplot(221, projection="3d")
        for identity, positions in paths:
            axis.plot(*positions.T, label=f"ID {identity}")
            axis.scatter(*positions[0], marker="o", s=30)
        axis.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="First six toss paths; circles = initial positions")
        axis.legend(fontsize=8)
        for panel, values, title, unit in (
            (222, initial[:, :3], "Initial world position", "m"),
            (223, initial[:, 7:10], "Initial world linear velocity", "m/s"),
            (224, initial[:, 10:13], "Initial local angular velocity", "rad/s"),
        ):
            axis = fig.add_subplot(panel)
            for component, name in enumerate(("x", "y", "z")):
                axis.hist(values[:, component], bins=35, alpha=.5, label=name)
            axis.set(title=title, xlabel=unit, ylabel="Trajectories")
            axis.legend()
            axis.grid(alpha=.15)
        Path(figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=140)
        plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/cube-toss")
    parser.add_argument("--output", default="docs/results/cube_toss_audit.json")
    parser.add_argument("--figure", default="docs/assets/cube-toss-distribution.png")
    args = parser.parse_args()
    result = audit(args.dataset, args.output, args.figure)
    print({key: result[key] for key in ("total_trajectories", "toss_trajectories", "nonzero_action_values", "passed")})
    if not result["passed"]:
        raise SystemExit("Saved toss data failed the zero-action/finite-state/initial-contact audit")
