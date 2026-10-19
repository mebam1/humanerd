"""Audit every saved OOD throw, including contact order and deterministic GT replay."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json

import mujoco
import numpy as np

from cube.experiments.common import load_manifest, load_trajectory, write_json
from cube.experiments.tracking import progress
from cube.sim import SceneSpec, Simulator


def bounds(values):
    values = np.asarray(values)
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist()}


def replay_contacts(sim, states):
    """Replay zero-action native physics and inspect every physics-boundary contact.

    Read contact geometry directly to avoid computing per-contact Jacobians that
    are only needed by training event labels. The integration, solver settings,
    initial float64 state, and substep boundaries match Simulator.step exactly.
    """
    first = {}
    maximum_error = 0.
    exact_storage_replay = True
    names = {index: mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, index)
             for index in range(sim.model.ngeom)}
    for expected in states[1:]:
        for _ in range(sim.spec.substeps):
            sim.data.qacc_warmstart[:] = 0
            mujoco.mj_step(sim.model, sim.data)
            mujoco.mj_kinematics(sim.model, sim.data)
            mujoco.mj_collision(sim.model, sim.data)
            for contact in sim.data.contact[:sim.data.ncon]:
                if contact.dist <= contact.includemargin:
                    for geom in (contact.geom1, contact.geom2):
                        name = names[int(geom)]
                        if name != "object":
                            first.setdefault(name, float(sim.data.time))
        actual = sim.state()
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
        exact_storage_replay &= bool(np.array_equal(actual.astype(expected.dtype), expected))
    return first, maximum_error, exact_storage_replay


def audit(dataset, output, expected_trajectories=1000):
    root, manifest = load_manifest(dataset)
    if manifest.get("status") != "complete":
        raise ValueError("Only a completed dataset can be audited")
    if manifest.get("backend") != "mujoco":
        raise ValueError("This exact native-physics replay audit requires backend=mujoco")
    entries = manifest["trajectories"]
    scenarios = Counter(entry["scenario"] for entry in entries)
    splits = Counter(entry["split"] for entry in entries)
    expected = {"toss_wall": expected_trajectories // 2, "toss_valley": expected_trajectories // 2}
    if expected_trajectories % 2:
        raise ValueError("expected_trajectories must be even for an exact 50:50 mixture")
    failures = {}
    caches = {}
    initial = {scenario: [] for scenario in expected}
    hits = Counter()
    first_contact_times = {"wall": [], "floor": [], "valley": []}
    nonzero_actions = nonfinite = initial_contacts = mismatched_train_states = 0
    unsafe_initial = replay_mismatches = nonpositive_wall_vx = 0
    quaternion_error = maximum_replay_error = maximum_penetration = 0.
    geometry_checks = []

    def fail(identity, reason):
        failures.setdefault(str(identity), []).append(reason)

    for entry in progress(entries, desc="Audit OOD GT", unit="trajectory"):
        identity, scenario = entry["id"], entry["scenario"]
        if scenario not in expected:
            fail(identity, f"Unexpected scenario: {scenario}")
            continue
        arrays = load_trajectory(root, entry, keys=["states", "actions", "contact_count", "substep_penetration"])
        states, actions = arrays["states"], arrays["actions"]
        if states.shape != (entry["steps"] + 1, 13) or actions.shape != (entry["steps"], 6):
            fail(identity, "Unexpected rigid state/action shape")
            continue
        count_nonzero = int(np.count_nonzero(actions))
        nonzero_actions += count_nonzero
        if count_nonzero:
            fail(identity, "Nonzero external action")
        if not np.isfinite(states).all():
            nonfinite += 1
            fail(identity, "Nonfinite stored state")
            continue
        norm_error = float(np.max(np.abs(np.linalg.norm(states[:, 3:7].astype(np.float64), axis=1) - 1)))
        quaternion_error = max(quaternion_error, norm_error)
        if norm_error >= 1e-6:
            fail(identity, "Stored quaternion norm error >= 1e-6")
        maximum_penetration = max(maximum_penetration, float(np.max(arrays["substep_penetration"])))

        scene = {key: value for key, value in entry["scene_spec"].items() if key != "seed"}
        cache_key = json.dumps(scene, sort_keys=True)
        if cache_key not in caches:
            spec = SceneSpec(**scene)
            sim = Simulator(spec)
            train = Simulator(replace(spec, scenario="toss"))
            caches[cache_key] = sim, train
            if scenario == "toss_valley":
                normals = [sim.data.geom_xmat[index].reshape(3, 3)[:, 2]
                           for index in range(sim.model.ngeom)
                           if mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, index)
                           in {"valley_left", "valley_right"}]
                normals = np.asarray(normals)
                angle = np.deg2rad(10)
                valid_geometry = bool(normals.shape == (2, 3)
                                      and np.allclose(np.sort(normals[:, 0]), [-np.sin(angle), np.sin(angle)], atol=1e-9)
                                      and np.allclose(normals[:, 1], 0, atol=1e-9)
                                      and np.allclose(normals[:, 2], np.cos(angle), atol=1e-9))
                geometry_checks.append({"scenario": scenario, "each_plane_degrees": spec.valley_angle_degrees,
                                        "normals": normals.tolist(), "valid": valid_geometry})
        sim, train = caches[cache_key]
        initial_state = sim.reset_episode(entry["seed"])
        expected_train = train.reset_episode(entry["seed"]).astype(states.dtype)
        initial[scenario].append(np.asarray(states[0], dtype=np.float64))
        if arrays["contact_count"][0] or sim.diagnostics()["contact_count"]:
            initial_contacts += 1
            fail(identity, "Contact already active at the initial pose")
        if not np.array_equal(initial_state.astype(states.dtype), states[0]):
            fail(identity, "Stored initial state differs from its scene seed")
        selection = np.s_[:] if scenario == "toss_valley" else np.r_[0:7, 8:13]
        if not np.array_equal(states[0][selection], expected_train[selection]):
            mismatched_train_states += 1
            fail(identity, "Initial pose or train-distribution velocity was changed")
        if scenario == "toss_wall" and states[0, 7] <= 0:
            nonpositive_wall_vx += 1
            fail(identity, "Wall toss does not travel toward +x")
        position = states[0, :3]
        if (np.any(position < np.asarray(sim.spec.toss_position_min) - 1e-6)
                or np.any(position > np.asarray(sim.spec.toss_position_max) + 1e-6)):
            unsafe_initial += 1
            fail(identity, "Initial position is outside the configured train launch range")

        contacts, replay_error, exact = replay_contacts(sim, states)
        maximum_replay_error = max(maximum_replay_error, replay_error)
        if not exact:
            replay_mismatches += 1
            fail(identity, "Native zero-action replay does not match saved states at storage precision")
        if scenario == "toss_wall":
            wall, floor = contacts.get("wall"), contacts.get("floor")
            hits["wall_contact"] += wall is not None
            hits["wall_branch_flat_floor_contact"] += floor is not None
            ordered = wall is not None and floor is not None and wall < floor
            hits["wall_before_floor"] += ordered
            if not ordered:
                fail(identity, "Wall branch did not hit the wall before landing on the flat floor")
            if wall is not None:
                first_contact_times["wall"].append(wall)
            if floor is not None:
                first_contact_times["floor"].append(floor)
        else:
            surfaces = {name: time for name, time in contacts.items() if name in {"valley_left", "valley_right"}}
            hits["valley_contact"] += bool(surfaces)
            hits["valley_left_contact"] += "valley_left" in surfaces
            hits["valley_right_contact"] += "valley_right" in surfaces
            if not surfaces:
                fail(identity, "Valley branch never reached either receiving plane")
            else:
                first_contact_times["valley"].append(min(surfaces.values()))

    mixture_valid = len(entries) == expected_trajectories and scenarios == expected
    split_valid = splits == {"ood": expected_trajectories}
    report = {"dataset": str(dataset), "dataset_fingerprint": manifest.get("dataset_fingerprint"),
              "backend": manifest["backend"], "generator_version": manifest.get("generator_version"),
              "total_trajectories": len(entries), "total_transitions": sum(entry["steps"] for entry in entries),
              "splits": dict(splits), "scenarios": dict(scenarios), "expected_scenarios": expected,
              "mixture_50_50_valid": mixture_valid, "ood_only_valid": split_valid,
              "nonzero_action_values": nonzero_actions, "nonfinite_trajectories": nonfinite,
              "initial_contact_trajectories": initial_contacts, "out_of_range_initial_positions": unsafe_initial,
              "train_initial_distribution_mismatch_trajectories": mismatched_train_states,
              "nonpositive_wall_x_velocity_trajectories": nonpositive_wall_vx,
              "quaternion_norm_max_error": quaternion_error, "replay_mismatch_trajectories": replay_mismatches,
              "replay_max_absolute_float64_to_storage_error": maximum_replay_error,
              "replay_comparison": "Exact equality after casting native float64 states to the saved storage dtype",
              "contact_observation": "Every dt/substeps physics boundary; no endpoint-only wall-contact approximation",
              "physical_contact_counts": dict(hits), "maximum_substep_penetration_m": maximum_penetration,
              "valley_geometry": geometry_checks,
              "first_contact_seconds": {name: bounds(values) if values else None
                                        for name, values in first_contact_times.items()},
              "initial_state_ranges": {scenario: {"position_world_m": bounds(np.asarray(values)[:, :3]),
                                                   "linear_velocity_world_m_s": bounds(np.asarray(values)[:, 7:10]),
                                                   "angular_velocity_local_rad_s": bounds(np.asarray(values)[:, 10:13])}
                                       for scenario, values in initial.items() if values},
              "failed_trajectories": failures}
    report["passed"] = bool(mixture_valid and split_valid and not failures
                            and geometry_checks and all(item["valid"] for item in geometry_checks))
    write_json(output, report)
    for sim, train in caches.values():
        sim.close()
        train.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/cube-toss-ood-v2")
    parser.add_argument("--output", default="docs/results/ood_toss_v2_audit.json")
    parser.add_argument("--expected-trajectories", type=int, default=1000)
    args = parser.parse_args()
    result = audit(args.dataset, args.output, args.expected_trajectories)
    print({key: result[key] for key in ("total_trajectories", "scenarios", "physical_contact_counts",
                                       "replay_mismatch_trajectories", "passed")})
    if not result["passed"]:
        raise SystemExit("OOD toss data failed the audit; inspect failed_trajectories in the saved report")
