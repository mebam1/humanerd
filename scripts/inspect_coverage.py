"""Measure local-attention coverage and the independent-additive null control.

Run through Docker Compose; no trained checkpoints or external data are needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch

from cube.experiments.common import write_json
from cube.models import apply_delta, build_model
from cube.sim import SceneSpec, Simulator


def world_twist(state: np.ndarray) -> np.ndarray:
    nq = (len(state) + 1) // 2
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, np.asarray(state[3:7], dtype=np.float64))
    return np.r_[state[nq:nq + 3], rotation.reshape(3, 3) @ state[nq + 3:nq + 6]]


@torch.no_grad()
def inspect_coverage(seeds: list[int], threshold: float = .08) -> dict:
    profiles = {
        "physical_density": {"points_per_face": 64, "floor_points": 256, "window_size": 64},
        "smoke_density": {"points_per_face": 16, "floor_points": 64, "window_size": 32},
        "undersampled_density": {"points_per_face": 4, "floor_points": 16, "window_size": 16},
        "undersampled_window64": {"points_per_face": 4, "floor_points": 16, "window_size": 64},
    }
    records = []
    for profile, sampling in profiles.items():
        for serialization in ("hilbert", "morton"):
            for layers in (1, 3):
                config = {"name": "hierarchical", "hidden_dim": 64, "heads": 4,
                          "local_layers": layers, "global_layers": 2,
                          "serialization": serialization, **sampling}
                model = None
                for seed in seeds:
                    sim = Simulator(SceneSpec(scenario="rest", seed=seed))
                    if model is None:
                        model = build_model(sim, config)
                    else:
                        model.set_scene(sim)
                    state = torch.tensor(sim.state()[None], dtype=torch.float32)
                    action = torch.zeros(1, sim.action_dim)
                    coverage = model.coverage(state, action, threshold=threshold)
                    records.append({"profile": profile, "seed": seed, "scenario": "rest",
                                    "serialization": serialization, "local_layers": layers,
                                    **sampling, "threshold_m": threshold,
                                    "sample_count": int(model.features.local_points.shape[0]),
                                    "physical_contact_count": sim.diagnostics()["contact_count"],
                                    **coverage})
    summaries = []
    for profile in profiles:
        for serialization in ("hilbert", "morton"):
            for layers in (1, 3):
                selected = [r for r in records if (r["profile"], r["serialization"], r["local_layers"]) ==
                            (profile, serialization, layers)]
                near = sum(r["near_pairs"] for r in selected)
                covered = sum(r["covered_pairs"] for r in selected)
                summaries.append({"profile": profile, "serialization": serialization,
                                  "local_layers": layers, "seeds": len(selected),
                                  "near_pairs": near, "covered_pairs": covered,
                                  "weighted_coverage": covered / near if near else None,
                                  "states_without_near_pairs": sum(r["near_pairs"] == 0 for r in selected)})
    return {"threshold_m": threshold, "seeds": seeds, "records": records, "summary": summaries,
            "interpretation": "Direct object/environment sampled pairs within 0.08m sharing at least one local-attention window; not transitive reachability. Null coverage means no nearby sampled pair, not zero-percent coverage. Physical contacts can exist even when a sparse surface cloud has no nearby sampled pair. Layers 1 and 3 use respectively unshifted and both unshifted/half-window-shifted partitions; a third layer repeats the unshifted partition.",
            "learned_accuracy_claim": False}


@torch.no_grad()
def independent_null(seeds: list[int]) -> dict:
    records = []
    for layout in ("same_face", "cross_face"):
        for seed in seeds:
            torch.manual_seed(1000 + seed)
            sims = {support: Simulator(SceneSpec(scenario="intervention", support=support,
                                                support_layout=layout, seed=seed))
                    for support in ("none", "A", "B", "AB")}
            initial = sims["none"].state()
            model = build_model(sims["none"], {"name": "independent", "hidden_dim": 32, "heads": 4,
                                              "points_per_face": 4, "floor_points": 16,
                                              "local_layers": 1, "global_layers": 1})
            state, action = torch.tensor(initial[None], dtype=torch.float32), torch.zeros(1, 6)
            deltas, predicted_twists, true_twists = {}, {}, {}
            for support, sim in sims.items():
                np.testing.assert_array_equal(initial, sim.state())
                model.set_scene(sim)
                delta = model(state, action)
                predicted = apply_delta(state, delta).numpy()[0]
                deltas[support] = delta.numpy()[0]
                predicted_twists[support] = world_twist(predicted) - world_twist(initial)
                true_twists[support] = world_twist(sim.step()) - world_twist(initial)
            def contrast(values):
                return values["AB"] - values["A"] - values["B"] + values["none"]
            delta_null, twist_null, truth = map(contrast, (deltas, predicted_twists, true_twists))
            records.append({"layout": layout, "seed": seed, "network_initialization_seed": 1000 + seed,
                            "predicted_delta_C_AB": delta_null,
                            "predicted_delta_max_abs": float(np.abs(delta_null).max()),
                            "predicted_world_twist_C_AB": twist_null,
                            "predicted_world_twist_norm": float(np.linalg.norm(twist_null)),
                            "ground_truth_C_AB": truth, "ground_truth_norm": float(np.linalg.norm(truth))})
    tolerance = 2e-6
    maximum = max(r["predicted_world_twist_norm"] for r in records)
    return {"records": records, "seeds": seeds, "model": "independent", "trained": False,
            "max_predicted_world_twist_coupling_norm": maximum,
            "max_predicted_delta_component": max(r["predicted_delta_max_abs"] for r in records),
            "mean_ground_truth_coupling_norm": float(np.mean([r["ground_truth_norm"] for r in records])),
            "tolerance": tolerance, "passed": maximum < tolerance,
            "interpretation": "An architecture null test using identical current states and independent per-static-geometry responses. Untrained network weights are sufficient to test additivity. Near-zero predicted coupling does not indicate dynamics accuracy; it verifies the intended inability of this control model to represent non-additive support coupling."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/quick")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.output)
    coverage = inspect_coverage(args.seeds)
    null = independent_null(args.seeds)
    write_json(output / "coverage.json", coverage)
    write_json(output / "independent_null.json", null)
    for row in coverage["summary"]:
        print(row)
    print({k: v for k, v in null.items() if k != "records"})
    if not null["passed"]:
        raise SystemExit("Independent-additive null control failed")


if __name__ == "__main__":
    main()
