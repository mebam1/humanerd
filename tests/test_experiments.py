"""Integration tests use actual simulator trajectories and actual fitted weights."""
import numpy as np
import pytest
import torch

from cube.experiments import ant_gait_action, coupling, evaluate, generate_dataset, train, validate
from cube.experiments.common import load_manifest, state_metrics
from cube.experiments.learning import _transfer_weights, load_checkpoint


def tiny_config(kind="cube"):
    return {"scene": {"kind": kind}, "seed": 51, "scenarios": ["impact", "rest"],
            "ood_scenarios": ["floor_wall"] if kind == "cube" else [],
            "trajectories_per_split": {"train": 1, "val": 1, "test": 1},
            "ood_trajectories": 1, "steps": 5}


def training_config():
    return {"epochs": 1, "batch_size": 5, "learning_rate": 0.001,
            "model": {"hidden_dim": 16, "heads": 4, "points_per_face": 4,
                      "floor_points": 4, "window_size": 8, "local_layers": 1, "global_layers": 1}}


def test_dataset_has_disjoint_trajectory_seeds_and_heldout_environments(tmp_path):
    generate_dataset(tiny_config(), str(tmp_path))
    _, manifest = load_manifest(tmp_path)
    entries = manifest["trajectories"]
    assert len({entry["seed"] for entry in entries}) == len(entries)
    assert all(entry["split"] == "ood" for entry in entries if entry["scenario"] == "floor_wall")
    assert all(entry["scenario"] != "floor_wall" for entry in entries if entry["split"] == "train")
    for entry in entries:
        data = np.load(tmp_path / entry["file"])
        assert data["states"].shape == (6, 13)
        assert data["actions"].shape == (5, 6)
        assert np.isfinite(data["states"]).all()
        assert data["impact_event"].shape == (5,)
        assert data["persistent_contact"].shape == (5,)
        assert np.all(data["substep_penetration"] >= 0)


def test_ood_overlap_is_rejected(tmp_path):
    cfg = tiny_config()
    cfg["ood_scenarios"] = ["impact"]
    with pytest.raises(ValueError, match="OOD"):
        generate_dataset(cfg, str(tmp_path))


def test_training_checkpoint_normalization_and_autonomous_rollout(tmp_path):
    dataset, run, evaluation = tmp_path / "data", tmp_path / "run", tmp_path / "evaluation"
    generate_dataset(tiny_config(), str(dataset))
    cfg = training_config()
    cfg["rollout_steps"] = 2
    report = train(str(dataset), str(run), cfg, model_name="hierarchical", seed=3)
    model, payload = load_checkpoint(report["checkpoint"])
    assert np.isfinite(report["best_val_loss"])
    _, manifest = load_manifest(dataset)
    train_ids = {entry["id"] for entry in manifest["trajectories"] if entry["split"] == "train"}
    assert set(payload["normalization"]["trajectory_ids"]) == train_ids
    assert payload["normalization"]["fit_split"] == "train"
    result = evaluate(report["checkpoint"], str(dataset), str(evaluation), rollout_horizon=4)
    assert result["mean_latency_end_to_end_ms"] > 0
    assert result["divergence_rate"] == 0
    rollout = np.load(next(evaluation.glob("rollout_*.npz")))
    assert rollout["predicted"].shape == (5, 13)
    assert np.allclose(np.linalg.norm(rollout["predicted"][:, 3:7], axis=1), 1, atol=1e-5)


def test_budget_cannot_be_mislabeled(tmp_path):
    generate_dataset(tiny_config(), str(tmp_path / "data"))
    with pytest.raises(ValueError, match="available"):
        train(str(tmp_path / "data"), str(tmp_path / "run"), {**training_config(), "trajectory_budget": 64})


def test_contact_counterfactuals_share_exact_initial_state(tmp_path):
    result = coupling(str(tmp_path), seeds=[4])
    assert {record["layout"] for record in result["records"]} == {"same_face", "cross_face"}
    for record in result["records"]:
        initial = record["conditions"]["none"]["initial_state"]
        assert all(np.array_equal(condition["initial_state"], initial) for condition in record["conditions"].values())
        delta = record["velocity_deltas"]
        expected = delta["AB"] - delta["A"] - delta["B"] + delta["none"]
        assert np.allclose(record["ground_truth_C_AB"], expected)
        assert record["ground_truth_norm"] > 0


def test_physics_and_ant_gait_are_finite(tmp_path):
    assert validate(str(tmp_path))["passed"]
    from cube.sim import SceneSpec, Simulator
    sim = Simulator(SceneSpec(kind="ant", scenario="walk"))
    action = ant_gait_action(sim.state(), 0.2)
    assert action.shape == (8,)
    assert np.max(np.abs(action)) <= 1
    assert np.isfinite(sim.step(action)).all()


def test_angular_error_compares_common_world_frame():
    truth = np.zeros((1, 13))
    truth[0, 3] = 1
    truth[0, 10] = 1
    prediction = truth.copy()
    prediction[0, 3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    prediction[0, 10:13] = [0, -1, 0]
    assert state_metrics(prediction, truth)["angular_velocity_rmse"] < 1e-12


def test_structural_transfer_does_not_freeze_untrained_cube_joint_head(tmp_path):
    from cube.models import build_model
    from cube.sim import SceneSpec, Simulator
    source = build_model(Simulator(SceneSpec()), training_config()["model"])
    target = build_model(Simulator(SceneSpec(kind="two_link")), training_config()["model"])
    for name, parameter in source.named_parameters():
        if name.startswith("aggregation.joint_decoder."):
            parameter.data.fill_(123)
    checkpoint = tmp_path / "source.pt"
    torch.save({"state_dict": source.state_dict(), "metadata": source.metadata}, checkpoint)
    report = _transfer_weights(target, str(checkpoint), "aggregation-frozen")
    assert report["novel_joint_head"]
    assert not any(name.startswith("aggregation.joint_decoder.") for name in report["loaded_parameters"])
    joint_head = [parameter for name, parameter in target.named_parameters() if name.startswith("aggregation.joint_decoder.")]
    assert joint_head and all(parameter.requires_grad for parameter in joint_head)
    assert all(not torch.all(parameter == 123) for parameter in joint_head)


def test_evaluation_rejects_nonpositive_horizon(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        evaluate("unused.pt", "unused-dataset", str(tmp_path), rollout_horizon=0)
