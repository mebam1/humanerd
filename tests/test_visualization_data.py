"""Visualization uses real dataset storage and strictly autonomous model history."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from cube.experiments.common import load_trajectory
from cube.experiments.data import generate_dataset
from cube.experiments.learning import train
from cube.sim import SceneSpec, Simulator
from cube.visualization_data import list_trajectories, predict_trajectory, select_trajectories


def dataset_config(storage="npz"):
    return {"scene": {"kind": "cube"}, "seed": 8, "scenarios": ["impact", "rest"],
            "ood_scenarios": ["floor_wall"], "trajectories_per_split": {"train": 2, "val": 1, "test": 1},
            "ood_trajectories": 1, "steps": 4,
            "storage": {"format": storage, "trajectories_per_shard": 3},
            "generation": {"workers": 1, "progress": False}}


@pytest.mark.parametrize("storage", ["npz", "sharded_npy"])
def test_selection_reads_both_storage_formats_and_lists_total_before_limit(tmp_path, storage):
    generate_dataset(dataset_config(storage), str(tmp_path))
    result = list_trajectories(tmp_path, limit=2)
    assert result["total_matches"] == 4
    assert result["kind"] == "cube" and result["backend"] == "mujoco"
    assert len(result["trajectories"]) == 2
    root, _, entries = select_trajectories(tmp_path / "manifest.json", scenario="rest", limit=1)
    assert entries[0]["split"] == "train" and entries[0]["scenario"] == "rest"
    assert load_trajectory(root, entries[0])["states"].shape == (5, 13)
    assert list_trajectories(tmp_path, default_split=None)["total_matches"] == 9


def test_explicit_id_ignores_default_split_but_obeys_explicit_filters(tmp_path):
    generate_dataset(dataset_config(), str(tmp_path))
    root, _, entries = select_trajectories(tmp_path, split="ood")
    identity = entries[0]["id"]
    assert select_trajectories(root, trajectory_id=str(identity))[2][0]["split"] == "ood"
    with pytest.raises(ValueError, match="No trajectories match"):
        select_trajectories(root, trajectory_id=identity, split="train")
    with pytest.raises(ValueError, match="No trajectories match"):
        select_trajectories(root, trajectory_id=identity, scenario="rest")
    with pytest.raises(ValueError, match="does not exist"):
        select_trajectories(root, trajectory_id=999999)
    for limit in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            list_trajectories(root, limit=limit)


class RecordedModel(torch.nn.Module):
    """Known +1 position delta makes teacher forcing visible on every call."""

    def __init__(self, bad_call=None, bad_value=float("nan")):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.nq, self.nv, self.action_dim = 7, 6, 6
        self.config = {"name": "recording_test", "history": 3}
        self.calls, self.bad_call, self.bad_value = [], bad_call, bad_value

    def set_scene(self, scene):
        self.scene = scene

    def forward(self, state, action, history=None):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert state.dtype == torch.float32
        self.calls.append({"state": state.clone(), "action": action.clone(),
                           "states": history["states"].clone(), "actions": history["actions"].clone()})
        delta = torch.zeros((len(state), 12), device=state.device)
        delta[:, 0] = self.bad_value if len(self.calls) == self.bad_call else 1.
        return delta


def rollout_inputs():
    truth = np.zeros((5, 13), dtype=np.float64)
    truth[:, 3] = 1.
    truth[:, 0] = [0., 100., 200., 300., 400.]
    actions = np.tile(np.arange(4, dtype=np.float64)[:, None], (1, 6))
    return truth, actions


def test_rollout_history_never_uses_future_ground_truth(monkeypatch):
    from cube import visualization_data
    model = RecordedModel()
    monkeypatch.setattr(visualization_data, "load_checkpoint", lambda *args: (model, {"seed": 17, "epoch": 2}))
    truth, actions = rollout_inputs()
    output = predict_trajectory("model.pt", asdict(SceneSpec()), truth, actions, horizon=100, progress_config=False)
    assert output["requested_horizon"] == 100
    assert output["available_steps"] == output["completed_steps"] == 4
    assert output["diverged"] is False and output["reason"] is None
    assert output["checkpoint_meta"]["seed"] == 17
    assert output["predicted"].dtype == np.float32
    np.testing.assert_array_equal(output["predicted"][:, 0], np.arange(5))
    np.testing.assert_array_equal(output["truth"], truth)
    np.testing.assert_array_equal(model.calls[0]["states"][0, :, 0], [0, 0, 0])
    np.testing.assert_array_equal(model.calls[1]["states"][0, :, 0], [0, 0, 1])
    np.testing.assert_array_equal(model.calls[2]["states"][0, :, 0], [0, 1, 2])
    np.testing.assert_array_equal(model.calls[3]["states"][0, :, 0], [1, 2, 3])
    for index, call in enumerate(model.calls):
        np.testing.assert_array_equal(call["state"][0, 0], index)
        np.testing.assert_array_equal(call["actions"][0, :, 0], np.maximum(np.arange(index - 2, index + 1), 0))
        np.testing.assert_array_equal(call["action"][0], actions[index])
    assert model.scene._warp_model is None


@pytest.mark.parametrize("bad_value,reason", [(float("nan"), "Non-finite"), (20000., "magnitude")])
def test_divergence_returns_only_valid_prefix_without_gt_fallback(bad_value, reason):
    model = RecordedModel(bad_call=2, bad_value=bad_value)
    truth, actions = rollout_inputs()
    result = predict_trajectory("model.pt", asdict(SceneSpec()), truth, actions,
                                loaded_checkpoint=(model, {}), progress_config=False)
    assert result["diverged"] and reason in result["reason"]
    assert result["completed_steps"] == 1
    assert result["predicted"].shape == result["truth"].shape == (2, 13)
    assert result["actions"].shape == (1, 6)
    assert result["predicted"][-1, 0] == 1.0
    assert result["truth"][-1, 0] == 100.0
    assert np.isfinite(result["predicted"]).all()


def test_wrong_robot_checkpoint_rejected_and_scene_closed(monkeypatch):
    from cube import visualization_data
    model = RecordedModel()
    ant = Simulator(SceneSpec(kind="ant"))
    closed = []
    original_close = ant.close
    def close():
        closed.append(True)
        original_close()
    monkeypatch.setattr(ant, "close", close)
    monkeypatch.setattr(visualization_data, "make_sim", lambda spec: ant)
    states = np.stack([ant.state()] * 3)
    with pytest.raises(ValueError, match="Checkpoint morphology"):
        predict_trajectory("cube.pt", asdict(ant.spec), states, np.zeros((2, 8)),
                           loaded_checkpoint=(model, {}))
    assert closed == [True]


def test_rollout_of_real_fitted_checkpoint_from_sharded_dataset(tmp_path):
    dataset = tmp_path / "data"
    generate_dataset(dataset_config("sharded_npy"), str(dataset))
    trained = train(str(dataset), str(tmp_path / "trained"), model_name="nerd_like",
                    config={"epochs": 1, "batch_size": 8, "progress": {"enabled": False},
                            "model": {"hidden_dim": 16, "heads": 4, "history": 3,
                                      "points_per_face": 4, "floor_points": 4, "window_size": 8,
                                      "global_layers": 1, "local_layers": 1}})
    root, _, entries = select_trajectories(dataset, split="test", limit=1)
    values = load_trajectory(root, entries[0], keys=["states", "actions"])
    result = predict_trajectory(trained["checkpoint"], entries[0]["scene_spec"],
                                values["states"], values["actions"], horizon=3, progress_config=False)
    assert result["model_name"] == "nerd_like"
    assert result["completed_steps"] == 3 and not result["diverged"]
    assert result["checkpoint_meta"]["epoch"] == 1
    np.testing.assert_allclose(np.linalg.norm(result["predicted"][:, 3:7], axis=1), 1., atol=1e-5)
    np.testing.assert_array_equal(result["predicted"][0], values["states"][0].astype(np.float32))


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_invalid_horizon_fails_before_checkpoint_loading(horizon):
    truth, actions = rollout_inputs()
    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        predict_trajectory("unused.pt", {}, truth, actions, horizon=horizon)
