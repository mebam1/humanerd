"""Long comparisons resume completed work without changing the scientific plan."""
import copy
import json

import pytest

from cube.experiments import generate_dataset
from cube.experiments import protocols


@pytest.fixture
def suite_config(tmp_path):
    dataset = tmp_path / "data"
    generate_dataset({
        "seed": 42,
        "scene": {"kind": "cube"},
        "scenarios": ["impact"],
        "ood_scenarios": ["floor_wall"],
        "trajectories_per_split": {"train": 2, "val": 1, "test": 1},
        "ood_trajectories": 1,
        "steps": 3,
        "generation": {"progress": False},
    }, str(dataset))
    training = {
        "epochs": 1,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": .001,
        "progress": {"enabled": False},
        "wandb": {"mode": "disabled", "project": "suite-test", "tags": ["shared"]},
        "model": {"hidden_dim": 8, "heads": 2, "points_per_face": 4,
                  "floor_points": 4, "window_size": 8, "local_layers": 1, "global_layers": 1},
    }
    training_path = tmp_path / "train.json"
    training_path.write_text(json.dumps(training), encoding="utf-8")
    return {
        "dataset": str(dataset),
        "training_config": str(training_path),
        "training": {"model": {"face_tokens": 2}, "wandb": {"group": "one-study"}},
        "budgets": [2],
        "seeds": [7],
        "models": ["state_mlp", {"name": "hierarchical", "label": "faces",
                                 "training_overrides": {"batch_size": 1, "gradient_accumulation_steps": 2}}],
        "evaluation": {"device": "cpu", "batch_size": 2, "max_trajectories": 1,
                       "save_rollouts": False},
        "rollout_horizon": 2,
    }


def test_filtered_suite_resumes_remaining_models_and_preserves_shared_configuration(suite_config, tmp_path, monkeypatch):
    output = tmp_path / "comparison"
    original = copy.deepcopy(suite_config)
    calls = []
    real_train = protocols.train

    def observed_train(dataset, directory, config, model, seed, **kwargs):
        calls.append((model, copy.deepcopy(config)))
        return real_train(dataset, directory, config, model, seed, **kwargs)

    monkeypatch.setattr(protocols, "train", observed_train)
    first = protocols.run_suite(suite_config, str(output), model_filter=["state_mlp"])
    assert not first["completed"]
    assert first["planned_runs"] == 2 and first["completed_runs"] == 1
    assert [row["model"] for row in first["runs"]] == ["state_mlp"]
    original_checkpoint = (output / "state_mlp_budget2_seed7" / "best.pt").read_bytes()

    final = protocols.run_suite(suite_config, str(output), resume=True)
    assert final["completed"] and final["completed_runs"] == 2
    assert [name for name, _ in calls] == ["state_mlp", "hierarchical"]
    assert original_checkpoint == (output / "state_mlp_budget2_seed7" / "best.pt").read_bytes()
    assert all(row["effective_batch_size"] == 2 for row in final["runs"])
    assert all(row["ood_metrics"] is not None for row in final["runs"])
    assert all("test/rollout/position_rmse" in row["evaluation_metrics"] for row in final["runs"])
    for _, cfg in calls:
        assert cfg["model"]["hidden_dim"] == 8 and cfg["model"]["face_tokens"] == 2
        assert cfg["wandb"]["project"] == "suite-test"
        assert cfg["wandb"]["tags"] == ["shared"] and cfg["wandb"]["group"] == "one-study"
        assert cfg["evaluation"]["save_rollouts"] is False
        assert "training_overrides" not in cfg["model"]
    assert calls[0][1]["batch_size"] == 2 and calls[1][1]["batch_size"] == 1
    assert suite_config == original

    again = protocols.run_suite(suite_config, str(output), resume=True)
    assert len(calls) == 2 and again["runs"] == final["runs"]


@pytest.mark.parametrize("mutation", ["training", "dataset", "seed"])
def test_suite_resume_rejects_changed_plan_before_training(suite_config, tmp_path, monkeypatch, mutation):
    output = tmp_path / "comparison"
    protocols.run_suite(suite_config, str(output), model_filter=["state_mlp"])
    changed = copy.deepcopy(suite_config)
    if mutation == "training":
        changed["training"]["epochs"] = 2
    elif mutation == "seed":
        changed["seeds"] = [9]
    else:
        manifest_path = tmp_path / "data" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["test_change"] = "dataset changed after training"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def forbidden_train(*args, **kwargs):
        pytest.fail("A changed scientific plan must be rejected before any training")

    monkeypatch.setattr(protocols, "train", forbidden_train)
    with pytest.raises(ValueError, match="configuration or dataset changed"):
        protocols.run_suite(changed, str(output), resume=True)


def test_evaluation_failure_resume_reuses_completed_training(suite_config, tmp_path, monkeypatch):
    output = tmp_path / "comparison"
    suite_config["models"] = ["state_mlp"]
    real_evaluate = protocols.evaluate

    def interrupted_evaluation(*args, **kwargs):
        raise RuntimeError("interrupted before test evaluation")

    monkeypatch.setattr(protocols, "evaluate", interrupted_evaluation)
    with pytest.raises(RuntimeError, match="interrupted"):
        protocols.run_suite(suite_config, str(output))
    assert (output / "state_mlp_budget2_seed7" / "training.json").exists()
    incomplete = json.loads((output / "suite.json").read_text(encoding="utf-8"))
    assert not incomplete["completed"] and incomplete["completed_runs"] == 0

    def forbidden_train(*args, **kwargs):
        pytest.fail("An evaluation retry must reuse the completed training checkpoint")

    monkeypatch.setattr(protocols, "train", forbidden_train)
    monkeypatch.setattr(protocols, "evaluate", real_evaluate)
    resumed = protocols.run_suite(suite_config, str(output), resume=True)
    assert resumed["completed"] and resumed["completed_runs"] == 1
    assert (output / "state_mlp_budget2_seed7" / "ood" / "evaluation.json").exists()


def test_existing_suite_requires_resume_and_missing_artifacts_are_not_accepted(suite_config, tmp_path):
    output = tmp_path / "comparison"
    suite_config["models"] = ["state_mlp"]
    protocols.run_suite(suite_config, str(output))
    with pytest.raises(ValueError, match="Use --resume"):
        protocols.run_suite(suite_config, str(output))
    (output / "state_mlp_budget2_seed7" / "ood" / "evaluation.json").unlink()
    with pytest.raises(ValueError, match="artifacts are missing"):
        protocols.run_suite(suite_config, str(output), resume=True)
