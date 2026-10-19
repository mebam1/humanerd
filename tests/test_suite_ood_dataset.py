"""Separate OOD sources must not change training/test or mix suite identities."""

import hashlib
import json
from pathlib import Path

import pytest

from cube.experiments import protocols
from cube.experiments.storage import content_hash


@pytest.fixture
def mocked_suite(tmp_path, monkeypatch):
    datasets = {}
    calls = []
    for name, entries in (
        ("train-data", [{"id": 0, "split": "train"}, {"id": 1, "split": "test"}]),
        ("ood-v2", [{"id": 12000, "split": "ood"}, {"id": 12500, "split": "ood"}]),
    ):
        root = tmp_path / name
        root.mkdir()
        manifest = {"kind": "cube", "trajectories": entries, "dataset_fingerprint": name}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        datasets[str(root)] = (root, manifest)

    def fake_train(dataset, directory, config, model, seed, **kwargs):
        calls.append(("train", dataset))
        root = Path(directory)
        root.mkdir(parents=True)
        checkpoint = root / "best.pt"
        checkpoint.write_text("placeholder checkpoint", encoding="utf-8")
        return {"checkpoint": str(checkpoint), "elapsed_seconds": 0,
                "effective_batch_size": 1, "optimizer_steps": 1, "wandb": {},
                "transfer": {"total_parameters": 1}}

    def fake_evaluate(checkpoint, dataset, directory, split="test", *args, **kwargs):
        calls.append((split, dataset))
        root = Path(directory)
        root.mkdir(parents=True)
        (root / "evaluation.json").write_text("{}", encoding="utf-8")
        return {"mean_one_step": {"position_rmse": .1}, "divergence_rate": 0,
                "mean_latency_end_to_end_ms": 0, "evaluated_trajectories": 1}

    class FakeTracker:
        def __init__(self, *args, **kwargs):
            pass

        def log(self, *args, **kwargs):
            pass

        def finish(self, *args, **kwargs):
            pass

    monkeypatch.setattr(protocols, "load_manifest", lambda path: datasets[str(path)])
    monkeypatch.setattr(protocols, "train", fake_train)
    monkeypatch.setattr(protocols, "evaluate", fake_evaluate)
    monkeypatch.setattr(protocols, "RunTracker", FakeTracker)
    config = {"dataset": str(tmp_path / "train-data"), "ood_dataset": str(tmp_path / "ood-v2"),
              "models": ["state_mlp"], "budgets": [1], "seeds": [0],
              "training": {"progress": {"enabled": False}, "wandb": {"mode": "disabled"}},
              "evaluation": {}, "rollout_horizon": 100}
    return config, datasets, calls


def test_separate_ood_source_routes_evaluation_and_records_manifest_identity(mocked_suite, tmp_path):
    config, datasets, calls = mocked_suite
    output = tmp_path / "suite"
    report = protocols.run_suite(config, str(output))
    assert calls == [("train", config["dataset"]), ("test", config["dataset"]), ("ood", config["ood_dataset"])]
    identity = hashlib.sha256((Path(config["ood_dataset"]) / "manifest.json").read_bytes()).hexdigest()
    assert report["ood_dataset"]["manifest_sha256"] == identity
    assert report["ood_dataset"]["dataset_fingerprint"] == "ood-v2"
    assert report["runs"][0]["ood_metrics"] == {"position_rmse": .1}
    plan = json.loads((output / "suite_plan.json").read_text(encoding="utf-8"))
    assert plan["configuration"]["ood_dataset_sha256"] == identity
    calls.clear()
    assert protocols.run_suite(config, str(output), resume=True)["completed"]
    assert calls == []


@pytest.mark.parametrize("change", ["ood_manifest", "remove_override"])
def test_changed_ood_source_rejects_resume_before_any_model_work(mocked_suite, tmp_path, change):
    config, datasets, calls = mocked_suite
    output = tmp_path / "suite"
    protocols.run_suite(config, str(output))
    calls.clear()
    if change == "ood_manifest":
        manifest_path = Path(config["ood_dataset"]) / "manifest.json"
        manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        config.pop("ood_dataset")
    with pytest.raises(ValueError, match="configuration or dataset changed"):
        protocols.run_suite(config, str(output), resume=True)
    assert calls == []


@pytest.mark.parametrize("change,match", [("empty", "must contain"), ("kind", "same body kind")])
def test_invalid_explicit_ood_source_is_rejected(mocked_suite, tmp_path, change, match):
    config, datasets, calls = mocked_suite
    manifest = datasets[config["ood_dataset"]][1]
    if change == "empty":
        manifest["trajectories"] = []
    else:
        manifest["kind"] = "ant"
    with pytest.raises(ValueError, match=match):
        protocols.run_suite(config, str(tmp_path / "suite"))
    assert calls == []


def test_absent_override_preserves_original_suite_signature(mocked_suite, tmp_path):
    config, datasets, calls = mocked_suite
    config.pop("ood_dataset")
    output = tmp_path / "suite"
    report = protocols.run_suite(config, str(output))
    original_signature = {
        "dataset_sha256": hashlib.sha256((Path(config["dataset"]) / "manifest.json").read_bytes()).hexdigest(),
        "budgets": [1], "seeds": [0],
        "models": [{"name": "state_mlp", "label": "state_mlp"}],
        "training": {**config["training"], "evaluation": {}},
        "evaluation": {}, "rollout_horizon": 100,
    }
    assert report["signature"] == content_hash(original_signature)
    assert "ood_dataset" not in report
    assert calls == [("train", config["dataset"]), ("test", config["dataset"])]
