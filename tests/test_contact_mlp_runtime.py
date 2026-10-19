"""Contact inputs are current-state queries with train-only, resumable statistics."""
import json

import numpy as np
import pytest
import torch

from cube.experiments import evaluate, generate_dataset, train
from cube.experiments.common import load_manifest, load_trajectory
from cube.experiments.learning import load_checkpoint
from cube.experiments.training_runtime import SceneCache, TrajectoryCache, fit_input_normalization
from cube.models import build_model


@pytest.fixture
def contact_data(tmp_path):
    directory = tmp_path / "data"
    generate_dataset({"scene": {"kind": "cube"}, "seed": 71,
                      "scenarios": ["toss", "rest"], "ood_scenarios": ["toss_valley"],
                      "trajectories_per_split": {"train": 2, "val": 1, "test": 1},
                      "ood_trajectories": 1, "steps": 6,
                      "generation": {"progress": False}}, str(directory))
    return directory


def contact_settings():
    return {"device": "cpu", "epochs": 2, "batch_size": 4,
            "gradient_accumulation_steps": 2, "max_steps": 2, "learning_rate": 0.001,
            "normalization_chunk_size": 3, "input_normalization_batch_size": 3,
            "wandb": {"mode": "disabled"}, "progress": {"enabled": False},
            "evaluation": {"warmup_batches": 0},
            "model": {"history": 1, "hidden_dims": [16, 8], "normalize_input": True,
                      "normalize_output": True, "input_normalization_epsilon": 1e-5}}


def test_contact_input_statistics_match_all_train_current_states(contact_data):
    root, manifest = load_manifest(contact_data)
    entries = [entry for entry in manifest["trajectories"] if entry["split"] == "train"]
    cache, scenes = TrajectoryCache(root, size=1), SceneCache(size=1)
    model = build_model(scenes.get(entries[0])[0], {"name": "contact_mlp", "hidden_dims": [8]})
    stats = fit_input_normalization(model, entries, cache, scenes, 3, 1e-5, {"enabled": False})
    vectors = []
    for entry in entries:
        scenes.activate(model, entry)
        trajectory = load_trajectory(root, entry, keys=["states", "actions"])
        with torch.no_grad():
            vectors.append(model.raw_input(torch.tensor(np.array(trajectory["states"][:-1]), dtype=torch.float32),
                                           torch.tensor(np.array(trajectory["actions"]), dtype=torch.float32)).double())
    vectors = torch.cat(vectors)
    np.testing.assert_allclose(stats["mean"], vectors.mean(0), atol=1e-7)
    np.testing.assert_allclose(stats["scale"], (vectors.var(0, unbiased=False)+1e-5).sqrt(), atol=1e-7)
    assert stats["feature_dimension"] == 71
    assert stats["transitions"] == len(vectors) == 24
    assert set(stats["trajectory_ids"]) == {entry["id"] for entry in entries}
    assert len(cache.values) == len(scenes.values) == 1
    # Population epsilon scaling stays finite even for zero-action dimensions.
    assert np.isfinite(stats["scale"]).all() and min(stats["scale"]) > 0
    with pytest.raises(ValueError, match="train-only"):
        fit_input_normalization(model, [next(entry for entry in manifest["trajectories"] if entry["split"] == "val")],
                                cache, scenes, 3)


def test_contact_statistics_never_request_contact_or_future_labels(contact_data, monkeypatch):
    import cube.experiments.training_runtime as runtime
    root, manifest = load_manifest(contact_data)
    entry = next(entry for entry in manifest["trajectories"] if entry["split"] == "train")
    original = runtime.load_trajectory
    requests = []

    def only_current_sources(root, entry, keys=None):
        requests.append(keys)
        assert keys == ["states", "actions"]
        result = original(root, entry, keys=keys)
        # The unused terminal state must not affect current-input statistics.
        result["states"] = np.array(result["states"], copy=True)
        result["states"][-1] = np.nan
        return result

    monkeypatch.setattr(runtime, "load_trajectory", only_current_sources)
    scenes = SceneCache()
    model = build_model(scenes.get(entry)[0], {"name": "contact_mlp", "hidden_dims": [8]})
    stats = fit_input_normalization(model, [entry], TrajectoryCache(root), scenes, 2,
                                    progress_config={"enabled": False})
    assert stats["transitions"] == entry["steps"]
    assert np.isfinite(stats["mean"]).all()
    assert requests


def test_contact_training_checkpoint_resume_and_ood_evaluation(contact_data, tmp_path, monkeypatch):
    import cube.experiments.training_runtime as runtime
    cfg = contact_settings()
    full = train(str(contact_data), str(tmp_path / "full"), cfg, "contact_mlp", seed=5)
    partial = train(str(contact_data), str(tmp_path / "resumed"), {**cfg, "max_steps": 1}, "contact_mlp", seed=5)
    checkpoint_before = torch.load(partial["last_checkpoint"], map_location="cpu", weights_only=False)

    def forbidden_refit(*args, **kwargs):
        raise AssertionError("Resume must restore normalization without refitting")

    monkeypatch.setattr(runtime, "fit_input_normalization", forbidden_refit)
    monkeypatch.setattr(runtime, "fit_normalization", forbidden_refit)
    resumed = train(str(contact_data), str(tmp_path / "resumed"), cfg, "contact_mlp", seed=5,
                    resume=partial["last_checkpoint"])
    expected = torch.load(full["last_checkpoint"], map_location="cpu", weights_only=False)
    actual = torch.load(resumed["last_checkpoint"], map_location="cpu", weights_only=False)
    for key, value in expected["state_dict"].items():
        torch.testing.assert_close(actual["state_dict"][key], value, atol=0, rtol=0)
    assert actual["input_normalization"] == checkpoint_before["input_normalization"] == expected["input_normalization"]
    assert actual["input_normalization"]["fit_split"] == "train"
    assert actual["input_normalization"]["dataset_manifest_sha256"] == actual["dataset_fingerprint"]
    model, _ = load_checkpoint(resumed["last_checkpoint"])
    original_buffers = {key: value.clone() for key, value in model.named_buffers()}
    result = evaluate(resumed["last_checkpoint"], str(contact_data), str(tmp_path / "eval"),
                      split="ood", device="cpu", batch_size=3, rollout_horizon=6, save_rollouts=False)
    assert result["evaluated_trajectories"] == 1
    row = result["trajectories"][0]
    assert row["local_pair_coverage_initial"]["applicable"] is False
    assert row["local_pair_coverage_initial"]["coverage"] is None
    assert "contact_slot_diagnostics_initial" in row
    assert row["rollout_steps"] == 6
    reloaded, _ = load_checkpoint(resumed["last_checkpoint"])
    for key, value in reloaded.named_buffers():
        torch.testing.assert_close(value, original_buffers[key], atol=0, rtol=0)
    # JSON serialization must not emit NaN or fake point-cloud coverage.
    json.loads((tmp_path / "eval" / "evaluation.json").read_text())


def test_contact_dedicated_configs_do_not_enable_remote_logging():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    suite = json.loads((root / "configs/suite_contact_mlp.json").read_text())
    cfg = json.loads((root / suite["training_config"]).read_text())
    assert suite["models"] == ["contact_mlp"]
    assert suite["dataset"] == "data/cube-toss"
    assert suite["ood_dataset"] == "data/cube-toss-ood-v2"
    assert suite["evaluation"]["max_trajectories"] == 1000 and suite["rollout_horizon"] == 100
    assert cfg["model"]["hidden_dims"] == [512, 256, 128, 64]
    assert cfg["model"]["history"] == 1
    assert cfg["epochs"] == 5 and suite["budgets"] == [10000] and suite["seeds"] == [0]
    assert cfg["wandb"]["mode"] == "disabled"
