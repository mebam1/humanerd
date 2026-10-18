"""Runtime behavior: exact coverage, bounded caches, resumability, and telemetry modes."""
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cube.experiments import evaluate, generate_dataset, train
from cube.experiments.common import load_manifest, load_trajectory
from cube.experiments.learning import load_checkpoint
from cube.experiments.tracking import RunTracker
from cube.experiments.training_runtime import (BatchPlan, SceneCache, TrajectoryCache,
                                               fit_normalization, fixed_subset, scene_key)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    generate_dataset({"scene": {"kind": "cube"}, "seed": 18, "scenarios": ["impact", "rest"],
                      "ood_scenarios": [], "trajectories_per_split": {"train": 2, "val": 1, "test": 1},
                      "steps": 5, "generation": {"progress": False}}, str(root))
    return root


def settings():
    return {"epochs": 2, "batch_size": 3, "gradient_accumulation_steps": 2,
            "learning_rate": .001, "progress": {"enabled": False},
            "scheduler": {"name": "linear", "warmup_epochs": 0, "min_lr_ratio": .1},
            "model": {"hidden_dim": 8, "heads": 2, "points_per_face": 4, "floor_points": 4,
                      "window_size": 8, "local_layers": 1, "global_layers": 1}}


def test_batch_plan_covers_every_transition_once_and_packs_across_trajectories(dataset):
    root, manifest = load_manifest(dataset)
    entries = [entry for entry in manifest["trajectories"] if entry["split"] == "train"]
    cache = TrajectoryCache(root, size=1)
    plan = BatchPlan(entries, cache, batch_size=3, rollout_steps=2)
    batches = list(plan.iterate(17, epoch=2))
    pairs = [(entry["id"], int(index)) for batch in batches for entry, indices in batch for index in indices]
    assert len(pairs) == plan.transitions == 16
    assert len(set(pairs)) == len(pairs)
    assert set(pairs) == {(entry["id"], index) for entry in entries for index in range(4)}
    assert len(batches) == plan.batches
    assert any(len(batch) > 1 for batch in batches)
    assert all(len({scene_key(entry) for entry, _ in batch}) == 1 for batch in batches)
    assert len(cache.values) <= 1
    repeated = [(entry["id"], indices.tolist()) for batch in plan.iterate(17, epoch=2) for entry, indices in batch]
    assert repeated == [(entry["id"], indices.tolist()) for batch in batches for entry, indices in batch]


def test_streamed_normalization_matches_full_train_only_reference(dataset):
    from cube.models import state_delta
    root, manifest = load_manifest(dataset)
    entries = [entry for entry in manifest["trajectories"] if entry["split"] == "train"]
    cache = TrajectoryCache(root, size=1)
    normalization = fit_normalization(entries, cache, .02, 3, {"enabled": False})
    values = [load_trajectory(root, entry)["states"] for entry in entries]
    deltas = torch.cat([state_delta(torch.tensor(states[:-1], dtype=torch.float32),
                                   torch.tensor(states[1:], dtype=torch.float32)) for states in values])
    np.testing.assert_allclose(normalization["delta_mean"], deltas.mean(0).numpy(), atol=1e-7)
    np.testing.assert_allclose(normalization["delta_std"], deltas.std(0, unbiased=False).clamp_min(.02).numpy(), atol=1e-7)
    assert normalization["transitions"] == 20
    assert len(cache.values) == 1
    assert set(normalization["trajectory_ids"]) == {entry["id"] for entry in entries}


def test_batch_plan_interleaves_scenarios_without_losing_transition_coverage():
    entries = [{"id": scene*3+trajectory, "steps": 40,
                "scene_spec": {"kind": "cube", "scenario": scenario, "seed": trajectory}}
               for scene, scenario in enumerate(("impact", "rest", "slide")) for trajectory in range(3)]
    plan = BatchPlan(entries, None, batch_size=4)
    batches = list(plan.iterate(17, epoch=2))
    scenes = [scene_key(batch[0][0]) for batch in batches]
    assert all(len({scene_key(entry) for entry, _ in batch}) == 1 for batch in batches)
    # A grouped epoch would have only two switches; the deterministic interleaved
    # order switches often, so gradients see all contact cases throughout the epoch.
    assert sum(left != right for left, right in zip(scenes, scenes[1:])) > 20
    assert len(batches) == plan.batches == 90
    pairs = [(entry["id"], int(index)) for batch in batches for entry, indices in batch for index in indices]
    assert len(pairs) == len(set(pairs)) == plan.transitions == 360
    assert set(pairs) == {(entry["id"], index) for entry in entries for index in range(40)}
    repeated = [[(entry["id"], indices.tolist()) for entry, indices in batch] for batch in plan.iterate(17, epoch=2)]
    assert repeated == [[(entry["id"], indices.tolist()) for entry, indices in batch] for batch in batches]


def test_scene_cache_reuses_seed_independent_geometry(dataset):
    _, manifest = load_manifest(dataset)
    entry = manifest["trajectories"][0]
    other = copy.deepcopy(entry)
    other["scene_spec"]["seed"] += 1337
    cache = SceneCache(size=1)
    assert cache.get(entry)[0] is cache.get(other)[0]
    cache.get(manifest["trajectories"][-1])
    assert len(cache.values) == 1


def test_fixed_validation_subset_is_repeatable_and_scenario_stratified():
    entries = [{"id": index, "scenario": "rare" if index == 0 else "common"} for index in range(20)]
    subset = fixed_subset(entries, 4)
    assert subset == fixed_subset(entries, 4)
    assert len(subset) == 4
    assert {entry["scenario"] for entry in subset} == {"rare", "common"}


def test_partial_epoch_resume_matches_uninterrupted_weights(dataset, tmp_path):
    cfg = {**settings(), "max_steps": 2}
    uninterrupted = train(str(dataset), str(tmp_path / "full"), cfg, "state_mlp", seed=4)
    run_dir = tmp_path / "resumed"
    first = train(str(dataset), str(run_dir), {**cfg, "max_steps": 1}, "state_mlp", seed=4)
    first_last = torch.load(first["last_checkpoint"], map_location="cpu", weights_only=False)
    assert first_last["batch_cursor"] > 0
    assert first_last["global_step"] == 1
    resumed = train(str(dataset), str(run_dir), cfg, "state_mlp", seed=4, resume=first["last_checkpoint"])
    expected = torch.load(uninterrupted["last_checkpoint"], map_location="cpu", weights_only=False)
    actual = torch.load(resumed["last_checkpoint"], map_location="cpu", weights_only=False)
    assert actual["global_step"] == expected["global_step"] == 2
    for key in expected["state_dict"]:
        torch.testing.assert_close(actual["state_dict"][key], expected["state_dict"][key], atol=0, rtol=0)
    assert actual["scheduler_state_dict"] == expected["scheduler_state_dict"]
    assert actual["epoch_samples"] == expected["epoch_samples"]
    assert actual["normalization"] == expected["normalization"]
    assert (run_dir / "best.pt").is_file()


def test_resume_rejects_changed_dataset_or_optimization_contract(dataset, tmp_path):
    cfg = {**settings(), "max_steps": 1}
    report = train(str(dataset), str(tmp_path / "run"), cfg, "state_mlp")
    with pytest.raises(ValueError, match="already contains"):
        train(str(dataset), str(tmp_path / "run"), cfg, "state_mlp")
    with pytest.raises(ValueError, match="settings changed"):
        train(str(dataset), str(tmp_path / "run"), {**cfg, "batch_size": 9}, "state_mlp", resume=report["last_checkpoint"])
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["test_mutation"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        train(str(dataset), str(tmp_path / "run"), cfg, "state_mlp", resume=report["last_checkpoint"])


def test_streamed_evaluation_uses_fixed_subset_and_can_disable_rollout_archives(dataset, tmp_path):
    cfg = {**settings(), "max_steps": 1, "evaluation": {"retain_trajectory_metrics": False, "warmup_batches": 1}}
    report = train(str(dataset), str(tmp_path / "run"), cfg, "state_mlp")
    result = evaluate(report["checkpoint"], str(dataset), str(tmp_path / "eval"), rollout_horizon=2,
                      batch_size=2, max_trajectories=1, save_rollouts=False)
    assert result["evaluated_trajectories"] == 1
    assert result["available_trajectories"] == 2
    assert result["trajectories"] == []
    assert result["mean_latency_end_to_end_ms"] > 0
    assert result["mean_rollout"]
    assert "mean_impact" in result and "mean_persistent_contact" in result
    rows = (tmp_path / "eval" / "trajectories.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["one_step_batch_size"] == 2
    assert not list((tmp_path / "eval").glob("rollout_*.npz"))
    _, checkpoint = load_checkpoint(report["checkpoint"])
    assert checkpoint["normalization"]["fit_split"] == "train"


def test_wandb_disabled_is_sdk_and_credential_independent(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    tracker = RunTracker({"mode": "disabled"}, {}, tmp_path)
    tracker.log({"x": 1})
    tracker.finish()
    assert tracker.run is None


def test_wandb_online_mode_metrics_config_and_resume_identity(tmp_path, monkeypatch):
    calls, logs, finishes = [], [], []
    run = SimpleNamespace(id="stable-run", url="https://wandb.ai/example/run", settings=SimpleNamespace(mode="online"),
                          define_metric=lambda *args, **kwargs: None,
                          log=logs.append, summary={}, finish=lambda **kwargs: finishes.append(kwargs))
    fake = SimpleNamespace(Settings=lambda **kwargs: kwargs, init=lambda **kwargs: calls.append(kwargs) or run)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_API_KEY", "never-upload-this-secret")
    tracker = RunTracker({"mode": "online", "project": "research", "api_key": "also-secret"},
                         {"architecture": "test"}, tmp_path, run_id="stable-run")
    tracker.log({"optimizer_step": 1, "train/loss": .2})
    tracker.finish({"best_val_loss": .1})
    assert calls[0]["resume"] == "must" and calls[0]["id"] == "stable-run"
    assert calls[0]["mode"] == "online"
    assert calls[0]["settings"] == {"disable_git": True, "save_code": False}
    assert "secret" not in str(calls)
    assert logs == [{"optimizer_step": 1, "train/loss": .2}]
    assert run.summary["best_val_loss"] == .1
    assert finishes == [{"exit_code": 0}]


def test_wandb_online_refuses_silent_offline_fallback(tmp_path, monkeypatch):
    finished = []
    run = SimpleNamespace(settings=SimpleNamespace(mode="offline"), finish=lambda **kwargs: finished.append(kwargs))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Settings=lambda **kwargs: kwargs, init=lambda **kwargs: run))
    with pytest.raises(RuntimeError, match="not silently changed"):
        RunTracker({"mode": "online"}, {}, tmp_path)
    assert finished == [{"exit_code": 1}]

