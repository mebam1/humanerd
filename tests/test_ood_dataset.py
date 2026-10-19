"""Versioned OOD-only datasets keep training data separate and retain public IDs."""

from collections import Counter

import numpy as np
import pytest

from cube.experiments.common import load_manifest, load_trajectory
from cube.experiments.data import _entries, _settings, dataset_plan, generate_dataset
from cube.sim import Simulator


def ood_config(**overrides):
    return {
        "seed": 100,
        "ood_only": True,
        "trajectory_id_offset": 12000,
        "scene": {"kind": "cube", "valley_angle_degrees": 10.0},
        "scenarios": [],
        "ood_scenarios": ["toss_wall", "toss_valley"],
        "trajectories_per_split": {"train": 0, "val": 0, "test": 0},
        "ood_trajectories": 2,
        "steps": 4,
        "storage": {"format": "sharded_npy", "trajectories_per_shard": 3, "dtype": "float64"},
        "generation": {"workers": 1, "progress": False},
        **overrides,
    }


def test_versioned_ood_plan_and_public_ids_use_dense_shard_rows():
    config = ood_config(ood_trajectories=500, steps=100,
                        storage={"format": "sharded_npy", "trajectories_per_shard": 256})
    plan = dataset_plan(config)
    entries = list(_entries(_settings(config, "mujoco", "cpu")))
    assert plan["splits"] == {"train": 0, "val": 0, "test": 0, "ood": 1000}
    assert plan["transitions"] == 100000
    assert plan["shards"] == 4
    assert [entry["id"] for entry in entries] == list(range(12000, 13000))
    assert Counter(entry["scenario"] for entry in entries) == {"toss_wall": 500, "toss_valley": 500}
    assert all(entry["split"] == "ood" for entry in entries)
    assert (entries[0]["shard"], entries[0]["row"]) == ("shards/000000", 0)
    assert (entries[256]["shard"], entries[256]["row"]) == ("shards/000001", 0)
    assert (entries[-1]["shard"], entries[-1]["row"]) == ("shards/000003", 231)


def test_ood_only_shards_replay_and_resume_without_touching_training(tmp_path):
    config = ood_config()
    generate_dataset(config, str(tmp_path))
    root, manifest = load_manifest(tmp_path)
    assert manifest["ood_only"] is True
    assert manifest["trajectory_id_offset"] == 12000
    assert len(manifest["trajectories"]) == 4
    for entry in manifest["trajectories"]:
        assert entry["split"] == "ood"
        data = load_trajectory(root, entry)
        np.testing.assert_array_equal(data["actions"], np.zeros((4, 6)))
        sim = Simulator(entry["scene_spec"])
        try:
            np.testing.assert_array_equal(data["states"][0], sim.state())
            for state in data["states"][1:]:
                np.testing.assert_array_equal(state, sim.step())
        finally:
            sim.close()
    manifest_bytes = (tmp_path / "manifest.json").read_bytes()
    assert generate_dataset(config, str(tmp_path), resume=True)["resumed_complete"]
    assert (tmp_path / "manifest.json").read_bytes() == manifest_bytes
    with pytest.raises(ValueError, match="incompatible"):
        generate_dataset({**config, "trajectory_id_offset": 12001}, str(tmp_path), resume=True)


@pytest.mark.parametrize("overrides,match", [
    ({"ood_only": False}, "nonempty"),
    ({"ood_only": 1}, "true or false"),
    ({"scenarios": ["toss"]}, "empty training"),
    ({"ood_scenarios": []}, "nonempty ood"),
    ({"trajectories_per_split": {"train": 1, "val": 0, "test": 0}}, "zero train"),
    ({"trajectory_id_offset": -1}, "trajectory_id_offset"),
    ({"rigid_action_mode": "random_wrench"}, "toss requires"),
])
def test_ood_only_requires_explicit_consistent_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        dataset_plan(ood_config(**overrides))
