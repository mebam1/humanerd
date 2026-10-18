"""Real small-scale simulations exercise the same storage path as NeRD-size data."""
import json

import numpy as np
import pytest

from cube.experiments.common import load_manifest, load_trajectory
from cube.experiments.data import dataset_plan, generate_dataset


def config(workers=1):
    return {"seed": 7, "scene": {"kind": "cube"}, "scenarios": ["impact", "rest"],
            "ood_scenarios": ["floor_wall"], "trajectories_per_split": {"train": 2, "val": 1, "test": 1},
            "ood_trajectories": 1, "steps": 4,
            "storage": {"format": "sharded_npy", "trajectories_per_shard": 3, "dtype": "float64"},
            "generation": {"workers": workers, "progress": False}}


def test_plan_exact_cube_budget_without_generation():
    cfg = config()
    cfg.update(scenarios=["free_fall", "impact", "rebound", "settle", "rest", "slide", "edge", "wall_only"],
               trajectories_per_split={"train": 1250, "val": 128, "test": 256}, steps=100, ood_trajectories=2048)
    plan = dataset_plan(cfg)
    assert plan["splits"] == {"train": 10000, "val": 1024, "test": 2048, "ood": 2048}
    assert plan["transitions_per_split"]["train"] == 1_000_000
    assert plan["estimated_uncompressed_bytes"] > 0


def test_shards_match_npz_and_resume_without_rewriting(tmp_path):
    cfg = config()
    legacy_cfg = {**cfg, "storage": {"format": "npz", "dtype": "float64"}}
    generate_dataset(legacy_cfg, str(tmp_path / "npz"))
    report = generate_dataset(cfg, str(tmp_path / "sharded"))
    root, manifest = load_manifest(tmp_path / "sharded")
    old_root, legacy = load_manifest(tmp_path / "npz")
    assert report["trajectories"] == 9
    assert manifest["format_version"] == 2
    assert len(manifest["shards"]) == 3
    for entry, old_entry in zip(manifest["trajectories"], legacy["trajectories"]):
        actual, expected = load_trajectory(root, entry), load_trajectory(old_root, old_entry)
        assert isinstance(actual["states"], np.memmap)
        assert not actual["states"].flags.writeable
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
        assert set(load_trajectory(root, entry, keys=["states", "actions"])) == {"states", "actions"}
    saved_times = {path: path.stat().st_mtime_ns for path in root.glob("shards/*/*.npy")}
    manifest_bytes = (root / "manifest.json").read_bytes()
    generate_dataset(cfg, str(root), resume=True)
    assert (root / "manifest.json").read_bytes() == manifest_bytes
    assert saved_times == {path: path.stat().st_mtime_ns for path in saved_times}
    with pytest.raises(FileExistsError, match="resume"):
        generate_dataset(cfg, str(root))
    with pytest.raises(ValueError, match="incompatible"):
        generate_dataset({**cfg, "seed": 8}, str(root), resume=True)


def test_resume_recovers_committed_shard_not_yet_in_manifest(tmp_path):
    cfg = config()
    generate_dataset(cfg, str(tmp_path))
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"], manifest["shards"] = "incomplete", manifest["shards"][:1]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        load_manifest(tmp_path)
    generate_dataset(cfg, str(tmp_path), resume=True)
    assert len(load_manifest(tmp_path)[1]["trajectories"]) == 9


def test_legacy_resume_preserves_manifest_identity(tmp_path):
    cfg = {**config(), "storage": {"format": "npz"}}
    generate_dataset(cfg, str(tmp_path))
    manifest_bytes = (tmp_path / "manifest.json").read_bytes()
    assert generate_dataset(cfg, str(tmp_path), resume=True)["resumed_complete"]
    assert (tmp_path / "manifest.json").read_bytes() == manifest_bytes


def test_parallel_generation_is_deterministic(tmp_path):
    generate_dataset(config(1), str(tmp_path / "serial"))
    generate_dataset(config(2), str(tmp_path / "parallel"))
    serial_root, serial = load_manifest(tmp_path / "serial")
    parallel_root, parallel = load_manifest(tmp_path / "parallel")
    assert serial["dataset_fingerprint"] == parallel["dataset_fingerprint"]
    for left, right in zip(serial["trajectories"], parallel["trajectories"]):
        assert left == right
        left_data, right_data = load_trajectory(serial_root, left), load_trajectory(parallel_root, right)
        for key in left_data:
            np.testing.assert_array_equal(left_data[key], right_data[key])


def test_resume_rejects_corrupt_shard_schema(tmp_path):
    cfg = config()
    generate_dataset(cfg, str(tmp_path))
    path = tmp_path / "shards/000000/metadata.json"
    metadata = json.loads(path.read_text())
    metadata["schema_hash"] = "wrong"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="schema"):
        generate_dataset(cfg, str(tmp_path), resume=True)


def test_endpoint_labels_explicit_and_warp_workers_guarded(tmp_path):
    cfg = config()
    cfg["generation"]["record_substeps"] = False
    generate_dataset(cfg, str(tmp_path))
    manifest = load_manifest(tmp_path)[1]
    assert manifest["record_substeps"] is False
    assert "endpoint" in manifest["event_temporal_resolution"]
    with pytest.raises(ValueError, match="workers=1"):
        dataset_plan(config(2), backend="warp", device="cuda:0")
