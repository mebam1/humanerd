"""Atomic trajectory shards and bounded, read-only memory-map access."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np


def content_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    """A manifest never exposes a partially written JSON document."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def array_schema(steps: int, state_dim: int, action_dim: int, dtype: str) -> dict:
    return {
        "states": {"shape": [steps + 1, state_dim], "dtype": dtype},
        "actions": {"shape": [steps, action_dim], "dtype": dtype},
        "contact_count": {"shape": [steps + 1], "dtype": "int32"},
        "penetration": {"shape": [steps + 1], "dtype": dtype},
        "min_distance": {"shape": [steps + 1], "dtype": dtype},
        "impact_event": {"shape": [steps], "dtype": "bool"},
        "persistent_contact": {"shape": [steps], "dtype": "bool"},
        "substep_penetration": {"shape": [steps], "dtype": dtype},
        "first_impact_time": {"shape": [steps], "dtype": dtype},
    }


@lru_cache(maxsize=64)
def _mapped_array(path: str) -> np.ndarray:
    # Do not explicitly close evicted mappings: a caller may still own a row view.
    return np.load(path, mmap_mode="r", allow_pickle=False)


def read_trajectory(root: Path, entry: dict, keys=None) -> dict:
    if "shard" in entry:
        selected = keys if keys is not None else entry["array_keys"]
        return {key: _mapped_array(str((root / entry["shard"] / f"{key}.npy").resolve()))[int(entry["row"])]
                for key in selected}
    with np.load(root / entry["file"], allow_pickle=False) as data:
        return {key: data[key] for key in (data.files if keys is None else keys)}


def validate_shard(root: Path, shard: str, fingerprint: str, schema: dict,
                   expected_entries: list[dict] | None = None) -> dict:
    """Check committed metadata and array headers/sizes without loading payloads."""
    directory = root / shard
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_fingerprint") != fingerprint or metadata.get("schema_hash") != content_hash(schema):
        raise ValueError(f"Incompatible dataset/schema in {shard}")
    entries = metadata["trajectories"]
    if expected_entries is not None and entries != expected_entries:
        raise ValueError(f"Trajectory identities differ in {shard}")
    for key, spec in schema.items():
        path = directory / f"{key}.npy"
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if data.shape != (len(entries), *spec["shape"]) or data.dtype != np.dtype(spec["dtype"]):
            raise ValueError(f"Unexpected array shape/dtype: {path}")
        # NPY offset + raw payload is exact; also catches truncated/appended shards.
        if path.stat().st_size != data.offset + data.nbytes:
            raise ValueError(f"Incomplete array payload: {path}")
        del data
    return {"path": shard, "trajectories": len(entries),
            "first_id": entries[0]["id"], "last_id": entries[-1]["id"]}


def expand_manifest(root: Path, manifest: dict) -> dict:
    if manifest.get("status", "complete") != "complete":
        raise ValueError("Dataset is incomplete; resume generation before training or evaluation")
    if manifest.get("format_version") != 2:
        return manifest
    entries = []
    for shard in manifest["shards"]:
        metadata = json.loads((root / shard["path"] / "metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("dataset_fingerprint") != manifest["dataset_fingerprint"]
                or metadata.get("schema_hash") != manifest["schema_hash"]):
            raise ValueError(f"Dataset metadata mismatch in {shard['path']}")
        entries.extend(metadata["trajectories"])
    if len(entries) != manifest["expected_trajectories"] or len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("Dataset manifest has missing or duplicate trajectory identities")
    return {**manifest, "trajectories": entries}
