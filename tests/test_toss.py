"""Unforced, reproducible and spatially diverse rigid-body throws."""

import json
import os

import numpy as np
import pytest

from cube.experiments.common import load_manifest, load_trajectory
from cube.experiments.data import dataset_plan, generate_dataset
from cube.sim import SceneSpec, Simulator


def small_config(scenarios=None, **overrides):
    return {"scene": {"kind": "cube"}, "scenarios": scenarios or ["toss"],
            "ood_scenarios": [], "trajectories_per_split": {"train": 2, "val": 1, "test": 1},
            "steps": 6, "generation": {"progress": False}, **overrides}


def test_toss_initial_conditions_are_diverse_safe_and_repeatable():
    spec = SceneSpec(scenario="toss", kind="cuboid", half_extents=(.3, .2, .1))
    sim = Simulator(spec)
    samples = []
    for seed in range(512):
        state = sim.reset_episode(seed)
        samples.append(state)
        rotation = sim.data.xmat[sim.root_body_id].reshape(3, 3)
        vertical_radius = np.abs(rotation[2]) @ np.array(spec.half_extents)
        assert state[2] - vertical_radius >= .02
        assert sim.diagnostics()["contact_count"] == 0
        assert np.isclose(np.linalg.norm(state[3:7]), 1)
    states = np.asarray(samples)
    for selection, lower, upper in (
        (slice(0, 3), spec.toss_position_min, spec.toss_position_max),
        (slice(7, 10), spec.toss_linear_velocity_min, spec.toss_linear_velocity_max),
        (slice(10, 13), spec.toss_angular_velocity_min, spec.toss_angular_velocity_max),
    ):
        assert np.all(states[:, selection] >= lower)
        assert np.all(states[:, selection] <= upper)
        assert np.all(np.ptp(states[:, selection], axis=0) > .9 * (np.asarray(upper) - lower))
    # Isotropic quaternion second moment detects axis-only / Euler-angle sampling.
    np.testing.assert_allclose(states[:, 3:7].T @ states[:, 3:7] / len(states), np.eye(4) / 4, atol=.035)
    np.testing.assert_array_equal(sim.reset_episode(17), states[17])
    np.testing.assert_array_equal(Simulator(SceneSpec(**{**spec.to_dict(), "seed": 17})).state(), states[17])
    assert SceneSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_unforced_toss_has_ballistic_translation_and_changing_orientation():
    sim = Simulator(SceneSpec(scenario="toss", seed=7))
    initial = sim.state()
    for _ in range(5):
        final = sim.step(np.zeros(6))
        assert sim.diagnostics()["contact_count"] == 0
    np.testing.assert_allclose(final[7:10], initial[7:10] + np.asarray(sim.spec.gravity) * .1, atol=1e-10)
    assert abs(np.dot(initial[3:7], final[3:7])) < .99
    assert np.linalg.norm(final[:2] - initial[:2]) > .05


@pytest.mark.parametrize("override,match", [
    ({"kind": "ant"}, "only supported"),
    ({"toss_position_min": (-1, -1, .3)}, "circumscribed radius"),
    ({"toss_position_min": (2, -1, .8)}, "componentwise"),
    ({"toss_linear_velocity_max": (1, 2)}, "three finite"),
    ({"toss_angular_velocity_max": (1, 2, np.inf)}, "three finite"),
    ({"toss_angular_velocity_min": (-8, 9, -8)}, "componentwise"),
])
def test_toss_rejects_invalid_ranges_before_compilation(override, match):
    with pytest.raises(ValueError, match=match):
        SceneSpec(scenario="toss", **override)


@pytest.mark.parametrize("kind", ["cube", "cuboid"])
def test_rigid_data_defaults_to_zero_actions_even_with_nonzero_scale(tmp_path, kind):
    cfg = small_config(["impact", "slide", "edge", "toss"], scene={"kind": kind}, action_scale=10)
    generate_dataset(cfg, str(tmp_path))
    root, manifest = load_manifest(tmp_path)
    assert manifest["generator_version"] == 3
    assert manifest["rigid_action_mode"] == "zero"
    assert manifest["action_policy"] == dataset_plan(cfg)["action_policy"]
    assert manifest["action_policy"]["zero_scenarios"] == cfg["scenarios"]
    for entry in manifest["trajectories"]:
        arrays = load_trajectory(root, entry)
        np.testing.assert_array_equal(arrays["actions"], np.zeros((6, 6)))
        # The recorded trajectory must match direct unforced replay, including
        # reset of any previously applied wrench in the cached scene model.
        sim = Simulator(entry["scene_spec"])
        np.testing.assert_array_equal(arrays["states"][0], sim.state())
        for expected in arrays["states"][1:]:
            np.testing.assert_array_equal(expected, sim.step())


def test_random_wrench_opt_in_and_resume_policy_rejection(tmp_path):
    cfg = small_config(["impact", "rest", "settle", "free_fall"], rigid_action_mode="random_wrench")
    generate_dataset(cfg, str(tmp_path))
    root, manifest = load_manifest(tmp_path)
    for entry in manifest["trajectories"]:
        actions = load_trajectory(root, entry, keys=["actions"])["actions"]
        assert bool(np.any(actions)) == (entry["scenario"] == "impact")
    with pytest.raises(ValueError, match="incompatible"):
        generate_dataset({**cfg, "rigid_action_mode": "zero"}, str(tmp_path), resume=True)
    saved = (tmp_path / "manifest.json").read_bytes()
    assert generate_dataset(cfg, str(tmp_path), resume=True)["resumed_complete"]
    assert (tmp_path / "manifest.json").read_bytes() == saved


def test_generator_version_rejects_legacy_resume(tmp_path, monkeypatch):
    import cube.experiments.data as data
    cfg = small_config()
    monkeypatch.setattr(data, "GENERATOR_VERSION", 2)
    generate_dataset(cfg, str(tmp_path))
    monkeypatch.setattr(data, "GENERATOR_VERSION", 3)
    with pytest.raises(ValueError, match="incompatible"):
        generate_dataset(cfg, str(tmp_path), resume=True)


@pytest.mark.parametrize("config,match", [
    (small_config(rigid_action_mode="random_wrench"), "toss requires"),
    (small_config(rigid_action_mode="wrong"), "zero or random_wrench"),
    (small_config(["rest"], scene={"kind": "ant"}, rigid_action_mode="zero"), "only applies"),
    (small_config(action_scale=-1), "nonnegative"),
])
def test_invalid_action_policy_is_rejected_by_dry_run(config, match):
    with pytest.raises(ValueError, match=match):
        dataset_plan(config)


@pytest.mark.parametrize("kind,mode", [("ant", "ant_gait_mixture"), ("two_link", "random_motor_control")])
def test_articulated_control_remains_active(tmp_path, kind, mode):
    cfg = small_config(["rest"], scene={"kind": kind})
    generate_dataset(cfg, str(tmp_path))
    root, manifest = load_manifest(tmp_path)
    assert manifest["rigid_action_mode"] is None
    assert manifest["action_policy"]["mode"] == mode
    assert any(np.any(load_trajectory(root, entry, keys=["actions"])["actions"]) for entry in manifest["trajectories"])


@pytest.mark.skipif(os.getenv("CUBE_TEST_WARP") != "1", reason="Explicit opt-in: real Warp kernel compilation")
def test_toss_initialization_shared_with_real_warp():
    spec = SceneSpec(scenario="toss", seed=17)
    reference = Simulator(spec)
    warp = Simulator(spec, backend="warp", device=os.getenv("CUBE_WARP_DEVICE", "cuda:0"))
    np.testing.assert_array_equal(reference.state(), warp.state())
    for _ in range(5):
        np.testing.assert_allclose(warp.step(), reference.step(), atol=2e-4, rtol=2e-4)
