"""OOD throws retain random poses while testing new contact geometry."""

from collections import Counter

import mujoco
import numpy as np
import pytest

from cube.experiments.common import load_manifest, load_trajectory
from cube.experiments.data import dataset_plan, generate_dataset
from cube.sim import SceneSpec, Simulator


@pytest.mark.parametrize("scenario", ["toss_wall", "toss_valley"])
def test_ood_toss_keeps_diverse_safe_seeded_train_poses(scenario):
    train = Simulator(SceneSpec(scenario="toss"))
    ood = Simulator(SceneSpec(scenario=scenario))
    states = []
    for seed in range(256):
        expected = train.reset_episode(seed)
        actual = ood.reset_episode(seed)
        # Changing the environment must not replace the random launch pose
        # with the old fixed, floor-level wall-contact initialization.
        np.testing.assert_array_equal(actual[:7], expected[:7])
        if scenario == "toss_valley":
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual[7] > 0
            np.testing.assert_array_equal(actual[8:], expected[8:])
        assert ood.diagnostics()["contact_count"] == 0
        states.append(actual)

    states = np.asarray(states)
    assert np.all(np.ptp(states[:, :3], axis=0) > (.85, .85, .85))
    # The quaternion second moment rejects axis-only and Euler-angle sampling.
    np.testing.assert_allclose(states[:, 3:7].T @ states[:, 3:7] / len(states),
                               np.eye(4) / 4, atol=.06)
    np.testing.assert_array_equal(ood.reset_episode(17), states[17])
    recreated = Simulator(SceneSpec(scenario=scenario, seed=17))
    np.testing.assert_array_equal(recreated.state(), states[17])


def test_valley_is_two_planes_each_ten_degrees_with_a_lower_centre():
    sim = Simulator(SceneSpec(scenario="toss_valley"))
    ids = [mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
           for name in ("valley_left", "valley_right")]
    assert min(ids) >= 0
    assert all(sim.model.geom_type[index] == mujoco.mjtGeom.mjGEOM_PLANE for index in ids)
    assert all(sim.model.geom_bodyid[index] == 0 for index in ids)
    normals = np.array([sim.data.geom_xmat[index].reshape(3, 3)[:, 2] for index in ids])
    angle = np.deg2rad(10)
    np.testing.assert_allclose(np.sort(normals[:, 0]), (-np.sin(angle), np.sin(angle)), atol=1e-9)
    np.testing.assert_allclose(normals[:, 1], 0, atol=1e-9)
    np.testing.assert_allclose(normals[:, 2], np.cos(angle), atol=1e-9)
    # The upper collision envelope is a V, not a uniformly sloping floor
    # or a ridge with its centre higher than the two outer sides.
    for x in (-2., -.5, 0., .5, 2.):
        heights = [(np.dot(normal, sim.data.geom_xpos[index]) - normal[0] * x) / normal[2]
                   for index, normal in zip(ids, normals)]
        np.testing.assert_allclose(max(heights), abs(x) * np.tan(angle), atol=1e-9)


@pytest.mark.parametrize("scenario", ["toss_wall", "toss_valley"])
def test_unforced_ood_throws_reach_the_intended_surfaces(scenario):
    sim = Simulator(SceneSpec(scenario=scenario))
    observed_surfaces = set()
    for seed in range(20):
        sim.reset_episode(seed)
        first_contacts = {}
        for _ in range(100):
            assert np.isfinite(sim.step(record_substeps=True)).all()
            for snapshot in sim.last_substep_contacts:
                for contact in snapshot["contacts"]:
                    if contact["distance"] > contact["inclusion_margin"]:
                        continue
                    for name in (contact["geom1_name"], contact["geom2_name"]):
                        if name != "object":
                            first_contacts.setdefault(name, snapshot["time"])
                            observed_surfaces.add(name)
        if scenario == "toss_wall":
            assert "wall" in first_contacts, f"Seed {seed} missed the wall"
            assert "floor" in first_contacts, f"Seed {seed} never landed on the flat floor"
            assert first_contacts["wall"] < first_contacts["floor"], f"Seed {seed} landed before hitting the wall"
        else:
            assert set(first_contacts) & {"valley_left", "valley_right"}, f"Seed {seed} missed the valley"
    if scenario == "toss_valley":
        assert observed_surfaces == {"valley_left", "valley_right"}


def test_ood_dataset_is_half_wall_half_valley_with_replayable_zero_actions(tmp_path):
    config = {"seed": 27, "scene": {"kind": "cube"}, "scenarios": ["toss"],
              "ood_scenarios": ["toss_wall", "toss_valley"], "ood_trajectories": 3,
              "trajectories_per_split": {"train": 1, "val": 1, "test": 1},
              "steps": 6, "generation": {"progress": False}, "action_scale": 10}
    plan = dataset_plan(config)
    assert plan["splits"] == {"train": 1, "val": 1, "test": 1, "ood": 6}
    generate_dataset(config, str(tmp_path))
    root, manifest = load_manifest(tmp_path)
    ood = [entry for entry in manifest["trajectories"] if entry["split"] == "ood"]
    assert Counter(entry["scenario"] for entry in ood) == {"toss_wall": 3, "toss_valley": 3}
    assert all(entry["scenario"] == "toss" for entry in manifest["trajectories"] if entry["split"] != "ood")
    for entry in ood:
        arrays = load_trajectory(root, entry)
        np.testing.assert_array_equal(arrays["actions"], np.zeros((6, 6)))
        sim = Simulator(entry["scene_spec"])
        np.testing.assert_array_equal(arrays["states"][0], sim.state())
        for expected in arrays["states"][1:]:
            np.testing.assert_array_equal(sim.step(), expected)
    saved_manifest = (tmp_path / "manifest.json").read_bytes()
    assert generate_dataset(config, str(tmp_path), resume=True)["resumed_complete"]
    assert saved_manifest == (tmp_path / "manifest.json").read_bytes()


@pytest.mark.parametrize("scenario,overrides", [
    ("toss_valley", {"valley_angle_degrees": 0}),
    ("toss_valley", {"valley_angle_degrees": 90}),
    ("toss_valley", {"toss_position_min": (-1, -1, .4)}),
    ("toss_wall", {"toss_position_max": (2.4, 1, 1.8)}),
    ("toss_wall", {"kind": "ant"}),
    ("toss_valley", {"kind": "two_link"}),
])
def test_ood_toss_rejects_unsafe_geometry_or_initial_ranges(scenario, overrides):
    with pytest.raises(ValueError):
        SceneSpec(scenario=scenario, **overrides)


@pytest.mark.parametrize("scenario", ["toss_wall", "toss_valley"])
def test_ood_throw_data_cannot_opt_into_random_external_wrenches(scenario):
    config = {"scene": {"kind": "cube"}, "scenarios": ["rest"],
              "ood_scenarios": [scenario], "rigid_action_mode": "random_wrench"}
    with pytest.raises(ValueError, match="toss requires"):
        dataset_plan(config)
