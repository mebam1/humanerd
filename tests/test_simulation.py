"""Physics checks test state conventions and observed responses, not XML strings."""

import json
import os
import numpy as np
import pytest
import mujoco

from cube.sim import SceneSpec, Simulator


@pytest.mark.parametrize("kind,joints", [("cube", 0), ("cuboid", 0), ("two_link", 1), ("ant", 8)])
def test_shape_actuators_metadata(kind, joints):
    sim = Simulator(SceneSpec(kind=kind))
    assert sim.model.nq == 7 + joints
    assert sim.model.nv == 6 + joints
    assert sim.action_dim == (joints or 6)
    assert np.isclose(np.sum(sim.model.body_mass), sim.spec.mass)
    meta = json.loads(json.dumps(sim.kinematic_metadata()))
    assert len(meta["bodies"]) == sim.model.nbody
    assert SceneSpec.from_dict(json.loads(json.dumps(sim.spec.to_dict()))) == sim.spec
    assert np.isfinite(sim.step()).all()


def test_free_fall_matches_semi_implicit_euler():
    sim = Simulator(SceneSpec(scenario="free_fall", dt=.02, substeps=4))
    before = sim.state()
    after = sim.step()
    g = np.asarray(sim.spec.gravity)
    h = sim.dt / sim.spec.substeps
    expected_pos = before[:3] + before[7:10] * sim.dt + g * h*h * (4 * 5 / 2)
    np.testing.assert_allclose(after[7:10], before[7:10] + g * sim.dt, atol=1e-10)
    np.testing.assert_allclose(after[:3], expected_pos, atol=1e-10)


def test_free_root_wrench_acts_in_world_frame():
    sim = Simulator(SceneSpec(scenario="free_fall", gravity=(0, 0, 0), mass=2))
    state = sim.state()
    state[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    state[7:] = 0
    sim.reset(state)
    after = sim.step(np.array([2, 0, 0, 0, 0, 0]))
    np.testing.assert_allclose(after[7:10], [.02, 0, 0], atol=1e-10)


def test_contact_transition_replay_has_no_warmstart_history():
    sim = Simulator(SceneSpec(scenario="impact"))
    for _ in range(3):
        sim.step()
    state = sim.state()
    next_state = sim.step(np.zeros(6))
    sim.reset(state)
    replay = sim.step(np.zeros(6))
    np.testing.assert_allclose(next_state, replay, atol=1e-10)


def test_rest_stays_supported_and_slide_loses_speed():
    rest = Simulator(SceneSpec(scenario="rest"))
    start = rest.state()
    for _ in range(100):
        rest.step()
    assert abs(rest.state()[2] - start[2]) < .01
    assert np.linalg.norm(rest.state()[7:10]) < .01
    assert rest.diagnostics()["contact_count"] >= 1
    slide = Simulator(SceneSpec(scenario="slide"))
    speed = np.linalg.norm(slide.state()[7:9])
    for _ in range(20):
        slide.step()
    assert np.linalg.norm(slide.state()[7:9]) < speed * .5


def test_bounce_has_upward_separation_and_low_damping():
    sim = Simulator(SceneSpec(scenario="bounce", seed=1))
    vz = [sim.step()[9] for _ in range(20)]
    assert max(vz) > .2


@pytest.mark.parametrize("layout", ["same_face", "cross_face"])
def test_intervention_preserves_initial_state_and_existing_support(layout):
    sims = {s: Simulator(SceneSpec(scenario="intervention", support=s, support_layout=layout)) for s in ("none", "A", "B", "AB")}
    for sim in sims.values():
        np.testing.assert_array_equal(sim.state(), sims["none"].state())
    for name in ("A", "B"):
        a = sims[name].model.geom(f"support_{name}")
        ab = sims["AB"].model.geom(f"support_{name}")
        np.testing.assert_array_equal(a.pos, ab.pos)
        np.testing.assert_array_equal(a.size, ab.size)
    deltas = {name: sim.step()[7:] - sim._initial_state[7:] for name, sim in sims.items()}
    coupling = deltas["AB"] - deltas["A"] - deltas["B"] + deltas["none"]
    assert np.linalg.norm(coupling) > 1e-3


def test_wall_only_does_not_accidentally_add_floor():
    sim = Simulator(SceneSpec(scenario="wall_only"))
    assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1
    assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "wall") >= 0


@pytest.mark.parametrize("kind,scenario", [("cube", "rest"), ("ant", "walk"), ("ant", "rest")])
def test_episode_seeds_change_initial_conditions_without_changing_physics(kind, scenario):
    first = Simulator(SceneSpec(kind=kind, scenario=scenario, seed=1))
    second = Simulator(SceneSpec(kind=kind, scenario=scenario, seed=2))
    assert not np.array_equal(first.state(), second.state())
    np.testing.assert_array_equal(first.model.body_mass, second.model.body_mass)
    np.testing.assert_array_equal(first.model.geom_size, second.model.geom_size)
    again = Simulator(SceneSpec(kind=kind, scenario=scenario, seed=1))
    np.testing.assert_array_equal(first.state(), again.state())


def test_contact_information_is_geometry_and_current_velocity_only():
    sim = Simulator(SceneSpec(scenario="rest"))
    contacts = sim.contact_info()
    assert contacts
    assert all("impulse" not in c and "force" not in c for c in contacts)
    assert all(np.isclose(np.linalg.norm(c["normal"]), 1) for c in contacts)


def test_substep_contact_recording_preserves_transition_and_tracks_approach():
    spec = SceneSpec(scenario="impact", dt=.04, substeps=8, seed=3)
    sim = Simulator(spec)
    initial = sim.state()
    plain = sim.step()
    sim.reset(initial)
    recorded = sim.step(record_substeps=True)
    np.testing.assert_allclose(recorded, plain, atol=1e-10)
    assert len(sim.last_substep_contacts) == spec.substeps + 1
    np.testing.assert_array_equal(sim.last_substep_contacts[0]["state"], initial)
    np.testing.assert_array_equal(sim.last_substep_contacts[-1]["state"], recorded)
    np.testing.assert_allclose([r["time"] for r in sim.last_substep_contacts], np.linspace(0, spec.dt, spec.substeps + 1))
    contacts = [c for record in sim.last_substep_contacts for c in record["contacts"]]
    assert any(c["relative_normal_velocity"] < -.1 for c in contacts)
    assert all("impulse" not in c and "force" not in c for c in contacts)


@pytest.mark.skipif(os.getenv("CUBE_TEST_WARP") != "1", reason="Explicit opt-in: real Warp kernel compilation")
@pytest.mark.parametrize("kind", ["cube", "ant"])
def test_warp_executes_real_step_and_agrees_in_free_flight(kind):
    device = os.getenv("CUBE_WARP_DEVICE", "cuda:0")
    spec = SceneSpec(kind=kind, scenario="free_fall")
    reference = Simulator(spec)
    warp = Simulator(spec, backend="warp", device=device)
    assert warp._warp_model is not None
    initial = warp.state()
    result = warp.step()
    np.testing.assert_allclose(result, reference.step(), atol=2e-4, rtol=2e-4)
    warp.reset(initial)
    np.testing.assert_allclose(warp.step(record_substeps=True), result, atol=2e-5)
    assert len(warp.last_substep_contacts) == spec.substeps + 1
