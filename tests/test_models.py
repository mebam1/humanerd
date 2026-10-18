"""Physics conventions and meaningful differentiable-model integration checks."""
import itertools

import mujoco
import numpy as np
import pytest
import torch

from cube.models import GeometryFeatures, apply_delta, build_model, hilbert_keys, state_delta, transfer_weights
from cube.models.serialization import local_pair_coverage
from cube.models.state import quat_normalize
from cube.sim import SceneSpec, Simulator


SMALL = {"hidden_dim": 16, "heads": 2, "local_layers": 2, "global_layers": 1,
         "points_per_face": 4, "floor_points": 16, "window_size": 8}


def tensors(sim):
    return torch.tensor(sim.state(), dtype=torch.float32)[None], torch.zeros(1, sim.action_dim)


@pytest.mark.parametrize("joints", [0, 1, 8])
def test_relative_state_roundtrip_and_quaternion_sign(joints):
    torch.manual_seed(4)
    state = torch.randn(3, 13 + 2*joints, dtype=torch.float64)
    following = torch.randn_like(state)
    state[:, 3:7] = quat_normalize(state[:, 3:7])
    following[:, 3:7] = quat_normalize(following[:, 3:7])
    delta = state_delta(state, following)
    reconstructed = apply_delta(state, delta)
    assert torch.allclose(reconstructed[:, :3], following[:, :3], atol=1e-9)
    assert torch.allclose(reconstructed[:, 7:], following[:, 7:], atol=1e-9)
    assert torch.allclose((reconstructed[:, 3:7]*following[:, 3:7]).sum(-1).abs(), torch.ones(3, dtype=torch.float64), atol=1e-9)
    opposite = following.clone()
    opposite[:, 3:7] *= -1
    assert torch.allclose(state_delta(state, opposite), delta, atol=1e-9)


def test_zero_delta_has_finite_rotation_gradients():
    state = torch.zeros(2, 13, dtype=torch.float64)
    state[:, 3] = 1
    delta = torch.zeros(2, 12, dtype=torch.float64, requires_grad=True)
    loss = apply_delta(state, delta).square().sum()
    loss.backward()
    assert torch.isfinite(delta.grad).all()


def test_hilbert_curve_has_grid_neighbor_continuity():
    grid = torch.tensor(list(itertools.product(range(4), repeat=3)))
    keys = hilbert_keys(grid, bits=2)
    assert len(keys.unique()) == 64
    ordered = grid[keys.argsort()]
    assert torch.equal((ordered[1:]-ordered[:-1]).abs().sum(-1), torch.ones(63, dtype=torch.long))


@pytest.mark.parametrize("kind", ["cube", "two_link", "ant"])
def test_torch_fk_matches_mujoco_pose_and_origin_jacobian_velocity(kind):
    sim = Simulator(SceneSpec(kind=kind, scenario="rest"))
    state = sim.state()
    rng = np.random.default_rng(12)
    state[3:7] = np.array([.9, .2, -.1, .3]) / np.linalg.norm([.9, .2, -.1, .3])
    state[7:sim.model.nq] += rng.uniform(-.2, .2, sim.model.nq-7)
    state[sim.model.nq:] = rng.uniform(-1., 1., sim.model.nv)
    sim.reset(state)
    geometry = GeometryFeatures(sim, SMALL).double()
    s = torch.tensor(sim.state(), dtype=torch.float64)[None].requires_grad_()
    p, q, v, w = geometry.forward_kinematics(s)
    np.testing.assert_allclose(p.detach().numpy()[0], sim.data.xpos, atol=2e-7)
    np.testing.assert_allclose(np.abs((q.detach().numpy()[0]*sim.data.xquat).sum(-1)), 1., atol=2e-7)
    for bid in geometry.dynamic_ids:
        jp, jr = np.zeros((3, sim.model.nv)), np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, jp, jr, bid)
        np.testing.assert_allclose(v.detach().numpy()[0, bid], jp @ sim.data.qvel, atol=2e-7)
        np.testing.assert_allclose(w.detach().numpy()[0, bid], jr @ sim.data.qvel, atol=2e-7)
    (p.square().sum() + v.square().sum() + q.square().sum()).backward()
    assert torch.isfinite(s.grad).all()
    if kind != "cube":
        assert s.grad[0, 7:sim.model.nq].abs().sum() > 0


def test_cube_physical_faces_edges_material_and_static_velocities():
    sim = Simulator(SceneSpec(scenario="rest"))
    geometry = GeometryFeatures(sim, SMALL)
    state, action = tensors(sim)
    state[:, sim.model.nq:sim.model.nq+6] = torch.tensor([1., 2., 3., .2, .3, .4])
    f = geometry(state, action)
    assert geometry.patch_count == 6
    assert f["points"].shape[-1] == 21
    assert f["body"].shape[-1] == 42
    assert torch.allclose(f["velocities"][:, ~geometry.dynamic], torch.zeros_like(f["velocities"][:, ~geometry.dynamic]))
    # n=2 includes all corners on each physical face, with distinct face normals.
    assert torch.allclose(geometry.local_points[geometry.dynamic].abs(), torch.full((24, 3), .2))
    bounce = GeometryFeatures(Simulator(SceneSpec(scenario="bounce")), SMALL)
    assert not torch.equal(geometry.material, bounce.material)


@pytest.mark.parametrize("name", ["hierarchical", "state_mlp", "nerd_like", "deepsets", "flat", "independent"])
@pytest.mark.parametrize("kind", ["cube", "ant"])
def test_models_forward_shapes_and_finite_gradients(name, kind):
    torch.manual_seed(0)
    sim = Simulator(SceneSpec(kind=kind, scenario="rest"))
    model = build_model(sim, {**SMALL, "name": name, "history": 2})
    state, action = tensors(sim)
    state.requires_grad_()
    delta = model(state, action)
    assert delta.shape == (1, 2*sim.model.nv)
    predicted = apply_delta(state, delta)
    predicted = apply_delta(predicted, model(predicted, action))
    predicted.square().mean().backward()
    assert torch.isfinite(predicted).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize("ablation", [{"mean_pool": True}, {"random_groups": True},
    {"no_face_exchange": True}, {"no_velocity": True}, {"no_inertia": True},
    {"serialization": "morton"}, {"face_tokens": 2}, {"face_tokens": 4}])
def test_ablation_options_run(ablation):
    sim = Simulator(SceneSpec(scenario="rest"))
    model = build_model(sim, {**SMALL, **ablation})
    state, action = tensors(sim)
    assert torch.isfinite(model(state, action)).all()


def test_pooling_is_invariant_to_within_face_point_permutation():
    torch.manual_seed(9)
    sim = Simulator(SceneSpec(scenario="rest"))
    model = build_model(sim, SMALL)
    state, action = tensors(sim)
    f = model.features(state, action)
    h = torch.randn(1, f["points"].shape[1], SMALL["hidden_dim"])
    expected = model.aggregation.patch_pool(h, f, model.features)
    original = model.features.patch_index.clone()
    model.features.patch_index.copy_(original.flip(-1))
    actual = model.aggregation.patch_pool(h, f, model.features)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_coverage_reports_only_close_object_environment_pairs():
    points = torch.tensor([[[0., 0., 0.], [.01, 0., 0.], [2., 0., 0.], [2.01, 0., 0.]]])
    result = local_pair_coverage(points, torch.tensor([True, False, True, False]), 4, 1, .02)
    assert result == {"near_pairs": 2, "covered_pairs": 2, "coverage": 1.}


def test_scene_swap_retains_parameters_and_cannot_transfer_geometry():
    sim = Simulator(SceneSpec(scenario="rest"))
    model = build_model(sim, SMALL)
    identities = [id(p) for p in model.parameters()]
    count = model.features.local_points.shape[0]
    model.set_scene(Simulator(SceneSpec(scenario="floor_wall")))
    assert identities == [id(p) for p in model.parameters()]
    assert model.features.local_points.shape[0] > count
    assert not any(k.startswith("features.") for k in model.state_dict())


def test_cube_to_ant_transfer_preserves_target_geometry_and_freezes_encoder():
    source = build_model(Simulator(SceneSpec(scenario="rest")), SMALL)
    target_sim = Simulator(SceneSpec(kind="ant", scenario="rest"))
    target = build_model(target_sim, SMALL)
    shape = target.features.local_points.clone()
    report = transfer_weights(source, target, freeze="point")
    assert report["transferred_tensors"] > 0
    assert torch.equal(shape, target.features.local_points)
    assert all(not p.requires_grad for p in target.point_encoder.parameters())
    assert all(p.requires_grad for p in target.aggregation.parameters())
    state, action = tensors(target_sim)
    target(state, action).square().mean().backward()
    assert all(p.grad is None for p in target.point_encoder.parameters())
    assert any(p.grad is not None for p in target.aggregation.parameters())


def test_independent_baseline_has_zero_counterfactual_coupling_by_construction():
    torch.manual_seed(21)
    simulations = [Simulator(SceneSpec(scenario="intervention", support=s)) for s in ("none", "A", "B", "AB")]
    model = build_model(simulations[0], {**SMALL, "name": "independent"})
    state, action = tensors(simulations[0])
    responses = []
    with torch.no_grad():
        for sim in simulations:
            model.set_scene(sim)
            responses.append(model(state, action))
    coupling = responses[3] - responses[1] - responses[2] + responses[0]
    assert torch.allclose(coupling, torch.zeros_like(coupling), atol=2e-7)


def test_new_joint_head_is_not_copied_or_frozen_in_structural_transfer():
    source = build_model(Simulator(SceneSpec()), SMALL)
    target = build_model(Simulator(SceneSpec(kind="two_link")), SMALL)
    initial = {name: p.clone() for name, p in target.aggregation.joint_decoder.named_parameters()}
    report = transfer_weights(source, target, freeze="aggregation")
    assert report["new_joint_head_trainable"]
    for name, p in target.aggregation.joint_decoder.named_parameters():
        assert p.requires_grad
        assert torch.equal(p, initial[name])
    assert not any(p.requires_grad for p in target.aggregation.base_decoder.parameters())


def test_wall_face_samples_follow_current_position_inside_physical_bounds():
    sim = Simulator(SceneSpec(scenario="floor_wall"))
    model = build_model(sim, {**SMALL, "points_per_face": 16})
    state, action = tensors(sim)
    state[:, 1] = 1.25
    f = model.features(state, action)
    wall = next(g for g in model.metadata["geoms"] if g.get("name") == "wall")
    indices = model.features.point_geom == wall["id"]
    samples = f["world_points"][0, indices]
    origin = torch.tensor(wall["pos"])
    extent = torch.tensor(wall["size"])
    assert ((samples-origin).abs() <= extent + 1e-6).all()
    # Every wall face lies on an exact box boundary; tangential points are local.
    assert (((samples-origin).abs()-extent).abs().amin(-1) < 1e-6).all()
    near_face = ((samples[:, 0] - .2).abs() < 1e-5) & (f["normals"][0, indices, 0] < -.99)
    assert ((samples[near_face, 1] - 1.25).abs() <= .5 + 1e-6).all()
