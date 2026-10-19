"""Analytic NeRD-style contacts are current-state geometry, never a point cloud."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from cube.models import apply_delta, build_model, metadata_from_scene
from cube.models.contact import ContactFeatures, ContactMLP
from cube.models.state import quat_multiply, quat_rotate
from cube.sim import SceneSpec, Simulator


def _state(z=.25, x=0., y=0., dtype=torch.float32):
    state = torch.zeros(1, 13, dtype=dtype)
    state[0, :3] = torch.tensor([x, y, z], dtype=dtype)
    state[0, 3] = 1.
    return state


def _corners(dtype=torch.float32):
    # NeRD's eight fixed material points: x is the least-significant bit.
    return torch.tensor([[.2 if i & 1 else -.2,
                          .2 if i & 2 else -.2,
                          .2 if i & 4 else -.2] for i in range(8)], dtype=dtype)


def _flat_metadata():
    return deepcopy(metadata_from_scene(Simulator(SceneSpec(scenario="rest"))))


def _obstacle_metadata(quat=(1., 0., 0., 0.)):
    metadata = _flat_metadata()
    floor = next(g for g in metadata["geoms"] if int(g["type"]) == 0)
    floor.update(type=6, name="isolated_obstacle", pos=[0., 0., 0.],
                 quat=list(quat), size=[1., 2., 3.])
    return metadata


@pytest.mark.parametrize("z,expected_depth", [(.25, .05), (.15, -.05)])
def test_flat_plane_gives_signed_distance_projection_and_fixed_material_points(z, expected_depth):
    features = ContactFeatures(_flat_metadata())
    state = _state(z)
    result = features(state, torch.zeros(1, 6))
    contacts = result["contacts"][0]
    assert contacts.shape == (8, 10)
    assert result["contact_mask"].dtype == torch.bool
    assert torch.equal(result["contact_mask"][0], torch.tensor([True] * 4 + [False] * 4))
    corners = _corners()
    torch.testing.assert_close(contacts[:4, :3], corners[:4])
    expected_projection = corners[:4].clone()
    expected_projection[:, 2] = -z
    torch.testing.assert_close(contacts[:4, 3:6], expected_projection)
    torch.testing.assert_close(contacts[:4, 6:9], torch.tensor([[0., 0., 1.]]).expand(4, -1))
    torch.testing.assert_close(contacts[:4, 9], torch.full((4,), expected_depth))
    torch.testing.assert_close(result["distances"][0], corners[:, 2] + z)
    assert torch.count_nonzero(contacts[4:]) == 0


def test_far_contacts_are_entirely_zero_including_the_fixed_vertex():
    result = ContactFeatures(_flat_metadata())(_state(.31), torch.zeros(1, 6))
    assert not result["contact_mask"].any()
    assert torch.count_nonzero(result["contacts"]) == 0
    assert (result["distances"] > .1).all()
    assert torch.count_nonzero(result["input"][:, 6:62]) == 0


@pytest.mark.parametrize("margin_target", ["dynamic", "environment"])
def test_contact_activation_uses_four_times_larger_geom_margin(margin_target):
    metadata = _flat_metadata()
    target = next(g for g in metadata["geoms"] if
                  (int(g["body_id"]) != 0) == (margin_target == "dynamic"))
    target["margin"] = .03
    result = ContactFeatures(metadata)(_state(.31), torch.zeros(1, 6))
    assert torch.equal(result["contact_mask"][0], torch.tensor([True] * 4 + [False] * 4))
    torch.testing.assert_close(result["contacts"][0, :4, 9], torch.full((4,), .11))


def test_missing_environment_has_no_contact_or_fake_distance():
    metadata = _flat_metadata()
    metadata["geoms"] = [g for g in metadata["geoms"] if int(g["body_id"]) != 0]
    result = ContactFeatures(metadata)(_state(.1), torch.zeros(1, 6))
    assert torch.isposinf(result["distances"]).all()
    assert not result["contact_mask"].any()
    assert torch.count_nonzero(result["contacts"]) == 0
    assert torch.isfinite(result["input"]).all()


@pytest.mark.parametrize("contype,conaffinity", [(0, 0), (2, 2)])
def test_disabled_or_incompatible_collision_masks_do_not_create_contacts(contype, conaffinity):
    metadata = _flat_metadata()
    floor = next(g for g in metadata["geoms"] if int(g["body_id"]) == 0)
    floor["contype"], floor["conaffinity"] = contype, conaffinity
    result = ContactFeatures(metadata)(_state(.15), torch.zeros(1, 6))
    assert not result["contact_mask"].any()
    assert torch.isposinf(result["distances"]).all()


def test_cuboid_uses_its_eight_physical_vertices():
    sim = Simulator(SceneSpec(kind="cuboid", scenario="rest", half_extents=(.3, .1, .15)))
    result = ContactFeatures(sim)(_state(.19), torch.zeros(1, 6))
    expected = torch.tensor([[-.3, -.1, -.15], [.3, -.1, -.15],
                             [-.3, .1, -.15], [.3, .1, -.15]])
    torch.testing.assert_close(result["contacts"][0, :4, :3], expected)
    torch.testing.assert_close(result["contacts"][0, :4, 9], torch.full((4,), .04))


def test_batch_members_have_independent_queries_and_activation_masks():
    features = ContactFeatures(_flat_metadata())
    states = torch.cat((_state(.25), _state(.31), _state(.15)), 0)
    actions = torch.zeros(3, 6)
    batched = features(states, actions)
    for index in range(3):
        individual = features(states[index:index + 1], actions[index:index + 1])
        for key in ("contacts", "contact_mask", "distances", "input"):
            torch.testing.assert_close(batched[key][index:index + 1], individual[key])


@pytest.mark.parametrize("center_x", [1.25, .75])
def test_box_face_query_is_exact_outside_and_inside(center_x):
    state = _state(.21, x=center_x, y=.13)
    result = ContactFeatures(_obstacle_metadata())(state, torch.zeros(1, 6))
    expected_distances = center_x + _corners()[:, 0] - 1.
    torch.testing.assert_close(result["distances"][0], expected_distances)
    active = result["contact_mask"][0]
    contacts = result["contacts"][0, active]
    expected_projection = _corners()[active].clone()
    expected_projection[:, 0] = 1. - center_x
    torch.testing.assert_close(contacts[:, 3:6], expected_projection)
    torch.testing.assert_close(contacts[:, 6:9], torch.tensor([[1., 0., 0.]]).expand(len(contacts), -1))
    if center_x < 1.:
        assert active.all()
        assert (contacts[:, 9] < 0).all()


def test_box_edge_returns_euclidean_gap_and_diagonal_normal():
    state = _state(.1, x=1.25, y=2.25)
    result = ContactFeatures(_obstacle_metadata())(state, torch.zeros(1, 6))
    # Corner zero is (1.05, 2.05, -.1), nearest obstacle point (1, 2, -.1).
    contact = result["contacts"][0, 0]
    torch.testing.assert_close(contact[3:6], torch.tensor([-.25, -.25, -.2]))
    torch.testing.assert_close(contact[6:9], torch.tensor([1., 1., 0.]) / math.sqrt(2))
    torch.testing.assert_close(contact[9], torch.tensor(math.sqrt(.05 ** 2 + .05 ** 2)))


def test_rotated_box_uses_oriented_faces():
    q = (math.sqrt(.5), 0., 0., math.sqrt(.5))
    state = _state(.21, x=.13, y=1.25)
    result = ContactFeatures(_obstacle_metadata(q))(state, torch.zeros(1, 6))
    torch.testing.assert_close(result["distances"][0], 1.25 + _corners()[:, 1] - 1., atol=3e-7, rtol=1e-5)
    contacts = result["contacts"][0, result["contact_mask"][0]]
    torch.testing.assert_close(contacts[:, 6:9], torch.tensor([[0., 1., 0.]]).expand(len(contacts), -1), atol=3e-7, rtol=1e-5)
    torch.testing.assert_close(contacts[:, 4], torch.full((len(contacts),), -.25), atol=3e-7, rtol=1e-5)


@pytest.mark.parametrize("center_x", [-1., 0., 1.])
def test_valley_selects_highest_support_halfspace_for_each_vertex(center_x):
    metadata = metadata_from_scene(Simulator(SceneSpec(scenario="toss_valley")))
    angle = math.radians(10.)
    state = _state(.2 + abs(center_x) * math.tan(angle), x=center_x)
    result = ContactFeatures(metadata)(state, torch.zeros(1, 6))
    world = _corners() + state[0, :3]
    expected_depth = world[:, 2] * math.cos(angle) - world[:, 0].abs() * math.sin(angle)
    expected_normals = torch.stack((-torch.sign(world[:, 0]) * math.sin(angle),
                                    torch.zeros(8), torch.full((8,), math.cos(angle))), -1)
    torch.testing.assert_close(result["distances"][0], expected_depth, atol=3e-7, rtol=1e-5)
    active = result["contact_mask"][0]
    assert active[:4].all()
    torch.testing.assert_close(result["contacts"][0, active, 6:9], expected_normals[active], atol=3e-7, rtol=1e-5)
    expected_surface = world[active] - expected_depth[active, None] * expected_normals[active]
    torch.testing.assert_close(result["contacts"][0, active, 3:6] + state[0, :3], expected_surface, atol=3e-7, rtol=1e-5)


def test_input_layout_matches_nerd_explicit_contact_fields():
    state = _state()
    state[0, 7:] = torch.tensor([1., 2., 3., .4, .5, .6])
    action = torch.tensor([[2., 3., 4., 5., 6., 7.]])
    result = ContactFeatures(_flat_metadata())(state, action)
    vector = result["input"]
    assert vector.shape == (1, 71)
    torch.testing.assert_close(vector[:, :3], state[:, 7:10])
    torch.testing.assert_close(vector[:, 3:6], state[:, 10:13])
    torch.testing.assert_close(vector[:, 6:30], result["contacts"][:, :, 6:9].flatten(1))
    torch.testing.assert_close(vector[:, 30:54], result["contacts"][:, :, 3:6].flatten(1))
    torch.testing.assert_close(vector[:, 54:62], result["contacts"][:, :, 9])
    torch.testing.assert_close(vector[:, 62:68], action)
    torch.testing.assert_close(vector[:, 68:71], torch.tensor([[0., 0., -1.]]))


def test_contact_query_uses_supplied_state_not_live_simulator_contacts():
    sim = Simulator(SceneSpec(scenario="rest"))
    features = ContactFeatures(sim)
    state = _state(.25)
    action = torch.zeros(1, 6)
    near = features(state, action)
    sim.reset(_state(4.)[0].numpy())
    repeated = features(state, action)
    torch.testing.assert_close(repeated["input"], near["input"])
    assert near["contact_mask"].any()
    assert not features(_state(4.), action)["contact_mask"].any()


@pytest.mark.parametrize("obstacle", [False, True])
def test_body_frame_inputs_are_invariant_to_world_translation_and_yaw(obstacle):
    metadata = _obstacle_metadata() if obstacle else _flat_metadata()
    state = _state(.21, x=1.25, y=.13) if obstacle else _state(.25, x=.31, y=-.17)
    state[0, 7:] = torch.tensor([.7, -.4, .2, .3, .5, -.8])
    action = torch.tensor([[1., 2., 3., .3, -.4, .5]])
    original = ContactFeatures(metadata)(state, action)
    yaw = .71
    q = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
    translation = torch.tensor([2., -1., .7])
    transformed = deepcopy(metadata)
    for geom in transformed["geoms"]:
        if int(geom["body_id"]) == 0:
            geom["pos"] = (quat_rotate(q, torch.tensor(geom["pos"])) + translation).tolist()
            geom["quat"] = quat_multiply(q, torch.tensor(geom["quat"])).tolist()
    shifted = state.clone()
    shifted[:, :3] = quat_rotate(q, state[:, :3]) + translation
    shifted[:, 3:7] = quat_multiply(q[None], state[:, 3:7])
    shifted[:, 7:10] = quat_rotate(q, state[:, 7:10])
    rotated_action = torch.cat((quat_rotate(q, action[:, :3]), quat_rotate(q, action[:, 3:])), -1)
    actual = ContactFeatures(transformed)(shifted, rotated_action)
    torch.testing.assert_close(actual["input"], original["input"], atol=8e-7, rtol=1e-5)
    assert torch.equal(actual["contact_mask"], original["contact_mask"])


def test_static_collision_geometry_respects_nested_fixed_body_transforms():
    metadata = _flat_metadata()
    parent = deepcopy(metadata["bodies"][0])
    parent.update(id=2, parent=0, pos=[1., 2., .3],
                  quat=[math.sqrt(.5), 0., 0., math.sqrt(.5)])
    child = deepcopy(metadata["bodies"][0])
    child.update(id=3, parent=2, pos=[0., 0., .2],
                 quat=[math.sqrt(.5), 0., math.sqrt(.5), 0.])
    metadata["bodies"].extend((parent, child))
    floor = next(g for g in metadata["geoms"] if int(g["type"]) == 0)
    floor.update(body_id=3, pos=[0., 0., 0.],
                 quat=[math.sqrt(.5), 0., -math.sqrt(.5), 0.])
    # The nested transforms put the horizontal support at z=.5. Its yaw and
    # horizontal origin must not change any body-relative contact values.
    actual = ContactFeatures(metadata)(_state(.75, x=1.2, y=2.1), torch.zeros(1, 6))
    expected = ContactFeatures(_flat_metadata())(_state(.25), torch.zeros(1, 6))
    torch.testing.assert_close(actual["contacts"], expected["contacts"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual["distances"], expected["distances"], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("center_x", [.75, .8, 1.25])
def test_box_query_gradients_are_finite_inside_on_surface_and_outside(center_x):
    state = _state(.21, x=center_x, y=.13).requires_grad_()
    result = ContactFeatures(_obstacle_metadata())(state, torch.zeros(1, 6))
    result["contacts"].square().sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert state.grad[0, 0].abs() > 0


def test_quaternion_sign_does_not_change_contacts_or_predictions():
    model = build_model(Simulator(SceneSpec(scenario="rest")), "contact_mlp")
    state = _state(.25)
    state[:, 3:7] = torch.tensor([[.98, .1, -.08, .05]])
    state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
    opposite = state.clone()
    opposite[:, 3:7] *= -1
    action = torch.zeros(1, 6)
    torch.testing.assert_close(model.features(state, action)["input"], model.features(opposite, action)["input"])
    torch.testing.assert_close(model(state, action), model(opposite, action))


def test_factory_never_builds_or_samples_point_cloud_features(monkeypatch):
    import cube.models.dynamics as dynamics
    import cube.models.geometry as geometry

    def forbidden(*args, **kwargs):
        raise AssertionError("contact_mlp must not construct or sample a point cloud")

    monkeypatch.setattr(dynamics, "GeometryFeatures", forbidden)
    monkeypatch.setattr(geometry, "GeometryFeatures", forbidden)
    monkeypatch.setattr(geometry, "_primitive_patches", forbidden)
    sim = Simulator(SceneSpec(scenario="rest"))
    model = build_model(sim, {"name": "contact_mlp"})
    assert isinstance(model, ContactMLP)
    assert isinstance(model.features, ContactFeatures)
    assert model(_state(), torch.zeros(1, 6)).shape == (1, 12)
    assert [(layer.in_features, layer.out_features) for layer in model.modules()
            if isinstance(layer, nn.Linear)] == [(71, 512), (512, 256), (256, 128), (128, 64), (64, 12)]
    assert sum(isinstance(layer, nn.ReLU) for layer in model.modules()) == 4


def test_contacts_and_autonomous_rollout_keep_finite_nonzero_gradients():
    torch.manual_seed(83)
    model = build_model(Simulator(SceneSpec(scenario="rest")), "contact_mlp")
    state = _state(.24).requires_grad_()
    action = torch.tensor([[.1, -.2, .3, .2, .1, -.1]], requires_grad=True)
    features = model.features(state, action)
    contact_gradient, = torch.autograd.grad(features["contacts"].square().sum(), state, retain_graph=True)
    assert torch.isfinite(contact_gradient).all()
    assert contact_gradient[0, 2].abs() > 0
    next_state = apply_delta(state, model(state, action))
    following = apply_delta(next_state, model(next_state, action))
    following.square().mean().backward()
    assert torch.isfinite(following).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert action.grad.abs().sum() > 0
    gradients = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0


def test_scene_swap_updates_contacts_without_replacing_parameters():
    model = build_model(Simulator(SceneSpec(scenario="rest")), "contact_mlp")
    state = _state(.6, x=2.2)
    action = torch.zeros(1, 6)
    assert not model.features(state, action)["contact_mask"].any()
    parameters = [id(p) for p in model.parameters()]
    model.set_scene(Simulator(SceneSpec(scenario="toss_wall")))
    assert [id(p) for p in model.parameters()] == parameters
    assert model.features(state, action)["contact_mask"].any()
    assert not any(key.startswith("features.") for key in model.state_dict())


@pytest.mark.parametrize("kind", ["two_link", "ant"])
def test_unsupported_articulated_bodies_fail_explicitly(kind):
    with pytest.raises(ValueError):
        build_model(Simulator(SceneSpec(kind=kind, scenario="rest")), "contact_mlp")


def test_history_other_than_one_is_not_silently_ignored():
    with pytest.raises(ValueError, match="history"):
        build_model(Simulator(SceneSpec(scenario="rest")), {"name": "contact_mlp", "history": 2})


@pytest.mark.parametrize("dynamic", [False, True])
def test_unsupported_collision_primitive_fails_explicitly(dynamic):
    metadata = _flat_metadata()
    geom = next(g for g in metadata["geoms"] if (int(g["body_id"]) != 0) == dynamic)
    geom["type"] = 2  # Sphere is not the promised exact plane/box collision implementation.
    with pytest.raises(ValueError):
        ContactFeatures(metadata)
