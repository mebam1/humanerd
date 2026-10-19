"""The OOD valley observation contains the exposed collision surface only."""
import math

import pytest
import torch

from cube.models import GeometryFeatures, metadata_from_scene
from cube.sim import SceneSpec, Simulator


@pytest.mark.parametrize("root_x", [-4., -.2, 0., .2, 4.])
def test_valley_surface_samples_stay_exposed_after_root_recentering(root_x):
    sim = Simulator(SceneSpec(scenario="toss_valley"))
    geometry = GeometryFeatures(sim, {"floor_points": 256, "floor_extent": 1.5, "points_per_face": 4})
    state = torch.tensor(sim.state(), dtype=torch.float32)[None]
    state[:, :3] = torch.tensor([root_x, 1.25, 2.])
    state[:, 3:7] = torch.tensor([1., 0., 0., 0.])
    state.requires_grad_()
    features = geometry(state, torch.zeros(1, sim.action_dim))
    angle = math.radians(10)
    for name, side in (("valley_left", -1), ("valley_right", 1)):
        geom = next(g for g in geometry.metadata["geoms"] if g["name"] == name)
        indices = geometry.point_geom == geom["id"]
        samples = features["world_points"][0, indices]
        assert len(samples) == 256
        assert len(torch.unique(samples.detach(), dim=0)) == 256
        assert (side * samples[:, 0] >= -1e-6).all()
        assert torch.allclose(samples[:, 2], samples[:, 0].abs() * math.tan(angle), atol=1e-6)
        expected_normal = samples.new_tensor([-side * math.sin(angle), 0., math.cos(angle)])
        assert torch.allclose(features["normals"][0, indices], expected_normal.expand_as(samples), atol=1e-6)
        assert torch.allclose(samples[:, 1].amin(), samples.new_tensor(-.25), atol=1e-6)
        assert torch.allclose(samples[:, 1].amax(), samples.new_tensor(2.75), atol=1e-6)
    features["world_points"].square().sum().backward()
    assert torch.isfinite(state.grad).all()
    assert state.grad[0, 1].abs() > 0


def test_raw_mujoco_metadata_retains_valley_names_for_exposed_sampling():
    sim = Simulator(SceneSpec(scenario="toss_valley"))
    metadata = metadata_from_scene(sim.model)
    assert {g["name"] for g in metadata["geoms"] if g["type"] == 0} == {"valley_left", "valley_right"}
    geometry = GeometryFeatures(metadata, {"floor_points": 16})
    static = geometry.local_points[~geometry.dynamic]
    assert torch.allclose(static[:, 2], static[:, 0].abs() * math.tan(math.radians(10)), atol=1e-6)


def test_flat_floor_keeps_full_root_centered_sampling_grid():
    sim = Simulator(SceneSpec(scenario="toss"))
    geometry = GeometryFeatures(sim, {"floor_points": 16, "floor_extent": 1.5})
    state = torch.tensor(sim.state(), dtype=torch.float32)[None]
    state[:, :3] = torch.tensor([-.4, .7, 1.2])
    features = geometry(state, torch.zeros(1, sim.action_dim))
    grid = torch.linspace(-1.5, 1.5, 4)
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    expected = torch.stack((x.flatten() - .4, y.flatten() + .7, torch.zeros(16)), -1)
    assert torch.allclose(features["world_points"][0, ~geometry.dynamic], expected, atol=1e-6)
