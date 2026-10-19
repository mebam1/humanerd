"""Playback must preserve recorded states and expose, rather than hide, model drift."""
import os

import imageio.v2 as imageio
import numpy as np
import pytest

from cube.rendering import render_states, render_trajectory
from cube.sim import Simulator


def _trajectory():
    sim = Simulator({"scenario": "free_fall"})
    try:
        return np.asarray([sim.state()] + [sim.step() for _ in range(5)])
    finally:
        sim.close()


@pytest.mark.parametrize("kwargs,match", [({"width": 63}, "width"), ({"height": 0}, "height"),
                                         ({"stride": 0}, "stride"), ({"stride": 1.5}, "stride"),
                                         ({"camera": "unknown"}, "camera")])
def test_render_input_settings(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        render_states(_trajectory(), {}, tmp_path / "bad.mp4", **kwargs)


def test_render_state_guards(tmp_path):
    states = _trajectory()
    with pytest.raises(ValueError, match="nonempty"):
        render_states(states[:0], {}, tmp_path / "bad.mp4")
    with pytest.raises(ValueError, match="same shape"):
        render_states(states, {}, tmp_path / "bad.mp4", predicted=states[:2])
    invalid = states.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        render_states(invalid, {}, tmp_path / "bad.mp4")
    invalid = states.copy()
    invalid[0, 3:7] = 0
    with pytest.raises(ValueError, match="quaternions"):
        render_states(invalid, {}, tmp_path / "bad.mp4")
    with pytest.raises(ValueError, match="state_dim"):
        render_states(states, {"kind": "ant"}, tmp_path / "bad.mp4")


def test_renderer_initialization_failure_releases_simulator(tmp_path, monkeypatch):
    import cube.rendering as rendering
    states = _trajectory()
    closed = []
    def failed_renderer(*args, **kwargs):
        raise RuntimeError("No OpenGL context")
    monkeypatch.setattr(rendering.mujoco, "Renderer", failed_renderer)
    monkeypatch.setattr(Simulator, "close", lambda self: closed.append(self))
    with pytest.raises(RuntimeError, match="OpenGL"):
        render_states(states, {}, tmp_path / "bad.mp4")
    assert len(closed) == 1


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "osmesa", reason="requires headless OSMesa Docker runtime")
def test_actual_comparison_video_does_not_integrate(tmp_path, monkeypatch):
    states = _trajectory()
    predicted = states.copy()
    predicted[1:, 0] += 1.
    def no_step(*args, **kwargs):
        raise AssertionError("Playback must never integrate physics")
    monkeypatch.setattr(Simulator, "step", no_step)
    path = tmp_path / "comparison.mp4"
    result = render_states(states, {}, path, predicted=predicted, width=320, height=240, stride=2,
                           progress_config=False)
    assert result["frames"] == 3
    assert result["fps"] == 25
    assert result["camera"]["fit"] == "ground_truth"
    assert result["camera"]["shared_panels"] is True
    poster = imageio.imread(result["poster"])
    assert poster.shape == (240, 640, 3)
    with imageio.get_reader(path) as reader:
        frames = list(reader.iter_data())
    assert len(frames) == 3
    assert frames[0].shape == poster.shape
    np.testing.assert_array_equal(poster[70:200, :320], poster[70:200, 320:])
    assert not np.array_equal(frames[-1][70:200, :320], frames[-1][70:200, 320:])


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "osmesa", reason="requires headless OSMesa Docker runtime")
def test_floor_wall_camera_keeps_cube_visible():
    import mujoco
    from cube.rendering import _camera_for_states

    sim = Simulator({"scenario": "floor_wall"})
    try:
        states = np.asarray([sim.state()] + [sim.step() for _ in range(5)])
        camera, info = _camera_for_states(sim, states, 320, 240)
        assert info["wall_aware"]
        object_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "object")
        with mujoco.Renderer(sim.model, height=240, width=320) as renderer:
            renderer.enable_segmentation_rendering()
            for state in states:
                sim.reset(state)
                renderer.update_scene(sim.data, camera)
                segmentation = renderer.render()
                visible = ((segmentation[..., 0] == object_id)
                           & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))
                assert visible.sum() > .02 * 320 * 240
    finally:
        sim.close()


def test_legacy_wrapper_scene_spec_and_key(tmp_path, monkeypatch):
    import cube.rendering as rendering
    import json
    states = _trajectory()
    archive = tmp_path / "states.npz"
    np.savez(archive, predicted=states)
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps({"scene_spec": {"scenario": "free_fall"}}), encoding="utf-8")
    captured = {}
    def capture(value, spec, output, **kwargs):
        captured.update(states=value, scene=spec, output=output, **kwargs)
        return {"video": str(output)}
    monkeypatch.setattr(rendering, "render_states", capture)
    render_trajectory(archive, scene, tmp_path / "x.mp4", key="predicted")
    np.testing.assert_array_equal(captured["states"], states)
    assert captured["scene"] == {"scenario": "free_fall"}
    assert captured["label"] == "PREDICTED"
    with pytest.raises(ValueError, match="available keys"):
        render_trajectory(archive, scene, tmp_path / "x.mp4", key="missing")
