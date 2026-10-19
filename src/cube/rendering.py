"""Headless state playback and synchronized GT/model comparison, without stepping physics."""
from __future__ import annotations

from contextlib import ExitStack
import json
from numbers import Integral
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cube.experiments.tracking import progress
from cube.sim import Simulator


def _positive_integer(value, name, *, even=False):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1 or (even and value % 2):
        qualifier = "positive even integer" if even else "positive integer"
        raise ValueError(f"{name} must be a {qualifier}")
    return int(value)


def _states_array(states, name):
    array = np.asarray(states)
    if array.ndim != 2 or not array.shape[0] or array.shape[1] < 13:
        raise ValueError(f"{name} must be a nonempty [frames, state_dim] array with state_dim >= 13")
    if (not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array)
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must contain finite real numeric states")
    if np.any(np.linalg.norm(array[:, 3:7], axis=1) < 1e-12):
        raise ValueError(f"{name} root quaternions must be nonzero")
    return array


def _camera_for_states(sim, states, width, height):
    """Fit a sphere enclosing GT geometry, independent of predicted drift."""
    dynamic = np.flatnonzero(sim.model.geom_bodyid != 0)
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    radius = sim.model.geom_rbound[dynamic, None]
    for state in states:
        sim.reset(state)
        centers = sim.data.geom_xpos[dynamic]
        lower = np.minimum(lower, np.min(centers - radius, axis=0))
        upper = np.maximum(upper, np.max(centers + radius, axis=0))
    # Include the contact plane without fitting the entire infinite floor or wall.
    lower[2] = min(0., lower[2])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (lower + upper) * .5
    camera.azimuth = 130.
    camera.elevation = -24.
    # MuJoCo's azimuth describes the viewing direction toward lookat. A camera
    # on the far side of an opaque wall can hide every dynamic geom even when
    # the GT-fitted bounds are correct. View diagonally toward a named wall
    # from the object's side, while retaining the same GT fit for both panels.
    wall = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    wall_aware = False
    if wall >= 0:
        toward_wall = sim.data.geom_xpos[wall, :2] - camera.lookat[:2]
        if np.linalg.norm(toward_wall) > 1e-8:
            camera.azimuth = float(np.rad2deg(np.arctan2(toward_wall[1], toward_wall[0])) + 45.)
            wall_aware = True
    half_fov = np.deg2rad(float(sim.model.vis.global_.fovy)) / 2
    half_fov = min(half_fov, np.arctan(np.tan(half_fov) * width / height))
    # Leave room for panel captions; both panels retain the same scale and pose.
    camera.distance = max(.5, float(np.linalg.norm(upper - lower) * .5) / np.sin(half_fov) * 1.25)
    info = {"lookat": camera.lookat.tolist(), "distance": float(camera.distance),
            "azimuth": float(camera.azimuth), "elevation": float(camera.elevation),
            "gt_bounds": [lower.tolist(), upper.tolist()], "wall_aware": wall_aware}
    return camera, info


def _label_frame(pixels, title, timestamp, index, font, *, error=None):
    frame = Image.fromarray(pixels)
    draw = ImageDraw.Draw(frame)
    width, height = frame.size
    top = min(58, max(28, height // 8))
    bottom = min(32, max(18, height // 15))
    draw.rectangle((0, 0, width, top), fill=(15, 23, 42))
    draw.rectangle((0, height - bottom, width, height), fill=(15, 23, 42))
    accent = (103, 232, 249) if error is None else (253, 186, 116)
    draw.text((10, 5), title, font=font, fill=accent)
    draw.text((10, 7 + font.size), f"t = {timestamp:.3f} s   |   frame {index}", font=font, fill="white")
    footer = "Recorded state playback" if error is None else f"Root position error: {error:.4f} m"
    draw.text((10, height - bottom + 4), footer, font=font, fill="white")
    return np.asarray(frame)


def render_states(states, scene: dict, output, *, predicted=None, width=640, height=480,
                  stride=1, camera="fixed", progress_config=None, label="GROUND TRUTH"):
    """Stream an MP4 and PNG poster from recorded states.

    ``predicted`` adds a synchronized right-hand panel. Both panels use one
    camera fitted to GT only, so camera tracking cannot hide prediction drift.
    ``follow`` translates that common camera with the GT root; ``fixed`` keeps
    the complete GT path in view. Input state index determines time, including
    when stride skips frames. No Simulator.step or MuJoCo integration is called.
    """
    width = _positive_integer(width, "width", even=True)
    height = _positive_integer(height, "height", even=True)
    stride = _positive_integer(stride, "stride")
    if camera not in {"fixed", "follow"}:
        raise ValueError("camera must be 'fixed' or 'follow'")
    states = _states_array(states, "states")
    if predicted is not None:
        predicted = _states_array(predicted, "predicted")
        if predicted.shape != states.shape:
            raise ValueError("predicted and states must have exactly the same shape")
    out = Path(output)
    if out.suffix.lower() != ".mp4":
        raise ValueError("output must have an .mp4 extension")
    poster = out.with_suffix(".png")
    with ExitStack() as resources:
        sim = Simulator(scene)
        resources.callback(sim.close)
        expected = sim.model.nq + sim.model.nv
        if states.shape[1] != expected:
            raise ValueError(f"Scene requires state_dim={expected}; received {states.shape[1]}")
        fps = 1. / sim.dt / stride
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Scene timestep must produce a finite positive frame rate")
        view, camera_info = _camera_for_states(sim, states, width, height)
        camera_info.update({"mode": camera, "fit": "ground_truth", "shared_panels": predicted is not None})
        fixed_lookat = view.lookat.copy()
        sim.model.vis.global_.offwidth = width
        sim.model.vis.global_.offheight = height
        renderer = mujoco.Renderer(sim.model, height=height, width=width)
        resources.callback(renderer.close)
        font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        font = ImageFont.truetype(str(font_path), size=max(10, min(18, height // 32)))
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=7, macro_block_size=1)
        resources.callback(writer.close)
        frames = 0
        indices = progress(range(0, len(states), stride), config=progress_config, desc="Render", unit="frame")
        resources.callback(indices.close)
        for index in indices:
            if camera == "follow":
                view.lookat[:] = fixed_lookat + states[index, :3] - states[0, :3]
            sim.reset(states[index])
            renderer.update_scene(sim.data, view)
            pixels = _label_frame(renderer.render(), label, index * sim.dt, index, font)
            if predicted is not None:
                sim.reset(predicted[index])
                renderer.update_scene(sim.data, view)
                error = float(np.linalg.norm(predicted[index, :3] - states[index, :3]))
                right = _label_frame(renderer.render(), "MODEL ROLLOUT", index * sim.dt, index, font, error=error)
                pixels = np.concatenate((pixels, right), axis=1)
            writer.append_data(pixels)
            if frames == 0:
                imageio.imwrite(poster, pixels)
            frames += 1
    return {"video": str(out), "poster": str(poster), "frames": frames, "fps": fps,
            "width": width * (2 if predicted is not None else 1), "height": height, "camera": camera_info}


def render_trajectory(trajectory, scene, output, key="states", width=960, height=640, stride=1):
    """Backward-compatible NPZ/scene-JSON entry point for stored trajectories."""
    spec = json.loads(Path(scene).read_text(encoding="utf-8-sig"))
    spec = spec.get("scene", spec.get("scene_spec", spec))
    with np.load(trajectory, allow_pickle=False) as arrays:
        if key not in arrays:
            raise ValueError(f"Trajectory has no array '{key}'; available keys: {', '.join(arrays.files)}")
        states = arrays[key]
    label = "GROUND TRUTH" if key == "states" else key.replace("_", " ").upper()
    return render_states(states, spec, output, width=width, height=height, stride=stride, label=label)
