"""Headless, state-based MuJoCo rendering; no re-simulation of the trajectory."""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from cube.sim import Simulator


def render_trajectory(trajectory, scene, output, key="states", width=960, height=640, stride=1):
    if stride < 1:
        raise ValueError("stride must be positive")
    spec = json.loads(Path(scene).read_text(encoding="utf-8"))
    if "scene" in spec:
        spec = spec["scene"]
    sim = Simulator(spec)
    states = np.load(trajectory, allow_pickle=False)[key]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sim.model.vis.global_.offwidth = width
    sim.model.vis.global_.offheight = height
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.4 if sim.spec.kind == "ant" else 1.8
    camera.azimuth = 130
    camera.elevation = -24
    fps = max(1, round(1 / sim.dt / stride))
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=7, macro_block_size=1)
    frames = 0
    try:
        for state in states[::stride]:
            sim.reset(state)
            camera.lookat[:] = state[:3]
            camera.lookat[2] = max(.15, camera.lookat[2] * .6)
            renderer.update_scene(sim.data, camera)
            pixels = renderer.render()
            writer.append_data(pixels)
            if frames == 0:
                imageio.imwrite(out.with_suffix(".png"), pixels)
            frames += 1
    finally:
        writer.close()
        renderer.close()
        sim.close()
    return {"video": str(out), "poster": str(out.with_suffix(".png")), "frames": frames, "fps": fps}
