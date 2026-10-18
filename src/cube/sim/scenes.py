"""Reproducible MJCF scenes shared by native MuJoCo and MuJoCo Warp."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


SCENARIOS = (
    "free_fall", "impact", "bounce", "rebound", "settle", "rest", "slide",
    "edge", "floor_wall", "wall_only", "wall", "intervention", "walk",
)


@dataclass(frozen=True)
class SceneSpec:
    kind: str = "cube"
    scenario: str = "free_fall"
    seed: int = 0
    dt: float = 0.02
    substeps: int = 4
    half_extents: tuple[float, float, float] = (0.2, 0.2, 0.2)
    mass: float = 1.0
    friction: float = 0.8
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    support: str = "AB"
    support_layout: str = "same_face"
    contact_timeconst: float = 0.012
    contact_damping: float | None = None
    contact_margin: float = 0.001
    solver_iterations: int = 50
    actuator_gear: float = 20.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "half_extents", tuple(self.half_extents))
        object.__setattr__(self, "gravity", tuple(self.gravity))
        if self.kind not in {"cube", "cuboid", "two_link", "ant"}:
            raise ValueError(f"Unknown kind: {self.kind}")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        if self.dt <= 0 or self.substeps < 1 or int(self.substeps) != self.substeps:
            raise ValueError("dt and integer substeps must be positive")
        if len(self.half_extents) != 3 or min(self.half_extents) <= 0 or self.mass <= 0:
            raise ValueError("Three half extents and mass must be positive")
        if len(self.gravity) != 3 or self.friction < 0:
            raise ValueError("gravity must have three components; friction must be nonnegative")
        if self.support not in {"none", "empty", "A", "B", "AB"}:
            raise ValueError("support must be none, A, B, or AB")
        if self.support_layout not in {"same_face", "cross_face"}:
            raise ValueError("support_layout must be same_face or cross_face")
        if self.contact_timeconst <= 0 or (self.contact_damping is not None and self.contact_damping <= 0):
            raise ValueError("Contact solver parameters must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SceneSpec":
        return cls(**value)


def _vec(values) -> str:
    return " ".join(f"{float(x):.10g}" for x in values)


def _environment(spec: SceneSpec) -> str:
    hx, hy, hz = spec.half_extents
    if spec.scenario == "intervention":
        geoms = []
        # The same support dimensions and pose are retained in all four worlds.
        # No floor is present: only the intervention supports can exert contact.
        if "A" in spec.support:
            geoms.append(f'<geom name="support_A" type="box" pos="{-0.65 * hx} 0 -0.05" size="{0.15 * hx} {0.7 * hy} .05" rgba=".25 .65 .35 1"/>')
        if "B" in spec.support:
            if spec.support_layout == "same_face":
                geoms.append(f'<geom name="support_B" type="box" pos="{0.65 * hx} 0 -0.05" size="{0.15 * hx} {0.7 * hy} .05" rgba=".65 .35 .25 1"/>')
            else:
                geoms.append(f'<geom name="support_B" type="box" pos="{hx + 0.05} 0 {hz}" size=".05 {0.7 * hy} {0.7 * hz}" rgba=".65 .35 .25 1"/>')
        return "\n".join(geoms)
    result = []
    if spec.scenario != "wall_only":
        result.append('<geom name="floor" type="plane" size="5 5 .1" rgba=".65 .70 .75 1"/>')
    if spec.scenario in {"floor_wall", "wall", "wall_only"}:
        result.append(f'<geom name="wall" type="box" pos="{hx + 0.05} 0 1.5" size=".05 3 1.5" rgba=".45 .5 .6 1"/>')
    return "\n".join(result)


def _rigid(spec: SceneSpec) -> tuple[str, str]:
    return (
        f'<body name="root" pos="0 0 1"><freejoint name="root_free"/>'
        f'<geom name="object" type="box" size="{_vec(spec.half_extents)}" mass="{spec.mass}" rgba=".2 .55 .9 1"/></body>',
        "",
    )


def _two_link(spec: SceneSpec) -> tuple[str, str]:
    hx, hy, hz = spec.half_extents
    body = f'''<body name="root" pos="0 0 .65">
      <freejoint name="root_free"/>
      <geom name="base_box" type="box" size="{hx} {hy * .55} {hz * .5}" mass="{spec.mass * .5}" rgba=".2 .55 .9 1"/>
      <body name="link" pos="{hx} 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" limited="true" range="-1.5 1.5" damping=".08" armature=".005"/>
        <geom name="link_box" type="box" pos="{hx} 0 0" size="{hx} {hy * .4} {hz * .4}" mass="{spec.mass * .5}" rgba=".9 .55 .2 1"/>
      </body>
    </body>'''
    return body, f'<motor name="hinge_motor" joint="hinge" gear="{spec.actuator_gear}" ctrllimited="true" ctrlrange="-1 1"/>'


ANT_LEGS = (("front_left", 1, 1), ("front_right", 1, -1), ("back_left", -1, 1), ("back_right", -1, -1))


def _ant(spec: SceneSpec) -> tuple[str, str]:
    # Original box/capsule morphology: four legs, each with yaw hip + pitch knee.
    bodies, actuators = [], []
    for name, sx, sy in ANT_LEGS:
        axis = (-sy / math.sqrt(2), sx / math.sqrt(2), 0)
        upper = (sx * .16, sy * .16, -.035)
        lower = (sx * .12, sy * .12, -.265)
        bodies.append(f'''<body name="{name}_upper" pos="{sx * .18} {sy * .105} 0">
          <joint name="{name}_hip" axis="0 0 1" range="-.65 .65"/>
          <geom name="{name}_upper_geom" type="capsule" fromto="0 0 0 {_vec(upper)}" size=".035" mass="{spec.mass * .065}"/>
          <body name="{name}_lower" pos="{_vec(upper)}">
            <joint name="{name}_knee" axis="{_vec(axis)}" range="-.7 .8"/>
            <geom name="{name}_lower_geom" type="capsule" fromto="0 0 0 {_vec(lower)}" size=".03" mass="{spec.mass * .06}"/>
          </body>
        </body>''')
        for joint in ("hip", "knee"):
            actuators.append(f'<motor name="{name}_{joint}_motor" joint="{name}_{joint}" gear="{spec.actuator_gear}" ctrllimited="true" ctrlrange="-1 1"/>')
    return (
        f'<body name="root" pos="0 0 .34"><freejoint name="root_free"/>'
        f'<geom name="torso" type="box" size=".22 .12 .075" mass="{spec.mass * .5}" rgba=".2 .55 .9 1"/>'
        + "\n".join(bodies) + "</body>",
        "\n".join(actuators),
    )


def scene_xml(spec: SceneSpec) -> str:
    """Return complete MJCF. The control timestep is dt; physics uses dt/substeps."""
    body, actuator = (_ant(spec) if spec.kind == "ant" else _two_link(spec) if spec.kind == "two_link" else _rigid(spec))
    damping = spec.contact_damping
    if damping is None:
        damping = .25 if spec.scenario in {"bounce", "rebound"} else 1.0
    return f'''<mujoco model="cube_contact_{spec.kind}">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="{spec.dt / spec.substeps}" gravity="{_vec(spec.gravity)}" integrator="Euler" solver="Newton" cone="pyramidal" jacobian="dense" iterations="{spec.solver_iterations}" tolerance="1e-8">
        <flag warmstart="disable"/>
      </option>
      <size njmax="512" nconmax="128"/>
      <default>
        <geom friction="{spec.friction} .005 .0001" condim="3" margin="{spec.contact_margin}" solref="{spec.contact_timeconst} {damping}" solimp=".95 .99 .001"/>
        <joint type="hinge" limited="true" damping=".15" armature=".015"/>
      </default>
      <worldbody>
        <light pos="0 -3 5" dir="0 .5 -1"/>
        <camera name="overview" pos="2.4 -3.4 2" xyaxes=".817 .577 0 -.244 .346 .906"/>
        {_environment(spec)}
        {body}
      </worldbody>
      <actuator>{actuator}</actuator>
    </mujoco>'''
