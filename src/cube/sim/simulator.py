"""A common state/action interface for two independently executed solvers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import replace
import numpy as np
import mujoco

from .scenes import SceneSpec, scene_xml


class Simulator:
    """State = qpos(wxyz quaternion) + native MuJoCo qvel.

    Free-joint translation velocity is world-frame; angular velocity is root-frame.
    Rigid actions are world-frame forces/torques at root COM (N/Nm); articulated
    actions are normalized motor controls, multiplied by actuator gear (Nm).
    """

    def __init__(self, spec: SceneSpec | dict[str, Any] | None = None, backend: str = "mujoco", device: str = "cpu") -> None:
        self.spec = SceneSpec.from_dict(spec) if isinstance(spec, dict) else spec or SceneSpec()
        if backend not in {"mujoco", "warp"}:
            raise ValueError("backend must be mujoco or warp")
        self.backend, self.device = backend, device
        self.xml = scene_xml(self.spec)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.dt = self.spec.dt
        self.action_dim = int(self.model.nu) if self.model.nu else 6
        self.root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "root")
        self._warp_model = None
        self._warp_data = None
        self.last_substep_contacts: list[dict[str, Any]] = []
        self._initial_state = self._make_initial_state()
        self.reset()
        if backend == "warp":
            try:
                import warp as wp
                import mujoco_warp as mjw
            except ImportError as exc:
                raise RuntimeError("Warp backend requires warp-lang and mujoco-warp; use the Docker image.") from exc
            self._wp, self._mjw = wp, mjw
            wp.init()
            # Explicit device lookup raises on unavailable CUDA. Never substitute a solver/device.
            with wp.ScopedDevice(device):
                self._warp_model = mjw.put_model(self.model)
                self._warp_data = mjw.put_data(self.model, self.data, nworld=1, nconmax=128, njmax=512)

    def _make_initial_state(self) -> np.ndarray:
        rng = np.random.default_rng(self.spec.seed)
        q = self.model.qpos0.copy()
        v = np.zeros(self.model.nv)
        s = self.spec.scenario
        hx, hy, hz = self.spec.half_extents
        if self.spec.kind in {"cube", "cuboid"}:
            if s == "toss":
                q[:3] = rng.uniform(self.spec.toss_position_min, self.spec.toss_position_max)
                # Normalizing an isotropic 4D normal samples the unit 3-sphere;
                # its antipodal quaternion pairs induce Haar-uniform SO(3).
                quaternion = rng.normal(size=4)
                while np.linalg.norm(quaternion) < 1e-12:
                    quaternion = rng.normal(size=4)
                q[3:7] = quaternion / np.linalg.norm(quaternion)
                v[:3] = rng.uniform(self.spec.toss_linear_velocity_min, self.spec.toss_linear_velocity_max)
                v[3:6] = rng.uniform(self.spec.toss_angular_velocity_min, self.spec.toss_angular_velocity_max)
            elif s == "free_fall":
                q[2] = hz + rng.uniform(.45, .9)
                v[:3] = rng.uniform(-.15, .15, 3)
            elif s in {"impact", "bounce", "rebound"}:
                q[2] = hz + rng.uniform(.02, .06)
                v[2] = -rng.uniform(1., 2.)
            elif s == "settle":
                angle = rng.uniform(.1, .25)
                q[3:7] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
                q[2] = hz * np.cos(angle) + hx * np.sin(angle) + .03
                v[:3] = rng.uniform(-.2, .2, 3)
                v[3:] = rng.uniform(-.6, .6, 3)
            elif s in {"rest", "slide"}:
                q[2] = hz + .0001
                # Distinct episode seeds must produce distinct resting trajectories,
                # otherwise a trajectory-level split still leaks identical examples.
                q[:2] = rng.uniform(-.03, .03, 2)
                yaw = rng.uniform(-.12, .12)
                q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                if s == "slide":
                    v[0] = rng.uniform(.6, 1.4)
                    v[1] = rng.uniform(-.3, .3)
                else:
                    v[:2] = rng.uniform(-.004, .004, 2)
            elif s == "edge":
                angle = rng.uniform(.35, .65)
                q[3:7] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
                q[2] = hz * np.cos(angle) + hx * np.sin(angle) + .0001
                v[4] = rng.uniform(.5, 1.0)
            elif s in {"wall", "floor_wall", "wall_only"}:
                q[2] = hz + .0001 if s != "wall_only" else hz + .6
                q[0] = -.0001
                v[0] = rng.uniform(.5, 1.2)
                v[2] = -.1 if s != "wall_only" else 0
            elif s == "intervention":
                # No support-dependent RNG/state: all four counterfactuals start identical.
                q[:3] = [0, 0, hz + .0001]
                v[2] = -.4
                v[0] = .3 if self.spec.support_layout == "cross_face" else 0
                v[3:] = rng.uniform(-.15, .15, 3)
            else:
                q[2] = hz + .0001
        elif self.spec.kind == "two_link":
            q[2] = .45 if s not in {"rest", "slide"} else hz * .5 + .001
            q[7] = rng.uniform(-.35, .35)
            v[:3] = rng.uniform(-.1, .1, 3)
        elif self.spec.kind == "ant":
            q[2] = .34
            q[:2] = rng.uniform(-.02, .02, 2)
            yaw = rng.uniform(-.03, .03)
            q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            if s in {"free_fall", "impact"}:
                q[2] += .2
            joint_jitter = .02 if s in {"rest", "walk"} else .06
            q[7:] = rng.uniform(-joint_jitter, joint_jitter, self.model.nq - 7)
        return np.concatenate((q, v))

    def state(self) -> np.ndarray:
        return np.concatenate((self.data.qpos, self.data.qvel)).copy()

    def reset_episode(self, seed: int) -> np.ndarray:
        """Reset seeded episode randomness while reusing the compiled scene model."""
        self.spec = replace(self.spec, seed=int(seed))
        self._initial_state = self._make_initial_state()
        return self.reset()

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        state = self._initial_state if state is None else np.asarray(state, dtype=np.float64)
        if state.shape != (self.model.nq + self.model.nv,) or not np.isfinite(state).all():
            raise ValueError(f"Expected finite state of shape {(self.model.nq + self.model.nv,)}")
        norm = np.linalg.norm(state[3:7])
        if norm < 1e-12:
            raise ValueError("Root quaternion must be nonzero")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = state[:self.model.nq]
        self.data.qpos[3:7] /= norm
        self.data.qvel[:] = state[self.model.nq:]
        mujoco.mj_forward(self.model, self.data)
        self.data.qacc_warmstart[:] = 0
        self.last_substep_contacts = []
        if self._warp_model is not None:
            with self._wp.ScopedDevice(self.device):
                self._warp_data = self._mjw.put_data(self.model, self.data, nworld=1, nconmax=128, njmax=512)
        return self.state()

    def _kinematics_and_contacts(self) -> None:
        # Diagnostics use only current geometry/velocity. No native dynamics step
        # occurs in the Warp branch and no solver impulse is returned as a feature.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_comVel(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

    def step(self, action: np.ndarray | None = None, *, record_substeps: bool = False) -> np.ndarray:
        """Advance one control step, optionally retaining every physics-boundary contact.

        Substep snapshots are evaluation labels, never model inputs. Recording on
        Warp synchronizes device state at every substep and therefore costs more
        than throughput measurement with recording disabled.
        """
        action = np.zeros(self.action_dim) if action is None else np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"Expected finite action of shape {(self.action_dim,)}")
        self.data.xfrc_applied[:] = 0
        self.data.qfrc_applied[:] = 0
        if self.model.nu:
            self.data.ctrl[:] = np.clip(action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        else:
            self.data.xfrc_applied[self.root_body_id, :] = action
        self.last_substep_contacts = []
        if record_substeps:
            self.last_substep_contacts.append({"time": float(self.data.time), "state": self.state().tolist(), "contacts": self.contact_info()})
        if self.backend == "mujoco":
            for _ in range(self.spec.substeps):
                self.data.qacc_warmstart[:] = 0
                mujoco.mj_step(self.model, self.data)
                if record_substeps:
                    self.last_substep_contacts.append({"time": float(self.data.time), "state": self.state().tolist(), "contacts": self.contact_info()})
        else:
            with self._wp.ScopedDevice(self.device):
                if self.model.nu:
                    self._warp_data.ctrl.assign(self.data.ctrl.astype(np.float32).reshape(1, -1))
                self._warp_data.xfrc_applied.assign(self.data.xfrc_applied.astype(np.float32).reshape(1, self.model.nbody, 6))
                for _ in range(self.spec.substeps):
                    self._warp_data.qacc_warmstart.zero_()
                    self._mjw.step(self._warp_model, self._warp_data)
                    if record_substeps:
                        self.data.qpos[:] = self._warp_data.qpos.numpy()[0]
                        self.data.qvel[:] = self._warp_data.qvel.numpy()[0]
                        self.data.time += self.dt / self.spec.substeps
                        self.last_substep_contacts.append({"time": float(self.data.time), "state": self.state().tolist(), "contacts": self.contact_info()})
                # Only the integration state is needed. This avoids a host-side
                # contact-capacity copy and is compatible with both IO API versions.
                self.data.qpos[:] = self._warp_data.qpos.numpy()[0]
                self.data.qvel[:] = self._warp_data.qvel.numpy()[0]
                if not record_substeps:
                    self.data.time += self.dt
        self._kinematics_and_contacts()
        result = self.state()
        if not np.isfinite(result).all():
            raise FloatingPointError(f"Nonfinite {self.backend} state")
        return result

    def contact_info(self) -> list[dict[str, Any]]:
        self._kinematics_and_contacts()
        result = []
        for contact in self.data.contact[:self.data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            normal = np.asarray(contact.frame[:3]).copy()
            velocities = []
            for bid in (b1, b2):
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jp, jr, contact.pos, bid)
                velocities.append(jp @ self.data.qvel)
            result.append({
                "geom1": g1, "geom2": g2, "body1": b1, "body2": b2,
                "geom1_name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                "geom2_name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2),
                "distance": float(contact.dist), "position": contact.pos.tolist(),
                "inclusion_margin": float(contact.includemargin),
                "normal": normal.tolist(),
                "relative_normal_velocity": float(np.dot(velocities[1] - velocities[0], normal)),
            })
        return result

    def diagnostics(self) -> dict[str, float | int]:
        contacts = self.contact_info()
        distances = [c["distance"] for c in contacts]
        return {"contact_count": sum(c["distance"] <= c["inclusion_margin"] for c in contacts),
                "min_distance": min(distances, default=float("inf")),
                "penetration": max(0., -min(distances, default=0.))}

    def kinematic_metadata(self) -> dict[str, Any]:
        m = self.model
        bodies = [{"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i),
                   "parent": int(m.body_parentid[i]), "pos": m.body_pos[i].tolist(),
                   "quat": m.body_quat[i].tolist(), "mass": float(m.body_mass[i]),
                   "inertia": m.body_inertia[i].tolist(), "ipos": m.body_ipos[i].tolist(),
                   "iquat": m.body_iquat[i].tolist()} for i in range(m.nbody)]
        joints = [{"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i),
                   "body_id": int(m.jnt_bodyid[i]), "pos": m.jnt_pos[i].tolist(),
                   "axis": m.jnt_axis[i].tolist(), "qposadr": int(m.jnt_qposadr[i]),
                   "dofadr": int(m.jnt_dofadr[i]), "type": int(m.jnt_type[i]),
                   "range": m.jnt_range[i].tolist()} for i in range(m.njnt)]
        geoms = [{"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i),
                  "body_id": int(m.geom_bodyid[i]), "type": int(m.geom_type[i]),
                  "pos": m.geom_pos[i].tolist(), "quat": m.geom_quat[i].tolist(),
                  "size": m.geom_size[i].tolist(), "friction": m.geom_friction[i].tolist(),
                  "solref": m.geom_solref[i].tolist(), "solimp": m.geom_solimp[i].tolist(),
                  "margin": float(m.geom_margin[i]),
                  "contype": int(m.geom_contype[i]), "conaffinity": int(m.geom_conaffinity[i])}
                 for i in range(m.ngeom)]
        return {"bodies": bodies, "joints": joints, "geoms": geoms,
                "gravity": m.opt.gravity.tolist(), "dt": self.dt,
                "nq": m.nq, "nv": m.nv, "nu": m.nu,
                "qpos0": m.qpos0.tolist(), "root_body_id": self.root_body_id,
                "actuator_gear": m.actuator_gear.tolist(),
                "actuator_trnid": m.actuator_trnid.tolist(),
                "ctrlrange": m.actuator_ctrlrange.tolist()}

    def save_xml(self, path: str | Path) -> None:
        Path(path).write_text(self.xml, encoding="utf-8")

    def close(self) -> None:
        """Release references to device allocations; no renderer is allocated by default."""
        self._warp_data = None
        self._warp_model = None


def make_simulator(spec: SceneSpec | dict[str, Any] | None = None, backend: str = "mujoco", device: str = "cpu") -> Simulator:
    return Simulator(spec, backend=backend, device=device)
