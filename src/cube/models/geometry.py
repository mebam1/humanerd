"""Current-state surface reconstruction with differentiable rigid-body kinematics."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .state import quat_conjugate, quat_multiply, quat_normalize, quat_rotate, rotvec_to_quat


def metadata_from_scene(scene: Any) -> dict:
    if isinstance(scene, dict):
        return scene
    if hasattr(scene, "kinematic_metadata"):
        return scene.kinematic_metadata()
    model = getattr(scene, "model", scene)
    if not hasattr(model, "body_parentid"):
        raise TypeError("Expected simulator, MuJoCo model, or kinematic_metadata dictionary")
    bodies = [{"id": i, "parent": int(model.body_parentid[i]),
               **{key: getattr(model, "body_" + key)[i].tolist()
                  for key in ("pos", "quat", "mass", "inertia", "ipos", "iquat")}}
              for i in range(model.nbody)]
    joints = [{"id": i, "body_id": int(model.jnt_bodyid[i]),
               **{key: getattr(model, "jnt_" + key)[i].tolist()
                  for key in ("pos", "axis", "qposadr", "dofadr", "type", "range")}}
              for i in range(model.njnt)]
    geoms = [{"id": i, "name": model.geom(i).name if hasattr(model, "geom") else None,
              "body_id": int(model.geom_bodyid[i]),
              **{key: getattr(model, "geom_" + key)[i].tolist()
                 for key in ("type", "pos", "quat", "size", "friction", "contype", "conaffinity", "solref", "solimp", "margin")}}
             for i in range(model.ngeom)]
    return {"bodies": bodies, "joints": joints, "geoms": geoms,
            "gravity": model.opt.gravity.tolist(), "dt": float(model.opt.timestep),
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
            "actuator_trnid": model.actuator_trnid.tolist(),
            "actuator_gear": model.actuator_gear.tolist(), "qpos0": model.qpos0.tolist()}


def _valley_side(geom: dict) -> int:
    """Only the named static valley planes have a buried half to exclude."""
    if int(geom["type"]) != 0 or int(geom["body_id"]) != 0:
        return 0
    return {"valley_left": -1, "valley_right": 1}.get(geom.get("name"), 0)


def _primitive_patches(kind: int, size: list, n: int, extent: float) -> list[tuple[Tensor, Tensor]]:
    """Physical box faces; six consistent parameterized patches for curved primitives."""
    grid = torch.linspace(-1, 1, n)
    a, b = torch.meshgrid(grid, grid, indexing="ij")
    a, b = a.flatten(), b.flatten()
    size = torch.tensor(size, dtype=torch.float32)
    patches = []
    if kind in (0, 6, 2, 4):
        faces = [(2, 1)] if kind == 0 else [(axis, sign) for axis in range(3) for sign in (-1, 1)]
        for axis, sign in faces:
            other = [i for i in range(3) if i != axis]
            p = torch.zeros(n*n, 3)
            p[:, axis] = sign
            p[:, other[0]], p[:, other[1]] = a, b
            normals = torch.zeros_like(p)
            normals[:, axis] = sign
            if kind == 0:
                p[:, 2] = 0
                p[:, :2] *= extent
            elif kind == 6:
                p *= size
            else:
                p = torch.nn.functional.normalize(p, dim=-1)
                radii = size if kind == 4 else size[0].repeat(3)
                normals = torch.nn.functional.normalize(p / radii.clamp_min(1e-6), dim=-1)
                p *= radii
            patches.append((p, normals))
        return patches
    if kind in (3, 5):  # capsule/cylinder z axis in compiled geom frame
        radius, length = float(size[0]), float(size[1])
        for quadrant in range(4):
            theta = (a + 1) * math.pi / 4 + quadrant * math.pi / 2
            normals = torch.stack((theta.cos(), theta.sin(), torch.zeros_like(theta)), -1)
            p = normals * radius
            p[:, 2] = b * length
            patches.append((p, normals))
        for sign in (-1, 1):
            theta = (a + 1) * math.pi
            radial = (b + 1) / 2
            if kind == 3:
                phi = radial * math.pi / 2
                normals = torch.stack((phi.sin()*theta.cos(), phi.sin()*theta.sin(), sign*phi.cos()), -1)
                p = normals * radius
            else:
                p = torch.stack((radial*theta.cos()*radius, radial*theta.sin()*radius, torch.zeros_like(a)), -1)
                normals = torch.zeros_like(p)
                normals[:, 2] = sign
            p[:, 2] += sign * length
            patches.append((p, normals))
        return patches
    raise ValueError(f"Surface sampler does not implement MuJoCo geom type {kind}; mesh/hfield need explicit sampling")


class GeometryFeatures(nn.Module):
    """No future contacts, solver impulses, teacher pose, or detached FK in this path."""
    point_dim = 21
    body_dim = 42

    def __init__(self, scene: Any, config: dict | None = None):
        super().__init__()
        self.metadata = metadata_from_scene(scene)
        self.config = config or {}
        m = self.metadata
        self.nq, self.nv, self.nu = m["nq"], m["nv"], m["nu"]
        free = [j for j in m["joints"] if int(j["type"]) == 0]
        if len(free) != 1 or free[0]["qposadr"] != 0 or any(int(j["type"]) != 3 for j in m["joints"] if j not in free):
            raise ValueError("Models currently require one free root followed by hinge joints")
        self.root_id = int(free[0]["body_id"])
        if self.nq != 7 + len(m["joints"]) - 1 or self.nv != self.nq - 1:
            raise ValueError("Unsupported generalized state layout")
        self.joints_by_body = {i: [] for i in range(len(m["bodies"]))}
        for j in m["joints"]:
            self.joints_by_body[int(j["body_id"])].append(j)
        self.dynamic_ids = []
        for body in m["bodies"]:
            if int(body["id"]) == self.root_id or int(body["parent"]) in self.dynamic_ids:
                self.dynamic_ids.append(int(body["id"]))
        self.dynamic_lookup = {body: i for i, body in enumerate(self.dynamic_ids)}
        points, normals, body_ids, material, patch_ids, plane_ids, geom_ids = [], [], [], [], [], [], []
        patch_body = []
        n = max(2, math.ceil(math.sqrt(self.config.get("points_per_face", 16))))
        floor_n = max(2, math.ceil(math.sqrt(self.config.get("floor_points", 64))))
        for geom in m["geoms"]:
            if int(geom.get("contype", 1)) == 0 and int(geom.get("conaffinity", 1)) == 0:
                continue
            gid, bid, kind = int(geom["id"]), int(geom["body_id"]), int(geom["type"])
            gq = torch.tensor(geom["quat"], dtype=torch.float32)
            gp = torch.tensor(geom["pos"], dtype=torch.float32)
            for p, normal in _primitive_patches(kind, geom["size"], floor_n if kind == 0 else n,
                                                float(self.config.get("floor_extent", 1.5))):
                side = _valley_side(geom)
                if side:
                    # Keep the complete grid density on the exposed half of
                    # each plane; its other half lies under the opposite slope.
                    extent = float(self.config.get("floor_extent", 1.5))
                    p[:, 0] = (p[:, 0] + side * extent) / 2
                count = p.shape[0]
                points.append(quat_rotate(gq, p) + gp)
                normals.append(quat_rotate(gq, normal))
                body_ids.extend([bid] * count)
                geom_ids.extend([gid] * count)
                material.extend([list(geom["friction"]) + list(geom.get("solref", [.02, 1.])) +
                                 list(geom.get("solimp", [.9, .95, .001, .5, 2.])) +
                                 [float(geom.get("margin", 0.))]] * count)
                plane_ids.extend([gid if kind == 0 else -1] * count)
                if bid in self.dynamic_lookup:
                    patch_ids.extend([len(patch_body)] * count)
                    patch_body.append(self.dynamic_lookup[bid])
                else:
                    patch_ids.extend([-1] * count)
        if not points or not patch_body:
            raise ValueError("Need at least one sampled dynamic geometry")
        self._buffer("local_points", torch.cat(points))
        self._buffer("local_normals", torch.cat(normals))
        self._buffer("point_body", torch.tensor(body_ids, dtype=torch.long))
        self._buffer("point_geom", torch.tensor(geom_ids, dtype=torch.long))
        self._buffer("material", torch.tensor(material, dtype=torch.float32))
        self._buffer("patch_ids", torch.tensor(patch_ids, dtype=torch.long))
        self._buffer("plane_ids", torch.tensor(plane_ids, dtype=torch.long))
        self._buffer("dynamic", torch.tensor([b in self.dynamic_lookup for b in body_ids]))
        self._buffer("patch_body", torch.tensor(patch_body, dtype=torch.long))
        self._buffer("gravity", torch.tensor(m["gravity"], dtype=torch.float32))
        self._buffer("body_pos", torch.tensor([b["pos"] for b in m["bodies"]], dtype=torch.float32))
        self._buffer("body_quat", torch.tensor([b["quat"] for b in m["bodies"]], dtype=torch.float32))
        self._buffer("body_com", torch.tensor([b["ipos"] for b in m["bodies"]], dtype=torch.float32))
        self._buffer("body_iquat", torch.tensor([b["iquat"] for b in m["bodies"]], dtype=torch.float32))
        self._buffer("body_inertia", torch.tensor([b["inertia"] for b in m["bodies"]], dtype=torch.float32))
        self._buffer("body_mass", torch.tensor([b["mass"] for b in m["bodies"]], dtype=torch.float32))
        self._buffer("qpos0", torch.tensor(m.get("qpos0", [0.] * self.nq), dtype=torch.float32))
        self.patch_count = len(patch_body)
        grouping = self.patch_ids.clone()
        if self.config.get("random_groups", False):
            dynamic_indices = torch.where(self.dynamic)[0]
            permutation = torch.randperm(len(dynamic_indices), generator=torch.Generator().manual_seed(2026))
            grouping[dynamic_indices] = grouping[dynamic_indices[permutation]]
        indices = [torch.where(grouping == i)[0] for i in range(self.patch_count)]
        maximum = max(len(i) for i in indices)
        patch_index = torch.zeros(self.patch_count, maximum, dtype=torch.long)
        patch_mask = torch.ones(self.patch_count, maximum, dtype=torch.bool)
        for i, index in enumerate(indices):
            patch_index[i, :len(index)] = index
            patch_mask[i, :len(index)] = False
        self._buffer("patch_index", patch_index)
        self._buffer("patch_mask", patch_mask)

    def _buffer(self, name: str, value: Tensor):
        # Scene buffers are reconstructed from metadata, never transferred as learned weights.
        self.register_buffer(name, value, persistent=False)

    def forward_kinematics(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = state.shape[0]
        zero = state.new_zeros(batch, 3)
        identity = state.new_tensor([1., 0., 0., 0.]).expand(batch, -1)
        positions, rotations, velocities, angular = [zero], [identity], [zero], [zero]
        for body in self.metadata["bodies"][1:]:
            bid, parent = int(body["id"]), int(body["parent"])
            if bid == self.root_id:
                p, q = state[:, :3], quat_normalize(state[:, 3:7])
                v, w = state[:, self.nq:self.nq+3], quat_rotate(q, state[:, self.nq+3:self.nq+6])
            else:
                pq, pp, pv, pw = rotations[parent], positions[parent], velocities[parent], angular[parent]
                offset = quat_rotate(pq, self.body_pos[bid].expand(batch, -1))
                p, q = pp + offset, quat_multiply(pq, self.body_quat[bid].expand(batch, -1))
                v, w = pv + torch.linalg.cross(pw, offset), pw
                for joint in self.joints_by_body[bid]:
                    if int(joint["type"]) != 3:
                        continue
                    jpos = state.new_tensor(joint["pos"]).expand(batch, -1)
                    jaxis = state.new_tensor(joint["axis"]).expand(batch, -1)
                    angle = state[:, int(joint["qposadr"])] - self.qpos0[int(joint["qposadr"])]
                    jrot = rotvec_to_quat(jaxis * angle[:, None])
                    pivot = p + quat_rotate(q, jpos)
                    axis = quat_rotate(q, jaxis)
                    q_next = quat_multiply(q, jrot)
                    p_next = pivot - quat_rotate(q_next, jpos)
                    dw = axis * state[:, self.nq + int(joint["dofadr"]), None]
                    v = v + torch.linalg.cross(w, p_next - p) + torch.linalg.cross(dw, p_next - pivot)
                    p, q, w = p_next, q_next, w + dw
            positions.append(p)
            rotations.append(q)
            velocities.append(v)
            angular.append(w)
        return tuple(torch.stack(values, 1) for values in (positions, rotations, velocities, angular))

    def forward(self, state: Tensor, action: Tensor) -> dict[str, Tensor]:
        if state.ndim != 2 or state.shape[-1] != self.nq + self.nv:
            raise ValueError("state must be [batch,nq+nv]")
        batch = state.shape[0]
        bp, bq, bv, bw = self.forward_kinematics(state)
        pb = self.point_body
        point_q = bq[:, pb]
        local = self.local_points.unsqueeze(0).expand(batch, -1, -1)
        offset = quat_rotate(point_q, local)
        world_points = bp[:, pb] + offset
        world_normals = quat_rotate(point_q, self.local_normals.unsqueeze(0).expand(batch, -1, -1))
        # A large finite wall needs samples near the robot, not only at distant
        # box corners. Localize tangential grids while keeping exact face planes
        # and clipping every sample to the physical box. Small supports retain
        # their complete original face grid.
        for geom in self.metadata["geoms"]:
            if int(geom["type"]) != 6 or int(geom["body_id"]) in self.dynamic_lookup:
                continue
            indices = torch.where(self.point_geom == int(geom["id"]))[0]
            if not len(indices):
                continue
            bid = int(geom["body_id"])
            gq = quat_multiply(bq[:, bid], state.new_tensor(geom["quat"]).expand(batch, -1))
            gp = bp[:, bid] + quat_rotate(bq[:, bid], state.new_tensor(geom["pos"]).expand(batch, -1))
            inverse = quat_conjugate(gq)
            root_local = quat_rotate(inverse, state[:, :3] - gp)
            box_local = quat_rotate(inverse[:, None], world_points[:, indices] - gp[:, None])
            box_normals = quat_rotate(inverse[:, None], world_normals[:, indices])
            size = state.new_tensor(geom["size"])
            radius = size.clamp_max(float(self.config.get("environment_box_extent", .5)))
            center = torch.maximum(torch.minimum(root_local, size - radius), -size + radius)
            candidate = center[:, None] + box_local / size.clamp_min(1e-8) * radius
            localized = torch.where(box_normals.abs() > .5, box_local, candidate)
            world_points = world_points.index_copy(1, indices, gp[:, None] + quat_rotate(gq[:, None], localized))
        # Resample each infinite plane around the current projected root position.
        # The projection stays in the graph, while plane geometry remains static.
        for geom in self.metadata["geoms"]:
            if int(geom["type"]) != 0:
                continue
            indices = torch.where(self.plane_ids == int(geom["id"]))[0]
            if len(indices) == 0:
                continue
            bid = int(geom["body_id"])
            gq = quat_multiply(bq[:, bid], state.new_tensor(geom["quat"]).expand(batch, -1))
            gp = bp[:, bid] + quat_rotate(bq[:, bid], state.new_tensor(geom["pos"]).expand(batch, -1))
            normal = quat_rotate(gq, state.new_tensor([0., 0., 1.]).expand(batch, -1))
            diff = state[:, :3] - gp
            translation = diff - (diff * normal).sum(-1, keepdim=True) * normal
            side = _valley_side(geom)
            if side:
                # A root-centred translation must not carry valley samples
                # through the seam onto the buried half. Slide the full grid
                # along the exposed half, stopping its nearest edge at x=0.
                local = quat_rotate(quat_conjugate(gq), translation)
                extent = float(self.config.get("floor_extent", 1.5))
                shift_x = side * (side * local[:, 0] - extent / 2).clamp_min(0)
                local = torch.stack((shift_x, local[:, 1], torch.zeros_like(shift_x)), -1)
                translation = quat_rotate(gq, local)
            delta = torch.zeros_like(world_points).index_add(1, indices, translation[:, None, :].expand(-1, len(indices), -1))
            world_points = world_points + delta
        world_velocity = bv[:, pb] + torch.linalg.cross(bw[:, pb], world_points - bp[:, pb])
        root_q = quat_normalize(state[:, 3:7])
        inv = quat_conjugate(root_q)
        positions = quat_rotate(inv[:, None], world_points - state[:, None, :3])
        normals = quat_rotate(inv[:, None], world_normals)
        velocity = quat_rotate(inv[:, None], world_velocity)
        if self.config.get("no_velocity", False):
            velocity = torch.zeros_like(velocity)
        static = (~self.dynamic).to(state.dtype)[None, :, None].expand(batch, -1, -1)
        point = torch.cat((positions, normals, velocity, self.material[None].expand(batch, -1, -1), static), -1)
        patch_points = positions[:, self.patch_index]
        patch_normals = normals[:, self.patch_index]
        valid = (~self.patch_mask).to(state.dtype)[None, :, :, None]
        denominator = valid.sum(2).clamp_min(1)
        centers = (patch_points * valid).sum(2) / denominator
        nrm = (patch_normals * valid).sum(2) / denominator
        extents = patch_points.masked_fill(~valid.bool(), -torch.inf).amax(2) - patch_points.masked_fill(~valid.bool(), torch.inf).amin(2)
        patch_geometry = torch.cat((centers, nrm, extents), -1)
        body_features = []
        for bid in self.dynamic_ids:
            local_q = quat_normalize(quat_multiply(inv, bq[:, bid]))
            local_q = torch.where(local_q[:, :1] < 0, -local_q, local_q)
            principal_q = quat_multiply(local_q, self.body_iquat[bid].expand(batch, -1))
            axes = torch.eye(3, device=state.device, dtype=state.dtype).expand(batch, -1, -1)
            matrix = quat_rotate(principal_q[:, None], axes).transpose(-1, -2)
            inertia = matrix @ torch.diag_embed(self.body_inertia[bid].expand(batch, -1)) @ matrix.transpose(-1, -2)
            mass = self.body_mass[bid].expand(batch, 1)
            if self.config.get("no_inertia", False):
                inertia, mass = torch.zeros_like(inertia), torch.zeros_like(mass)
            wrench = state.new_zeros(batch, 6)
            axis = state.new_zeros(batch, 3)
            joint_info = state.new_zeros(batch, 3)
            if bid == self.root_id and action.shape[-1] == 6 and self.nu == 0:
                wrench = torch.cat((quat_rotate(inv, action[:, :3]), quat_rotate(inv, action[:, 3:])), -1)
            for joint in self.joints_by_body[bid]:
                if int(joint["type"]) != 3:
                    continue
                jid = int(joint["id"])
                axis = quat_rotate(local_q, state.new_tensor(joint["axis"]).expand(batch, -1))
                torque = state.new_zeros(batch)
                mapping = self.metadata.get("actuator_trnid", [])
                gears = self.metadata.get("actuator_gear", [])
                for aid, transmission in enumerate(mapping):
                    if int(transmission[0]) == jid and aid < action.shape[-1]:
                        torque = torque + action[:, aid] * (float(gears[aid][0]) if gears else 1.)
                if not mapping and jid > 0 and jid-1 < action.shape[-1]:
                    torque = action[:, jid-1]
                joint_info = torch.stack((state[:, int(joint["qposadr"])], state[:, self.nq+int(joint["dofadr"])], torque), -1)
            com_local = quat_rotate(local_q, self.body_com[bid].expand(batch, -1))
            body_v = quat_rotate(inv, bv[:, bid])
            body_w = quat_rotate(inv, bw[:, bid])
            # no_velocity removes explicit point velocity only, matching the proposal ablation.
            body_features.append(torch.cat((quat_rotate(inv, bp[:, bid] - state[:, :3]), local_q,
                body_v, body_w, mass, inertia.flatten(1), com_local,
                quat_rotate(inv, self.gravity.expand(batch, -1)), wrench,
                state.new_full((batch, 1), float(self.metadata["dt"])), axis, joint_info), -1))
        return {"points": point, "positions": positions, "normals": normals, "velocities": velocity,
                "world_points": world_points, "dynamic": self.dynamic, "patch_geometry": patch_geometry,
                "body": torch.stack(body_features, 1), "body_world_positions": bp,
                "body_world_quaternions": bq, "body_world_velocities": bv, "body_world_angular": bw}
