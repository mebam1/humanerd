"""NeRD-style fixed contact slots and its current-step MLP baseline.

No surface point cloud is sampled here. Eight material corners query exact
static collision primitives at the supplied (including predicted) cube pose.
See documentation/05_models.md for sources and the Cube-specific adaptations.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .geometry import metadata_from_scene
from .state import quat_conjugate, quat_multiply, quat_normalize, quat_rotate


CONTACT_CONFIG = {
    "name": "contact_mlp", "history": 1, "hidden_dims": [512, 256, 128, 64], "activation": "relu",
    "contact_min_distance": .1, "contact_thickness_multiplier": 4.,
    "normalize_input": True, "normalize_output": True,
    "input_normalization_epsilon": 1e-5,
}


class ContactFeatures(nn.Module):
    """Body-frame (p0, p1, normal, signed distance), eight fixed box corners.

Each corner selects the environment primitive with minimum signed distance.
Positive distance means separation; negative means penetration. Selection is
piecewise differentiable, as is the contact-distance activation mask.
"""
    input_dim = 71
    contact_slots = 8

    def __init__(self, scene_or_metadata: Any, config: dict | None = None):
        super().__init__()
        self.config = {**CONTACT_CONFIG, **(config or {})}
        self.metadata = metadata_from_scene(scene_or_metadata)
        m = self.metadata
        self.nq, self.nv, self.nu = int(m["nq"]), int(m["nv"]), int(m["nu"])
        joints = m["joints"]
        if ((self.nq, self.nv, self.nu) != (7, 6, 0) or len(joints) != 1
                or int(joints[0]["type"]) != 0 or int(joints[0]["qposadr"]) != 0):
            raise ValueError("contact_mlp requires one unactuated free box (nq=7, nv=6, nu=0)")
        self.root_id = int(joints[0]["body_id"])
        bodies = {int(body["id"]): body for body in m["bodies"]}
        dynamic_ids = {self.root_id}
        for _ in bodies:
            dynamic_ids.update(bid for bid, body in bodies.items()
                               if int(body["parent"]) in dynamic_ids)
        colliders = [g for g in m["geoms"]
                     if int(g.get("contype", 1)) or int(g.get("conaffinity", 1))]
        dynamic = [g for g in colliders if int(g["body_id"]) in dynamic_ids]
        if len(dynamic) != 1 or int(dynamic[0]["type"]) != 6 or int(dynamic[0]["body_id"]) != self.root_id:
            raise ValueError("contact_mlp supports exactly one box collision geom on the free root")
        box = dynamic[0]
        size = torch.tensor(box["size"], dtype=torch.float32)
        if not torch.isfinite(size).all() or not (size > 0).all():
            raise ValueError("Box half extents must be finite and positive")
        signs = torch.tensor([[1. if i & (1 << axis) else -1. for axis in range(3)]
                              for i in range(8)])
        corners = quat_rotate(torch.tensor(box["quat"], dtype=torch.float32), signs * size) + torch.tensor(box["pos"], dtype=torch.float32)
        self.register_buffer("corners", corners, persistent=False)
        gravity = torch.tensor(m["gravity"], dtype=torch.float32)
        self.register_buffer("gravity_direction", torch.nn.functional.normalize(gravity, dim=0), persistent=False)

        minimum = float(self.config["contact_min_distance"])
        multiplier = float(self.config["contact_thickness_multiplier"])
        if not math.isfinite(minimum) or minimum < 0 or not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError("Contact threshold parameters must be finite and nonnegative")
        world_poses = {0: (torch.zeros(3), torch.tensor([1., 0., 0., 0.]))}

        def fixed_pose(bid: int) -> tuple[Tensor, Tensor]:
            if bid not in world_poses:
                body = bodies[bid]
                parent_p, parent_q = fixed_pose(int(body["parent"]))
                world_poses[bid] = (parent_p + quat_rotate(parent_q, torch.tensor(body["pos"], dtype=torch.float32)),
                                    quat_multiply(parent_q, torch.tensor(body["quat"], dtype=torch.float32)))
            return world_poses[bid]

        self.environment_kinds = []
        self.environment_ids = []
        positions, rotations, sizes, thresholds = [], [], [], []
        for geom in colliders:
            if int(geom["body_id"]) in dynamic_ids:
                continue
            # MuJoCo's collision bitmask rule; visual-only/incompatible geoms
            # must not create otherwise invisible contact inputs.
            if not ((int(box.get("contype", 1)) & int(geom.get("conaffinity", 1))) or
                    (int(geom.get("contype", 1)) & int(box.get("conaffinity", 1)))):
                continue
            kind = int(geom["type"])
            if kind not in (0, 6):
                raise ValueError("contact_mlp environment supports exact planes and boxes only")
            bp, bq = fixed_pose(int(geom["body_id"]))
            positions.append(bp + quat_rotate(bq, torch.tensor(geom["pos"], dtype=torch.float32)))
            rotations.append(quat_multiply(bq, torch.tensor(geom["quat"], dtype=torch.float32)))
            half = torch.tensor(geom["size"], dtype=torch.float32)
            if kind == 6 and (not torch.isfinite(half).all() or not (half > 0).all()):
                raise ValueError("Environment box half extents must be finite and positive")
            sizes.append(half)
            thresholds.append(max(minimum, multiplier * max(float(box.get("margin", 0.)),
                                                            float(geom.get("margin", 0.)))))
            self.environment_kinds.append(kind)
            self.environment_ids.append(int(geom["id"]))
        for name, values, width in (("environment_positions", positions, 3),
                                     ("environment_rotations", rotations, 4),
                                     ("environment_sizes", sizes, 3)):
            self.register_buffer(name, torch.stack(values) if values else torch.empty(0, width), persistent=False)
        self.register_buffer("thresholds", torch.tensor(thresholds), persistent=False)

    @staticmethod
    def _box_surface(point: Tensor, half: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        closest = torch.maximum(torch.minimum(point, half), -half)
        offset = point - closest
        outside_distance = torch.linalg.vector_norm(offset, dim=-1)
        outside_normal = offset / outside_distance[..., None].clamp_min(1e-12)
        clearance, axis = (half - point.abs()).min(-1)
        sign = torch.where(point.gather(-1, axis[..., None]) >= 0, 1., -1.)
        inside_normal = torch.zeros_like(point).scatter(-1, axis[..., None], sign)
        inside_surface = point + inside_normal * clearance[..., None]
        outside = (point.abs() > half).any(-1)
        return (torch.where(outside[..., None], closest, inside_surface),
                torch.where(outside[..., None], outside_normal, inside_normal),
                torch.where(outside, outside_distance, -clearance))

    def forward(self, state: Tensor, action: Tensor) -> dict[str, Tensor]:
        if state.ndim != 2 or state.shape[-1] != 13 or action.shape != (state.shape[0], 6):
            raise ValueError("contact_mlp expects state [B,13] and world wrench [B,6]")
        batch = state.shape[0]
        q = quat_normalize(state[:, 3:7])
        inverse = quat_conjugate(q)
        p0_body = self.corners[None].expand(batch, -1, -1)
        p0_world = quat_rotate(q[:, None], p0_body) + state[:, None, :3]
        projections, normals, distances = [], [], []
        for i, kind in enumerate(self.environment_kinds):
            gp, gq = self.environment_positions[i], self.environment_rotations[i]
            local = quat_rotate(quat_conjugate(gq), p0_world - gp)
            if kind == 0:
                d = local[..., 2]
                normal = torch.zeros_like(local)
                normal[..., 2] = 1.
                surface = local - d[..., None] * normal
            else:
                surface, normal, d = self._box_surface(local, self.environment_sizes[i])
            projections.append(quat_rotate(gq, surface) + gp)
            normals.append(quat_rotate(gq, normal))
            distances.append(d)
        if distances:
            distance, selected = torch.stack(distances, 2).min(-1)
            index = selected[..., None, None].expand(-1, -1, 1, 3)
            p1_world = torch.stack(projections, 2).gather(2, index).squeeze(2)
            normal_world = torch.stack(normals, 2).gather(2, index).squeeze(2)
            active = distance <= self.thresholds[selected]
            p1_body = quat_rotate(inverse[:, None], p1_world - state[:, None, :3])
            normal_body = quat_rotate(inverse[:, None], normal_world)
            raw = torch.cat((p0_body, p1_body, normal_body, distance[..., None]), -1)
            contacts = torch.where(active[..., None], raw, torch.zeros_like(raw))
        else:
            distance = state.new_full((batch, 8), float("inf"))
            selected = torch.full((batch, 8), -1, dtype=torch.long, device=state.device)
            active = torch.zeros((batch, 8), dtype=torch.bool, device=state.device)
            contacts = state.new_zeros(batch, 8, 10)
        # p0 is fixed in the body frame and is not an NN input in NeRD's code.
        # MuJoCo stores translational velocity in world coordinates, angular
        # velocity in body coordinates. Both wrench components arrive in world.
        vector = torch.cat((quat_rotate(inverse, state[:, 7:10]), state[:, 10:13],
                            contacts[:, :, 6:9].flatten(1), contacts[:, :, 3:6].flatten(1),
                            contacts[:, :, 9], quat_rotate(inverse, action[:, :3]),
                            quat_rotate(inverse, action[:, 3:]),
                            quat_rotate(inverse, self.gravity_direction[None].expand(batch, -1))), -1)
        return {"contacts": contacts, "contact_mask": active, "distances": distance,
                "selected_environment": selected, "input": vector}


class ContactMLP(nn.Module):
    """Current-step NeRD MLP ablation with explicit contact inputs, for Cube."""
    kind = "contact_mlp"
    nq, nv, nu = 7, 6, 0
    state_dim, action_dim, delta_dim, input_dim = 13, 6, 12, 71

    def __init__(self, scene_or_metadata: Any, config: dict | None = None):
        super().__init__()
        self.config = {**CONTACT_CONFIG, **(config or {}), "name": self.kind}
        if self.config["history"] != 1:
            raise ValueError("contact_mlp follows NeRD's MLP ablation: history must equal 1")
        if self.config["activation"] != "relu":
            raise ValueError("contact_mlp follows NeRD's MLP ablation: activation must be relu")
        hidden = self.config["hidden_dims"]
        if not hidden or any(not isinstance(n, int) or n <= 0 for n in hidden):
            raise ValueError("hidden_dims must contain positive integer widths")
        self.features = ContactFeatures(scene_or_metadata, self.config)
        self.metadata = self.features.metadata
        layers: list[nn.Module] = []
        width = self.input_dim
        for following in hidden:
            layers.extend((nn.Linear(width, following), nn.ReLU()))
            width = following
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Linear(width, self.delta_dim)
        for name, width in (("input", self.input_dim), ("output", self.delta_dim)):
            self.register_buffer(name + "_mean", torch.zeros(width))
            self.register_buffer(name + "_scale", torch.ones(width))

    def set_scene(self, scene_or_metadata: Any) -> "ContactMLP":
        reference = next(self.parameters())
        features = ContactFeatures(scene_or_metadata, self.config).to(reference)
        if not torch.allclose(features.corners, self.features.corners):
            raise ValueError("contact_mlp uses fixed robot geometry; construct a new model for another box")
        self.features, self.metadata = features, features.metadata
        return self

    def raw_input(self, state: Tensor, action: Tensor) -> Tensor:
        return self.features(state, action)["input"]

    @torch.no_grad()
    def _set_normalization(self, prefix: str, mean: Tensor, scale: Tensor) -> None:
        target_mean, target_scale = getattr(self, prefix + "_mean"), getattr(self, prefix + "_scale")
        mean = torch.as_tensor(mean, device=target_mean.device, dtype=target_mean.dtype)
        scale = torch.as_tensor(scale, device=target_scale.device, dtype=target_scale.dtype)
        if (mean.shape != target_mean.shape or scale.shape != target_scale.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all()):
            raise ValueError(f"Invalid {prefix} normalization shape or values")
        target_mean.copy_(mean)
        target_scale.copy_(scale)

    def set_input_normalization(self, mean: Tensor, scale: Tensor) -> None:
        self._set_normalization("input", mean, scale)

    def set_output_normalization(self, mean: Tensor, scale: Tensor) -> None:
        self._set_normalization("output", mean, scale)

    def encode_state(self, state: Tensor, action: Tensor, history: Any = None) -> Tensor:
        vector = self.raw_input(state, action)
        if self.config["normalize_input"]:
            vector = (vector - self.input_mean) / self.input_scale
        return self.encoder(vector)

    def forward(self, state: Tensor, action: Tensor, history: Any = None) -> Tensor:
        delta = self.decoder(self.encode_state(state, action, history))
        return delta * self.output_scale + self.output_mean if self.config["normalize_output"] else delta

    def freeze_transfer(self, mode: str = "none") -> "ContactMLP":
        if mode != "none":
            raise ValueError("contact_mlp has no point encoder or aggregation modules to freeze")
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return self

    @torch.no_grad()
    def contact_diagnostics(self, state: Tensor, action: Tensor) -> dict:
        features = self.features(state, action)
        distances = features["distances"]
        finite = distances[torch.isfinite(distances)]
        return {"slots_per_state": 8, "active_slots": features["contact_mask"].sum(-1).tolist(),
                "minimum_signed_distance": float(finite.min()) if finite.numel() else None,
                "source": "analytic_current_state_plane_box_queries"}
