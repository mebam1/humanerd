"""Surface hierarchy, controlled ablations, and honestly named research baselines."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .geometry import GeometryFeatures, metadata_from_scene
from .serialization import local_pair_coverage, serialization_order


def mlp(input_dim: int, hidden: int, output: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, output))


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        normalized = self.norm(x)
        x = x + self.attention(normalized, normalized, normalized, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(x)


class RelativePointBlock(nn.Module):
    """Window attention with learnable relative position/velocity/normal bias."""
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.bias = mlp(7, heads*2, heads)
        self.output = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

    def forward(self, x: Tensor, physical: Tensor, window: int, shift: int) -> Tensor:
        batch, count, width = x.shape
        pad_right = (-(count + shift)) % window
        normalized = self.norm(x)
        padded = F.pad(normalized, (0, 0, shift, pad_right))
        raw = F.pad(physical, (0, 0, shift, pad_right))
        groups = padded.shape[1] // window
        chunks = padded.reshape(batch * groups, window, width)
        raw = raw.reshape(batch * groups, window, physical.shape[-1])
        qkv = self.qkv(chunks).reshape(batch*groups, window, 3, self.heads, width//self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        rel_position = raw[:, None, :, :3] - raw[:, :, None, :3]
        rel_velocity = raw[:, None, :, 6:9] - raw[:, :, None, 6:9]
        normal_dot = (raw[:, None, :, 3:6] * raw[:, :, None, 3:6]).sum(-1, keepdim=True)
        bias = self.bias(torch.cat((rel_position, rel_velocity, normal_dot), -1)).permute(0, 3, 1, 2)
        valid = F.pad(torch.ones(batch, count, device=x.device, dtype=torch.bool), (shift, pad_right)).reshape(batch*groups, window)
        bias = bias.masked_fill(~valid[:, None, None, :], -torch.inf)
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.)
        value = value.transpose(1, 2).reshape(batch, groups*window, width)[:, shift:shift+count]
        x = x + self.output(value)
        return x + self.ff(x)


class PointEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        width, heads = config["hidden_dim"], config["heads"]
        self.input = mlp(GeometryFeatures.point_dim, width, width)
        self.blocks = nn.ModuleList([RelativePointBlock(width, heads) for _ in range(config["local_layers"])])

    def forward(self, physical: Tensor, flat: bool = False) -> Tensor:
        order = serialization_order(physical[..., :3], self.config["serialization"])
        inverse = torch.argsort(order, dim=-1)
        raw = physical.gather(1, order[..., None].expand(-1, -1, physical.shape[-1]))
        x = self.input(raw)
        window = raw.shape[1] if flat else int(self.config["window_size"])
        for i, block in enumerate(self.blocks):
            x = block(x, raw, window, 0 if flat else (i % 2) * (window // 2))
        return x.gather(1, inverse[..., None].expand(-1, -1, x.shape[-1]))


class Hierarchy(nn.Module):
    """Point->patch->link->articulation parameters independent of geometry count."""
    def __init__(self, config: dict):
        super().__init__()
        width, heads = config["hidden_dim"], config["heads"]
        self.config = config
        self.patch_query = mlp(9, width, width)
        self.query_offsets = nn.Parameter(torch.randn(config["face_tokens"], width) / math.sqrt(width))
        self.face_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.body_input = mlp(GeometryFeatures.body_dim, width, width)
        self.link_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.articulation_query = nn.Parameter(torch.zeros(1, 1, width))
        self.global_blocks = nn.ModuleList([AttentionBlock(width, heads) for _ in range(config["global_layers"])])
        # Learned prior distinguishing self, parent-child, other links, global token.
        self.structural_bias = nn.Parameter(torch.zeros(4))
        self.pooled_readout = mlp(width, width, width)
        self.base_decoder = mlp(width, width, 12)
        self.joint_decoder = mlp(width * 2, width, 2)

    def patch_pool(self, h: Tensor, features: dict, geometry: GeometryFeatures) -> Tensor:
        batch, _, width = h.shape
        count = geometry.patch_count
        points = h[:, geometry.patch_index].reshape(batch*count, -1, width)
        mask = geometry.patch_mask[None].expand(batch, -1, -1).reshape(batch*count, -1)
        query = self.patch_query(features["patch_geometry"])[..., None, :] + self.query_offsets[None, None]
        if self.config["mean_pool"]:
            valid = (~mask).to(h.dtype)[..., None]
            pooled = (points * valid).sum(1) / valid.sum(1).clamp_min(1)
            return pooled.reshape(batch, count, 1, width).expand(-1, -1, self.config["face_tokens"], -1) + query
        result = self.face_attention(query.reshape(batch*count, -1, width), points, points,
                                     key_padding_mask=mask, need_weights=False)[0]
        return result.reshape(batch, count, self.config["face_tokens"], width) + query

    def forward(self, h: Tensor, features: dict, geometry: GeometryFeatures,
                kind: str) -> tuple[Tensor, Tensor]:
        body = self.body_input(features["body"])
        if kind in ("deepsets", "flat"):
            # Pool only dynamic points after local environment interaction.
            pooled = self.pooled_readout(h[:, geometry.dynamic].mean(1))
            return pooled + body.mean(1), body + pooled[:, None]
        faces = self.patch_pool(h, features, geometry)
        links = []
        for i in range(len(geometry.dynamic_ids)):
            index = torch.where(geometry.patch_body == i)[0]
            if len(index):
                local_faces = faces[:, index].flatten(1, 2)
                if not self.config["no_face_exchange"]:
                    # Exchange among faces belonging to the same rigid link first.
                    local_faces = self.global_blocks[0](local_faces) if self.global_blocks else local_faces
                query = body[:, i:i+1]
                links.append(query + self.link_attention(query, local_faces, local_faces, need_weights=False)[0])
            else:
                links.append(body[:, i:i+1])
        links = torch.cat(links, 1)
        art = self.articulation_query.expand(h.shape[0], -1, -1) + body.mean(1, keepdim=True)
        tokens = torch.cat((art, links), 1)
        count = len(geometry.dynamic_ids)
        relations = torch.full((count+1, count+1), 2, device=h.device, dtype=torch.long)
        relations[0, :] = relations[:, 0] = 3
        relations.fill_diagonal_(0)
        for i, bid in enumerate(geometry.dynamic_ids):
            parent = int(geometry.metadata["bodies"][bid]["parent"])
            if parent in geometry.dynamic_lookup:
                j = geometry.dynamic_lookup[parent]
                relations[i+1, j+1] = relations[j+1, i+1] = 1
        bias = self.structural_bias[relations]
        for block in self.global_blocks:
            tokens = block(tokens, bias)
        return tokens[:, 0], tokens[:, 1:]

    def decode(self, latent: Tensor, links: Tensor, geometry: GeometryFeatures) -> Tensor:
        base = self.base_decoder(latent)
        joints = sorted((j for j in geometry.metadata["joints"] if int(j["type"]) == 3), key=lambda j: j["qposadr"])
        if not joints:
            return base
        values = []
        for joint in joints:
            link = links[:, geometry.dynamic_lookup[int(joint["body_id"])]]
            values.append(self.joint_decoder(torch.cat((latent, link), -1)))
        values = torch.stack(values, 1)
        return torch.cat((base, values[..., 0], values[..., 1]), -1)


DEFAULT_CONFIG = {"name": "hierarchical", "hidden_dim": 64, "heads": 4,
                  "local_layers": 2, "global_layers": 2, "window_size": 32,
                  "points_per_face": 16, "floor_points": 64, "floor_extent": 1.5,
                  "face_tokens": 1, "serialization": "hilbert", "history": 1,
                  "no_velocity": False, "no_inertia": False, "mean_pool": False,
                  "random_groups": False, "no_face_exchange": False}


class DynamicsModel(nn.Module):
    def __init__(self, scene_or_metadata: Any, config: dict | None = None):
        super().__init__()
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        aliases = {"proposed": "hierarchical", "point_face_body": "hierarchical", "mlp": "state_mlp",
                   "flat_transformer": "flat", "independent_additive": "independent"}
        self.config["name"] = aliases.get(self.config["name"], self.config["name"])
        self.kind = self.config["name"]
        if self.kind not in ("hierarchical", "state_mlp", "nerd_like", "deepsets", "flat", "independent"):
            raise ValueError(f"Unknown model {self.kind}")
        if self.config["hidden_dim"] % self.config["heads"]:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.config["face_tokens"] not in (1, 2, 4):
            raise ValueError("face_tokens must be 1, 2, or 4")
        if self.config["history"] < 1 or self.config["window_size"] < 1:
            raise ValueError("history and window_size must be positive")
        self.metadata = metadata_from_scene(scene_or_metadata)
        self.features = GeometryFeatures(self.metadata, self.config)
        self.nq, self.nv, self.nu = self.features.nq, self.features.nv, self.features.nu
        self.action_dim = self.nu or 6
        self.state_dim, self.delta_dim = self.nq + self.nv, 2 * self.nv
        width = self.config["hidden_dim"]
        self.point_encoder = PointEncoder(self.config) if self.kind in ("hierarchical", "deepsets", "flat") else nn.Identity()
        self.aggregation = Hierarchy(self.config) if self.kind in ("hierarchical", "deepsets", "flat") else nn.Identity()
        context_dim = self.state_dim + self.action_dim + GeometryFeatures.body_dim + 2 * GeometryFeatures.point_dim
        if self.kind in ("state_mlp", "nerd_like", "independent"):
            self.state_encoder = mlp(context_dim, width, width)
        if self.kind in ("state_mlp", "nerd_like"):
            self.state_decoder = mlp(width, width, self.delta_dim)
        if self.kind == "nerd_like":
            self.contact_encoder = mlp(2 * GeometryFeatures.point_dim, width, width)
            self.time_embedding = nn.Parameter(torch.randn(self.config["history"], width) / math.sqrt(width))
            self.history_blocks = nn.ModuleList([AttentionBlock(width, self.config["heads"]) for _ in range(self.config["global_layers"])])
        if self.kind == "independent":
            self.local_response = mlp(2 * GeometryFeatures.point_dim + GeometryFeatures.body_dim, width, self.delta_dim)
            self.free_response = mlp(self.state_dim + self.action_dim + GeometryFeatures.body_dim, width, self.delta_dim)

    def set_scene(self, scene_or_metadata: Any) -> "DynamicsModel":
        """Swap static context without replacing learned weights or optimizer parameters."""
        metadata = metadata_from_scene(scene_or_metadata)
        if (metadata["nq"], metadata["nv"], metadata["nu"]) != (self.nq, self.nv, self.nu):
            raise ValueError("set_scene requires same DOFs; construct a target model for morphology transfer")
        reference = next(self.parameters())
        self.metadata = metadata
        self.features = GeometryFeatures(metadata, self.config).to(device=reference.device, dtype=reference.dtype)
        return self

    def _context(self, state: Tensor, action: Tensor, features: dict) -> Tensor:
        static = features["points"][:, ~self.features.dynamic]
        if static.shape[1]:
            static_summary = torch.cat((static.mean(1), static.amax(1)), -1)
        else:
            static_summary = state.new_zeros(state.shape[0], 2 * GeometryFeatures.point_dim)
        return torch.cat((state, action, features["body"].mean(1), static_summary), -1)

    def _nearest_contact(self, features: dict) -> Tensor:
        dynamic = features["points"][:, self.features.dynamic]
        static = features["points"][:, ~self.features.dynamic]
        if not static.shape[1]:
            empty = torch.zeros_like(dynamic)
            empty[..., 9] = 10.
            return empty
        distances = torch.cdist(dynamic[..., :3], static[..., :3])
        distance, nearest = distances.min(-1)
        other = static.gather(1, nearest[..., None].expand(-1, -1, static.shape[-1]))
        return torch.cat((other[..., :3] - dynamic[..., :3], other[..., 3:6],
                          other[..., 6:9] - dynamic[..., 6:9], distance[..., None], other[..., 9:-1]), -1)

    def _encode(self, state: Tensor, action: Tensor, history: Tensor | dict | None = None) -> tuple[Tensor, Tensor, dict]:
        features = self.features(state, action)
        if self.kind in ("state_mlp", "independent"):
            latent = self.state_encoder(self._context(state, action, features))
            return latent, latent[:, None].expand(-1, len(self.features.dynamic_ids), -1), features
        if self.kind == "nerd_like":
            if isinstance(history, dict):
                states, actions = history["states"], history.get("actions")
            else:
                states, actions = history, None
            if states is None:
                states = state[:, None]
            # Inclusive-current history: replace its last slot with the predicted state
            # so rollout never uses a stale teacher pose and history=10 stays ten steps.
            states = torch.cat((states[:, :-1], state[:, None]), 1)[:, -self.config["history"]:]
            if actions is None:
                actions = action[:, None].expand(-1, states.shape[1], -1)
            else:
                actions = torch.cat((actions[:, :-1], action[:, None]), 1)[:, -states.shape[1]:]
            if states.shape[1] < self.config["history"]:
                padding = self.config["history"] - states.shape[1]
                states = torch.cat((states[:, :1].expand(-1, padding, -1), states), 1)
                actions = torch.cat((actions[:, :1].expand(-1, padding, -1), actions), 1)
            tokens = []
            for i in range(states.shape[1]):
                current = features if i == states.shape[1]-1 else self.features(states[:, i], actions[:, i])
                contact = self._nearest_contact(current)
                summary = torch.cat((contact.mean(1), contact.amax(1)), -1)
                tokens.append(self.state_encoder(self._context(states[:, i], actions[:, i], current)) + self.contact_encoder(summary))
            x = torch.stack(tokens, 1) + self.time_embedding[None]
            for block in self.history_blocks:
                x = block(x)
            latent = x[:, -1]
            return latent, latent[:, None].expand(-1, len(self.features.dynamic_ids), -1), features
        h = self.point_encoder(features["points"], flat=self.kind == "flat")
        latent, links = self.aggregation(h, features, self.features, self.kind)
        return latent, links, features

    def encode_state(self, state: Tensor, action: Tensor, history: Tensor | dict | None = None) -> Tensor:
        return self._encode(state, action, history)[0]

    def forward(self, state: Tensor, action: Tensor, history: Tensor | dict | None = None) -> Tensor:
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(f"action must be [batch,{self.action_dim}]")
        latent, links, features = self._encode(state, action, history)
        if self.kind in ("state_mlp", "nerd_like"):
            return self.state_decoder(latent)
        if self.kind == "independent":
            dynamic = features["points"][:, self.features.dynamic]
            body = features["body"].mean(1)
            free = self.free_response(torch.cat((state, action, body), -1))
            total = torch.zeros_like(free)
            # Each static geometry contributes independently. Selecting only a
            # nearest support would change A's input when B appears and would
            # inadvertently enable non-additive counterfactual effects.
            for geom in self.features.metadata["geoms"]:
                if int(geom["body_id"]) in self.features.dynamic_lookup:
                    continue
                indices = torch.where(self.features.point_geom == int(geom["id"]))[0]
                if not len(indices):
                    continue
                static = features["points"][:, indices]
                # Chunk dynamic points to bound activation sizes for dense clouds.
                contribution = torch.zeros_like(free)
                for start in range(0, dynamic.shape[1], 32):
                    current = dynamic[:, start:start+32]
                    first = current[:, :, None].expand(-1, -1, static.shape[1], -1)
                    second = static[:, None].expand(-1, current.shape[1], -1, -1)
                    relative = second[..., :3] - first[..., :3]
                    distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
                    contact = torch.cat((relative, second[..., 3:6], second[..., 6:9]-first[..., 6:9],
                                         distance, second[..., 9:-1]), -1)
                    context = body[:, None, None].expand(-1, current.shape[1], static.shape[1], -1)
                    response = self.local_response(torch.cat((first, contact, context), -1))
                    gate = torch.exp(-distance.square() / .1**2)
                    contribution = contribution + (response * gate).sum((1, 2)) / (dynamic.shape[1] * static.shape[1])
                total = total + contribution
            return free + total
        return self.aggregation.decode(latent, links, self.features)

    def freeze_transfer(self, mode: str = "none") -> "DynamicsModel":
        if mode not in ("none", "point", "aggregation"):
            raise ValueError("freeze mode must be none, point, or aggregation")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not ((mode == "point" and name.startswith("point_encoder.")) or
                                          (mode == "aggregation" and not name.startswith("point_encoder."))))
        return self

    @torch.no_grad()
    def coverage(self, state: Tensor, action: Tensor, threshold: float = .08) -> dict:
        features = self.features(state, action)
        return local_pair_coverage(features["positions"], self.features.dynamic,
            self.config["window_size"], self.config["local_layers"], threshold, self.config["serialization"])


def build_model(scene_or_metadata: Any, config: dict | str | None = None) -> DynamicsModel:
    return DynamicsModel(scene_or_metadata, {"name": config} if isinstance(config, str) else config)


def transfer_weights(source: DynamicsModel, target: DynamicsModel, freeze: str = "none") -> dict:
    """Transfer every compatible learned tensor; scene buffers are nonpersistent."""
    existing = target.state_dict()
    new_joint_head = source.nq == 7 and target.nq > 7
    compatible = {k: v for k, v in source.state_dict().items() if k in existing and v.shape == existing[k].shape
                  and not (new_joint_head and k.startswith("aggregation.joint_decoder."))}
    target.load_state_dict(compatible, strict=False)
    target.freeze_transfer(freeze)
    if new_joint_head and isinstance(target.aggregation, Hierarchy):
        for parameter in target.aggregation.joint_decoder.parameters():
            parameter.requires_grad_(True)
    return {"transferred_tensors": len(compatible), "skipped_tensors": len(source.state_dict())-len(compatible),
            "freeze": freeze, "new_joint_head_trainable": new_joint_head}
