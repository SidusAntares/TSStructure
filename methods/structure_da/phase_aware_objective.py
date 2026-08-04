"""Source-only prototypes and unified objectives for phase-aware T/S tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .eden_alignment import EDENDomainAlignmentOutput
from .quality_fusion import TwoScaleQualityLossOutput
from .temporal_geometry import warp_to_identity_tangent
from .temporal_selection import TrendStructurePhaseSelectionOutput


def _finite(name: str, value: float) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def _positive_int(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _validate_nonnegative_dataclass(instance) -> None:
    for name, value in vars(instance).items():
        value = _finite(name, value)
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
        object.__setattr__(instance, name, value)


@dataclass(frozen=True)
class PhaseAwareSemanticFeatures:
    aligned_structure_srvf: Tensor
    aligned_structure_support: Tensor
    shape_feature: Tensor
    trend_embedding: Tensor
    structure_embedding: Tensor
    shape_valid: Tensor
    component_valid: Tensor


@dataclass(frozen=True)
class PrototypeAlignmentConfig:
    num_classes: int
    canonical_grid_size: int
    srvf_dim: int
    shape_dim: int
    raw_dim: int
    prototype_momentum: float = 0.99
    radius_buffer_size: int = 2048
    min_radius_samples: int = 32
    q_inner_quantile: float = 0.75
    q_outer_quantile: float = 0.95
    feature_inner_quantile: float = 0.75
    min_common_support: float = 0.05
    min_source_class_samples_for_separation: int = 2
    q_temperature: float = 0.10
    z_temperature: float = 0.10
    trend_temperature: float = 0.10
    structure_temperature: float = 0.10
    q_separation_margin: float = 1.0
    target_q_margin: float = 0.10
    raw_pull_confidence: float = 0.50
    raw_huber_delta: float = 0.10
    eps: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("num_classes", "canonical_grid_size", "srvf_dim", "shape_dim", "raw_dim", "radius_buffer_size", "min_radius_samples", "min_source_class_samples_for_separation"):
            minimum = 2 if name in ("num_classes", "canonical_grid_size") else 1
            object.__setattr__(self, name, _positive_int(name, getattr(self, name), minimum))
        floating = (
            "prototype_momentum", "q_inner_quantile", "q_outer_quantile",
            "feature_inner_quantile", "min_common_support", "q_temperature",
            "z_temperature", "trend_temperature", "structure_temperature",
            "q_separation_margin", "target_q_margin", "raw_pull_confidence",
            "raw_huber_delta", "eps",
        )
        for name in floating:
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if not 0 <= self.prototype_momentum < 1:
            raise ValueError("prototype_momentum must lie in [0, 1)")
        for name in ("q_inner_quantile", "q_outer_quantile", "feature_inner_quantile"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie in (0, 1)")
        if self.q_inner_quantile >= self.q_outer_quantile:
            raise ValueError("q_inner_quantile must be less than q_outer_quantile")
        if not 0 <= self.min_common_support <= 1:
            raise ValueError("min_common_support must lie in [0, 1]")
        for name in ("q_temperature", "z_temperature", "trend_temperature", "structure_temperature", "raw_huber_delta", "eps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name in ("q_separation_margin", "target_q_margin", "raw_pull_confidence"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.raw_pull_confidence > 1:
            raise ValueError("raw_pull_confidence must not exceed 1")


@dataclass(frozen=True)
class PrototypeAlignmentWeights:
    q_compact: float = 1.0
    q_separate: float = 1.0
    z_proto: float = 1.0
    q_to_z_source: float = 1.0
    raw_proto: float = 1.0
    q_to_z_target: float = 1.0
    z_pull: float = 1.0
    q_to_raw_target: float = 1.0
    raw_pull: float = 1.0

    def __post_init__(self) -> None:
        _validate_nonnegative_dataclass(self)


def support_weighted_srvf_distance(
    sample_srvf: Tensor,
    sample_support: Tensor,
    prototype_srvf: Tensor,
    prototype_support: Tensor,
    integration_weights: Tensor,
    *,
    min_common_support: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(sample_srvf, Tensor) or sample_srvf.ndim != 3:
        raise ValueError("sample_srvf must have shape [B, K, D]")
    b, k, d = sample_srvf.shape
    if prototype_srvf.ndim != 3 or prototype_srvf.shape[1:] != (k, d):
        raise ValueError("prototype_srvf must have shape [C, K, D]")
    if sample_support.shape != (b, k) or prototype_support.shape != (prototype_srvf.shape[0], k):
        raise ValueError("support tensors have incompatible shapes")
    if integration_weights.shape != (k,):
        raise ValueError("integration_weights must have shape [K]")
    tensors = (sample_srvf, sample_support, prototype_srvf, prototype_support, integration_weights)
    if any(not x.is_floating_point() for x in tensors):
        raise ValueError("distance inputs must be floating point")
    if any(x.device != sample_srvf.device or x.dtype != sample_srvf.dtype for x in tensors):
        raise ValueError("distance inputs must share dtype and device")
    if any(not torch.isfinite(x).all().item() for x in tensors):
        raise ValueError("distance inputs must be finite")
    if torch.any((sample_support < 0) | (sample_support > 1)).item() or torch.any(
        (prototype_support < 0) | (prototype_support > 1)
    ).item():
        raise ValueError("support values must lie in [0, 1]")
    if torch.any(integration_weights < 0).item():
        raise ValueError("integration_weights must be nonnegative")
    min_common_support = _finite("min_common_support", min_common_support)
    eps = _finite("eps", eps)
    if not 0 <= min_common_support <= 1 or eps <= 0:
        raise ValueError("invalid support threshold or eps")
    common = torch.minimum(sample_support[:, None, :], prototype_support[None, :, :])
    mass = (common * integration_weights).sum(-1)
    squared = (sample_srvf[:, None] - prototype_srvf[None]).square().sum(-1)
    distance = torch.sqrt((squared * common * integration_weights).sum(-1) / (mass + eps) + eps)
    valid = mass >= min_common_support
    distance = torch.where(valid, distance, torch.zeros_like(distance))
    return distance, mass, valid


def cosine_prototype_distance(
    feature: Tensor, prototype: Tensor, prototype_ready: Tensor, *, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    if (
        not isinstance(feature, Tensor)
        or not isinstance(prototype, Tensor)
        or not feature.is_floating_point()
        or not prototype.is_floating_point()
        or feature.ndim != 2
        or prototype.ndim != 2
        or feature.shape[1] != prototype.shape[1]
    ):
        raise ValueError("feature and prototype must have shapes [B,D] and [C,D]")
    if feature.dtype != prototype.dtype or feature.device != prototype.device:
        raise ValueError("feature and prototype must share dtype and device")
    if prototype_ready.dtype != torch.bool or prototype_ready.shape != (prototype.shape[0],) or prototype_ready.device != feature.device:
        raise ValueError("prototype_ready must have shape [C]")
    if not torch.isfinite(feature).all().item() or not torch.isfinite(prototype).all().item():
        raise ValueError("features must be finite")
    eps = _finite("eps", eps)
    if eps <= 0:
        raise ValueError("eps must be positive")
    normalized_feature = feature / (feature.norm(dim=-1, keepdim=True) + eps)
    normalized_prototype = prototype / (prototype.norm(dim=-1, keepdim=True) + eps)
    distance = (1 - normalized_feature @ normalized_prototype.T).clamp(0, 2)
    valid = prototype_ready[None, :].expand(feature.shape[0], -1)
    distance = torch.where(valid, distance, torch.zeros_like(distance))
    return normalized_feature, distance, valid


def _masked_softmax(logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    if logits.shape != mask.shape or logits.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("logits and mask must have matching [B,C] shape")
    row_valid = mask.any(-1)
    safe = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(safe, -1) * mask.to(logits.dtype)
    probabilities = torch.where(row_valid[:, None], probabilities, torch.zeros_like(probabilities))
    return probabilities, row_valid


def _graph_zero(*references: Tensor) -> Tensor:
    if not references:
        return torch.tensor(0.0)
    return sum(
        (
            torch.where(torch.isfinite(x), x, torch.zeros_like(x)).sum() * 0
            for x in references
        ),
        references[0].new_zeros(()),
    )


def _resolve_class_radius(
    class_radius: Tensor,
    class_ready: Tensor,
    global_radius: Tensor,
    global_ready: Tensor,
) -> tuple[Tensor, Tensor]:
    """Resolve each class radius without copying a fallback into class state."""
    resolved = torch.where(class_ready, class_radius, global_radius.expand_as(class_radius))
    return resolved, class_ready | global_ready


@dataclass(frozen=True)
class PrototypeAlignmentLossOutput:
    total_loss: Tensor
    source_shape_loss: Tensor
    source_raw_loss: Tensor
    target_semantic_loss: Tensor
    q_compact_loss: Tensor
    q_separate_loss: Tensor
    z_proto_loss: Tensor
    q_to_z_source_loss: Tensor
    trend_proto_loss: Tensor
    structure_proto_loss: Tensor
    q_to_z_target_loss: Tensor
    z_pull_loss: Tensor
    q_to_trend_target_loss: Tensor
    q_to_structure_target_loss: Tensor
    trend_pull_loss: Tensor
    structure_pull_loss: Tensor
    source_q_compact_count: Tensor
    source_q_pair_count: Tensor
    source_z_proto_count: Tensor
    source_trend_proto_count: Tensor
    source_structure_proto_count: Tensor
    source_relation_count: Tensor
    target_teacher_count: Tensor
    target_inner_count: Tensor
    target_middle_count: Tensor
    target_outer_count: Tensor
    target_z_pull_count: Tensor
    target_trend_relation_count: Tensor
    target_structure_relation_count: Tensor
    target_trend_pull_count: Tensor
    target_structure_pull_count: Tensor
    target_pseudo_label: Tensor
    target_teacher_mask: Tensor
    target_inner_mask: Tensor
    target_middle_mask: Tensor
    target_outer_mask: Tensor


class PhaseAwarePrototypeAlignment(nn.Module):
    """Maintain source-only q/z/T/S prototypes and compute dual-space losses."""

    def __init__(self, config: PrototypeAlignmentConfig, weights: PrototypeAlignmentWeights | None = None) -> None:
        super().__init__()
        if not isinstance(config, PrototypeAlignmentConfig):
            raise ValueError("config must be PrototypeAlignmentConfig")
        self.config = config
        self.weights = weights or PrototypeAlignmentWeights()
        if not isinstance(self.weights, PrototypeAlignmentWeights):
            raise ValueError("weights must be PrototypeAlignmentWeights")
        c, k, dq, dz, dr, n = config.num_classes, config.canonical_grid_size, config.srvf_dim, config.shape_dim, config.raw_dim, config.radius_buffer_size
        integration = torch.full((k,), 1 / (k - 1))
        integration[[0, -1]] *= 0.5
        self.register_buffer("integration_weights", integration)
        for name, shape in (("q_prototype", (c, k, dq)), ("q_support", (c, k)), ("z_prototype", (c, dz)), ("trend_prototype", (c, dr)), ("structure_prototype", (c, dr))):
            self.register_buffer(name, torch.zeros(shape))
        for name in ("q", "z", "trend", "structure"):
            self.register_buffer(f"{name}_prototype_ready", torch.zeros(c, dtype=torch.bool))
            self.register_buffer(f"{name}_update_count", torch.zeros(c, dtype=torch.long))
            self.register_buffer(f"{name}_distance_buffer", torch.zeros(c, n))
            self.register_buffer(f"{name}_distance_count", torch.zeros(c, dtype=torch.long))
            self.register_buffer(f"{name}_distance_index", torch.zeros(c, dtype=torch.long))
            self.register_buffer(f"{name}_radius_inner", torch.zeros(c))
            self.register_buffer(f"{name}_radius_ready", torch.zeros(c, dtype=torch.bool))
            self.register_buffer(f"{name}_global_radius_inner", torch.zeros(()))
            self.register_buffer(f"{name}_global_radius_inner_ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer("q_radius_outer", torch.zeros(c))
        self.register_buffer("q_global_radius_outer", torch.zeros(()))
        self.register_buffer("q_global_radius_outer_ready", torch.zeros((), dtype=torch.bool))

    def _validate_features(self, value: PhaseAwareSemanticFeatures, name: str) -> None:
        if not isinstance(value, PhaseAwareSemanticFeatures):
            raise ValueError(f"{name} must be PhaseAwareSemanticFeatures")
        q, support, z, trend, structure = value.aligned_structure_srvf, value.aligned_structure_support, value.shape_feature, value.trend_embedding, value.structure_embedding
        if q.ndim != 3 or q.shape[1:] != (self.config.canonical_grid_size, self.config.srvf_dim):
            raise ValueError("aligned_structure_srvf has invalid shape")
        b = q.shape[0]
        expected = ((support, (b, self.config.canonical_grid_size)), (z, (b, self.config.shape_dim)), (trend, (b, self.config.raw_dim)), (structure, (b, self.config.raw_dim)))
        if any(x.shape != shape for x, shape in expected):
            raise ValueError("semantic feature shapes are invalid")
        floating = (q, support, z, trend, structure)
        if any(not x.is_floating_point() or x.dtype != q.dtype or x.device != q.device for x in floating):
            raise ValueError("semantic floating tensors must share dtype and device")
        if not torch.isfinite(support).all().item() or torch.any((support < 0) | (support > 1)).item():
            raise ValueError("support must be finite in [0,1]")
        for tensor, valid in ((q, value.shape_valid), (z, value.shape_valid), (trend, value.component_valid), (structure, value.component_valid)):
            if valid.dtype != torch.bool or valid.shape != (b,) or valid.device != q.device:
                raise ValueError("valid masks must be boolean [B]")
            if not torch.isfinite(tensor[valid]).all().item():
                raise ValueError("valid semantic values must be finite")

    def _validate_labels(self, labels: Tensor, batch: int, device: torch.device) -> None:
        if labels.dtype != torch.long or labels.shape != (batch,) or labels.device != device:
            raise ValueError("source_labels must be long [B]")
        if torch.any((labels < 0) | (labels >= self.config.num_classes)).item():
            raise ValueError("source_labels are out of range")

    def _resolved_radius(self, branch: str, *, outer: bool = False) -> tuple[Tensor, Tensor]:
        suffix = "outer" if outer else "inner"
        radius = getattr(self, f"{branch}_radius_{suffix}")
        ready = getattr(self, f"{branch}_radius_ready")
        global_radius = getattr(self, f"{branch}_global_radius_{suffix}")
        global_ready = getattr(self, f"{branch}_global_radius_{suffix}_ready")
        return _resolve_class_radius(radius, ready, global_radius, global_ready)

    def _append(self, branch: str, class_index: int, values: Tensor) -> None:
        if values.numel() == 0:
            return
        buffer = getattr(self, f"{branch}_distance_buffer")
        count = getattr(self, f"{branch}_distance_count")
        index = getattr(self, f"{branch}_distance_index")
        n = buffer.shape[1]
        original_n = values.numel()
        start = int(index[class_index].item())
        if original_n > n:
            values = values[-n:]
            start = (start + original_n - n) % n
        slots = (torch.arange(values.numel(), device=buffer.device) + start) % n
        buffer[class_index, slots] = values
        index[class_index] = (int(index[class_index].item()) + original_n) % n
        count[class_index] = min(n, int(count[class_index].item()) + original_n)

    def _refresh_radii(self) -> None:
        cfg = self.config
        for branch in ("q", "z", "trend", "structure"):
            buffer = getattr(self, f"{branch}_distance_buffer")
            count = getattr(self, f"{branch}_distance_count")
            inner = getattr(self, f"{branch}_radius_inner")
            ready = getattr(self, f"{branch}_radius_ready")
            pooled = []
            for c in range(cfg.num_classes):
                amount = int(count[c].item())
                values = buffer[c, :amount]
                if amount:
                    pooled.append(values)
                if amount >= cfg.min_radius_samples:
                    inner[c] = torch.quantile(values, cfg.q_inner_quantile if branch == "q" else cfg.feature_inner_quantile)
                    ready[c] = True
                    if branch == "q":
                        self.q_radius_outer[c] = torch.quantile(values, cfg.q_outer_quantile)
            if sum(x.numel() for x in pooled) >= cfg.min_radius_samples:
                values = torch.cat(pooled)
                getattr(self, f"{branch}_global_radius_inner").copy_(torch.quantile(values, cfg.q_inner_quantile if branch == "q" else cfg.feature_inner_quantile))
                getattr(self, f"{branch}_global_radius_inner_ready").fill_(True)
                if branch == "q":
                    self.q_global_radius_outer.copy_(torch.quantile(values, cfg.q_outer_quantile))
                    self.q_global_radius_outer_ready.fill_(True)

    def _batch_q(self, source: PhaseAwareSemanticFeatures, labels: Tensor):
        c, k, d = self.config.num_classes, self.config.canonical_grid_size, self.config.srvf_dim
        prototypes = source.aligned_structure_srvf.new_zeros(c, k, d)
        supports = source.aligned_structure_support.new_zeros(c, k)
        counts = labels.new_zeros(c)
        for cls in range(c):
            mask = (labels == cls) & source.shape_valid
            counts[cls] = mask.sum()
            if mask.any():
                s = source.aligned_structure_support[mask]
                supports[cls] = s.sum(0) / mask.sum().clamp_min(1)
                denominator = s.sum(0)
                numerator = (source.aligned_structure_srvf[mask] * s[..., None]).sum(0)
                prototypes[cls] = torch.where(denominator[:, None] > 0, numerator / denominator[:, None].clamp_min(self.config.eps), torch.zeros_like(numerator))
        return prototypes, supports, counts

    def _batch_feature(self, feature: Tensor, valid: Tensor, labels: Tensor):
        result = feature.new_zeros(self.config.num_classes, feature.shape[1])
        counts = labels.new_zeros(self.config.num_classes)
        normalized = F.normalize(feature, dim=-1, eps=self.config.eps)
        for cls in range(self.config.num_classes):
            mask = (labels == cls) & valid
            counts[cls] = mask.sum()
            if mask.any():
                result[cls] = F.normalize(normalized[mask].mean(0), dim=0, eps=self.config.eps)
        return result, counts

    @torch.no_grad()
    def update_source_state(self, source: PhaseAwareSemanticFeatures, source_labels: Tensor) -> None:
        self._validate_features(source, "source")
        self._validate_labels(source_labels, source.aligned_structure_srvf.shape[0], source.aligned_structure_srvf.device)
        old_q, old_support, old_q_ready = self.q_prototype.clone(), self.q_support.clone(), self.q_prototype_ready.clone()
        old = {name: (getattr(self, f"{name}_prototype").clone(), getattr(self, f"{name}_prototype_ready").clone()) for name in ("z", "trend", "structure")}
        q_distance, _, q_valid = support_weighted_srvf_distance(source.aligned_structure_srvf, source.aligned_structure_support, old_q, old_support, self.integration_weights, min_common_support=self.config.min_common_support, eps=self.config.eps)
        feature_distances = {}
        for name, feature in (("z", source.shape_feature), ("trend", source.trend_embedding), ("structure", source.structure_embedding)):
            feature_distances[name] = cosine_prototype_distance(feature, old[name][0], old[name][1], eps=self.config.eps)[1]
        for cls in range(self.config.num_classes):
            class_mask = source_labels == cls
            self._append("q", cls, q_distance[class_mask & source.shape_valid & old_q_ready[cls] & q_valid[:, cls], cls])
            for name, valid in (("z", source.shape_valid), ("trend", source.component_valid), ("structure", source.component_valid)):
                self._append(name, cls, feature_distances[name][class_mask & valid & old[name][1][cls], cls])
        self._refresh_radii()
        batch_q, batch_support, q_counts = self._batch_q(source, source_labels)
        batch_features = {
            "z": self._batch_feature(source.shape_feature, source.shape_valid, source_labels),
            "trend": self._batch_feature(source.trend_embedding, source.component_valid, source_labels),
            "structure": self._batch_feature(source.structure_embedding, source.component_valid, source_labels),
        }
        m = self.config.prototype_momentum
        for cls in range(self.config.num_classes):
            if q_counts[cls] > 0:
                if not self.q_prototype_ready[cls]:
                    self.q_prototype[cls], self.q_support[cls] = batch_q[cls], batch_support[cls]
                else:
                    new_support = m * self.q_support[cls] + (1 - m) * batch_support[cls]
                    numerator = m * self.q_support[cls, :, None] * self.q_prototype[cls] + (1 - m) * batch_support[cls, :, None] * batch_q[cls]
                    self.q_prototype[cls] = torch.where(new_support[:, None] > 0, numerator / new_support[:, None].clamp_min(self.config.eps), torch.zeros_like(numerator))
                    self.q_support[cls] = new_support
                self.q_prototype_ready[cls] = True
                self.q_update_count[cls] += 1
            for name in ("z", "trend", "structure"):
                batch, counts = batch_features[name]
                if counts[cls] > 0:
                    prototype = getattr(self, f"{name}_prototype")
                    ready = getattr(self, f"{name}_prototype_ready")
                    prototype[cls] = batch[cls] if not ready[cls] else F.normalize(m * prototype[cls] + (1 - m) * batch[cls], dim=0, eps=self.config.eps)
                    ready[cls] = True
                    getattr(self, f"{name}_update_count")[cls] += 1

    def _mean(self, values: Tensor, mask: Tensor, *refs: Tensor) -> tuple[Tensor, Tensor]:
        count = mask.sum()
        if mask.any():
            return values[mask].mean(), count
        return _graph_zero(values, *refs), count

    def _prototype_ce(self, feature: Tensor, labels: Tensor, valid: Tensor, prototype: Tensor, ready: Tensor, temperature: float):
        normalized, distance, class_valid = cosine_prototype_distance(feature, prototype, ready, eps=self.config.eps)
        logits = normalized @ F.normalize(prototype, dim=-1, eps=self.config.eps).T / temperature
        mask = valid & ready[labels] & (ready.sum() >= 2)
        if mask.any():
            return F.cross_entropy(logits[mask].masked_fill(~ready[None], torch.finfo(logits.dtype).min), labels[mask]), mask.sum(), distance, logits, class_valid
        return _graph_zero(feature), mask.sum(), distance, logits, class_valid

    def _kl(self, teacher: Tensor, student: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        active = mask.any(-1)
        if active.any():
            value = (teacher.detach() * (torch.log(teacher.detach() + self.config.eps) - torch.log(student + self.config.eps))).sum(-1)
            return value[active].mean(), active.sum()
        return _graph_zero(student), active.sum()

    def _raw_huber(self, error: Tensor) -> Tensor:
        delta = self.config.raw_huber_delta
        return torch.where(error <= delta, error.square() / (2 * delta), error - delta / 2)

    def forward(self, source: PhaseAwareSemanticFeatures, source_labels: Tensor, target: PhaseAwareSemanticFeatures, target_shape_feature_da: Tensor) -> PrototypeAlignmentLossOutput:
        self._validate_features(source, "source")
        self._validate_features(target, "target")
        self._validate_labels(source_labels, source.aligned_structure_srvf.shape[0], source.aligned_structure_srvf.device)
        if target_shape_feature_da.shape != target.shape_feature.shape or target_shape_feature_da.dtype != target.shape_feature.dtype or target_shape_feature_da.device != target.shape_feature.device or not torch.isfinite(target_shape_feature_da).all().item():
            raise ValueError("target_shape_feature_da must match target shape feature")
        cfg, w = self.config, self.weights
        q_inner, q_inner_ready = self._resolved_radius("q", outer=False)
        q_outer, q_outer_ready = self._resolved_radius("q", outer=True)
        q_source, _, q_source_valid = support_weighted_srvf_distance(source.aligned_structure_srvf, source.aligned_structure_support, self.q_prototype, self.q_support, self.integration_weights, min_common_support=cfg.min_common_support, eps=cfg.eps)
        rows = torch.arange(source_labels.numel(), device=source_labels.device)
        true_q = q_source[rows, source_labels]
        compact_mask = source.shape_valid & self.q_prototype_ready[source_labels] & q_source_valid[rows, source_labels] & q_inner_ready[source_labels]
        compact_values = (true_q / (q_inner[source_labels] + cfg.eps) - 1).clamp_min(0).square()
        q_compact, q_compact_count = self._mean(compact_values, compact_mask, source.aligned_structure_srvf)

        batch_q, batch_support, batch_counts = self._batch_q(source, source_labels)
        center_distance, _, center_valid = support_weighted_srvf_distance(batch_q, batch_support, batch_q, batch_support, self.integration_weights, min_common_support=cfg.min_common_support, eps=cfg.eps)
        pair_values, pair_masks = [], []
        for c in range(cfg.num_classes):
            for k in range(c + 1, cfg.num_classes):
                ready = batch_counts[c] >= cfg.min_source_class_samples_for_separation and batch_counts[k] >= cfg.min_source_class_samples_for_separation and center_valid[c, k] and q_inner_ready[c] and q_inner_ready[k]
                pair_values.append((cfg.q_separation_margin - center_distance[c, k] / (q_inner[c] + q_inner[k] + cfg.eps)).clamp_min(0).square())
                pair_masks.append(torch.as_tensor(ready, device=source_labels.device))
        pair_values = torch.stack(pair_values)
        pair_mask = torch.stack(pair_masks).bool()
        q_separate, q_pair_count = self._mean(pair_values, pair_mask, source.aligned_structure_srvf)

        z_proto, z_count, _, z_logits, _ = self._prototype_ce(source.shape_feature, source_labels, source.shape_valid, self.z_prototype, self.z_prototype_ready, cfg.z_temperature)
        trend_proto, trend_count, _, _, _ = self._prototype_ce(source.trend_embedding, source_labels, source.component_valid, self.trend_prototype, self.trend_prototype_ready, cfg.trend_temperature)
        structure_proto, structure_count, _, _, _ = self._prototype_ce(source.structure_embedding, source_labels, source.component_valid, self.structure_prototype, self.structure_prototype_ready, cfg.structure_temperature)

        q_class_mask = q_source_valid & self.q_prototype_ready[None] & q_inner_ready[None] & self.z_prototype_ready[None]
        pq_source, pq_source_valid = _masked_softmax(-q_source / (q_inner[None] + cfg.eps) / cfg.q_temperature, q_class_mask)
        pz_source, _ = _masked_softmax(z_logits, q_class_mask)
        normalized_q_source = q_source / (q_inner[None] + cfg.eps)
        nearest_q = normalized_q_source.masked_fill(~q_class_mask, torch.inf).argmin(-1)
        relation_rows = source.shape_valid & pq_source_valid & (q_class_mask.sum(-1) >= 2) & (nearest_q == source_labels)
        relation_mask = q_class_mask & relation_rows[:, None]
        q_to_z_source, source_relation_count = self._kl(pq_source, pz_source, relation_mask)

        source_shape = w.q_compact * q_compact + w.q_separate * q_separate + w.z_proto * z_proto + w.q_to_z_source * q_to_z_source
        source_raw = w.raw_proto * 0.5 * (trend_proto + structure_proto)

        with torch.no_grad():
            q_target, _, q_target_valid = support_weighted_srvf_distance(target.aligned_structure_srvf.detach(), target.aligned_structure_support, self.q_prototype, self.q_support, self.integration_weights, min_common_support=cfg.min_common_support, eps=cfg.eps)
            teacher_classes = q_target_valid & self.q_prototype_ready[None] & q_inner_ready[None] & q_outer_ready[None] & self.z_prototype_ready[None]
            delta_q = q_target / (q_inner[None] + cfg.eps)
            masked_delta = delta_q.masked_fill(~teacher_classes, torch.inf)
            sorted_delta, _ = masked_delta.sort(-1)
            yq = masked_delta.argmin(-1)
            margin = sorted_delta[:, 1] - sorted_delta[:, 0]
            normalized_z = F.normalize(target.shape_feature.detach(), dim=-1, eps=cfg.eps)
            z_teacher_logits = normalized_z @ F.normalize(self.z_prototype, dim=-1, eps=cfg.eps).T
            yz = z_teacher_logits.masked_fill(~teacher_classes, -torch.inf).argmax(-1)
            tr = torch.arange(yq.numel(), device=yq.device)
            enough = teacher_classes.sum(-1) >= 2
            selected_q = q_target[tr, yq]
            teacher = target.shape_valid & enough & (yq == yz) & (margin >= cfg.target_q_margin) & (selected_q <= q_outer[yq])
            inner_mask = teacher & (selected_q <= q_inner[yq])
            middle_mask = teacher & (selected_q > q_inner[yq]) & (selected_q <= q_outer[yq])
            outer_mask = target.shape_valid & enough & (selected_q > q_outer[yq])
            pseudo = torch.where(teacher, yq, torch.full_like(yq, -1))
            pq_target, _ = _masked_softmax(-q_target / (q_inner[None] + cfg.eps) / cfg.q_temperature, teacher_classes)

        normalized_da, z_distance, _ = cosine_prototype_distance(target_shape_feature_da, self.z_prototype, self.z_prototype_ready, eps=cfg.eps)
        z_da_logits = normalized_da @ F.normalize(self.z_prototype, dim=-1, eps=cfg.eps).T / cfg.z_temperature
        pz_target, _ = _masked_softmax(z_da_logits, teacher_classes)
        q_to_z_mask = teacher_classes & teacher[:, None]
        q_to_z_target, _ = self._kl(pq_target, pz_target, q_to_z_mask)
        z_radius, z_radius_ready = self._resolved_radius("z", outer=False)
        safe_y = pseudo.clamp_min(0)
        z_excess = (z_distance[torch.arange(pseudo.numel(), device=pseudo.device), safe_y] - z_radius[safe_y]).clamp_min(0).square()
        z_pull_mask = middle_mask & z_radius_ready[safe_y]
        z_pull, z_pull_count = self._mean(z_excess, z_pull_mask, target_shape_feature_da)

        def target_raw(feature: Tensor, prototype: Tensor, ready: Tensor, temperature: float, branch: str):
            normalized, distance, _ = cosine_prototype_distance(feature, prototype, ready, eps=cfg.eps)
            logits = normalized @ F.normalize(prototype, dim=-1, eps=cfg.eps).T / temperature
            classes = teacher_classes & ready[None]
            probabilities, row_valid = _masked_softmax(logits, classes)
            branch_pq = pq_target * classes.to(pq_target.dtype)
            branch_pq = branch_pq / branch_pq.sum(-1, keepdim=True).clamp_min(cfg.eps)
            relation_rows = teacher & target.component_valid & row_valid & (classes.sum(-1) >= 2) & classes[torch.arange(safe_y.numel(), device=safe_y.device), safe_y]
            relation, relation_count = self._kl(branch_pq, probabilities, classes & relation_rows[:, None])
            radius, radius_ready = self._resolved_radius(branch, outer=False)
            prediction = probabilities.argmax(-1)
            confidence = probabilities[torch.arange(safe_y.numel(), device=safe_y.device), safe_y]
            pseudo_class_ready = classes[
                torch.arange(safe_y.numel(), device=safe_y.device), safe_y
            ]
            pull_mask = (
                teacher
                & target.component_valid
                & pseudo_class_ready
                & (prediction == safe_y)
                & (confidence >= cfg.raw_pull_confidence)
                & radius_ready[safe_y]
            )
            excess = (distance[torch.arange(safe_y.numel(), device=safe_y.device), safe_y] - radius[safe_y]).clamp_min(0)
            pull, pull_count = self._mean(self._raw_huber(excess), pull_mask, feature)
            return relation, relation_count, pull, pull_count

        q_trend, trend_relation_count, trend_pull, trend_pull_count = target_raw(target.trend_embedding, self.trend_prototype, self.trend_prototype_ready, cfg.trend_temperature, "trend")
        q_structure, structure_relation_count, structure_pull, structure_pull_count = target_raw(target.structure_embedding, self.structure_prototype, self.structure_prototype_ready, cfg.structure_temperature, "structure")
        target_semantic = w.q_to_z_target * q_to_z_target + w.z_pull * z_pull + w.q_to_raw_target * 0.5 * (q_trend + q_structure) + w.raw_pull * 0.5 * (trend_pull + structure_pull)
        total = source_shape + source_raw + target_semantic
        return PrototypeAlignmentLossOutput(
            total, source_shape, source_raw, target_semantic,
            q_compact, q_separate, z_proto, q_to_z_source,
            trend_proto, structure_proto, q_to_z_target, z_pull,
            q_trend, q_structure, trend_pull, structure_pull,
            q_compact_count, q_pair_count, z_count, trend_count, structure_count,
            source_relation_count, teacher.sum(), inner_mask.sum(), middle_mask.sum(), outer_mask.sum(),
            z_pull_count, trend_relation_count, structure_relation_count, trend_pull_count, structure_pull_count,
            pseudo, teacher, inner_mask, middle_mask, outer_mask,
        )


@dataclass(frozen=True)
class TrendLedGeometryLossOutput:
    total_loss: Tensor
    candidate_loss: Tensor
    center_loss: Tensor
    valid_candidate_sample_count: Tensor
    source_center_count: Tensor


class TrendLedGeometryObjective(nn.Module):
    def __init__(self, candidate_weight: float = 1.0, center_weight: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.candidate_weight = _finite("candidate_weight", candidate_weight)
        self.center_weight = _finite("center_weight", center_weight)
        self.eps = _finite("eps", eps)
        if min(self.candidate_weight, self.center_weight) < 0 or self.eps <= 0:
            raise ValueError("weights must be nonnegative and eps positive")

    def forward(self, selection: TrendStructurePhaseSelectionOutput, source_mask: Tensor) -> TrendLedGeometryLossOutput:
        legal = selection.candidate_legal_mask
        widths = selection.candidates.interval_widths
        batch = legal.shape[0] if legal.ndim == 2 else -1
        if (
            legal.ndim != 2
            or legal.dtype != torch.bool
            or widths.ndim != 3
            or widths.shape[:2] != legal.shape
            or widths.shape[-1] < 1
            or selection.candidate_softmin_score.shape != (batch,)
            or selection.selected_candidate_index.dtype != torch.long
            or selection.selected_candidate_index.shape != (batch,)
            or selection.phase_valid.dtype != torch.bool
            or selection.phase_valid.shape != (batch,)
            or source_mask.dtype != torch.bool
            or source_mask.shape != (batch,)
        ):
            raise ValueError("selection/source_mask shapes are invalid")
        tensor_fields = (
            widths,
            selection.candidate_softmin_score,
            legal,
            selection.selected_candidate_index,
            selection.phase_valid,
            source_mask,
        )
        if any(value.device != widths.device for value in tensor_fields):
            raise ValueError("selection/source_mask tensors must share a device")
        if not widths.is_floating_point() or not selection.candidate_softmin_score.is_floating_point():
            raise ValueError("candidate widths and scores must be floating point")
        if widths.dtype != selection.candidate_softmin_score.dtype:
            raise ValueError("candidate widths and scores must share a dtype")
        if not torch.isfinite(widths).all().item():
            raise ValueError("candidate widths must be finite")
        if torch.any((selection.selected_candidate_index < -1) | (selection.selected_candidate_index >= legal.shape[1])).item():
            raise ValueError("selected_candidate_index is out of range")
        candidate_valid = legal.any(-1)
        if not torch.isfinite(selection.candidate_softmin_score[candidate_valid]).all().item():
            raise ValueError("scores for valid candidates must be finite")
        candidate_loss = selection.candidate_softmin_score[candidate_valid].mean() if candidate_valid.any() else _graph_zero(selection.candidate_softmin_score, selection.candidates.interval_widths)
        batch, _, intervals = widths.shape
        safe = selection.selected_candidate_index.clamp_min(0)
        chosen = widths[torch.arange(batch, device=widths.device), safe]
        identity = torch.full_like(chosen, 1 / intervals)
        chosen = torch.where((selection.selected_candidate_index >= 0)[:, None], chosen, identity)
        tangent = warp_to_identity_tangent(chosen, eps=self.eps).tangent
        center_mask = source_mask & selection.phase_valid
        if center_mask.any():
            mean = tangent[center_mask].sum(0) / center_mask.sum().clamp_min(1)
            center_loss = mean.square().sum()
        else:
            center_loss = _graph_zero(tangent, widths)
        total = self.candidate_weight * candidate_loss + self.center_weight * center_loss
        return TrendLedGeometryLossOutput(total, candidate_loss, center_loss, candidate_valid.sum(), center_mask.sum())

    def forward_pair(
        self,
        source_selection: TrendStructurePhaseSelectionOutput,
        target_selection: TrendStructurePhaseSelectionOutput,
    ) -> TrendLedGeometryLossOutput:
        source_widths = source_selection.candidates.interval_widths
        target_widths = target_selection.candidates.interval_widths
        if source_widths.device != target_widths.device:
            raise ValueError("source and target candidates must share a device")
        if source_widths.dtype != target_widths.dtype:
            raise ValueError("source and target candidates must share a dtype")
        if source_widths.shape[1] != target_widths.shape[1]:
            raise ValueError("source and target candidate count must match")
        if source_widths.shape[2] != target_widths.shape[2]:
            raise ValueError("source and target canonical grid must match")

        source_result = self.forward(
            source_selection,
            torch.ones(
                source_widths.shape[0],
                dtype=torch.bool,
                device=source_widths.device,
            ),
        )
        target_result = self.forward(
            target_selection,
            torch.zeros(
                target_widths.shape[0],
                dtype=torch.bool,
                device=target_widths.device,
            ),
        )
        candidate_count = (
            source_result.valid_candidate_sample_count
            + target_result.valid_candidate_sample_count
        )
        candidate_numerator = (
            source_result.candidate_loss
            * source_result.valid_candidate_sample_count
            + target_result.candidate_loss
            * target_result.valid_candidate_sample_count
        )
        candidate_loss = torch.where(
            candidate_count > 0,
            candidate_numerator / candidate_count.clamp_min(1),
            _graph_zero(
                source_selection.candidate_softmin_score,
                target_selection.candidate_softmin_score,
            ),
        )
        center_loss = source_result.center_loss
        total = (
            self.candidate_weight * candidate_loss
            + self.center_weight * center_loss
        )
        return TrendLedGeometryLossOutput(
            total,
            candidate_loss,
            center_loss,
            candidate_count,
            source_result.source_center_count,
        )


@dataclass(frozen=True)
class PhaseAwareTaskLossWeights:
    classification: float = 1.0
    quality: float = 1.0
    source_shape: float = 1.0
    source_raw: float = 1.0
    global_domain: float = 1.0
    target_semantic: float = 1.0

    def __post_init__(self) -> None:
        _validate_nonnegative_dataclass(self)


@dataclass(frozen=True)
class PhaseAwareTaskLossOutput:
    total_loss: Tensor
    classification_loss: Tensor
    quality_loss: Tensor
    source_shape_loss: Tensor
    source_raw_loss: Tensor
    global_domain_loss: Tensor
    target_semantic_loss: Tensor
    prototype: PrototypeAlignmentLossOutput


class PhaseAwareTaskObjective(nn.Module):
    def __init__(self, weights: PhaseAwareTaskLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or PhaseAwareTaskLossWeights()
        if not isinstance(self.weights, PhaseAwareTaskLossWeights):
            raise ValueError("weights must be PhaseAwareTaskLossWeights")

    def forward(self, source_logits: Tensor, source_labels: Tensor, quality_loss: TwoScaleQualityLossOutput, alignment: EDENDomainAlignmentOutput, prototype: PrototypeAlignmentLossOutput) -> PhaseAwareTaskLossOutput:
        if source_logits.ndim != 2 or source_labels.dtype != torch.long or source_labels.shape != (source_logits.shape[0],):
            raise ValueError("source logits/labels have invalid shape")
        classification = F.cross_entropy(source_logits, source_labels)
        losses = (classification, quality_loss.total_loss, prototype.source_shape_loss, prototype.source_raw_loss, alignment.loss, prototype.target_semantic_loss)
        if any(x.ndim != 0 or not torch.isfinite(x).item() for x in losses):
            raise FloatingPointError("all task losses must be finite scalar tensors")
        w = self.weights
        total = w.classification * classification + w.quality * losses[1] + w.source_shape * losses[2] + w.source_raw * losses[3] + w.global_domain * losses[4] + w.target_semantic * losses[5]
        return PhaseAwareTaskLossOutput(total, *losses, prototype)
