"""Domain-level Shape statistics estimated from stable target labels.

The statistic is deliberately downstream of confirmed domain phase. Stable
labels already carry ``aligned_q_shape`` in the source reference coordinate;
this module never estimates or applies a sample-level warp and never consumes
target ground-truth labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch
from torch import Tensor

from .prototype_bank import SourcePrototypeBank, support_aware_q_distance
from .stable_target_labels import StableTargetLabelScanResult


_EPS = 1e-8


class DomainShapeStatus(str, Enum):
    UNAVAILABLE = "unavailable"
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class TargetShapeClassCenter:
    class_id: int
    center_q: Tensor
    center_support: Tensor
    sample_count: int
    effective_weight: float
    source_distance: float
    residual_q: Tensor
    valid: bool
    reject_reason: str | None


@dataclass(frozen=True)
class DomainShapeConfig:
    shape_min_valid_classes: int
    shape_min_samples_per_class: int
    shape_shared_ratio_min: float
    shape_leave_one_out_drift_max: float
    shape_center_drift_max: float
    shape_effect_norm_max: float
    shape_confirmation_patience: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.shape_min_valid_classes, bool)
            or not isinstance(self.shape_min_valid_classes, int)
            or self.shape_min_valid_classes < 2
        ):
            raise ValueError("shape_min_valid_classes must be an integer >= 2")
        if (
            isinstance(self.shape_min_samples_per_class, bool)
            or not isinstance(self.shape_min_samples_per_class, int)
            or self.shape_min_samples_per_class < 1
        ):
            raise ValueError("shape_min_samples_per_class must be an integer >= 1")
        if (
            isinstance(self.shape_confirmation_patience, bool)
            or not isinstance(self.shape_confirmation_patience, int)
            or self.shape_confirmation_patience < 2
        ):
            raise ValueError("shape_confirmation_patience must be an integer >= 2")
        if not math.isfinite(float(self.shape_shared_ratio_min)) or not (
            0.0 <= float(self.shape_shared_ratio_min) <= 1.0
        ):
            raise ValueError("shape_shared_ratio_min must lie in [0, 1]")
        for name in (
            "shape_leave_one_out_drift_max",
            "shape_center_drift_max",
            "shape_effect_norm_max",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class DomainShapeState:
    scan_index: int
    status: DomainShapeStatus
    class_centers: tuple[TargetShapeClassCenter, ...]
    valid_classes: tuple[int, ...]
    delta: Tensor | None
    interactions: tuple[Tensor, ...]
    rho_shape: float | None
    leave_one_out_drift: float | None
    center_drift: float | None
    confirmation_age: int


def _integration_weights(grid_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if grid_size < 2:
        raise ValueError("Shape grid must contain at least two points")
    weights = torch.full(
        (grid_size,),
        1.0 / (grid_size - 1),
        device=device,
        dtype=dtype,
    )
    weights[[0, -1]] *= 0.5
    return weights


def _l2_energy(q: Tensor, weights: Tensor) -> Tensor:
    if q.ndim != 2:
        raise ValueError("q must have shape [K,D]")
    if weights.shape != (q.shape[0],):
        raise ValueError("integration weights must have shape [K]")
    return (weights * q.square().sum(dim=-1)).sum()


def _l2_norm(q: Tensor, weights: Tensor) -> Tensor:
    return torch.sqrt(_l2_energy(q, weights).clamp_min(0.0))


def _valid_stable_shape(q: Tensor, support: Tensor) -> bool:
    return (
        isinstance(q, Tensor)
        and isinstance(support, Tensor)
        and q.ndim == 2
        and support.shape == q.shape[:1]
        and q.is_floating_point()
        and support.is_floating_point()
        and bool(torch.isfinite(q).all().item())
        and bool(torch.isfinite(support).all().item())
        and not bool(torch.any((support < 0.0) | (support > 1.0)).item())
        and bool(torch.any(support > 0.0).item())
    )


def _empty_center(
    *,
    class_id: int,
    sample_count: int,
    source_q: Tensor,
    source_support: Tensor,
    reason: str,
) -> TargetShapeClassCenter:
    return TargetShapeClassCenter(
        class_id=class_id,
        center_q=torch.zeros_like(source_q).detach(),
        center_support=torch.zeros_like(source_support).detach(),
        sample_count=sample_count,
        effective_weight=0.0,
        source_distance=float("nan"),
        residual_q=torch.zeros_like(source_q).detach(),
        valid=False,
        reject_reason=reason,
    )


def _build_class_centers(
    scan_result: StableTargetLabelScanResult,
    bank: SourcePrototypeBank,
    config: DomainShapeConfig,
) -> tuple[TargetShapeClassCenter, ...]:
    labels_by_class: dict[int, list] = {}
    for label in scan_result.stable_labels:
        labels_by_class.setdefault(int(label.class_id), []).append(label)

    centers: list[TargetShapeClassCenter] = []
    num_classes = int(bank.ready.numel())
    for class_id in sorted(labels_by_class):
        labels = labels_by_class[class_id]
        if not 0 <= class_id < num_classes:
            # Stable labels outside the source closed-set are structurally invalid.
            continue
        source_q = bank.shape_srvf[class_id].detach()
        source_support = bank.shape_support[class_id].detach()
        if not bool(bank.ready[class_id].item()):
            centers.append(
                _empty_center(
                    class_id=class_id,
                    sample_count=0,
                    source_q=source_q,
                    source_support=source_support,
                    reason="source_prototype_not_ready",
                )
            )
            continue

        usable = [
            label
            for label in labels
            if _valid_stable_shape(label.aligned_q_shape, label.aligned_q_support)
            and label.aligned_q_shape.shape == source_q.shape
            and label.aligned_q_support.shape == source_support.shape
        ]
        sample_count = len(usable)
        if sample_count < config.shape_min_samples_per_class:
            centers.append(
                _empty_center(
                    class_id=class_id,
                    sample_count=sample_count,
                    source_q=source_q,
                    source_support=source_support,
                    reason="insufficient_samples",
                )
            )
            continue

        q_stack = torch.stack(
            [
                label.aligned_q_shape.detach().to(
                    device=source_q.device, dtype=source_q.dtype
                )
                for label in usable
            ]
        )
        support_stack = torch.stack(
            [
                label.aligned_q_support.detach().to(
                    device=source_q.device, dtype=source_q.dtype
                )
                for label in usable
            ]
        )
        support_sum = support_stack.sum(dim=0)
        center_support = support_stack.mean(dim=0)
        center_q = (q_stack * support_stack.unsqueeze(-1)).sum(dim=0) / (
            support_sum.unsqueeze(-1) + _EPS
        )
        weights = _integration_weights(
            center_q.shape[0], device=center_q.device, dtype=center_q.dtype
        )
        effective_weight = float(
            (support_stack * weights.unsqueeze(0)).sum().item()
        )
        distance = support_aware_q_distance(
            center_q.unsqueeze(0),
            source_q.unsqueeze(0),
            center_support.unsqueeze(0),
            source_support.unsqueeze(0),
            weights,
            eps=_EPS,
        )
        if not bool(distance.valid[0, 0].item()):
            centers.append(
                TargetShapeClassCenter(
                    class_id=class_id,
                    center_q=center_q.detach(),
                    center_support=center_support.detach(),
                    sample_count=sample_count,
                    effective_weight=effective_weight,
                    source_distance=float("nan"),
                    residual_q=torch.zeros_like(center_q).detach(),
                    valid=False,
                    reject_reason="no_common_source_support",
                )
            )
            continue
        residual = center_q - source_q.to(device=center_q.device, dtype=center_q.dtype)
        centers.append(
            TargetShapeClassCenter(
                class_id=class_id,
                center_q=center_q.detach(),
                center_support=center_support.detach(),
                sample_count=sample_count,
                effective_weight=effective_weight,
                source_distance=float(distance.distance[0, 0].item()),
                residual_q=residual.detach(),
                valid=True,
                reject_reason=None,
            )
        )
    return tuple(centers)


def _unavailable_state(
    *,
    scan_index: int,
    class_centers: tuple[TargetShapeClassCenter, ...],
    valid_classes: tuple[int, ...],
) -> DomainShapeState:
    return DomainShapeState(
        scan_index=scan_index,
        status=DomainShapeStatus.UNAVAILABLE,
        class_centers=class_centers,
        valid_classes=valid_classes,
        delta=None,
        interactions=(),
        rho_shape=None,
        leave_one_out_drift=None,
        center_drift=None,
        confirmation_age=0,
    )


def update_domain_shape_state(
    stable_label_scan: StableTargetLabelScanResult,
    source_prototype_bank: SourcePrototypeBank,
    config: DomainShapeConfig,
    *,
    previous_state: DomainShapeState | None = None,
) -> DomainShapeState:
    """Estimate and gate the shared Domain Shape effect ``Delta``.

    Only ``StableTargetLabel.aligned_q_shape`` and source Shape prototypes are
    consumed. Stable target confidence is intentionally ignored as an
    aggregation weight, so classes are equally weighted when estimating
    ``Delta`` regardless of target pseudo-label frequency.
    """
    if not isinstance(stable_label_scan, StableTargetLabelScanResult):
        raise TypeError("stable_label_scan must be a StableTargetLabelScanResult")
    if not isinstance(source_prototype_bank, SourcePrototypeBank):
        raise TypeError("source_prototype_bank must be a SourcePrototypeBank")
    if not isinstance(config, DomainShapeConfig):
        raise TypeError("config must be a DomainShapeConfig")
    if previous_state is not None and not isinstance(previous_state, DomainShapeState):
        raise TypeError("previous_state must be a DomainShapeState or None")

    scan_index = 0 if previous_state is None else previous_state.scan_index + 1
    class_centers = _build_class_centers(
        stable_label_scan, source_prototype_bank, config
    )
    valid_centers = tuple(center for center in class_centers if center.valid)
    valid_classes = tuple(center.class_id for center in valid_centers)
    min_classes = max(2, config.shape_min_valid_classes)
    if len(valid_centers) < min_classes:
        return _unavailable_state(
            scan_index=scan_index,
            class_centers=class_centers,
            valid_classes=valid_classes,
        )

    residuals = torch.stack([center.residual_q for center in valid_centers])
    delta = residuals.mean(dim=0)
    interactions_tensor = residuals - delta.unsqueeze(0)
    interactions = tuple(row.detach() for row in interactions_tensor)
    weights = _integration_weights(
        delta.shape[0], device=delta.device, dtype=delta.dtype
    )
    delta_energy = _l2_energy(delta, weights)
    interaction_energy = torch.stack(
        [_l2_energy(row, weights) for row in interactions_tensor]
    ).mean()
    rho_shape = float(
        (delta_energy / (delta_energy + interaction_energy + _EPS)).item()
    )

    loo_drifts: list[Tensor] = []
    for index in range(residuals.shape[0]):
        keep = torch.ones(residuals.shape[0], dtype=torch.bool, device=residuals.device)
        keep[index] = False
        delta_without = residuals[keep].mean(dim=0)
        loo_drifts.append(_l2_norm(delta_without - delta, weights))
    leave_one_out_drift = float(torch.stack(loo_drifts).max().item())
    effect_norm = float(_l2_norm(delta, weights).item())

    center_drift: float | None = None
    if previous_state is not None and previous_state.delta is not None:
        previous_delta = previous_state.delta.to(device=delta.device, dtype=delta.dtype)
        if previous_delta.shape != delta.shape:
            raise ValueError("previous Domain Shape delta shape does not match current grid")
        center_drift = float(_l2_norm(delta - previous_delta, weights).item())

    passes = (
        rho_shape >= config.shape_shared_ratio_min
        and leave_one_out_drift <= config.shape_leave_one_out_drift_max
        and effect_norm <= config.shape_effect_norm_max
        and (
            center_drift is None
            or center_drift <= config.shape_center_drift_max
        )
    )
    if not passes:
        return DomainShapeState(
            scan_index=scan_index,
            status=DomainShapeStatus.REJECTED,
            class_centers=class_centers,
            valid_classes=valid_classes,
            delta=delta.detach(),
            interactions=interactions,
            rho_shape=rho_shape,
            leave_one_out_drift=leave_one_out_drift,
            center_drift=center_drift,
            confirmation_age=0,
        )

    stable_previous = (
        previous_state is not None
        and previous_state.status
        in (DomainShapeStatus.PROVISIONAL, DomainShapeStatus.CONFIRMED)
        and previous_state.delta is not None
    )
    confirmation_age = previous_state.confirmation_age + 1 if stable_previous else 1
    status = (
        DomainShapeStatus.CONFIRMED
        if confirmation_age >= config.shape_confirmation_patience
        else DomainShapeStatus.PROVISIONAL
    )
    return DomainShapeState(
        scan_index=scan_index,
        status=status,
        class_centers=class_centers,
        valid_classes=valid_classes,
        delta=delta.detach(),
        interactions=interactions,
        rho_shape=rho_shape,
        leave_one_out_drift=leave_one_out_drift,
        center_drift=center_drift,
        confirmation_age=confirmation_age,
    )
