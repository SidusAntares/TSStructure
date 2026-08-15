"""Pure control-flow helpers for experiment 14 shared Domain Phase iteration.

The project convention is ``gamma(u_source) = u_target``.  A shared Domain
Phase is therefore source->target; its inverse is used only for Teacher target
pseudo-label generation.  This module contains no target labels and no model
selection logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .confirmed_phase_view import (
    align_target_positions_to_source,
    map_source_positions_to_target,
)
from .phase_geometry import phase_distance
from .shared_domain_phase_diagnostic import (
    sample_equal_weights,
    weighted_frechet_mean_gamma,
)

PHASE_ARMS = ("NO_PHASE", "STATIC_DOMAIN_PHASE", "ITERATIVE_DOMAIN_PHASE")


@dataclass(frozen=True)
class SharedDomainPhaseEstimate:
    gamma: Tensor
    valid: bool
    participating_classes: tuple[int, ...]
    accepted_count_by_class: tuple[int, ...]
    valid_gamma_count_by_class: tuple[int, ...]
    class_centers: tuple[tuple[int, Tensor], ...]
    invalid_reason: str | None


def identity_phase(k: int = 128) -> Tensor:
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer >=2")
    return torch.linspace(0.0, 1.0, k, dtype=torch.float64)


def validate_phase(gamma: Tensor, *, name: str = "gamma") -> Tensor:
    if not isinstance(gamma, Tensor) or gamma.ndim != 1 or gamma.numel() < 2:
        raise ValueError(f"{name} must have shape [K] with K>=2")
    value = gamma.detach().cpu().double().contiguous()
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")
    if abs(float(value[0])) > 1e-6 or abs(float(value[-1]) - 1.0) > 1e-6:
        raise ValueError(f"{name} must preserve [0,1] endpoints")
    if not torch.all(value[1:] > value[:-1]).item():
        raise ValueError(f"{name} must be strictly increasing")
    return value


def map_source_batch_positions(positions: Tensor, mask: Tensor, gamma: Tensor) -> Tensor:
    """Apply source->target ``gamma`` independently to each batch row."""
    if positions.ndim != 2 or mask.shape != positions.shape:
        raise ValueError("positions/mask must have shape [B,L]")
    rows = [
        map_source_positions_to_target(positions[i], mask[i], gamma)
        for i in range(positions.shape[0])
    ]
    return torch.stack(rows, dim=0) if rows else positions.clone()


def map_target_batch_positions(positions: Tensor, mask: Tensor, gamma: Tensor) -> Tensor:
    """Apply target->source ``gamma^{-1}`` to a batch."""
    return align_target_positions_to_source(positions, mask, gamma)


def phase_distance_value(a: Tensor, b: Tensor) -> float:
    return float(phase_distance(validate_phase(a, name="a"), validate_phase(b, name="b")).item())


def build_shared_domain_phase(
    gammas_by_class: Mapping[int, Sequence[Tensor]],
    *,
    accepted_count_by_class: Mapping[int, int],
    num_classes: int,
    k_reg: int,
) -> SharedDomainPhaseEstimate:
    """Two-stage Fisher--Rao center: within class, then equal weight over classes.

    Only numerically valid registration gammas should be passed in
    ``gammas_by_class``.  Confidence/mask eligibility is represented separately
    by ``accepted_count_by_class`` so numerical solver failures can be audited
    without becoming a hidden Phase-specific reliability gate.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >=2")
    if k_reg < 2:
        raise ValueError("k_reg must be >=2")
    accepted = tuple(int(accepted_count_by_class.get(c, 0)) for c in range(num_classes))
    valid_counts: list[int] = []
    class_centers: list[tuple[int, Tensor]] = []
    for class_id in range(num_classes):
        rows = [validate_phase(g, name=f"gamma[class={class_id}]") for g in gammas_by_class.get(class_id, ())]
        if any(g.numel() != k_reg for g in rows):
            raise ValueError("all Phase observations must share k_reg")
        valid_counts.append(len(rows))
        if not rows:
            continue
        stack = torch.stack(rows, dim=0)
        center = weighted_frechet_mean_gamma(stack, sample_equal_weights(stack.shape[0])).gamma
        class_centers.append((class_id, validate_phase(center, name=f"class_center[{class_id}]")))

    participating = tuple(class_id for class_id, _ in class_centers)
    if len(class_centers) < 2:
        return SharedDomainPhaseEstimate(
            gamma=identity_phase(k_reg),
            valid=False,
            participating_classes=participating,
            accepted_count_by_class=accepted,
            valid_gamma_count_by_class=tuple(valid_counts),
            class_centers=tuple(class_centers),
            invalid_reason="fewer_than_two_nonempty_class_phase_centers",
        )

    centers = torch.stack([center for _, center in class_centers], dim=0)
    shared = weighted_frechet_mean_gamma(centers, sample_equal_weights(centers.shape[0])).gamma
    return SharedDomainPhaseEstimate(
        gamma=validate_phase(shared, name="shared_phase"),
        valid=True,
        participating_classes=participating,
        accepted_count_by_class=accepted,
        valid_gamma_count_by_class=tuple(valid_counts),
        class_centers=tuple(class_centers),
        invalid_reason=None,
    )


def actual_phase_for_arm(
    arm: str,
    *,
    identity: Tensor,
    first_phase: Tensor | None,
    previous_phase: Tensor,
    proposal: SharedDomainPhaseEstimate | None,
    phase_epoch: int,
) -> Tensor:
    """Resolve the Phase used by one arm at a Phase-window boundary.

    ``phase_epoch`` is 1-based for the upcoming Phase-guided epoch.  Invalid
    proposals never replace the previous actual Phase.
    """
    if arm not in PHASE_ARMS:
        raise ValueError(f"unknown Phase arm {arm!r}")
    ident = validate_phase(identity, name="identity")
    previous = validate_phase(previous_phase, name="previous_phase")
    if arm == "NO_PHASE":
        return ident
    if arm == "STATIC_DOMAIN_PHASE":
        if first_phase is not None:
            return validate_phase(first_phase, name="first_phase")
        if proposal is not None and proposal.valid:
            return validate_phase(proposal.gamma, name="proposal")
        return previous
    if phase_epoch < 1:
        raise ValueError("phase_epoch must be >=1")
    if proposal is not None and proposal.valid:
        return validate_phase(proposal.gamma, name="proposal")
    return previous


def class_center_distance_rows(estimate: SharedDomainPhaseEstimate, shared_gamma: Tensor) -> list[dict]:
    shared = validate_phase(shared_gamma, name="shared_gamma")
    rows: list[dict] = []
    center_map = dict(estimate.class_centers)
    for class_id in range(len(estimate.accepted_count_by_class)):
        center = center_map.get(class_id)
        rows.append({
            "class_id": int(class_id),
            "accepted_count": int(estimate.accepted_count_by_class[class_id]),
            "valid_gamma_count": int(estimate.valid_gamma_count_by_class[class_id]),
            "has_class_center": center is not None,
            "distance_class_center_to_shared": (
                float("nan") if center is None else phase_distance_value(center, shared)
            ),
        })
    return rows
