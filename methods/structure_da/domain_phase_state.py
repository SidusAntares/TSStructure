"""Robust class phase centers and M=0/1/2 domain phase grouping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from .phase_geometry import (
    pairwise_phase_distances,
    phase_distance,
    sqrt_mean_gamma,
    sqrt_median_gamma,
)
from .target_hypothesis_scan import TargetHypothesisScanResult


@dataclass(frozen=True)
class DomainPhaseConfig:
    phase_min_samples_per_class: float
    phase_class_dispersion_max: float
    phase_class_diameter_max: float
    phase_group_dispersion_max: float
    phase_group_diameter_max: float
    phase_group_core_separation: float
    phase_global_radius: float
    phase_confirmation_patience: int
    phase_center_drift_max: float
    phase_group_cap: int = 2
    phase_min_classes_per_group: int = 2

    def __post_init__(self) -> None:
        if self.phase_group_cap != 2:
            raise ValueError("phase_group_cap must equal 2")
        if self.phase_min_classes_per_group != 2:
            raise ValueError("phase_min_classes_per_group must equal 2")
        if self.phase_confirmation_patience < 2:
            raise ValueError("phase_confirmation_patience must be at least 2")
        thresholds = {
            "phase_min_samples_per_class": self.phase_min_samples_per_class,
            "phase_class_dispersion_max": self.phase_class_dispersion_max,
            "phase_class_diameter_max": self.phase_class_diameter_max,
            "phase_group_dispersion_max": self.phase_group_dispersion_max,
            "phase_group_diameter_max": self.phase_group_diameter_max,
            "phase_group_core_separation": self.phase_group_core_separation,
            "phase_global_radius": self.phase_global_radius,
            "phase_center_drift_max": self.phase_center_drift_max,
        }
        for name, value in thresholds.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class PhaseClassCenter:
    class_id: int
    center_gamma: Tensor
    candidate_count: int
    effective_evidence_count: float
    dispersion: float
    diameter: float
    median_distance: float
    center_drift: float | None
    valid: bool
    reject_reason: str | None


class PhaseGroupStatus(str, Enum):
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class PhaseGroup:
    group_id: int
    member_classes: tuple[int, ...]
    center_gamma: Tensor
    within_dispersion: float
    diameter: float
    core_radius: float
    sample_evidence_count: float
    class_count: int
    center_drift: float | None
    status: PhaseGroupStatus
    confirmation_age: int


@dataclass(frozen=True)
class DomainPhaseState:
    scan_index: int
    m: int
    class_centers: tuple[PhaseClassCenter, ...]
    valid_phase_classes: tuple[int, ...]
    groups: tuple[PhaseGroup, ...]
    rejected_classes: tuple[int, ...]


@dataclass(frozen=True)
class _GroupCandidate:
    member_indices: tuple[int, ...]
    center_gamma: Tensor
    within_dispersion: float
    diameter: float
    core_radius: float


def _previous_class_centers(
    previous_state: DomainPhaseState | None,
) -> dict[int, PhaseClassCenter]:
    if previous_state is None:
        return {}
    return {center.class_id: center for center in previous_state.class_centers}


def _class_reject_reason(
    *,
    evidence: float,
    dispersion: float,
    diameter: float,
    center_drift: float | None,
    config: DomainPhaseConfig,
) -> str | None:
    if evidence < config.phase_min_samples_per_class:
        return "insufficient_evidence"
    if dispersion > config.phase_class_dispersion_max:
        return "class_dispersion_exceeded"
    if diameter > config.phase_class_diameter_max:
        return "class_diameter_exceeded"
    if center_drift is not None and center_drift > config.phase_center_drift_max:
        return "class_center_drift_exceeded"
    return None


def _build_class_centers(
    scan_result: TargetHypothesisScanResult,
    config: DomainPhaseConfig,
    previous_state: DomainPhaseState | None,
) -> tuple[PhaseClassCenter, ...]:
    by_class: dict[int, list] = {}
    for hypothesis in scan_result.hypotheses:
        by_class.setdefault(int(hypothesis.class_id), []).append(hypothesis)
    previous = _previous_class_centers(previous_state)
    centers: list[PhaseClassCenter] = []
    for class_id in sorted(by_class):
        hypotheses = by_class[class_id]
        gammas = torch.stack(
            [hypothesis.gamma.detach().cpu().double() for hypothesis in hypotheses]
        )
        center = sqrt_median_gamma(gammas)
        distances = torch.stack(
            [phase_distance(gamma, center) for gamma in gammas]
        )
        pairwise = pairwise_phase_distances(gammas)
        evidence = float(sum(float(item.evidence_weight) for item in hypotheses))
        dispersion = float(distances.square().mean().item())
        diameter = float(pairwise.max().item())
        median_distance = float(distances.median().item())
        prior = previous.get(class_id)
        center_drift = (
            float(phase_distance(center, prior.center_gamma).item())
            if prior is not None
            else None
        )
        reason = _class_reject_reason(
            evidence=evidence,
            dispersion=dispersion,
            diameter=diameter,
            center_drift=center_drift,
            config=config,
        )
        centers.append(
            PhaseClassCenter(
                class_id=class_id,
                center_gamma=center.detach(),
                candidate_count=len(hypotheses),
                effective_evidence_count=evidence,
                dispersion=dispersion,
                diameter=diameter,
                median_distance=median_distance,
                center_drift=center_drift,
                valid=reason is None,
                reject_reason=reason,
            )
        )
    return tuple(centers)


def _group_metrics(
    member_indices: tuple[int, ...], class_centers: tuple[PhaseClassCenter, ...]
) -> _GroupCandidate:
    gammas = torch.stack([class_centers[index].center_gamma for index in member_indices])
    center = sqrt_mean_gamma(gammas)
    distances = torch.stack([phase_distance(gamma, center) for gamma in gammas])
    diameter = float(pairwise_phase_distances(gammas).max().item())
    return _GroupCandidate(
        member_indices=member_indices,
        center_gamma=center.detach(),
        within_dispersion=float(distances.square().mean().item()),
        diameter=diameter,
        core_radius=float(distances.max().item()),
    )


def _identity_like(gamma: Tensor) -> Tensor:
    return torch.linspace(0.0, 1.0, gamma.numel(), dtype=torch.float64)


def _group_is_feasible(group: _GroupCandidate, config: DomainPhaseConfig) -> bool:
    return (
        len(group.member_indices) >= config.phase_min_classes_per_group
        and group.within_dispersion <= config.phase_group_dispersion_max
        and group.diameter <= config.phase_group_diameter_max
        and float(phase_distance(group.center_gamma, _identity_like(group.center_gamma)).item())
        <= config.phase_global_radius
    )


def _try_m1(
    valid_indices: tuple[int, ...],
    class_centers: tuple[PhaseClassCenter, ...],
    config: DomainPhaseConfig,
) -> tuple[_GroupCandidate, tuple[int, ...]] | None:
    gammas = torch.stack([class_centers[index].center_gamma for index in valid_indices])
    robust_center = sqrt_median_gamma(gammas)
    core = tuple(
        index
        for index in valid_indices
        if float(phase_distance(class_centers[index].center_gamma, robust_center).item())
        <= config.phase_group_diameter_max
    )
    if len(core) < config.phase_min_classes_per_group:
        return None
    group = _group_metrics(core, class_centers)
    if not _group_is_feasible(group, config):
        return None
    outliers = tuple(index for index in valid_indices if index not in core)
    return group, outliers


def _farthest_pair(
    valid_indices: tuple[int, ...], class_centers: tuple[PhaseClassCenter, ...]
) -> tuple[int, int]:
    best_pair = (valid_indices[0], valid_indices[1])
    best_distance = -1.0
    for position, left in enumerate(valid_indices):
        for right in valid_indices[position + 1 :]:
            distance = float(
                phase_distance(
                    class_centers[left].center_gamma,
                    class_centers[right].center_gamma,
                ).item()
            )
            if distance > best_distance:
                best_distance = distance
                best_pair = (left, right)
    return best_pair


def _try_m2(
    valid_indices: tuple[int, ...],
    class_centers: tuple[PhaseClassCenter, ...],
    config: DomainPhaseConfig,
) -> tuple[tuple[_GroupCandidate, _GroupCandidate], tuple[int, ...]] | None:
    if len(valid_indices) < 4:
        return None
    seed_left, seed_right = _farthest_pair(valid_indices, class_centers)
    centers = (
        class_centers[seed_left].center_gamma,
        class_centers[seed_right].center_gamma,
    )
    previous_membership: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    membership: tuple[tuple[int, ...], tuple[int, ...]]
    for _ in range(50):
        assigned = [[], []]
        for index in valid_indices:
            gamma = class_centers[index].center_gamma
            distances = (
                float(phase_distance(gamma, centers[0]).item()),
                float(phase_distance(gamma, centers[1]).item()),
            )
            assigned[0 if distances[0] <= distances[1] else 1].append(index)
        membership = (tuple(assigned[0]), tuple(assigned[1]))
        if not membership[0] or not membership[1]:
            return None
        if membership == previous_membership:
            break
        centers = tuple(
            sqrt_mean_gamma(
                torch.stack([class_centers[index].center_gamma for index in members])
            )
            for members in membership
        )
        previous_membership = membership

    retained: list[tuple[int, ...]] = []
    rejected: list[int] = []
    for group_index, members in enumerate(membership):
        kept = tuple(
            index
            for index in members
            if float(
                phase_distance(class_centers[index].center_gamma, centers[group_index]).item()
            )
            <= config.phase_group_diameter_max
        )
        rejected.extend(index for index in members if index not in kept)
        retained.append(kept)
    if any(len(members) < config.phase_min_classes_per_group for members in retained):
        return None
    groups = tuple(_group_metrics(members, class_centers) for members in retained)
    if not all(_group_is_feasible(group, config) for group in groups):
        return None
    separation = float(phase_distance(groups[0].center_gamma, groups[1].center_gamma).item())
    if not (
        separation > groups[0].core_radius + groups[1].core_radius
        and separation >= config.phase_group_core_separation
    ):
        return None
    ordered = tuple(
        sorted(groups, key=lambda group: min(class_centers[i].class_id for i in group.member_indices))
    )
    return (ordered[0], ordered[1]), tuple(sorted(rejected))


def _materialize_groups(
    candidates: tuple[_GroupCandidate, ...],
    class_centers: tuple[PhaseClassCenter, ...],
    previous_state: DomainPhaseState | None,
    config: DomainPhaseConfig,
) -> tuple[PhaseGroup, ...]:
    previous_by_members = {}
    if previous_state is not None and previous_state.m == len(candidates):
        previous_by_members = {
            group.member_classes: group for group in previous_state.groups
        }
    groups: list[PhaseGroup] = []
    for group_id, candidate in enumerate(candidates):
        members = tuple(sorted(class_centers[index].class_id for index in candidate.member_indices))
        prior = previous_by_members.get(members)
        drift = (
            float(phase_distance(candidate.center_gamma, prior.center_gamma).item())
            if prior is not None
            else None
        )
        stable = prior is not None and drift <= config.phase_center_drift_max
        age = prior.confirmation_age + 1 if stable else 1
        status = (
            PhaseGroupStatus.CONFIRMED
            if stable and age >= config.phase_confirmation_patience
            else PhaseGroupStatus.PROVISIONAL
        )
        groups.append(
            PhaseGroup(
                group_id=group_id,
                member_classes=members,
                center_gamma=candidate.center_gamma.detach(),
                within_dispersion=candidate.within_dispersion,
                diameter=candidate.diameter,
                core_radius=candidate.core_radius,
                sample_evidence_count=float(
                    sum(
                        class_centers[index].effective_evidence_count
                        for index in candidate.member_indices
                    )
                ),
                class_count=len(candidate.member_indices),
                center_drift=drift,
                status=status,
                confirmation_age=age,
            )
        )
    return tuple(groups)


@torch.no_grad()
def update_domain_phase_state(
    scan_result: TargetHypothesisScanResult,
    config: DomainPhaseConfig,
    previous_state: DomainPhaseState | None = None,
) -> DomainPhaseState:
    """Build robust phase classes, M=0/1/2 groups and confirmation state."""
    if not isinstance(scan_result, TargetHypothesisScanResult):
        raise TypeError("scan_result must be a TargetHypothesisScanResult")
    class_centers = _build_class_centers(scan_result, config, previous_state)
    valid_indices = tuple(
        index for index, center in enumerate(class_centers) if center.valid
    )
    valid_classes = tuple(class_centers[index].class_id for index in valid_indices)
    scan_index = 0 if previous_state is None else previous_state.scan_index + 1
    invalid_classes = {
        center.class_id for center in class_centers if not center.valid
    }
    if len(valid_indices) < config.phase_min_classes_per_group:
        return DomainPhaseState(
            scan_index=scan_index,
            m=0,
            class_centers=class_centers,
            valid_phase_classes=valid_classes,
            groups=(),
            rejected_classes=tuple(sorted(center.class_id for center in class_centers)),
        )

    m1 = _try_m1(valid_indices, class_centers, config)
    if m1 is not None:
        candidate, outlier_indices = m1
        groups = _materialize_groups((candidate,), class_centers, previous_state, config)
        rejected = invalid_classes | {
            class_centers[index].class_id for index in outlier_indices
        }
        return DomainPhaseState(
            scan_index=scan_index,
            m=1,
            class_centers=class_centers,
            valid_phase_classes=valid_classes,
            groups=groups,
            rejected_classes=tuple(sorted(rejected)),
        )

    m2 = _try_m2(valid_indices, class_centers, config)
    if m2 is not None:
        candidates, outlier_indices = m2
        groups = _materialize_groups(candidates, class_centers, previous_state, config)
        rejected = invalid_classes | {
            class_centers[index].class_id for index in outlier_indices
        }
        active = {class_id for group in groups for class_id in group.member_classes}
        rejected.update(set(valid_classes) - active)
        return DomainPhaseState(
            scan_index=scan_index,
            m=2,
            class_centers=class_centers,
            valid_phase_classes=valid_classes,
            groups=groups,
            rejected_classes=tuple(sorted(rejected)),
        )

    return DomainPhaseState(
        scan_index=scan_index,
        m=0,
        class_centers=class_centers,
        valid_phase_classes=valid_classes,
        groups=(),
        rejected_classes=tuple(sorted(center.class_id for center in class_centers)),
    )
