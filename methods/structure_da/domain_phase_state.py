"""Robust class phase centers and M=0/1/2 domain phase grouping."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

import torch
from torch import Tensor

from .phase_geometry import (
    pairwise_phase_distances,
    phase_distance,
    sqrt_mean_gamma,
    sqrt_median_gamma,
)
from .target_hypothesis_scan import (
    PairwiseClassAlignment,
    TargetHypothesisScanResult,
)


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
    # Identity evidence is deliberately disabled until calibration supplies
    # explicit numerical tolerances.  ``None`` must never be interpreted as
    # evidence for identity.
    phase_identity_radius: float | None = None
    phase_identity_gain_ratio_min: float | None = None

    def __post_init__(self) -> None:
        if self.phase_group_cap != 2:
            raise ValueError("phase_group_cap must equal 2")
        if self.phase_min_classes_per_group != 2:
            raise ValueError("phase_min_classes_per_group must equal 2")
        if self.phase_confirmation_patience < 2:
            raise ValueError("phase_confirmation_patience must be at least 2")
        if (self.phase_identity_radius is None) != (self.phase_identity_gain_ratio_min is None):
            raise ValueError(
                "phase_identity_radius and phase_identity_gain_ratio_min must both be set or both be None"
            )
        thresholds = {
            "phase_min_samples_per_class": self.phase_min_samples_per_class,
            "phase_class_dispersion_max": self.phase_class_dispersion_max,
            "phase_class_diameter_max": self.phase_class_diameter_max,
            "phase_group_dispersion_max": self.phase_group_dispersion_max,
            "phase_group_diameter_max": self.phase_group_diameter_max,
            "phase_group_core_separation": self.phase_group_core_separation,
            "phase_global_radius": self.phase_global_radius,
            "phase_center_drift_max": self.phase_center_drift_max,
            **(
                {}
                if self.phase_identity_radius is None
                else {
                    "phase_identity_radius": self.phase_identity_radius,
                    "phase_identity_gain_ratio_min": self.phase_identity_gain_ratio_min,
                }
            ),
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


class PhaseDecisionStatus(str, Enum):
    """Domain-level decision, distinct from non-identity group model order M."""

    UNCONFIRMED = "unconfirmed"
    IDENTITY_CONFIRMED = "identity_confirmed"
    NONIDENTITY_CONFIRMED = "nonidentity_confirmed"


class CandidatePhaseCompatibilityStatus(str, Enum):
    NO_CONFIRMED_GROUP = "no_confirmed_group"
    UNUSABLE = "unusable"
    COMPATIBLE = "compatible"
    RESIDUAL = "residual"


@dataclass(frozen=True)
class CandidatePhaseCompatibility:
    sample_id: int
    class_id: int
    status: CandidatePhaseCompatibilityStatus
    assigned_group_id: int | None
    nearest_group_id: int | None
    phase_distance_to_group: float | None
    gamma: Tensor | None


@dataclass(frozen=True)
class ResidualPhaseEvidence:
    sample_id: int
    class_id: int
    gamma: Tensor
    nearest_group_id: int | None
    phase_distance_to_group: float


@dataclass(frozen=True)
class ResidualPhaseGroupCandidate:
    member_classes: tuple[int, ...]
    center_gamma: Tensor
    sample_evidence_count: float
    within_dispersion: float
    diameter: float


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
    decision_status: PhaseDecisionStatus = PhaseDecisionStatus.UNCONFIRMED
    decision_stability_age: int = 0
    identity_evidence_classes: tuple[int, ...] = ()
    identity_evidence_count: float = 0.0
    residual_evidence_count: int = 0
    residual_evidence_classes: tuple[int, ...] = ()


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


def _alignment_lookup(
    scan_result: TargetHypothesisScanResult,
) -> dict[tuple[int, int], PairwiseClassAlignment]:
    return {
        (int(item.sample_id), int(item.class_id)): item
        for item in scan_result.pairwise_alignments
    }


def _identity_evidence_summary(
    scan_result: TargetHypothesisScanResult,
    config: DomainPhaseConfig,
) -> tuple[tuple[int, ...], float]:
    """Collect explicit identity evidence from geometry-first primary labels.

    Identity is disabled unless both calibrated tolerances are supplied.  A
    missing non-identity group is therefore never reinterpreted as identity.
    """
    if (
        config.phase_identity_radius is None
        or config.phase_identity_gain_ratio_min is None
    ):
        return (), 0.0
    lookup = _alignment_lookup(scan_result)
    counts: dict[int, float] = {}
    for candidate in scan_result.candidate_pseudo_labels:
        alignment = lookup.get((int(candidate.sample_id), int(candidate.class_id)))
        if (
            alignment is None
            or alignment.gamma is None
            or not alignment.numerically_valid
            or alignment.t_gain_ratio is None
            or not math.isfinite(float(alignment.t_gain_ratio))
            or alignment.q_shape_distance is None
            or not math.isfinite(float(alignment.q_shape_distance))
        ):
            continue
        identity = _identity_like(alignment.gamma)
        distance = float(phase_distance(alignment.gamma, identity).item())
        if (
            distance <= float(config.phase_identity_radius)
            and float(alignment.t_gain_ratio)
            >= float(config.phase_identity_gain_ratio_min)
        ):
            class_id = int(candidate.class_id)
            counts[class_id] = counts.get(class_id, 0.0) + 1.0
    valid_classes = tuple(
        sorted(
            class_id
            for class_id, count in counts.items()
            if count >= config.phase_min_samples_per_class
        )
    )
    total = float(sum(counts[class_id] for class_id in valid_classes))
    return valid_classes, total


def _confirmed_groups(state: DomainPhaseState) -> tuple[PhaseGroup, ...]:
    return tuple(
        group for group in state.groups if group.status is PhaseGroupStatus.CONFIRMED
    )


@torch.no_grad()
def evaluate_sample_class_phase_compatibility(
    *,
    sample_id: int,
    class_id: int,
    alignment: PairwiseClassAlignment | None,
    state: DomainPhaseState,
    config: DomainPhaseConfig,
    require_phase_evidence_eligible: bool = True,
) -> CandidatePhaseCompatibility:
    """Compare one cached sample/class gamma with the confirmed Domain Phase.

    ``require_phase_evidence_eligible=True`` is the conservative Round-B mode
    used when collecting evidence capable of changing the Domain Phase model.
    Stable-label validation may set it to ``False``: once Domain Phase has been
    confirmed, a numerically valid candidate gamma can be checked against the
    group center even if that sample was not reliable enough to *estimate* the
    domain-level Phase in the first place.
    """
    sample_id = int(sample_id)
    class_id = int(class_id)
    gamma = None if alignment is None else alignment.gamma
    if state.decision_status is PhaseDecisionStatus.IDENTITY_CONFIRMED:
        if (
            alignment is None
            or gamma is None
            or not alignment.numerically_valid
            or config.phase_identity_radius is None
        ):
            return CandidatePhaseCompatibility(
                sample_id=sample_id,
                class_id=class_id,
                status=CandidatePhaseCompatibilityStatus.UNUSABLE,
                assigned_group_id=None,
                nearest_group_id=None,
                phase_distance_to_group=None,
                gamma=None if gamma is None else gamma.detach().cpu().double(),
            )
        identity = _identity_like(gamma)
        distance = float(phase_distance(gamma, identity).item())
        compatible = distance <= float(config.phase_identity_radius)
        return CandidatePhaseCompatibility(
            sample_id=sample_id,
            class_id=class_id,
            status=(
                CandidatePhaseCompatibilityStatus.COMPATIBLE
                if compatible
                else CandidatePhaseCompatibilityStatus.RESIDUAL
            ),
            assigned_group_id=-1 if compatible else None,
            nearest_group_id=-1,
            phase_distance_to_group=distance,
            gamma=gamma.detach().cpu().double(),
        )

    confirmed = _confirmed_groups(state)
    if not confirmed:
        return CandidatePhaseCompatibility(
            sample_id=sample_id,
            class_id=class_id,
            status=CandidatePhaseCompatibilityStatus.NO_CONFIRMED_GROUP,
            assigned_group_id=None,
            nearest_group_id=None,
            phase_distance_to_group=None,
            gamma=None if gamma is None else gamma.detach().cpu().double(),
        )
    if (
        alignment is None
        or gamma is None
        or not alignment.numerically_valid
        or (require_phase_evidence_eligible and not alignment.phase_evidence_eligible)
    ):
        return CandidatePhaseCompatibility(
            sample_id=sample_id,
            class_id=class_id,
            status=CandidatePhaseCompatibilityStatus.UNUSABLE,
            assigned_group_id=None,
            nearest_group_id=None,
            phase_distance_to_group=None,
            gamma=None if gamma is None else gamma.detach().cpu().double(),
        )

    assigned = next(
        (group for group in confirmed if class_id in group.member_classes),
        None,
    )
    distances = [
        (float(phase_distance(gamma, group.center_gamma).item()), group)
        for group in confirmed
    ]
    _nearest_distance, nearest = min(distances, key=lambda item: item[0])
    comparison_group = assigned or nearest
    comparison_distance = float(
        phase_distance(gamma, comparison_group.center_gamma).item()
    )
    compatible = comparison_distance <= config.phase_group_diameter_max
    return CandidatePhaseCompatibility(
        sample_id=sample_id,
        class_id=class_id,
        status=(
            CandidatePhaseCompatibilityStatus.COMPATIBLE
            if compatible
            else CandidatePhaseCompatibilityStatus.RESIDUAL
        ),
        assigned_group_id=None if assigned is None else assigned.group_id,
        nearest_group_id=nearest.group_id,
        phase_distance_to_group=comparison_distance,
        gamma=gamma.detach().cpu().double(),
    )


@torch.no_grad()
def evaluate_candidate_phase_compatibility(
    scan_result: TargetHypothesisScanResult,
    state: DomainPhaseState,
    config: DomainPhaseConfig,
) -> tuple[CandidatePhaseCompatibility, ...]:
    """Compare each primary candidate gamma with confirmed Domain Phase groups.

    This Round-B evidence path keeps ``phase_evidence_eligible`` as a hard
    requirement because residuals may alter the Domain Phase model itself.
    Round-C stable-label validation calls the single-pair helper above with the
    weaker, post-confirmation semantics.
    """
    lookup = _alignment_lookup(scan_result)
    return tuple(
        evaluate_sample_class_phase_compatibility(
            sample_id=int(candidate.sample_id),
            class_id=int(candidate.class_id),
            alignment=lookup.get((int(candidate.sample_id), int(candidate.class_id))),
            state=state,
            config=config,
            require_phase_evidence_eligible=True,
        )
        for candidate in scan_result.candidate_pseudo_labels
    )


def collect_residual_phase_evidence(
    compatibility: tuple[CandidatePhaseCompatibility, ...],
) -> tuple[ResidualPhaseEvidence, ...]:
    result = []
    for item in compatibility:
        if (
            item.status is CandidatePhaseCompatibilityStatus.RESIDUAL
            and item.gamma is not None
            and item.phase_distance_to_group is not None
        ):
            result.append(
                ResidualPhaseEvidence(
                    sample_id=item.sample_id,
                    class_id=item.class_id,
                    gamma=item.gamma.detach().cpu().double(),
                    nearest_group_id=item.nearest_group_id,
                    phase_distance_to_group=float(item.phase_distance_to_group),
                )
            )
    return tuple(result)


def detect_residual_phase_group(
    compatibility: tuple[CandidatePhaseCompatibility, ...],
    state: DomainPhaseState,
    config: DomainPhaseConfig,
) -> ResidualPhaseGroupCandidate | None:
    """Return a supported residual group, never one produced by a lone outlier."""
    residuals = collect_residual_phase_evidence(compatibility)
    by_class: dict[int, list[ResidualPhaseEvidence]] = {}
    for item in residuals:
        by_class.setdefault(int(item.class_id), []).append(item)

    centers: list[PhaseClassCenter] = []
    for class_id in sorted(by_class):
        evidence = by_class[class_id]
        if len(evidence) < config.phase_min_samples_per_class:
            continue
        gammas = torch.stack([item.gamma for item in evidence])
        center = sqrt_median_gamma(gammas)
        distances = torch.stack([phase_distance(gamma, center) for gamma in gammas])
        pairwise = pairwise_phase_distances(gammas)
        dispersion = float(distances.square().mean().item())
        diameter = float(pairwise.max().item())
        if (
            dispersion > config.phase_class_dispersion_max
            or diameter > config.phase_class_diameter_max
        ):
            continue
        centers.append(
            PhaseClassCenter(
                class_id=class_id,
                center_gamma=center.detach(),
                candidate_count=len(evidence),
                effective_evidence_count=float(len(evidence)),
                dispersion=dispersion,
                diameter=diameter,
                median_distance=float(distances.median().item()),
                center_drift=None,
                valid=True,
                reject_reason=None,
            )
        )
    if len(centers) < config.phase_min_classes_per_group:
        return None
    center_tuple = tuple(centers)
    valid_indices = tuple(range(len(center_tuple)))
    candidate = _try_m1(valid_indices, center_tuple, config)
    if candidate is None:
        return None
    group, _outliers = candidate
    confirmed = _confirmed_groups(state)
    for existing in confirmed:
        separation = float(
            phase_distance(group.center_gamma, existing.center_gamma).item()
        )
        if not (
            separation >= config.phase_group_core_separation
            and separation > group.core_radius + existing.core_radius
        ):
            return None
    member_classes = tuple(
        sorted(center_tuple[index].class_id for index in group.member_indices)
    )
    evidence_count = float(
        sum(center_tuple[index].effective_evidence_count for index in group.member_indices)
    )
    return ResidualPhaseGroupCandidate(
        member_classes=member_classes,
        center_gamma=group.center_gamma.detach(),
        sample_evidence_count=evidence_count,
        within_dispersion=group.within_dispersion,
        diameter=group.diameter,
    )


def _model_signature(groups: tuple[PhaseGroup, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(group.member_classes) for group in groups))


def _decision_stability_age(
    groups: tuple[PhaseGroup, ...],
    previous_state: DomainPhaseState | None,
    *,
    identity_classes: tuple[int, ...],
) -> int:
    if groups:
        if (
            previous_state is not None
            and previous_state.groups
            and _model_signature(groups) == _model_signature(previous_state.groups)
            and all(
                group.center_drift is not None
                for group in groups
            )
        ):
            return previous_state.decision_stability_age + 1
        return 1
    if identity_classes:
        if (
            previous_state is not None
            and previous_state.m == 0
            and previous_state.identity_evidence_classes == identity_classes
        ):
            return previous_state.decision_stability_age + 1
        return 1
    return 0


def _build_grouped_state(
    *,
    scan_index: int,
    class_centers: tuple[PhaseClassCenter, ...],
    valid_classes: tuple[int, ...],
    invalid_classes: set[int],
    valid_indices: tuple[int, ...],
    config: DomainPhaseConfig,
    previous_state: DomainPhaseState | None,
) -> DomainPhaseState:
    # M=2 is evaluated before M=1.  The existing separation/core constraints
    # are the structural guard against splitting one coherent group merely
    # because two clusters always fit more tightly than one.
    m2 = _try_m2(valid_indices, class_centers, config) if len(valid_indices) >= 4 else None
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

    return DomainPhaseState(
        scan_index=scan_index,
        m=0,
        class_centers=class_centers,
        valid_phase_classes=valid_classes,
        groups=(),
        rejected_classes=tuple(sorted(center.class_id for center in class_centers)),
    )


@torch.no_grad()
def update_domain_phase_state(
    scan_result: TargetHypothesisScanResult,
    config: DomainPhaseConfig,
    previous_state: DomainPhaseState | None = None,
) -> DomainPhaseState:
    """Update group model and explicit domain-level Phase decision state."""
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
        state = DomainPhaseState(
            scan_index=scan_index,
            m=0,
            class_centers=class_centers,
            valid_phase_classes=valid_classes,
            groups=(),
            rejected_classes=tuple(sorted(center.class_id for center in class_centers)),
        )
    else:
        state = _build_grouped_state(
            scan_index=scan_index,
            class_centers=class_centers,
            valid_classes=valid_classes,
            invalid_classes=invalid_classes,
            valid_indices=valid_indices,
            config=config,
            previous_state=previous_state,
        )

    identity_classes, identity_count = _identity_evidence_summary(scan_result, config)
    stability_age = _decision_stability_age(
        state.groups,
        previous_state,
        identity_classes=identity_classes,
    )
    provisional = replace(
        state,
        decision_stability_age=stability_age,
        identity_evidence_classes=identity_classes,
        identity_evidence_count=identity_count,
    )

    compatibility = evaluate_candidate_phase_compatibility(
        scan_result,
        provisional,
        config,
    )
    residuals = collect_residual_phase_evidence(compatibility)
    residual_group = detect_residual_phase_group(
        compatibility,
        provisional,
        config,
    )
    residual_classes = tuple(sorted({item.class_id for item in residuals}))

    decision = PhaseDecisionStatus.UNCONFIRMED
    if provisional.m > 0:
        all_confirmed = bool(provisional.groups) and all(
            group.status is PhaseGroupStatus.CONFIRMED
            for group in provisional.groups
        )
        if (
            all_confirmed
            and stability_age >= config.phase_confirmation_patience
            and residual_group is None
        ):
            decision = PhaseDecisionStatus.NONIDENTITY_CONFIRMED
    elif (
        len(identity_classes) >= config.phase_min_classes_per_group
        and stability_age >= config.phase_confirmation_patience
    ):
        decision = PhaseDecisionStatus.IDENTITY_CONFIRMED

    return replace(
        provisional,
        decision_status=decision,
        residual_evidence_count=len(residuals),
        residual_evidence_classes=residual_classes,
    )
