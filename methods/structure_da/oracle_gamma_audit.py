"""Post-hoc helpers for the 07 oracle true-class gamma audit.

This module never solves a registration problem.  It receives gamma values that
were already produced by the 06 oracle true-class Stage-A cache, separates
basic numerical usability from the *current production* Phase-evidence
eligibility rules, and exposes small pure helpers used by the audit script and
unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .sample_phase_diagnostic import RawShapeValidation, TOnlyPhaseRegistration
from .target_hypothesis_scan import PhaseHypothesisScanConfig


PRODUCTION_RULE_ORDER = (
    "pre_support",
    "gamma_increment",
    "gamma_speed",
    "gamma_roughness",
    "gamma_deviation",
    "gain_unavailable",
    "gain",
    "shape_support",
    "q_cdf_unavailable",
)


@dataclass(frozen=True)
class CurrentPhaseLegalityDecision:
    """Current production Phase-evidence eligibility for one cached gamma.

    The numerical-validity boundary is intentionally external.  The 07 audit
    first evaluates every numerically usable oracle gamma on the frozen
    classifier and only then applies this decision post hoc.
    """

    accepted: bool
    reject_reasons: tuple[str, ...]
    shape_support_valid: bool
    q_cdf_available: bool


def audit_current_phase_legality(
    registration: TOnlyPhaseRegistration,
    shape_validation: RawShapeValidation,
    config: PhaseHypothesisScanConfig,
) -> CurrentPhaseLegalityDecision:
    """Reproduce the current production scanner's eligibility gates post hoc.

    This deliberately does *not* use ``registration.t_only_legal``.  06's
    T-only legality is a diagnostic construct that excludes S-SRVF to preserve
    Stage-A independence and additionally tracks target-T validity.  Production
    ``TargetPhaseHypothesisScanner.phase_evidence_eligible`` instead uses:

    - pre-registration T common support;
    - configured gamma increment/speed/roughness/deviation bounds;
    - T registration gain ratio;
    - aligned S-SRVF support computability;
    - availability of the source class S-distance empirical CDF.

    Source q95/outer-range rejection is diagnostic-only in production and is
    therefore intentionally absent here.
    """
    if not registration.numerically_valid or registration.gamma is None:
        raise ValueError("current Phase legality is defined here only for numerically valid gamma")

    reasons: list[str] = []
    if float(registration.pre_common_support_t) < float(config.registration_min_common_support):
        reasons.append("pre_support")

    minimum_increment = registration.gamma_min_increment
    if minimum_increment is None or float(minimum_increment) < float(config.registration_min_increment):
        reasons.append("gamma_increment")

    max_speed = registration.gamma_max_local_speed
    if max_speed is None or float(max_speed) > float(config.registration_max_local_speed):
        reasons.append("gamma_speed")

    roughness = registration.gamma_roughness
    if roughness is None or float(roughness) > float(config.registration_max_roughness):
        reasons.append("gamma_roughness")

    deviation = registration.phase_deviation
    if deviation is None or float(deviation) > float(config.registration_max_deviation):
        reasons.append("gamma_deviation")

    gain = registration.t_gain_ratio
    if gain is None or not math.isfinite(float(gain)):
        reasons.append("gain_unavailable")
    elif float(gain) > float(config.registration_gain_ratio_max):
        reasons.append("gain")

    shape_support_valid = bool(shape_validation.computable)
    q_cdf_available = bool(
        shape_validation.computable and shape_validation.q_distance_percentile is not None
    )
    if not shape_support_valid:
        reasons.append("shape_support")
    elif not q_cdf_available:
        reasons.append("q_cdf_unavailable")

    return CurrentPhaseLegalityDecision(
        accepted=not reasons,
        reject_reasons=tuple(reasons),
        shape_support_valid=shape_support_valid,
        q_cdf_available=q_cdf_available,
    )


def conditional_probability(numerator_mask: Sequence[bool], condition_mask: Sequence[bool]) -> float:
    if len(numerator_mask) != len(condition_mask):
        raise ValueError("mask lengths must match")
    denominator = sum(bool(value) for value in condition_mask)
    if denominator == 0:
        return float("nan")
    numerator = sum(bool(a) and bool(b) for a, b in zip(numerator_mask, condition_mask))
    return float(numerator / denominator)


def rule_threshold_descriptions(config: PhaseHypothesisScanConfig) -> dict[str, str]:
    """Human/machine-readable descriptions of the actual current gates."""
    return {
        "pre_support": f"pre_common_support_t < {float(config.registration_min_common_support):.12g}",
        "gamma_increment": f"gamma_min_increment < {float(config.registration_min_increment):.12g}",
        "gamma_speed": f"gamma_max_local_speed > {float(config.registration_max_local_speed):.12g}",
        "gamma_roughness": f"gamma_roughness > {float(config.registration_max_roughness):.12g}",
        "gamma_deviation": f"phase_deviation > {float(config.registration_max_deviation):.12g}",
        "gain_unavailable": "t_gain_ratio missing or non-finite",
        "gain": f"t_gain_ratio > {float(config.registration_gain_ratio_max):.12g}",
        "shape_support": "support_aware_q_distance.valid == false",
        "q_cdf_unavailable": "source class q_distance_samples is empty",
    }
