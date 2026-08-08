from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from methods.structure_da.target_hypothesis_scan import (
    CandidatePseudoLabel,
    PairwiseClassAlignment,
    TargetClassPhaseHypothesis,
    TargetHypothesisScanResult,
)


def _gamma(power: float, size: int = 32) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, size, dtype=torch.float64).pow(power)


def _hypothesis(
    sample_id: int,
    class_id: int,
    power: float,
    *,
    evidence_weight: float = 1.0,
) -> TargetClassPhaseHypothesis:
    return TargetClassPhaseHypothesis(
        sample_id=sample_id,
        class_id=class_id,
        gamma=_gamma(power),
        t_identity_error=1.0,
        t_registered_error=0.5,
        t_gain_ratio=0.5,
        q_shape_distance=0.1,
        q_distance_percentile=0.1,
        common_support_t=1.0,
        common_support_shape=1.0,
        roughness=0.0,
        phase_deviation=0.0,
        preferred=True,
        ambiguous_class=False,
        evidence_weight=evidence_weight,
    )


def _scan(specs: list[tuple[int, float] | tuple[int, float, float]]) -> TargetHypothesisScanResult:
    hypotheses = []
    for sample_id, spec in enumerate(specs):
        class_id, power, *weight = spec
        hypotheses.append(
            _hypothesis(
                sample_id,
                int(class_id),
                float(power),
                evidence_weight=float(weight[0]) if weight else 1.0,
            )
        )
    return TargetHypothesisScanResult(
        hypotheses=tuple(hypotheses),
        num_samples=len({hypothesis.sample_id for hypothesis in hypotheses}),
        num_pairwise_attempted=0,
        num_pre_support_rejected=0,
        num_solver_failed=0,
        num_gamma_rejected=0,
        num_gain_rejected=0,
        num_shape_support_rejected=0,
        num_outer_rejected=0,
        samples_with_zero_hypothesis=0,
        samples_with_one_hypothesis=len(hypotheses),
        samples_with_two_hypotheses=0,
    )


def _config(**overrides):
    from methods.structure_da.domain_phase_state import DomainPhaseConfig

    values = dict(
        phase_min_samples_per_class=1.0,
        phase_class_dispersion_max=0.02,
        phase_class_diameter_max=0.3,
        phase_group_dispersion_max=0.02,
        phase_group_diameter_max=0.3,
        phase_group_core_separation=0.15,
        phase_global_radius=1.5,
        phase_confirmation_patience=2,
        phase_center_drift_max=0.08,
    )
    values.update(overrides)
    return DomainPhaseConfig(**values)


def _update(specs, *, config=None, previous_state=None):
    from methods.structure_da.domain_phase_state import update_domain_phase_state

    return update_domain_phase_state(
        _scan(specs),
        config or _config(),
        previous_state=previous_state,
    )


def test_three_close_class_centers_form_m1_before_considering_m2() -> None:
    state = _update([(0, 0.98), (1, 1.0), (2, 1.02)])
    assert state.m == 1
    assert state.groups[0].member_classes == (0, 1, 2)
    assert state.rejected_classes == ()


def test_m1_rejects_one_obvious_outlier_to_g0() -> None:
    state = _update(
        [(0, 0.98), (1, 1.0), (2, 1.02), (3, 2.4)],
        config=_config(phase_group_diameter_max=0.18),
    )
    assert state.m == 1
    assert state.groups[0].member_classes == (0, 1, 2)
    assert state.rejected_classes == (3,)


def test_four_close_classes_remain_m1_when_m2_separation_is_not_supported() -> None:
    state = _update(
        [(0, 0.96), (1, 0.99), (2, 1.02), (3, 1.05)],
        config=_config(
            phase_group_diameter_max=0.4,
            phase_group_core_separation=0.3,
        ),
    )
    assert state.m == 1
    assert state.groups[0].member_classes == (0, 1, 2, 3)


def test_two_supported_separated_groups_are_not_hidden_by_feasible_m1() -> None:
    state = _update(
        [(0, 0.55), (1, 0.57), (2, 1.75), (3, 1.80)],
        config=_config(
            phase_group_diameter_max=1.5,
            phase_group_dispersion_max=1.0,
            phase_group_core_separation=0.15,
        ),
    )
    assert state.m == 2
    assert [group.member_classes for group in state.groups] == [(0, 1), (2, 3)]


def test_two_clear_clusters_form_deterministically_ordered_m2_when_m1_fails() -> None:
    state = _update(
        [(0, 0.55), (1, 0.57), (2, 1.75), (3, 1.8)],
        config=_config(
            phase_group_diameter_max=0.12,
            phase_group_dispersion_max=0.01,
            phase_group_core_separation=0.2,
        ),
    )
    assert state.m == 2
    assert [group.group_id for group in state.groups] == [0, 1]
    assert [group.member_classes for group in state.groups] == [(0, 1), (2, 3)]


def test_three_classes_or_a_single_outlier_cannot_force_m2() -> None:
    three = _update(
        [(0, 0.55), (1, 0.57), (2, 1.8)],
        config=_config(phase_group_diameter_max=0.12),
    )
    lone_outlier = _update(
        [(0, 0.95), (1, 1.0), (2, 1.05), (3, 2.5)],
        config=_config(
            phase_group_diameter_max=0.12,
            phase_group_dispersion_max=1e-6,
        ),
    )
    assert three.m != 2
    assert lone_outlier.m != 2


def test_failed_constraints_return_m0_without_fake_identity_group() -> None:
    state = _update(
        [(0, 0.55), (1, 0.8), (2, 1.3), (3, 2.0)],
        config=_config(
            phase_group_diameter_max=1e-6,
            phase_group_dispersion_max=1e-12,
            phase_group_core_separation=1.0,
        ),
    )
    assert state.m == 0
    assert state.groups == ()
    assert state.rejected_classes == (0, 1, 2, 3)


def test_confirmation_requires_stable_membership_and_small_group_drift() -> None:
    from methods.structure_da.domain_phase_state import PhaseGroupStatus

    config = _config()
    first = _update([(0, 0.98), (1, 1.0), (2, 1.02)], config=config)
    assert first.scan_index == 0
    assert first.groups[0].status is PhaseGroupStatus.PROVISIONAL
    assert first.groups[0].confirmation_age == 1

    second = _update(
        [(0, 0.98), (1, 1.0), (2, 1.02)],
        config=config,
        previous_state=first,
    )
    assert second.scan_index == 1
    assert second.groups[0].status is PhaseGroupStatus.CONFIRMED
    assert second.groups[0].confirmation_age == 2

    changed = _update(
        [(0, 0.98), (1, 1.0), (3, 1.02)],
        config=config,
        previous_state=second,
    )
    assert changed.groups[0].status is PhaseGroupStatus.PROVISIONAL
    assert changed.groups[0].confirmation_age == 1


def test_progressive_membership_growth_preserves_confirmation_age() -> None:
    from methods.structure_da.domain_phase_state import (
        PhaseDecisionStatus,
        PhaseGroupStatus,
    )

    config = _config(phase_center_drift_max=0.12)
    first = _update([(0, 0.98), (1, 1.0)], config=config)
    assert first.groups[0].member_classes == (0, 1)
    assert first.groups[0].confirmation_age == 1

    second = _update(
        [(0, 0.98), (1, 1.0), (2, 1.02)],
        config=config,
        previous_state=first,
    )
    assert second.groups[0].member_classes == (0, 1, 2)
    assert second.groups[0].status is PhaseGroupStatus.CONFIRMED
    assert second.groups[0].confirmation_age == 2
    assert second.decision_stability_age == 2
    assert second.decision_status is PhaseDecisionStatus.NONIDENTITY_CONFIRMED

    third = _update(
        [(0, 0.98), (1, 1.0), (2, 1.02), (3, 1.04)],
        config=config,
        previous_state=second,
    )
    assert third.groups[0].member_classes == (0, 1, 2, 3)
    assert third.groups[0].status is PhaseGroupStatus.CONFIRMED
    assert third.groups[0].confirmation_age == 3
    assert third.decision_stability_age == 3
    assert third.decision_status is PhaseDecisionStatus.NONIDENTITY_CONFIRMED


def test_progressive_member_replacement_still_resets_confirmation() -> None:
    from methods.structure_da.domain_phase_state import PhaseGroupStatus

    config = _config(phase_center_drift_max=0.12)
    first = _update([(0, 0.98), (1, 1.0), (2, 1.02)], config=config)
    second = _update(
        [(0, 0.98), (1, 1.0), (2, 1.02)],
        config=config,
        previous_state=first,
    )
    changed = _update(
        [(0, 0.98), (1, 1.0), (3, 1.02)],
        config=config,
        previous_state=second,
    )
    assert changed.groups[0].member_classes == (0, 1, 3)
    assert changed.groups[0].status is PhaseGroupStatus.PROVISIONAL
    assert changed.groups[0].confirmation_age == 1
    assert changed.decision_stability_age == 1


def test_group_center_drift_over_threshold_resets_confirmation() -> None:
    from methods.structure_da.domain_phase_state import PhaseGroupStatus

    config = _config(phase_center_drift_max=0.03)
    first = _update([(0, 0.98), (1, 1.0), (2, 1.02)], config=config)
    confirmed = _update(
        [(0, 0.98), (1, 1.0), (2, 1.02)], config=config, previous_state=first
    )
    prior_group = replace(confirmed.groups[0], center_gamma=_gamma(1.5))
    synthetic_previous = replace(confirmed, groups=(prior_group,))

    current = _update(
        [(0, 0.98), (1, 1.0), (2, 1.02)],
        config=config,
        previous_state=synthetic_previous,
    )
    assert current.groups[0].center_drift is not None
    assert current.groups[0].center_drift > config.phase_center_drift_max
    assert current.groups[0].status is PhaseGroupStatus.PROVISIONAL
    assert current.groups[0].confirmation_age == 1


def test_evidence_weights_only_change_effective_count_not_median_membership() -> None:
    state = _update(
        [(0, 0.9, 0.5), (0, 1.1, 0.5), (1, 1.0), (2, 1.02)],
        config=_config(phase_min_samples_per_class=1.0),
    )
    center = next(item for item in state.class_centers if item.class_id == 0)
    assert center.candidate_count == 2
    assert center.effective_evidence_count == pytest.approx(1.0)
    assert center.valid


def test_invalid_class_centers_are_rejected_and_outputs_are_detached() -> None:
    state = _update(
        [(0, 1.0, 0.4), (1, 1.0), (2, 1.02)],
        config=_config(phase_min_samples_per_class=1.0),
    )
    invalid = next(item for item in state.class_centers if item.class_id == 0)
    assert not invalid.valid
    assert invalid.reject_reason is not None
    assert 0 in state.rejected_classes
    for center in state.class_centers:
        assert center.center_gamma.requires_grad is False
        assert center.center_gamma.device.type == "cpu"
    for group in state.groups:
        assert group.center_gamma.requires_grad is False
        assert group.center_gamma.device.type == "cpu"


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase_group_cap", 3),
        ("phase_min_classes_per_group", 1),
        ("phase_confirmation_patience", 1),
        ("phase_global_radius", -1.0),
    ],
)
def test_domain_phase_config_rejects_invalid_structural_values(field, value) -> None:
    values = _config().__dict__ | {field: value}
    from methods.structure_da.domain_phase_state import DomainPhaseConfig

    with pytest.raises(ValueError):
        DomainPhaseConfig(**values)



def _pairwise_alignment(
    sample_id: int,
    class_id: int,
    power: float,
    *,
    gain_ratio: float = 0.5,
    eligible: bool = True,
    q_distance: float = 0.1,
) -> PairwiseClassAlignment:
    gamma = _gamma(power)
    return PairwiseClassAlignment(
        sample_id=sample_id,
        class_id=class_id,
        gamma=gamma,
        t_identity_error=1.0,
        t_registered_error=gain_ratio,
        t_gain_ratio=gain_ratio,
        pre_common_support_t=1.0,
        common_support_t=1.0,
        gamma_finite=True,
        gamma_endpoint_error=0.0,
        gamma_strictly_increasing=True,
        gamma_min_increment=0.01,
        gamma_max_local_speed=1.0,
        gamma_roughness=0.0,
        phase_deviation=0.0,
        q_shape_distance=q_distance,
        q_distance_percentile=0.1,
        common_support_shape=1.0,
        numerically_valid=True,
        phase_evidence_eligible=eligible,
        reject_reasons=(),
        solver_error=None,
    )


def _candidate(sample_id: int, class_id: int, *, eligible: bool = True) -> CandidatePseudoLabel:
    return CandidatePseudoLabel(
        sample_id=sample_id,
        class_id=class_id,
        q_shape_distance=0.1,
        q_distance_percentile=0.1,
        phase_evidence_eligible=eligible,
        ambiguous=False,
    )


def _scan_with_pairwise(
    alignments: list[PairwiseClassAlignment],
    *,
    hypotheses: tuple[TargetClassPhaseHypothesis, ...] = (),
) -> TargetHypothesisScanResult:
    candidates = tuple(_candidate(item.sample_id, item.class_id, eligible=item.phase_evidence_eligible) for item in alignments)
    sample_ids = {item.sample_id for item in alignments}
    return TargetHypothesisScanResult(
        hypotheses=hypotheses,
        num_samples=len(sample_ids),
        num_pairwise_attempted=len(alignments),
        num_pre_support_rejected=0,
        num_solver_failed=0,
        num_gamma_rejected=0,
        num_gain_rejected=0,
        num_shape_support_rejected=0,
        num_outer_rejected=0,
        samples_with_zero_hypothesis=0,
        samples_with_one_hypothesis=len(sample_ids),
        samples_with_two_hypotheses=0,
        pairwise_alignments=tuple(alignments),
        candidate_pseudo_labels=candidates,
    )


def test_progressive_evidence_can_upgrade_confirmed_m1_to_m2() -> None:
    from methods.structure_da.domain_phase_state import PhaseDecisionStatus

    config = _config(
        phase_group_diameter_max=0.18,
        phase_group_dispersion_max=0.02,
        phase_group_core_separation=0.20,
    )
    first = _update([(0, 0.55), (1, 0.57)], config=config)
    second = _update([(0, 0.55), (1, 0.57)], config=config, previous_state=first)
    assert second.m == 1
    assert second.decision_status is PhaseDecisionStatus.NONIDENTITY_CONFIRMED

    third = _update(
        [(0, 0.55), (1, 0.57), (2, 1.75), (3, 1.80)],
        config=config,
        previous_state=second,
    )
    assert third.m == 2
    assert third.decision_status is PhaseDecisionStatus.UNCONFIRMED

    fourth = _update(
        [(0, 0.55), (1, 0.57), (2, 1.75), (3, 1.80)],
        config=config,
        previous_state=third,
    )
    assert fourth.m == 2
    assert fourth.decision_status is PhaseDecisionStatus.NONIDENTITY_CONFIRMED


def test_unconfirmed_is_not_implicitly_identity() -> None:
    from methods.structure_da.domain_phase_state import PhaseDecisionStatus

    state = _update([], config=_config())
    assert state.m == 0
    assert state.decision_status is PhaseDecisionStatus.UNCONFIRMED
    assert state.identity_evidence_classes == ()


def test_identity_requires_explicit_calibrated_evidence_and_stability() -> None:
    from methods.structure_da.domain_phase_state import PhaseDecisionStatus, update_domain_phase_state

    config = _config(
        phase_identity_radius=0.02,
        phase_identity_gain_ratio_min=0.98,
    )
    scan = _scan_with_pairwise(
        [
            _pairwise_alignment(0, 0, 1.0, gain_ratio=0.99, eligible=False),
            _pairwise_alignment(1, 1, 1.0, gain_ratio=0.995, eligible=False),
        ]
    )
    first = update_domain_phase_state(scan, config)
    assert first.m == 0
    assert first.decision_status is PhaseDecisionStatus.UNCONFIRMED
    assert first.identity_evidence_classes == (0, 1)

    second = update_domain_phase_state(scan, config, previous_state=first)
    assert second.m == 0
    assert second.decision_status is PhaseDecisionStatus.IDENTITY_CONFIRMED


def test_identity_support_growth_preserves_decision_stability_age() -> None:
    from methods.structure_da.domain_phase_state import PhaseDecisionStatus, update_domain_phase_state

    config = _config(
        phase_identity_radius=0.02,
        phase_identity_gain_ratio_min=0.98,
    )
    first_scan = _scan_with_pairwise(
        [
            _pairwise_alignment(0, 0, 1.0, gain_ratio=0.99, eligible=False),
            _pairwise_alignment(1, 1, 1.0, gain_ratio=0.995, eligible=False),
        ]
    )
    first = update_domain_phase_state(first_scan, config)
    assert first.identity_evidence_classes == (0, 1)
    assert first.decision_stability_age == 1

    grown_scan = _scan_with_pairwise(
        [
            _pairwise_alignment(0, 0, 1.0, gain_ratio=0.99, eligible=False),
            _pairwise_alignment(1, 1, 1.0, gain_ratio=0.995, eligible=False),
            _pairwise_alignment(2, 2, 1.0, gain_ratio=0.992, eligible=False),
        ]
    )
    second = update_domain_phase_state(grown_scan, config, previous_state=first)
    assert second.identity_evidence_classes == (0, 1, 2)
    assert second.decision_stability_age == 2
    assert second.decision_status is PhaseDecisionStatus.IDENTITY_CONFIRMED


def _confirmed_m1_state(power: float = 1.0):
    from methods.structure_da.domain_phase_state import (
        DomainPhaseState,
        PhaseDecisionStatus,
        PhaseGroup,
        PhaseGroupStatus,
    )

    group = PhaseGroup(
        group_id=0,
        member_classes=(0, 1),
        center_gamma=_gamma(power),
        within_dispersion=0.0,
        diameter=0.0,
        core_radius=0.0,
        sample_evidence_count=4.0,
        class_count=2,
        center_drift=0.0,
        status=PhaseGroupStatus.CONFIRMED,
        confirmation_age=2,
    )
    return DomainPhaseState(
        scan_index=1,
        m=1,
        class_centers=(),
        valid_phase_classes=(0, 1),
        groups=(group,),
        rejected_classes=(),
        decision_status=PhaseDecisionStatus.NONIDENTITY_CONFIRMED,
        decision_stability_age=2,
    )


def test_compatible_nonmember_candidate_is_assigned_to_nearest_confirmed_group() -> None:
    from methods.structure_da.domain_phase_state import (
        CandidatePhaseCompatibilityStatus,
        evaluate_candidate_phase_compatibility,
    )

    state = _confirmed_m1_state(power=1.0)
    scan = _scan_with_pairwise([_pairwise_alignment(10, 2, 1.01)])
    compatibility = evaluate_candidate_phase_compatibility(scan, state, _config())
    assert len(compatibility) == 1
    assert compatibility[0].status is CandidatePhaseCompatibilityStatus.COMPATIBLE
    assert compatibility[0].assigned_group_id == 0
    assert compatibility[0].nearest_group_id == 0


def test_single_far_candidate_is_residual_but_cannot_create_second_group() -> None:
    from methods.structure_da.domain_phase_state import (
        CandidatePhaseCompatibilityStatus,
        detect_residual_phase_group,
        evaluate_candidate_phase_compatibility,
    )

    config = _config(phase_group_diameter_max=0.12, phase_group_core_separation=0.15)
    scan = _scan_with_pairwise([_pairwise_alignment(10, 2, 2.0)])
    compatibility = evaluate_candidate_phase_compatibility(scan, _confirmed_m1_state(), config)
    assert compatibility[0].status is CandidatePhaseCompatibilityStatus.RESIDUAL
    assert detect_residual_phase_group(compatibility, _confirmed_m1_state(), config) is None


def test_residuals_from_only_one_class_cannot_create_second_group() -> None:
    from methods.structure_da.domain_phase_state import (
        detect_residual_phase_group,
        evaluate_candidate_phase_compatibility,
    )

    config = _config(phase_group_diameter_max=0.12, phase_group_core_separation=0.15)
    scan = _scan_with_pairwise(
        [
            _pairwise_alignment(10, 2, 2.0),
            _pairwise_alignment(11, 2, 2.02),
            _pairwise_alignment(12, 2, 1.98),
        ]
    )
    compatibility = evaluate_candidate_phase_compatibility(scan, _confirmed_m1_state(), config)
    assert detect_residual_phase_group(compatibility, _confirmed_m1_state(), config) is None


def test_multi_sample_multi_class_residual_cluster_can_form_new_group_candidate() -> None:
    from methods.structure_da.domain_phase_state import (
        detect_residual_phase_group,
        evaluate_candidate_phase_compatibility,
    )

    config = _config(
        phase_min_samples_per_class=2.0,
        phase_group_diameter_max=0.15,
        phase_group_dispersion_max=0.02,
        phase_group_core_separation=0.20,
    )
    scan = _scan_with_pairwise(
        [
            _pairwise_alignment(10, 2, 2.00),
            _pairwise_alignment(11, 2, 2.02),
            _pairwise_alignment(12, 3, 1.98),
            _pairwise_alignment(13, 3, 2.01),
        ]
    )
    state = _confirmed_m1_state()
    compatibility = evaluate_candidate_phase_compatibility(scan, state, config)
    residual_group = detect_residual_phase_group(compatibility, state, config)
    assert residual_group is not None
    assert residual_group.member_classes == (2, 3)
    assert residual_group.sample_evidence_count == pytest.approx(4.0)
