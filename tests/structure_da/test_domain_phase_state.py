from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from methods.structure_da.target_hypothesis_scan import (
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


def test_m1_has_priority_when_a_two_group_partition_could_be_tighter() -> None:
    state = _update(
        [(0, 0.85), (1, 0.95), (2, 1.05), (3, 1.15)],
        config=_config(phase_group_diameter_max=0.4),
    )
    assert state.m == 1
    assert state.groups[0].member_classes == (0, 1, 2, 3)


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
