from __future__ import annotations

import torch

from methods.structure_da.domain_shape_state import (
    DomainShapeConfig,
    DomainShapeStatus,
    update_domain_shape_state,
)
from methods.structure_da.prototype_bank import SourcePrototypeBank
from methods.structure_da.stable_target_labels import (
    StableTargetLabel,
    StableTargetLabelScanResult,
)


def _config(**overrides) -> DomainShapeConfig:
    values = dict(
        shape_min_valid_classes=2,
        shape_min_samples_per_class=1,
        shape_shared_ratio_min=0.5,
        shape_leave_one_out_drift_max=2.0,
        shape_center_drift_max=0.25,
        shape_effect_norm_max=10.0,
        shape_confirmation_patience=2,
    )
    values.update(overrides)
    return DomainShapeConfig(**values)


def _bank(num_classes: int = 3, grid: int = 5, dim: int = 2) -> SourcePrototypeBank:
    return SourcePrototypeBank(
        trend_srvf=torch.zeros(num_classes, grid, dim),
        shape_srvf=torch.zeros(num_classes, grid, dim),
        trend_support=torch.ones(num_classes, grid),
        shape_support=torch.ones(num_classes, grid),
        fused=torch.eye(num_classes),
        class_counts=torch.full((num_classes,), 10),
        ready=torch.ones(num_classes, dtype=torch.bool),
        q_distance_samples=tuple(torch.zeros(0) for _ in range(num_classes)),
        f_distance_samples=tuple(torch.zeros(0) for _ in range(num_classes)),
        q_quantiles=torch.zeros(num_classes, 3),
        f_quantiles=torch.zeros(num_classes, 3),
        version=1,
    )


def _label(sample_id: int, class_id: int, residual: float, confidence: float = 0.9):
    q = torch.zeros(5, 2)
    q[:, 0] = residual
    return StableTargetLabel(
        sample_id=sample_id,
        class_id=class_id,
        group_id=0,
        aligned_q_shape=q,
        aligned_q_support=torch.ones(5),
        fused_repr=torch.zeros(3),
        confidence_summary=confidence,
    )


def _scan(labels: list[StableTargetLabel], num_classes: int = 3) -> StableTargetLabelScanResult:
    counts = [0] * num_classes
    for label in labels:
        counts[label.class_id] += 1
    return StableTargetLabelScanResult(
        candidates=(),
        stable_labels=tuple(labels),
        num_samples=len(labels),
        num_without_confirmed_phase=0,
        num_candidate_views=len(labels),
        num_classifier_pass=len(labels),
        num_fused_pass=len(labels),
        num_q_pass=len(labels),
        num_stable_labels=len(labels),
        num_ambiguous_rejected=0,
        stable_class_counts=tuple(counts),
    )


def test_same_class_residuals_give_zero_interaction_and_rho_near_one() -> None:
    scan = _scan([_label(0, 0, 1.0), _label(1, 1, 1.0), _label(2, 2, 1.0)])
    state = update_domain_shape_state(scan, _bank(), _config())

    assert state.status is DomainShapeStatus.PROVISIONAL
    assert state.delta is not None
    torch.testing.assert_close(state.delta[:, 0], torch.ones(5))
    assert all(torch.allclose(xi, torch.zeros_like(xi)) for xi in state.interactions)
    assert state.rho_shape is not None and state.rho_shape > 0.999999


def test_mutually_inconsistent_class_residuals_lower_shared_ratio() -> None:
    scan = _scan([_label(0, 0, -1.0), _label(1, 1, 0.0), _label(2, 2, 1.0)])
    state = update_domain_shape_state(
        scan,
        _bank(),
        _config(shape_shared_ratio_min=0.0),
    )
    assert state.rho_shape is not None
    assert state.rho_shape < 0.1


def test_equal_class_weighting_ignores_target_sample_imbalance_and_confidence() -> None:
    labels = [_label(i, 0, 2.0, confidence=0.01) for i in range(10)]
    labels += [_label(10, 1, 0.0, confidence=0.999)]
    state = update_domain_shape_state(
        _scan(labels),
        _bank(),
        _config(shape_shared_ratio_min=0.0),
    )
    assert state.delta is not None
    # Equal class weighting: (2 + 0) / 2 = 1, not sample-weighted 20/11.
    torch.testing.assert_close(state.delta[:, 0], torch.ones(5))


def test_insufficient_valid_classes_is_unavailable_without_fake_delta() -> None:
    state = update_domain_shape_state(
        _scan([_label(0, 0, 1.0)]),
        _bank(),
        _config(),
    )
    assert state.status is DomainShapeStatus.UNAVAILABLE
    assert state.delta is None
    assert state.interactions == ()
    assert state.confirmation_age == 0


def test_leave_one_out_drift_gate_rejects_unstable_shared_effect() -> None:
    scan = _scan([_label(0, 0, 0.0), _label(1, 1, 0.0), _label(2, 2, 3.0)])
    state = update_domain_shape_state(
        scan,
        _bank(),
        _config(
            shape_shared_ratio_min=0.0,
            shape_leave_one_out_drift_max=0.1,
        ),
    )
    assert state.leave_one_out_drift is not None
    assert state.leave_one_out_drift > 0.1
    assert state.status is DomainShapeStatus.REJECTED


def test_first_valid_scan_is_provisional_and_repeated_stable_scan_confirms() -> None:
    scan = _scan([_label(0, 0, 1.0), _label(1, 1, 1.0), _label(2, 2, 1.0)])
    first = update_domain_shape_state(scan, _bank(), _config())
    second = update_domain_shape_state(
        scan,
        _bank(),
        _config(),
        previous_state=first,
    )
    assert first.status is DomainShapeStatus.PROVISIONAL
    assert first.confirmation_age == 1
    assert second.status is DomainShapeStatus.CONFIRMED
    assert second.confirmation_age == 2
    assert second.center_drift == 0.0


def test_large_delta_drift_rejects_and_resets_confirmation() -> None:
    first_scan = _scan([_label(0, 0, 0.5), _label(1, 1, 0.5), _label(2, 2, 0.5)])
    second_scan = _scan([_label(0, 0, 2.0), _label(1, 1, 2.0), _label(2, 2, 2.0)])
    config = _config(shape_center_drift_max=0.1)
    first = update_domain_shape_state(first_scan, _bank(), config)
    second = update_domain_shape_state(
        second_scan,
        _bank(),
        config,
        previous_state=first,
    )
    assert second.center_drift is not None and second.center_drift > 0.1
    assert second.status is DomainShapeStatus.REJECTED
    assert second.confirmation_age == 0


def test_statistics_api_has_no_target_true_label_input_and_detaches_tensors() -> None:
    import inspect

    signature = inspect.signature(update_domain_shape_state)
    assert "target_labels" not in signature.parameters
    q = torch.ones(5, 2, requires_grad=True)
    label = StableTargetLabel(
        sample_id=0,
        class_id=0,
        group_id=0,
        aligned_q_shape=q,
        aligned_q_support=torch.ones(5),
        fused_repr=torch.zeros(3),
        confidence_summary=0.7,
    )
    state = update_domain_shape_state(
        _scan([label, _label(1, 1, 1.0)]),
        _bank(),
        _config(),
    )
    assert state.delta is not None and not state.delta.requires_grad
    assert all(not center.center_q.requires_grad for center in state.class_centers)
