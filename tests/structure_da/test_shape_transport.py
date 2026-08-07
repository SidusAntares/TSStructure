from __future__ import annotations

import torch

from methods.structure_da.decomposition import SymmetricTimeKernelDecomposition
from methods.structure_da.domain_phase_state import (
    DomainPhaseState,
    PhaseGroup,
    PhaseGroupStatus,
)
from methods.structure_da.domain_shape_state import (
    DomainShapeState,
    DomainShapeStatus,
)
from methods.structure_da.shape_transport import (
    apply_domain_shape_effect,
    build_synthetic_source_example,
    correct_target_shape_to_source,
    inverse_vector_srvf,
    map_source_positions_to_target,
    synthesize_source_shape_to_target,
)


def _shape_state(delta: torch.Tensor, status=DomainShapeStatus.CONFIRMED):
    return DomainShapeState(
        scan_index=1,
        status=status,
        class_centers=(),
        valid_classes=(0, 1),
        delta=delta,
        interactions=(),
        rho_shape=1.0,
        leave_one_out_drift=0.0,
        center_drift=0.0,
        confirmation_age=2,
    )


def _phase_state(status=PhaseGroupStatus.CONFIRMED, *, include_class=True):
    grid = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    gamma = grid.square()
    members = (0, 1) if include_class else (1, 2)
    group = PhaseGroup(
        group_id=0,
        member_classes=members,
        center_gamma=gamma,
        within_dispersion=0.0,
        diameter=0.0,
        core_radius=0.0,
        sample_evidence_count=4.0,
        class_count=2,
        center_drift=0.0,
        status=status,
        confirmation_age=2 if status is PhaseGroupStatus.CONFIRMED else 1,
    )
    return DomainPhaseState(
        scan_index=1,
        m=1,
        class_centers=(),
        valid_phase_classes=members,
        groups=(group,),
        rejected_classes=(),
    )


def test_delta_zero_keeps_source_q_unchanged() -> None:
    q = torch.randn(7, 3)
    out = synthesize_source_shape_to_target(q, _shape_state(torch.zeros_like(q)), 1.0)
    torch.testing.assert_close(out, q)


def test_lambda_zero_keeps_source_q_unchanged() -> None:
    q = torch.randn(7, 3)
    delta = torch.randn_like(q)
    torch.testing.assert_close(apply_domain_shape_effect(q, delta, 0.0), q)


def test_lambda_one_adds_full_delta() -> None:
    q = torch.randn(7, 3)
    delta = torch.randn_like(q)
    torch.testing.assert_close(apply_domain_shape_effect(q, delta, 1.0), q + delta)


def test_vector_srvf_inverse_is_finite() -> None:
    q = torch.randn(2, 11, 4)
    curve = inverse_vector_srvf(q, torch.zeros(2, 4))
    assert curve.shape == q.shape
    assert torch.isfinite(curve).all()


def test_vector_srvf_inverse_reconstructs_constant_velocity_curve() -> None:
    grid = torch.linspace(0.0, 1.0, 17)
    velocity = torch.tensor([2.0, -1.0, 0.5])
    velocity_norm = torch.linalg.vector_norm(velocity)
    q_row = velocity / torch.sqrt(velocity_norm)
    q = q_row.expand(grid.numel(), -1).clone()
    initial = torch.tensor([3.0, 4.0, -2.0])
    reconstructed = inverse_vector_srvf(q, initial, grid=grid)
    expected = initial + grid[:, None] * velocity
    torch.testing.assert_close(reconstructed, expected, atol=1e-5, rtol=1e-5)


def test_target_to_source_correction_recovers_source_shape() -> None:
    source_q = torch.randn(9, 2)
    delta = torch.randn_like(source_q)
    target_q = source_q + delta
    corrected = correct_target_shape_to_source(target_q, _shape_state(delta))
    torch.testing.assert_close(corrected, source_q)


def test_source_to_target_phase_uses_gamma_not_inverse_gamma() -> None:
    source_positions = torch.tensor([0.25, 0.5, 0.75])
    mapped = map_source_positions_to_target(
        source_positions,
        torch.ones(3, dtype=torch.bool),
        _phase_state().groups[0].center_gamma,
    )
    expected = source_positions.square()
    inverse_values = torch.sqrt(source_positions)
    torch.testing.assert_close(mapped, expected, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(mapped, inverse_values)


def _build(phase_state, shape_status=DomainShapeStatus.CONFIRMED):
    grid = torch.linspace(0.0, 1.0, 9)
    velocity = torch.tensor([1.0, 0.5])
    norm = torch.linalg.vector_norm(velocity)
    q = (velocity / torch.sqrt(norm)).expand(9, -1).clone()
    structure = grid[:, None] * velocity
    return build_synthetic_source_example(
        source_sample_id=7,
        class_id=0,
        source_structure_function=structure,
        source_q_shape=q,
        source_q_support=torch.ones(9),
        source_positions=torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
        mask=torch.ones(5, dtype=torch.bool),
        phase_state=phase_state,
        domain_shape_state=_shape_state(torch.zeros_like(q), status=shape_status),
        decomposition=SymmetricTimeKernelDecomposition(),
        lambda_delta=1.0,
    )


def test_g0_or_provisional_phase_cannot_create_phase_shape_example() -> None:
    g0 = DomainPhaseState(
        scan_index=1,
        m=0,
        class_centers=(),
        valid_phase_classes=(),
        groups=(),
        rejected_classes=(0,),
    )
    assert _build(g0) is None
    assert _build(_phase_state(PhaseGroupStatus.PROVISIONAL)) is None
    assert _build(_phase_state(include_class=False)) is None


def test_unconfirmed_domain_shape_cannot_create_synthetic_example() -> None:
    assert _build(_phase_state(), DomainShapeStatus.PROVISIONAL) is None


def test_synthetic_example_uses_frozen_slow_operator_and_keeps_true_class() -> None:
    example = _build(_phase_state())
    assert example is not None
    assert example.class_id == 0
    assert example.group_id == 0
    assert example.trend_tokens.shape == example.structure_tokens.shape == (5, 2)
    assert torch.isfinite(example.trend_tokens).all()
    assert torch.isfinite(example.structure_tokens).all()
