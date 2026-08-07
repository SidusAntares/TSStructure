from __future__ import annotations

import pytest
import torch

from methods.structure_da import (
    SourcePrototypeBank,
    SupportAwareDistanceOutput,
    support_aware_q_distance,
)


def _bank(**overrides) -> SourcePrototypeBank:
    values = dict(
        trend_srvf=torch.zeros(3, 6, 4),
        shape_srvf=torch.randn(3, 6, 4),
        trend_support=torch.ones(3, 6),
        shape_support=torch.ones(3, 6),
        fused=torch.randn(3, 8),
        class_counts=torch.tensor([5, 5, 5]),
        ready=torch.ones(3, dtype=torch.bool),
        q_distance_samples=(torch.zeros(0), torch.zeros(0), torch.zeros(0)),
        f_distance_samples=(torch.zeros(0), torch.zeros(0), torch.zeros(0)),
        q_quantiles=torch.zeros(3, 3),
        f_quantiles=torch.zeros(3, 3),
        version=0,
    )
    values.update(overrides)
    return SourcePrototypeBank(**values)


def test_bank_ready_classes_reports_ready_only() -> None:
    bank = _bank(ready=torch.tensor([True, False, True]))
    assert bank.ready_classes() == [0, 2]


def test_bank_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="class_counts"):
        _bank(class_counts=torch.tensor([5, 5]))
    with pytest.raises(ValueError, match="ready"):
        _bank(ready=torch.tensor([True, False]))


def _weights(grid: int = 6) -> torch.Tensor:
    w = torch.full((grid,), 1.0 / (grid - 1))
    w[[0, -1]] *= 0.5
    return w / w.sum()


def test_support_aware_distance_shapes_and_validity() -> None:
    q_a = torch.randn(3, 6, 4)
    q_b = torch.randn(2, 6, 4)
    sup_a = torch.rand(3, 6)
    sup_b = torch.rand(2, 6)

    output = support_aware_q_distance(q_a, q_b, sup_a, sup_b, _weights())

    assert isinstance(output, SupportAwareDistanceOutput)
    assert output.distance_sq.shape == (3, 2)
    assert output.distance.shape == (3, 2)
    assert output.valid.shape == (3, 2)
    assert output.common_support.shape == (3, 2)
    assert output.valid.dtype == torch.bool
    assert (output.common_support >= 0).all()
    assert torch.isfinite(output.distance).all()
    torch.testing.assert_close(output.distance, output.distance_sq.sqrt())


def test_invalid_support_is_masked_not_large_distance() -> None:
    q_a = torch.randn(3, 6, 4)
    q_b = torch.randn(2, 6, 4)
    sup_a = torch.ones(3, 6)
    sup_b = torch.zeros(2, 6)

    output = support_aware_q_distance(q_a, q_b, sup_a, sup_b, _weights())

    assert not output.valid.any().item()
    # The masked distance stays finite, never a giant fake constant.
    assert torch.isfinite(output.distance).all()
    assert output.distance.max().item() < 1e6


def test_low_support_positions_weight_prototype_less() -> None:
    # Two query SRVFs differ only at one grid point; when support at that point
    # is zero the distance must shrink relative to when it is one.
    q = torch.zeros(2, 6, 2)
    q[0, 3, :] = torch.tensor([2.0, 0.0])
    q[1, 3, :] = torch.tensor([2.0, 0.0])
    q_b = torch.zeros(1, 6, 2)
    high = torch.ones(2, 6)
    low = torch.ones(2, 6)
    low[0, 3] = 0.0

    d_high = support_aware_q_distance(q, q_b, high, torch.ones(1, 6), _weights())
    d_low = support_aware_q_distance(q, q_b, low, torch.ones(1, 6), _weights())

    # Row 0 (zero support at the differing point) must be closer than row 1.
    assert d_low.distance[0, 0].item() < d_low.distance[1, 0].item()
    assert d_high.distance[0, 0].item() == pytest.approx(d_high.distance[1, 0].item())


def test_bank_tensors_are_not_parameters() -> None:
    bank = _bank()
    for value in (
        bank.trend_srvf,
        bank.shape_srvf,
        bank.trend_support,
        bank.shape_support,
        bank.fused,
        bank.q_quantiles,
        bank.f_quantiles,
    ):
        assert not isinstance(value, torch.nn.Parameter)
