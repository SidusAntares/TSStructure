from __future__ import annotations

import pytest
import torch

from methods.structure_da import (
    PairwisePhaseCandidate,
    empirical_cdf,
    shape_distance_to_prototype,
)


def _weights(grid: int = 64) -> torch.Tensor:
    w = torch.full((grid,), 1.0 / (grid - 1))
    w[[0, -1]] *= 0.5
    return w / w.sum()


def test_empirical_cdf_monotonic() -> None:
    samples = torch.tensor([0.1, 0.2, 0.4, 0.8])
    queries = torch.tensor([0.05, 0.1, 0.2, 0.4, 0.8, 1.0])
    u = empirical_cdf(samples, queries)
    expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    torch.testing.assert_close(u, expected)
    # Monotonic in queries.
    assert torch.all(u[1:] >= u[:-1])


def test_empirical_cdf_empty_returns_zero() -> None:
    u = empirical_cdf(torch.zeros(0), torch.tensor([0.5]))
    assert u.item() == 0.0


def test_shape_distance_to_prototype_valid() -> None:
    grid = torch.linspace(0, 1, 8)
    q = torch.zeros(8, 4)
    proto = torch.zeros(8, 4)
    proto[2] = 1.0
    support = torch.ones(8)
    distance, valid = shape_distance_to_prototype(q, support, proto, support, _weights(8))
    assert valid
    assert distance > 0.0
    assert torch.isfinite(torch.tensor(distance))


def test_shape_distance_invalid_when_no_common_support() -> None:
    q = torch.zeros(8, 4)
    proto = torch.zeros(8, 4)
    support = torch.ones(8)
    zero_support = torch.zeros(8)
    _, valid = shape_distance_to_prototype(q, support, proto, zero_support, _weights(8))
    assert not valid


def test_pairwise_candidate_rejected_carries_reason() -> None:
    candidate = PairwisePhaseCandidate(
        sample_id=0,
        class_id=1,
        gamma=torch.linspace(0, 1, 8),
        t_identity_error=1.0,
        t_registered_error=2.0,
        t_gain_ratio=2.0,
        common_support=0.0,
        roughness=1.0,
        min_increment=1e-6,
        max_local_speed=1.0,
        phase_deviation=0.1,
        legal=False,
        reject_reason="solver_error",
    )
    assert not candidate.legal
    assert candidate.reject_reason == "solver_error"
