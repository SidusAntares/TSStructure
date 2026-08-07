from __future__ import annotations

import pytest
import torch

from methods.structure_da import (
    FdasrsfDP2RegistrationAdapter,
    check_gamma_legality,
    resample_gamma,
    warp_q_gamma,
    warp_support_gamma,
)
from methods.structure_da.temporal_registration import invert_monotone_warp


@pytest.fixture(scope="module")
def grid() -> torch.Tensor:
    return torch.linspace(0, 1, 64)


def _vector_curve(grid: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [torch.sin(3.14159 * grid), torch.cos(6.28318 * grid)], dim=-1
    )


def test_dp2_identity_recovers_identity(grid) -> None:
    q = _vector_curve(grid)
    adapter = FdasrsfDP2RegistrationAdapter()
    gamma = adapter.register(q, q.clone(), grid)
    assert gamma.shape == (64,)
    assert (gamma - grid).abs().max().item() < 1e-3
    legality = check_gamma_legality(gamma, grid)
    assert legality.legal


def test_dp2_known_nonidentity_direction(grid) -> None:
    q = _vector_curve(grid)
    gamma_true = grid ** 1.5
    gamma_inv = invert_monotone_warp(gamma_true)
    # target = (source, gamma_true^{-1}); then (target, gamma_true) ~ source
    target_q = warp_q_gamma(q, gamma_inv).squeeze(0)
    adapter = FdasrsfDP2RegistrationAdapter()
    gamma_hat = adapter.register(q, target_q, grid)

    # gamma_hat must point toward gamma_true, not its inverse.
    diff_true = (gamma_hat - gamma_true).abs().max().item()
    diff_inv = (gamma_hat - gamma_inv).abs().max().item()
    assert diff_true < diff_inv
    assert diff_true < 0.05

    # Registered error must be smaller than identity error.
    e_reg = ((warp_q_gamma(target_q, gamma_hat).squeeze(0) - q).square().mean()).sqrt().item()
    e_id = ((target_q - q).square().mean()).sqrt().item()
    assert e_reg < e_id


def test_dp2_supports_vector_valued_q(grid) -> None:
    # D >= 2 input is required and handled as a single multivariate curve.
    q = _vector_curve(grid)
    adapter = FdasrsfDP2RegistrationAdapter()
    gamma = adapter.register(q, q.clone() + 1e-6, grid)
    assert torch.isfinite(gamma).all()
    assert gamma.requires_grad is False


def test_warp_support_gamma_does_not_multiply_sqrt_derivative(grid) -> None:
    gamma = grid ** 1.5
    support = torch.ones_like(grid)
    warped = warp_support_gamma(support, gamma, grid)
    # Support is a scalar function; warping identity support by gamma must stay 1.
    assert (warped - 1.0).abs().max().item() < 1e-2


def test_resample_gamma_to_target_grid(grid) -> None:
    gamma = grid ** 1.5
    target_grid = torch.linspace(0, 1, 32)
    resampled = resample_gamma(gamma, grid, target_grid)
    assert resampled.shape == (32,)
    assert resampled[0].item() == pytest.approx(0.0, abs=1e-6)
    assert resampled[-1].item() == pytest.approx(1.0, abs=1e-6)
    assert torch.all(resampled[1:] >= resampled[:-1])


def _make_adapter() -> FdasrsfDP2RegistrationAdapter:
    return FdasrsfDP2RegistrationAdapter()


def test_gamma_legality_rejects_nan(grid) -> None:
    bad = torch.full((64,), float("nan"))
    legality = check_gamma_legality(bad, grid)
    assert not legality.legal
    assert legality.finite is False


def test_gamma_legality_rejects_bad_endpoints(grid) -> None:
    bad = grid.clone()
    bad[0] = 0.2
    legality = check_gamma_legality(bad, grid)
    assert not legality.legal
    assert legality.endpoint_error > 1e-6


def test_gamma_legality_rejects_non_increasing(grid) -> None:
    bad = grid.clone()
    bad[20], bad[21] = bad[21], bad[20]
    legality = check_gamma_legality(bad, grid)
    assert not legality.legal
    assert legality.strictly_increasing is False


def test_gamma_legality_rejects_min_increment(grid) -> None:
    bad = grid.clone()
    # collapse a pair to zero increment
    bad[30] = bad[29]
    legality = check_gamma_legality(
        bad, grid, registration_min_increment=1e-4
    )
    assert not legality.legal
    assert legality.min_increment < 1e-4


def test_gamma_legality_rejects_excess_speed(grid) -> None:
    bad = grid.clone()
    bad[31:] = 0.5 * (bad[31:] + 1.0)
    legality = check_gamma_legality(
        bad, grid, registration_max_local_speed=5.0
    )
    assert not legality.legal
    assert legality.max_local_speed > 5.0


def test_gamma_legality_rejects_excess_roughness(grid) -> None:
    bad = grid.clone()
    step = 1.0 / 63
    for index in range(10, 54, 4):
        bad[index] = bad[index] + 0.01 * ((-1) ** index)
    legality = check_gamma_legality(
        bad, grid, registration_max_roughness=1e-6
    )
    # The wiggly gamma has high roughness.
    assert legality.roughness > 1e-6
    assert not legality.legal


def test_gamma_legality_rejects_excess_deviation(grid) -> None:
    bad = grid.clone()
    bad[:] = 0.5 + 0.5 * grid  # large deviation from identity but legal otherwise
    legality = check_gamma_legality(
        bad, grid, registration_max_deviation=0.2
    )
    assert not legality.legal
    assert legality.phase_deviation > 0.2
