from __future__ import annotations

import pytest
import torch

from methods.structure_da import (
    FdasrsfCurveRegistrationAdapter,
    check_gamma_legality,
    resample_gamma,
    warp_q_gamma,
    warp_support_gamma,
)
from methods.structure_da.temporal_registration import invert_monotone_warp


@pytest.fixture(scope="module")
def grid() -> torch.Tensor:
    return torch.linspace(0, 1, 128)


def _vector_curve(grid: torch.Tensor, dims: int = 2) -> torch.Tensor:
    channels = [torch.sin((1 + d) * 3.14159 * grid) for d in range(dims)]
    return torch.stack(channels, dim=-1)


def _adapter() -> FdasrsfCurveRegistrationAdapter:
    return FdasrsfCurveRegistrationAdapter(registration_lambda=0.0)


def test_curve_dp_identity_recovers_identity(grid) -> None:
    q = _vector_curve(grid)
    adapter = _adapter()
    gamma = adapter.register(q, q.clone())
    assert gamma.shape == (128,)
    assert (gamma - grid).abs().max().item() < 1e-3
    legality = check_gamma_legality(gamma, grid)
    assert legality.legal


def test_curve_dp_known_nonidentity_direction(grid) -> None:
    q = _vector_curve(grid)
    gamma_true = grid ** 1.5
    gamma_inv = invert_monotone_warp(gamma_true)
    # target = (source, gamma_true^{-1}); then (target, gamma_true) ~ source
    target_q = warp_q_gamma(q, gamma_inv).squeeze(0)
    adapter = _adapter()
    gamma_hat = adapter.register(q, target_q)

    # One shared gamma for the vector-valued curve.
    assert gamma_hat.shape == (128,)
    assert gamma_hat.ndim == 1
    assert torch.isfinite(gamma_hat).all()
    assert torch.all(gamma_hat[1:] >= gamma_hat[:-1])
    assert gamma_hat[0].item() == pytest.approx(0.0, abs=1e-3)
    assert gamma_hat[-1].item() == pytest.approx(1.0, abs=1e-3)

    # gamma_hat must point toward gamma_true, not its inverse.
    diff_true = (gamma_hat - gamma_true).abs().max().item()
    diff_inv = (gamma_hat - gamma_inv).abs().max().item()
    assert diff_true < diff_inv
    assert diff_true < 0.05

    # Registered error must be clearly smaller than identity error.
    e_reg = ((warp_q_gamma(target_q, gamma_hat).squeeze(0) - q).square().mean()).sqrt().item()
    e_id = ((target_q - q).square().mean()).sqrt().item()
    assert e_reg < e_id


@pytest.mark.parametrize("dims", [2, 4, 8])
def test_curve_dp_supports_various_dimensionality(grid, dims) -> None:
    # A [K, D] curve always yields exactly one [K] gamma, never a [K, D] matrix.
    q = _vector_curve(grid, dims=dims)
    adapter = _adapter()
    gamma = adapter.register(q, q.clone() + 1e-6)
    assert gamma.ndim == 1
    assert gamma.shape == (128,)
    assert gamma.requires_grad is False
    assert torch.isfinite(gamma).all()


def test_shared_gamma_is_joint_curve_solve_not_channel_average(monkeypatch) -> None:
    """Regression guard: the vector-valued warp must come from the curve solver
    directly, never from averaging per-channel gammas of the utility solver."""
    from fdasrsf import curve_functions as cf

    grid = torch.linspace(0, 1, 64)
    src = _vector_curve(grid, dims=3)
    tgt = src.clone() + 0.01
    expected_gamma = (torch.linspace(0, 1, 64) ** 1.4).to(torch.float64)

    captured = {}

    def fake_curve_reparam(q1, q2, lam=0.0, method="DP"):
        captured["q1_shape"] = q1.shape
        captured["q2_shape"] = q2.shape
        captured["method"] = method
        return expected_gamma.numpy()

    def forbidden_utility(*args, **kwargs):
        raise AssertionError(
            "utility_functions.optimum_reparam must not be used "
            "for vector-valued shared registration"
        )

    monkeypatch.setattr(cf, "optimum_reparam_curve", fake_curve_reparam)
    import fdasrsf.utility_functions as uf

    monkeypatch.setattr(uf, "optimum_reparam", forbidden_utility)

    adapter = FdasrsfCurveRegistrationAdapter(registration_lambda=0.0)
    gamma = adapter.register(src, tgt)

    # The curve solver receives [D, K] and returns the shared gamma unchanged.
    assert captured["q1_shape"] == (3, 64)
    assert captured["q2_shape"] == (3, 64)
    assert captured["method"] == "DP"
    torch.testing.assert_close(gamma, expected_gamma, rtol=0, atol=0)


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


def test_gamma_legality_rejects_nan(grid) -> None:
    bad = torch.full((128,), float("nan"))
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
    bad[40], bad[41] = bad[41], bad[40]
    legality = check_gamma_legality(bad, grid)
    assert not legality.legal
    assert legality.strictly_increasing is False


def test_gamma_legality_rejects_min_increment(grid) -> None:
    bad = grid.clone()
    # collapse a pair to zero increment
    bad[60] = bad[59]
    legality = check_gamma_legality(
        bad, grid, registration_min_increment=1e-4
    )
    assert not legality.legal
    assert legality.min_increment < 1e-4


def test_gamma_legality_rejects_excess_speed(grid) -> None:
    bad = grid.clone()
    bad[64:] = 0.5 * (bad[64:] + 1.0)
    legality = check_gamma_legality(
        bad, grid, registration_max_local_speed=5.0
    )
    assert not legality.legal
    assert legality.max_local_speed > 5.0


def test_gamma_legality_rejects_excess_roughness(grid) -> None:
    # Build a strictly increasing gamma with large oscillating local speed so
    # the log-speed derivative (roughness) is high while increments stay > 0.
    bad = grid.clone()
    step = 1.0 / 127
    base_speed = 1.0
    oscillating = base_speed + 0.6 * torch.sin(12.0 * grid)
    cumulative = torch.cat([torch.zeros(1), torch.cumsum(oscillating[1:], dim=0)])
    bad = cumulative / cumulative[-1] * 1.0
    # Re-anchor endpoints exactly.
    bad = (bad - bad[0]) / (bad[-1] - bad[0])
    legality = check_gamma_legality(
        bad, grid, registration_max_roughness=1e-6
    )
    assert legality.strictly_increasing
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
