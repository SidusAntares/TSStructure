from __future__ import annotations

import numpy as np
import pytest
import torch


def _identity(size: int = 32) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, size, dtype=torch.float64)


def test_phase_distance_identity_symmetry_and_nonidentity() -> None:
    from methods.structure_da.phase_geometry import phase_distance

    identity = _identity()
    warped = identity.square()

    assert phase_distance(identity, identity).item() == pytest.approx(0.0, abs=1e-10)
    distance_ab = phase_distance(identity, warped)
    distance_ba = phase_distance(warped, identity)
    assert distance_ab.item() > 0.0
    torch.testing.assert_close(distance_ab, distance_ba, rtol=0.0, atol=1e-12)
    assert torch.isfinite(distance_ab)
    assert distance_ab.requires_grad is False


def test_pairwise_phase_distances_are_finite_symmetric_with_zero_diagonal() -> None:
    from methods.structure_da.phase_geometry import pairwise_phase_distances

    identity = _identity()
    gammas = torch.stack((identity, identity.square(), identity.sqrt()))
    distances = pairwise_phase_distances(gammas)

    assert distances.shape == (3, 3)
    assert torch.isfinite(distances).all()
    torch.testing.assert_close(distances, distances.T, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(torch.diag(distances), torch.zeros(3, dtype=torch.float64))


def test_sqrt_median_passes_k_by_n_to_fdasrsf(monkeypatch) -> None:
    import fdasrsf.utility_functions as utility_functions
    from methods.structure_da.phase_geometry import sqrt_median_gamma

    gammas = torch.stack((_identity(), _identity().square(), _identity().sqrt()))
    captured: dict[str, tuple[int, ...]] = {}

    def fake_sqrt_median(matrix: np.ndarray):
        captured["shape"] = matrix.shape
        return matrix[:, 0], None, None, None

    monkeypatch.setattr(utility_functions, "SqrtMedian", fake_sqrt_median)
    result = sqrt_median_gamma(gammas)

    assert captured["shape"] == (32, 3)
    assert result.dtype == torch.float64
    assert result.device.type == "cpu"
    assert result.requires_grad is False


def test_sqrt_mean_passes_k_by_n_and_parallel_false_to_fdasrsf(monkeypatch) -> None:
    import fdasrsf.utility_functions as utility_functions
    from methods.structure_da.phase_geometry import sqrt_mean_gamma

    gammas = torch.stack((_identity(), _identity().square(), _identity().sqrt()))
    captured: dict[str, object] = {}

    def fake_sqrt_mean(matrix: np.ndarray, *, parallel: bool):
        captured["shape"] = matrix.shape
        captured["parallel"] = parallel
        return None, matrix[:, 0], None, None

    monkeypatch.setattr(utility_functions, "SqrtMean", fake_sqrt_mean)
    result = sqrt_mean_gamma(gammas)

    assert captured == {"shape": (32, 3), "parallel": False}
    assert result.dtype == torch.float64
    assert result.device.type == "cpu"
    assert result.requires_grad is False


def test_sqrt_mean_falls_back_when_fdasrsf_does_not_converge(monkeypatch) -> None:
    import fdasrsf.utility_functions as utility_functions
    from methods.structure_da.phase_geometry import phase_distance, sqrt_mean_gamma

    def broken_sqrt_mean(*args, **kwargs):
        raise IndexError("index 501 is out of bounds for axis 0 with size 501")

    monkeypatch.setattr(utility_functions, "SqrtMean", broken_sqrt_mean)
    grid = _identity()
    gammas = torch.stack((grid.pow(2.00), grid.pow(2.02)))

    result = sqrt_mean_gamma(gammas)

    assert result.dtype == torch.float64
    assert result.device.type == "cpu"
    assert result.requires_grad is False
    assert torch.isfinite(result).all()
    assert result[0].item() == pytest.approx(0.0, abs=1e-12)
    assert result[-1].item() == pytest.approx(1.0, abs=1e-12)
    assert torch.all(torch.diff(result) > 0.0)
    left = phase_distance(gammas[0], result).item()
    right = phase_distance(gammas[1], result).item()
    assert left == pytest.approx(right, rel=0.0, abs=1e-8)


@pytest.mark.parametrize("reducer_name", ["sqrt_median_gamma", "sqrt_mean_gamma"])
def test_identical_gammas_bypass_fdasrsf_without_nan(monkeypatch, reducer_name: str) -> None:
    import fdasrsf.utility_functions as utility_functions
    from methods.structure_da import phase_geometry

    def forbidden(*args, **kwargs):
        raise AssertionError("fdasrsf must be bypassed for identical gammas")

    monkeypatch.setattr(utility_functions, "SqrtMedian", forbidden)
    monkeypatch.setattr(utility_functions, "SqrtMean", forbidden)
    gamma = _identity()
    result = getattr(phase_geometry, reducer_name)(torch.stack((gamma, gamma, gamma)))

    torch.testing.assert_close(result, gamma)
    assert torch.isfinite(result).all()
    assert result.requires_grad is False
