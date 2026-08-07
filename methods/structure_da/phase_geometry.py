"""CPU Fisher--Rao geometry helpers for monotone phase warps."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _as_gamma(gamma: Tensor, *, name: str) -> Tensor:
    if not isinstance(gamma, Tensor) or gamma.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional torch.Tensor [K]")
    if gamma.numel() < 2:
        raise ValueError(f"{name} must contain at least two grid points")
    result = gamma.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _as_gammas(gammas: Tensor) -> Tensor:
    if not isinstance(gammas, Tensor) or gammas.ndim != 2:
        raise ValueError("gammas must be a two-dimensional torch.Tensor [N,K]")
    if gammas.shape[0] == 0 or gammas.shape[1] < 2:
        raise ValueError("gammas must contain at least one warp and two grid points")
    result = gammas.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not torch.isfinite(result).all():
        raise ValueError("gammas must contain only finite values")
    return result


def gamma_to_psi(gamma: Tensor) -> Tensor:
    """Map one sampled warp ``gamma[K]`` to normalized ``sqrt(gamma_dot)``."""
    gamma_cpu = _as_gamma(gamma, name="gamma")
    interval_count = gamma_cpu.numel() - 1
    derivative = torch.diff(gamma_cpu) * interval_count
    if torch.any(derivative < -1e-12):
        raise ValueError("gamma must be monotonically nondecreasing")
    psi = torch.sqrt(derivative.clamp_min(0.0))
    step = 1.0 / interval_count
    norm = torch.sqrt((psi.square().sum() * step).clamp_min(0.0))
    if not torch.isfinite(norm) or norm.item() <= 0.0:
        raise ValueError("gamma has zero or non-finite square-root velocity norm")
    return (psi / norm).detach()


def phase_distance(gamma_a: Tensor, gamma_b: Tensor) -> Tensor:
    """Return the Fisher--Rao distance between two sampled monotone warps."""
    a = _as_gamma(gamma_a, name="gamma_a")
    b = _as_gamma(gamma_b, name="gamma_b")
    if a.shape != b.shape:
        raise ValueError("gamma_a and gamma_b must have the same shape")
    psi_a = gamma_to_psi(a)
    psi_b = gamma_to_psi(b)
    step = 1.0 / (a.numel() - 1)
    inner = (psi_a * psi_b).sum() * step
    distance = torch.acos(inner.clamp(-1.0, 1.0))
    return distance.detach().to(dtype=torch.float64, device="cpu")


def pairwise_phase_distances(gammas: Tensor) -> Tensor:
    """Return the symmetric pairwise Fisher--Rao distance matrix ``[N,N]``."""
    gammas_cpu = _as_gammas(gammas)
    count = gammas_cpu.shape[0]
    distances = torch.zeros((count, count), dtype=torch.float64)
    for left in range(count):
        for right in range(left + 1, count):
            value = phase_distance(gammas_cpu[left], gammas_cpu[right])
            distances[left, right] = value
            distances[right, left] = value
    return distances


def _all_phase_identical(gammas: Tensor) -> bool:
    if gammas.shape[0] <= 1:
        return True
    distances = pairwise_phase_distances(gammas)
    return bool(torch.all(distances <= 1e-10).item())


def _validate_reducer_result(result, *, size: int, name: str) -> Tensor:
    array = np.asarray(result, dtype=np.float64)
    if array.shape != (size,):
        raise RuntimeError(f"fdasrsf {name} returned shape {array.shape}, expected {(size,)}")
    tensor = torch.from_numpy(array.copy()).to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"fdasrsf {name} returned non-finite values")
    return tensor.detach()


def sqrt_median_gamma(gammas: Tensor) -> Tensor:
    """Compute the unweighted fdasrsf square-root median of ``gammas[N,K]``."""
    gammas_cpu = _as_gammas(gammas)
    if _all_phase_identical(gammas_cpu):
        return gammas_cpu[0].clone().detach()
    from fdasrsf import utility_functions

    gamma_matrix_np = gammas_cpu.numpy().T
    gam_median, _psi_median, _psi, _vec = utility_functions.SqrtMedian(
        gamma_matrix_np
    )
    return _validate_reducer_result(
        gam_median, size=gammas_cpu.shape[1], name="SqrtMedian"
    )


def sqrt_mean_gamma(gammas: Tensor) -> Tensor:
    """Compute the unweighted fdasrsf square-root mean of ``gammas[N,K]``."""
    gammas_cpu = _as_gammas(gammas)
    if _all_phase_identical(gammas_cpu):
        return gammas_cpu[0].clone().detach()
    from fdasrsf import utility_functions

    gamma_matrix_np = gammas_cpu.numpy().T
    _mu, gam_mean, _psi, _vec = utility_functions.SqrtMean(
        gamma_matrix_np,
        parallel=False,
    )
    return _validate_reducer_result(
        gam_mean, size=gammas_cpu.shape[1], name="SqrtMean"
    )
