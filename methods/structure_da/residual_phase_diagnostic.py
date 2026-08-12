"""Pure helpers for experiment 11 residual-Phase diagnostics.

The project convention is ``gamma(u_source) = u_target``.  Experiment 10
shared centers use the same direction.  Therefore the diagnostic factorization

    gamma_sample = delta_shared o residual

implies

    residual = delta_shared^{-1} o gamma_sample.

All composition/inversion in this module is deterministic piecewise-linear
function algebra on the canonical registration grid.  No registration solver,
classifier outcome, legality filter, clustering, or training logic lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .phase_geometry import phase_distance
from .shared_domain_phase_diagnostic import (
    WeightedFrechetMeanResult,
    sample_equal_weights,
    weighted_frechet_mean_gamma,
)
from .temporal_registration import invert_monotone_warp


@dataclass(frozen=True)
class ResidualReconstructionAudit:
    residuals: Tensor
    reconstructed: Tensor
    phase_errors: Tensor
    max_error: float
    mean_error: float
    fail_count: int
    tolerance: float


@dataclass(frozen=True)
class ResidualCenterSummary:
    class_id: int
    center: Tensor
    center_to_identity: float
    dispersion_mean_squared: float
    distance_median: float
    distance_q25: float
    distance_q75: float
    distance_q90: float
    estimator: WeightedFrechetMeanResult


def _as_gamma(gamma: Tensor, *, name: str) -> Tensor:
    if not isinstance(gamma, Tensor) or gamma.ndim != 1 or gamma.numel() < 2:
        raise ValueError(f"{name} must have shape [K] with K>=2")
    value = gamma.detach().cpu().double().contiguous()
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")
    if abs(float(value[0])) > 1e-8 or abs(float(value[-1]) - 1.0) > 1e-8:
        raise ValueError(f"{name} must preserve [0,1] endpoints")
    if not torch.all(value[1:] > value[:-1]).item():
        raise ValueError(f"{name} must be strictly increasing")
    return value


def _evaluate_piecewise_linear(values: Tensor, query: Tensor) -> Tensor:
    """Evaluate a sampled monotone function on a uniform [0,1] grid."""
    values = _as_gamma(values, name="values")
    if not isinstance(query, Tensor) or query.ndim != 1:
        raise ValueError("query must have shape [K]")
    q = query.detach().cpu().double().contiguous()
    if not torch.isfinite(q).all().item() or torch.any((q < 0.0) | (q > 1.0)).item():
        raise ValueError("query must be finite and lie in [0,1]")
    scaled = q * float(values.numel() - 1)
    lower = torch.floor(scaled).long().clamp(0, values.numel() - 2)
    upper = lower + 1
    frac = scaled - lower.double()
    result = values[lower] + frac * (values[upper] - values[lower])
    result = torch.where(q == 0.0, values[0].expand_as(result), result)
    result = torch.where(q == 1.0, values[-1].expand_as(result), result)
    return result.detach()


def compose_phase(left: Tensor, right: Tensor) -> Tensor:
    """Return ``left o right`` for source->target sampled Phase functions."""
    left = _as_gamma(left, name="left")
    right = _as_gamma(right, name="right")
    if left.shape != right.shape:
        raise ValueError("left and right must share shape")
    composed = _evaluate_piecewise_linear(left, right)
    composed[0] = 0.0
    composed[-1] = 1.0
    if not torch.all(composed[1:] > composed[:-1]).item():
        raise RuntimeError("Phase composition lost strict monotonicity")
    return composed.detach()


def inverse_phase(gamma: Tensor) -> Tensor:
    """Return sampled ``gamma^{-1}`` on the same canonical grid."""
    value = _as_gamma(gamma, name="gamma")
    inverse = invert_monotone_warp(value).detach().cpu().double().contiguous()
    inverse[0] = 0.0
    inverse[-1] = 1.0
    if not torch.all(inverse[1:] > inverse[:-1]).item():
        raise RuntimeError("Phase inverse lost strict monotonicity")
    return inverse


def residual_from_shared(sample_gamma: Tensor, shared_gamma: Tensor) -> Tensor:
    """Compute ``shared^{-1} o sample`` under the project source->target convention."""
    sample = _as_gamma(sample_gamma, name="sample_gamma")
    shared = _as_gamma(shared_gamma, name="shared_gamma")
    if sample.shape != shared.shape:
        raise ValueError("sample_gamma and shared_gamma must share shape")
    # Do not first sample shared^{-1} on the uniform grid and then interpolate
    # that sampled inverse: the inverse's breakpoints live at shared(grid), not
    # at the uniform grid, which introduces avoidable discretization error.
    # Query the piecewise-linear inverse directly at gamma_sample(u).
    residual = invert_monotone_warp(shared, sample).detach().cpu().double().contiguous()
    residual[0] = 0.0
    residual[-1] = 1.0
    if not torch.all(residual[1:] > residual[:-1]).item():
        raise RuntimeError("residual Phase lost strict monotonicity")
    return residual


def reconstruct_from_shared_residual(shared_gamma: Tensor, residual_gamma: Tensor) -> Tensor:
    """Reconstruct ``shared o residual`` in the project source->target direction."""
    return compose_phase(shared_gamma, residual_gamma)


def build_residual_population(
    sample_gammas: Tensor,
    labels: Sequence[int],
    shared_minus_class: Mapping[int, Tensor],
    *,
    reconstruction_tolerance: float = 1e-7,
) -> ResidualReconstructionAudit:
    """Factor every sample gamma and enforce the reconstruction hard gate."""
    if not isinstance(sample_gammas, Tensor) or sample_gammas.ndim != 2:
        raise ValueError("sample_gammas must have shape [N,K]")
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.ndim != 1 or labels_arr.size != sample_gammas.shape[0]:
        raise ValueError("labels must have shape [N]")
    tolerance = float(reconstruction_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("reconstruction_tolerance must be positive and finite")

    residuals: list[Tensor] = []
    reconstructed: list[Tensor] = []
    errors: list[float] = []
    for gamma, class_id in zip(sample_gammas, labels_arr.tolist()):
        if int(class_id) not in shared_minus_class:
            raise KeyError(f"missing leave-one-class-out shared Phase for class {class_id}")
        shared = shared_minus_class[int(class_id)]
        residual = residual_from_shared(gamma, shared)
        rebuilt = reconstruct_from_shared_residual(shared, residual)
        error = float(phase_distance(gamma.detach().cpu().double(), rebuilt).item())
        residuals.append(residual)
        reconstructed.append(rebuilt)
        errors.append(error)

    error_tensor = torch.tensor(errors, dtype=torch.float64)
    fail_count = int(torch.sum(error_tensor > tolerance).item())
    result = ResidualReconstructionAudit(
        residuals=torch.stack(residuals),
        reconstructed=torch.stack(reconstructed),
        phase_errors=error_tensor,
        max_error=float(error_tensor.max().item()) if error_tensor.numel() else float("nan"),
        mean_error=float(error_tensor.mean().item()) if error_tensor.numel() else float("nan"),
        fail_count=fail_count,
        tolerance=tolerance,
    )
    if fail_count:
        raise RuntimeError(
            "residual Phase composition reconstruction failed: "
            f"fail_count={fail_count}, max_dGamma={result.max_error:.6g}, "
            f"tolerance={tolerance:.6g}"
        )
    return result


def phase_distances_to_center(gammas: Tensor, center: Tensor) -> Tensor:
    if not isinstance(gammas, Tensor) or gammas.ndim != 2:
        raise ValueError("gammas must have shape [N,K]")
    return torch.tensor(
        [float(phase_distance(row, center).item()) for row in gammas],
        dtype=torch.float64,
    )


def residual_center_summary(class_id: int, residuals: Tensor) -> ResidualCenterSummary:
    if not isinstance(residuals, Tensor) or residuals.ndim != 2 or residuals.shape[0] == 0:
        raise ValueError("residuals must have shape [N,K] with N>0")
    estimator = weighted_frechet_mean_gamma(
        residuals.detach().cpu().double(), sample_equal_weights(residuals.shape[0])
    )
    center = estimator.gamma
    identity = torch.linspace(0.0, 1.0, center.numel(), dtype=torch.float64)
    center_to_identity = float(phase_distance(center, identity).item())
    distances = phase_distances_to_center(residuals, center)
    q = torch.quantile(distances, torch.tensor([0.25, 0.5, 0.75, 0.9], dtype=torch.float64))
    return ResidualCenterSummary(
        class_id=int(class_id),
        center=center,
        center_to_identity=center_to_identity,
        dispersion_mean_squared=float(distances.square().mean().item()),
        distance_median=float(q[1].item()),
        distance_q25=float(q[0].item()),
        distance_q75=float(q[2].item()),
        distance_q90=float(q[3].item()),
        estimator=estimator,
    )


def deterministic_class_folds(
    sample_ids: Sequence[int],
    labels: Sequence[int],
    *,
    n_folds: int = 5,
    seed: int = 20260812,
) -> np.ndarray:
    """Assign reproducible within-class folds without using Phase/classifier values."""
    ids = np.asarray(sample_ids, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if ids.ndim != 1 or labels_arr.shape != ids.shape:
        raise ValueError("sample_ids and labels must share shape [N]")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    folds = np.full(ids.size, -1, dtype=np.int64)
    for class_id in sorted(np.unique(labels_arr).tolist()):
        indices = np.flatnonzero(labels_arr == class_id)
        if indices.size < n_folds:
            raise ValueError(f"class {class_id} has fewer than {n_folds} samples")
        # Sort before randomization so assignment does not depend on DataLoader/cache order.
        ordered = indices[np.argsort(ids[indices], kind="mergesort")]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(class_id)]))
        permuted = ordered[rng.permutation(ordered.size)]
        for position, index in enumerate(permuted.tolist()):
            folds[int(index)] = int(position % n_folds)
    if np.any(folds < 0):
        raise RuntimeError("some samples were not assigned to a fold")
    return folds


def pairwise_center_distances(centers: Tensor) -> Tensor:
    if not isinstance(centers, Tensor) or centers.ndim != 2:
        raise ValueError("centers must have shape [M,K]")
    count = int(centers.shape[0])
    result = torch.zeros((count, count), dtype=torch.float64)
    for i in range(count):
        for j in range(i + 1, count):
            d = phase_distance(centers[i], centers[j])
            result[i, j] = d
            result[j, i] = d
    return result
