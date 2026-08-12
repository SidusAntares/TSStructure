"""Pure helpers for experiment 10 shared-Domain-Phase diagnostics.

This module estimates only diagnostic Fisher--Rao Frechet centers from already
computed oracle true-class sample gammas.  It never runs registration, clusters
samples, selects a Phase group count, or uses classifier outcomes to change a
center.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .phase_population_diagnostic import gamma_to_unit_phase_vectors


@dataclass(frozen=True)
class WeightedFrechetMeanResult:
    gamma: Tensor
    objective: float
    iterations: int
    converged: bool
    tangent_norm: float


def _as_labels(labels: Sequence[int], n: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size != int(n):
        raise ValueError("labels must have shape [N]")
    return values


def class_balanced_weights(labels: Sequence[int], *, held_out_class: int | None = None) -> Tensor:
    """Return sample weights with equal total mass for each included class.

    If ``held_out_class`` is given, that class receives exactly zero weight and
    the remaining classes each receive total mass ``1/(C-1)``.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.ndim != 1 or labels_arr.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    classes = sorted(np.unique(labels_arr).tolist())
    if held_out_class is not None:
        held_out_class = int(held_out_class)
        if held_out_class not in classes:
            raise ValueError("held_out_class is not present in labels")
        included = [c for c in classes if c != held_out_class]
    else:
        included = classes
    if not included:
        raise ValueError("at least one class must remain")
    weights = np.zeros(labels_arr.size, dtype=np.float64)
    class_mass = 1.0 / float(len(included))
    for class_id in included:
        indices = np.flatnonzero(labels_arr == class_id)
        if indices.size == 0:
            raise RuntimeError("included class unexpectedly has no samples")
        weights[indices] = class_mass / float(indices.size)
    total = float(weights.sum())
    if not math.isfinite(total) or abs(total - 1.0) > 1e-12:
        raise RuntimeError("class-balanced weights failed to normalize")
    return torch.from_numpy(weights)


def sample_equal_weights(n: int) -> Tensor:
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    return torch.full((n,), 1.0 / float(n), dtype=torch.float64)


def _validate_weights(weights: Tensor, n: int) -> Tensor:
    if not isinstance(weights, Tensor) or weights.ndim != 1 or weights.numel() != int(n):
        raise ValueError("weights must have shape [N]")
    w = weights.detach().cpu().double().contiguous()
    if not torch.isfinite(w).all().item() or torch.any(w < 0).item():
        raise ValueError("weights must be finite and non-negative")
    total = w.sum()
    if total.item() <= 0.0:
        raise ValueError("weights must contain positive mass")
    return w / total


def _sphere_vector_to_gamma(vector: Tensor) -> Tensor:
    mu = vector.detach().cpu().double().contiguous()
    if mu.ndim != 1 or mu.numel() < 1:
        raise ValueError("vector must have shape [K-1]")
    # All sample Phase SRVFs lie in the non-negative orthant.  Tiny negative
    # values can arise numerically during sphere log/exp iterations.
    mu = mu.clamp_min(0.0)
    mu = mu / torch.linalg.vector_norm(mu).clamp_min(1e-15)
    increments = mu.square()
    gamma = torch.empty(mu.numel() + 1, dtype=torch.float64)
    gamma[0] = 0.0
    gamma[1:] = torch.cumsum(increments, dim=0)
    gamma[-1] = 1.0
    return gamma.detach()


def weighted_frechet_mean_gamma(
    gammas: Tensor,
    weights: Tensor,
    *,
    max_iterations: int = 256,
    tolerance: float = 1e-10,
) -> WeightedFrechetMeanResult:
    """Weighted Fisher--Rao/Karcher mean on the Phase SRVF unit sphere.

    This minimizes the diagnostic objective ``sum_i w_i d_Gamma(gamma_i,mu)^2``
    using the same discretized Fisher--Rao geometry as ``phase_distance``.
    It is not a pointwise average of gamma functions.
    """
    if not isinstance(gammas, Tensor) or gammas.ndim != 2 or gammas.shape[0] == 0:
        raise ValueError("gammas must have shape [N,K] with N>0")
    points = gamma_to_unit_phase_vectors(gammas).double()
    w = _validate_weights(weights, points.shape[0])
    active = w > 0
    points = points[active]
    w = w[active]
    w = w / w.sum()
    if points.shape[0] == 1:
        gamma = _sphere_vector_to_gamma(points[0])
        return WeightedFrechetMeanResult(gamma, 0.0, 0, True, 0.0)

    initial = (points * w[:, None]).sum(dim=0)
    initial_norm = torch.linalg.vector_norm(initial)
    if not torch.isfinite(initial_norm) or initial_norm.item() <= 1e-15:
        raise RuntimeError("cannot initialize weighted Fisher--Rao mean")
    mu = initial / initial_norm
    converged = False
    last_norm = float("inf")
    iterations = 0

    for iteration in range(1, int(max_iterations) + 1):
        cosine = (points @ mu).clamp(-1.0, 1.0)
        theta = torch.acos(cosine)
        sine = torch.sin(theta)
        scale = torch.where(theta <= 1e-12, torch.ones_like(theta), theta / sine.clamp_min(1e-15))
        tangents = scale[:, None] * (points - cosine[:, None] * mu[None, :])
        tangents = torch.where((theta <= 1e-12)[:, None], torch.zeros_like(tangents), tangents)
        tangent_mean = (tangents * w[:, None]).sum(dim=0)
        norm = torch.linalg.vector_norm(tangent_mean)
        if not torch.isfinite(norm):
            raise RuntimeError("weighted Fisher--Rao mean produced non-finite tangent")
        last_norm = float(norm.item())
        iterations = iteration
        if last_norm <= float(tolerance):
            converged = True
            break
        mu = torch.cos(norm) * mu + torch.sin(norm) * tangent_mean / norm
        mu_norm = torch.linalg.vector_norm(mu)
        if not torch.isfinite(mu_norm) or mu_norm.item() <= 1e-15:
            raise RuntimeError("weighted Fisher--Rao mean produced invalid sphere point")
        mu = mu / mu_norm

    # Evaluate the formal weighted squared-distance objective before converting
    # back to gamma.  Clamping protects only acos numerical precision.
    theta = torch.acos((points @ mu).clamp(-1.0, 1.0))
    _sphere_objective = float((w * theta.square()).sum().item())
    gamma = _sphere_vector_to_gamma(mu)
    objective = weighted_objective(gammas, gamma, weights)
    return WeightedFrechetMeanResult(
        gamma=gamma,
        objective=objective,
        iterations=iterations,
        converged=converged,
        tangent_norm=last_norm,
    )


def center_phase_distance_matrix(gammas: Tensor) -> Tensor:
    """Pairwise formal Fisher--Rao distances for diagnostic center gammas."""
    vectors = gamma_to_unit_phase_vectors(gammas).double()
    inner = vectors @ vectors.T
    distance = torch.acos(inner.clamp(-1.0, 1.0))
    distance.fill_diagonal_(0.0)
    return distance.detach().cpu().double()


def phase_distance_to_identity_for_gamma(gamma: Tensor) -> float:
    if gamma.ndim != 1:
        raise ValueError("gamma must have shape [K]")
    identity = torch.linspace(0.0, 1.0, gamma.numel(), dtype=torch.float64)
    pair = torch.stack([gamma.detach().cpu().double(), identity], dim=0)
    return float(center_phase_distance_matrix(pair)[0, 1].item())


def weighted_objective(gammas: Tensor, center: Tensor, weights: Tensor) -> float:
    """Evaluate ``sum_i w_i d_Gamma(gamma_i,center)^2`` exactly in the discretization."""
    points = gamma_to_unit_phase_vectors(gammas).double()
    center_vec = gamma_to_unit_phase_vectors(center.detach().cpu().double().unsqueeze(0))[0]
    w = _validate_weights(weights, points.shape[0])
    theta = torch.acos((points @ center_vec).clamp(-1.0, 1.0))
    return float((w * theta.square()).sum().item())
