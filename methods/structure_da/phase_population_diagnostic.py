"""Pure Phase-population helpers for experiment 09.

The functions in this module only describe the population of already-computed
monotone gamma warps.  They never solve registrations, filter samples, cluster
warps, choose a group count, or construct representative Domain Phase values.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


def gamma_to_unit_phase_vectors(gammas: Tensor) -> Tensor:
    """Map ``gammas[N,K]`` to Euclidean unit vectors for Fisher--Rao geometry.

    ``phase_geometry.phase_distance`` uses normalized ``sqrt(gamma_dot)`` with
    the discretized L2 inner product. Multiplication by ``sqrt(step)`` converts
    those functions into ordinary Euclidean unit vectors, so
    ``acos(v_i @ v_j)`` is exactly the same discretized Phase distance.
    """
    if not isinstance(gammas, Tensor) or gammas.ndim != 2 or gammas.shape[1] < 2:
        raise ValueError("gammas must have shape [N,K] with K>=2")
    values = gammas.detach().cpu().double().contiguous()
    if values.shape[0] == 0 or not torch.isfinite(values).all().item():
        raise ValueError("gammas must be finite and non-empty")
    interval_count = int(values.shape[1] - 1)
    derivative = torch.diff(values, dim=1) * interval_count
    if torch.any(derivative < -1e-12).item():
        raise ValueError("gammas must be monotonically nondecreasing")
    psi = torch.sqrt(derivative.clamp_min(0.0))
    step = 1.0 / interval_count
    norms = torch.sqrt((psi.square().sum(dim=1) * step).clamp_min(1e-15))
    psi = psi / norms[:, None]
    vectors = psi * math.sqrt(step)
    vectors = vectors / torch.linalg.vector_norm(vectors, dim=1, keepdim=True).clamp_min(1e-15)
    return vectors.detach()


def phase_distance_to_identity(vectors: Tensor) -> Tensor:
    """Fisher--Rao distance from each Phase vector to ``id(t)=t``."""
    if vectors.ndim != 2 or vectors.shape[1] < 1:
        raise ValueError("vectors must have shape [N,D]")
    values = vectors.detach().cpu().double()
    identity = torch.full(
        (values.shape[1],), 1.0 / math.sqrt(values.shape[1]), dtype=torch.float64
    )
    inner = values @ identity
    return torch.acos(inner.clamp(-1.0, 1.0)).detach()


def phase_pair_distances(vectors: Tensor, left: Sequence[int], right: Sequence[int]) -> np.ndarray:
    """Return formal Fisher--Rao distances for paired row indices."""
    if len(left) != len(right):
        raise ValueError("left/right lengths must match")
    values = vectors.detach().cpu().double()
    li = torch.as_tensor(left, dtype=torch.long)
    ri = torch.as_tensor(right, dtype=torch.long)
    inner = (values[li] * values[ri]).sum(dim=1)
    return torch.acos(inner.clamp(-1.0, 1.0)).numpy()


def sample_within_cross_pairs(
    labels: Sequence[int], *, count_each: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample label-blind population pairs, then retain within/cross strata.

    Pair proposals are uniform over sample indices. Labels are used only after
    proposal to separate the two oracle-only audit strata. No Phase sample is
    filtered based on class or downstream outcome.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.ndim != 1 or labels_arr.size < 2:
        raise ValueError("labels must contain at least two samples")
    count_each = int(count_each)
    if count_each <= 0:
        raise ValueError("count_each must be positive")
    rng = np.random.default_rng(int(seed))
    within_l: list[np.ndarray] = []
    within_r: list[np.ndarray] = []
    cross_l: list[np.ndarray] = []
    cross_r: list[np.ndarray] = []
    nw = nc = 0
    n = labels_arr.size
    while nw < count_each or nc < count_each:
        batch = max(8192, 4 * (count_each - min(nw, count_each) + count_each - min(nc, count_each)))
        batch = min(batch, 1_000_000)
        left = rng.integers(0, n, size=batch, dtype=np.int64)
        right = rng.integers(0, n, size=batch, dtype=np.int64)
        distinct = left != right
        left, right = left[distinct], right[distinct]
        same = labels_arr[left] == labels_arr[right]
        if nw < count_each and np.any(same):
            take = min(count_each - nw, int(np.sum(same)))
            within_l.append(left[same][:take]); within_r.append(right[same][:take]); nw += take
        if nc < count_each and np.any(~same):
            take = min(count_each - nc, int(np.sum(~same)))
            cross_l.append(left[~same][:take]); cross_r.append(right[~same][:take]); nc += take
    return (
        np.concatenate(within_l)[:count_each], np.concatenate(within_r)[:count_each],
        np.concatenate(cross_l)[:count_each], np.concatenate(cross_r)[:count_each],
    )


def common_language_effect(within: np.ndarray, cross: np.ndarray) -> dict[str, float]:
    """Effect size for whether a within-class Phase distance is smaller.

    ``prob_within_smaller`` estimates P(D_within < D_cross), with half credit
    for ties. ``rank_biserial`` maps that probability to [-1,1]; positive
    values mean within-class distances tend to be smaller.
    """
    a = np.asarray(within, dtype=np.float64)
    b = np.asarray(cross, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {"prob_within_smaller": float("nan"), "rank_biserial": float("nan")}
    bs = np.sort(b)
    less = np.searchsorted(bs, a, side="left")
    leq = np.searchsorted(bs, a, side="right")
    greater = b.size - leq
    ties = leq - less
    probability = float(np.mean((greater + 0.5 * ties) / b.size))
    return {"prob_within_smaller": probability, "rank_biserial": 2.0 * probability - 1.0}


def exact_phase_knn(
    vectors: Tensor,
    *,
    k_max: int,
    device: torch.device,
    block_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact kNN under Fisher--Rao Phase distance without a full NxN matrix.

    Neighbor order under ``acos(v_i @ v_j)`` is exactly reverse inner-product
    order. The computation is blockwise and stores only the k nearest rows.
    Float32 matrix products are used for tractability; returned distances are
    the same Fisher--Rao formula, not Euclidean gamma distance.
    """
    values = vectors.detach().to(device=device, dtype=torch.float32)
    n = int(values.shape[0])
    k_max = int(k_max)
    if not 1 <= k_max < n:
        raise ValueError("k_max must lie in [1,N-1]")
    block_size = max(1, int(block_size))
    all_indices = np.empty((n, k_max), dtype=np.int64)
    all_distances = np.empty((n, k_max), dtype=np.float32)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        similarity = values[start:stop] @ values.T
        local_rows = torch.arange(stop - start, device=device)
        global_rows = torch.arange(start, stop, device=device)
        similarity[local_rows, global_rows] = -torch.inf
        top_similarity, top_indices = torch.topk(similarity, k=k_max, dim=1, largest=True, sorted=True)
        distances = torch.acos(top_similarity.clamp(-1.0, 1.0))
        all_indices[start:stop] = top_indices.detach().cpu().numpy()
        all_distances[start:stop] = distances.detach().cpu().numpy()
    return all_indices, all_distances


def knn_label_statistics(
    neighbor_indices: np.ndarray,
    labels: Sequence[int],
    *,
    k_values: Sequence[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Oracle-only same-class fraction and class entropy for Phase neighbors."""
    neighbors = np.asarray(neighbor_indices, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if neighbors.ndim != 2 or neighbors.shape[0] != labels_arr.size:
        raise ValueError("neighbor_indices must have shape [N,K]")
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in sorted(set(int(v) for v in k_values)):
        if not 1 <= k <= neighbors.shape[1]:
            raise ValueError("each k must lie within available neighbors")
        neighbor_labels = labels_arr[neighbors[:, :k]]
        same = np.mean(neighbor_labels == labels_arr[:, None], axis=1)
        entropy = np.zeros(labels_arr.size, dtype=np.float64)
        for row in range(labels_arr.size):
            _classes, counts = np.unique(neighbor_labels[row], return_counts=True)
            probs = counts.astype(np.float64) / float(k)
            entropy[row] = -float(np.sum(probs * np.log(probs)))
        results[k] = (same.astype(np.float64), entropy)
    return results


def deterministic_random_subset(n: int, size: int, seed: int) -> np.ndarray:
    n, size = int(n), int(size)
    if n <= 0 or size <= 0:
        raise ValueError("n and size must be positive")
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=min(n, size), replace=False).astype(np.int64))


def deterministic_equal_class_subset(
    labels: Sequence[int], *, per_class: int, seed: int
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.ndim != 1 or labels_arr.size == 0:
        raise ValueError("labels must be non-empty")
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for class_id in sorted(np.unique(labels_arr).tolist()):
        indices = np.flatnonzero(labels_arr == class_id)
        count = min(int(per_class), int(indices.size))
        selected.append(rng.choice(indices, size=count, replace=False).astype(np.int64))
    return np.sort(np.concatenate(selected))
