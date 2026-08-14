"""Frozen-feature structure helpers for experiment 13A-2.

The module only computes label-free observables from frozen source/target LTAE
features and fixed raw-candidate assignments.  Target true labels are not an
input to any function here.  Oracle correctness is consumed only by the
separate summary helpers at the bottom of the module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .candidate_evidence_diagnostic import spearman_correlation


DEFAULT_K_VALUES = (5, 10, 20, 50)


def _as_2d_float(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty shape [N,D]")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = _as_2d_float(values, "features")
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, eps)


def cosine_distance_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("a/b must share shape [N,D]")
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    yn = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), eps)
    return 1.0 - np.sum(xn * yn, axis=1)


def class_centroids(features: np.ndarray, labels: Sequence[int], num_classes: int) -> np.ndarray:
    x = _as_2d_float(features, "features").astype(np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (x.shape[0],):
        raise ValueError("labels must have shape [N]")
    out = np.full((int(num_classes), x.shape[1]), np.nan, dtype=np.float64)
    for cid in range(int(num_classes)):
        mask = y == cid
        if np.any(mask):
            out[cid] = x[mask].mean(axis=0)
    if not np.isfinite(out).all():
        raise ValueError("every source class must have at least one feature")
    return out


def loo_class_centroid_distances(
    features: np.ndarray, labels: Sequence[int], num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine distances to leave-one-out source class centroids.

    Returns ``(distances, full_centroids)``.  The LOO distances form the source
    reference distribution; target samples are compared with the full source
    class centroid because the target sample is not a source member.
    """
    x = _as_2d_float(features, "features").astype(np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (x.shape[0],):
        raise ValueError("labels must have shape [N]")
    sums = np.zeros((int(num_classes), x.shape[1]), dtype=np.float64)
    counts = np.zeros(int(num_classes), dtype=np.int64)
    for cid in range(int(num_classes)):
        mask = y == cid
        sums[cid] = x[mask].sum(axis=0)
        counts[cid] = int(mask.sum())
    if np.any(counts < 2):
        raise ValueError("LOO source calibration requires at least two samples per class")
    loo_centers = (sums[y] - x) / (counts[y, None] - 1)
    distances = cosine_distance_rows(x, loo_centers)
    centroids = sums / counts[:, None]
    return distances.astype(np.float64), centroids.astype(np.float64)


def empirical_percentiles_by_class(
    reference_values: Sequence[float], reference_classes: Sequence[int],
    query_values: Sequence[float], query_classes: Sequence[int], num_classes: int,
) -> np.ndarray:
    ref = np.asarray(reference_values, dtype=np.float64)
    ref_cls = np.asarray(reference_classes, dtype=np.int64)
    query = np.asarray(query_values, dtype=np.float64)
    query_cls = np.asarray(query_classes, dtype=np.int64)
    if ref.shape != ref_cls.shape or query.shape != query_cls.shape:
        raise ValueError("value/class arrays must share shape")
    out = np.full(query.shape, np.nan, dtype=np.float64)
    for cid in range(int(num_classes)):
        samples = np.sort(ref[(ref_cls == cid) & np.isfinite(ref)])
        mask = (query_cls == cid) & np.isfinite(query)
        if samples.size == 0 or not np.any(mask):
            continue
        out[mask] = np.searchsorted(samples, query[mask], side="right") / samples.size
    return out


@dataclass(frozen=True)
class CosineKNNResult:
    indices: np.ndarray
    distances: np.ndarray
    nearest_same_distance: np.ndarray | None
    nearest_other_distance: np.ndarray | None


def exact_cosine_knn(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    k_max: int,
    device: torch.device | str = "cpu",
    chunk_size: int = 1024,
    exclude_self: bool = False,
    query_reference_indices: Sequence[int] | None = None,
    query_groups: Sequence[int] | None = None,
    reference_groups: Sequence[int] | None = None,
) -> CosineKNNResult:
    """Exact chunked cosine KNN without FAISS/sklearn dependency.

    ``exclude_self`` is valid only when every query has a corresponding row in
    the reference matrix, supplied through ``query_reference_indices``.
    Optional group arrays additionally return exact nearest same-group and
    other-group distances using the same similarity matrix.
    """
    q = _normalize_rows(query_features)
    r = _normalize_rows(reference_features)
    if int(k_max) < 1 or int(chunk_size) < 1:
        raise ValueError("k_max and chunk_size must be positive")
    available = r.shape[0] - (1 if exclude_self else 0)
    if int(k_max) > available:
        raise ValueError("k_max exceeds available reference neighbours")
    if exclude_self:
        if query_reference_indices is None:
            if q.shape[0] != r.shape[0]:
                raise ValueError("self exclusion needs query_reference_indices")
            query_reference_indices = np.arange(q.shape[0], dtype=np.int64)
        qri = np.asarray(query_reference_indices, dtype=np.int64)
        if qri.shape != (q.shape[0],):
            raise ValueError("query_reference_indices must have shape [N_query]")
    else:
        qri = None
    use_groups = query_groups is not None or reference_groups is not None
    if use_groups:
        if query_groups is None or reference_groups is None:
            raise ValueError("query_groups/reference_groups must be supplied together")
        qg = np.asarray(query_groups, dtype=np.int64)
        rg = np.asarray(reference_groups, dtype=np.int64)
        if qg.shape != (q.shape[0],) or rg.shape != (r.shape[0],):
            raise ValueError("group arrays do not match feature counts")
    else:
        qg = rg = None

    dev = torch.device(device)
    ref = torch.as_tensor(r, dtype=torch.float32, device=dev)
    ref_groups_t = None if rg is None else torch.as_tensor(rg, dtype=torch.long, device=dev)
    indices_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    same_parts: list[np.ndarray] = []
    other_parts: list[np.ndarray] = []
    for start in range(0, q.shape[0], int(chunk_size)):
        end = min(q.shape[0], start + int(chunk_size))
        qt = torch.as_tensor(q[start:end], dtype=torch.float32, device=dev)
        sim = qt @ ref.T
        if qri is not None:
            local = torch.arange(end - start, device=dev)
            cols = torch.as_tensor(qri[start:end], dtype=torch.long, device=dev)
            sim[local, cols] = -torch.inf
        top_sim, top_idx = torch.topk(sim, k=int(k_max), dim=1, largest=True, sorted=True)
        indices_parts.append(top_idx.detach().cpu().numpy().astype(np.int64))
        distance_parts.append((1.0 - top_sim).detach().cpu().numpy().astype(np.float64))
        if qg is not None and ref_groups_t is not None:
            group = torch.as_tensor(qg[start:end], dtype=torch.long, device=dev)
            same_mask = ref_groups_t.unsqueeze(0) == group.unsqueeze(1)
            if qri is not None:
                local = torch.arange(end - start, device=dev)
                cols = torch.as_tensor(qri[start:end], dtype=torch.long, device=dev)
                same_mask[local, cols] = False
            same_sim = sim.masked_fill(~same_mask, -torch.inf).max(dim=1).values
            other_sim = sim.masked_fill(same_mask, -torch.inf).max(dim=1).values
            same_dist = 1.0 - same_sim
            other_dist = 1.0 - other_sim
            same_dist[~torch.isfinite(same_sim)] = torch.nan
            other_dist[~torch.isfinite(other_sim)] = torch.nan
            same_parts.append(same_dist.detach().cpu().numpy().astype(np.float64))
            other_parts.append(other_dist.detach().cpu().numpy().astype(np.float64))
        del sim, top_sim, top_idx
    return CosineKNNResult(
        indices=np.concatenate(indices_parts, axis=0),
        distances=np.concatenate(distance_parts, axis=0),
        nearest_same_distance=np.concatenate(same_parts) if same_parts else None,
        nearest_other_distance=np.concatenate(other_parts) if other_parts else None,
    )


def categorical_entropy(labels: np.ndarray, num_classes: int) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError("labels must have shape [N,K]")
    out = np.zeros(arr.shape[0], dtype=np.float64)
    for cid in range(int(num_classes)):
        p = np.mean(arr == cid, axis=1)
        mask = p > 0
        out[mask] -= p[mask] * np.log(p[mask])
    return out


def _mutual_fraction(knn_indices: np.ndarray, k: int) -> np.ndarray:
    nbr = np.asarray(knn_indices[:, :int(k)], dtype=np.int64)
    n = nbr.shape[0]
    rows = np.repeat(np.arange(n, dtype=np.int64), int(k))
    cols = nbr.reshape(-1)
    keys = rows * n + cols
    reverse = cols * n + rows
    sorted_keys = np.sort(keys)
    pos = np.searchsorted(sorted_keys, reverse)
    found = (pos < sorted_keys.size)
    found[found] &= sorted_keys[pos[found]] == reverse[found]
    return found.reshape(n, int(k)).mean(axis=1)


def source_knn_observables(
    knn: CosineKNNResult,
    source_labels: Sequence[int],
    candidates: Sequence[int],
    *,
    num_classes: int,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, np.ndarray]:
    labels = np.asarray(source_labels, dtype=np.int64)
    cand = np.asarray(candidates, dtype=np.int64)
    if cand.shape != (knn.indices.shape[0],):
        raise ValueError("candidate count does not match KNN queries")
    out: dict[str, np.ndarray] = {}
    for k in k_values:
        nbr_idx = knn.indices[:, :int(k)]
        nbr_dist = knn.distances[:, :int(k)]
        nbr_labels = labels[nbr_idx]
        same = nbr_labels == cand[:, None]
        out[f"source_knn{k}_candidate_fraction"] = same.mean(axis=1)
        out[f"source_knn{k}_label_entropy"] = categorical_entropy(nbr_labels, num_classes)
        candidate_sum = np.where(same, nbr_dist, 0.0).sum(axis=1)
        other_sum = np.where(~same, nbr_dist, 0.0).sum(axis=1)
        n_candidate = same.sum(axis=1)
        n_other = (~same).sum(axis=1)
        candidate_mean = np.divide(candidate_sum, n_candidate, out=np.full(cand.shape, np.nan), where=n_candidate > 0)
        other_mean = np.divide(other_sum, n_other, out=np.full(cand.shape, np.nan), where=n_other > 0)
        out[f"source_knn{k}_candidate_mean_distance"] = candidate_mean
        out[f"source_knn{k}_other_mean_distance"] = other_mean
        out[f"source_knn{k}_candidate_vs_other_distance_margin"] = other_mean - candidate_mean
    if knn.nearest_same_distance is not None:
        out["source_nearest_candidate_distance"] = knn.nearest_same_distance
    if knn.nearest_other_distance is not None:
        out["source_nearest_other_distance"] = knn.nearest_other_distance
    if knn.nearest_same_distance is not None and knn.nearest_other_distance is not None:
        out["source_nearest_candidate_vs_other_margin"] = knn.nearest_other_distance - knn.nearest_same_distance
    return out


def target_knn_observables(
    knn: CosineKNNResult,
    raw_pred: Sequence[int],
    raw_posterior: np.ndarray,
    *,
    num_classes: int,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, np.ndarray]:
    pred = np.asarray(raw_pred, dtype=np.int64)
    post = np.asarray(raw_posterior, dtype=np.float64)
    if pred.shape != (knn.indices.shape[0],) or post.shape[0] != pred.size:
        raise ValueError("target prediction arrays do not match KNN queries")
    out: dict[str, np.ndarray] = {}
    row = np.arange(pred.size)
    for k in k_values:
        nbr_idx = knn.indices[:, :int(k)]
        nbr_dist = knn.distances[:, :int(k)]
        nbr_pred = pred[nbr_idx]
        same = nbr_pred == pred[:, None]
        out[f"target_knn{k}_mean_distance"] = nbr_dist.mean(axis=1)
        out[f"target_knn{k}_median_distance"] = np.median(nbr_dist, axis=1)
        out[f"target_knn{k}_nearest_distance"] = nbr_dist[:, 0]
        out[f"target_knn{k}_mean_cosine_similarity"] = 1.0 - nbr_dist.mean(axis=1)
        out[f"target_knn{k}_candidate_fraction"] = same.mean(axis=1)
        out[f"target_knn{k}_label_entropy"] = categorical_entropy(nbr_pred, num_classes)
        candidate_post = post[nbr_idx, pred[:, None]]
        out[f"target_knn{k}_candidate_posterior_mean"] = candidate_post.mean(axis=1)
        out[f"target_knn{k}_mutual_rate"] = _mutual_fraction(knn.indices, int(k))
    if knn.nearest_same_distance is not None:
        out["target_nearest_same_candidate_distance"] = knn.nearest_same_distance
    if knn.nearest_other_distance is not None:
        out["target_nearest_other_candidate_distance"] = knn.nearest_other_distance
    if knn.nearest_same_distance is not None and knn.nearest_other_distance is not None:
        out["target_neighborhood_margin"] = knn.nearest_other_distance - knn.nearest_same_distance
        ratio = np.divide(
            knn.nearest_other_distance,
            knn.nearest_same_distance + 1e-12,
            out=np.full(pred.shape, np.nan, dtype=np.float64),
            where=np.isfinite(knn.nearest_same_distance) & np.isfinite(knn.nearest_other_distance),
        )
        out["target_neighborhood_distance_ratio"] = ratio
    return out


def pool_centroid_distances(features: np.ndarray, groups: Sequence[int], num_groups: int) -> np.ndarray:
    x = _as_2d_float(features, "features").astype(np.float64)
    g = np.asarray(groups, dtype=np.int64)
    centers = class_centroids(x, g, int(num_groups))
    return cosine_distance_rows(x, centers[g])


def pca_project(features: np.ndarray, components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    x = _as_2d_float(features, "features").astype(np.float64)
    if int(components) < 1 or int(components) > x.shape[1]:
        raise ValueError("invalid PCA component count")
    centered = x - x.mean(axis=0, keepdims=True)
    # D is small for frozen LTAE features; covariance eigendecomposition avoids
    # materialising an N x N matrix for tens of thousands of target parcels.
    cov = centered.T @ centered
    denom = max(x.shape[0] - 1, 1)
    cov /= denom
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval = np.maximum(eigval[order], 0.0)
    basis = eigvec[:, order[:int(components)]]
    projected = centered @ basis
    total = float(eigval.sum())
    ratio = eigval[:int(components)] / total if total > 0 else np.zeros(int(components))
    return projected.astype(np.float64), ratio.astype(np.float64)


def fixed_quantile_grid(
    a: Sequence[float], b: Sequence[float], correctness: Sequence[bool],
    *, a_higher_is_support: bool, b_higher_is_support: bool,
) -> list[dict]:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    c = np.asarray(correctness, dtype=bool)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep] if a_higher_is_support else -x[keep]
    y = y[keep] if b_higher_is_support else -y[keep]
    c = c[keep]
    if x.size == 0:
        return []
    x_edges = np.quantile(x, [0.25, 0.50, 0.75])
    y_edges = np.quantile(y, [0.25, 0.50, 0.75])
    xb = np.digitize(x, x_edges, right=True)
    yb = np.digitize(y, y_edges, right=True)
    rows: list[dict] = []
    for xi in range(4):
        for yi in range(4):
            mask = (xb == xi) & (yb == yi)
            n = int(mask.sum())
            correct_n = int(c[mask].sum()) if n else 0
            rows.append({
                "a_support_quartile": xi + 1,
                "b_support_quartile": yi + 1,
                "n": n,
                "correct_n": correct_n,
                "wrong_n": n - correct_n,
                "precision": float(correct_n / n) if n else float("nan"),
            })
    return rows


def fixed_top_intersections(
    a: Sequence[float], b: Sequence[float], correctness: Sequence[bool],
    *, a_higher_is_support: bool, b_higher_is_support: bool,
    coverages: Sequence[float] = (0.25, 0.10),
) -> list[dict]:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    c = np.asarray(correctness, dtype=bool)
    keep = np.isfinite(x) & np.isfinite(y)
    valid_idx = np.flatnonzero(keep)
    if valid_idx.size == 0:
        return []
    xs = x[keep] if a_higher_is_support else -x[keep]
    ys = y[keep] if b_higher_is_support else -y[keep]
    order_x = np.argsort(-xs, kind="mergesort")
    order_y = np.argsort(-ys, kind="mergesort")
    rows: list[dict] = []
    for coverage in coverages:
        n_each = max(1, int(math.ceil(float(coverage) * valid_idx.size)))
        a_sel = np.zeros(valid_idx.size, dtype=bool); a_sel[order_x[:n_each]] = True
        b_sel = np.zeros(valid_idx.size, dtype=bool); b_sel[order_y[:n_each]] = True
        joint = a_sel & b_sel
        n = int(joint.sum())
        correct_n = int(c[valid_idx][joint].sum()) if n else 0
        rows.append({
            "requested_each_coverage": float(coverage),
            "n_valid": int(valid_idx.size),
            "n_each": int(n_each),
            "joint_n": n,
            "joint_coverage": float(n / valid_idx.size),
            "correct_n": correct_n,
            "wrong_n": n - correct_n,
            "precision": float(correct_n / n) if n else float("nan"),
        })
    return rows


def complementarity_diagnostics(
    a: Sequence[float], b: Sequence[float], correctness: Sequence[bool],
    *, a_higher_is_support: bool, b_higher_is_support: bool,
) -> tuple[float, list[dict], list[dict]]:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    c = np.asarray(correctness, dtype=bool)
    keep = np.isfinite(x) & np.isfinite(y)
    xr = x[keep] if a_higher_is_support else -x[keep]
    yr = y[keep] if b_higher_is_support else -y[keep]
    rho = spearman_correlation(xr, yr)
    grid = fixed_quantile_grid(
        x, y, c, a_higher_is_support=a_higher_is_support, b_higher_is_support=b_higher_is_support,
    )
    joint = fixed_top_intersections(
        x, y, c, a_higher_is_support=a_higher_is_support, b_higher_is_support=b_higher_is_support,
    )
    return rho, grid, joint
