"""Pure statistics helpers for experiment 13A candidate-evidence diagnostics.

All helpers operate on already generated label-free observables plus oracle-only
correctness arrays.  They never train a model, select a pseudo-label class, or
construct a TRAINABLE gate.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def prediction_margin(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must have shape [N,C], C>=2")
    part = np.partition(p, p.shape[1] - 2, axis=1)
    return part[:, -1] - part[:, -2]


def prediction_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("probabilities must have shape [N,C]")
    q = np.clip(p, eps, 1.0)
    return -np.sum(q * np.log(q), axis=1)


def jensen_shannon_rows(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(p, dtype=np.float64)
    b = np.asarray(q, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("posterior arrays must share shape [N,C]")
    a = np.clip(a, eps, 1.0); a = a / a.sum(axis=1, keepdims=True)
    b = np.clip(b, eps, 1.0); b = b / b.sum(axis=1, keepdims=True)
    m = 0.5 * (a + b)
    kl_a = np.sum(a * (np.log(a) - np.log(m)), axis=1)
    kl_b = np.sum(b * (np.log(b) - np.log(m)), axis=1)
    return 0.5 * (kl_a + kl_b)


def candidate_margin(values: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.int64)
    if arr.ndim != 2 or c.shape != (arr.shape[0],):
        raise ValueError("values/candidate dimensions do not match")
    own = arr[np.arange(arr.shape[0]), c]
    masked = arr.copy()
    masked[np.arange(arr.shape[0]), c] = -np.inf
    return own - np.max(masked, axis=1)


def cosine_similarity_matrix(features: np.ndarray, references: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    r = np.asarray(references, dtype=np.float64)
    if x.ndim != 2 or r.ndim != 2 or x.shape[1] != r.shape[1]:
        raise ValueError("features/references must have shape [N,D]/[C,D]")
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), eps)
    return x @ r.T


def distribution_summary(values: Sequence[float]) -> dict:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {key: float("nan") for key in ("mean", "std", "median", "q25", "q75", "iqr")}
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def bootstrap_mean_ci(values: Sequence[float], *, seed: int, reps: int = 500) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or int(reps) <= 0:
        return float("nan"), float("nan")
    if x.size == 1:
        value = float(x[0]); return value, value
    rng = np.random.default_rng(int(seed))
    # Chunk replicates to avoid a large temporary array for very large pools.
    means: list[np.ndarray] = []
    remaining = int(reps)
    while remaining:
        take = min(100, remaining)
        indices = rng.integers(0, x.size, size=(take, x.size))
        means.append(x[indices].mean(axis=1))
        remaining -= take
    boot = np.concatenate(means)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    keep = np.isfinite(s)
    y = y[keep]; s = s[keep]
    n_pos = int(y.sum()); n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    rank_sum = float(ranks[y].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    keep = np.isfinite(s)
    y = y[keep]; s = s[keep]
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    # Tie-aware average precision: add an entire equal-score block at once,
    # so binary/discrete evidence is not affected by arbitrary stable ordering.
    order = np.argsort(-s, kind="mergesort")
    y_ord = y[order]
    s_ord = s[order]
    tp = 0
    seen = 0
    ap = 0.0
    start = 0
    recall_prev = 0.0
    while start < s_ord.size:
        end = start + 1
        while end < s_ord.size and s_ord[end] == s_ord[start]:
            end += 1
        tp += int(y_ord[start:end].sum())
        seen += int(end - start)
        recall = tp / positives
        precision = tp / seen
        ap += (recall - recall_prev) * precision
        recall_prev = recall
        start = end
    return float(ap)


def precision_at_coverages(labels: Sequence[bool], scores: Sequence[float], coverages=(0.05, 0.10, 0.20, 0.30, 0.50, 1.0)) -> list[dict]:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    keep = np.isfinite(s)
    y = y[keep]; s = s[keep]
    if y.size == 0:
        return [{"coverage": float(c), "n_selected": 0, "precision": float("nan")} for c in coverages]
    order = np.argsort(-s, kind="mergesort")
    rows = []
    for coverage in coverages:
        n = max(1, min(y.size, int(math.ceil(float(coverage) * y.size))))
        selected = y[order[:n]]
        rows.append({"coverage": float(coverage), "n_selected": int(n), "precision": float(np.mean(selected))})
    return rows


def wrong_reject_at_correct_retention(labels: Sequence[bool], scores: Sequence[float], retention: float) -> dict:
    if not 0 < retention <= 1:
        raise ValueError("retention must lie in (0,1]")
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    keep = np.isfinite(s)
    y = y[keep]; s = s[keep]
    correct = s[y]; wrong = s[~y]
    if correct.size == 0 or wrong.size == 0:
        return {"threshold": float("nan"), "correct_retention": float("nan"), "wrong_reject_rate": float("nan")}
    # Reliability score: larger is better. Reject the low-score tail.
    threshold = float(np.quantile(correct, 1.0 - retention, method="lower"))
    return {
        "threshold": threshold,
        "correct_retention": float(np.mean(correct >= threshold)),
        "wrong_reject_rate": float(np.mean(wrong < threshold)),
    }


def confidence_quantile_masks(raw_prob: Sequence[float]) -> Mapping[str, np.ndarray]:
    x = np.asarray(raw_prob, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("raw_prob must be one-dimensional")
    if x.size == 0:
        return {name: np.zeros(0, dtype=bool) for name in ("all", "q0_25", "q25_50", "q50_75", "q75_90", "q90_100", "top25", "top10")}
    q25, q50, q75, q90 = np.quantile(x, [0.25, 0.50, 0.75, 0.90])
    return {
        "all": np.ones(x.size, dtype=bool),
        "q0_25": x <= q25,
        "q25_50": (x > q25) & (x <= q50),
        "q50_75": (x > q50) & (x <= q75),
        "q75_90": (x > q75) & (x <= q90),
        "q90_100": x > q90,
        "top25": x >= q75,
        "top10": x >= q90,
    }


def standardized_mean_difference(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64); y = np.asarray(b, dtype=np.float64)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    vx = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
    vy = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    pooled = math.sqrt(max(0.5 * (vx + vy), 0.0))
    if pooled <= 1e-12:
        return 0.0 if abs(float(np.mean(x) - np.mean(y))) <= 1e-12 else float("inf")
    return float((np.mean(x) - np.mean(y)) / pooled)


def confidence_matched_indices(
    correct: Sequence[bool], raw_prob: Sequence[float], raw_margin: Sequence[float], *, seed: int, bins: int = 10
) -> tuple[np.ndarray, dict]:
    """Deterministic within-pool 2-D coarsened exact matching.

    Correct and wrong candidates are matched only within the same joint
    quantile cell of raw candidate probability and raw top1-top2 margin.  The
    caller invokes this separately per raw candidate class, so no cross-class
    matching can occur.
    """
    y = np.asarray(correct, dtype=bool)
    p = np.asarray(raw_prob, dtype=np.float64)
    m = np.asarray(raw_margin, dtype=np.float64)
    if y.shape != p.shape or y.shape != m.shape or y.ndim != 1:
        raise ValueError("matching arrays must share shape [N]")
    valid = np.isfinite(p) & np.isfinite(m)
    if not np.any(valid) or y[valid].sum() == 0 or (~y[valid]).sum() == 0:
        return np.zeros(y.size, dtype=bool), {
            "n_correct_before": int(np.sum(y & valid)), "n_wrong_before": int(np.sum((~y) & valid)),
            "n_correct_matched": 0, "n_wrong_matched": 0,
            "raw_prob_smd_before": standardized_mean_difference(p[y & valid], p[(~y) & valid]),
            "raw_prob_smd_after": float("nan"),
            "raw_margin_smd_before": standardized_mean_difference(m[y & valid], m[(~y) & valid]),
            "raw_margin_smd_after": float("nan"),
        }
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)
    p_edges = np.unique(np.quantile(p[valid], quantiles))
    m_edges = np.unique(np.quantile(m[valid], quantiles))
    # digitize internal boundaries only; repeated quantiles are harmless.
    p_bin = np.digitize(p, p_edges[1:-1], right=True)
    m_bin = np.digitize(m, m_edges[1:-1], right=True)
    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    for pb in np.unique(p_bin[valid]):
        for mb in np.unique(m_bin[valid]):
            cell = valid & (p_bin == pb) & (m_bin == mb)
            pos = np.flatnonzero(cell & y)
            neg = np.flatnonzero(cell & ~y)
            n = min(pos.size, neg.size)
            if n == 0:
                continue
            if pos.size > n:
                pos = rng.choice(pos, size=n, replace=False)
            if neg.size > n:
                neg = rng.choice(neg, size=n, replace=False)
            chosen.extend(np.asarray(pos, dtype=np.int64).tolist())
            chosen.extend(np.asarray(neg, dtype=np.int64).tolist())
    mask = np.zeros(y.size, dtype=bool)
    if chosen:
        mask[np.asarray(chosen, dtype=np.int64)] = True
    return mask, {
        "n_correct_before": int(np.sum(y & valid)), "n_wrong_before": int(np.sum((~y) & valid)),
        "n_correct_matched": int(np.sum(y & mask)), "n_wrong_matched": int(np.sum((~y) & mask)),
        "raw_prob_smd_before": standardized_mean_difference(p[y & valid], p[(~y) & valid]),
        "raw_prob_smd_after": standardized_mean_difference(p[y & mask], p[(~y) & mask]),
        "raw_margin_smd_before": standardized_mean_difference(m[y & valid], m[(~y) & valid]),
        "raw_margin_smd_after": standardized_mean_difference(m[y & mask], m[(~y) & mask]),
        "raw_prob_mean_correct_after": float(np.mean(p[y & mask])) if np.any(y & mask) else float("nan"),
        "raw_prob_mean_wrong_after": float(np.mean(p[(~y) & mask])) if np.any((~y) & mask) else float("nan"),
        "raw_margin_mean_correct_after": float(np.mean(m[y & mask])) if np.any(y & mask) else float("nan"),
        "raw_margin_mean_wrong_after": float(np.mean(m[(~y) & mask])) if np.any((~y) & mask) else float("nan"),
    }


def summarize_evidence(
    values: Sequence[float], correctness: Sequence[bool], *, higher_is_reliable: bool, seed: int, bootstrap_reps: int = 500
) -> dict:
    raw = np.asarray(values, dtype=np.float64)
    y = np.asarray(correctness, dtype=bool)
    if raw.shape != y.shape:
        raise ValueError("evidence/correctness arrays must share shape")
    finite = np.isfinite(raw)
    raw = raw[finite]; y = y[finite]
    score = raw if higher_is_reliable else -raw
    correct_values = raw[y]; wrong_values = raw[~y]
    csum = distribution_summary(correct_values); wsum = distribution_summary(wrong_values)
    c_lo, c_hi = bootstrap_mean_ci(correct_values, seed=seed + 1, reps=bootstrap_reps)
    w_lo, w_hi = bootstrap_mean_ci(wrong_values, seed=seed + 2, reps=bootstrap_reps)
    diff_values = None
    if correct_values.size and wrong_values.size and int(bootstrap_reps) > 0:
        rng = np.random.default_rng(seed + 3)
        diffs: list[np.ndarray] = []
        remaining = int(bootstrap_reps)
        while remaining:
            take = min(50, remaining)
            ci = rng.integers(0, correct_values.size, size=(take, correct_values.size))
            wi = rng.integers(0, wrong_values.size, size=(take, wrong_values.size))
            diffs.append(correct_values[ci].mean(axis=1) - wrong_values[wi].mean(axis=1))
            remaining -= take
        boot_diff = np.concatenate(diffs)
        diff_lo, diff_hi = map(float, np.quantile(boot_diff, [0.025, 0.975]))
        diff_values = (float(np.mean(correct_values) - np.mean(wrong_values)), diff_lo, diff_hi)
    elif correct_values.size and wrong_values.size:
        diff_values = (float(np.mean(correct_values) - np.mean(wrong_values)), float("nan"), float("nan"))
    else:
        diff_values = (float("nan"), float("nan"), float("nan"))
    out = {
        "n": int(raw.size), "n_correct": int(y.sum()), "n_wrong": int((~y).sum()),
        "base_precision": float(np.mean(y)) if y.size else float("nan"),
        "correct_mean": csum["mean"], "correct_std": csum["std"], "correct_median": csum["median"],
        "correct_q25": csum["q25"], "correct_q75": csum["q75"], "correct_iqr": csum["iqr"],
        "wrong_mean": wsum["mean"], "wrong_std": wsum["std"], "wrong_median": wsum["median"],
        "wrong_q25": wsum["q25"], "wrong_q75": wsum["q75"], "wrong_iqr": wsum["iqr"],
        "correct_mean_ci_low": c_lo, "correct_mean_ci_high": c_hi,
        "wrong_mean_ci_low": w_lo, "wrong_mean_ci_high": w_hi,
        "mean_difference_correct_minus_wrong": diff_values[0],
        "mean_difference_ci_low": diff_values[1], "mean_difference_ci_high": diff_values[2],
        "effect_direction": "higher_in_correct" if diff_values[0] > 0 else ("lower_in_correct" if diff_values[0] < 0 else "equal"),
        "higher_is_reliable": bool(higher_is_reliable),
        "auroc": auroc(y, score), "auprc": auprc(y, score),
    }
    for retention in (0.95, 0.90):
        veto = wrong_reject_at_correct_retention(y, score, retention)
        suffix = int(round(retention * 100))
        out[f"wrong_reject_at_correct_retention_{suffix}"] = veto["wrong_reject_rate"]
        out[f"achieved_correct_retention_{suffix}"] = veto["correct_retention"]
    return out


def rankdata(values: Sequence[float]) -> np.ndarray:
    return _average_ranks(np.asarray(values, dtype=np.float64))


def spearman_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64); y = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    rx = rankdata(x[keep]); ry = rankdata(y[keep])
    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def two_by_two_precision(a: Sequence[float], b: Sequence[float], correctness: Sequence[bool]) -> list[dict]:
    x = np.asarray(a, dtype=np.float64); y = np.asarray(b, dtype=np.float64); c = np.asarray(correctness, dtype=bool)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]; y = y[keep]; c = c[keep]
    if x.size == 0:
        return []
    ax = float(np.median(x)); by = float(np.median(y))
    rows = []
    for a_high in (False, True):
        for b_high in (False, True):
            mask = (x >= ax if a_high else x < ax) & (y >= by if b_high else y < by)
            rows.append({
                "a_high": bool(a_high), "b_high": bool(b_high), "n": int(mask.sum()),
                "precision": float(np.mean(c[mask])) if np.any(mask) else float("nan"),
                "a_median_threshold": ax, "b_median_threshold": by,
            })
    return rows
