"""Pure helpers for experiment 12 Stage-2 bootstrap temporal-state diagnostics.

This module contains only offline metric/statistics helpers.  It does not train
models, update pseudo labels, estimate Phase from semantic predictions, or
select a bootstrap state from target metrics.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _entropy_rows(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape [N,C]")
    clipped = np.clip(probs, eps, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def prediction_margins(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("probabilities must have shape [N,C] with C>=2")
    partitioned = np.partition(probs, kth=probs.shape[1] - 2, axis=1)
    return partitioned[:, -1] - partitioned[:, -2]


def confusion_matrix(labels: Sequence[int], predictions: Sequence[int], num_classes: int) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("labels and predictions must have matching shape [N]")
    matrix = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for truth, pred in zip(y.tolist(), p.tolist()):
        if not 0 <= truth < num_classes or not 0 <= pred < num_classes:
            raise ValueError("class id outside confusion-matrix range")
        matrix[truth, pred] += 1
    return matrix


def row_normalize_confusion(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("confusion matrix must be square")
    denominator = values.sum(axis=1, keepdims=True)
    return np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)


def per_class_metrics(labels: Sequence[int], probabilities: np.ndarray, class_names: Sequence[str]) -> list[dict]:
    y = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != y.size or probs.shape[1] != len(class_names):
        raise ValueError("label/probability/class dimensions do not match")
    pred = probs.argmax(axis=1)
    max_prob = probs.max(axis=1)
    margin = prediction_margins(probs)
    entropy = _entropy_rows(probs)
    rows: list[dict] = []
    for class_id, class_name in enumerate(class_names):
        true_mask = y == class_id
        pred_mask = pred == class_id
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        support = int(true_mask.sum())
        pred_count = int(pred_mask.sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        true_prob = probs[true_mask, class_id]
        rows.append({
            "class_id": int(class_id),
            "class_name": str(class_name),
            "true_support": support,
            "predicted_count": pred_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_true_class_probability": float(np.mean(true_prob)) if support else float("nan"),
            "median_true_class_probability": float(np.median(true_prob)) if support else float("nan"),
            "mean_max_probability": float(np.mean(max_prob[true_mask])) if support else float("nan"),
            "mean_prediction_margin": float(np.mean(margin[true_mask])) if support else float("nan"),
            "mean_prediction_entropy": float(np.mean(entropy[true_mask])) if support else float("nan"),
            "correct_count": tp,
            "wrong_count": int(support - tp),
            "predicted_to_true_support_ratio": float(pred_count / support) if support else float("nan"),
        })
    return rows


def overall_metrics(labels: Sequence[int], probabilities: np.ndarray, class_names: Sequence[str]) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    rows = per_class_metrics(y, probs, class_names)
    pred = probs.argmax(axis=1)
    true_prob = probs[np.arange(y.size), y]
    max_prob = probs.max(axis=1)
    margin = prediction_margins(probs)
    entropy = _entropy_rows(probs)
    counts = np.bincount(pred, minlength=len(class_names)).astype(np.float64)
    dist = counts / max(float(counts.sum()), 1.0)
    nonzero = dist > 0
    predicted_entropy = float(-np.sum(dist[nonzero] * np.log(dist[nonzero])))
    return {
        "accuracy": float(np.mean(pred == y)) if y.size else float("nan"),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "mean_true_class_probability": float(np.mean(true_prob)) if y.size else float("nan"),
        "mean_max_probability": float(np.mean(max_prob)) if y.size else float("nan"),
        "mean_prediction_margin": float(np.mean(margin)) if y.size else float("nan"),
        "mean_prediction_entropy": float(np.mean(entropy)) if y.size else float("nan"),
        "predicted_class_entropy": predicted_entropy,
        "largest_predicted_class_fraction": float(np.max(dist)) if y.size else float("nan"),
        "n_samples": int(y.size),
    }


def add_identity_deltas(condition_rows: list[dict], identity_rows: Sequence[dict]) -> list[dict]:
    identity = {int(row["class_id"]): row for row in identity_rows}
    result: list[dict] = []
    for row in condition_rows:
        base = identity[int(row["class_id"])]
        enriched = dict(row)
        enriched.update({
            "delta_recall_vs_identity": float(row["recall"] - base["recall"]),
            "delta_precision_vs_identity": float(row["precision"] - base["precision"]),
            "delta_f1_vs_identity": float(row["f1"] - base["f1"]),
            "delta_true_prob_vs_identity": float(
                row["mean_true_class_probability"] - base["mean_true_class_probability"]
            ),
        })
        result.append(enriched)
    return result


def hard_transition_rows(
    labels: Sequence[int],
    before_probabilities: np.ndarray,
    after_probabilities: np.ndarray,
    class_names: Sequence[str],
) -> list[dict]:
    y = np.asarray(labels, dtype=np.int64)
    before = np.asarray(before_probabilities).argmax(axis=1)
    after = np.asarray(after_probabilities).argmax(axis=1)
    if y.size != before.size or y.size != after.size:
        raise ValueError("transition arrays have inconsistent lengths")
    rows: list[dict] = []
    scopes: list[tuple[int | None, str]] = [(None, "ALL")] + [
        (index, str(name)) for index, name in enumerate(class_names)
    ]
    for class_id, class_name in scopes:
        mask = np.ones(y.size, dtype=bool) if class_id is None else y == int(class_id)
        before_correct = before[mask] == y[mask]
        after_correct = after[mask] == y[mask]
        wc = int(np.sum(~before_correct & after_correct))
        cw = int(np.sum(before_correct & ~after_correct))
        cc = int(np.sum(before_correct & after_correct))
        ww = int(np.sum(~before_correct & ~after_correct))
        rows.append({
            "class_id": "ALL" if class_id is None else int(class_id),
            "class_name": class_name,
            "n_samples": int(mask.sum()),
            "wrong_to_correct": wc,
            "correct_to_wrong": cw,
            "correct_to_correct": cc,
            "wrong_to_wrong": ww,
            "net_correct_gain": int(wc - cw),
        })
    return rows


def distribution_summary(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {key: float("nan") for key in ("mean", "median", "q10", "q25", "q75", "q90")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
    }


def classwise_semantic_distribution(
    labels: Sequence[int], probabilities: np.ndarray, class_names: Sequence[str]
) -> list[dict]:
    y = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    max_prob = probs.max(axis=1)
    margin = prediction_margins(probs)
    entropy = _entropy_rows(probs)
    true_prob = probs[np.arange(y.size), y]
    metrics = {
        "max_probability": max_prob,
        "prediction_margin": margin,
        "prediction_entropy": entropy,
        "true_class_probability": true_prob,
    }
    rows: list[dict] = []
    for class_id, class_name in enumerate(class_names):
        mask = y == class_id
        for metric_name, values in metrics.items():
            summary = distribution_summary(values[mask])
            rows.append({
                "class_id": int(class_id),
                "class_name": str(class_name),
                "metric": metric_name,
                "n_samples": int(mask.sum()),
                **summary,
            })
    return rows
