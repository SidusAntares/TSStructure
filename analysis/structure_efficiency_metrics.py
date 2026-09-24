"""Pure numeric metrics for structure-efficiency diagnostics."""

import numpy as np


EPS = 1e-12


def effective_rank(values):
    """Entropy effective rank using singular values."""
    singular = np.linalg.svd(np.asarray(values, dtype=np.float64), compute_uv=False)
    total = singular.sum()
    if total <= EPS:
        return 0.0
    probabilities = singular / total
    return float(np.exp(-(probabilities * np.log(probabilities + EPS)).sum()))


def candidate_effective_number(weights):
    """Effective number of candidates for each sample and anchor."""
    values = np.asarray(weights, dtype=np.float64)
    entropy = -(values * np.log(values + EPS)).sum(axis=1)
    return np.exp(entropy)


def macro_f1(labels, predictions, classes=None):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if classes is None:
        classes = np.unique(np.concatenate((labels, predictions)))
    scores = []
    for label in classes:
        true_positive = np.sum((labels == label) & (predictions == label))
        false_positive = np.sum((labels != label) & (predictions == label))
        false_negative = np.sum((labels == label) & (predictions != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def quantile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {key: float("nan") for key in ("mean", "median", "p10", "p50", "p90")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, .1)),
        "p50": float(np.quantile(values, .5)),
        "p90": float(np.quantile(values, .9)),
    }


def adjacent_cosine(tokens):
    values = np.asarray(tokens, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    values = values / np.maximum(norms, EPS)
    return np.sum(values * np.roll(values, -1, axis=1), axis=-1).reshape(-1)


def within_class_dispersion(features, labels):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    features /= np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), EPS)
    rows = {}
    for label in np.unique(labels):
        chosen = features[labels == label]
        centroid = chosen.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), EPS)
        rows[int(label)] = float(np.mean(1 - chosen @ centroid))
    return rows


def nearest_source_coverage(source, target, chunk_size=512):
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    source /= np.maximum(np.linalg.norm(source, axis=-1, keepdims=True), EPS)
    target /= np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), EPS)
    chunks = []
    for start in range(0, len(target), chunk_size):
        chunks.append((target[start:start + chunk_size] @ source.T).max(axis=1))
    return np.concatenate(chunks) if chunks else np.empty(0)
