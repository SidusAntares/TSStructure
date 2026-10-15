#!/usr/bin/env python3
"""Frozen Source-only P0 audit for Anchor matching calibration evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.nn import functional as F

from analysis.structure_semantic_features import (
    create_model_from_config,
    semantic_loader,
)
from analysis.audit_e_representation_chain import _dataset as make_dataset
from analysis.summarize_reliable_composition_ab import (
    official_create_splits as create_official_splits,
)
from dataset import PixelSetData


P0R_PROTOCOL_VERSION = "p0r-v1"
CALIBRATION_PROTOCOL = {
    "normal_change_quantile": 0.95,
    "minimum_total": 50,
    "minimum_per_class": 10,
    "bootstrap_repeats": 200,
    "bootstrap_confidence": 0.95,
    "stable_auroc_lower_bound": 0.5,
    "same_class_shuffle_repeats": 20,
    "conditional_permutation_repeats": 200,
    "minimum_baseline_auprc_increment": 0.01,
    "conditional_permutation_alpha": 0.05,
    "minimum_morphology_positive_fraction": 0.05,
}
PERTURBATION_VERSION = "reverse_and_four_contiguous_blocks_v1"
VIEW_SEED_VERSION = "sha256_seed_domain_parcel_view_v1"
REQUIRED_OUTPUTS = (
    "matching_distribution.csv",
    "view_repeatability.csv",
    "morphology_counterfactual.csv",
    "calibration_candidates.csv",
    "anchor_reliability_summary.csv",
    "repeatability_controls.csv",
    "repeatability_baselines.csv",
    "morphology_conditional_controls.csv",
    "per_class_evidence.csv",
    "bootstrap_protocol.json",
    "calibration_decision.json",
    "checkpoint_manifest.json",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=path.stem + ".", suffix=".json", delete=False,
    )
    try:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno()); handle.close()
        os.replace(handle.name, path)
    finally:
        if not handle.closed:
            handle.close()
        Path(handle.name).unlink(missing_ok=True)


def _write_csv(path, rows, fieldnames=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def deterministic_view_seed(seed, domain, parcel_index, view_id):
    payload = f"{int(seed)}|{domain}|{int(parcel_index)}|{int(view_id)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def seeded_dataset_item(dataset, index, seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1)); torch.manual_seed(int(seed))
        return dataset[index]
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


class DeterministicViewDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, domain, global_seed, view_id):
        self.dataset = dataset
        self.domain = str(domain)
        self.global_seed = int(global_seed)
        self.view_id = int(view_id)
        self.parcels = np.asarray(dataset.get_parcel_indices(), dtype=np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        seed = deterministic_view_seed(
            self.global_seed, self.domain, self.parcels[index], self.view_id,
        )
        return seeded_dataset_item(self.dataset, index, seed)

    def get_shapes(self):
        return self.dataset.get_shapes()

    def get_parcel_indices(self):
        return self.parcels.copy()

    def get_labels(self):
        return self.dataset.get_labels()


def split_ids_are_disjoint(train_ids, val_ids):
    return set(map(int, train_ids)).isdisjoint(set(map(int, val_ids)))


def pair_views_by_parcel(first, second):
    def mapping(packet):
        keys = list(zip(map(str, packet["domain"]), map(int, packet["parcel_index"])))
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate (domain, parcel_index) in a view")
        return {key: index for index, key in enumerate(keys)}
    left, right = mapping(first), mapping(second)
    if set(left) != set(right):
        raise ValueError("independent views do not contain identical parcels")
    keys = sorted(left)
    def reordered(packet, index, sample_count):
        result = {}
        identity = list(index) == list(range(sample_count))
        for key, value in packet.items():
            array = np.asarray(value)
            is_metadata = key in {"window_starts", "window_scales"}
            if array.ndim and len(array) == sample_count and not is_metadata:
                result[key] = array if identity else array[index]
            else:
                result[key] = array
        return result
    return (
        reordered(first, [left[item] for item in keys], len(left)),
        reordered(second, [right[item] for item in keys], len(right)),
    )


def _circular_indices(start, scale, grid_points):
    return {(int(start) + offset) % int(grid_points) for offset in range(int(scale))}


def circular_window_overlap(start_a, start_b, scale=24, grid_points=64, scale_b=None):
    scale_b = scale if scale_b is None else scale_b
    return bool(
        _circular_indices(start_a, scale, grid_points)
        & _circular_indices(start_b, scale_b, grid_points)
    )


def circular_window_iou(start_a, start_b, scale=24, grid_points=64, scale_b=None):
    """IoU of the exact circular extractor indices used by two windows."""
    scale_b = scale if scale_b is None else scale_b
    first = _circular_indices(start_a, scale, grid_points)
    second = _circular_indices(start_b, scale_b, grid_points)
    return len(first & second) / max(len(first | second), 1)


def class_matched_derangements(parcels, labels, repeats=None, seed=1):
    """Return deterministic same-class, never-self parcel controls."""
    parcels = np.asarray(parcels)
    labels = np.asarray(labels)
    if parcels.ndim != 1 or labels.shape != parcels.shape:
        raise ValueError("parcels and labels must be aligned one-dimensional arrays")
    if len(np.unique(parcels)) != len(parcels):
        raise ValueError("parcel IDs must be unique before derangement")
    repeats = (
        CALIBRATION_PROTOCOL["same_class_shuffle_repeats"]
        if repeats is None else int(repeats)
    )
    rng = np.random.default_rng(seed)
    output = np.empty((repeats, len(parcels)), dtype=np.int64)
    for clazz in np.unique(labels):
        indices = np.flatnonzero(labels == clazz)
        if len(indices) < 2:
            raise ValueError(f"class {clazz} has fewer than two parcels for a shuffled control")
        for repeat in range(repeats):
            order = rng.permutation(indices)
            shift = int(rng.integers(1, len(indices)))
            partners = np.roll(order, shift)
            output[repeat, order] = partners
    if np.any(output == np.arange(len(parcels))[None, :]):
        raise RuntimeError("same-class control unexpectedly contains a self match")
    return output


def repeatability_rows(
    similarity_a, similarity_b, parcels, starts, scales, grid_points=64,
    class_labels=None, control="paired",
):
    left = np.asarray(similarity_a); right = np.asarray(similarity_b)
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("paired similarity must share [B,N,M]")
    starts = np.asarray(starts); scales = np.asarray(scales)
    if starts.shape != scales.shape or starts.shape != (left.shape[1],):
        raise ValueError("window starts/scales must match candidate count")
    class_labels = (
        np.full(len(parcels), -1, dtype=np.int64)
        if class_labels is None else np.asarray(class_labels, dtype=np.int64)
    )
    rows = []
    for direction, source, target in (
        ("view0_to_view1", left, right), ("view1_to_view0", right, left),
    ):
        source_best = source.argmax(1); target_best = target.argmax(1)
        source_q = source.max(1)
        for sample, parcel in enumerate(np.asarray(parcels)):
            for anchor in range(left.shape[2]):
                a = int(source_best[sample, anchor]); b = int(target_best[sample, anchor])
                rows.append({
                    "parcel_index": int(parcel), "anchor": anchor,
                    "class_id": int(class_labels[sample]), "control": control,
                    "direction": direction, "level": "parcel_anchor",
                    "q": float(source_q[sample, anchor]),
                    "source_window": a, "target_window": b,
                    "exact_match": int(a == b),
                    "profile_cosine": float(np.dot(
                        source[sample, :, anchor], target[sample, :, anchor],
                    ) / max(
                        np.linalg.norm(source[sample, :, anchor])
                        * np.linalg.norm(target[sample, :, anchor]), 1e-12,
                    )),
                    "profile_mean_absolute_change": float(np.mean(np.abs(
                        source[sample, :, anchor] - target[sample, :, anchor],
                    ))),
                    "label": int(circular_window_overlap(
                        starts[a], starts[b], scales[a], grid_points, scales[b],
                    )),
                    "window_iou": float(circular_window_iou(
                        starts[a], starts[b], scales[a], grid_points, scales[b],
                    )),
                })
    return rows


def deterministic_block_reorder(windows, seed, blocks=4):
    values = torch.as_tensor(windows)
    if values.ndim != 4 or values.shape[2] < blocks:
        raise ValueError("windows must be [B,N,T,C] with enough time points")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(blocks, generator=generator).tolist()
    if order == list(range(blocks)):
        order = order[1:] + order[:1]
    pieces = torch.tensor_split(values, blocks, dim=2)
    return torch.cat([pieces[index] for index in order], dim=2)


def perturb_windows(windows, seed):
    values = torch.as_tensor(windows)
    return values.flip(2), deterministic_block_reorder(values, seed)


def freeze_morphology_thresholds(normal_absolute_change, quantile=None):
    values = np.asarray(normal_absolute_change, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("normal view changes must retain Anchor axis")
    quantile = CALIBRATION_PROTOCOL["normal_change_quantile"] if quantile is None else quantile
    return np.quantile(values.reshape(-1, values.shape[-1]), float(quantile), axis=0)


def morphology_proxy_labels(original, perturbed, thresholds):
    original = np.asarray(original, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if original.shape != perturbed.shape or thresholds.shape != (original.shape[-1],):
        raise ValueError("morphology inputs must share [...,Anchor] and Anchor thresholds")
    delta = original - perturbed
    return delta > thresholds, delta


def cross_view_morphology_records(
    similarity_view0, similarity_view1, perturbed_view0, perturbed_view1,
    thresholds, parcels, labels, starts, perturbation,
):
    """Build decoupled morphology records: q from one view, label from the other."""
    view0 = np.asarray(similarity_view0, dtype=np.float64)
    view1 = np.asarray(similarity_view1, dtype=np.float64)
    changed0 = np.asarray(perturbed_view0, dtype=np.float64)
    changed1 = np.asarray(perturbed_view1, dtype=np.float64)
    if not (view0.shape == view1.shape == changed0.shape == changed1.shape):
        raise ValueError("cross-view morphology arrays must share [B,N,M]")
    proxy0, delta0 = morphology_proxy_labels(view0, changed0, thresholds)
    proxy1, delta1 = morphology_proxy_labels(view1, changed1, thresholds)
    parcels = np.asarray(parcels); labels = np.asarray(labels); starts = np.asarray(starts)
    rows = []
    for direction, predictor, proxy, delta in (
        ("view0_to_view1", view0, proxy1, delta1),
        ("view1_to_view0", view1, proxy0, delta0),
    ):
        for sample in range(view0.shape[0]):
            for window in range(view0.shape[1]):
                for anchor in range(view0.shape[2]):
                    rows.append({
                        "parcel_index": int(parcels[sample]),
                        "class_id": int(labels[sample]),
                        "window_index": window,
                        "window_start": int(starts[window]),
                        "anchor": anchor,
                        "direction": direction,
                        "perturbation": str(perturbation),
                        "q": float(predictor[sample, window, anchor]),
                        "opposite_view_delta": float(delta[sample, window, anchor]),
                        "proxy_label": int(proxy[sample, window, anchor]),
                    })
    return rows


def conditional_permute(values, strata, seed=1):
    """Permute values only within predeclared class/time/q-band strata."""
    values = np.asarray(values)
    strata = np.asarray(strata, dtype=object)
    if len(values) != len(strata):
        raise ValueError("values and strata must align")
    keys = [tuple(item) if isinstance(item, (list, tuple, np.ndarray)) else (item,) for item in strata]
    rng = np.random.default_rng(seed)
    output = values.copy()
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        indices = np.asarray(indices, dtype=np.int64)
        output[indices] = values[rng.permutation(indices)]
    return output


@dataclass(frozen=True)
class MonotonicCalibration:
    slope: float
    intercept: float
    fit_split: str = "source_train"
    sample_count: int = 0
    positive_count: int = 0
    negative_count: int = 0

    def predict_proba(self, q):
        values = self.slope * np.asarray(q, dtype=np.float64) + self.intercept
        return 1. / (1. + np.exp(-np.clip(values, -40., 40.)))


def fit_monotonic_calibrator(q, labels, groups, minimum_total=None, minimum_per_class=None):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if not (len(q) == len(labels) == len(groups)):
        raise ValueError("q, labels, and parcel groups must align")
    minimum_total = CALIBRATION_PROTOCOL["minimum_total"] if minimum_total is None else minimum_total
    minimum_per_class = CALIBRATION_PROTOCOL["minimum_per_class"] if minimum_per_class is None else minimum_per_class
    counts = np.bincount(labels, minlength=2)
    if np.count_nonzero(counts) < 2:
        raise ValueError("both proxy classes are required for calibration")
    if len(labels) < minimum_total or counts.min() < minimum_per_class:
        raise ValueError("minimum support is not met for calibration")

    def objective(parameters):
        slope, intercept = parameters
        logits = np.clip(slope * q + intercept, -40., 40.)
        return float(np.mean(np.logaddexp(0., logits) - labels * logits))

    prevalence = np.clip(labels.mean(), 1e-6, 1. - 1e-6)
    result = minimize(
        objective, np.array([1., math.log(prevalence / (1. - prevalence))]),
        method="L-BFGS-B", bounds=((0., None), (None, None)),
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"monotonic calibration failed: {result.message}")
    return MonotonicCalibration(
        slope=float(result.x[0]), intercept=float(result.x[1]),
        sample_count=len(labels), positive_count=int(counts[1]), negative_count=int(counts[0]),
    )


def _ece(labels, scores, bins=10, sample_weight=None):
    labels = np.asarray(labels); scores = np.asarray(scores)
    weights = (
        np.ones(len(labels), dtype=np.float64)
        if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )
    total = weights.sum()
    edges = np.linspace(0., 1., bins + 1)
    result = 0.
    for index in range(bins):
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1]
        )
        if selected.any():
            selected_weight = weights[selected]
            mass = selected_weight.sum()
            result += mass / max(total, 1e-12) * abs(
                np.average(labels[selected], weights=selected_weight)
                - np.average(scores[selected], weights=selected_weight)
            )
    return float(result)


def _metric(labels, scores, name, sample_weight=None):
    if name == "auroc":
        return float(roc_auc_score(labels, scores, sample_weight=sample_weight))
    if name == "auprc":
        return float(average_precision_score(labels, scores, sample_weight=sample_weight))
    if name == "brier":
        return float(brier_score_loss(labels, scores, sample_weight=sample_weight))
    if name == "ece":
        return _ece(labels, scores, sample_weight=sample_weight)
    raise ValueError(f"unknown calibration metric: {name}")


def bootstrap_metric_interval(
    labels, scores, groups, metric, repeats=None, confidence=None, seed=1,
    group_classes=None, return_metadata=False,
):
    labels = np.asarray(labels); scores = np.asarray(scores); groups = np.asarray(groups)
    repeats = CALIBRATION_PROTOCOL["bootstrap_repeats"] if repeats is None else int(repeats)
    confidence = CALIBRATION_PROTOCOL["bootstrap_confidence"] if confidence is None else float(confidence)
    estimate = _metric(labels, scores, metric)
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    strata = [np.arange(len(unique), dtype=np.int64)]
    stratified = group_classes is not None
    if group_classes is not None:
        group_classes = np.asarray(group_classes)
        if len(group_classes) != len(groups):
            raise ValueError("group_classes must align with observations")
        per_group = []
        for position in range(len(unique)):
            current = np.unique(group_classes[inverse == position])
            if len(current) != 1:
                raise ValueError("each parcel group must have one Source class")
            per_group.append(current[0])
        per_group = np.asarray(per_group)
        strata = [np.flatnonzero(per_group == clazz) for clazz in np.unique(per_group)]
    values = []
    for _ in range(repeats):
        sampled = np.concatenate([
            rng.choice(positions, size=len(positions), replace=True) for positions in strata
        ])
        multiplicity = np.bincount(sampled, minlength=len(unique))
        weights = multiplicity[inverse]
        selected = weights > 0
        if np.unique(labels[selected]).size < 2 and metric in ("auroc", "auprc"):
            continue
        values.append(_metric(
            labels[selected], scores[selected], metric,
            sample_weight=weights[selected],
        ))
    if not values:
        low = high = float("nan")
    else:
        alpha = (1. - confidence) / 2.
        low = float(np.quantile(values, alpha)); high = float(np.quantile(values, 1. - alpha))
    result = (estimate, low, high)
    if return_metadata:
        result += ({
            "method": "parcel_cluster_bootstrap",
            "seed": int(seed), "repeats": repeats,
            "confidence": confidence,
            "population_group_count": len(unique),
            "resampled_group_count": len(unique),
            "stratified": stratified,
            "invalid_replicates": repeats - len(values),
        },)
    return result


def assess_joint_reliability(repeatability, morphology):
    if repeatability == "stable" and morphology == "stable":
        return "calibratable"
    if repeatability == "failed" and morphology == "failed":
        return "not_calibratable"
    return "uncertain"


def assess_anchor_reliability(repeatability, morphology):
    """Apply the preregistered two-evidence decision and emit auditable reasons."""
    reasons = []
    if not repeatability.get("supported", False):
        reasons.append("repeatability_unsupported")
    else:
        if not repeatability.get("monotonic_generalizes", False):
            reasons.append("repeatability_not_monotonic_on_val")
        if not repeatability.get("baseline_increment_supported", False):
            reasons.append("repeatability_no_baseline_increment")
        if not repeatability.get("paired_shuffle_effect_supported", False):
            reasons.append("repeatability_no_paired_shuffle_effect")
    if not morphology.get("supported", False):
        reasons.append("morphology_unsupported")
    else:
        if not morphology.get("cross_view_generalizes", False):
            reasons.append("morphology_cross_view_not_generalized")
        if not morphology.get("exceeds_normal_variation", False):
            reasons.append("morphology_not_above_normal_variation")
        if not morphology.get("conditional_control_supported", False):
            reasons.append("morphology_proxy_artifact_not_excluded")
        perturbations = morphology.get("perturbations", {})
        if perturbations and not any(value == "pass" for value in perturbations.values()):
            reasons.append("morphology_no_effective_perturbation")
    repeat_pass = repeatability.get("supported", False) and not any(
        reason.startswith("repeatability_") for reason in reasons
    )
    morphology_pass = morphology.get("supported", False) and not any(
        reason.startswith("morphology_") for reason in reasons
    )
    if repeat_pass and morphology_pass:
        status = "calibratable"
    elif (
        repeatability.get("supported", False) and morphology.get("supported", False)
        and not repeat_pass and not morphology_pass
    ):
        status = "not_calibratable"
    else:
        status = "uncertain"
    return {"status": status, "reason_codes": reasons}


def build_calibration_decision(
    source, checkpoint_sha, train_fingerprint, val_fingerprint, anchors, seed,
):
    return {
        "protocol_version": P0R_PROTOCOL_VERSION,
        "source": source, "checkpoint_sha256": checkpoint_sha,
        "split_fingerprints": {"source_train": train_fingerprint, "source_val": val_fingerprint},
        "seed": int(seed), "anchors": list(anchors),
        "target_data_used": False, "unified_matching_probability": None,
    }


def build_ab_readiness(decisions, expected_sources=None):
    per_source = {}
    for source, decision in decisions.items():
        statuses = [anchor.get("status") for anchor in decision.get("anchors", [])]
        per_source[source] = {
            "anchor_count": len(statuses),
            "calibratable_count": statuses.count("calibratable"),
            "uncertain_count": statuses.count("uncertain"),
            "not_calibratable_count": statuses.count("not_calibratable"),
        }
    missing_sources = sorted(set(expected_sources or ()) - set(decisions))
    if missing_sources or not per_source or any(row["calibratable_count"] == 0 for row in per_source.values()):
        status = "blocked"
    elif all(row["calibratable_count"] == row["anchor_count"] for row in per_source.values()):
        status = "ready"
    else:
        status = "ready_with_limitations"
    return {
        "protocol_version": P0R_PROTOCOL_VERSION, "status": status,
        "experiment_feasible": status != "blocked",
        "credible_occurrence_interpretation": status == "ready",
        "sources": per_source, "missing_sources": missing_sources,
    }


def freeze_for_audit(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def recompute_window_tokens_and_similarity(branch, windows):
    encoded = branch.token_generator(windows, return_encoded_components=True)
    tokens = encoded["shape_token"] if isinstance(encoded, dict) else encoded
    similarity = branch.shapelet_dictionary.compute_similarity(tokens)
    return tokens, similarity


def _calibration_metrics(calibrator, q, labels, groups, seed, group_classes=None):
    scores = calibrator.predict_proba(q)
    intervals, metadata = _joint_calibration_bootstrap(
        labels, scores, groups, seed=seed, group_classes=group_classes,
    )
    row = {}
    for metric, (estimate, low, high) in intervals.items():
        row[metric] = estimate; row[f"{metric}_ci_low"] = low; row[f"{metric}_ci_high"] = high
        row[f"{metric}_bootstrap_invalid_replicates"] = metadata["invalid_replicates"]
        row[f"{metric}_bootstrap_group_count"] = metadata["population_group_count"]
    row["sample_count"] = len(labels)
    row["positive_count"] = int(np.asarray(labels).sum())
    row["negative_count"] = int(len(labels) - np.asarray(labels).sum())
    return row


def _joint_calibration_bootstrap(labels, scores, groups, seed, group_classes=None):
    """Shared full-population parcel bootstrap for all calibration metrics.

    Bootstrap parcel multiplicities are generated once and reused across all
    four metrics. Ranking metrics reuse one fixed score order, avoiding tens of
    thousands of redundant sorts on large Source validation splits.
    """
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    unique, inverse = np.unique(groups, return_inverse=True)
    if group_classes is None:
        strata = [np.arange(len(unique), dtype=np.int64)]
        stratified = False
    else:
        group_classes = np.asarray(group_classes)
        per_group = np.empty(len(unique), dtype=group_classes.dtype)
        for position in range(len(unique)):
            values = np.unique(group_classes[inverse == position])
            if len(values) != 1:
                raise ValueError("each parcel group must have one Source class")
            per_group[position] = values[0]
        strata = [np.flatnonzero(per_group == clazz) for clazz in np.unique(per_group)]
        stratified = True
    repeats = CALIBRATION_PROTOCOL["bootstrap_repeats"]
    rng = np.random.default_rng(seed)
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = labels[order].astype(np.float64)
    sorted_scores = scores[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    squared_error = np.square(scores - labels)
    bins = np.minimum((scores * 10).astype(np.int64), 9)
    values = {metric: [] for metric in ("auroc", "auprc", "brier", "ece")}
    invalid = 0
    for _ in range(repeats):
        sampled = np.concatenate([
            rng.choice(positions, size=len(positions), replace=True) for positions in strata
        ])
        multiplicity = np.bincount(sampled, minlength=len(unique)).astype(np.float64)
        weights = multiplicity[inverse]
        positive = float(np.dot(weights, labels)); negative = float(weights.sum() - positive)
        if positive <= 0 or negative <= 0:
            invalid += 1; continue
        sorted_w = weights[order]
        pos_w = sorted_w * sorted_y; neg_w = sorted_w * (1. - sorted_y)
        pos_group = np.add.reduceat(pos_w, group_starts)
        neg_group = np.add.reduceat(neg_w, group_starts)
        negatives_after = np.cumsum(neg_group[::-1])[::-1] - neg_group
        auc = float(np.dot(pos_group, negatives_after + .5 * neg_group) / (positive * negative))
        cumulative_positive = np.cumsum(pos_group)
        cumulative_total = np.cumsum(pos_group + neg_group)
        precision = np.divide(
            cumulative_positive, cumulative_total,
            out=np.zeros_like(cumulative_positive), where=cumulative_total > 0,
        )
        auprc = float(np.dot(precision, pos_group) / positive)
        total_weight = max(float(weights.sum()), 1e-12)
        brier = float(np.dot(weights, squared_error) / total_weight)
        ece = 0.
        for band in range(10):
            selected = bins == band
            mass = float(weights[selected].sum())
            if mass:
                observed = float(np.dot(weights[selected], labels[selected]) / mass)
                predicted = float(np.dot(weights[selected], scores[selected]) / mass)
                ece += mass / total_weight * abs(observed - predicted)
        for metric, value in (("auroc", auc), ("auprc", auprc), ("brier", brier), ("ece", ece)):
            values[metric].append(value)
    alpha = (1. - CALIBRATION_PROTOCOL["bootstrap_confidence"]) / 2.
    intervals = {}
    for metric, replicas in values.items():
        estimate = _metric(labels, scores, metric)
        if replicas:
            low, high = np.quantile(replicas, [alpha, 1. - alpha])
        else:
            low = high = float("nan")
        intervals[metric] = (estimate, float(low), float(high))
    return intervals, {
        "method": "shared_full_population_parcel_cluster_bootstrap",
        "seed": int(seed), "repeats": repeats,
        "population_group_count": len(unique), "resampled_group_count": len(unique),
        "stratified": stratified, "invalid_replicates": invalid,
    }


def _evidence_status(rows):
    supported = [row for row in rows if row.get("fit_status") == "fit"]
    if len(supported) != len(rows) or not rows:
        return "unsupported"
    stable = all(
        float(row["slope"]) > 0.
        and float(row["auroc_ci_low"]) > CALIBRATION_PROTOCOL["stable_auroc_lower_bound"]
        for row in supported
    )
    return "stable" if stable else "failed"


def _config_dict(raw):
    return dict(raw if isinstance(raw, dict) else vars(raw))


def _random_view_transform(config):
    from torchvision.transforms import transforms
    from transforms import Identity, Normalize, RandomSamplePixels, RandomSampleTimeSteps, RandomTemporalShift, ToTensor
    shift = (
        RandomTemporalShift(config.max_shift_aug, config.shift_aug_p)
        if bool(getattr(config, "with_shift_aug", False)) else Identity()
    )
    return transforms.Compose([
        RandomSamplePixels(int(config.num_pixels)),
        RandomSampleTimeSteps(int(config.seq_length)), shift, Normalize(), ToTensor(),
    ])


def load_source_runtime(checkpoint, data_root, device, seed=1):
    checkpoint = Path(checkpoint); data_root = Path(data_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing Source checkpoint: {checkpoint}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"missing data root: {data_root}")
    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "config" not in packet or "state_dict" not in packet:
        raise ValueError("Source checkpoint must contain config and state_dict")
    config = SimpleNamespace(**_config_dict(packet["config"])); config.data_root = str(data_root)
    if getattr(config, "shape_representation", None) != "phase_moment":
        raise ValueError("P0 requires original G phase_moment Source checkpoint")
    if int(getattr(config, "shapelet_count", -1)) != 16:
        raise ValueError("P0 requires 16 learned Anchors")
    model = create_model_from_config(config)
    model.load_state_dict(packet["state_dict"], strict=True)
    model = freeze_for_audit(model.to(device))
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if not getattr(config, "classes", None):
        raise ValueError("Source checkpoint must contain its frozen closed-set class mapping")
    source_protocol_dataset = PixelSetData(
        str(data_root), config.source, list(config.classes), closed_set=True,
        combine_spring_and_winter=bool(config.combine_spring_and_winter),
    )
    eligible = {config.source: source_protocol_dataset.get_parcel_indices().tolist()}
    protocol = {
        "closed_set": True, "source": config.source,
        "classes": list(config.classes),
        "class_to_idx": {name: index for index, name in enumerate(config.classes)},
        "eligible_source_samples": len(eligible[config.source]),
        "seed": int(seed), "val_ratio": float(config.val_ratio),
        "test_ratio": float(config.test_ratio), "target_data_used": False,
    }
    splits = create_official_splits(
        [config.source], config.num_folds, eligible,
        config.val_ratio, config.test_ratio,
    )[0]
    datasets = {
        name: make_dataset(data_root, config.source, config, splits[config.source][split])
        for name, split in (("source_train", "train"), ("source_val", "val"))
    }
    if not split_ids_are_disjoint(
        datasets["source_train"].get_parcel_indices(), datasets["source_val"].get_parcel_indices(),
    ):
        raise RuntimeError("Source Train/Val parcel leakage")
    transform = _random_view_transform(config)
    datasets["source_train"].transform = transform
    datasets["source_val"].transform = transform
    return model, config, protocol, splits, datasets


def _window_metadata(branch):
    starts, scales = [], []
    for index, scale in enumerate(branch.window_extractor.scales):
        indices = getattr(branch.window_extractor, f"indices_{index}")
        starts.extend(indices[:, 0].detach().cpu().tolist())
        scales.extend([int(scale)] * indices.shape[0])
    return np.asarray(starts, dtype=np.int64), np.asarray(scales, dtype=np.int64)


@torch.inference_mode()
def extract_view(model, dataset, domain, split, view_id, seed, device, batch_size, pixel_budget):
    wrapped = DeterministicViewDataset(dataset, domain, seed, view_id)
    target = torch.device(device)
    storage = {key: [] for key in (
        "parcel_index", "label", "similarity", "token_norm", "reverse_similarity",
        "reorder_similarity", "pooling_weight",
        "reverse_token_cosine", "reorder_token_cosine",
    )}
    starts, scales = _window_metadata(model.structure_branch)
    if len(set(scales.tolist())) != 1:
        raise ValueError("P0 morphology perturbation currently requires one fixed window scale")
    for raw in semantic_loader(wrapped, batch_size, pixel_budget, num_workers=0):
        batch = {key: value.to(target) if torch.is_tensor(value) else value for key, value in raw.items()}
        output = model.forward_with_temporal_shift(
            batch["pixels"], batch["valid_pixels"], batch["positions"], batch["extra"],
            temporal_shift=0, return_dict=True,
        )
        groups, _ = model.structure_branch.window_extractor(output["exposed_curve"])
        windows = torch.cat(groups, dim=1)
        recomputed, similarity = recompute_window_tokens_and_similarity(
            model.structure_branch, windows,
        )
        torch.testing.assert_close(recomputed, output["shape_tokens"], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(similarity, output["shapelet_similarity"], rtol=2e-5, atol=2e-6)
        reversed_windows, reordered_windows = perturb_windows(
            windows, deterministic_view_seed(seed, domain, int(view_id), 99),
        )
        reversed_tokens = model.structure_branch.token_generator(
            reversed_windows, return_encoded_components=True,
        )["shape_token"]
        reordered_tokens = model.structure_branch.token_generator(
            reordered_windows, return_encoded_components=True,
        )["shape_token"]
        beta = float(model.structure_branch.shapelet_dictionary.beta)
        values = {
            "parcel_index": batch["parcel_index"], "label": batch["label"],
            "similarity": similarity, "token_norm": recomputed.norm(dim=-1),
            "reverse_similarity": model.structure_branch.shapelet_dictionary.compute_similarity(reversed_tokens),
            "reorder_similarity": model.structure_branch.shapelet_dictionary.compute_similarity(reordered_tokens),
            "pooling_weight": torch.softmax(beta * similarity, dim=1),
            "reverse_token_cosine": F.cosine_similarity(recomputed, reversed_tokens, dim=-1),
            "reorder_token_cosine": F.cosine_similarity(recomputed, reordered_tokens, dim=-1),
        }
        for key, value in values.items():
            storage[key].append(value.detach().cpu())
    arrays = {key: torch.cat(value).numpy() for key, value in storage.items()}
    arrays["domain"] = np.asarray([str(domain)] * len(arrays["parcel_index"]))
    arrays["split"] = np.asarray([str(split)] * len(arrays["parcel_index"]))
    arrays["view_id"] = np.full(len(arrays["parcel_index"]), int(view_id), dtype=np.int64)
    arrays["window_starts"] = starts; arrays["window_scales"] = scales
    return arrays


def _split_fingerprint(dataset, domain):
    parcels = np.asarray(dataset.get_parcel_indices(), dtype=np.int64)
    payload = str(domain).encode() + b"\0" + parcels.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _raw_cache_signature(checkpoint_sha, split_fingerprints, seed, branch):
    starts, scales = _window_metadata(branch)
    payload = {
        "protocol_version": P0R_PROTOCOL_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "split_fingerprints": split_fingerprints,
        "seed": int(seed), "view_seed_version": VIEW_SEED_VERSION,
        "perturbation_version": PERTURBATION_VERSION,
        "window_starts": starts.tolist(), "window_scales": scales.tolist(),
        "token_generator": type(branch.token_generator).__name__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def _save_raw_view_cache(path, packet):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **packet)
    os.replace(temporary, path)


def _load_raw_view_cache(path):
    with np.load(path, allow_pickle=False) as packet:
        return {key: packet[key] for key in packet.files}


def load_or_extract_views(
    output, model, datasets, domain, seed, device, batch_size, pixel_budget,
    checkpoint_sha,
):
    split_fingerprints = {
        split: _split_fingerprint(dataset, domain) for split, dataset in datasets.items()
    }
    signature, cache_protocol = _raw_cache_signature(
        checkpoint_sha, split_fingerprints, seed, model.structure_branch,
    )
    cache = Path(output) / "cache"
    manifest_path = cache / "raw_cache_manifest.json"
    reusable = False
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            reusable = manifest.get("signature") == signature and all(
                (cache / f"{split}_view{view}.npz").is_file()
                for split in datasets for view in (0, 1)
            )
        except (OSError, ValueError, json.JSONDecodeError):
            reusable = False
    views = {}
    for split, dataset in datasets.items():
        packets = []
        for view_id in (0, 1):
            path = cache / f"{split}_view{view_id}.npz"
            if reusable:
                packet = _load_raw_view_cache(path)
            else:
                packet = extract_view(
                    model, dataset, domain, split, view_id, seed, device,
                    batch_size, pixel_budget,
                )
                _save_raw_view_cache(path, packet)
            packets.append(packet)
        views[split] = tuple(packets)
    if not reusable:
        _atomic_json(manifest_path, {
            "protocol_version": P0R_PROTOCOL_VERSION,
            "signature": signature, "protocol": cache_protocol,
            "contents": "compact similarities, window metadata, parcel/class IDs; no high-dimensional tokens",
        })
    return views, split_fingerprints, signature


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "q05": float(np.quantile(values, .05)), "q25": float(np.quantile(values, .25)),
        "q50": float(np.quantile(values, .5)), "q75": float(np.quantile(values, .75)),
        "q95": float(np.quantile(values, .95)),
    }


def matching_distribution_rows(packet, split, view_id, beta):
    similarity = packet["similarity"]; weights = packet["pooling_weight"]
    top_anchor = similarity.argmax(2)
    rows = []
    for anchor in range(similarity.shape[-1]):
        qmax = similarity[:, :, anchor].max(1)
        selected = similarity[:, :, anchor].argmax(1)
        entropy = -(weights[:, :, anchor] * np.log(np.maximum(weights[:, :, anchor], 1e-12))).sum(1)
        counts = np.bincount(selected, minlength=similarity.shape[1])
        row = {"split": split, "view_id": view_id, "anchor": anchor, "sample_count": len(qmax)}
        row.update({f"qmax_{key}": value for key, value in _quantiles(qmax).items()})
        row.update({f"window_entropy_{key}": value for key, value in _quantiles(entropy).items()})
        row["modal_window"] = int(counts.argmax())
        row["modal_window_fraction"] = float(counts.max() / counts.sum())
        row["selected_window_coverage"] = float(np.count_nonzero(counts) / len(counts))
        row["anchor_utilization_fraction"] = float(np.mean(top_anchor == anchor))
        row["shape_token_norm_mean"] = float(np.mean(packet["token_norm"]))
        row["shape_token_norm_std"] = float(np.std(packet["token_norm"]))
        rows.append(row)
    return rows


def _window_repeatability_rows(first, second, split):
    rows = []
    change = np.abs(first["similarity"] - second["similarity"])
    for anchor in range(change.shape[-1]):
        values = change[:, :, anchor].reshape(-1)
        row = {
            "split": split, "level": "window_anchor", "direction": "symmetric",
            "anchor": anchor, "sample_count": len(values),
        }
        row.update({f"absolute_change_{key}": value for key, value in _quantiles(values).items()})
        rows.append(row)
    return rows


def _repeatability_band_rows(rows, split, train_edges):
    output = []
    for anchor in sorted({row["anchor"] for row in rows}):
        for direction in ("view0_to_view1", "view1_to_view0"):
            selected = [row for row in rows if row["anchor"] == anchor and row["direction"] == direction]
            q = np.asarray([row["q"] for row in selected])
            label = np.asarray([row["label"] for row in selected])
            bands = np.digitize(q, train_edges[(anchor, direction)][1:-1], right=True)
            for band, name in enumerate(("low", "medium", "high")):
                current = bands == band
                if not current.any():
                    continue
                output.append({
                    "split": split, "level": "parcel_anchor_q_band",
                    "direction": direction, "anchor": anchor,
                    "similarity_band": name, "sample_count": int(current.sum()),
                    "overlap_repeatability": float(label[current].mean()),
                    "unstable_fraction": float(1. - label[current].mean()),
                    "qmax_mean": float(q[current].mean()),
                })
    return output


def _repeatability_aggregate(rows, starts, scales, train_mode=None, grid_points=64):
    output = []
    anchors = sorted({row["anchor"] for row in rows})
    for anchor in anchors:
        for direction in ("view0_to_view1", "view1_to_view0"):
            selected = [row for row in rows if row["anchor"] == anchor and row["direction"] == direction]
            q = np.asarray([row["q"] for row in selected]); label = np.asarray([row["label"] for row in selected])
            exact = np.asarray([row["exact_match"] for row in selected])
            row = {
                "level": "parcel_anchor",
                "anchor": anchor, "direction": direction, "sample_count": len(selected),
                "overlap_repeatability": float(label.mean()), "exact_repeatability": float(exact.mean()),
                "profile_cosine_mean": float(np.mean([item["profile_cosine"] for item in selected])),
                "profile_cosine_median": float(np.median([item["profile_cosine"] for item in selected])),
                "profile_mean_absolute_change": float(np.mean([
                    item["profile_mean_absolute_change"] for item in selected
                ])),
            }
            row.update({f"qmax_{key}": value for key, value in _quantiles(q).items()})
            if train_mode is not None:
                row["fixed_position_baseline_overlap"] = float(np.mean([
                    circular_window_overlap(
                        starts[train_mode[anchor]], starts[item["target_window"]],
                        scales[train_mode[anchor]], grid_points, scales[item["target_window"]],
                    ) for item in selected
                ]))
            output.append(row)
    return output


def _shuffled_repeatability(first, second, starts, scales, seed):
    labels = np.asarray(first["label"])
    counts = {clazz: int(np.sum(labels == clazz)) for clazz in np.unique(labels)}
    eligible = np.asarray([counts[clazz] >= 2 for clazz in labels])
    if not eligible.any():
        return []
    first_similarity = first["similarity"][eligible]
    second_similarity = second["similarity"][eligible]
    parcels = first["parcel_index"][eligible]
    labels = labels[eligible]
    permutations = class_matched_derangements(
        parcels, labels, seed=seed,
    )
    grid_points = 64
    overlap = np.asarray([
        [int(circular_window_overlap(starts[a], starts[b], scales[a], grid_points, scales[b]))
         for b in range(len(starts))]
        for a in range(len(starts))
    ], dtype=np.int8)
    labels_by_repeat = []
    first_best = first_similarity.argmax(1)
    for repeat, indices in enumerate(permutations):
        second_best = second_similarity[indices].argmax(1)
        labels_by_repeat.append(np.stack([
            overlap[first_best, second_best], overlap[second_best, first_best],
        ], axis=0))
    return {
        "labels": np.stack(labels_by_repeat, axis=0),
        "parcel_index": parcels, "class_id": labels,
        "directions": ("view0_to_view1", "view1_to_view0"),
    }


def _paired_shuffle_control_rows(paired, shuffled, split, seed):
    output = []
    if not shuffled:
        return output
    anchors = sorted({int(row["anchor"]) for row in paired})
    for anchor in anchors:
        for direction_index, direction in enumerate(shuffled["directions"]):
            observed = [row for row in paired if row["anchor"] == anchor and row["direction"] == direction]
            parcel_position = {
                int(parcel): index for index, parcel in enumerate(shuffled["parcel_index"])
            }
            observed = [row for row in observed if int(row["parcel_index"]) in parcel_position]
            if not observed:
                continue
            observed_label = np.asarray([row["label"] for row in observed], dtype=np.float64)
            control_label = np.asarray([
                shuffled["labels"][:, direction_index, parcel_position[int(row["parcel_index"])], anchor].mean()
                for row in observed
            ])
            effect = observed_label - control_label
            parcels = np.asarray([row["parcel_index"] for row in observed])
            classes = np.asarray([row["class_id"] for row in observed])
            # Compute the paired effect with a full-population, class-stratified
            # parcel bootstrap; no observation or parcel subsampling is used.
            rng = np.random.default_rng(seed + 1000 + anchor)
            if len(np.unique(parcels)) != len(parcels):
                raise ValueError("paired repeatability must contain one row per parcel and anchor")
            strata = [np.flatnonzero(classes == clazz) for clazz in np.unique(classes)]
            replicated = []
            for _ in range(CALIBRATION_PROTOCOL["bootstrap_repeats"]):
                sampled = np.concatenate([
                    rng.choice(positions, len(positions), replace=True) for positions in strata
                ])
                replicated.append(float(effect[sampled].mean()))
            alpha = (1. - CALIBRATION_PROTOCOL["bootstrap_confidence"]) / 2
            low, high = np.quantile(replicated, [alpha, 1. - alpha])
            output.append({
                "split": split, "anchor": anchor, "direction": direction,
                "sample_count": len(effect),
                "paired_repeatability": float(observed_label.mean()),
                "same_class_shuffled_repeatability": float(control_label.mean()),
                "paired_minus_shuffled": float(effect.mean()),
                "paired_minus_shuffled_ci_low": float(low),
                "paired_minus_shuffled_ci_high": float(high),
                "shuffle_repeats": CALIBRATION_PROTOCOL["same_class_shuffle_repeats"],
                "bootstrap_group_count": len(parcels),
            })
    return output


def _repeatability_design(rows, classes, windows, include_position, include_class, include_q):
    matrix = []
    for row in rows:
        values = [1.0]
        if include_position:
            values.extend(float(row["source_window"] == window) for window in windows[1:])
        if include_class:
            values.extend(float(row["class_id"] == clazz) for clazz in classes[1:])
        if include_q:
            values.append(float(row["q"]))
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64)


def _fit_logistic_design(train_x, train_y, val_x, nonnegative_last=False):
    train_y = np.asarray(train_y, dtype=np.float64)
    if np.unique(train_y).size < 2:
        raise ValueError("both repeatability labels are required")
    def objective(coefficients):
        logits = np.clip(train_x @ coefficients, -40., 40.)
        penalty = 1e-4 * np.square(coefficients[1:]).sum()
        return float(np.mean(np.logaddexp(0., logits) - train_y * logits) + penalty)
    prevalence = np.clip(train_y.mean(), 1e-6, 1 - 1e-6)
    initial = np.zeros(train_x.shape[1]); initial[0] = math.log(prevalence / (1 - prevalence))
    bounds = [(None, None)] * train_x.shape[1]
    if nonnegative_last:
        bounds[-1] = (0., None)
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"repeatability baseline fit failed: {result.message}")
    logits = np.clip(val_x @ result.x, -40., 40.)
    return 1. / (1. + np.exp(-logits)), result.x


def repeatability_baseline_rows(train_rows, val_rows, anchor, seed):
    train = [row for row in train_rows if row["anchor"] == anchor and row["direction"] == "view0_to_view1"]
    val = [row for row in val_rows if row["anchor"] == anchor and row["direction"] == "view0_to_view1"]
    classes = sorted({row["class_id"] for row in train})
    windows = sorted({row["source_window"] for row in train})
    train_y = np.asarray([row["label"] for row in train]); val_y = np.asarray([row["label"] for row in val])
    val_groups = np.asarray([row["parcel_index"] for row in val])
    output = []
    for name, include_position, include_class, include_q in (
        ("constant", False, False, False), ("position_only", True, False, False),
        ("class_position", True, True, False), ("class_position_qmax", True, True, True),
    ):
        train_x = _repeatability_design(
            train, classes, windows, include_position, include_class, include_q,
        )
        val_x = _repeatability_design(
            val, classes, windows, include_position, include_class, include_q,
        )
        if name == "constant":
            scores = np.full(len(val), train_y.mean())
            coefficients = np.asarray([math.log(np.clip(train_y.mean(), 1e-6, 1-1e-6) / np.clip(1-train_y.mean(), 1e-6, 1))])
        else:
            scores, coefficients = _fit_logistic_design(
                train_x, train_y, val_x, nonnegative_last=include_q,
            )
        row = {
            "anchor": anchor, "model": name, "split": "source_val",
            "train_sample_count": len(train), "sample_count": len(val),
            "train_base_rate": float(train_y.mean()), "val_base_rate": float(val_y.mean()),
            "qmax_coefficient": float(coefficients[-1]) if include_q else "",
        }
        for metric in ("auroc", "auprc", "brier", "ece"):
            try:
                estimate, low, high = bootstrap_metric_interval(
                    val_y, scores, val_groups, metric, seed=seed + anchor,
                    group_classes=np.asarray([row_["class_id"] for row_ in val]),
                )
            except ValueError:
                estimate = low = high = float("nan")
            row.update({metric: estimate, f"{metric}_ci_low": low, f"{metric}_ci_high": high})
        output.append(row)
    return output


def _morphology_summary(split, view, thresholds, train_edges):
    rows = []
    for perturbation, changed in (
        ("reverse", view["reverse_similarity"]), ("block_reorder", view["reorder_similarity"]),
    ):
        token_cosine = view[
            "reverse_token_cosine" if perturbation == "reverse" else "reorder_token_cosine"
        ].reshape(-1)
        labels, delta = morphology_proxy_labels(view["similarity"], changed, thresholds)
        for anchor in range(view["similarity"].shape[-1]):
            q = view["similarity"][:, :, anchor].reshape(-1)
            d = delta[:, :, anchor].reshape(-1); y = labels[:, :, anchor].reshape(-1)
            bands = np.digitize(q, train_edges[anchor][1:-1], right=True)
            for band in range(3):
                selected = bands == band
                if not selected.any():
                    continue
                row = {
                    "split": split, "view_id": int(view["view_id"][0]),
                    "perturbation": perturbation, "anchor": anchor,
                    "similarity_band": ("low", "medium", "high")[band],
                    "sample_count": int(selected.sum()),
                    "normal_change_threshold": float(thresholds[anchor]),
                    "proxy_positive_fraction": float(y[selected].mean()),
                    "symmetric_or_small_change_fraction": float(
                        np.mean(np.abs(d[selected]) <= thresholds[anchor])
                    ),
                    "window_token_cosine_mean": float(token_cosine.mean()),
                    "window_token_cosine_q05": float(np.quantile(token_cosine, .05)),
                }
                row.update({f"original_q_{key}": value for key, value in _quantiles(q[selected]).items()})
                row.update({f"delta_q_{key}": value for key, value in _quantiles(d[selected]).items()})
                rows.append(row)
    return rows


def _cross_view_morphology_all(views, thresholds):
    left, right = pair_views_by_parcel(*views)
    packets = []
    for perturbation, key in (
        ("reverse", "reverse_similarity"),
        ("block_reorder", "reorder_similarity"),
    ):
        proxy_left, delta_left = morphology_proxy_labels(
            left["similarity"], left[key], thresholds,
        )
        proxy_right, delta_right = morphology_proxy_labels(
            right["similarity"], right[key], thresholds,
        )
        for direction, q, proxy, delta in (
            ("view0_to_view1", left["similarity"], proxy_right, delta_right),
            ("view1_to_view0", right["similarity"], proxy_left, delta_left),
        ):
            packets.append({
                "perturbation": perturbation, "direction": direction,
                "q": q, "proxy_label": proxy, "delta": delta,
                "parcel_index": left["parcel_index"], "class_id": left["label"],
                "window_starts": left["window_starts"],
                "thresholds": np.asarray(thresholds),
            })
    return packets


def morphology_conditional_control_rows(train_rows, val_rows, seed):
    output = []
    anchor_count = train_rows[0]["q"].shape[-1]
    for anchor in range(anchor_count):
        train_q = np.concatenate([packet["q"][:, :, anchor].reshape(-1) for packet in train_rows])
        edges = np.quantile(train_q, [0., 1/3, 2/3, 1.])
        for perturbation in ("reverse", "block_reorder"):
            for direction in ("view0_to_view1", "view1_to_view0"):
                selected = next((
                    packet for packet in val_rows
                    if packet["perturbation"] == perturbation
                    and packet["direction"] == direction
                ), None)
                if selected is None:
                    continue
                q = selected["q"][:, :, anchor].reshape(-1)
                y = selected["proxy_label"][:, :, anchor].reshape(-1).astype(np.int64)
                if np.unique(y).size < 2:
                    observed = float("nan"); permutation_p = float("nan")
                else:
                    bands = np.digitize(q, edges[1:-1], right=True)
                    classes = np.repeat(selected["class_id"], selected["q"].shape[1])
                    starts = np.tile(selected["window_starts"], selected["q"].shape[0])
                    strata = np.empty((len(q), 3), dtype=object)
                    strata[:, 0] = classes; strata[:, 1] = starts; strata[:, 2] = bands
                    keys = [tuple(row) for row in strata.tolist()]
                    q_residual = q.astype(np.float64).copy()
                    y_residual = y.astype(np.float64).copy()
                    positions = {}
                    for index, key in enumerate(keys):
                        positions.setdefault(key, []).append(index)
                    for indices in positions.values():
                        indices = np.asarray(indices, dtype=np.int64)
                        q_residual[indices] -= q_residual[indices].mean()
                        y_residual[indices] -= y_residual[indices].mean()
                    denominator = max(
                        float(np.linalg.norm(q_residual) * np.linalg.norm(y_residual)),
                        1e-12,
                    )
                    observed = float(np.dot(q_residual, y_residual) / denominator)
                    null = []
                    rng = np.random.default_rng(seed + anchor * 1000)
                    parcel_ids = np.repeat(selected["parcel_index"], selected["q"].shape[1])
                    unique_parcels, parcel_inverse = np.unique(parcel_ids, return_inverse=True)
                    parcel_contribution = np.bincount(
                        parcel_inverse, weights=q_residual * y_residual,
                        minlength=len(unique_parcels),
                    )
                    for _ in range(CALIBRATION_PROTOCOL["conditional_permutation_repeats"]):
                        signs = rng.choice((-1., 1.), size=len(unique_parcels))
                        null.append(float(np.dot(signs, parcel_contribution) / denominator))
                    permutation_p = float(
                        (1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null))
                    )
                output.append({
                    "anchor": anchor, "split": "source_val", "perturbation": perturbation,
                    "direction": direction, "sample_count": len(q),
                    "positive_count": int(y.sum()), "negative_count": int(len(y) - y.sum()),
                    "proxy_positive_fraction": float(y.mean()),
                    "delta_mean": float(selected["delta"][:, :, anchor].mean()),
                    "normal_change_threshold": float(selected["thresholds"][anchor]),
                    "observed_conditional_correlation": observed,
                    "conditional_permutation_p": permutation_p,
                    "permutation_repeats": CALIBRATION_PROTOCOL["conditional_permutation_repeats"],
                    "conditional_randomization": "parcel_cluster_sign_flip_after_class_time_qband_residualization",
                    "parcel_group_count": len(unique_parcels) if np.unique(y).size >= 2 else 0,
                })
    return output


def per_class_evidence_rows(repeat_rows, morphology_rows, split):
    rows = []
    for anchor in sorted({row["anchor"] for row in repeat_rows}):
        for clazz in sorted({row["class_id"] for row in repeat_rows}):
            selected = [
                row for row in repeat_rows if row["anchor"] == anchor
                and row["class_id"] == clazz
            ]
            if selected:
                rows.append({
                    "split": split, "evidence": "repeatability", "anchor": anchor,
                    "class_id": clazz, "sample_count": len(selected),
                    "positive_fraction": float(np.mean([row["label"] for row in selected])),
                    "score_mean": float(np.mean([row["q"] for row in selected])),
                    "effect_mean": float(np.mean([row["window_iou"] for row in selected])),
                })
    if morphology_rows:
        anchor_count = morphology_rows[0]["q"].shape[-1]
        classes = sorted(set(np.concatenate([packet["class_id"] for packet in morphology_rows]).tolist()))
        for anchor in range(anchor_count):
            for clazz in classes:
                for perturbation in ("reverse", "block_reorder"):
                    packets = [packet for packet in morphology_rows if packet["perturbation"] == perturbation]
                    selected_q, selected_y, selected_delta = [], [], []
                    for packet in packets:
                        mask = packet["class_id"] == clazz
                        if mask.any():
                            selected_q.append(packet["q"][mask, :, anchor].reshape(-1))
                            selected_y.append(packet["proxy_label"][mask, :, anchor].reshape(-1))
                            selected_delta.append(packet["delta"][mask, :, anchor].reshape(-1))
                    if not selected_q:
                        continue
                    q = np.concatenate(selected_q); y = np.concatenate(selected_y)
                    delta = np.concatenate(selected_delta)
                    rows.append({
                        "split": split, "evidence": "morphology_" + perturbation,
                        "anchor": anchor, "class_id": clazz,
                        "sample_count": len(q),
                        "positive_fraction": float(y.mean()),
                        "score_mean": float(q.mean()),
                        "effect_mean": float(delta.mean()),
                    })
    return rows


def _fit_candidate(evidence, anchor, train_q, train_y, train_groups, validations, seed):
    rows = []
    try:
        model = fit_monotonic_calibrator(train_q, train_y, train_groups)
    except ValueError as error:
        for validation in validations:
            name, q, y, groups = validation[:4]
            rows.append({
                "evidence": evidence, "anchor": anchor, "validation": name,
                "fit_status": "unsupported", "reason": str(error),
                "train_sample_count": len(train_y),
                "train_positive_count": int(np.sum(train_y)),
                "train_negative_count": int(len(train_y) - np.sum(train_y)),
            })
        return rows
    for validation in validations:
        name, q, y, groups = validation[:4]
        group_classes = validation[4] if len(validation) > 4 else None
        row = {
            "evidence": evidence, "anchor": anchor, "validation": name,
            "fit_status": "fit", **asdict(model),
        }
        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
        if (
            len(y) < CALIBRATION_PROTOCOL["minimum_total"]
            or counts.min() < CALIBRATION_PROTOCOL["minimum_per_class"]
        ):
            row.update({
                "fit_status": "unsupported",
                "reason": "Source Val proxy labels do not meet predeclared support",
                "sample_count": len(y), "positive_count": int(counts[1]),
                "negative_count": int(counts[0]),
            })
            rows.append(row)
            continue
        row.update(_calibration_metrics(
            model, q, y, groups, seed + anchor, group_classes=group_classes,
        ))
        rows.append(row)
    return rows


def _calibration_rows(train_views, val_views, thresholds, seed):
    train_left, train_right = pair_views_by_parcel(*train_views)
    val_left, val_right = pair_views_by_parcel(*val_views)
    starts = train_left["window_starts"]; scales = train_left["window_scales"]
    train_repeat = repeatability_rows(
        train_left["similarity"], train_right["similarity"], train_left["parcel_index"],
        starts, scales, class_labels=train_left["label"],
    )
    val_repeat = repeatability_rows(
        val_left["similarity"], val_right["similarity"], val_left["parcel_index"],
        starts, scales, class_labels=val_left["label"],
    )
    train_cross = _cross_view_morphology_all(train_views, thresholds)
    val_cross = _cross_view_morphology_all(val_views, thresholds)
    rows = []
    for anchor in range(train_left["similarity"].shape[-1]):
        train_direction = [row for row in train_repeat if row["anchor"] == anchor and row["direction"] == "view0_to_view1"]
        validations = []
        for direction in ("view0_to_view1", "view1_to_view0"):
            selected = [row for row in val_repeat if row["anchor"] == anchor and row["direction"] == direction]
            validations.append((
                direction, np.asarray([row["q"] for row in selected]),
                np.asarray([row["label"] for row in selected]),
                np.asarray([row["parcel_index"] for row in selected]),
                np.asarray([row["class_id"] for row in selected]),
            ))
        rows.extend(_fit_candidate(
            "repeatability", anchor,
            np.asarray([row["q"] for row in train_direction]),
            np.asarray([row["label"] for row in train_direction]),
            np.asarray([row["parcel_index"] for row in train_direction]),
            validations, seed,
        ))

        for perturbation in ("reverse", "block_reorder"):
            train_selected = [packet for packet in train_cross if packet["perturbation"] == perturbation]
            val_selected = [packet for packet in val_cross if packet["perturbation"] == perturbation]
            train_q = np.concatenate([packet["q"][:, :, anchor].reshape(-1) for packet in train_selected])
            train_y = np.concatenate([packet["proxy_label"][:, :, anchor].reshape(-1) for packet in train_selected])
            train_groups = np.concatenate([
                np.repeat(packet["parcel_index"], packet["q"].shape[1]) for packet in train_selected
            ])
            val_q = np.concatenate([packet["q"][:, :, anchor].reshape(-1) for packet in val_selected])
            val_y = np.concatenate([packet["proxy_label"][:, :, anchor].reshape(-1) for packet in val_selected])
            val_groups = np.concatenate([
                np.repeat(packet["parcel_index"], packet["q"].shape[1]) for packet in val_selected
            ])
            val_classes = np.concatenate([
                np.repeat(packet["class_id"], packet["q"].shape[1]) for packet in val_selected
            ])
            rows.extend(_fit_candidate(
                "morphology_" + perturbation, anchor,
                train_q, train_y, train_groups,
                [("source_val_cross_view",
                  val_q, val_y, val_groups, val_classes)],
                seed,
            ))
        train_q = np.concatenate([packet["q"][:, :, anchor].reshape(-1) for packet in train_cross])
        train_y = np.concatenate([packet["proxy_label"][:, :, anchor].reshape(-1) for packet in train_cross])
        train_groups = np.concatenate([
            np.repeat(packet["parcel_index"], packet["q"].shape[1]) for packet in train_cross
        ])
        val_q = np.concatenate([packet["q"][:, :, anchor].reshape(-1) for packet in val_cross])
        val_y = np.concatenate([packet["proxy_label"][:, :, anchor].reshape(-1) for packet in val_cross])
        val_groups = np.concatenate([
            np.repeat(packet["parcel_index"], packet["q"].shape[1]) for packet in val_cross
        ])
        val_classes = np.concatenate([
            np.repeat(packet["class_id"], packet["q"].shape[1]) for packet in val_cross
        ])
        rows.extend(_fit_candidate(
            "morphology_sensitivity", anchor,
            train_q, train_y, train_groups,
            [("source_val_cross_view",
              val_q, val_y, val_groups, val_classes)],
            seed,
        ))
    return rows, train_repeat, val_repeat, train_cross, val_cross


def _reliability_rows(calibration_rows, baseline_rows, control_rows, morphology_controls):
    output = []
    for anchor in sorted({int(row["anchor"]) for row in calibration_rows}):
        selected = [row for row in calibration_rows if int(row["anchor"]) == anchor]
        repeat_candidates = [row for row in selected if row["evidence"] == "repeatability"]
        reverse_rows = [row for row in selected if row["evidence"] == "morphology_reverse"]
        reorder_rows = [row for row in selected if row["evidence"] == "morphology_block_reorder"]
        repeatability_status = _evidence_status(repeat_candidates)
        reverse = _evidence_status(reverse_rows); reorder = _evidence_status(reorder_rows)
        baseline = {row["model"]: row for row in baseline_rows if row["anchor"] == anchor}
        increment = float("nan")
        if "class_position" in baseline and "class_position_qmax" in baseline:
            increment = float(baseline["class_position_qmax"]["auprc"]) - float(baseline["class_position"]["auprc"])
        paired_controls = [row for row in control_rows if row["anchor"] == anchor]
        morphology_anchor_controls = [row for row in morphology_controls if row["anchor"] == anchor]
        repeatability = {
            "supported": repeatability_status != "unsupported",
            "monotonic_generalizes": repeatability_status == "stable",
            "baseline_increment_supported": bool(np.isfinite(increment) and increment >= CALIBRATION_PROTOCOL["minimum_baseline_auprc_increment"]),
            "paired_shuffle_effect_supported": bool(
                paired_controls and all(float(row["paired_minus_shuffled_ci_low"]) > 0 for row in paired_controls)
            ),
        }
        perturbation_status = {}
        for name, evidence_rows in (("reverse", reverse_rows), ("block_reorder", reorder_rows)):
            controls = [row for row in morphology_anchor_controls if row["perturbation"] == name]
            conditional = bool(controls and all(
                np.isfinite(float(row["conditional_permutation_p"]))
                and float(row["conditional_permutation_p"]) <= CALIBRATION_PROTOCOL["conditional_permutation_alpha"]
                for row in controls
            ))
            perturbation_status[name] = "pass" if _evidence_status(evidence_rows) == "stable" and conditional else "fail"
        morphology = {
            "supported": reverse != "unsupported" or reorder != "unsupported",
            "cross_view_generalizes": reverse == "stable" or reorder == "stable",
            "exceeds_normal_variation": bool(
                morphology_anchor_controls and any(
                    float(row["proxy_positive_fraction"])
                    >= CALIBRATION_PROTOCOL["minimum_morphology_positive_fraction"]
                    for row in morphology_anchor_controls
                )
            ),
            "conditional_control_supported": any(value == "pass" for value in perturbation_status.values()),
            "perturbations": perturbation_status,
        }
        decision = assess_anchor_reliability(repeatability, morphology)
        output.append({
            "anchor": anchor, "repeatability_status": repeatability_status,
            "repeatability_evidence_status": repeatability_status,
            "repeatability_auprc_increment": increment,
            "morphology_reverse_status": reverse,
            "morphology_block_reorder_status": reorder,
            "morphology_status": "stable" if any(value == "pass" for value in perturbation_status.values()) else "failed",
            "joint_status": decision["status"],
            "reason_codes": json.dumps(decision["reason_codes"]),
        })
    return output


def _plot_calibration(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fitted = [row for row in rows if row.get("fit_status") == "fit"]
    if not fitted:
        return
    fig, axis = plt.subplots(figsize=(8, 4))
    for evidence in sorted({row["evidence"] for row in fitted}):
        selected = [row for row in fitted if row["evidence"] == evidence]
        axis.scatter(
            [int(row["anchor"]) for row in selected],
            [float(row["auroc"]) for row in selected], s=14, label=evidence, alpha=.7,
        )
    axis.axhline(.5, color="black", linewidth=1, linestyle="--")
    axis.set(xlabel="Anchor", ylabel="Source Val AUROC", ylim=(0, 1))
    axis.legend(fontsize=7); fig.tight_layout()
    fig.savefig(Path(output) / "calibration_validation.png", dpi=140)
    plt.close(fig)


def _plot_matching_distribution(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["split"] in ("source_train", "source_val")]
    if not selected:
        return
    fig, axis = plt.subplots(figsize=(9, 4))
    for split, marker in (("source_train", "o"), ("source_val", "x")):
        current = [row for row in selected if row["split"] == split and int(row["view_id"]) == 0]
        axis.scatter(
            [int(row["anchor"]) for row in current],
            [float(row["qmax_q50"]) for row in current],
            marker=marker, s=22, label=split,
        )
    axis.set(xlabel="Anchor", ylabel="Median parcel qmax", xticks=range(16))
    axis.legend(); fig.tight_layout()
    fig.savefig(Path(output) / "matching_distribution.png", dpi=140)
    plt.close(fig)


def settings_fingerprint(checkpoint_sha, data_root, seed, batch_size, pixel_budget):
    payload = {
        "protocol_version": P0R_PROTOCOL_VERSION,
        "checkpoint_sha256": checkpoint_sha, "data_root": str(Path(data_root)),
        "seed": int(seed), "batch_size": int(batch_size), "pixel_budget": int(pixel_budget),
        "calibration_protocol": CALIBRATION_PROTOCOL,
        "view_seed_version": VIEW_SEED_VERSION,
        "perturbation_version": PERTURBATION_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_source(args):
    output = Path(args.output_root) / args.source_alias
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256_file(args.checkpoint)
    fingerprint = settings_fingerprint(
        checkpoint_sha, args.data_root, args.seed, args.batch_size, args.pixel_budget,
    )
    if args.settings_fingerprint and args.settings_fingerprint != fingerprint:
        raise ValueError("scheduler/analysis settings fingerprint mismatch")
    model, config, protocol, splits, datasets = load_source_runtime(
        args.checkpoint, args.data_root, args.device, args.seed,
    )
    views, split_fingerprints, cache_signature = load_or_extract_views(
        output, model, datasets, config.source, args.seed, args.device,
        args.batch_size, args.pixel_budget, checkpoint_sha,
    )
    train0, train1 = pair_views_by_parcel(*views["source_train"])
    normal_change = np.abs(train0["similarity"] - train1["similarity"])
    thresholds = freeze_morphology_thresholds(normal_change)
    train_edges = [
        np.quantile(train0["similarity"][:, :, anchor].reshape(-1), [0., 1/3, 2/3, 1.])
        for anchor in range(train0["similarity"].shape[-1])
    ]
    matching = []
    morphology = []
    beta = float(model.structure_branch.shapelet_dictionary.beta)
    for split, packets in views.items():
        for view_id, packet in enumerate(packets):
            matching.extend(matching_distribution_rows(packet, split, view_id, beta))
            morphology.extend(_morphology_summary(split, packet, thresholds, train_edges))
    calibration, train_repeat, val_repeat, train_cross, val_cross = _calibration_rows(
        views["source_train"], views["source_val"], thresholds, args.seed,
    )
    starts = train0["window_starts"]; scales = train0["window_scales"]
    train_shuffled = _shuffled_repeatability(train0, train1, starts, scales, args.seed)
    val0, val1 = pair_views_by_parcel(*views["source_val"])
    val_shuffled = _shuffled_repeatability(val0, val1, starts, scales, args.seed + 1)
    repeatability_controls = (
        _paired_shuffle_control_rows(train_repeat, train_shuffled, "source_train", args.seed)
        + _paired_shuffle_control_rows(val_repeat, val_shuffled, "source_val", args.seed + 1)
    )
    repeatability_baselines = []
    for anchor in range(train0["similarity"].shape[-1]):
        repeatability_baselines.extend(
            repeatability_baseline_rows(train_repeat, val_repeat, anchor, args.seed)
        )
    morphology_controls = morphology_conditional_control_rows(
        train_cross, val_cross, args.seed,
    )
    per_class = (
        per_class_evidence_rows(train_repeat, train_cross, "source_train")
        + per_class_evidence_rows(val_repeat, val_cross, "source_val")
    )
    train_modes = {}
    for anchor in range(train0["similarity"].shape[-1]):
        selected = train0["similarity"][:, :, anchor].argmax(1)
        train_modes[anchor] = int(np.bincount(selected, minlength=train0["similarity"].shape[1]).argmax())
    repeatability = []
    repeat_edges = {}
    for anchor in range(train0["similarity"].shape[-1]):
        for direction in ("view0_to_view1", "view1_to_view0"):
            selected = [
                row for row in train_repeat
                if row["anchor"] == anchor and row["direction"] == direction
            ]
            repeat_edges[(anchor, direction)] = np.quantile(
                [row["q"] for row in selected], [0., 1/3, 2/3, 1.],
            )
    for split, rows in (("source_train", train_repeat), ("source_val", val_repeat)):
        for row in _repeatability_aggregate(
            rows, train0["window_starts"], train0["window_scales"], train_modes,
            model.structure_branch.window_extractor.grid_points,
        ):
            row["split"] = split
            repeatability.append(row)
        first, second = pair_views_by_parcel(*views[split])
        repeatability.extend(_window_repeatability_rows(first, second, split))
        repeatability.extend(_repeatability_band_rows(rows, split, repeat_edges))
    reliability = _reliability_rows(
        calibration, repeatability_baselines,
        [row for row in repeatability_controls if row["split"] == "source_val"],
        morphology_controls,
    )
    _write_csv(output / "matching_distribution.csv", matching)
    _write_csv(output / "view_repeatability.csv", repeatability)
    _write_csv(output / "morphology_counterfactual.csv", morphology)
    _write_csv(output / "calibration_candidates.csv", calibration)
    _write_csv(output / "anchor_reliability_summary.csv", reliability)
    _write_csv(output / "repeatability_controls.csv", repeatability_controls)
    _write_csv(output / "repeatability_baselines.csv", repeatability_baselines)
    _write_csv(output / "morphology_conditional_controls.csv", morphology_controls)
    _write_csv(output / "per_class_evidence.csv", per_class)
    _atomic_json(output / "bootstrap_protocol.json", {
        "protocol_version": P0R_PROTOCOL_VERSION,
        "method": "full_population_class_stratified_parcel_cluster_bootstrap",
        "repeats": CALIBRATION_PROTOCOL["bootstrap_repeats"],
        "confidence": CALIBRATION_PROTOCOL["bootstrap_confidence"],
        "seed": int(args.seed), "observation_subsampling": False,
        "parcel_subsampling": False,
    })
    _plot_calibration(output, calibration)
    _plot_matching_distribution(output, matching)
    counts = {status: sum(row["joint_status"] == status for row in reliability) for status in (
        "calibratable", "uncertain", "not_calibratable",
    )}
    (output / "P0R_source_report.md").write_text(
        "\n".join([
            f"# P0-R Anchor matching calibration: {args.source_alias}", "",
            f"- Source checkpoint: `{args.checkpoint}`", f"- Source Train parcels: {len(train0['parcel_index'])}",
            f"- Source Val parcels: {len(views['source_val'][0]['parcel_index'])}",
            f"- Joint statuses: `{json.dumps(counts, sort_keys=True)}`", "",
            "Repeatability and morphology sensitivity are separate proxy evidence streams; no unified matching probability is produced.",
            "Morphology proxy labels partly depend on original similarity, so monotonicity does not imply statistical independence or true occurrence probability.",
        ]) + "\n", encoding="utf-8",
    )
    calibration_by_anchor = {}
    for anchor in range(train0["similarity"].shape[-1]):
        current = [row for row in calibration if int(row["anchor"]) == anchor]
        def candidate(evidence):
            rows = [row for row in current if row["evidence"] == evidence and row.get("fit_status") == "fit"]
            if not rows:
                return None
            row = rows[0]
            result = {key: row.get(key) for key in (
                "slope", "intercept", "sample_count", "positive_count", "negative_count",
                "auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "brier", "ece",
            )}
            result["validation_metrics"] = [{
                key: item.get(key) for key in (
                    "validation", "sample_count", "positive_count", "negative_count",
                    "auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "brier", "ece",
                )
            } for item in rows]
            return result
        reliability_row = next(row for row in reliability if int(row["anchor"]) == anchor)
        calibration_by_anchor[anchor] = {
            "anchor": anchor, "status": reliability_row["joint_status"],
            "reason_codes": json.loads(reliability_row["reason_codes"]),
            "repeatability": candidate("repeatability"),
            "morphology": candidate("morphology_sensitivity"),
            "morphology_by_perturbation": {
                "reverse": candidate("morphology_reverse"),
                "block_reorder": candidate("morphology_block_reorder"),
            },
            "score_ranges": {
                "parcel_qmax": [
                    float(train0["similarity"][:, :, anchor].max(1).min()),
                    float(train0["similarity"][:, :, anchor].max(1).max()),
                ],
                "window_q": [
                    float(train0["similarity"][:, :, anchor].min()),
                    float(train0["similarity"][:, :, anchor].max()),
                ],
                "source_val_parcel_qmax": [
                    float(val0["similarity"][:, :, anchor].max(1).min()),
                    float(val0["similarity"][:, :, anchor].max(1).max()),
                ],
                "source_val_window_q": [
                    float(val0["similarity"][:, :, anchor].min()),
                    float(val0["similarity"][:, :, anchor].max()),
                ],
            },
            "proxy_support": {
                "repeatability_train_positive_fraction": float(np.mean([
                    row["label"] for row in train_repeat if row["anchor"] == anchor
                ])),
                "repeatability_val_positive_fraction": float(np.mean([
                    row["label"] for row in val_repeat if row["anchor"] == anchor
                ])),
                "morphology_train_positive_fraction": float(np.mean(np.concatenate([
                    packet["proxy_label"][:, :, anchor].reshape(-1) for packet in train_cross
                ]))),
                "morphology_val_positive_fraction": float(np.mean(np.concatenate([
                    packet["proxy_label"][:, :, anchor].reshape(-1) for packet in val_cross
                ]))),
            },
            "repeatability_directions": ["view0_to_view1", "view1_to_view0"],
        }
    decision = build_calibration_decision(
        args.source_alias, checkpoint_sha,
        split_fingerprints["source_train"], split_fingerprints["source_val"],
        list(calibration_by_anchor.values()), args.seed,
    )
    decision["cache_signature"] = cache_signature
    decision["ab_usefulness"] = {
        "parcel_support_candidate_anchors": [
            anchor for anchor, row in calibration_by_anchor.items()
            if row["status"] == "calibratable" and row["repeatability"] is not None
        ],
        "window_activation_candidate_anchors": [
            anchor for anchor, row in calibration_by_anchor.items()
            if row["status"] == "calibratable"
            and row["morphology"] is not None
        ],
    }
    _atomic_json(output / "calibration_decision.json", decision)
    manifest = {
        "status": "complete", "source_alias": args.source_alias,
        "protocol_version": P0R_PROTOCOL_VERSION,
        "source_domain": config.source, "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_sha256": checkpoint_sha, "settings_fingerprint": fingerprint,
        "classes": list(config.classes), "protocol": protocol,
        "split_parcel_sha256": split_fingerprints,
        "sample_counts": {split: len(datasets[split]) for split in ("source_train", "source_val")},
        "calibration_protocol": CALIBRATION_PROTOCOL,
        "morphology_thresholds": thresholds.tolist(),
        "view_seed_version": VIEW_SEED_VERSION, "perturbation_version": PERTURBATION_VERSION,
        "target_data_used": False, "training_performed": False,
        "cache_signature": cache_signature,
    }
    _atomic_json(output / "checkpoint_manifest.json", manifest)
    print(f"P0_SOURCE_COMPLETE|source={args.source_alias}|output={output}", flush=True)


def merge_sources(output_root, previous_root=None):
    output_root = Path(output_root)
    rows, missing, decisions = [], [], {}
    for source in ("AT1", "FR1", "FR2", "DK1"):
        folder = output_root / source
        manifest = folder / "checkpoint_manifest.json"
        reliability = folder / "anchor_reliability_summary.csv"
        decision_path = folder / "calibration_decision.json"
        if not manifest.is_file() or not reliability.is_file() or not decision_path.is_file():
            missing.append(source); continue
        try:
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            missing.append(source); continue
        if not (
            manifest_payload.get("status") == "complete"
            and manifest_payload.get("protocol_version") == P0R_PROTOCOL_VERSION
            and manifest_payload.get("source_alias") == source
            and decision_payload.get("protocol_version") == P0R_PROTOCOL_VERSION
            and decision_payload.get("source") == source
            and decision_payload.get("checkpoint_sha256")
            == manifest_payload.get("checkpoint_sha256")
            and decision_payload.get("split_fingerprints")
            == {
                "source_train": manifest_payload.get("split_parcel_sha256", {}).get("source_train"),
                "source_val": manifest_payload.get("split_parcel_sha256", {}).get("source_val"),
            }
        ):
            missing.append(source); continue
        decisions[source] = decision_payload
        with reliability.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append({"source": source, **row})
    _write_csv(output_root / "P0R_4source_comparison.csv", rows)
    readiness = build_ab_readiness(
        decisions, expected_sources=("AT1", "FR1", "FR2", "DK1"),
    )
    if missing:
        readiness["blocking_reason"] = "missing_or_incomplete_sources"
    previous = {
        "status": "unavailable", "path": str(previous_root) if previous_root else None,
        "note": "No legacy P0 value is inferred when its verified result directory is unavailable.",
    }
    if previous_root:
        old_csv = Path(previous_root) / "P0_4source_comparison.csv"
        if old_csv.is_file():
            with old_csv.open(newline="", encoding="utf-8") as stream:
                old_rows = list(csv.DictReader(stream))
            previous = {
                "status": "available", "path": str(old_csv),
                "anchor_decision_count": len(old_rows),
                "calibratable_count": sum(
                    row.get("joint_status") == "calibratable" for row in old_rows
                ),
                "p0r_calibratable_count": sum(
                    row.get("joint_status") == "calibratable" for row in rows
                ),
                "comparison_scope": "status counts only; protocols differ",
            }
    readiness["previous_p0_comparison"] = previous
    _atomic_json(output_root / "AB_readiness.json", readiness)
    counts = {}
    for row in rows:
        counts[row["joint_status"]] = counts.get(row["joint_status"], 0) + 1
    lines = [
        "# P0-R Anchor Matching Calibration Audit", "",
        f"Completed Sources: {4 - len(missing)}/4.",
        f"Joint Anchor statuses: `{json.dumps(counts, sort_keys=True)}`.", "",
        "Repeatability and morphology sensitivity were fitted and validated separately using Source-only labels and predeclared criteria.",
        "No unified matching probability was generated and no parameter was written back to training.",
    ]
    if missing:
        lines.extend(["", "## Missing Sources", "", *[f"- {item}" for item in missing]])
    (output_root / "P0R_FINAL_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source", "summary"), required=True)
    parser.add_argument("--source-alias")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root", default="experiments/anchor_matching_calibration_p0r_seed1")
    parser.add_argument("--previous-root", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--settings-fingerprint", default="")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "summary":
        merge_sources(args.output_root, args.previous_root or None)
    else:
        for field in ("source_alias", "checkpoint", "data_root"):
            if not getattr(args, field):
                raise ValueError(f"--{field.replace('_', '-')} is required in source mode")
        run_source(args)


if __name__ == "__main__":
    main()
