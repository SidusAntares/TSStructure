#!/usr/bin/env python3
"""Frozen A/B/C audit for cross-domain structure evidence and Query signals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from analysis.structure_semantic_features import (
    EVENT_COMPONENTS,
    FrozenModels,
    _semantic_collate,
    build_abc_datasets,
    extract_checkpoint_features,
    load_feature_split,
    load_frozen_models,
    replay_abc_datasets,
    response_to_anchor_events,
)
from analysis.diagnose_eg_pair import sha256_file, validate_source


PAIR_NAMES = {
    "FR2_DK1": (
        ("spring_barley", "winter_wheat"),
        ("winter_barley", "winter_wheat"),
        ("horsebeans", "meadow"),
    ),
    "FR1_FR2": (
        ("winter_barley", "winter_wheat"),
        ("spring_barley", "winter_wheat"),
        ("horsebeans", "meadow"),
    ),
}

STAGE_ORDER = ("extract", "A", "B", "C", "report")
STAGE_ARTIFACTS = {
    "extract": ("feature_cache_manifest.json", "checkpoint_manifest.json"),
    "A": (
        "A_anchor_event_semantics.csv", "A_pairwise_discrimination.csv",
        "A_transferability_validation.csv", "A_anchor_candidates.json",
        "A_representative_windows.png", "A_findings.md",
    ),
    "B": (
        "B_source_calibration.json", "B_sample_scores.csv",
        "B_coverage_error_curves.csv", "B_per_class_reliability.csv",
        "B_conflict_cases.csv", "B_findings.md",
    ),
    "C": (
        "C_pairwise_model_sensitivity.csv", "C_local_perturbations.csv",
        "C_gradient_groups.csv", "C_gradient_samples.csv", "C_findings.md",
    ),
    "report": ("checkpoint_report.md",),
}


def canonical_class(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def fixed_class_pairs(task, classes):
    lookup = {canonical_class(name): index for index, name in enumerate(classes)}
    return [
        {
            "y_id": lookup[left], "c_id": lookup[right],
            "y_name": classes[lookup[left]], "c_name": classes[lookup[right]],
        }
        for left, right in PAIR_NAMES[task]
        if left in lookup and right in lookup
    ]


def _effect_vector(left, right):
    n_left, n_right = len(left), len(right)
    if n_left < 2 or n_right < 2:
        return np.full(left.shape[1:] if n_left else right.shape[1:], np.nan)
    pooled = (
        (n_left - 1) * left.var(axis=0, ddof=1)
        + (n_right - 1) * right.var(axis=0, ddof=1)
    ) / max(n_left + n_right - 2, 1)
    return (left.mean(axis=0) - right.mean(axis=0)) / np.sqrt(pooled + 1e-8)


def _bootstrap_projection_ci(left, right, direction, seed, repeats=200):
    rng = np.random.default_rng(seed)
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if len(left) < 2 or len(right) < 2 or not np.isfinite(norm) or norm <= 0:
        return float("nan"), float("nan")
    unit = direction / norm
    values = []
    for _ in range(int(repeats)):
        a = left[rng.integers(0, len(left), len(left))].mean(0)
        b = right[rng.integers(0, len(right), len(right))].mean(0)
        values.append(float((a - b) @ unit))
    return tuple(float(value) for value in np.quantile(values, (.025, .975)))


def pairwise_event_statistics(
    events, labels, y_id, c_id, split, role, seed=1,
):
    events = np.asarray(events, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if events.ndim != 3 or events.shape[2] != len(EVENT_COMPONENTS):
        raise ValueError("events must be [N,anchor,6]")
    y_selected, c_selected = labels == int(y_id), labels == int(c_id)
    rows = []
    for anchor in range(events.shape[1]):
        left, right = events[y_selected, anchor], events[c_selected, anchor]
        row = {
            "role": str(role), "split": str(split), "anchor_id": anchor,
            "y_id": int(y_id), "c_id": int(c_id),
            "support_y": int(y_selected.sum()), "support_c": int(c_selected.sum()),
        }
        if len(left) < 2 or len(right) < 2:
            row.update({"status": "N/A", "effect_norm": float("nan"),
                        "projection_ci_low": float("nan"),
                        "projection_ci_high": float("nan"),
                        "dominant_component": "N/A"})
            for component in EVENT_COMPONENTS:
                row[f"difference_{component}"] = float("nan")
                row[f"effect_{component}"] = float("nan")
        else:
            difference = left.mean(0) - right.mean(0)
            effect = _effect_vector(left, right)
            pooled_std = np.divide(
                difference, effect,
                out=np.sqrt(
                    ((len(left) - 1) * left.var(0, ddof=1)
                     + (len(right) - 1) * right.var(0, ddof=1))
                    / max(len(left) + len(right) - 2, 1) + 1e-8
                ),
                where=np.abs(effect) > 1e-12,
            )
            low, high = _bootstrap_projection_ci(
                left, right, effect, int(seed) + anchor,
            )
            row.update({
                "status": "ok", "effect_norm": float(np.linalg.norm(effect)),
                "projection_ci_low": low, "projection_ci_high": high,
                "dominant_component": EVENT_COMPONENTS[int(np.nanargmax(np.abs(effect)))],
            })
            for index, component in enumerate(EVENT_COMPONENTS):
                row[f"difference_{component}"] = float(difference[index])
                row[f"effect_{component}"] = float(effect[index])
                row[f"mean_y_{component}"] = float(left[:, index].mean())
                row[f"mean_c_{component}"] = float(right[:, index].mean())
                row[f"pooled_std_{component}"] = float(max(abs(pooled_std[index]), 1e-6))
        rows.append(row)
    return rows


@dataclass(frozen=True)
class JointSupport:
    center: np.ndarray
    precision: np.ndarray
    threshold: float
    marginal_low: np.ndarray
    marginal_high: np.ndarray


def _squared_mahalanobis(values, center, precision):
    delta = np.asarray(values, dtype=np.float64) - center
    return np.einsum("ni,ij,nj->n", delta, precision, delta)


def fit_joint_support(values, quantile=.975, seed=1):
    del seed
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("joint support needs at least three [N,D] observations")
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-6)
    robust_radius = np.sqrt(np.square((values - center) / scale).sum(1))
    trim = robust_radius <= np.quantile(robust_radius, .975)
    fitted = values[trim] if int(trim.sum()) > values.shape[1] + 1 else values
    covariance = LedoitWolf().fit(fitted).covariance_
    precision = np.linalg.pinv(covariance)
    distances = _squared_mahalanobis(values, center, precision)
    return JointSupport(
        center=center, precision=precision,
        threshold=float(np.quantile(distances, quantile)),
        marginal_low=np.quantile(values, .01, axis=0),
        marginal_high=np.quantile(values, .99, axis=0),
    )


def joint_support_mask(support, values):
    values = np.asarray(values, dtype=np.float64)
    marginal = (
        (values >= support.marginal_low).all(1)
        & (values <= support.marginal_high).all(1)
    )
    joint = _squared_mahalanobis(values, support.center, support.precision) <= support.threshold
    return marginal & joint


def _row_effect(row):
    return np.asarray([row[f"effect_{name}"] for name in EVENT_COMPONENTS], dtype=float)


def _cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else float("nan")


def select_source_only_candidates(
    train_rows, val_rows, source_train_events, target_train_events,
    min_support=20, min_effect=.5, min_direction_cosine=.25,
    min_target_range=.5, role="teacher", pair=None, seed=1,
):
    source_train_events = np.asarray(source_train_events)
    target_train_events = np.asarray(target_train_events)
    by_anchor = {int(row["anchor_id"]): row for row in val_rows}
    candidates = []
    for train in train_rows:
        anchor = int(train["anchor_id"])
        val = by_anchor.get(anchor)
        if not val or train["status"] != "ok" or val["status"] != "ok":
            continue
        train_effect, val_effect = _row_effect(train), _row_effect(val)
        direction = _cosine(train_effect, val_effect)
        support = fit_joint_support(source_train_events[:, anchor], seed=seed + anchor)
        range_rate = float(joint_support_mask(support, target_train_events[:, anchor]).mean())
        minimum = min(
            int(train["support_y"]), int(train["support_c"]),
            int(val["support_y"]), int(val["support_c"]),
        )
        if (
            minimum >= int(min_support)
            and float(train["effect_norm"]) >= float(min_effect)
            and direction >= float(min_direction_cosine)
            and range_rate >= float(min_target_range)
        ):
            candidates.append({
                "role": str(role), "anchor_id": anchor,
                "pair_y": None if pair is None else int(pair[0]),
                "pair_c": None if pair is None else int(pair[1]),
                "minimum_source_support": minimum,
                "source_train_effect_norm": float(train["effect_norm"]),
                "source_val_effect_norm": float(val["effect_norm"]),
                "source_direction_cosine": direction,
                "target_unlabeled_range_rate": range_rate,
                "effect": train_effect.tolist(),
                "mean_y": [train[f"mean_y_{name}"] for name in EVENT_COMPONENTS],
                "mean_c": [train[f"mean_c_{name}"] for name in EVENT_COMPONENTS],
                "pooled_std": [train[f"pooled_std_{name}"] for name in EVENT_COMPONENTS],
                "joint_support": {
                    "center": support.center.tolist(),
                    "precision": support.precision.tolist(),
                    "threshold": support.threshold,
                    "marginal_low": support.marginal_low.tolist(),
                    "marginal_high": support.marginal_high.tolist(),
                },
            })
    return candidates


def fit_source_structure_support(candidates):
    candidates = [dict(row) for row in candidates]
    wrong_role = {row.get("role") for row in candidates} - {"teacher"}
    if wrong_role:
        raise ValueError("B accepts EMA Teacher-space candidates only")
    return {
        "status": "ok" if candidates else "no_reliable_candidates",
        "role": "teacher", "fit_split": "source_train",
        "selection_uses_target_truth": False,
        "candidates": candidates,
    }


def _candidate_support(candidate):
    raw = candidate["joint_support"]
    return JointSupport(
        center=np.asarray(raw["center"], dtype=float),
        precision=np.asarray(raw["precision"], dtype=float),
        threshold=float(raw["threshold"]),
        marginal_low=np.asarray(raw["marginal_low"], dtype=float),
        marginal_high=np.asarray(raw["marginal_high"], dtype=float),
    )


def score_structure_support(calibration, events, predictions):
    events = np.asarray(events, dtype=float)
    predictions = np.asarray(predictions, dtype=np.int64)
    if events.ndim != 3 or len(events) != len(predictions):
        raise ValueError("events/predictions must align")
    margin = np.full(len(events), np.nan)
    range_fraction = np.full(len(events), np.nan)
    evidence_count = np.zeros(len(events), dtype=np.int64)
    for sample in range(len(events)):
        scores, ranges, weights = [], [], []
        predicted = int(predictions[sample])
        for candidate in calibration.get("candidates", []):
            y_id, c_id = int(candidate["pair_y"]), int(candidate["pair_c"])
            if predicted not in (y_id, c_id):
                continue
            anchor = int(candidate["anchor_id"])
            value = events[sample, anchor]
            effect = np.asarray(candidate["effect"], dtype=float)
            unit = effect / max(np.linalg.norm(effect), 1e-12)
            midpoint = .5 * (
                np.asarray(candidate["mean_y"], dtype=float)
                + np.asarray(candidate["mean_c"], dtype=float)
            )
            scale = np.maximum(np.asarray(candidate["pooled_std"], dtype=float), 1e-6)
            signed = float(((value - midpoint) / scale) @ unit)
            if predicted == c_id:
                signed = -signed
            weight = max(
                float(candidate.get("source_direction_cosine", 0.)), 0.,
            ) * max(float(candidate.get("source_train_effect_norm", 0.)), 1e-6)
            scores.append(signed); weights.append(weight)
            ranges.append(bool(joint_support_mask(_candidate_support(candidate), value[None])[0]))
        if scores:
            weights = np.asarray(weights, dtype=float)
            if weights.sum() <= 0:
                weights = np.ones_like(weights)
            margin[sample] = float(np.average(scores, weights=weights))
            range_fraction[sample] = float(np.mean(ranges))
            evidence_count[sample] = len(scores)
    available = evidence_count > 0
    return {
        "margin": margin,
        "range_fraction": range_fraction,
        "evidence_count": evidence_count,
        "available": available,
        "out_of_range": available & (range_fraction < .5),
        "conflict": available & (margin < 0),
    }


def calibrate_source_threshold(scores, correct_mask, minimum_coverage=.5):
    correct_mask = np.asarray(correct_mask, dtype=bool)
    values = np.asarray(scores["margin"], dtype=float)
    eligible = np.asarray(scores["available"], dtype=bool) & np.isfinite(values)
    if not eligible.any():
        return {
            "fit_split": "source_val", "status": "no_reliable_candidates",
            "structure_threshold": float("nan"), "source_val_coverage": 0.,
            "source_val_accuracy": float("nan"),
        }
    candidates = np.unique(values[eligible])
    best = None
    for threshold in candidates:
        accepted = eligible & (values >= threshold) & ~np.asarray(scores["out_of_range"])
        coverage = float(accepted.sum() / max(eligible.sum(), 1))
        if coverage + 1e-12 < float(minimum_coverage) or not accepted.any():
            continue
        accuracy = float(correct_mask[accepted].mean())
        key = (accuracy, coverage, threshold)
        if best is None or key > best[0]:
            best = (key, threshold, coverage, accuracy)
    if best is None:
        threshold = float(np.nanmin(values[eligible]))
        accepted = eligible & (values >= threshold)
        coverage = float(accepted.sum() / eligible.sum())
        accuracy = float(correct_mask[accepted].mean())
    else:
        _, threshold, coverage, accuracy = best
    return {
        "fit_split": "source_val", "status": "ok",
        "structure_threshold": float(threshold),
        "source_val_coverage": coverage, "source_val_accuracy": accuracy,
        "minimum_coverage": float(minimum_coverage),
    }


def _sample_key_digest(keys):
    serialized = "\n".join(f"{domain}\t{int(parcel)}" for domain, parcel in keys)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _metric_row(method, selected, labels, predictions, eligible, digest, point):
    selected = np.asarray(selected, dtype=bool)
    accepted = int(selected.sum())
    return {
        "curve_point": point, "method": method,
        "eligible": int(eligible.sum()), "accepted": accepted,
        "coverage": float(accepted / max(int(eligible.sum()), 1)),
        "error_rate": float((predictions[selected] != labels[selected]).mean())
        if accepted else float("nan"),
        "accuracy": float((predictions[selected] == labels[selected]).mean())
        if accepted else float("nan"),
        "sample_key_digest": digest,
    }


def coverage_error_curves(
    keys, labels, predictions, confidence, anchor_reliability, structure_score,
    confidence_threshold=.9, structure_threshold=0.,
):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=float)
    anchor_reliability = np.asarray(anchor_reliability, dtype=float)
    structure_score = np.asarray(structure_score, dtype=float)
    if not all(len(value) == len(keys) for value in (
        labels, predictions, confidence, anchor_reliability, structure_score,
    )):
        raise ValueError("all B comparison arrays must use the same Target keys")
    eligible = confidence >= float(confidence_threshold)
    structure_selected = eligible & np.isfinite(structure_score) & (
        structure_score >= float(structure_threshold)
    )
    count = int(structure_selected.sum())
    digest = _sample_key_digest(keys)

    def top_count(score):
        selected = np.zeros(len(score), dtype=bool)
        indices = np.flatnonzero(eligible)
        if count:
            order = indices[np.argsort(-score[indices], kind="stable")]
            selected[order[:count]] = True
        return selected

    rows = [
        _metric_row("confidence_raw_0.9", eligible, labels, predictions, eligible, digest, "raw"),
        _metric_row("structure_support", structure_selected, labels, predictions, eligible, digest, "matched"),
        _metric_row("confidence_matched", top_count(confidence), labels, predictions, eligible, digest, "matched"),
        _metric_row("anchor_reliability_matched", top_count(anchor_reliability), labels, predictions, eligible, digest, "matched"),
    ]
    for accepted in sorted(set(np.linspace(1, max(int(eligible.sum()), 1), 20, dtype=int))):
        if accepted > int(eligible.sum()):
            continue
        for method, score in (
            ("confidence", confidence), ("anchor_reliability", anchor_reliability),
            ("structure_support", np.nan_to_num(structure_score, nan=-np.inf)),
        ):
            selected = np.zeros(len(score), dtype=bool)
            indices = np.flatnonzero(eligible)
            order = indices[np.argsort(-score[indices], kind="stable")]
            selected[order[:accepted]] = True
            rows.append(_metric_row(
                method, selected, labels, predictions, eligible, digest,
                f"count_{accepted}",
            ))
    return rows


def _matched_masks(confidence, anchor, structure, confidence_threshold, structure_threshold):
    confidence = np.asarray(confidence, dtype=float)
    eligible = confidence >= confidence_threshold
    structure_mask = eligible & np.isfinite(structure) & (structure >= structure_threshold)
    count = int(structure_mask.sum())

    def top(score):
        result = np.zeros(len(score), dtype=bool)
        indices = np.flatnonzero(eligible)
        if count:
            ordered = indices[np.argsort(-np.asarray(score)[indices], kind="stable")]
            result[ordered[:count]] = True
        return result

    return {
        "confidence_raw_0.9": eligible,
        "confidence_matched": top(confidence),
        "anchor_reliability_matched": top(anchor),
        "structure_support": structure_mask,
    }


def run_b_stage(cache_dir, output_dir, classes, confidence_threshold=.9):
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    candidate_packet = json.loads(
        (output_dir / "A_anchor_candidates.json").read_text(encoding="utf-8")
    )
    teacher_candidates = candidate_packet.get("roles", {}).get("teacher", [])
    calibration = fit_source_structure_support(teacher_candidates)
    source_val = load_feature_split(cache_dir, "teacher", "source_val")
    target_val = load_feature_split(cache_dir, "teacher", "target_val")
    target_train = load_feature_split(cache_dir, "teacher", "target_train")
    source_scores = score_structure_support(
        calibration, source_val["events"], source_val["prediction"],
    )
    threshold = calibrate_source_threshold(
        source_scores, source_val["prediction"] == source_val["label"],
    )
    calibration.update(threshold)
    cache_manifest = json.loads(
        (cache_dir / "feature_cache_manifest.json").read_text(encoding="utf-8")
    )
    calibration["teacher_applied_shifts"] = cache_manifest["applied_shifts"]["teacher"]
    calibration["teacher_source_target_coordinate_consistent"] = len(
        set(calibration["teacher_applied_shifts"].values())
    ) == 1
    calibration["confidence_threshold"] = float(confidence_threshold)
    _write_json(output_dir / "B_source_calibration.json", calibration)

    target_scores = score_structure_support(
        calibration, target_val["events"], target_val["prediction"],
    )
    target_train_scores = score_structure_support(
        calibration, target_train["events"], target_train["prediction"],
    )
    anchor = target_val["similarity"].max(axis=1).mean(axis=1)
    keys = tuple(zip(target_val["domain"].tolist(), target_val["parcel_index"].tolist()))
    structure_threshold = float(calibration.get("structure_threshold", float("nan")))
    effective_threshold = structure_threshold if np.isfinite(structure_threshold) else float("inf")
    curve_rows = coverage_error_curves(
        keys, target_val["label"], target_val["prediction"],
        target_val["confidence"], anchor, target_scores["margin"],
        confidence_threshold=confidence_threshold,
        structure_threshold=effective_threshold,
    )
    _write_csv(output_dir / "B_coverage_error_curves.csv", curve_rows)

    masks = _matched_masks(
        target_val["confidence"], anchor, target_scores["margin"],
        confidence_threshold, effective_threshold,
    )
    sample_rows, conflict_rows = [], []
    for index, (domain, parcel) in enumerate(keys):
        row = {
            "split": "target_val",
            "domain": domain, "parcel_index": int(parcel),
            "true_class": classes[int(target_val["label"][index])],
            "teacher_prediction": classes[int(target_val["prediction"][index])],
            "teacher_confidence": float(target_val["confidence"][index]),
            "confidence_gate_0.9": bool(masks["confidence_raw_0.9"][index]),
            "anchor_reliability": float(anchor[index]),
            "structure_margin": float(target_scores["margin"][index]),
            "structure_range_fraction": float(target_scores["range_fraction"][index]),
            "structure_evidence_count": int(target_scores["evidence_count"][index]),
            "structure_available": bool(target_scores["available"][index]),
            "structure_conflict": bool(target_scores["conflict"][index]),
            "structure_out_of_range": bool(target_scores["out_of_range"][index]),
            **{f"accepted_{method}": bool(mask[index]) for method, mask in masks.items()},
        }
        sample_rows.append(row)
        if row["confidence_gate_0.9"] and (
            row["structure_conflict"] or row["structure_out_of_range"]
        ):
            conflict_rows.append(row)
    train_anchor = target_train["similarity"].max(axis=1).mean(axis=1)
    train_structure_mask = (
        (target_train["confidence"] >= confidence_threshold)
        & target_train_scores["available"]
        & ~target_train_scores["out_of_range"]
        & (target_train_scores["margin"] >= effective_threshold)
    )
    for index in range(len(target_train["parcel_index"])):
        row = {
            "split": "target_train",
            "domain": str(target_train["domain"][index]),
            "parcel_index": int(target_train["parcel_index"][index]),
            "true_class": "",
            "teacher_prediction": classes[int(target_train["prediction"][index])],
            "teacher_prediction_id": int(target_train["prediction"][index]),
            "teacher_confidence": float(target_train["confidence"][index]),
            "confidence_gate_0.9": bool(target_train["confidence"][index] >= confidence_threshold),
            "anchor_reliability": float(train_anchor[index]),
            "structure_margin": float(target_train_scores["margin"][index]),
            "structure_range_fraction": float(target_train_scores["range_fraction"][index]),
            "structure_evidence_count": int(target_train_scores["evidence_count"][index]),
            "structure_available": bool(target_train_scores["available"][index]),
            "structure_conflict": bool(target_train_scores["conflict"][index]),
            "structure_out_of_range": bool(target_train_scores["out_of_range"][index]),
            "accepted_structure_support": bool(train_structure_mask[index]),
        }
        sample_rows.append(row)
        if row["confidence_gate_0.9"] and (
            row["structure_conflict"] or row["structure_out_of_range"]
        ):
            conflict_rows.append(row)
    _write_csv(output_dir / "B_sample_scores.csv", sample_rows)
    _write_csv(output_dir / "B_conflict_cases.csv", conflict_rows)

    per_class = []
    labels, predictions = target_val["label"], target_val["prediction"]
    for class_id, class_name in enumerate(classes):
        truth = labels == class_id
        for method, mask in masks.items():
            selected = truth & mask
            per_class.append({
                "scope": "class", "class_id": class_id, "class_name": class_name,
                "method": method, "support": int(truth.sum()),
                "accepted": int(selected.sum()),
                "coverage": float(selected.sum() / max(int(truth.sum()), 1)),
                "accuracy": float((predictions[selected] == labels[selected]).mean())
                if selected.any() else float("nan"),
            })
    bins = ((.9, .95), (.95, .98), (.98, 1.000001))
    for low, high in bins:
        selected = (target_val["confidence"] >= low) & (target_val["confidence"] < high)
        for method, accepted in (
            ("confidence_raw", selected),
            ("structure_support", selected & masks["structure_support"]),
        ):
            per_class.append({
                "scope": "confidence_bin", "class_id": "ALL",
                "class_name": "ALL", "method": method,
                "confidence_low": low, "confidence_high": min(high, 1.),
                "support": int(selected.sum()), "accepted": int(accepted.sum()),
                "coverage": float(accepted.sum() / max(int(selected.sum()), 1)),
                "accuracy": float((predictions[accepted] == labels[accepted]).mean())
                if accepted.any() else float("nan"),
                "mean_structure_margin": float(np.nanmean(target_scores["margin"][selected]))
                if selected.any() and np.isfinite(target_scores["margin"][selected]).any()
                else float("nan"),
            })
    _write_csv(output_dir / "B_per_class_reliability.csv", per_class)

    lookup = {canonical_class(name): index for index, name in enumerate(classes)}
    spring, wheat = lookup.get("spring_barley"), lookup.get("winter_wheat")
    if spring is None or wheat is None:
        spring_line = "Spring Barley→Winter Wheat: N/A (class unavailable)."
    else:
        wrong = (labels == spring) & (predictions == wheat) & masks["confidence_raw_0.9"]
        retained_wrong = wrong & masks["structure_support"]
        correct_spring = (labels == spring) & (predictions == spring) & masks["confidence_raw_0.9"]
        retained_correct = correct_spring & masks["structure_support"]
        spring_line = (
            f"Spring Barley→Winter Wheat high-confidence errors retained: "
            f"{int(retained_wrong.sum())}/{int(wrong.sum())}; correct high-confidence "
            f"Spring Barley retained: {int(retained_correct.sum())}/{int(correct_spring.sum())}."
        )
        if correct_spring.any() and not retained_correct.any():
            spring_line += (
                " FAIRNESS FAILURE: the structure gate eliminated every correct "
                "high-confidence Spring Barley example."
            )
    raw = next(row for row in curve_rows if row["method"] == "confidence_raw_0.9")
    structure = next(
        row for row in curve_rows
        if row["method"] == "structure_support" and row["curve_point"] == "matched"
    )
    matched_conf = next(
        row for row in curve_rows
        if row["method"] == "confidence_matched" and row["curve_point"] == "matched"
    )
    extra = (
        np.isfinite(structure["error_rate"]) and np.isfinite(matched_conf["error_rate"])
        and structure["error_rate"] < matched_conf["error_rate"]
    )
    (output_dir / "B_findings.md").write_text(
        "# B — Target pseudo-label reliability\n\n"
        f"- Teacher confidence gate: {confidence_threshold:.2f}; eligible: {raw['accepted']}.\n"
        f"- Structure candidates: {len(teacher_candidates)}; status: {calibration['status']}.\n"
        f"- Structure accepted: {structure['accepted']}; matched-confidence error: "
        f"{matched_conf['error_rate']}; structure error: {structure['error_rate']}.\n"
        f"- Incremental reliability beyond confidence at matched coverage: {bool(extra)}.\n"
        f"- {spring_line}\n\n"
        "All methods use the same Target Val keys and the same confidence>=0.9 candidate "
        "scope. Thresholds and candidates use Source only; Target truth appears only in this "
        "offline evaluation. Reducing coverage alone is not treated as success. Target inference "
        "is deterministic frozen-Teacher inference, not an exact replay of historical "
        "strong-augmentation batches.\n",
        encoding="utf-8",
    )
    return {
        "calibration": calibration, "scores": target_scores,
        "target_train_scores": target_train_scores,
        "target_train_structure_mask": train_structure_mask,
        "masks": masks, "curve_rows": curve_rows,
    }


def _move_batch(batch, device):
    target = torch.device(device)
    return {
        key: value.to(target) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _forward_from_response(model, spatial, shifted_positions, response):
    query = model.structure_branch.response_to_query(response)
    instance = model.temporal_encoder(
        spatial, shifted_positions, external_query=query,
    )
    return model.decoder(instance)


def _pair_margin_tensor(logits, y_id, c_id):
    y_id = torch.as_tensor(y_id, device=logits.device, dtype=torch.long).reshape(-1)
    c_id = torch.as_tensor(c_id, device=logits.device, dtype=torch.long).reshape(-1)
    if y_id.numel() == 1:
        y_id = y_id.expand(logits.shape[0])
    if c_id.numel() == 1:
        c_id = c_id.expand(logits.shape[0])
    return logits.gather(1, y_id[:, None]).squeeze(1) - logits.gather(
        1, c_id[:, None],
    ).squeeze(1)


def _prepared_response(model, batch, temporal_shift):
    device = next(model.parameters()).device
    batch = _move_batch(batch, device)
    with torch.no_grad():
        spatial = model.spatial_encoder(
            batch["pixels"], batch["valid_pixels"], batch["extra"],
        )
        structure = model.prepare_structure(
            spatial, batch["positions"], temporal_shift,
        )
    response = structure["shapelet_response"].detach()
    shifted_positions = batch["positions"] + temporal_shift
    return batch, spatial.detach(), shifted_positions, response


def normal_pair_margin_sensitivity(
    model, batch, y_id, c_id, temporal_shift=0,
):
    batch, spatial, shifted_positions, native = _prepared_response(
        model, batch, temporal_shift,
    )
    response = native.clone().requires_grad_(True)
    logits = _forward_from_response(model, spatial, shifted_positions, response)
    margin = _pair_margin_tensor(logits, y_id, c_id)
    gradient = torch.autograd.grad(margin.sum(), response, retain_graph=False)[0]
    flat = gradient.detach().cpu().numpy()
    events = response_to_anchor_events(flat)
    return {
        "logits": logits.detach().cpu().numpy(),
        "margin": margin.detach().cpu().numpy(),
        "response": native.cpu().numpy(),
        "response_events": response_to_anchor_events(native.cpu().numpy()),
        "response_gradient_flat": flat,
        "response_gradient": events,
    }


def apply_in_range_event_perturbation(
    model, batch, support, anchor_id, component_id, requested_delta,
    y_id, c_id, temporal_shift=0, max_halvings=12,
):
    batch, spatial, shifted_positions, native = _prepared_response(
        model, batch, temporal_shift,
    )
    if native.shape[0] != 1:
        raise ValueError("local perturbation expects one sampled observation")
    anchor_id, component_id = int(anchor_id), int(component_id)
    flat_index = component_id * 16 + anchor_id
    base_event = response_to_anchor_events(native.cpu().numpy())[0, anchor_id]
    with torch.no_grad():
        baseline_logits = _forward_from_response(
            model, spatial, shifted_positions, native,
        )
        baseline_margin = float(_pair_margin_tensor(
            baseline_logits, y_id, c_id,
        ).item())
    applied, feasible = float(requested_delta), False
    perturbed_event = base_event.copy()
    for _ in range(int(max_halvings) + 1):
        perturbed_event = base_event.copy()
        perturbed_event[component_id] += applied
        feasible = bool(joint_support_mask(support, perturbed_event[None])[0])
        if feasible:
            break
        applied *= .5
    if not feasible:
        return {
            "accepted": False, "joint_feasible": False,
            "requested_delta": float(requested_delta),
            "applied_delta": 0., "actual_margin_delta": float("nan"),
        }
    perturbed = native.clone()
    perturbed[0, flat_index] += applied
    with torch.no_grad():
        logits = _forward_from_response(model, spatial, shifted_positions, perturbed)
        changed = float(_pair_margin_tensor(logits, y_id, c_id).item())
    return {
        "accepted": True, "joint_feasible": True,
        "requested_delta": float(requested_delta),
        "applied_delta": float(applied),
        "baseline_margin": baseline_margin, "perturbed_margin": changed,
        "actual_margin_delta": changed - baseline_margin,
        "base_event": base_event, "perturbed_event": perturbed_event,
    }


def sample_query_gradients(model, batch, labels, temporal_shift=0):
    device = next(model.parameters()).device
    batch = _move_batch(batch, device)
    labels = torch.as_tensor(labels, device=device, dtype=torch.long)
    groups = {
        "response_to_query": tuple(model.structure_branch.response_to_query.parameters()),
        "external_query_projection": tuple(
            model.temporal_encoder.attention_heads.external_query_projection.parameters()
        ),
    }
    tracked = tuple(parameter for values in groups.values() for parameter in values)
    original_requires = [parameter.requires_grad for parameter in tracked]
    for parameter in tracked:
        parameter.requires_grad_(True)
        parameter.grad = None
    try:
        logits = model.forward_with_temporal_shift(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch["extra"], temporal_shift=temporal_shift, return_dict=True,
        )["logits"]
        losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        vectors = {name: [] for name in groups}
        zero = {name: [] for name in groups}
        for sample in range(len(losses)):
            gradients = torch.autograd.grad(
                losses[sample], tracked, retain_graph=sample + 1 < len(losses),
                allow_unused=True,
            )
            offset = 0
            for name, parameters in groups.items():
                pieces = []
                for parameter in parameters:
                    gradient = gradients[offset]
                    offset += 1
                    pieces.append(
                        torch.zeros_like(parameter).flatten()
                        if gradient is None else gradient.detach().flatten()
                    )
                vector = torch.cat(pieces).cpu().numpy()
                vectors[name].append(vector)
                zero[name].append(bool(np.linalg.norm(vector) == 0.))
        return {
            "vectors": {name: np.stack(values) for name, values in vectors.items()},
            "zero": {name: np.asarray(values, dtype=bool) for name, values in zero.items()},
            "loss": losses.detach().cpu().numpy(),
            "labels": labels.detach().cpu().numpy(),
        }
    finally:
        for parameter, required in zip(tracked, original_requires):
            parameter.requires_grad_(required)
            parameter.grad = None


def build_target_gradient_groups(scores, prediction, confidence, threshold):
    prediction = np.asarray(prediction, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=float)
    eligible = confidence >= .9
    supported = (
        eligible & np.asarray(scores["available"], dtype=bool)
        & ~np.asarray(scores["conflict"], dtype=bool)
        & ~np.asarray(scores["out_of_range"], dtype=bool)
        & (np.asarray(scores["margin"], dtype=float) >= float(threshold))
    )
    return {
        "supported": supported,
        "conflict_or_unsupported": eligible & ~supported,
        "prediction": prediction,
    }


def _mean_direction(values):
    values = np.asarray(values, dtype=float)
    return values.mean(0) if len(values) else np.asarray([])


def summarize_gradient_groups(packets, class_ids):
    rows = []
    means = {}
    for group, packet in packets.items():
        labels = np.asarray(class_ids[group], dtype=np.int64)
        for module, vectors in packet["vectors"].items():
            vectors = np.asarray(vectors, dtype=float)
            if len(vectors) != len(labels):
                raise ValueError("gradient vectors and class ids must align")
            for scope, class_id, selected in [
                ("overall", "ALL", np.ones(len(labels), dtype=bool)),
                *[("class", int(value), labels == value) for value in np.unique(labels)],
            ]:
                current = vectors[selected]
                mean = _mean_direction(current)
                means[(group, module, scope, class_id)] = mean
                rows.append({
                    "record_type": "group", "group": group, "module": module,
                    "scope": scope, "class_id": class_id,
                    "support": int(selected.sum()),
                    "gradient_dim": int(vectors.shape[1]),
                    "mean_gradient_norm": float(np.linalg.norm(current, axis=1).mean())
                    if len(current) else float("nan"),
                    "mean_direction_norm": float(np.linalg.norm(mean))
                    if len(mean) else float("nan"),
                    "zero_gradient_rate": float((np.linalg.norm(current, axis=1) == 0).mean())
                    if len(current) else float("nan"),
                })
    groups = list(packets)
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1:]:
            modules = set(packets[left]["vectors"]) & set(packets[right]["vectors"])
            for module in modules:
                scopes = set(
                    (scope, class_id) for group, mod, scope, class_id in means
                    if group == left and mod == module
                ) & set(
                    (scope, class_id) for group, mod, scope, class_id in means
                    if group == right and mod == module
                )
                for scope, class_id in scopes:
                    rows.append({
                        "record_type": "cosine", "group": f"{left}__{right}",
                        "module": module, "scope": scope, "class_id": class_id,
                        "support": "", "gradient_dim": len(means[(left, module, scope, class_id)]),
                        "direction_cosine": _cosine(
                            means[(left, module, scope, class_id)],
                            means[(right, module, scope, class_id)],
                        ),
                    })
    return rows


def _batch_for_parcels(dataset, parcel_ids):
    available = {
        int(parcel): index
        for index, parcel in enumerate(dataset.get_parcel_indices().tolist())
    }
    missing = [int(parcel) for parcel in parcel_ids if int(parcel) not in available]
    if missing:
        raise ValueError(f"sample keys missing from replayed split: {missing[:5]}")
    return _semantic_collate([dataset[available[int(parcel)]] for parcel in parcel_ids])


def _balanced_target_indices(supported, conflict, predictions, maximum, seed):
    rng = np.random.default_rng(seed)
    left, right = [], []
    for class_id in np.unique(predictions):
        a = np.flatnonzero(supported & (predictions == class_id))
        b = np.flatnonzero(conflict & (predictions == class_id))
        count = min(len(a), len(b), int(maximum))
        if count:
            left.extend(rng.choice(a, count, replace=False).tolist())
            right.extend(rng.choice(b, count, replace=False).tolist())
    return np.asarray(sorted(left), dtype=np.int64), np.asarray(sorted(right), dtype=np.int64)


def _stratified_indices(labels, maximum, seed, allowed=None):
    labels = np.asarray(labels, dtype=np.int64)
    allowed = np.ones(len(labels), dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    rng = np.random.default_rng(seed)
    result = []
    for class_id in np.unique(labels[allowed]):
        indices = np.flatnonzero(allowed & (labels == class_id))
        count = min(len(indices), int(maximum))
        result.extend(rng.choice(indices, count, replace=False).tolist())
    return np.asarray(sorted(result), dtype=np.int64)


def _empty_gradient_packet(model):
    dimensions = {
        "response_to_query": sum(
            parameter.numel() for parameter in model.structure_branch.response_to_query.parameters()
        ),
        "external_query_projection": sum(
            parameter.numel() for parameter in
            model.temporal_encoder.attention_heads.external_query_projection.parameters()
        ),
    }
    return {
        "vectors": {name: np.empty((0, dim)) for name, dim in dimensions.items()},
        "zero": {name: np.empty(0, dtype=bool) for name in dimensions},
        "loss": np.empty(0), "labels": np.empty(0, dtype=np.int64),
    }


def _gradient_sample_rows(group, packet, labels, parcels, domain):
    rows = []
    for module, vectors in packet["vectors"].items():
        for index, vector in enumerate(vectors):
            rows.append({
                "group": group, "domain": domain,
                "parcel_index": int(parcels[index]),
                "class_id": int(labels[index]), "module": module,
                "loss": float(packet["loss"][index]),
                "gradient_dim": int(len(vector)),
                "gradient_norm": float(np.linalg.norm(vector)),
                "zero_gradient": bool(np.linalg.norm(vector) == 0.),
                "finite": bool(np.isfinite(vector).all()),
            })
    return rows


def run_c_stage(
    frozen, datasets, cache_dir, output_dir, task, classes,
    seed=1, max_per_class=4,
):
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    student = frozen.student
    candidates_packet = json.loads(
        (output_dir / "A_anchor_candidates.json").read_text(encoding="utf-8")
    )
    student_candidates = candidates_packet.get("roles", {}).get("student", [])
    candidate_by_pair = defaultdict(list)
    for candidate in student_candidates:
        candidate_by_pair[candidate.get("pair", "")].append(candidate)
    unstable = set()
    transfer_path = output_dir / "A_transferability_validation.csv"
    if transfer_path.is_file():
        with transfer_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("role") == "student" and row.get("validation_status") == "failed":
                    unstable.add((row.get("pair"), int(row["anchor_id"])))

    target_val = load_feature_split(cache_dir, "student", "target_val")
    sensitivity_rows, perturbation_rows = [], []
    for pair in _configured_pairs(task, classes):
        if pair["y_id"] is None or pair["c_id"] is None:
            continue
        pair_key = f"{pair['y_key']}__{pair['c_key']}"
        candidates = candidate_by_pair.get(pair_key, [])
        diagnostic_candidates = candidates or [{
            "anchor_id": 0, "effect": [float("nan")] * 6,
            "pair": pair_key, "fallback": True,
        }]
        pair_mask = np.isin(target_val["label"], [pair["y_id"], pair["c_id"]])
        indices = _stratified_indices(
            target_val["label"], maximum=2, seed=seed, allowed=pair_mask,
        )
        for index in indices:
            batch = _batch_for_parcels(
                datasets["target_val"], [target_val["parcel_index"][index]],
            )
            result = normal_pair_margin_sensitivity(
                student, batch, pair["y_id"], pair["c_id"], temporal_shift=0,
            )
            for candidate in diagnostic_candidates:
                anchor = int(candidate["anchor_id"])
                gradient = result["response_gradient"][0, anchor]
                effect = np.asarray(candidate.get("effect", [np.nan] * 6), dtype=float)
                alignment = _cosine(gradient, effect) if np.isfinite(effect).all() else float("nan")
                row = {
                    "pair": pair_key, "parcel_index": int(target_val["parcel_index"][index]),
                    "true_class": classes[int(target_val["label"][index])],
                    "anchor_id": anchor,
                    "source_candidate": not candidate.get("fallback", False),
                    "target_validation_unstable": (pair_key, anchor) in unstable,
                    "normal_margin": float(result["margin"][0]),
                    "gradient_effect_direction_cosine": alignment,
                }
                for component_index, component in enumerate(EVENT_COMPONENTS):
                    row[f"sensitivity_{component}"] = float(gradient[component_index])
                sensitivity_rows.append(row)
                if "joint_support" in candidate:
                    component = int(np.argmax(np.abs(effect)))
                    direction = 1. if effect[component] >= 0 else -1.
                    requested = direction * .1 * float(candidate["pooled_std"][component])
                    perturbed = apply_in_range_event_perturbation(
                        student, batch, _candidate_support(candidate), anchor,
                        component, requested, pair["y_id"], pair["c_id"],
                    )
                    perturbation_rows.append({
                        "pair": pair_key,
                        "parcel_index": int(target_val["parcel_index"][index]),
                        "anchor_id": anchor,
                        "component": EVENT_COMPONENTS[component],
                        "source_candidate": True,
                        "target_validation_unstable": (pair_key, anchor) in unstable,
                        **{key: value for key, value in perturbed.items()
                           if key not in ("base_event", "perturbed_event")},
                    })
    _write_csv(output_dir / "C_pairwise_model_sensitivity.csv", sensitivity_rows)
    _write_csv(output_dir / "C_local_perturbations.csv", perturbation_rows)

    calibration = json.loads(
        (output_dir / "B_source_calibration.json").read_text(encoding="utf-8")
    )
    teacher_target = load_feature_split(cache_dir, "teacher", "target_train")
    target_scores = score_structure_support(
        calibration, teacher_target["events"], teacher_target["prediction"],
    )
    threshold = calibration.get("structure_threshold", float("inf"))
    if not np.isfinite(threshold):
        threshold = float("inf")
    target_groups = build_target_gradient_groups(
        target_scores, teacher_target["prediction"], teacher_target["confidence"],
        threshold,
    )
    supported_indices, conflict_indices = _balanced_target_indices(
        target_groups["supported"], target_groups["conflict_or_unsupported"],
        teacher_target["prediction"], max_per_class, seed,
    )
    source = load_feature_split(cache_dir, "student", "source_val")
    target_classes = np.unique(np.concatenate((
        teacher_target["prediction"][supported_indices],
        teacher_target["prediction"][conflict_indices],
    ))) if len(supported_indices) + len(conflict_indices) else np.asarray([], dtype=int)
    source_allowed = np.isin(source["label"], target_classes) if len(target_classes) else np.ones(len(source["label"]), dtype=bool)
    source_indices = _stratified_indices(
        source["label"], max_per_class, seed, allowed=source_allowed,
    )

    selections = {
        "source": (datasets["source_val"], source, source_indices, source["label"]),
        "target_supported": (
            datasets["target_train"], teacher_target, supported_indices,
            teacher_target["prediction"],
        ),
        "target_conflict": (
            datasets["target_train"], teacher_target, conflict_indices,
            teacher_target["prediction"],
        ),
    }
    packets, group_labels, sample_rows = {}, {}, []
    for group, (dataset, cache, indices, labels) in selections.items():
        current_labels = np.asarray(labels)[indices].astype(np.int64)
        parcels = cache["parcel_index"][indices].astype(np.int64)
        if len(indices):
            batch = _batch_for_parcels(dataset, parcels)
            packet = sample_query_gradients(
                student, batch, torch.as_tensor(current_labels), temporal_shift=0,
            )
        else:
            packet = _empty_gradient_packet(student)
        packets[group] = packet
        group_labels[group] = current_labels
        sample_rows.extend(_gradient_sample_rows(
            group, packet, current_labels, parcels,
            str(cache["domain"][0]) if len(cache["domain"]) else "",
        ))
    group_rows = summarize_gradient_groups(packets, group_labels)
    _write_csv(output_dir / "C_gradient_groups.csv", group_rows)
    _write_csv(output_dir / "C_gradient_samples.csv", sample_rows)

    unstable_rows = [row for row in sensitivity_rows if row["target_validation_unstable"]]
    support_counts = {group: len(labels) for group, labels in group_labels.items()}
    def class_cosines(group_name):
        return [
            float(row["direction_cosine"]) for row in group_rows
            if row.get("record_type") == "cosine"
            and row.get("group") == group_name
            and row.get("scope") == "class"
            and np.isfinite(float(row.get("direction_cosine", float("nan"))))
        ]

    supported_cosines = class_cosines("source__target_supported")
    conflict_cosines = class_cosines("source__target_conflict")
    if supported_cosines and conflict_cosines and np.mean(conflict_cosines) < np.mean(supported_cosines):
        diagnosis = (
            "The class-controlled local gradients are more consistent with a supervision-signal "
            "problem in conflict samples than in supported samples; this is diagnostic, not causal."
        )
    elif unstable_rows:
        diagnosis = (
            "The normal path is locally sensitive to Student evidence that failed Target validation, "
            "so evidence-use consistency remains a plausible issue; causality is undetermined."
        )
    else:
        diagnosis = "The local evidence does not distinguish supervision error from evidence-use error."
    (output_dir / "C_findings.md").write_text(
        "# C — Normal Query evidence and local gradients\n\n"
        f"- Sensitivity rows on Target Val: {len(sensitivity_rows)}; rows tied to "
        f"Target-validation-unstable Student evidence: {len(unstable_rows)}.\n"
        f"- In-distribution perturbations accepted: "
        f"{sum(bool(row.get('accepted')) for row in perturbation_rows)}/{len(perturbation_rows)}.\n"
        f"- Gradient sample support: {support_counts}.\n\n"
        "C1 uses the normal Student Query path. Sensitivity and feasible local perturbations "
        "are local diagnostics, not causal structural interventions. C2 uses Teacher pseudo-labels "
        "but differentiates the Student path without optimizer updates. Overall and class-conditional "
        "directions are both reported; a low Source/Target cosine alone is not evidence of negative "
        "transfer. These final-checkpoint gradients do not replay historical training gradients. "
        "For variant G, the training switch detached the projected Query correction after the "
        "external projection; it did not detach the entire structure branch.\n\n"
        f"Diagnostic assessment: {diagnosis}\n",
        encoding="utf-8",
    )
    return {
        "sensitivity_rows": sensitivity_rows,
        "perturbation_rows": perturbation_rows,
        "gradient_group_rows": group_rows,
        "gradient_sample_rows": sample_rows,
    }


def required_stage_artifacts(stage):
    if stage not in STAGE_ARTIFACTS:
        raise ValueError(f"unknown checkpoint stage: {stage}")
    return STAGE_ARTIFACTS[stage]


def _timestamp():
    return datetime.now(timezone.utc).astimezone().isoformat()


def _read_stage_state(path):
    path = Path(path)
    if not path.is_file():
        return {"stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def update_stage_status(path, stage, status, error=None, metadata=None):
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown checkpoint stage: {stage}")
    state = _read_stage_state(path)
    stages = dict(state.get("stages", {}))
    previous = dict(stages.get(stage, {}))
    stages[stage] = {
        **previous, "status": status, "updated_at": _timestamp(),
        "error": error, **(metadata or {}),
    }
    state["stages"] = stages
    _write_json(path, state)
    return state


def pending_checkpoint_stages(state):
    stages = state.get("stages", {})
    pending = []
    dependency_complete = True
    for stage in STAGE_ORDER:
        complete = stages.get(stage, {}).get("status") == "complete"
        if not complete and dependency_complete:
            pending.append(stage)
        elif not complete:
            pending.append(stage)
        dependency_complete = dependency_complete and complete
    return pending


def stage_artifacts_complete(output_dir, stage):
    output_dir = Path(output_dir)
    return all((output_dir / name).is_file() for name in required_stage_artifacts(stage))


def choose_next_method(evidence):
    if int(evidence.get("stable_candidates", 0)) <= 0:
        return "insufficient_evidence"
    if evidence.get("b_incremental") and evidence.get("conflict_gradient_anomaly"):
        return "target_query_gradient_control"
    if evidence.get("uses_unstable_evidence"):
        return "query_evidence_consistency"
    return "structure_representation_alignment"


def write_checkpoint_report(output_dir, variant, task):
    output_dir = Path(output_dir)
    sections = []
    for stage in "ABC":
        path = output_dir / f"{stage}_findings.md"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed stage finding: {path}")
        text = path.read_text(encoding="utf-8")
        body = "\n".join(text.splitlines()[1:]).strip()
        sections.append(f"## {stage}\n\n{body}")
    report = output_dir / "checkpoint_report.md"
    report.write_text(
        f"# {variant} {task} — ABC audit\n\n"
        + "\n\n".join(sections)
        + "\n\n## Interpretation boundary\n\n"
        "This frozen-checkpoint audit reports associations, local sensitivities, and "
        "local gradient directions. It does not prove a causal training mechanism or "
        "replay historical gradients.\n",
        encoding="utf-8",
    )
    return report


def _read_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_pair_report(audit_root, task):
    audit_root = Path(audit_root)
    rows = []
    for variant in ("E", "G"):
        root = audit_root / f"{variant}_{task}"
        transfer = _read_csv_rows(root / "A_transferability_validation.csv")
        curves = _read_csv_rows(root / "B_coverage_error_curves.csv")
        gradients = _read_csv_rows(root / "C_gradient_groups.csv")
        stable = sum(row.get("validation_status") == "stable" for row in transfer)
        failed = sum(row.get("validation_status") == "failed" for row in transfer)
        matched = {
            row.get("method"): row for row in curves
            if row.get("curve_point") == "matched"
        }
        rows.append({
            "variant": variant, "stable": stable, "failed": failed,
            "structure_error": matched.get("structure_support", {}).get("error_rate", "N/A"),
            "confidence_error": matched.get("confidence_matched", {}).get("error_rate", "N/A"),
            "gradient_rows": len(gradients),
        })
    path = audit_root / "pair_reports" / f"{task}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# E/G pair report — {task}", "",
        "| Variant | Stable evidence | Failed validation | Structure error | Matched-confidence error | Gradient rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['stable']} | {row['failed']} | "
            f"{row['structure_error']} | {row['confidence_error']} | {row['gradient_rows']} |"
        )
    lines.extend((
        "", "Anchor IDs are checkpoint-local and are not compared directly between E and G.",
    ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def collect_global_evidence(audit_root):
    audit_root = Path(audit_root)
    stable = 0
    b_deltas = []
    uses_unstable = False
    conflict_cosines = []
    for variant in ("E", "G"):
        for task in PAIR_NAMES:
            root = audit_root / f"{variant}_{task}"
            transfer = _read_csv_rows(root / "A_transferability_validation.csv")
            stable += sum(row.get("validation_status") == "stable" for row in transfer)
            curves = _read_csv_rows(root / "B_coverage_error_curves.csv")
            matched = {
                row.get("method"): row for row in curves
                if row.get("curve_point") == "matched"
            }
            structure = _float_or_nan(matched.get("structure_support", {}).get("error_rate"))
            confidence = _float_or_nan(matched.get("confidence_matched", {}).get("error_rate"))
            if np.isfinite(structure) and np.isfinite(confidence):
                b_deltas.append(confidence - structure)
            sensitivities = _read_csv_rows(root / "C_pairwise_model_sensitivity.csv")
            uses_unstable |= any(
                row.get("target_validation_unstable", "").lower() == "true"
                and any(abs(_float_or_nan(row.get(f"sensitivity_{name}"))) > 1e-8 for name in EVENT_COMPONENTS)
                for row in sensitivities
            )
            gradients = _read_csv_rows(root / "C_gradient_groups.csv")
            for row in gradients:
                if (
                    row.get("record_type") == "cosine"
                    and row.get("group") == "target_supported__target_conflict"
                    and row.get("scope") == "class"
                ):
                    value = _float_or_nan(row.get("direction_cosine"))
                    if np.isfinite(value):
                        conflict_cosines.append(value)
    return {
        "stable_candidates": stable,
        "b_incremental": bool(b_deltas) and float(np.mean(b_deltas)) > 0.,
        "mean_matched_error_reduction": float(np.mean(b_deltas)) if b_deltas else float("nan"),
        "uses_unstable_evidence": bool(uses_unstable),
        "conflict_gradient_anomaly": bool(conflict_cosines) and float(np.mean(conflict_cosines)) < 0.,
        "mean_class_conditioned_conflict_cosine": float(np.mean(conflict_cosines)) if conflict_cosines else float("nan"),
    }


def write_global_summary(audit_root):
    audit_root = Path(audit_root)
    evidence = collect_global_evidence(audit_root)
    recommendation = choose_next_method(evidence)
    labels = {
        "structure_representation_alignment": "结构证据的表示与对应约束",
        "target_query_gradient_control": "证据条件化的 Target Query 梯度控制",
        "query_evidence_consistency": "正常 Query 的证据使用一致性约束",
        "insufficient_evidence": "证据不足，暂不增加约束",
    }
    supporting = [f"A 事后稳定候选数={evidence['stable_candidates']}"]
    opposing = []
    (supporting if evidence["b_incremental"] else opposing).append(
        f"B 匹配覆盖率平均错误率改善={evidence['mean_matched_error_reduction']}"
    )
    (supporting if evidence["uses_unstable_evidence"] else opposing).append(
        f"C 使用验证失败证据={evidence['uses_unstable_evidence']}"
    )
    (supporting if evidence["conflict_gradient_anomaly"] else opposing).append(
        f"C 类别条件冲突梯度异常={evidence['conflict_gradient_anomaly']}"
    )
    path = audit_root / "ABC_summary.md"
    path.write_text(
        "# ABC summary\n\n"
        "## 1. A：跨域结构证据\n\n"
        f"跨四个 checkpoint 的事后稳定候选数：{evidence['stable_candidates']}。\n\n"
        "## 2. B：Target 伪标签\n\n"
        f"相同覆盖率下的平均错误率下降：{evidence['mean_matched_error_reduction']}；"
        f"是否提供 confidence 之外的信息：{evidence['b_incremental']}。\n\n"
        "## 3. C：监督与正常 Query\n\n"
        f"是否使用 Target 验证失败的 Student 证据：{evidence['uses_unstable_evidence']}；"
        f"类别条件冲突梯度异常：{evidence['conflict_gradient_anomaly']}。\n"
        "低余弦本身不被解释为负迁移，最终 checkpoint 梯度不代表历史训练轨迹。\n\n"
        "## 4. E/G 差异\n\n"
        "详见 `pair_reports/FR1_FR2.md` 与 `pair_reports/FR2_DK1.md`；Anchor ID 不跨 checkpoint 比较。\n\n"
        "## 5. 下一步方法选择\n\n"
        f"**{labels[recommendation]}** (`{recommendation}`)\n\n"
        "### 支持证据\n\n- " + "\n- ".join(supporting) + "\n\n"
        "### 反对证据\n\n- " + "\n- ".join(opposing) + "\n\n"
        "### 无法确定\n\n"
        "- Target Val 是已知重点类别对的针对性诊断，不是独立验证。\n"
        "- 冻结模型的相关性、局部扰动与最终 checkpoint 梯度不能证明因果机制，"
        "也不能重放历史训练梯度。\n",
        encoding="utf-8",
    )
    return path, evidence, recommendation


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _split_manifest(datasets, config):
    result = {}
    for split, dataset in datasets.items():
        parcels = dataset.get_parcel_indices().astype(np.int64)
        domain = config.source if split.startswith("source") else config.target
        rows = [{"domain": domain, "parcel_index": int(parcel)} for parcel in parcels]
        if split != "target_train":
            labels = dataset.get_labels().astype(np.int64)
            for row, label in zip(rows, labels):
                row["label"] = int(label)
        result[split] = rows
    return result


def _classes_from_checkpoint(path):
    packet = torch.load(path, map_location="cpu", weights_only=False)
    raw = packet["config"]
    config = raw if isinstance(raw, dict) else vars(raw)
    return list(config["classes"])


def run_checkpoint_stage(args):
    output_dir = Path(args.audit_root) / f"{args.variant}_{args.task}"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "stage_status.json"
    state = _read_stage_state(state_path)
    if (
        state.get("stages", {}).get(args.stage, {}).get("status") == "complete"
        and stage_artifacts_complete(output_dir, args.stage)
    ):
        return
    update_stage_status(state_path, args.stage, "running", metadata={"started_at": _timestamp()})
    started = time.monotonic()
    measure_cuda_memory = str(args.device).startswith("cuda") and torch.cuda.is_available()
    if measure_cuda_memory:
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    try:
        if args.stage == "extract":
            frozen = load_frozen_models(
                args.checkpoint, args.task, args.variant, args.device,
            )
            source = validate_source(args.source_checkpoint, args.task, "cpu")
            if list(frozen.config.classes) != source["classes"]:
                raise ValueError("Source/UDA class mapping mismatch")
            config, protocol, splits, datasets = build_abc_datasets(
                frozen.config, args.data_root, args.seed,
            )
            manifest = extract_checkpoint_features(
                frozen, datasets, {"source": config.source, "target": config.target},
                output_dir, device=args.device, batch_size=args.batch_size,
                pixel_budget=args.pixel_budget,
                representative_count=args.representative_count,
            )
            _write_json(output_dir / "checkpoint_manifest.json", {
                "git_commit": _git_commit(), "task": args.task,
                "variant": args.variant, "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "source_checkpoint": source,
                "classes": list(config.classes), "seed": args.seed,
                "protocol": protocol, "split_manifest": _split_manifest(datasets, config),
                "target_test_used": False,
                "target_train_labels_cached": False,
                "applied_shifts": manifest.applied_shifts,
                "args": vars(args),
            })
        elif args.stage == "A":
            run_a_stage(
                output_dir, output_dir, args.task,
                _classes_from_checkpoint(args.checkpoint), args.seed,
            )
        elif args.stage == "B":
            run_b_stage(
                output_dir, output_dir, _classes_from_checkpoint(args.checkpoint),
                confidence_threshold=args.confidence_threshold,
            )
        elif args.stage == "C":
            frozen = load_frozen_models(
                args.checkpoint, args.task, args.variant, args.device,
            )
            checkpoint_manifest = json.loads(
                (output_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
            )
            datasets = replay_abc_datasets(
                frozen.config, args.data_root, args.seed,
                checkpoint_manifest["split_manifest"],
            )
            run_c_stage(
                frozen, datasets, output_dir, output_dir, args.task,
                list(frozen.config.classes), args.seed, args.gradient_samples_per_class,
            )
        else:
            write_checkpoint_report(output_dir, args.variant, args.task)
        if not stage_artifacts_complete(output_dir, args.stage):
            raise RuntimeError(f"{args.stage} did not produce its artifact contract")
        completion_metadata = {"elapsed_seconds": time.monotonic() - started}
        if measure_cuda_memory:
            completion_metadata["peak_cuda_memory_bytes"] = int(
                torch.cuda.max_memory_allocated(torch.device(args.device))
            )
        update_stage_status(
            state_path, args.stage, "complete",
            metadata=completion_metadata,
        )
    except Exception as error:
        failure_metadata = {
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started,
        }
        if measure_cuda_memory:
            failure_metadata["peak_cuda_memory_bytes"] = int(
                torch.cuda.max_memory_allocated(torch.device(args.device))
            )
        update_stage_status(
            state_path, args.stage, "failed", error=str(error),
            metadata=failure_metadata,
        )
        raise


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("checkpoint", "pair-report", "summary"), required=True)
    parser.add_argument("--stage", choices=STAGE_ORDER)
    parser.add_argument("--task", choices=tuple(PAIR_NAMES))
    parser.add_argument("--variant", choices=("E", "G"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--representative-count", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=.9)
    parser.add_argument("--gradient-samples-per-class", type=int, default=4)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "checkpoint":
        missing = [name for name in ("stage", "task", "variant", "checkpoint", "source_checkpoint")
                   if getattr(args, name) is None]
        if missing:
            raise ValueError(f"checkpoint mode missing arguments: {missing}")
        run_checkpoint_stage(args)
    elif args.mode == "pair-report":
        if args.task is None:
            raise ValueError("pair-report mode requires --task")
        write_pair_report(args.audit_root, args.task)
    else:
        for task in PAIR_NAMES:
            if not (args.audit_root / "pair_reports" / f"{task}.md").is_file():
                write_pair_report(args.audit_root, task)
        write_global_summary(args.audit_root)


def decompose_pairwise_anchor_margin(estimator, features, y_id, c_id):
    features = np.asarray(features, dtype=np.float64)
    scaler = estimator.named_steps.get("scaler", estimator.named_steps.get("scale"))
    classifier = estimator.named_steps["classifier"]
    classes = [int(value) for value in classifier.classes_]
    if int(y_id) not in classes or int(c_id) not in classes:
        raise ValueError("pair classes are absent from the fitted probe")
    y_row, c_row = classes.index(int(y_id)), classes.index(int(c_id))
    if classifier.coef_.shape[0] != len(classes):
        raise ValueError("pairwise decomposition requires a multiclass linear probe")
    standardized = scaler.transform(features)
    delta_weight = classifier.coef_[y_row] - classifier.coef_[c_row]
    bias = float(classifier.intercept_[y_row] - classifier.intercept_[c_row])
    terms = standardized * delta_weight
    scores = estimator.decision_function(features)
    margin = scores[:, y_row] - scores[:, c_row]
    contributions = response_to_anchor_events(terms)
    reconstructed = contributions.sum((1, 2)) + bias
    error = np.abs(reconstructed - margin)
    epsilon = max(
        np.finfo(array.dtype).eps for array in (
            np.asarray(standardized), np.asarray(classifier.coef_), np.asarray(scores),
        ) if np.issubdtype(array.dtype, np.floating)
    )
    operations = terms.shape[1] + 8
    gamma = operations * epsilon / max(1. - operations * epsilon, epsilon)
    bound = 4. * gamma * (
        np.abs(terms).sum(1) + abs(bias) + np.abs(margin)
    ) + np.finfo(float).tiny
    if np.any(error > bound):
        raise RuntimeError("pairwise Anchor contributions do not reconstruct probe margin")
    return {
        "margin": margin, "bias": np.full(len(features), bias),
        "anchor_component_contribution": contributions,
        "reconstruction_error": error, "reconstruction_bound": bound,
    }


def _write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=path.stem + ".", suffix=".json", dir=path.parent,
        delete=False, mode="w", encoding="utf-8",
    )
    try:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
        handle.flush(); os.fsync(handle.fileno()); handle.close()
        os.replace(handle.name, path)
    finally:
        if not handle.closed:
            handle.close()
        Path(handle.name).unlink(missing_ok=True)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _configured_pairs(task, classes):
    lookup = {canonical_class(name): index for index, name in enumerate(classes)}
    rows = []
    for left, right in PAIR_NAMES[task]:
        rows.append({
            "y_key": left, "c_key": right,
            "y_id": lookup.get(left), "c_id": lookup.get(right),
            "y_name": classes[lookup[left]] if left in lookup else left,
            "c_name": classes[lookup[right]] if right in lookup else right,
        })
    return rows


def _anchor_semantic_rows(role, split, data):
    rows = []
    for anchor in range(data["events"].shape[1]):
        similarities = data["similarity"][:, :, anchor]
        pooling = data["pooling_weight"][:, :, anchor]
        row = {
            "record_type": "anchor_summary", "role": role,
            "split": split, "anchor_id": anchor,
            "sample_support": len(data["events"]),
            "window_support": int(np.prod(data["similarity"][:, :, anchor].shape)),
            "similarity_mean": float(similarities.mean()),
            "similarity_std": float(similarities.std()),
            "similarity_q90": float(np.quantile(similarities, .9)),
            "similarity_q95": float(np.quantile(similarities, .95)),
            "pooling_weight_mean": float(pooling.mean()),
            "above_uniform_pooling_fraction": float(
                (pooling >= 1. / max(pooling.shape[1], 1)).mean()
            ),
            "anchor_probability_mean": float(data["anchor_probability"][:, :, anchor].mean()),
        }
        for index, component in enumerate(EVENT_COMPONENTS):
            values = data["events"][:, anchor, index]
            row[f"{component}_mean"] = float(values.mean())
            row[f"{component}_std"] = float(values.std())
        rows.append(row)
    return rows


def _representative_semantic_rows(role, split, data):
    rows = []
    centers = np.asarray(data["window_centers_days"])
    for index, anchor in enumerate(data["representative_anchor"]):
        window = int(data["representative_window_index"][index])
        rows.append({
            "record_type": "representative_window",
            "role": role,
            "split": split,
            "anchor_id": int(anchor),
            "representative_kind": str(data["representative_kind"][index]),
            "parcel_index": int(data["representative_parcel_index"][index]),
            "window_index": window,
            "window_center_days": float(centers[window]),
            "similarity": float(data["representative_score"][index]),
        })
    return rows


def _plot_representatives(cache_dir, candidates, path):
    import matplotlib.pyplot as plt

    selected = {
        (row["role"], int(row["anchor_id"])) for row in candidates
    }
    records = []
    for role in ("student", "teacher"):
        role_anchors = sorted(anchor for item_role, anchor in selected if item_role == role)
        wanted = set(role_anchors[:1] or [0])
        for split in ("source_train", "target_train"):
            data = load_feature_split(cache_dir, role, split)
            for index in range(len(data["representative_anchor"])):
                anchor = int(data["representative_anchor"][index])
                if anchor not in wanted:
                    continue
                records.append({
                    "role": role, "split": split, "anchor": anchor,
                    "kind": str(data["representative_kind"][index]),
                    "parcel": int(data["representative_parcel_index"][index]),
                    "window": int(data["representative_window_index"][index]),
                    "curve": data["representative_curves"][index],
                    "standardized": data["representative_standardized"][index],
                    "difference": data["representative_differences"][index],
                    "mean": data["representative_mean"][index],
                    "std": data["representative_std"][index],
                })
    records = records[:16]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        fig, axis = plt.subplots(figsize=(8, 2))
        axis.text(.5, .5, "No representative windows", ha="center", va="center")
        axis.axis("off")
    else:
        fig, axes = plt.subplots(len(records), 4, figsize=(15, 3 * len(records)), squeeze=False)
        for row_axes, record in zip(axes, records):
            for axis, key, title in zip(
                row_axes[:3], ("curve", "standardized", "difference"),
                ("full latent PSE window", "channel-standardized", "first difference"),
            ):
                image = axis.imshow(record[key].T, aspect="auto", interpolation="nearest")
                axis.set_title(title)
                axis.set_xlabel("window step"); axis.set_ylabel("latent channel")
                fig.colorbar(image, ax=axis, fraction=.035)
            row_axes[3].plot(record["mean"], label="Mean")
            row_axes[3].plot(record["std"], label="Std")
            row_axes[3].set_title("per-channel Mean/Std")
            row_axes[3].legend(fontsize=7)
            row_axes[0].set_ylabel(
                f"{record['role']} {record['split']} A{record['anchor']} "
                f"{record['kind']} p{record['parcel']} w{record['window']}\nlatent channel"
            )
        fig.suptitle("Representative latent PSE windows (not phenological stages)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def run_a_stage(
    cache_dir, output_dir, task, classes, seed=1, thresholds=None,
):
    from analysis.structure_representation_chain_audit import make_linear_probe

    thresholds = {
        "min_support": 20, "min_effect": .5,
        "min_direction_cosine": .25, "min_target_range": .5,
        **(thresholds or {}),
    }
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    semantic_rows, discrimination_rows, transfer_rows = [], [], []
    all_candidates = []
    pair_specs = _configured_pairs(task, classes)
    role_payload = {}
    for role in ("student", "teacher"):
        data = {
            split: load_feature_split(cache_dir, role, split)
            for split in ("source_train", "source_val", "target_train", "target_val")
        }
        for split, values in data.items():
            semantic_rows.extend(_anchor_semantic_rows(role, split, values))
            semantic_rows.extend(_representative_semantic_rows(role, split, values))
        probe = make_linear_probe().fit(
            data["source_train"]["response"], data["source_train"]["label"],
        )
        role_candidates = []
        for pair in pair_specs:
            pair_key = f"{pair['y_key']}__{pair['c_key']}"
            if pair["y_id"] is None or pair["c_id"] is None:
                for split in ("source_train", "source_val", "target_val"):
                    labels = data[split]["label"]
                    support_y = (
                        0 if pair["y_id"] is None
                        else int((labels == pair["y_id"]).sum())
                    )
                    support_c = (
                        0 if pair["c_id"] is None
                        else int((labels == pair["c_id"]).sum())
                    )
                    for anchor in range(16):
                        discrimination_rows.append({
                            "record_type": "event_effect", "role": role,
                            "split": split, "pair": pair_key,
                            "anchor_id": anchor, "status": "N/A",
                            "support_y": support_y, "support_c": support_c,
                            "effect_norm": float("nan"),
                        })
                transfer_rows.append({
                    "role": role, "pair": pair_key, "anchor_id": "N/A",
                    "candidate": False, "validation_status": "N/A_missing_class",
                })
                continue
            stats = {}
            for split in ("source_train", "source_val", "target_val"):
                stats[split] = pairwise_event_statistics(
                    data[split]["events"], data[split]["label"],
                    pair["y_id"], pair["c_id"], split, role, seed,
                )
                discrimination_rows.extend([
                    {"record_type": "event_effect", "pair": pair_key, **row}
                    for row in stats[split]
                ])
            candidates = select_source_only_candidates(
                stats["source_train"], stats["source_val"],
                data["source_train"]["events"], data["target_train"]["events"],
                role=role, pair=(pair["y_id"], pair["c_id"]), seed=seed,
                **thresholds,
            )
            for candidate in candidates:
                candidate.update({
                    "pair": pair_key, "y_name": pair["y_name"],
                    "c_name": pair["c_name"],
                })
            role_candidates.extend(candidates)
            candidate_anchors = {int(row["anchor_id"]): row for row in candidates}
            target_by_anchor = {row["anchor_id"]: row for row in stats["target_val"]}
            train_by_anchor = {row["anchor_id"]: row for row in stats["source_train"]}
            for anchor, candidate in candidate_anchors.items():
                target = target_by_anchor[anchor]
                cosine = (
                    _cosine(_row_effect(train_by_anchor[anchor]), _row_effect(target))
                    if target["status"] == "ok" else float("nan")
                )
                transfer_rows.append({
                    "role": role, "pair": pair_key, "anchor_id": anchor,
                    "candidate": True,
                    "target_support_y": target["support_y"],
                    "target_support_c": target["support_c"],
                    "target_direction_cosine": cosine,
                    "validation_status": (
                        "stable" if np.isfinite(cosine) and cosine >= thresholds["min_direction_cosine"]
                        else "failed"
                    ),
                })
            for split in ("source_val", "target_val"):
                selected = np.isin(
                    data[split]["label"], [pair["y_id"], pair["c_id"]],
                )
                if not selected.any():
                    continue
                decomposition = decompose_pairwise_anchor_margin(
                    probe, data[split]["response"][selected],
                    pair["y_id"], pair["c_id"],
                )
                labels = data[split]["label"][selected]
                for class_id, class_name in (
                    (pair["y_id"], pair["y_name"]),
                    (pair["c_id"], pair["c_name"]),
                ):
                    class_selected = labels == class_id
                    for anchor in range(16):
                        row = {
                            "record_type": "probe_margin", "role": role,
                            "split": split, "pair": pair_key,
                            "anchor_id": anchor, "class_name": class_name,
                            "support_y": int(class_selected.sum()),
                            "support_c": "", "status": "ok",
                            "pair_margin_mean": float(decomposition["margin"][class_selected].mean()),
                            "anchor_margin_contribution": float(
                                decomposition["anchor_component_contribution"][class_selected, anchor].sum(1).mean()
                            ),
                            "max_reconstruction_error": float(decomposition["reconstruction_error"].max()),
                            "max_reconstruction_bound": float(decomposition["reconstruction_bound"].max()),
                        }
                        for component_index, component in enumerate(EVENT_COMPONENTS):
                            row[f"contribution_{component}"] = float(
                                decomposition["anchor_component_contribution"][
                                    class_selected, anchor, component_index
                                ].mean()
                            )
                        discrimination_rows.append(row)
        role_payload[role] = role_candidates
        all_candidates.extend(role_candidates)

    _write_csv(output_dir / "A_anchor_event_semantics.csv", semantic_rows)
    _write_csv(output_dir / "A_pairwise_discrimination.csv", discrimination_rows)
    _write_csv(output_dir / "A_transferability_validation.csv", transfer_rows)
    candidate_packet = {
        "selection_uses_target_truth": False,
        "spaces_are_checkpoint_role_local": True,
        "thresholds": thresholds,
        "roles": role_payload,
    }
    _write_json(output_dir / "A_anchor_candidates.json", candidate_packet)
    _plot_representatives(
        cache_dir, all_candidates, output_dir / "A_representative_windows.png",
    )
    stable = [row for row in transfer_rows if row.get("validation_status") == "stable"]
    failed = [row for row in transfer_rows if row.get("validation_status") == "failed"]
    missing = [row for row in transfer_rows if str(row.get("validation_status", "")).startswith("N/A")]
    stable_ids = [f"{row['role']}:{row['pair']}:A{row['anchor_id']}" for row in stable]
    failed_ids = [f"{row['role']}:{row['pair']}:A{row['anchor_id']}" for row in failed]
    missing_ids = sorted({f"{row['role']}:{row['pair']}" for row in missing})
    (output_dir / "A_findings.md").write_text(
        "# A — Anchor event semantics\n\n"
        "Student and EMA Teacher Anchor identities were analyzed in separate coordinate spaces.\n\n"
        f"- Source-only candidates: {len(all_candidates)}\n"
        f"- Post-selection Target Val stable candidates: {len(stable)}\n"
        f"- Shape-supported but class-direction validation failures: {len(failed)}\n"
        f"- Fixed pairs unavailable because of missing classes: {len(missing)}\n\n"
        f"- Stable evidence IDs: {stable_ids or ['none']}\n"
        f"- Shape-only / direction-failed IDs: {failed_ids or ['none']}\n"
        f"- Missing or insufficient pair IDs: {missing_ids or ['none']}\n\n"
        "Target Val truth was used only after candidate selection. These are targeted diagnostics, "
        "not an independent validation set. PSE window plots show latent morphology and are not "
        "labeled as phenological stages.\n",
        encoding="utf-8",
    )
    return candidate_packet


if __name__ == "__main__":
    main()
