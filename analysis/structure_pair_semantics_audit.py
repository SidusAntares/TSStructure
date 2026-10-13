#!/usr/bin/env python3
"""Frozen all-pair SC/K1/K2/SCP and structure-semantics audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TASKS = {
    "FR1_FR2": ("FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
    "FR2_DK1": ("FR2", "france/31TCJ/2017", "DK1", "denmark/32VNH/2017"),
    "AT1_DK1": ("AT1", "austria/33UVP/2017", "DK1", "denmark/32VNH/2017"),
    "DK1_AT1": ("DK1", "denmark/32VNH/2017", "AT1", "austria/33UVP/2017"),
}
EVENT_COMPONENTS = (
    "strength", "concentration",
    "phase_k1_cos", "phase_k1_sin", "phase_k2_cos", "phase_k2_sin",
)
FEATURE_GROUPS = {
    "SC": (0, 1), "SC_K1": (0, 1, 2, 3),
    "SC_K2": (0, 1, 4, 5), "SCP": (0, 1, 2, 3, 4, 5),
}
GROUPS = tuple(FEATURE_GROUPS)
ALLOWED_SPLITS = ("source_train", "source_val", "target_train", "target_val")
REQUIRED_OUTPUTS = (
    "T1_component_comparison.csv", "T1_pairwise_phase_increment.csv",
    "T2_pair_evidence_graph.csv", "T2_sample_pair_scores.csv",
    "T2_teacher_rank_groups.csv", "T2_incremental_value.csv",
    "T2_eligibility_ablation.csv", "T2_semantic_cases.md",
    "T2_semantic_cases.png", "checkpoint_manifest.json",
)


@dataclass(frozen=True)
class CheckpointScope:
    variant: str
    task: str
    checkpoint_sha256: str


def anchor_ids_comparable(left, right):
    return left == right


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def enumerate_class_pairs(labels, classes, min_support=10):
    labels = np.asarray(labels, dtype=np.int64)
    rows = []
    for y_id, c_id in combinations(range(len(classes)), 2):
        support_y, support_c = int((labels == y_id).sum()), int((labels == c_id).sum())
        rows.append({
            "y_id": y_id, "c_id": c_id, "y_name": classes[y_id], "c_name": classes[c_id],
            "support_y": support_y, "support_c": support_c,
            "status": "eligible" if min(support_y, support_c) >= int(min_support) else "insufficient_support",
        })
    return rows


def _cosine(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else float("nan")


def _candidate_projection(candidate, values, group):
    indices = np.asarray(FEATURE_GROUPS[group], dtype=int)
    effect = np.asarray(candidate["effect"], float)[indices]
    midpoint = .5 * (
        np.asarray(candidate["mean_y"], float)[indices]
        + np.asarray(candidate["mean_c"], float)[indices]
    )
    scale = np.maximum(np.asarray(candidate["pooled_std"], float)[indices], 1e-6)
    unit = effect / max(np.linalg.norm(effect), 1e-12)
    return ((np.asarray(values, float)[..., indices] - midpoint) / scale) @ unit


def signed_pair_evidence(candidate, events, y_id, c_id, group="SCP"):
    original = (int(candidate["pair_y"]), int(candidate["pair_c"]))
    requested = (int(y_id), int(c_id))
    if requested not in (original, original[::-1]):
        raise ValueError("requested classes do not match candidate pair")
    anchor = int(candidate["anchor_id"])
    score = _candidate_projection(candidate, np.asarray(events)[:, anchor], group)
    if requested == original[::-1]:
        score = -score
    return float(score[0]) if len(score) == 1 else score


def select_group_source_only(rows, uncertainty=.01):
    valid = [row for row in rows if np.isfinite(float(row["source_val_quality"]))]
    if not valid:
        return "SC", True
    ordered = sorted(valid, key=lambda row: (-float(row["source_val_quality"]), GROUPS.index(row["group"])))
    uncertain = len(ordered) > 1 and (
        float(ordered[0]["source_val_quality"]) - float(ordered[1]["source_val_quality"])
    ) <= float(uncertainty)
    return ordered[0]["group"], bool(uncertain)


def assert_unlabeled_target_train(arrays):
    if "label" in arrays:
        raise ValueError("Target Train must remain unlabeled")
    return True


def _softmax(logits):
    logits = np.asarray(logits, float)
    shifted = logits - logits.max(1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(1, keepdims=True)


def teacher_thresholds(source_val_logits):
    probability = _softmax(source_val_logits)
    ordered = np.sort(probability, axis=1)[:, ::-1]
    confidence = ordered[:, 0]
    margin = ordered[:, 0] - ordered[:, 1]
    entropy = -(probability * np.log(probability.clip(1e-12))).sum(1)
    return tuple(np.quantile(value, (.33, .67)).tolist() for value in (confidence, margin, entropy))


def teacher_rank_rows(logits, confidence_bounds, margin_bounds, entropy_bounds,
                      parcel_index, domain, labels=None):
    logits = np.asarray(logits, float); probability = _softmax(logits)
    order = np.argsort(-probability, axis=1); ranked = np.take_along_axis(probability, order, axis=1)
    entropy = -(probability * np.log(probability.clip(1e-12))).sum(1)
    rows = []
    for index in range(len(logits)):
        margin12 = float(ranked[index, 0] - ranked[index, 1])
        margin23 = float(ranked[index, 1] - ranked[index, 2]) if ranked.shape[1] > 2 else float("inf")
        if ranked[index, 0] <= confidence_bounds[0] or entropy[index] >= entropy_bounds[1]:
            group = "overall_uncertain"
        elif margin12 <= margin_bounds[0] and margin23 <= margin_bounds[0]:
            group = "top123_close"
        elif margin12 <= margin_bounds[0]:
            group = "top12_close"
        elif margin12 >= margin_bounds[1]:
            group = "top1_clear"
        else:
            group = "intermediate"
        row = {
            "domain": str(domain[index]), "parcel_index": int(parcel_index[index]),
            "teacher_top1": int(order[index, 0]), "teacher_top2": int(order[index, 1]),
            "teacher_top3": int(order[index, 2]) if order.shape[1] > 2 else -1,
            "confidence": float(ranked[index, 0]), "top1_top2_margin": margin12,
            "entropy": float(entropy[index]), "competition_group": group,
        }
        if labels is not None:
            truth = int(np.asarray(labels)[index]); rank = int(np.flatnonzero(order[index] == truth)[0]) + 1
            row.update(true_label=truth, true_rank=rank,
                       true_rank_group=f"true_top{rank}" if rank <= 3 else "true_below_top3")
        rows.append(row)
    return rows


def resolve_pair_graph(class_ids, pair_scores):
    class_ids = tuple(dict.fromkeys(int(value) for value in class_ids))
    expected = {tuple(sorted(pair)) for pair in combinations(class_ids, 2)}
    available = {tuple(sorted((int(a), int(b)))) for a, b in pair_scores}
    if expected - available:
        return {"status": "abstain_missing"}
    wins = {value: set() for value in class_ids}
    for left, right in expected:
        score = float(pair_scores[(left, right)])
        if not np.isfinite(score) or score == 0:
            return {"status": "abstain_missing"}
        winner, loser = (left, right) if score > 0 else (right, left)
        wins[winner].add(loser)
    champions = [value for value in class_ids if len(wins[value]) == len(class_ids) - 1]
    return {"status": "winner", "class_id": champions[0]} if len(champions) == 1 else {"status": "abstain_cycle"}


def validate_cache_identity(
    cache_dir, checkpoint_sha, classes, require_feature_files=True,
    expected_task=None, expected_variant=None,
):
    cache_dir = Path(cache_dir)
    feature = json.loads((cache_dir / "feature_cache_manifest.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((cache_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if checkpoint.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("cache checkpoint SHA mismatch")
    if list(checkpoint.get("classes", [])) != list(classes):
        raise ValueError("cache class mapping mismatch")
    if expected_task is not None and checkpoint.get("task") not in (None, expected_task):
        raise ValueError("cache task mismatch")
    if expected_variant is not None and checkpoint.get("variant") not in (None, expected_variant):
        raise ValueError("cache variant mismatch")
    if feature.get("target_test_used") or feature.get("target_train_labels_cached") or checkpoint.get("target_test_used"):
        raise ValueError("cache violates frozen split protocol")
    applied = feature.get("applied_shifts", {})
    student_shifts = list(applied.get("student", {}).values())
    teacher_shifts = list(applied.get("teacher", {}).values())
    if student_shifts and any(int(value) != 0 for value in student_shifts):
        raise ValueError("Student cache shift must remain zero")
    if teacher_shifts and len({int(value) for value in teacher_shifts}) != 1:
        raise ValueError("Teacher cache shift is inconsistent across splits")
    if require_feature_files:
        for role in ("student", "teacher"):
            for split in ALLOWED_SPLITS:
                path = cache_dir / "features" / f"{role}_{split}.npz"
                if not path.is_file():
                    raise FileNotFoundError(f"MISSING_CACHE_FEATURE|path={path}")
    return {"valid": True, "checkpoint_sha256": checkpoint_sha, "classes": list(classes)}


def _pair_mask(labels, y_id, c_id):
    labels = np.asarray(labels, dtype=np.int64)
    return (labels == int(y_id)) | (labels == int(c_id))


def _event_stats(events, labels, y_id, c_id):
    from analysis.structure_semantic_abc import pairwise_event_statistics
    return pairwise_event_statistics(events, labels, y_id, c_id, "source", "teacher", 1)


def build_pair_candidates(cache, classes, min_support=10):
    from analysis.structure_semantic_abc import fit_joint_support, joint_support_mask
    train, val, target = cache["source_train"], cache["source_val"], cache["target_train"]
    pairs = enumerate_class_pairs(train["label"], classes, min_support)
    rows = []
    for pair in pairs:
        y_id, c_id = pair["y_id"], pair["c_id"]
        train_rows = _event_stats(train["events"], train["label"], y_id, c_id)
        val_rows = _event_stats(val["events"], val["label"], y_id, c_id)
        for left, right in zip(train_rows, val_rows):
            anchor = int(left["anchor_id"])
            row = {**pair, "pair_y": y_id, "pair_c": c_id, "anchor_id": anchor,
                   "pair": f"{pair['y_name']}__{pair['c_name']}"}
            if pair["status"] != "eligible" or left["status"] != "ok" or right["status"] != "ok":
                row.update(status="insufficient_support", source_discriminable=False,
                           target_applicable=False)
                rows.append(row); continue
            effect = np.asarray([left[f"effect_{name}"] for name in EVENT_COMPONENTS], float)
            val_effect = np.asarray([right[f"effect_{name}"] for name in EVENT_COMPONENTS], float)
            sc = np.asarray(FEATURE_GROUPS["SC"])
            direction = _cosine(effect[sc], val_effect[sc])
            source_events = train["events"][:, anchor]
            support = fit_joint_support(source_events, seed=1 + anchor)
            in_range = joint_support_mask(support, target["events"][:, anchor])
            row.update({
                "status": "ok", "effect": effect.tolist(),
                "mean_y": [left[f"mean_y_{name}"] for name in EVENT_COMPONENTS],
                "mean_c": [left[f"mean_c_{name}"] for name in EVENT_COMPONENTS],
                "pooled_std": [left[f"pooled_std_{name}"] for name in EVENT_COMPONENTS],
                "source_direction_cosine_sc": direction,
                "source_effect_norm_sc": float(np.linalg.norm(effect[sc])),
                "source_discriminable": False,
                "target_train_range_rate": float(in_range.mean()),
                "target_applicable": bool(in_range.mean() >= .5),
                "joint_support": {
                    "center": support.center.tolist(), "precision": support.precision.tolist(),
                    "threshold": support.threshold, "marginal_low": support.marginal_low.tolist(),
                    "marginal_high": support.marginal_high.tolist(),
                },
            })
            pair_mask = _pair_mask(val["label"], y_id, c_id)
            projected = _candidate_projection(row, val["events"][pair_mask, anchor], "SCP")
            pair_labels = val["label"][pair_mask]
            prediction = np.where(projected >= 0, y_id, c_id)
            row["source_val_balanced_accuracy"] = float(balanced_accuracy_score(pair_labels, prediction))
            row["source_val_pair_auroc"] = float(roc_auc_score(pair_labels == y_id, projected))
            row["source_discriminable"] = bool(
                direction >= .25
                and np.linalg.norm(effect[sc]) >= .5
                and row["source_val_pair_auroc"] >= .6
            )
            rows.append(row)
    return rows


def _candidate_in_range(candidate, value):
    from analysis.structure_semantic_abc import JointSupport, joint_support_mask
    raw = candidate.get("joint_support")
    if raw is None:
        return False
    support = JointSupport(
        center=np.asarray(raw["center"]), precision=np.asarray(raw["precision"]),
        threshold=float(raw["threshold"]), marginal_low=np.asarray(raw["marginal_low"]),
        marginal_high=np.asarray(raw["marginal_high"]),
    )
    return bool(joint_support_mask(support, np.asarray(value)[None])[0])


def pair_score(candidates, events, y_id, c_id, group="SCP", strategy="source_target"):
    scores, weights, in_ranges = [], [], []
    for candidate in candidates:
        if (candidate.get("pair_y"), candidate.get("pair_c")) != tuple(sorted((int(y_id), int(c_id)))):
            continue
        if candidate.get("status") != "ok":
            continue
        anchor = int(candidate["anchor_id"])
        in_range = _candidate_in_range(candidate, np.asarray(events)[anchor])
        if strategy in ("source", "source_target") and not candidate.get("source_discriminable", False):
            continue
        if strategy in ("target", "source_target") and not in_range:
            continue
        scores.append(float(signed_pair_evidence(candidate, np.asarray(events)[None], y_id, c_id, group)))
        weights.append(max(float(candidate.get("source_direction_cosine_sc", 0)), 0.) + 1e-6)
        in_ranges.append(in_range)
    if not scores:
        return {"available": False, "score": np.nan, "evidence_count": 0, "range_fraction": np.nan}
    return {"available": True, "score": float(np.average(scores, weights=weights)),
            "evidence_count": len(scores), "range_fraction": float(np.mean(in_ranges))}


def _binary_quality(model, features, labels, y_id):
    labels = np.asarray(labels); mask = np.isfinite(features).all(1)
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return {
            "balanced_accuracy": np.nan, "pair_auroc": np.nan,
            "correct": np.zeros(len(labels), bool), "mask": mask,
            "labels": np.asarray(labels)[mask], "prediction": np.empty(0, dtype=int),
            "probability_y": np.empty(0, dtype=float),
        }
    probability = model.predict_proba(features[mask])
    classes = model[-1].classes_
    y_column = int(np.flatnonzero(classes == y_id)[0])
    prediction = model.predict(features[mask])
    correct = np.zeros(len(labels), bool); correct[mask] = prediction == labels[mask]
    return {"balanced_accuracy": float(balanced_accuracy_score(labels[mask], prediction)),
            "pair_auroc": float(roc_auc_score(labels[mask] == y_id, probability[:, y_column])),
            "correct": correct, "mask": mask, "labels": labels[mask],
            "prediction": prediction, "probability_y": probability[:, y_column]}


def _pair_features(arrays, anchors, components):
    if not anchors:
        return np.empty((len(arrays["events"]), 0))
    return arrays["events"][:, anchors][:, :, components].reshape(len(arrays["events"]), -1)


def _bootstrap_delta(left, right, repeats=400, seed=1):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if len(left) < 2 or len(left) != len(right):
        return np.nan, np.nan
    rng = np.random.default_rng(seed); delta = []
    for _ in range(int(repeats)):
        index = rng.integers(0, len(left), len(left))
        delta.append(float((right[index] - left[index]).mean()))
    return tuple(np.quantile(delta, (.025, .975)).tolist())


def _bootstrap_binary_metric(labels, values, metric, repeats=400, seed=1):
    labels, values = np.asarray(labels), np.asarray(values)
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return np.nan, np.nan
    indices = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    rng = np.random.default_rng(seed); samples = []
    for _ in range(int(repeats)):
        draw = np.concatenate([part[rng.integers(0, len(part), len(part))] for part in indices])
        samples.append(float(metric(labels[draw], values[draw])))
    return tuple(np.quantile(samples, (.025, .975)).tolist())


def _sample_digest(data, mask):
    values = sorted(
        f"{domain}:{int(parcel)}"
        for domain, parcel in zip(np.asarray(data["domain"])[mask], np.asarray(data["parcel_index"])[mask])
    )
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def run_t1(cache, candidates, classes, task, variant, output_dir):
    rows, increments = [], []
    pairs = enumerate_class_pairs(cache["source_train"]["label"], classes, 10)
    for pair in pairs:
        y_id, c_id = pair["y_id"], pair["c_id"]
        selected = sorted({int(row["anchor_id"]) for row in candidates
                           if row.get("pair_y") == y_id and row.get("pair_c") == c_id
                           and row.get("source_discriminable")})
        group_results = {}
        for group, components in FEATURE_GROUPS.items():
            record = {"task": task, "variant": variant, "scope": "class_pair", **pair,
                      "pair": pair["y_name"] + "__" + pair["c_name"], "group": group,
                      "anchor_count": len(selected), "selection_uses_target_val": False}
            if pair["status"] != "eligible" or not selected:
                record.update(status="insufficient_support", source_val_quality=np.nan,
                              target_val_quality=np.nan, target_coverage=0.)
                rows.append(record); continue
            train = cache["source_train"]; train_mask = _pair_mask(train["label"], y_id, c_id)
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=1))
            model.fit(_pair_features(train, selected, components)[train_mask], train["label"][train_mask])
            result = {}
            for split in ("source_val", "target_val"):
                data = cache[split]; mask = _pair_mask(data["label"], y_id, c_id)
                quality = _binary_quality(model, _pair_features(data, selected, components)[mask], data["label"][mask], y_id)
                result[split] = quality
                record[f"{split}_support"] = int(mask.sum())
                record[f"{split}_balanced_accuracy"] = quality["balanced_accuracy"]
                record[f"{split}_pair_auroc"] = quality["pair_auroc"]
                record[f"{split}_sample_digest"] = _sample_digest(data, mask)
                auc_low, auc_high = _bootstrap_binary_metric(
                    quality["labels"] == y_id, quality["probability_y"], roc_auc_score,
                    seed=1 if split == "source_val" else 2,
                )
                record[f"{split}_pair_auroc_ci_low"] = auc_low
                record[f"{split}_pair_auroc_ci_high"] = auc_high
            record["candidate_digest"] = hashlib.sha256(
                ",".join(map(str, selected)).encode("utf-8")
            ).hexdigest()
            record.update(status="ok", source_val_quality=result["source_val"]["pair_auroc"],
                          target_val_quality=result["target_val"]["pair_auroc"], target_coverage=1.)
            group_results[group] = (record, result)
            rows.append(record)
        usable = [value[0] for value in group_results.values()]
        chosen, uncertain = select_group_source_only(usable) if usable else ("SC", True)
        for group in GROUPS:
            if group not in group_results or "SC" not in group_results:
                continue
            current, current_result = group_results[group]
            base, base_result = group_results["SC"]
            low, high = _bootstrap_delta(base_result["target_val"]["correct"], current_result["target_val"]["correct"])
            increments.append({
                "task": task, "variant": variant, "pair": pair["y_name"] + "__" + pair["c_name"],
                "group": group, "selected_by_source_val": group == chosen,
                "source_selection_uncertain": uncertain,
                "source_val_increment_vs_sc": current["source_val_quality"] - base["source_val_quality"],
                "target_val_increment_vs_sc": current["target_val_quality"] - base["target_val_quality"],
                "target_paired_accuracy_delta_ci_low": low, "target_paired_accuracy_delta_ci_high": high,
                "k2_given_k1_increment": (
                    current["target_val_quality"] - group_results["SC_K1"][0]["target_val_quality"]
                    if group == "SCP" and "SC_K1" in group_results else np.nan
                ),
            })
    overall_rows = []
    for group in GROUPS:
        record = {"task": task, "variant": variant, "scope": "teacher_correctness",
                  "pair": "ALL", "group": group, "selection_uses_target_val": False}
        for split in ("source_val", "target_val"):
            data = cache[split]; probability = _softmax(data["logits"]); order = np.argsort(-probability, axis=1)
            scores = np.full(len(order), np.nan)
            for index, (top1, top2) in enumerate(order[:, :2]):
                key = tuple(sorted((int(top1), int(top2))))
                packet = pair_score(candidates, data["events"][index], key[0], key[1], group, "source")
                if packet["available"]:
                    scores[index] = packet["score"] if top1 == key[0] else -packet["score"]
            correct = data["prediction"] == data["label"]; finite = np.isfinite(scores)
            record[f"{split}_support"] = len(scores)
            record[f"{split}_coverage"] = float(finite.mean())
            record[f"{split}_correctness_auroc"] = (
                float(roc_auc_score(correct[finite], scores[finite]))
                if len(np.unique(correct[finite])) == 2 else np.nan
            )
            record[f"{split}_error_auprc"] = (
                float(average_precision_score(~correct[finite], -scores[finite]))
                if (~correct[finite]).any() else np.nan
            )
        record["source_val_quality"] = record["source_val_correctness_auroc"]
        record["target_val_quality"] = record["target_val_correctness_auroc"]
        overall_rows.append(record)
    chosen, uncertain = select_group_source_only(overall_rows)
    for row in overall_rows:
        row["selected_by_source_val"] = row["group"] == chosen
        row["source_selection_uncertain"] = uncertain
    rows.extend(overall_rows)
    _write_csv(Path(output_dir) / "T1_component_comparison.csv", rows)
    _write_csv(Path(output_dir) / "T1_pairwise_phase_increment.csv", increments)
    return rows, increments


def _error_metric_rows(labels, predictions, confidence, margin, structure, competition):
    error = np.asarray(labels) != np.asarray(predictions)
    methods = {"confidence_only": -np.asarray(confidence), "teacher_margin_only": -np.asarray(margin),
               "structure": -np.asarray(structure)}
    rows = []
    for group in ("ALL", *sorted(set(competition))):
        selected = np.ones(len(error), bool) if group == "ALL" else np.asarray(competition) == group
        for method, score in methods.items():
            finite = selected & np.isfinite(score)
            rows.append({
                "competition_group": group, "method": method, "support": int(selected.sum()),
                "available": int(finite.sum()), "coverage": float(finite.sum() / max(selected.sum(), 1)),
                "error_auroc": float(roc_auc_score(error[finite], score[finite])) if len(np.unique(error[finite])) == 2 else np.nan,
                "error_auprc": float(average_precision_score(error[finite], score[finite])) if error[finite].any() else np.nan,
            })
    return rows


def run_t2(cache, candidates, classes, task, variant, output_dir):
    output_dir = Path(output_dir)
    graph_rows = [{"task": task, "variant": variant, **{key: value for key, value in row.items() if key != "joint_support"}}
                  for row in candidates]
    _write_csv(output_dir / "T2_pair_evidence_graph.csv", graph_rows)
    thresholds = teacher_thresholds(cache["source_val"]["logits"])
    sample_rows, rank_rows, eligibility = [], [], []
    val_structure, val_margin, val_conf, val_pred, val_label, val_comp = [], [], [], [], [], []
    for split in ("target_train", "target_val"):
        data = cache[split]
        if split == "target_train": assert_unlabeled_target_train(data)
        ranked = teacher_rank_rows(data["logits"], *thresholds, data["parcel_index"], data["domain"],
                                   labels=data.get("label") if split == "target_val" else None)
        for index, rank in enumerate(ranked):
            top = [rank["teacher_top1"], rank["teacher_top2"], rank["teacher_top3"]]
            strategy_packets = {}
            for strategy in ("none", "source", "target", "source_target"):
                pair_scores = {}; edges = []
                for y_id, c_id in combinations(top, 2):
                    key = tuple(sorted((y_id, c_id)))
                    scored = pair_score(candidates, data["events"][index], key[0], key[1], "SCP", strategy)
                    oriented = scored["score"] if y_id == key[0] else -scored["score"]
                    if scored["available"]: pair_scores[key] = scored["score"]
                    edges.append((y_id, c_id, oriented, scored))
                    row = {"split": split, "domain": rank["domain"], "parcel_index": rank["parcel_index"],
                           "strategy": strategy, "y_id": y_id, "c_id": c_id,
                           "score_y_over_c": oriented, **scored}
                    if split == "target_val": row["true_label"] = rank["true_label"]
                    sample_rows.append(row)
                resolution = resolve_pair_graph(top, pair_scores)
                strategy_packets[strategy] = (edges, resolution)
            edge = strategy_packets["source_target"][0][0]
            structure = edge[2] if edge[3]["available"] else np.nan
            outside = []
            for class_id in range(len(classes)):
                if class_id in top: continue
                key = tuple(sorted((top[0], class_id)))
                packet = pair_score(candidates, data["events"][index], key[0], key[1], "SCP", "source_target")
                if packet["available"]:
                    oriented = packet["score"] if top[0] == key[0] else -packet["score"]
                    if oriented < 0: outside.append((oriented, class_id))
            outside_candidate = min(outside)[1] if outside else -1
            status = strategy_packets["source_target"][1]["status"]
            if not edge[3]["available"]: failure_reason = "candidate_or_support_missing"
            elif status == "abstain_cycle": failure_reason = "cyclic_conflict"
            elif split == "target_val" and rank["teacher_top1"] != rank["true_label"] and structure >= 0:
                failure_reason = "structure_not_discriminative"
            elif split == "target_val" and rank["teacher_top1"] != rank["true_label"] and strategy_packets["source_target"][1].get("class_id") != rank["true_label"]:
                failure_reason = "alternative_mismatch"
            else: failure_reason = "none"
            rank.update({
                "split": split, "structure_top1_top2_score": structure,
                "structure_sufficient": bool(edge[3]["available"]),
                "topk_resolution": status,
                "topk_candidate": strategy_packets["source_target"][1].get("class_id", -1),
                "supported_outside_top3_candidate": outside_candidate,
                "failure_reason": failure_reason,
            })
            rank_rows.append(rank)
            if split == "target_val":
                probability = _softmax(data["logits"][[index]])[0]
                val_structure.append(structure); val_margin.append(rank["top1_top2_margin"])
                val_conf.append(rank["confidence"]); val_pred.append(rank["teacher_top1"])
                val_label.append(rank["true_label"]); val_comp.append(rank["competition_group"])
                for strategy, (edges, resolution) in strategy_packets.items():
                    supported = edges[0][3]["available"]
                    conflict = supported and edges[0][2] < 0
                    eligibility.append({
                        "task": task, "variant": variant, "strategy": strategy,
                        "competition_group": rank["competition_group"], "sample_count": 1,
                        "evidence_available": int(supported), "evidence_missing": int(not supported),
                        "wrong_supervision_identified": int(conflict and rank["teacher_top1"] != rank["true_label"]),
                        "correct_supervision_rejected": int(conflict and rank["teacher_top1"] == rank["true_label"]),
                        "correct_alternative_proposed": int(resolution.get("status") == "winner" and resolution.get("class_id") == rank["true_label"] and rank["teacher_top1"] != rank["true_label"]),
                        "abstain": int(resolution["status"].startswith("abstain")),
                    })
    _write_csv(output_dir / "T2_sample_pair_scores.csv", sample_rows)
    _write_csv(output_dir / "T2_teacher_rank_groups.csv", rank_rows)
    incremental = _error_metric_rows(val_label, val_pred, val_conf, val_margin, val_structure, val_comp)
    for row in incremental: row.update(task=task, variant=variant)
    _write_csv(output_dir / "T2_incremental_value.csv", incremental)
    grouped = {}
    for row in eligibility:
        key = (row["strategy"], row["competition_group"])
        target = grouped.setdefault(key, {k: 0 for k in (
            "sample_count", "evidence_available", "evidence_missing", "wrong_supervision_identified",
            "correct_supervision_rejected", "correct_alternative_proposed", "abstain")})
        for field in target: target[field] += row[field]
    eligibility_rows = []
    for (strategy, group), values in grouped.items():
        eligibility_rows.append({"task": task, "variant": variant, "strategy": strategy,
                                 "competition_group": group, **values,
                                 "coverage": values["evidence_available"] / max(values["sample_count"], 1),
                                 "missing_rate": values["evidence_missing"] / max(values["sample_count"], 1)})
    _write_csv(output_dir / "T2_eligibility_ablation.csv", eligibility_rows)
    return rank_rows, eligibility_rows


def write_semantic_cases(cache, candidates, classes, task, variant, output_dir):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    focus = {name for name in ("spring_barley", "winter_barley", "winter_wheat") if name in classes}
    def category(row):
        if row.get("status") != "ok": return "insufficient"
        if row.get("source_discriminable") and row.get("target_applicable"): return "valid"
        return "failed"

    selected = []
    for label in ("valid", "failed", "insufficient"):
        pool = [row for row in candidates if category(row) == label]
        pool.sort(key=lambda row: (
            not (row.get("y_name") in focus or row.get("c_name") in focus),
            -float(row.get("source_effect_norm_sc", 0)), row["pair"], row["anchor_id"],
        ))
        for row in pool[:2]:
            selected.append({**row, "semantic_case_type": label})
    figure, axes = plt.subplots(
        max(len(selected), 1), 4,
        figsize=(18, 3 * max(len(selected), 1)), squeeze=False,
    )
    lines = ["# Structure semantic cases", "", f"Task={task}, variant={variant}.", ""]
    for row_index, candidate in enumerate(selected):
        anchor = int(candidate["anchor_id"])
        representatives = {}
        for split_index, split in enumerate(("source_train", "target_val")):
            data = cache[split]
            curve_axis = axes[row_index, 2 * split_index]
            weight_axis = axes[row_index, 2 * split_index + 1]
            representative = np.flatnonzero((data.get("representative_anchor", np.array([])) == anchor) & (data.get("representative_kind", np.array([])) == "top"))
            pair_sample = (data["label"] == candidate["y_id"]) | (data["label"] == candidate["c_id"])
            allowed_parcels = data["parcel_index"][pair_sample]
            representative = representative[
                np.isin(data.get("representative_parcel_index", np.array([], dtype=int))[representative], allowed_parcels)
            ]
            representatives[split] = representative
            for index in representative[:3]:
                curve = data["representative_standardized"][index]
                curve_axis.plot(curve.mean(1), alpha=.75)
            y_id, c_id = candidate["y_id"], candidate["c_id"]
            centers = np.asarray(data.get("window_centers_days", []))
            for class_id, style in ((y_id, "-"), (c_id, "--")):
                mask = data["label"] == class_id
                if mask.any():
                    values = data["pooling_weight"][mask, :, anchor].mean(0)
                    x_values = centers if len(centers) == len(values) else np.arange(len(values))
                    weight_axis.plot(
                        x_values, values, style, linewidth=2, label=classes[class_id],
                    )
            curve_axis.set_title(
                f"{split} local PSE {candidate['semantic_case_type']} {candidate['pair']} A{anchor}"
            )
            curve_axis.set_xlabel("local window index")
            weight_axis.set_title(f"{split} annual match weight")
            weight_axis.set_xlabel("window center day" if len(centers) else "window index")
            weight_axis.legend(fontsize=7)
        source_rep, target_rep = representatives["source_train"], representatives["target_val"]
        if len(source_rep) and len(target_rep):
            source_shape = cache["source_train"]["representative_standardized"][source_rep].mean(0).reshape(-1)
            target_shape = cache["target_val"]["representative_standardized"][target_rep].mean(0).reshape(-1)
            identity_stability = _cosine(source_shape, target_shape)
        else:
            identity_stability = np.nan
        phase_concentration = []
        for split in ("source_train", "target_val"):
            data = cache[split]
            selected_class = (data["label"] == candidate["y_id"]) | (data["label"] == candidate["c_id"])
            event = data["events"][selected_class, anchor]
            if len(event): phase_concentration.append(float(np.sqrt(event[:, 2] ** 2 + event[:, 3] ** 2).mean()))
        timing_interpretable = bool(np.isfinite(identity_stability) and identity_stability >= .5 and phase_concentration and min(phase_concentration) >= .2)
        lines.append(
            f"- [{candidate['semantic_case_type']}] {candidate['pair']} / A{anchor}: "
            f"source_discriminable={candidate.get('source_discriminable', False)}, "
            f"target_train_range={float(candidate.get('target_train_range_rate', np.nan)):.3f}, "
            f"cross_domain_shape_cosine={identity_stability}, "
            f"k1_concentration_min={(min(phase_concentration) if phase_concentration else np.nan)}, "
            f"relative_timing_interpretable={timing_interpretable}. Anchor ID is scoped to this checkpoint only."
        )
    if not selected:
        axes[0, 0].text(.5, .5, "No representative cases", ha="center")
        for axis in axes[0, 1:]: axis.axis("off")
        lines.append("- No representative Anchor–class pair; semantics remain undetermined.")
    figure.tight_layout(); figure.savefig(output_dir / "T2_semantic_cases.png", dpi=150); plt.close(figure)
    lines.extend(["", "Representative curves are local PSE windows, not raw NDVI. Relative timing is interpretable only when matching is stable and phase concentration is sufficient."])
    (output_dir / "T2_semantic_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_cache(cache_dir):
    from analysis.structure_semantic_features import load_feature_split
    result = {split: load_feature_split(cache_dir, "teacher", split) for split in ALLOWED_SPLITS}
    assert_unlabeled_target_train(result["target_train"])
    return result


def run_audit(cache_dir, output_dir, task, variant, checkpoint):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = packet["config"]; config = raw if isinstance(raw, dict) else vars(raw)
    classes = list(config["classes"]); digest = sha256_file(checkpoint)
    _, expected_source, _, expected_target = TASKS[task]
    if config.get("source") != expected_source or config.get("target") != expected_target:
        raise ValueError("checkpoint task/domain mismatch")
    if config.get("shape_representation") != "phase_moment" or config.get("shape_injection") != "current_query":
        raise ValueError("checkpoint representation is incompatible")
    validate_cache_identity(
        cache_dir, digest, classes, expected_task=task, expected_variant=variant,
    )
    cache = _load_cache(cache_dir)
    candidates = build_pair_candidates(cache, classes)
    run_t1(cache, candidates, classes, task, variant, output_dir)
    run_t2(cache, candidates, classes, task, variant, output_dir)
    write_semantic_cases(cache, candidates, classes, task, variant, output_dir)
    manifest = {
        "task": task, "variant": variant, "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest, "checkpoint_epoch": int(packet.get("epoch", -1)),
        "classes": classes, "source": config.get("source"), "target": config.get("target"),
        "global_temporal_shift": float(torch.as_tensor(packet.get("global_temporal_shift", 0)).reshape(-1)[0]),
        "target_test_used": False, "anchor_scope": f"{variant}:{task}:{digest}",
        "splits": list(ALLOWED_SPLITS), "selection_split": "source_val",
    }
    _write_json(output_dir / "checkpoint_manifest.json", manifest)


def load_generic_frozen(checkpoint, task, variant, device):
    from analysis.structure_semantic_features import FrozenModels, create_model_from_config
    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = packet.get("config"); config = dict(raw if isinstance(raw, dict) else vars(raw))
    _, source, _, target = TASKS[task]
    expected = {
        "source": source, "target": target, "shape_representation": "phase_moment",
        "shape_injection": "current_query", "shape_equivariance_weight": .05,
        "target_structure_detach_epochs": 0 if variant == "E" else 5,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items()
                  if config.get(key) != value and not (isinstance(value, float) and np.isclose(float(config.get(key, np.nan)), value))}
    if mismatches: raise ValueError(f"incompatible checkpoint: {mismatches}")
    namespace = SimpleNamespace(**config); student = create_model_from_config(namespace); teacher = create_model_from_config(namespace)
    student.load_state_dict(packet["state_dict"], strict=True); teacher.load_state_dict(packet["teacher_state_dict"], strict=True)
    for model in (student, teacher):
        model.to(device).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
    shift = int(torch.as_tensor(packet.get("global_temporal_shift", 0)).item())
    return FrozenModels(student, teacher, namespace, packet, {"student": 0, "teacher": shift})


def run_extract(checkpoint, source_checkpoint, cache_dir, task, variant, data_root, device, batch_size, pixel_budget):
    from analysis.structure_semantic_abc import _split_manifest
    from analysis.structure_semantic_features import (
        build_abc_datasets, create_model_from_config, extract_checkpoint_features,
    )
    frozen = load_generic_frozen(checkpoint, task, variant, device)
    source_packet = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_raw = source_packet["config"]; source_config = source_raw if isinstance(source_raw, dict) else vars(source_raw)
    expected_source = TASKS[task][1]
    if source_config.get("source") != expected_source or source_config.get("target") != expected_source:
        raise ValueError("Source checkpoint domain mismatch")
    for key, expected in (("shape_representation", "phase_moment"), ("shape_injection", "current_query")):
        if source_config.get(key) != expected: raise ValueError(f"Source checkpoint {key} mismatch")
    if list(source_config["classes"]) != list(frozen.config.classes): raise ValueError("Source/UDA class mapping mismatch")
    source_model = create_model_from_config(SimpleNamespace(**dict(source_config)))
    source_model.load_state_dict(source_packet["state_dict"], strict=True)
    del source_model
    config, protocol, _, datasets = build_abc_datasets(frozen.config, data_root, 1)
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    feature = extract_checkpoint_features(frozen, datasets, {"source": config.source, "target": config.target},
                                          cache_dir, device=device, batch_size=batch_size,
                                          pixel_budget=pixel_budget, representative_count=3)
    _write_json(cache_dir / "checkpoint_manifest.json", {
        "task": task, "variant": variant, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint), "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint), "classes": list(config.classes),
        "checkpoint_epoch": int(frozen.packet.get("epoch", -1)),
        "source_checkpoint_epoch": int(source_packet.get("epoch", -1)),
        "protocol": protocol, "split_manifest": _split_manifest(datasets, config),
        "target_test_used": False, "target_train_labels_cached": False,
        "applied_shifts": feature.applied_shifts,
    })


def write_summary(output_root):
    output_root = Path(output_root)
    component, increment, eligibility, incremental, ranks = [], [], [], [], []
    for path in output_root.glob("[EG]_*/T1_component_comparison.csv"):
        with path.open(newline="", encoding="utf-8") as stream: component.extend(csv.DictReader(stream))
    for path in output_root.glob("[EG]_*/T1_pairwise_phase_increment.csv"):
        with path.open(newline="", encoding="utf-8") as stream: increment.extend(csv.DictReader(stream))
    for path in output_root.glob("[EG]_*/T2_eligibility_ablation.csv"):
        with path.open(newline="", encoding="utf-8") as stream: eligibility.extend(csv.DictReader(stream))
    for path in output_root.glob("[EG]_*/T2_incremental_value.csv"):
        with path.open(newline="", encoding="utf-8") as stream: incremental.extend(csv.DictReader(stream))
    for path in output_root.glob("[EG]_*/T2_teacher_rank_groups.csv"):
        with path.open(newline="", encoding="utf-8") as stream: ranks.extend(csv.DictReader(stream))

    def numbers(rows, field, predicate=lambda row: True):
        values = []
        for row in rows:
            if not predicate(row): continue
            try: value = float(row.get(field, "nan"))
            except (TypeError, ValueError): continue
            if np.isfinite(value): values.append(value)
        return values

    completed = len(list(output_root.glob("[EG]_*/checkpoint_manifest.json")))
    k2 = numbers(increment, "target_val_increment_vs_sc", lambda row: row.get("group") == "SC_K2")
    k2_after_k1 = numbers(increment, "k2_given_k1_increment", lambda row: row.get("group") == "SCP")
    source_k2 = numbers(increment, "source_val_increment_vs_sc", lambda row: row.get("group") == "SC_K2")
    source_gain_target_loss = sum(
        1 for row in increment if row.get("group") == "SC_K2"
        and numbers([row], "source_val_increment_vs_sc") and numbers([row], "target_val_increment_vs_sc")
        and float(row["source_val_increment_vs_sc"]) > 0 and float(row["target_val_increment_vs_sc"]) < 0
    )
    qualified = [row for row in eligibility if row.get("strategy") == "source_target"]
    qualified_coverage = numbers(qualified, "coverage")
    wrong_found = sum(int(float(row.get("wrong_supervision_identified", 0))) for row in qualified)
    correct_rejected = sum(int(float(row.get("correct_supervision_rejected", 0))) for row in qualified)
    failure_counts = {}
    for row in ranks:
        reason = row.get("failure_reason", "unknown")
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    structure_auc = numbers(incremental, "error_auroc", lambda row: row.get("method") == "structure")
    confidence_auc = numbers(incremental, "error_auroc", lambda row: row.get("method") == "confidence_only")
    manifests_expected = 2 * len(TASKS)
    readiness = "not established"
    if completed == manifests_expected and qualified and structure_auc:
        readiness = "audit coverage complete; training readiness still requires human review of CSV evidence"

    def describe(values):
        if not values: return "unavailable"
        return f"n={len(values)}, mean={np.mean(values):.4f}, range=[{min(values):.4f}, {max(values):.4f}], positive={sum(v > 0 for v in values)}"

    text = (
        "# Frozen all-pair structure semantics summary\n\n"
        f"Completed compatible checkpoints: {completed}/{manifests_expected}. Target Test was not instantiated or accessed.\n\n"
        "## 1. K2 task/class-pair dependence\n\n"
        f"- K2 vs SC on Target Val: {describe(k2)}.\n"
        f"- K2 given K1 (SCP vs SC+K1): {describe(k2_after_k1)}.\n"
        f"- K2 vs SC on Source Val: {describe(source_k2)}; Source-gain/Target-loss relations={source_gain_target_loss}.\n"
        "- Pair-level effects, supports, identical sample/candidate digests and bootstrap intervals are in T1 CSVs.\n\n"
        "## 2. P3 failure sources after complete pair enumeration\n\n"
        f"- Offline failure counts: {json.dumps(failure_counts, sort_keys=True)}. Missing evidence and cycles remain abstentions, never zero votes or argmax inputs.\n\n"
        "## 3. Teacher competition and incremental value\n\n"
        f"- Structure error-AUROC strata: {describe(structure_auc)}; confidence-only strata: {describe(confidence_auc)}.\n"
        "- Rank/competition-specific results and supported candidates outside Top-3 are in T2_teacher_rank_groups.csv and T2_incremental_value.csv.\n\n"
        "## 4. Evidence eligibility\n\n"
        f"- Source-discriminable + Target-applicable strata={len(qualified)}, coverage={describe(qualified_coverage)}, wrong supervision identified={wrong_found}, correct supervision rejected={correct_rejected}.\n"
        "- Eligibility is only a qualification condition; Target Val determines its observed offline reliability.\n\n"
        "## 5. Anchor–Phase semantics\n\n"
        "- Representative PSE windows, annual matching-weight distributions, SC/K1/K2 evidence and timing interpretability gates are reported per checkpoint in T2_semantic_cases.md/png. Anchor IDs are never compared across checkpoint scopes.\n\n"
        "## 6. E4 evidence status\n\n"
        f"- {readiness}. This audit did not implement Reject/Relabel training or E4 and does not infer causality.\n"
    )
    (output_root / "FINAL_summary.md").write_text(text, encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("extract", "audit", "summary"), required=True)
    parser.add_argument("--task", choices=tuple(TASKS)); parser.add_argument("--variant", choices=("E", "G"))
    parser.add_argument("--checkpoint"); parser.add_argument("--source-checkpoint"); parser.add_argument("--cache-dir")
    parser.add_argument("--output-root", required=True); parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "summary": write_summary(args.output_root); return
    if not all((args.task, args.variant, args.checkpoint, args.cache_dir)):
        raise ValueError("checkpoint mode needs task, variant, checkpoint, and cache-dir")
    if args.mode == "extract":
        if not args.source_checkpoint: raise ValueError("extract needs source-checkpoint")
        run_extract(args.checkpoint, args.source_checkpoint, args.cache_dir, args.task, args.variant,
                    args.data_root, args.device, args.batch_size, args.pixel_budget)
    else:
        run_audit(args.cache_dir, Path(args.output_root) / f"{args.variant}_{args.task}",
                  args.task, args.variant, args.checkpoint)


if __name__ == "__main__":
    main()
