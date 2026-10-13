#!/usr/bin/env python3
"""Frozen conditional-increment audit for Teacher and pair-structure evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.structure_pair_semantics_audit import (
    _load_cache, build_pair_candidates, pair_score,
)


def teacher_continuous_features(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(1, keepdims=True)
    probability = np.exp(shifted); probability /= probability.sum(1, keepdims=True)
    ranking = np.argsort(-probability, axis=1)
    ordered = np.take_along_axis(probability, ranking, axis=1)
    entropy = -(probability * np.log(probability.clip(1e-12))).sum(1)
    return {
        "probability": probability, "ranking": ranking,
        "confidence": ordered[:, 0], "margin": ordered[:, 0] - ordered[:, 1],
        "entropy": entropy,
    }


class CalibratedReliability:
    def __init__(self, base, calibrator=None):
        self.base, self.calibrator = base, calibrator

    def predict_reliability(self, features):
        raw = self.base.predict_proba(features)[:, list(self.base[-1].classes_).index(1)]
        if self.calibrator is None:
            return raw
        return self.calibrator.predict_proba(raw[:, None])[:, 1]


@dataclass
class ReliabilityPair:
    status: str
    sample_count: int
    teacher_feature_dim: int
    combined_feature_dim: int
    teacher_model: Optional[CalibratedReliability]
    combined_model: Optional[CalibratedReliability]
    calibration_status: str = "not_requested"


def _fit_one(features, correct):
    model = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.), StandardScaler(),
        LogisticRegression(C=1., class_weight="balanced", max_iter=1000, random_state=1),
    )
    model.fit(np.asarray(features, float), np.asarray(correct, int))
    return model


def _calibrate(base, features, correct):
    correct = np.asarray(correct, int)
    if len(np.unique(correct)) < 2 or min(np.bincount(correct, minlength=2)) < 2:
        return None
    raw = base.predict_proba(features)[:, list(base[-1].classes_).index(1)]
    calibrator = LogisticRegression(C=1., max_iter=1000, random_state=1)
    calibrator.fit(raw[:, None], correct)
    return calibrator


def fit_reliability_pair(
    teacher_features, structure_features, correct, min_errors=5,
    calibration_teacher=None, calibration_structure=None, calibration_correct=None,
):
    teacher_features = np.asarray(teacher_features, float)
    structure_features = np.asarray(structure_features, float)
    correct = np.asarray(correct, int)
    if not (len(teacher_features) == len(structure_features) == len(correct)):
        raise ValueError("Teacher and Structure must use the same samples")
    result = ReliabilityPair(
        "unestimable_source_errors", len(correct), teacher_features.shape[1],
        teacher_features.shape[1] + structure_features.shape[1], None, None,
    )
    error_count = int((correct == 0).sum())
    if error_count < int(min_errors) or int((correct == 1).sum()) < int(min_errors):
        return result
    teacher = _fit_one(teacher_features, correct)
    combined_features = np.concatenate((teacher_features, structure_features), axis=1)
    combined = _fit_one(combined_features, correct)
    status = "not_requested"
    teacher_calibrator = combined_calibrator = None
    if calibration_teacher is not None:
        if calibration_structure is None or calibration_correct is None:
            raise ValueError("Source Val calibration inputs must be complete")
        if len(calibration_teacher) != len(calibration_structure):
            raise ValueError("Source Val calibration must use the same samples")
        teacher_calibrator = _calibrate(teacher, calibration_teacher, calibration_correct)
        combined_calibrator = _calibrate(
            combined, np.concatenate((calibration_teacher, calibration_structure), axis=1),
            calibration_correct,
        )
        status = "source_val" if teacher_calibrator is not None and combined_calibrator is not None else "source_val_unestimable"
    return ReliabilityPair(
        "ok", len(correct), teacher_features.shape[1], combined_features.shape[1],
        CalibratedReliability(teacher, teacher_calibrator),
        CalibratedReliability(combined, combined_calibrator), status,
    )


def _structure_rows(arrays, candidates):
    teacher = teacher_continuous_features(arrays["logits"])
    rows = []
    for index, ranking in enumerate(teacher["ranking"]):
        y_id, c_id = sorted(map(int, ranking[:2]))
        pair_candidates = candidates.get((y_id, c_id), []) if isinstance(candidates, dict) else candidates
        packet = pair_score(
            pair_candidates, arrays["events"][index], y_id, c_id,
            group="SCP", strategy="source_target",
        )
        oriented = packet["score"] if int(ranking[0]) == y_id else -packet["score"]
        rows.append({
            "structure_score_top1_over_top2": oriented,
            "anchor_support_count": packet["evidence_count"],
            "evidence_available": int(packet["available"]),
            "source_target_range_fraction": packet["range_fraction"],
        })
    return rows


def _feature_matrices(arrays, structure):
    teacher = teacher_continuous_features(arrays["logits"])
    teacher_matrix = np.column_stack((
        teacher["confidence"], teacher["margin"], teacher["entropy"],
    ))
    structure_matrix = np.asarray([[row[key] for key in (
        "structure_score_top1_over_top2", "anchor_support_count",
        "evidence_available", "source_target_range_fraction",
    )] for row in structure], float)
    return teacher, teacher_matrix, structure_matrix


def sample_evidence_rows(arrays, split, structure):
    if split == "target_train" and "label" in arrays:
        raise ValueError("Target Train must remain unlabeled")
    teacher = teacher_continuous_features(arrays["logits"])
    rows = []
    for index, packet in enumerate(structure):
        row = {
            "split": split, "domain": str(arrays["domain"][index]),
            "parcel_index": int(arrays["parcel_index"][index]),
            "teacher_logits": json.dumps(np.asarray(arrays["logits"][index], float).tolist()),
            "teacher_probability": json.dumps(teacher["probability"][index].tolist()),
            "teacher_ranking": json.dumps(teacher["ranking"][index].tolist()),
            "teacher_prediction": int(teacher["ranking"][index, 0]),
            "confidence": float(teacher["confidence"][index]),
            "margin": float(teacher["margin"][index]),
            "entropy": float(teacher["entropy"][index]), **packet,
        }
        if split != "target_train" and "label" in arrays:
            row["true_label"] = int(arrays["label"][index])
            row["teacher_correct"] = int(row["teacher_prediction"] == row["true_label"])
        rows.append(row)
    return rows


def _ece(correct, reliability, bins=10):
    correct, reliability = np.asarray(correct), np.asarray(reliability)
    result = 0.
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        selected = (reliability >= lower) & (reliability <= upper if upper == 1 else reliability < upper)
        if selected.any():
            result += selected.mean() * abs(correct[selected].mean() - reliability[selected].mean())
    return float(result)


def reliability_metrics(correct, reliability):
    correct, reliability = np.asarray(correct, int), np.asarray(reliability, float)
    error = 1 - correct; error_score = 1 - reliability
    return {
        "error_auroc": float(roc_auc_score(error, error_score)) if len(np.unique(error)) == 2 else np.nan,
        "error_auprc": float(average_precision_score(error, error_score)) if error.any() else np.nan,
        "brier": float(brier_score_loss(correct, reliability)), "ece": _ece(correct, reliability),
    }


def retention_curve(correct, score, method):
    order = np.argsort(-np.asarray(score, float)); correct = np.asarray(correct, int)
    rows = []
    for fraction in np.linspace(.1, 1., 10):
        count = max(1, int(round(len(order) * fraction))); chosen = order[:count]
        rows.append({"method": method, "retention_fraction": fraction,
                     "retained_count": count, "error_rate": float(1 - correct[chosen].mean())})
    return rows


def _write_csv(path, rows):
    rows = list(rows); fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _continuous_plot(path, rows):
    """Plot continuous relationships only; do not create post-hoc groups."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    target = [row for row in rows if row["split"] == "target_val" and "teacher_correct" in row]
    if not target:
        return False
    fields = ("confidence", "margin", "entropy", "structure_score_top1_over_top2")
    figure, axes = plt.subplots(1, len(fields), figsize=(14, 3.2), constrained_layout=True)
    correct = np.asarray([row["teacher_correct"] for row in target], float)
    for axis, field in zip(axes, fields):
        value = np.asarray([row.get(field, np.nan) for row in target], float)
        finite = np.isfinite(value)
        axis.scatter(value[finite], correct[finite], s=5, alpha=.12)
        if finite.sum() >= 10:
            order = np.argsort(value[finite]); ordered_x, ordered_y = value[finite][order], correct[finite][order]
            edges = np.array_split(np.arange(len(ordered_x)), min(20, len(ordered_x)))
            axis.plot([ordered_x[group].mean() for group in edges],
                      [ordered_y[group].mean() for group in edges], color="black", linewidth=1.5)
        axis.set(xlabel=field, ylabel="teacher correctness", ylim=(-.05, 1.05))
    figure.savefig(path, dpi=150); plt.close(figure)
    return True


def _classes_from_cache(cache_dir):
    manifest = Path(cache_dir) / "checkpoint_manifest.json"
    if manifest.is_file():
        values = json.loads(manifest.read_text(encoding="utf-8")).get("classes")
        if values:
            return list(values)
    raise ValueError("classes were not supplied and cache manifest has no class mapping")


def run(args):
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    classes = args.classes or _classes_from_cache(args.cache_dir)
    cache = _load_cache(args.cache_dir); raw_candidates = build_pair_candidates(cache, classes)
    candidates = {}
    for row in raw_candidates:
        pair = (int(row["pair_y"]), int(row["pair_c"]))
        candidates.setdefault(pair, []).append(row)
    structures = {split: _structure_rows(data, candidates) for split, data in cache.items()}
    matrices = {split: _feature_matrices(cache[split], structures[split]) for split in cache}
    train_correct = (cache["source_train"]["prediction"] == cache["source_train"]["label"]).astype(int)
    val_correct = (cache["source_val"]["prediction"] == cache["source_val"]["label"]).astype(int)
    fitted = fit_reliability_pair(
        matrices["source_train"][1], matrices["source_train"][2], train_correct,
        calibration_teacher=matrices["source_val"][1],
        calibration_structure=matrices["source_val"][2], calibration_correct=val_correct,
    )
    sample_rows = []
    for split in cache:
        sample_rows.extend(sample_evidence_rows(cache[split], split, structures[split]))
    metric_rows, curves = [], []
    if fitted.status == "ok":
        for split in ("source_val", "target_val"):
            data = cache[split]; correct = (data["prediction"] == data["label"]).astype(int)
            teacher_x, structure_x = matrices[split][1:]
            scores = {
                "confidence": matrices[split][0]["confidence"],
                "teacher_only": fitted.teacher_model.predict_reliability(teacher_x),
                "teacher_structure": fitted.combined_model.predict_reliability(np.concatenate((teacher_x, structure_x), 1)),
            }
            for method, values in scores.items():
                metric_rows.append({"task": args.task, "variant": args.variant, "split": split,
                                    "method": method, "sample_count": len(correct), **reliability_metrics(correct, values)})
                if split == "target_val": curves.extend({"task": args.task, "variant": args.variant, **row}
                                                        for row in retention_curve(correct, values, method))
    else:
        metric_rows.append({"task": args.task, "variant": args.variant, "split": "source_train",
                            "method": "all", "status": fitted.status, "sample_count": fitted.sample_count})
    _write_csv(output / "X1_sample_evidence.csv", sample_rows)
    _write_csv(output / "X1_teacher_vs_combined.csv", metric_rows)
    _write_csv(output / "X1_coverage_error.csv", curves)
    plotted = _continuous_plot(output / "X1_continuous_relationships.png", sample_rows)
    target_metrics = {row["method"]: row for row in metric_rows if row.get("split") == "target_val"}
    delta = "unavailable"
    if "teacher_only" in target_metrics and "teacher_structure" in target_metrics:
        delta = json.dumps({
            key: target_metrics["teacher_structure"][key] - target_metrics["teacher_only"][key]
            for key in ("error_auroc", "error_auprc", "brier", "ece")
        }, sort_keys=True)
    lines = ["# Teacher vs Structure conditional increment", "",
             f"Task={args.task}, variant={args.variant}, fit_status={fitted.status}, calibration={fitted.calibration_status}.",
             f"Target metric deltas (combined minus teacher-only): `{delta}`.",
             f"Continuous relationship plot generated: {plotted}.", "",
             "All fitting and calibration use Source labels only. Target Train labels are absent; Target Val is evaluation only. No confidence grouping or Target-selected threshold is created."]
    (output / "X1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser():
    value = argparse.ArgumentParser(); value.add_argument("--cache-dir", required=True)
    value.add_argument("--output-dir", required=True); value.add_argument("--task", required=True)
    value.add_argument("--variant", required=True); value.add_argument("--classes", nargs="+")
    return value


if __name__ == "__main__": run(parser().parse_args())
