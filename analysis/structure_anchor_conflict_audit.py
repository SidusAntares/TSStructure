#!/usr/bin/env python3
"""Frozen audit of within-pair anchor disagreement and domain instability."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis.structure_pair_semantics_audit import (
    _candidate_in_range,
    _load_cache,
    build_pair_candidates,
    signed_pair_evidence,
)
from analysis.structure_evidence_increment_audit import teacher_continuous_features
from analysis.structure_evidence_increment_audit import _classes_from_cache


def oriented_anchor_scores(scores, pair_y, pair_c, y_id, c_id):
    """Orient all anchor scores toward requested y over c."""
    original = (int(pair_y), int(pair_c))
    requested = (int(y_id), int(c_id))
    if requested == original:
        return np.asarray(scores, dtype=float)
    if requested == original[::-1]:
        return -np.asarray(scores, dtype=float)
    raise ValueError("requested classes do not match candidate pair")


def anchor_conflict_statistics(scores, weak_threshold=.1):
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "strong_conflict": False, "positive_strength": 0.,
            "negative_strength": 0., "dispersion": np.nan,
            "cancellation_ratio": np.nan, "anchor_count": 0,
        }
    positive = values[values > float(weak_threshold)]
    negative = -values[values < -float(weak_threshold)]
    absolute = np.abs(values).sum()
    cancellation = 0. if absolute <= 1e-12 else 1. - abs(values.sum()) / absolute
    return {
        "strong_conflict": bool(len(positive) and len(negative)),
        "positive_strength": float(positive.sum()),
        "negative_strength": float(negative.sum()),
        "dispersion": float(np.std(values)),
        "cancellation_ratio": float(np.clip(cancellation, 0., 1.)),
        "anchor_count": int(len(values)),
    }


def fit_source_conflict_reference(source_scores):
    rows = np.asarray(source_scores, dtype=float)
    if rows.ndim != 2 or rows.shape[0] < 2:
        raise ValueError("Source reference requires a two-dimensional sample-by-anchor array")
    finite_magnitude = np.abs(rows[np.isfinite(rows)])
    weak = float(max(np.quantile(finite_magnitude, .25), 1e-6)) if len(finite_magnitude) else .1
    packets = [anchor_conflict_statistics(row, weak) for row in rows]
    cancellation = np.asarray([row["cancellation_ratio"] for row in packets], float)
    dispersion = np.asarray([row["dispersion"] for row in packets], float)
    correlation = np.corrcoef(rows, rowvar=False) if rows.shape[1] > 1 else np.ones((1, 1))
    off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]
    return {
        "weak_threshold": weak,
        "conflict_threshold": float(np.nanquantile(cancellation, .95)),
        "dispersion_threshold": float(np.nanquantile(dispersion, .95)),
        "anchor_median": np.nanmedian(rows, axis=0),
        "anchor_stability": 1. / np.maximum(np.nanstd(rows, axis=0), .05),
        "source_anchor_correlation_mean": float(np.nanmean(off_diagonal)) if len(off_diagonal) else 1.,
        "source_conflict_rate": float(np.mean([row["strong_conflict"] for row in packets])),
        "source_dispersion_mean": float(np.nanmean(dispersion)),
        "source_sample_count": int(rows.shape[0]),
    }


def weighted_median(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    values, weights = values[valid], weights[valid]
    order = np.argsort(values); values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), .5 * weights.sum(), side="left")])


def aggregate_anchor_evidence(
    scores, weights, method="weighted_mean", reference=None, anchor_ids=None,
):
    scores, weights = np.asarray(scores, float), np.asarray(weights, float)
    anchor_ids = (
        np.arange(len(scores), dtype=int)
        if anchor_ids is None else np.asarray(anchor_ids, dtype=int)
    )
    if len(anchor_ids) != len(scores):
        raise ValueError("anchor_ids must align with scores")
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return {"available": False, "status": "no_evidence", "score": np.nan}
    scores, weights, anchor_ids = scores[valid], weights[valid], anchor_ids[valid]
    weak = .1 if reference is None else float(reference["weak_threshold"])
    stats = anchor_conflict_statistics(scores, weak)
    if method == "abstain_conflict":
        if reference is None:
            raise ValueError("abstain_conflict requires a Source reference")
        abnormal = (
            stats["strong_conflict"]
            and (
                stats["cancellation_ratio"] > float(reference["conflict_threshold"])
                or stats["dispersion"] > float(reference["dispersion_threshold"])
            )
        )
        if abnormal:
            return {**stats, "available": False, "status": "abstain_conflict", "score": np.nan}
        method = "weighted_mean"
    if method == "stability_weighted":
        if reference is None:
            raise ValueError("stability_weighted requires a Source reference")
        stability_values = np.asarray(reference["anchor_stability"], float)
        reference_ids = np.asarray(
            reference.get("anchor_ids", np.arange(len(stability_values))), dtype=int,
        )
        stability_by_anchor = dict(zip(reference_ids.tolist(), stability_values.tolist()))
        if any(int(anchor) not in stability_by_anchor for anchor in anchor_ids):
            return {**stats, "available": False, "status": "anchor_not_in_source_reference", "score": np.nan}
        stability = np.asarray([stability_by_anchor[int(anchor)] for anchor in anchor_ids])
        weights = weights * stability
        score = float(np.average(scores, weights=weights))
    elif method == "weighted_median":
        score = weighted_median(scores, weights)
    elif method == "weighted_mean":
        score = float(np.average(scores, weights=weights))
    else:
        raise ValueError(f"unknown aggregation method: {method}")
    return {**stats, "available": True, "status": "ok", "score": score}


def evidence_coverage(rows):
    rows = list(rows)
    return float(np.mean([bool(row.get("available", False)) for row in rows])) if rows else np.nan


def _pair_candidates(candidates, y_id, c_id):
    pair = tuple(sorted((int(y_id), int(c_id))))
    if isinstance(candidates, dict):
        return candidates.get(pair, [])
    return [row for row in candidates if (
        row.get("pair_y"), row.get("pair_c")
    ) == pair and row.get("status") == "ok" and row.get("source_discriminable", False)]


def _anchor_packet(candidates, events, y_id, c_id, require_range=True):
    rows = _pair_candidates(candidates, y_id, c_id)
    scores, weights, anchor_ids, ranges = [], [], [], []
    for row in rows:
        anchor = int(row["anchor_id"])
        in_range = _candidate_in_range(row, np.asarray(events)[anchor])
        if require_range and not in_range:
            continue
        scores.append(float(signed_pair_evidence(row, np.asarray(events)[None], y_id, c_id, "SCP")))
        weights.append(max(float(row.get("source_direction_cosine_sc", 0.)), 0.) + 1e-6)
        anchor_ids.append(anchor); ranges.append(in_range)
    return np.asarray(scores), np.asarray(weights), np.asarray(anchor_ids), np.asarray(ranges)


def _reference_rows(cache, candidates):
    references = {}
    grouped = defaultdict(list)
    for split in ("source_train", "source_val"):
        data = cache[split]
        for index, truth in enumerate(data["label"]):
            # Build Source norms only for every eligible pair containing the true class.
            for pair in sorted(candidates):
                if int(truth) not in pair:
                    continue
                scores, _, anchors, _ = _anchor_packet(
                    candidates, data["events"][index], *pair, require_range=False,
                )
                expected = tuple(row["anchor_id"] for row in _pair_candidates(candidates, *pair))
                if len(scores) == len(expected) and tuple(anchors) == expected:
                    grouped[pair].append(scores)
    for pair, values in grouped.items():
        if len(values) >= 2:
            references[pair] = fit_source_conflict_reference(np.stack(values))
            references[pair]["anchor_ids"] = np.asarray(
                [row["anchor_id"] for row in _pair_candidates(candidates, *pair)], dtype=int,
            )
    return references


def _write_csv(path, rows):
    rows = list(rows); fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _binary_metrics(truth, score):
    truth, score = np.asarray(truth, int), np.asarray(score, float)
    valid = np.isfinite(score)
    if valid.sum() == 0 or len(np.unique(truth[valid])) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(truth[valid], score[valid])), float(average_precision_score(truth[valid], score[valid]))


def run(args):
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(args.cache_dir)
    classes = args.classes or _classes_from_cache(args.cache_dir)
    raw_candidates = build_pair_candidates(cache, classes)
    candidates = defaultdict(list)
    for row in raw_candidates:
        if row.get("status") == "ok" and row.get("source_discriminable", False):
            candidates[(int(row["pair_y"]), int(row["pair_c"]))].append(row)
    candidates = dict(candidates)
    references = _reference_rows(cache, candidates)
    anchor_rows, sample_rows, ablations = [], [], []
    methods = ("weighted_mean", "stability_weighted", "weighted_median", "abstain_conflict")
    method_packets = defaultdict(list)
    for split, data in cache.items():
        teacher = teacher_continuous_features(data["logits"])
        for index, ranking in enumerate(teacher["ranking"]):
            top1, top2 = map(int, ranking[:2]); pair = tuple(sorted((top1, top2)))
            scores, weights, anchors, ranges = _anchor_packet(
                candidates, data["events"][index], top1, top2,
                require_range=split.startswith("target"),
            )
            reference = references.get(pair)
            if reference is None or not len(scores):
                continue
            stats = anchor_conflict_statistics(scores, reference["weak_threshold"])
            error = None if split == "target_train" else int(top1 != int(data["label"][index]))
            identity = {
                "split": split, "domain": str(data["domain"][index]),
                "parcel_index": int(data["parcel_index"][index]),
                "pair_y": top1, "pair_c": top2, "pair": f"{classes[top1]}__{classes[top2]}",
            }
            for anchor, score, weight, in_range in zip(anchors, scores, weights, ranges):
                anchor_rows.append({**identity, "anchor_id": int(anchor), "signed_score": float(score),
                                    "source_weight": float(weight), "in_source_target_range": int(in_range)})
            row = {**identity, **stats, "source_conflict_threshold": reference["conflict_threshold"],
                   "source_dispersion_threshold": reference["dispersion_threshold"],
                   "source_anchor_correlation_mean": reference["source_anchor_correlation_mean"],
                   "source_conflict_rate": reference["source_conflict_rate"],
                   "conflict_excess": stats["cancellation_ratio"] - reference["conflict_threshold"],
                   "dispersion_excess": stats["dispersion"] - reference["dispersion_threshold"]}
            if error is not None:
                row["teacher_error"] = error
            sample_rows.append(row)
            for method in methods:
                packet = aggregate_anchor_evidence(
                    scores, weights, method, reference, anchor_ids=anchors,
                )
                method_packets[(split, method)].append((packet, error, top1, top2, data.get("label", [None] * len(data["logits"]))[index]))
    conflict_metrics = []
    for split in ("source_val", "target_val"):
        subset = [row for row in sample_rows if row["split"] == split and "teacher_error" in row]
        truth = [row["teacher_error"] for row in subset]
        for field in ("cancellation_ratio", "dispersion", "negative_strength"):
            auroc, auprc = _binary_metrics(truth, [row[field] for row in subset])
            conflict_metrics.append({"split": split, "score": field, "sample_count": len(subset),
                                     "error_auroc": auroc, "error_auprc": auprc})
    for (split, method), packets in method_packets.items():
        available = [row[0] for row in packets]
        evaluable = [row for row in packets if row[1] is not None]
        selected = [row for row in evaluable if row[0]["available"]]
        correct_pair = [int((row[0]["score"] >= 0 and int(row[4]) == row[2]) or
                            (row[0]["score"] < 0 and int(row[4]) == row[3])) for row in selected]
        ablations.append({
            "split": split, "method": method, "sample_count": len(packets),
            "coverage": evidence_coverage(available), "available_count": len(selected),
            "pair_accuracy": float(np.mean(correct_pair)) if correct_pair else np.nan,
            "wrong_teacher_rejected": sum(row[1] == 1 and not row[0]["available"] for row in evaluable),
            "correct_teacher_rejected": sum(row[1] == 0 and not row[0]["available"] for row in evaluable),
        })
    _write_csv(output / "X2_anchor_pair_scores.csv", anchor_rows)
    _write_csv(output / "X2_source_target_disagreement.csv", sample_rows)
    _write_csv(output / "X2_conflict_vs_error.csv", conflict_metrics)
    _write_csv(output / "X2_aggregation_ablation.csv", ablations)
    target_detection = [row for row in conflict_metrics if row["split"] == "target_val"]
    target_ablation = [row for row in ablations if row["split"] == "target_val"]
    (output / "X2_report.md").write_text(
        "# Anchor disagreement audit\n\n"
        "All thresholds and stability weights are estimated from Source only. "
        "Target Train is unlabeled and Target Val labels are used only for evaluation. "
        "Disagreement is reported relative to the Source reference; it is not assumed to be erroneous.\n\n"
        f"Target error-detection statistics: `{json.dumps(target_detection, sort_keys=True)}`.\n\n"
        f"Target aggregation/coverage statistics: `{json.dumps(target_ablation, sort_keys=True)}`.\n",
        encoding="utf-8",
    )


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--cache-dir", required=True); value.add_argument("--output-dir", required=True)
    value.add_argument("--task", required=True); value.add_argument("--variant", required=True)
    value.add_argument("--classes", nargs="+")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
