#!/usr/bin/env python3
"""07 Oracle True-Class Gamma Effectiveness and Legality Audit.

This is a post-hoc, zero-exact-DP diagnostic.  It reuses the 06 Stage-A cache
of oracle true-class gamma values and answers only:

1) Does a numerically usable oracle true-class gamma improve the frozen target
   classifier relative to No Shift and the already-estimated TimeMatch scalar?
2) After that effectiveness is measured, does the *current production* Phase
   legality mechanism keep the beneficial gamma values?

No registration solver is called here.  No grouping, Stable Label, Teacher,
student update, class-center or group-center mechanism is evaluated.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, REPOSITORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import compare_stage2_phase_vs_timematch_shift as scalarcmp
import diagnose_sample_level_phase_validity as samplediag
import visualize_stage2_phase_alignment as phasevis
from methods.structure_da.oracle_gamma_audit import (
    PRODUCTION_RULE_ORDER,
    audit_current_phase_legality,
    conditional_probability,
    rule_threshold_descriptions,
)
from methods.structure_da.sample_phase_diagnostic import (
    RawShapeValidation,
    TOnlyPhaseRegistration,
    evaluate_shape_validation,
)
from methods.structure_da.stage2_trainer import DeviceBatchLoader, build_stage2_registration_extractor
from methods.structure_da import target_hypothesis_scan as targetscan


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _rate(values: Iterable[bool]) -> float:
    values = [bool(v) for v in values]
    return float(np.mean(values)) if values else float("nan")


def _safe_div(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def _binary_f1(labels: np.ndarray, predictions: np.ndarray, class_id: int) -> float:
    tp = int(np.sum((labels == class_id) & (predictions == class_id)))
    fp = int(np.sum((labels != class_id) & (predictions == class_id)))
    fn = int(np.sum((labels == class_id) & (predictions != class_id)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if not math.isfinite(precision) or not math.isfinite(recall) or precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _macro_f1(labels: np.ndarray, predictions: np.ndarray, class_ids: Sequence[int]) -> float:
    return float(np.mean([_binary_f1(labels, predictions, int(c)) for c in class_ids]))


def _transition(before_correct: bool, after_correct: bool) -> str:
    if before_correct and after_correct:
        return "correct_to_correct"
    if before_correct and not after_correct:
        return "correct_to_wrong"
    if not before_correct and after_correct:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def _numerical_failure_reason(record: TOnlyPhaseRegistration) -> str:
    reasons: list[str] = []
    if record.solver_error is not None or record.gamma is None:
        reasons.append(f"solver_failed:{record.solver_error or 'missing_gamma'}")
        return "+".join(reasons)
    if not bool(record.gamma_finite):
        reasons.append("non_finite")
    if record.gamma_endpoint_error is None or float(record.gamma_endpoint_error) > 1e-6:
        reasons.append("endpoint")
    if not bool(record.gamma_strictly_increasing):
        reasons.append("not_strictly_increasing")
    return "+".join(reasons) if reasons else "numerically_invalid"


def _load_06_stage_a_cache(path: Path) -> tuple[tuple[TOnlyPhaseRegistration, ...], list[int], list[int]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"06 Stage-A exact-DP cache is required and will not be regenerated by 07: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != samplediag.CACHE_SCHEMA:
        raise ValueError("unsupported 06 Stage-A cache schema")
    sample_ids = [int(v) for v in payload.get("sample_ids", ())]
    true_classes = [int(v) for v in payload.get("true_classes", ())]
    records = tuple(samplediag._registration_from_payload(row) for row in payload.get("records", ()))
    if len(records) != len(sample_ids) or len(records) != len(true_classes):
        raise ValueError("06 Stage-A cache record/sample/class lengths do not match")
    for record, sample_id, true_class in zip(records, sample_ids, true_classes):
        if int(record.sample_id) != sample_id or int(record.class_id) != true_class:
            raise ValueError("06 Stage-A cache record identity does not match cache metadata")
    return records, sample_ids, true_classes


def _load_timematch_shift(path: Path, *, source: str, target: str, seed: int, fold: int, model_checkpoint: Path) -> tuple[int, dict]:
    if not path.is_file():
        raise FileNotFoundError(
            "07 requires the already-computed 04 TimeMatch scalar summary; it does not reselect a shift "
            f"from target-test labels: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in (("source", source), ("target", target), ("seed", seed), ("fold", fold)):
        if str(payload.get(key)) != str(expected):
            raise ValueError(f"TimeMatch summary {key} does not match 07 runtime: {payload.get(key)!r} != {expected!r}")
    manifest_model = payload.get("model_checkpoint")
    if manifest_model is None or Path(str(manifest_model)).name != model_checkpoint.name:
        raise ValueError(
            "TimeMatch manifest model checkpoint does not match 07 Stage-1 checkpoint: "
            f"{manifest_model!r} vs {str(model_checkpoint)!r}"
        )
    result = payload.get("timematch_shift_result")
    if not isinstance(result, dict) or "selected_shift_days" not in result:
        raise ValueError("TimeMatch manifest does not contain timematch_shift_result.selected_shift_days")
    return int(result["selected_shift_days"]), payload


def _remap_target_cache_to_parcels(cache, loader):
    dataset = loader.dataset
    if not hasattr(dataset, "get_parcel_indices"):
        raise TypeError("target-test dataset does not expose get_parcel_indices()")
    parcels = [int(v) for v in dataset.get_parcel_indices().tolist()]
    mapped = []
    for value in cache.sample_ids.detach().cpu().tolist():
        local = int(value)
        if not 0 <= local < len(parcels):
            raise IndexError("target geometry cache local sample index is outside target-test dataset")
        mapped.append(parcels[local])
    return replace(cache, sample_ids=torch.tensor(mapped, dtype=torch.long))


def _validate_cache_alignment(cache, records: Sequence[TOnlyPhaseRegistration]) -> None:
    if len(cache.sample_ids) != len(records):
        raise ValueError("rebuilt target geometry cache length differs from 06 Stage-A cache")
    for record in records:
        sample_index = int(record.sample_index)
        if not 0 <= sample_index < len(cache.sample_ids):
            raise IndexError("06 registration sample_index is outside rebuilt target geometry cache")
        parcel = int(cache.sample_ids[sample_index].item())
        if parcel != int(record.sample_id):
            raise ValueError(
                "06 Stage-A cache order does not match rebuilt target geometry cache: "
                f"sample_index={sample_index}, cached={record.sample_id}, rebuilt={parcel}"
            )


def _evaluate_three_way(
    *,
    model,
    target_loader,
    records: Sequence[TOnlyPhaseRegistration],
    shift_days: int,
    time_scale_days: float,
    device: torch.device,
) -> list[dict]:
    record_by_parcel = {int(record.sample_id): record for record in records}
    gamma_by_parcel = {
        int(record.sample_id): record.gamma
        for record in records
        if record.numerically_valid and isinstance(record.gamma, Tensor)
    }
    rows: list[dict] = []
    with scalarcmp._timematch_time_extrapolation(model):
        for raw_batch in target_loader:
            batch = phasevis._move_batch(raw_batch, device)
            backbone = model.forward_backbone(
                batch["pixels"], batch["valid_pixels"], batch["positions"],
                batch.get("extra"), time_mask=batch.get("time_mask"),
            )
            native = backbone.normalized_positions.detach()
            mask = backbone.time_mask.detach()
            labels = batch["label"].long()
            parcels = batch["parcel_index"].detach().cpu().long()

            no_output = model.forward_from_backbone(
                backbone, batch["positions"], batch.get("extra"), return_geometry=False
            )
            scalar_positions = scalarcmp._scalar_positions(backbone, shift_days, time_scale_days)
            scalar_output = model.forward_from_backbone(
                backbone, batch["positions"], batch.get("extra"),
                temporal_positions_override=scalar_positions, return_geometry=False,
            )
            oracle_positions, oracle_available = samplediag._individual_positions(
                native, mask, parcels, gamma_by_parcel
            )
            oracle_output = model.forward_from_backbone(
                backbone, batch["positions"], batch.get("extra"),
                temporal_positions_override=oracle_positions, return_geometry=False,
            )

            outputs = {
                "no": no_output.logits.float(),
                "timematch": scalar_output.logits.float(),
                "oracle": oracle_output.logits.float(),
            }
            probs = {key: torch.softmax(value, dim=-1) for key, value in outputs.items()}
            preds = {key: value.argmax(dim=-1) for key, value in outputs.items()}

            for row_index, parcel in enumerate(parcels.tolist()):
                parcel = int(parcel)
                record = record_by_parcel.get(parcel)
                if record is None:
                    raise KeyError(f"target-test parcel {parcel} is missing from 06 Stage-A cache")
                true_class = int(labels[row_index].item())
                if true_class != int(record.class_id):
                    raise ValueError("target-test true label differs from 06 cached true class")
                numerically_valid = bool(record.numerically_valid and isinstance(record.gamma, Tensor))
                available = bool(oracle_available[row_index].item())
                if numerically_valid != available:
                    raise RuntimeError("oracle gamma numerical-validity and temporal-position availability disagree")

                row = {
                    "sample_id": parcel,
                    "true_class": true_class,
                    "gamma_numerically_valid": numerically_valid,
                    "numerical_failure_reason": "" if numerically_valid else _numerical_failure_reason(record),
                    "pred_no": int(preds["no"][row_index].item()),
                    "pred_timematch": int(preds["timematch"][row_index].item()),
                    "true_prob_no": float(probs["no"][row_index, true_class].item()),
                    "true_prob_timematch": float(probs["timematch"][row_index, true_class].item()),
                    "no_correct": bool(int(preds["no"][row_index].item()) == true_class),
                    "timematch_correct": bool(int(preds["timematch"][row_index].item()) == true_class),
                }
                if numerically_valid:
                    row.update({
                        "pred_oracle_gamma": int(preds["oracle"][row_index].item()),
                        "true_prob_oracle_gamma": float(probs["oracle"][row_index, true_class].item()),
                        "oracle_correct": bool(int(preds["oracle"][row_index].item()) == true_class),
                    })
                    row["delta_prob_vs_no"] = row["true_prob_oracle_gamma"] - row["true_prob_no"]
                    row["delta_prob_vs_timematch"] = row["true_prob_oracle_gamma"] - row["true_prob_timematch"]
                    row["beneficial_vs_no"] = bool(row["delta_prob_vs_no"] > 0.0)
                    row["beneficial_vs_timematch"] = bool(row["delta_prob_vs_timematch"] > 0.0)
                    row["transition_vs_no"] = _transition(row["no_correct"], row["oracle_correct"])
                    row["transition_vs_timematch"] = _transition(row["timematch_correct"], row["oracle_correct"])
                else:
                    row.update({
                        "pred_oracle_gamma": "",
                        "true_prob_oracle_gamma": float("nan"),
                        "oracle_correct": "",
                        "delta_prob_vs_no": float("nan"),
                        "delta_prob_vs_timematch": float("nan"),
                        "beneficial_vs_no": "",
                        "beneficial_vs_timematch": "",
                        "transition_vs_no": "",
                        "transition_vs_timematch": "",
                    })
                rows.append(row)
    return rows


def _attach_legality(
    rows: list[dict],
    *,
    records: Sequence[TOnlyPhaseRegistration],
    target_cache,
    source_bank,
    scan_config,
) -> dict[int, object]:
    row_by_parcel = {int(row["sample_id"]): row for row in rows}
    decisions = {}
    thresholds = rule_threshold_descriptions(scan_config)
    for record in records:
        row = row_by_parcel[int(record.sample_id)]
        row.update({
            "06_t_only_legal_diagnostic_only": bool(record.t_only_legal),
            "06_t_only_reject_reasons_diagnostic_only": "+".join(record.reject_reasons),
            "target_trend_valid_diagnostic_only": bool(record.target_trend_valid),
            "gamma_finite": bool(record.gamma_finite),
            "gamma_endpoint_error": record.gamma_endpoint_error,
            "gamma_strictly_increasing": bool(record.gamma_strictly_increasing),
            "gamma_min_increment": record.gamma_min_increment,
            "gamma_max_local_speed": record.gamma_max_local_speed,
            "gamma_roughness": record.gamma_roughness,
            "phase_deviation": record.phase_deviation,
            "pre_common_support_t": record.pre_common_support_t,
            "common_support_t": record.common_support_t,
            "t_identity_error": record.t_identity_error,
            "t_registered_error": record.t_registered_error,
            "t_gain_ratio": record.t_gain_ratio,
            "threshold_numerical_endpoint_error_max": 1e-6,
            "threshold_registration_min_common_support": float(scan_config.registration_min_common_support),
            "threshold_registration_min_increment": float(scan_config.registration_min_increment),
            "threshold_registration_max_local_speed": float(scan_config.registration_max_local_speed),
            "threshold_registration_max_roughness": float(scan_config.registration_max_roughness),
            "threshold_registration_max_deviation": float(scan_config.registration_max_deviation),
            "threshold_registration_gain_ratio_max": float(scan_config.registration_gain_ratio_max),
        })
        if not record.numerically_valid or record.gamma is None:
            row.update({
                "shape_support_valid": "",
                "shape_common_support": float("nan"),
                "q_distance_percentile": float("nan"),
                "q_cdf_available": "",
                "current_legality_accepted": "",
                "current_legality_reject_reason": "",
            })
            continue
        shape = evaluate_shape_validation(record, target_cache=target_cache, source_bank=source_bank)
        decision = audit_current_phase_legality(record, shape, scan_config)
        decisions[int(record.sample_id)] = decision
        row.update({
            "shape_support_valid": bool(decision.shape_support_valid),
            "shape_common_support": shape.common_support_shape,
            "q_distance_percentile": shape.q_distance_percentile,
            "q_cdf_available": bool(decision.q_cdf_available),
            "current_legality_accepted": bool(decision.accepted),
            "current_legality_reject_reason": "+".join(decision.reject_reasons),
        })
        for rule in PRODUCTION_RULE_ORDER:
            row[f"rejected_by_{rule}"] = bool(rule in decision.reject_reasons)
        for rule, description in thresholds.items():
            row[f"rule_{rule}_definition"] = description
    return decisions


def _summary_rows(rows: Sequence[dict], classes: Sequence[str]) -> list[dict]:
    valid = [row for row in rows if row["gamma_numerically_valid"]]
    valid_labels = np.asarray([int(row["true_class"]) for row in valid], dtype=np.int64)
    pred_no = np.asarray([int(row["pred_no"]) for row in valid], dtype=np.int64)
    pred_tm = np.asarray([int(row["pred_timematch"]) for row in valid], dtype=np.int64)
    pred_oracle = np.asarray([int(row["pred_oracle_gamma"]) for row in valid], dtype=np.int64)
    class_ids = list(range(len(classes)))

    def one(scope_rows: Sequence[dict], class_id: int | None, class_name: str) -> dict:
        scope = list(scope_rows)
        accepted = [bool(row["current_legality_accepted"]) for row in scope]
        rejected = [not value for value in accepted]
        ben_no = [bool(row["beneficial_vs_no"]) for row in scope]
        ben_tm = [bool(row["beneficial_vs_timematch"]) for row in scope]
        accepted_rows = [row for row in scope if bool(row["current_legality_accepted"])]
        rejected_rows = [row for row in scope if not bool(row["current_legality_accepted"])]
        reason_counts = {rule: 0 for rule in PRODUCTION_RULE_ORDER}
        for row in rejected_rows:
            for rule in PRODUCTION_RULE_ORDER:
                reason_counts[rule] += int(bool(row.get(f"rejected_by_{rule}", False)))
        primary_reason = max(reason_counts, key=reason_counts.get) if any(reason_counts.values()) else ""
        if class_id is None:
            labels = valid_labels
            pno, ptm, por = pred_no, pred_tm, pred_oracle
            no_metric = float((pno == labels).mean()) if labels.size else float("nan")
            tm_metric = float((ptm == labels).mean()) if labels.size else float("nan")
            or_metric = float((por == labels).mean()) if labels.size else float("nan")
            no_f1 = _macro_f1(labels, pno, class_ids) if labels.size else float("nan")
            tm_f1 = _macro_f1(labels, ptm, class_ids) if labels.size else float("nan")
            or_f1 = _macro_f1(labels, por, class_ids) if labels.size else float("nan")
        else:
            labels = valid_labels
            no_metric = _safe_div(sum(bool(row["no_correct"]) for row in scope), len(scope))
            tm_metric = _safe_div(sum(bool(row["timematch_correct"]) for row in scope), len(scope))
            or_metric = _safe_div(sum(bool(row["oracle_correct"]) for row in scope), len(scope))
            no_f1 = _binary_f1(labels, pred_no, class_id)
            tm_f1 = _binary_f1(labels, pred_tm, class_id)
            or_f1 = _binary_f1(labels, pred_oracle, class_id)
        return {
            "class_id": "ALL" if class_id is None else class_id,
            "class_name": class_name,
            "n_samples": len(rows) if class_id is None else sum(int(r["true_class"]) == class_id for r in rows),
            "n_numerically_valid": len(scope),
            "no_shift_acc_or_recall": no_metric,
            "timematch_acc_or_recall": tm_metric,
            "oracle_gamma_acc_or_recall": or_metric,
            "no_shift_f1": no_f1,
            "timematch_f1": tm_f1,
            "oracle_gamma_f1": or_f1,
            "no_shift_true_prob": _mean(row["true_prob_no"] for row in scope),
            "timematch_true_prob": _mean(row["true_prob_timematch"] for row in scope),
            "oracle_gamma_true_prob": _mean(row["true_prob_oracle_gamma"] for row in scope),
            "oracle_vs_no_true_prob_gain": _mean(row["delta_prob_vs_no"] for row in scope),
            "oracle_vs_timematch_true_prob_gain": _mean(row["delta_prob_vs_timematch"] for row in scope),
            "oracle_vs_no_beneficial_rate": _rate(ben_no),
            "oracle_vs_timematch_beneficial_rate": _rate(ben_tm),
            "accepted_count": sum(accepted),
            "accepted_rate": _rate(accepted),
            "rejected_count": sum(rejected),
            "rejected_rate": _rate(rejected),
            "primary_rejection_reason": primary_reason,
            "accepted_beneficial_rate_vs_no": _rate(bool(row["beneficial_vs_no"]) for row in accepted_rows),
            "rejected_beneficial_rate_vs_no": _rate(bool(row["beneficial_vs_no"]) for row in rejected_rows),
            "accepted_beneficial_rate_vs_timematch": _rate(bool(row["beneficial_vs_timematch"]) for row in accepted_rows),
            "rejected_beneficial_rate_vs_timematch": _rate(bool(row["beneficial_vs_timematch"]) for row in rejected_rows),
            "p_accepted_given_beneficial_no": conditional_probability(accepted, ben_no),
            "p_beneficial_no_given_accepted": conditional_probability(ben_no, accepted),
            "p_accepted_given_beneficial_timematch": conditional_probability(accepted, ben_tm),
            "p_beneficial_timematch_given_accepted": conditional_probability(ben_tm, accepted),
        }

    output = [one(valid, None, "ALL")]
    for class_id, class_name in enumerate(classes):
        scope = [row for row in valid if int(row["true_class"]) == class_id]
        output.append(one(scope, class_id, class_name))
    return output


def _partition_rows(rows: Sequence[dict]) -> list[dict]:
    valid = [row for row in rows if row["gamma_numerically_valid"]]
    output = []
    for name, accepted_value in (("accepted", True), ("rejected", False)):
        subset = [row for row in valid if bool(row["current_legality_accepted"]) == accepted_value]
        output.append({
            "partition": name,
            "count": len(subset),
            "no_shift_accuracy": _rate(bool(row["no_correct"]) for row in subset),
            "timematch_accuracy": _rate(bool(row["timematch_correct"]) for row in subset),
            "oracle_gamma_accuracy": _rate(bool(row["oracle_correct"]) for row in subset),
            "mean_delta_prob_vs_no": _mean(row["delta_prob_vs_no"] for row in subset),
            "mean_delta_prob_vs_timematch": _mean(row["delta_prob_vs_timematch"] for row in subset),
            "beneficial_rate_vs_no": _rate(bool(row["beneficial_vs_no"]) for row in subset),
            "beneficial_rate_vs_timematch": _rate(bool(row["beneficial_vs_timematch"]) for row in subset),
            "wrong_to_correct_vs_no": sum(row["transition_vs_no"] == "wrong_to_correct" for row in subset),
            "correct_to_wrong_vs_no": sum(row["transition_vs_no"] == "correct_to_wrong" for row in subset),
            "wrong_to_correct_vs_timematch": sum(row["transition_vs_timematch"] == "wrong_to_correct" for row in subset),
            "correct_to_wrong_vs_timematch": sum(row["transition_vs_timematch"] == "correct_to_wrong" for row in subset),
        })
    return output


def _rule_audit_rows(rows: Sequence[dict], scan_config) -> list[dict]:
    valid = [row for row in rows if row["gamma_numerically_valid"]]
    definitions = rule_threshold_descriptions(scan_config)
    output = []
    for rule in PRODUCTION_RULE_ORDER:
        subset = [row for row in valid if bool(row.get(f"rejected_by_{rule}", False))]
        output.append({
            "rule": rule,
            "current_definition": definitions[rule],
            "rejected_count": len(subset),
            "rejected_fraction_of_numerically_valid": _safe_div(len(subset), len(valid)),
            "mean_delta_prob_vs_no": _mean(row["delta_prob_vs_no"] for row in subset),
            "mean_delta_prob_vs_timematch": _mean(row["delta_prob_vs_timematch"] for row in subset),
            "beneficial_rate_vs_no": _rate(bool(row["beneficial_vs_no"]) for row in subset),
            "beneficial_rate_vs_timematch": _rate(bool(row["beneficial_vs_timematch"]) for row in subset),
            "wrong_to_correct_vs_no": sum(row["transition_vs_no"] == "wrong_to_correct" for row in subset),
            "correct_to_wrong_vs_no": sum(row["transition_vs_no"] == "correct_to_wrong" for row in subset),
            "wrong_to_correct_vs_timematch": sum(row["transition_vs_timematch"] == "wrong_to_correct" for row in subset),
            "correct_to_wrong_vs_timematch": sum(row["transition_vs_timematch"] == "correct_to_wrong" for row in subset),
        })
    return output


def _transition_rows(rows: Sequence[dict]) -> list[dict]:
    valid = [row for row in rows if row["gamma_numerically_valid"]]
    output = []
    for baseline, field in (("no_shift", "transition_vs_no"), ("timematch_scalar", "transition_vs_timematch")):
        for transition in ("wrong_to_correct", "correct_to_correct", "correct_to_wrong", "wrong_to_wrong"):
            subset = [row for row in valid if row[field] == transition]
            output.append({
                "baseline": baseline,
                "transition": transition,
                "count": len(subset),
                "fraction": _safe_div(len(subset), len(valid)),
            })
    return output


def _plot_overall(path: Path, summary_rows: Sequence[dict], dpi: int) -> None:
    row = summary_rows[0]
    labels = ["No Shift", "TimeMatch scalar", "Oracle true-class gamma"]
    acc = [row["no_shift_acc_or_recall"], row["timematch_acc_or_recall"], row["oracle_gamma_acc_or_recall"]]
    f1 = [row["no_shift_f1"], row["timematch_f1"], row["oracle_gamma_f1"]]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.36
    ax.bar(x - width / 2, acc, width, label="Accuracy")
    ax.bar(x + width / 2, f1, width, label="Macro-F1")
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("07 Oracle gamma effectiveness on numerically-valid target-test samples")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_class_recall(path: Path, summary_rows: Sequence[dict], dpi: int) -> None:
    rows = list(summary_rows[1:])
    x = np.arange(len(rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(x - width, [r["no_shift_acc_or_recall"] for r in rows], width, label="No Shift")
    ax.bar(x, [r["timematch_acc_or_recall"] for r in rows], width, label="TimeMatch scalar")
    ax.bar(x + width, [r["oracle_gamma_acc_or_recall"] for r in rows], width, label="Oracle gamma")
    ax.set_xticks(x, [r["class_name"] for r in rows], rotation=40, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Recall")
    ax.set_title("Per-class recall on numerically-valid oracle gamma population")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _write_readme(path: Path, *, shift_days: int, cache_path: Path, timematch_manifest: Path) -> None:
    text = f"""# 07 Oracle True-Class Gamma Effectiveness and Legality Audit

## 实验只回答两个问题

1. 已知 target 真实类别、与正确 source 类别原型配准得到的 oracle true-class gamma，是否能改善冻结分类器；
2. 在效果已经测完以后，当前生产代码的 Phase legality 是否能够保留真正有分类收益的 gamma。

本实验不重新求 gamma，不训练任何参数，不修改 Domain Phase 分组，不使用 Stable Label/Teacher，也不设计新的类别选择机制。

## 复用输入

- 06 Stage-A exact-DP cache：`{cache_path}`
- 04 TimeMatch manifest：`{timematch_manifest}`
- 复用的无监督 TimeMatch scalar shift：`{shift_days:+d}` 天

07 **没有 registration solver fallback**。如果 06 cache 不存在，脚本直接失败，不会重复 exact-DP。

## 问题 1：gamma 本身是否有效

主比较只在 `gamma_numerically_valid=true` 的 held-out target-test 样本上进行：

- solver 成功返回 gamma；
- gamma 为有限数；
- endpoint error <= 1e-6；
- gamma 严格单调递增。

在这一步之前不应用 registration gain、support、roughness、speed、deviation 或 S-SRVF 等当前 Phase 可靠性阈值。三路固定模型比较为：

1. No Shift；
2. 04 已经通过 target-train Inception Score 无监督选出的 TimeMatch scalar shift；
3. Oracle true-class gamma：`gamma_{{i,y_i}}^(-1)(t_i)`。

所有 PSE / decomposition / Time2Vec / LTAE / classifier 参数冻结且 `eval()`。

## 问题 2：当前生产 legality 是否筛得好

只有问题 1 的分类结果已经记录后，才对同一批 numerically-valid gamma 事后重放当前 `TargetPhaseHypothesisScanner.phase_evidence_eligible` 的实际条件：

- pre-registration T common support；
- gamma minimum increment；
- gamma maximum local speed；
- gamma roughness；
- gamma maximum deviation；
- T registration gain ratio；
- 对齐后 S-SRVF support 是否可计算；
- source class S-distance empirical CDF 是否可用。

代码审计特别确认：

- 06 的 `t_only_legal` **不能**直接当作当前生产 legality：06 为保持 Stage A 独立，故意不使用 S，而且还额外记录 target-T validity；
- 当前生产 scanner 的 source q95 / outer-range 只做 diagnostic，不是 Phase evidence eligibility gate，因此 07 不把它算作 rejection；
- `target_trend_valid` 在 07 中只作为诊断字段导出，不擅自增加为生产 legality gate。

## 主要输出

- `oracle_gamma_effectiveness_summary.csv`：每类 + `ALL`，No / TimeMatch / Oracle 三路分类结果及两个条件概率；
- `oracle_gamma_sample_level.csv`：逐样本数值有效性、三路预测/真类概率、收益、correct/wrong 转移、所有当前 legality 指标与阈值；
- `legality_rule_audit.csv`：每一条当前拒绝规则排掉多少 gamma，以及这些被拒 gamma 的真实分类收益；
- `legality_partition_summary.csv`：accepted 与 rejected gamma 的直接分类对照；
- `classification_transitions.csv`：Oracle gamma 相对 No Shift / TimeMatch 的四种硬分类状态变化；
- `overall_classification_comparison.png`：总体 Accuracy / Macro-F1；
- `per_class_recall_comparison.png`：逐类别 Recall；
- `summary.json` / `manifest.json`：机器可读协议与结果。

## 解释边界

实验结束后只允许回答：

- oracle true-class gamma 是否值得继续追求，尤其是否稳定优于 TimeMatch scalar；
- 当前 production legality 对 beneficial gamma 的保留率和纯度是否足够，以及具体哪条规则在漏掉/接受哪些 gamma。

本实验不讨论 Phase 分组、class/group center、S 如何选类别、Stable Label、Teacher 或新的训练机制。
"""
    path.write_text(text, encoding="utf-8")


def run(args) -> dict:
    calibration = torch.load(args.calibration_checkpoint.resolve(), map_location="cpu", weights_only=False)
    runtime = calibration.get("runtime_config")
    if not isinstance(runtime, dict):
        raise ValueError("calibration checkpoint is missing runtime_config")
    classes = [str(v) for v in runtime["classes"]]
    source = str(runtime["source"])
    target = str(runtime["target"])
    seed = int(runtime["seed"])
    fold = int(args.fold)
    data_root = str(args.data_root or runtime["data_root"])
    closed_set = bool(runtime.get("closed_set", True))
    combine = bool(runtime.get("combine_spring_and_winter", False))
    time_mode = str(runtime.get("time_coordinate_mode", "canonical_day_of_year"))
    val_ratio = float(runtime.get("val_ratio", 0.1))
    test_ratio = float(runtime.get("test_ratio", 0.2))
    time_scale_days = float(runtime.get("time_scale", 365.0))
    scan_config = samplediag._scan_config(runtime, workers=1)

    records, cache_sample_ids, cache_true_classes = _load_06_stage_a_cache(args.stage_a_cache.resolve())
    selected_shift, _timematch_payload = _load_timematch_shift(
        args.timematch_manifest.resolve(), source=source, target=target, seed=seed, fold=fold,
        model_checkpoint=args.model_checkpoint.resolve(),
    )

    device = torch.device(args.device)
    model_checkpoint = torch.load(args.model_checkpoint.resolve(), map_location="cpu", weights_only=False)
    model = phasevis._build_model(runtime, calibration, device, model_checkpoint=model_checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    source_all = phasevis._eligible_parcels(
        data_root, source, classes, closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
    )
    target_all = phasevis._eligible_parcels(
        data_root, target, classes, closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
    )
    splits = phasevis._reconstruct_fold_splits(
        source_all, target_all, source=source, target=target, seed=seed,
        val_ratio=val_ratio, test_ratio=test_ratio, fold=fold,
    )
    target_test_parcels = np.asarray(sorted(splits[target]["test"]), dtype=np.int64)
    target_test_loader = phasevis._selected_loader(
        data_root, target, classes, target_test_parcels,
        closed_set=closed_set, combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    target_meta = phasevis._metadata_dataset(
        data_root, target, classes, splits[target]["test"], closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
    )
    label_by_parcel = samplediag._label_map(target_meta)
    if set(cache_sample_ids) != set(label_by_parcel):
        missing = sorted(set(label_by_parcel).difference(cache_sample_ids))[:5]
        extra = sorted(set(cache_sample_ids).difference(label_by_parcel))[:5]
        raise ValueError(
            "06 Stage-A cache does not match current held-out target-test parcel population: "
            f"missing={missing}, extra={extra}"
        )
    for sample_id, true_class in zip(cache_sample_ids, cache_true_classes):
        if int(label_by_parcel[sample_id]) != int(true_class):
            raise ValueError("06 Stage-A cache true class differs from reconstructed target-test metadata")

    print(
        "ORACLE_GAMMA_07_EFFECTIVENESS_START|"
        f"target_test={len(cache_sample_ids)}|numerically_valid={sum(r.numerically_valid for r in records)}|"
        f"timematch_shift={selected_shift:+d}|exact_dp_calls=0",
        flush=True,
    )
    sample_rows = _evaluate_three_way(
        model=model, target_loader=target_test_loader, records=records,
        shift_days=selected_shift, time_scale_days=time_scale_days, device=device,
    )
    print("ORACLE_GAMMA_07_EFFECTIVENESS_READY|status=ready", flush=True)

    # Only after classifier effectiveness is recorded do we rebuild geometry
    # needed to audit the *current* production legality rules.
    reg_extractor = build_stage2_registration_extractor(model, device=device, k_reg=scan_config.k_reg)
    target_cache = targetscan._build_target_geometry_cache(
        model, DeviceBatchLoader(target_test_loader, device), device=device,
        shape_grid=model.temporal_module.structure_geometry.functional_lift.canonical_grid.detach().cpu(),
        shape_extractor=model.temporal_module.structure_geometry,
        reg_extractor=reg_extractor,
    )
    target_cache = _remap_target_cache_to_parcels(target_cache, target_test_loader)
    _validate_cache_alignment(target_cache, records)
    source_bank = samplediag._source_bank(calibration)
    _attach_legality(
        sample_rows, records=records, target_cache=target_cache,
        source_bank=source_bank, scan_config=scan_config,
    )
    print("ORACLE_GAMMA_07_LEGALITY_READY|status=ready", flush=True)

    summary_rows = _summary_rows(sample_rows, classes)
    partition_rows = _partition_rows(sample_rows)
    rule_rows = _rule_audit_rows(sample_rows, scan_config)
    transition_rows = _transition_rows(sample_rows)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "oracle_gamma_sample_level.csv", sample_rows)
    _write_csv(output_dir / "oracle_gamma_effectiveness_summary.csv", summary_rows)
    _write_csv(output_dir / "legality_partition_summary.csv", partition_rows)
    _write_csv(output_dir / "legality_rule_audit.csv", rule_rows)
    _write_csv(output_dir / "classification_transitions.csv", transition_rows)
    _plot_overall(output_dir / "overall_classification_comparison.png", summary_rows, args.dpi)
    _plot_class_recall(output_dir / "per_class_recall_comparison.png", summary_rows, args.dpi)

    all_row = summary_rows[0]
    result = {
        "protocol": "07_oracle_true_class_gamma_effectiveness_and_legality_audit",
        "source": source,
        "target": target,
        "seed": seed,
        "fold": fold,
        "target_test_samples": len(sample_rows),
        "numerically_valid_gamma": int(sum(bool(row["gamma_numerically_valid"]) for row in sample_rows)),
        "exact_dp_calls": 0,
        "stage_a_cache": str(args.stage_a_cache.resolve()),
        "timematch_manifest": str(args.timematch_manifest.resolve()),
        "timematch_selected_shift_days": selected_shift,
        "timematch_selection_source": "reused 04 target-train Inception-Score estimate; target-test labels not used",
        "question_1": {
            "no_shift_accuracy": all_row["no_shift_acc_or_recall"],
            "no_shift_macro_f1": all_row["no_shift_f1"],
            "timematch_accuracy": all_row["timematch_acc_or_recall"],
            "timematch_macro_f1": all_row["timematch_f1"],
            "oracle_gamma_accuracy": all_row["oracle_gamma_acc_or_recall"],
            "oracle_gamma_macro_f1": all_row["oracle_gamma_f1"],
            "oracle_vs_no_mean_true_prob_gain": all_row["oracle_vs_no_true_prob_gain"],
            "oracle_vs_timematch_mean_true_prob_gain": all_row["oracle_vs_timematch_true_prob_gain"],
            "oracle_vs_no_beneficial_rate": all_row["oracle_vs_no_beneficial_rate"],
            "oracle_vs_timematch_beneficial_rate": all_row["oracle_vs_timematch_beneficial_rate"],
        },
        "question_2": {
            "accepted_count": all_row["accepted_count"],
            "accepted_rate": all_row["accepted_rate"],
            "p_accepted_given_beneficial_no": all_row["p_accepted_given_beneficial_no"],
            "p_beneficial_no_given_accepted": all_row["p_beneficial_no_given_accepted"],
            "p_accepted_given_beneficial_timematch": all_row["p_accepted_given_beneficial_timematch"],
            "p_beneficial_timematch_given_accepted": all_row["p_beneficial_timematch_given_accepted"],
            "production_rule_order": list(PRODUCTION_RULE_ORDER),
            "q95_outer_range_is_gate": False,
            "target_trend_valid_is_production_gate": False,
            "06_t_only_legal_reused_as_production_legality": False,
        },
        "class_summary": summary_rows[1:],
        "legality_partition_summary": partition_rows,
        "legality_rule_audit": rule_rows,
    }
    _json_dump(output_dir / "summary.json", result)
    _json_dump(output_dir / "manifest.json", {
        "oracle_only_target_label_usage": "true label specifies the already-cached correct source prototype gamma and is used for post-hoc metrics only",
        "training_updates": False,
        "registration_solver_called": False,
        "domain_phase_grouping_modified": False,
        "stable_label_used": False,
        "teacher_used": False,
        "class_center_used": False,
        "group_center_used": False,
        "new_selection_mechanism_used": False,
        "effectiveness_evaluated_before_legality": True,
    })
    _write_readme(
        output_dir / "README_中文说明.md", shift_days=selected_shift,
        cache_path=args.stage_a_cache.resolve(), timematch_manifest=args.timematch_manifest.resolve(),
    )
    print(
        "ORACLE_GAMMA_07_DONE|"
        f"valid={result['numerically_valid_gamma']}|"
        f"acc_no={all_row['no_shift_acc_or_recall']:.4f}|"
        f"acc_tm={all_row['timematch_acc_or_recall']:.4f}|"
        f"acc_oracle={all_row['oracle_gamma_acc_or_recall']:.4f}|"
        f"accepted_rate={all_row['accepted_rate']:.4f}|exact_dp_calls=0",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-checkpoint", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-a-cache", type=Path, required=True)
    parser.add_argument("--timematch-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
