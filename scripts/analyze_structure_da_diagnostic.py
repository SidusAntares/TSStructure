#!/usr/bin/env python3
"""Summarize one V3 diagnostic run from its existing text log and outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any


THRESHOLDS = {
    "state_saturation_rate": 0.98,
    "candidate_collapse_rate": 0.95,
    "class_failure_rate": 0.90,
    "shape_valid_min": 0.05,
    "teacher_acceptance_min": 0.01,
    "teacher_acceptance_max": 0.99,
    "alpha_saturation_low": 0.02,
    "alpha_saturation_high": 0.98,
    "optimizer_skip_max": 0.20,
    "source_validation_drop": 0.20,
}
NOT_AVAILABLE = "NOT_AVAILABLE_IN_CURRENT_LOGS"


def _number(value: str) -> float | str:
    try:
        number = float(value)
    except ValueError:
        return value
    return number if math.isfinite(number) else value


def _identifier(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _pipe_record(line: str) -> tuple[str, dict[str, Any]] | None:
    if "|" not in line:
        return None
    pieces = line.strip().split("|")
    fields: dict[str, Any] = {}
    for piece in pieces[1:]:
        if "=" in piece:
            key, value = piece.split("=", 1)
            fields[key] = _number(value)
    return pieces[0], fields


def _last(records: list[dict[str, Any]], *names: str) -> Any:
    for record in reversed(records):
        for name in names:
            if name in record:
                return record[name]
    return NOT_AVAILABLE


def _series(records: list[dict[str, Any]], *names: str) -> list[float]:
    result: list[float] = []
    for record in records:
        value = next((record[name] for name in names if name in record), None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def _weighted_class_metric(
    classes: dict[str, dict[str, Any]], *names: str
) -> float | str:
    numerator = 0.0
    denominator = 0.0
    for record in classes.values():
        count = record.get("sample_count")
        value = next((record[name] for name in names if name in record), None)
        if isinstance(count, (int, float)) and isinstance(value, (int, float)):
            if math.isfinite(float(count)) and math.isfinite(float(value)) and count > 0:
                numerator += float(count) * float(value)
                denominator += float(count)
    return numerator / denominator if denominator > 0 else NOT_AVAILABLE


def _target_metrics(run_directory: Path) -> dict[str, Any]:
    files = sorted((run_directory / "fold_0").glob("test_metrics_*.json"))
    if not files:
        return {}
    return json.loads(files[-1].read_text(encoding="utf-8"))


def analyze_run(run_directory: Path) -> dict[str, Any]:
    run_directory = Path(run_directory)
    log_path = run_directory / "train.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    stderr_path = run_directory / "stderr.log"
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    phase_classes: dict[str, dict[str, list[dict[str, Any]]]] = {
        "source": defaultdict(list), "target": defaultdict(list)
    }
    train_steps: list[dict[str, Any]] = []
    train_epochs: list[dict[str, Any]] = []
    for line in text.splitlines():
        parsed = _pipe_record(line)
        if parsed is None:
            continue
        kind, fields = parsed
        if kind == "TRAIN_STEP":
            train_steps.append(fields)
        elif kind == "TRAIN_EPOCH":
            train_epochs.append(fields)
        elif kind == "PHASE_EPOCH":
            phase[str(fields.get("domain", "unknown"))].append(fields)
        elif kind == "PHASE_CLASS_EPOCH":
            split = str(fields.get("split", ""))
            label_source = str(fields.get("label_source", ""))
            if split == "target" and label_source != "target_geometry_pseudo":
                raise ValueError("target class diagnostics must use target_geometry_pseudo")
            if split in phase_classes:
                phase_classes[split][_identifier(fields.get("class_id", "unknown"))].append(fields)

    validation = [
        {"loss": float(loss), "accuracy": float(acc), "macro_f1": float(f1)}
        for loss, acc, f1 in re.findall(
            r"Validation result: loss=([0-9.eE+-]+), acc=([0-9.eE+-]+), f1=([0-9.eE+-]+)",
            text,
        )
    ]
    target_metrics = _target_metrics(run_directory)
    source_f1 = [record["macro_f1"] for record in validation]
    source_drop = (
        max(source_f1) - source_f1[-1] if source_f1 else NOT_AVAILABLE
    )
    source_phase = phase.get("source", [])
    target_phase = phase.get("target", [])
    class_summary = {
        split: {
            class_id: records[-1]
            for class_id, records in classes.items()
            if records
        }
        for split, classes in phase_classes.items()
    }
    source_classes = class_summary["source"]
    phase_summary = {
        "base_valid_rate": _weighted_class_metric(source_classes, "phase_base_valid_rate"),
        "identity_rate": _last(source_phase, "valid_identity_rate", "identity_rate"),
        "nonidentity_rate": _last(source_phase, "valid_nonidentity_rate", "nonidentity_rate"),
        "failure_rate": _last(source_phase, "failure_rate"),
        "candidate_trainable_rate": _last(source_phase, "candidate_trainable_rate"),
        "candidate_acceptable_rate": _last(source_phase, "candidate_acceptable_rate"),
        "candidate_collapse_rate": _last(source_phase, "candidate_collapse_rate"),
        "candidate_unique_count": _last(source_phase, "candidate_unique_count_mean", "candidate_unique_count"),
        "phase_magnitude_mean": _weighted_class_metric(source_classes, "phase_magnitude_mean"),
        "phase_magnitude_p95": _weighted_class_metric(source_classes, "phase_magnitude_p95"),
        "accepted_warp_shift": _weighted_class_metric(source_classes, "accepted_warp_shift_mean"),
        "shape_valid_rate": _last(source_phase, "shape_valid_rate", "S_shape_valid_rate"),
        "S_changed_preferred_rate": _last(source_phase, "S_changed_preferred_rate"),
        "S_veto_rate": _last(source_phase, "S_veto_rate"),
        "target_identity_rate": _last(target_phase, "valid_identity_rate", "identity_rate"),
        "target_failure_rate": _last(target_phase, "failure_rate"),
    }
    optimization = {
        "task_optimizer_success_rate": _last(train_epochs, "task_step_success_rate"),
        "optimizer_skip_rate": _last(train_epochs, "task_step_skip_rate", "task_optimizer_skip_rate"),
        "source_state_update_rate": _last(train_epochs, "source_state_update_rate"),
        "loss_nonfinite": bool(re.search(r"(?:loss|total|task|geometry|quality)=(?:nan|[+-]?inf)", text + stderr, re.I)),
        "amp_overflow": "overflow" in (text + stderr).casefold(),
        "oom": bool(re.search(r"out of memory|cuda oom", text + stderr, re.I)),
    }
    all_class_records = [record for classes in class_summary.values() for record in classes.values()]
    quality = {
        "alpha_trend_mean": _last(all_class_records, "alpha_trend_mean"),
        "alpha_structure_mean": _last(all_class_records, "alpha_structure_mean"),
    }
    shape_teacher = {
        "shape_valid_rate": phase_summary["shape_valid_rate"],
        "source_prototype_ready": _last(all_class_records, "shape_geometry_prototype_ready_rate"),
        "radius_ready": _last(all_class_records, "radius_ready_rate"),
        "target_teacher_valid": _last(train_steps, "teacher_rate"),
        "teacher_inner": _last(all_class_records, "geometry_teacher_inner_rate"),
        "teacher_middle": _last(all_class_records, "geometry_teacher_middle_rate"),
        "teacher_outer": _last(all_class_records, "geometry_teacher_outer_rate"),
        "q_z_consistency": _last(all_class_records, "q_z_consistency_rate"),
        "q_T_consistency": _last(all_class_records, "q_T_consistency_rate"),
        "q_S_consistency": _last(all_class_records, "q_S_consistency_rate"),
        "student_pull_enabled": _last(all_class_records, "student_pull_enabled_rate"),
    }
    failed_reasons: list[str] = []
    review_reasons: list[str] = []
    checkpoint = run_directory / "fold_0" / "model.pt"
    if not text or not train_epochs or not validation or not source_phase:
        failed_reasons.append("required_log_fields_missing")
    if not checkpoint.is_file():
        failed_reasons.append("checkpoint_missing")
    if not target_metrics:
        failed_reasons.append("target_metrics_missing")
    if optimization["loss_nonfinite"] or optimization["oom"]:
        failed_reasons.append("nonfinite_or_oom")
    if optimization["task_optimizer_success_rate"] == 0:
        failed_reasons.append("optimizer_never_stepped")
    if source_f1 and max(source_f1) <= 0:
        failed_reasons.append("source_classifier_no_learning_signal")
    failure_values = _series(source_phase, "failure_rate")
    if failure_values and min(failure_values) >= THRESHOLDS["state_saturation_rate"]:
        failed_reasons.append("phase_always_failed")

    for name in ("identity_rate", "nonidentity_rate"):
        value = phase_summary[name]
        if isinstance(value, (int, float)) and value >= THRESHOLDS["state_saturation_rate"]:
            review_reasons.append(f"{name}_saturated")
    collapse = phase_summary["candidate_collapse_rate"]
    if isinstance(collapse, (int, float)) and collapse >= THRESHOLDS["candidate_collapse_rate"]:
        review_reasons.append("candidate_collapse_rate_saturated")
    state_values = [
        phase_summary[name]
        for name in ("identity_rate", "nonidentity_rate", "failure_rate")
    ]
    if all(isinstance(value, (int, float)) for value in state_values):
        if abs(sum(float(value) for value in state_values) - 1.0) > 1e-3:
            failed_reasons.append("phase_state_ratios_invalid")
    shape_valid = phase_summary["shape_valid_rate"]
    if isinstance(shape_valid, (int, float)) and shape_valid < THRESHOLDS["shape_valid_min"]:
        review_reasons.append("shape_valid_rate_low")
    veto = phase_summary["S_veto_rate"]
    if isinstance(veto, (int, float)) and veto >= THRESHOLDS["state_saturation_rate"]:
        review_reasons.append("structure_veto_saturated")
    skip = optimization["optimizer_skip_rate"]
    if isinstance(skip, (int, float)) and skip > THRESHOLDS["optimizer_skip_max"]:
        review_reasons.append("optimizer_skip_high")
    if isinstance(source_drop, (int, float)) and source_drop > THRESHOLDS["source_validation_drop"]:
        review_reasons.append("source_validation_drop")
    for name, value in quality.items():
        if isinstance(value, (int, float)) and (
            value <= THRESHOLDS["alpha_saturation_low"]
            or value >= THRESHOLDS["alpha_saturation_high"]
        ):
            review_reasons.append(f"{name}_saturated")
    teacher = shape_teacher["target_teacher_valid"]
    if isinstance(teacher, (int, float)) and (
        teacher < THRESHOLDS["teacher_acceptance_min"]
        or teacher > THRESHOLDS["teacher_acceptance_max"]
    ):
        review_reasons.append("target_teacher_acceptance_extreme")
    prototype_ready = shape_teacher["source_prototype_ready"]
    if isinstance(prototype_ready, (int, float)) and prototype_ready <= 0:
        review_reasons.append("source_prototype_not_ready")
    for class_id, record in class_summary["source"].items():
        value = record.get("failure_rate")
        if isinstance(value, (int, float)) and value >= THRESHOLDS["class_failure_rate"]:
            review_reasons.append(f"source_class_{class_id}_failure")

    status = "FAILED" if failed_reasons else (
        "REVIEW_REQUIRED" if review_reasons else "PASS_FOR_PILOT4"
    )
    summary: dict[str, Any] = {
        "status": status,
        "thresholds": THRESHOLDS,
        "classification": {
            "source_validation_by_epoch": validation,
            "source_validation_best_macro_f1": max(source_f1) if source_f1 else NOT_AVAILABLE,
            "source_validation_last_macro_f1": source_f1[-1] if source_f1 else NOT_AVAILABLE,
            "source_validation_drop": source_drop,
            "target_macro_f1": target_metrics.get("macro_f1", NOT_AVAILABLE),
            "target_accuracy": target_metrics.get("accuracy", NOT_AVAILABLE),
        },
        "phase": phase_summary,
        "phase_by_class": class_summary,
        "shape_and_teacher": shape_teacher,
        "quality": quality,
        "optimization": optimization,
        "x1": {
            "registration_acceptance": phase_summary["candidate_acceptable_rate"],
            "relative_identity_gain": NOT_AVAILABLE,
            "phase_magnitude": phase_summary["phase_magnitude_mean"],
            "trend_phase_effect_on_structure": phase_summary["S_changed_preferred_rate"],
            "interclass_distance_preservation": NOT_AVAILABLE,
        },
        "failed_reasons": failed_reasons,
        "review_reasons": review_reasons,
    }
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    classification = summary["classification"]
    return "\n".join([
        "# Structure DA V3 diagnostic summary",
        "",
        f"**Status:** `{summary['status']}`",
        "",
        "## Classification",
        "",
        f"- Source validation best Macro-F1: {classification['source_validation_best_macro_f1']}",
        f"- Source validation last Macro-F1: {classification['source_validation_last_macro_f1']}",
        f"- Target test Macro-F1 (offline evaluation only): {classification['target_macro_f1']}",
        "",
        "## Review",
        "",
        f"- Failed reasons: {summary['failed_reasons'] or 'none'}",
        f"- Review reasons: {summary['review_reasons'] or 'none'}",
        "",
        "Target class diagnostics use `target_geometry_pseudo`; target true labels are not read for training-time analysis.",
    ]) + "\n"


def write_summary(run_directory: Path) -> dict[str, Any]:
    summary = analyze_run(run_directory)
    (run_directory / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (run_directory / "diagnostic_summary.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    summary = write_summary(args.run_directory)
    print(f"DIAGNOSTIC_STATUS={summary['status']}")
    return 1 if summary["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
