#!/usr/bin/env python3
"""Aggregate the fixed four-run V3 pilot and its preliminary X1 diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from analyze_structure_da_diagnostic import (  # noqa: E402
    NOT_AVAILABLE,
    analyze_run,
)


EXPECTED_RUNS = {
    "AT1_DK1_seed1", "AT1_DK1_seed2", "AT1_DK1_seed3", "DK1_FR2_seed1"
}


def _nested(record: dict[str, Any], *path: str) -> float | None:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _stats(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {"mean": NOT_AVAILABLE, "std": NOT_AVAILABLE,
                "min": NOT_AVAILABLE, "max": NOT_AVAILABLE,
                "range": NOT_AVAILABLE, "count": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "count": len(values),
    }


METRIC_PATHS = {
    "source_validation_macro_f1": ("classification", "source_validation_best_macro_f1"),
    "target_macro_f1": ("classification", "target_macro_f1"),
    "phase_failure": ("phase", "failure_rate"),
    "phase_identity": ("phase", "identity_rate"),
    "phase_nonidentity": ("phase", "nonidentity_rate"),
    "candidate_collapse": ("phase", "candidate_collapse_rate"),
    "phase_magnitude": ("phase", "phase_magnitude_mean"),
    "S_changed_preferred": ("phase", "S_changed_preferred_rate"),
    "S_veto": ("phase", "S_veto_rate"),
    "shape_valid": ("phase", "shape_valid_rate"),
    "alpha_trend": ("quality", "alpha_trend_mean"),
    "alpha_structure": ("quality", "alpha_structure_mean"),
    "optimizer_skip": ("optimization", "optimizer_skip_rate"),
    "teacher_acceptance": ("shape_and_teacher", "target_teacher_valid"),
}


def _group_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in METRIC_PATHS.items():
        values = [
            value for run in runs
            if (value := _nested(run.get("diagnostic", {}), *path)) is not None
        ]
        result[name] = _stats(values)
    return result


def _x1_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    available_x1 = [run.get("diagnostic", {}).get("x1", {}) for run in runs]
    registration = [
        float(value)
        for record in available_x1
        if isinstance((value := record.get("registration_acceptance")), (int, float))
    ]
    by_class_values: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        for class_id, record in (
            run.get("diagnostic", {}).get("phase_by_class", {}).get("source", {}).items()
        ):
            class_metrics = by_class_values.setdefault(class_id, {
                "registration_acceptance": [],
                "phase_magnitude": [],
                "trend_phase_effect_on_structure": [],
                "structure_veto": [],
            })
            for output_name, input_name in (
                ("registration_acceptance", "candidate_acceptable_rate"),
                ("phase_magnitude", "phase_magnitude_mean"),
                ("trend_phase_effect_on_structure", "structure_changed_preferred_rate"),
                ("structure_veto", "structure_veto_all_rate"),
            ):
                value = record.get(input_name)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    class_metrics[output_name].append(float(value))
    by_source_class = {
        class_id: {
            name: _stats(values) for name, values in metrics.items()
        } | {
            "relative_identity_gain": NOT_AVAILABLE,
            "interclass_distance_preservation": NOT_AVAILABLE,
        }
        for class_id, metrics in by_class_values.items()
    }
    result = {
        "title": "X1: Global Reference Diagnostic",
        "registration_acceptance": _stats(registration),
        "relative_identity_gain": NOT_AVAILABLE,
        "phase_magnitude": _stats([
            float(value) for record in available_x1
            if isinstance((value := record.get("phase_magnitude")), (int, float))
        ]),
        "trend_phase_effect_on_structure": NOT_AVAILABLE,
        "interclass_distance_preservation": NOT_AVAILABLE,
        "by_source_class": by_source_class,
    }
    if not registration:
        result["status"] = "INSUFFICIENT_DIAGNOSTICS"
    elif any(
        run.get("diagnostic", {}).get("status") == "REVIEW_REQUIRED" for run in runs
    ):
        result["status"] = "CLASS_SPECIFIC_FAILURE_REVIEW_NEEDED"
    else:
        result["status"] = "GLOBAL_REFERENCE_PRELIMINARILY_ACCEPTABLE"
    return result


def summarize_pilot4(pilot_root: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    at1 = [
        run for run in runs
        if run.get("source") == "AT1" and run.get("target") == "DK1"
    ]
    dk1 = [
        run for run in runs
        if run.get("source") == "DK1" and run.get("target") == "FR2"
    ]
    names = {str(run.get("run_name")) for run in runs}
    any_failed = any(
        int(run.get("exit_code", 1)) != 0
        or run.get("diagnostic", {}).get("status") == "FAILED"
        for run in runs
    )
    any_review = any(
        run.get("diagnostic", {}).get("status") == "REVIEW_REQUIRED"
        for run in runs
    )
    complete = names == EXPECTED_RUNS
    readiness = (
        "FAILED" if any_failed
        else "READY_FOR_FULL_EXPERIMENT" if complete and not any_review
        else "REVIEW_REQUIRED"
    )
    return {
        "readiness": readiness,
        "run_count": len(runs),
        "expected_runs_present": complete,
        "AT1_to_DK1_three_seeds": _group_metrics(at1),
        "DK1_to_FR2_seed1": _group_metrics(dk1),
        "x1": _x1_summary(runs),
    }


def _read_status_files(pilot_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted((pilot_root / ".launcher_status").glob("*.status")):
        fields = path.read_text(encoding="utf-8").rstrip("\n").split("\t")
        if len(fields) != 10:
            continue
        (run_name, source, target, seed, gpu, start, end, exit_code,
         diagnostic_status, output_path) = fields
        run_directory = Path(output_path)
        try:
            diagnostic = (
                analyze_run(run_directory)
                if (run_directory / "train.log").is_file()
                else {"status": "FAILED", "failed_reasons": ["train_log_missing"]}
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            diagnostic = {
                "status": "FAILED",
                "failed_reasons": [f"analysis_error:{type(error).__name__}:{error}"],
            }
        runs.append({
            "run_name": run_name,
            "source": source,
            "target": target,
            "seed": int(seed),
            "physical_gpu": gpu,
            "start_time": start,
            "end_time": end,
            "exit_code": int(exit_code),
            "diagnostic_status": diagnostic_status,
            "output_path": output_path,
            "command": (run_directory / "command.txt").read_text(
                encoding="utf-8", errors="replace"
            ).strip() if (run_directory / "command.txt").is_file() else NOT_AVAILABLE,
            "checkpoint_path": str(run_directory / "fold_0" / "model.pt"),
            "diagnostic": diagnostic,
        })
    return runs


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join([
        "# Structure DA V3 pilot4 summary",
        "",
        f"**Readiness:** `{summary['readiness']}`",
        "",
        "## AT1 to DK1 (three seeds)",
        "",
        "```json",
        json.dumps(summary["AT1_to_DK1_three_seeds"], indent=2),
        "```",
        "",
        "## DK1 to FR2 (seed 1)",
        "",
        "```json",
        json.dumps(summary["DK1_to_FR2_seed1"], indent=2),
        "```",
        "",
        "## X1: Global Reference Diagnostic",
        "",
        f"Status: `{summary['x1']['status']}`",
        "",
        "Unavailable current-log metrics are reported as `NOT_AVAILABLE_IN_CURRENT_LOGS`; no substitutes are inferred.",
    ]) + "\n"


def write_outputs(pilot_root: Path) -> dict[str, Any]:
    runs = _read_status_files(pilot_root)
    manifest = {
        "code_version": "a7751523794b48813ae9f294303889eed62ea2e7",
        "runs": runs,
    }
    (pilot_root / "pilot4_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    summary = summarize_pilot4(pilot_root, runs)
    (pilot_root / "pilot4_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (pilot_root / "pilot4_summary.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_root", type=Path)
    args = parser.parse_args()
    summary = write_outputs(args.pilot_root)
    print(f"PILOT4_STATUS={summary['readiness']}")
    return 1 if summary["readiness"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
