#!/usr/bin/env python3
"""Offline replay of Domain-Phase gating/grouping from Calibration-A caches.

This script never runs fdasrsf registration.  It rebuilds pairwise eligibility,
geometry-first candidate hypotheses, and the exact project DomainPhaseState from
stage2_calibration_pairwise.csv + stage2_calibration_geometry.pt.

Run from the TSStructure repository root on the server, where fdasrsf is installed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# `python scripts/analyze_stage2_phase_sweep.py` sets sys.path[0] to scripts/,
# not the repository root.  Resolve the TSStructure root from this file so that
# project imports work regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from methods.structure_da.domain_phase_state import DomainPhaseConfig, update_domain_phase_state
from methods.structure_da.phase_geometry import phase_distance
from methods.structure_da.target_hypothesis_scan import (
    PairwiseClassAlignment,
    TargetHypothesisScanResult,
    _select_candidate_and_hypotheses,
)


def _float(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key, "")
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _bool(row: Dict[str, str], key: str) -> bool:
    value = str(row.get(key, "")).strip().lower()
    return value in {"1", "true", "yes", "y"}


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sample_order(rows: Iterable[Dict[str, str]]) -> List[int]:
    seen = set()
    result = []
    for row in rows:
        sample_id = int(row["sample_id"])
        if sample_id not in seen:
            seen.add(sample_id)
            result.append(sample_id)
    return result


def _scenario_definitions() -> List[Dict[str, object]]:
    # Diagnostic scenarios only.  They do not change the training config.
    return [
        {
            "name": "current",
            "use_roughness_gate": True,
            "use_shape_outer_gate": True,
            "max_roughness": 20.0,
            "max_speed": 4.0,
            "gain_ratio_max": 0.95,
            "class_center_drift_max": 0.05,
            "ambiguity_margin": 0.05,
        },
        {
            "name": "no_rough_no_outer",
            "use_roughness_gate": False,
            "use_shape_outer_gate": False,
            "max_roughness": None,
            "max_speed": 4.0,
            "gain_ratio_max": 0.95,
            "class_center_drift_max": 0.05,
            "ambiguity_margin": 0.05,
        },
        {
            "name": "no_rough_no_outer_no_class_drift",
            "use_roughness_gate": False,
            "use_shape_outer_gate": False,
            "max_roughness": None,
            "max_speed": 4.0,
            "gain_ratio_max": 0.95,
            "class_center_drift_max": 1.0e6,
            "ambiguity_margin": 0.05,
        },
        {
            "name": "no_rough_no_outer_speed10",
            "use_roughness_gate": False,
            "use_shape_outer_gate": False,
            "max_roughness": None,
            "max_speed": 10.01,
            "gain_ratio_max": 0.95,
            "class_center_drift_max": 1.0e6,
            "ambiguity_margin": 0.05,
        },
        {
            "name": "no_rough_no_outer_gain098",
            "use_roughness_gate": False,
            "use_shape_outer_gate": False,
            "max_roughness": None,
            "max_speed": 4.0,
            "gain_ratio_max": 0.98,
            "class_center_drift_max": 1.0e6,
            "ambiguity_margin": 0.05,
        },
        {
            "name": "no_rough_no_outer_margin003",
            "use_roughness_gate": False,
            "use_shape_outer_gate": False,
            "max_roughness": None,
            "max_speed": 4.0,
            "gain_ratio_max": 0.95,
            "class_center_drift_max": 1.0e6,
            "ambiguity_margin": 0.03,
        },
    ]


def _eligible(row: Dict[str, str], scenario: Dict[str, object], base: Dict[str, object]) -> Tuple[bool, Tuple[str, ...]]:
    reasons: List[str] = []
    if not _bool(row, "numerically_valid"):
        reasons.append("numerical")
        return False, tuple(reasons)

    min_support = float(base["stage2_registration_min_common_support"])
    pre_support = _float(row, "pre_common_support_t")
    if pre_support is None or pre_support < min_support:
        reasons.append("pre_support")

    min_increment = float(base["stage2_registration_min_increment"])
    increment = _float(row, "gamma_min_increment")
    if increment is None or increment < min_increment:
        reasons.append("gamma_increment")

    speed = _float(row, "gamma_max_local_speed")
    if speed is None or speed > float(scenario["max_speed"]):
        reasons.append("gamma_speed")

    if bool(scenario["use_roughness_gate"]):
        roughness = _float(row, "gamma_roughness")
        max_roughness = float(scenario["max_roughness"])
        if roughness is None or roughness > max_roughness:
            reasons.append("gamma_roughness")

    deviation = _float(row, "phase_deviation")
    if deviation is None or deviation > float(base["stage2_registration_max_deviation"]):
        reasons.append("gamma_deviation")

    gain = _float(row, "t_gain_ratio")
    if gain is None or gain > float(scenario["gain_ratio_max"]):
        reasons.append("gain")

    if _float(row, "q_shape_distance") is None or _float(row, "common_support_shape") is None:
        reasons.append("shape_support")
    if _float(row, "q_distance_percentile") is None:
        reasons.append("q_cdf_unavailable")

    if bool(scenario["use_shape_outer_gate"]):
        q_distance = _float(row, "q_shape_distance")
        q_outer = _float(row, "source_q_outer")
        if q_distance is None or q_outer is None or q_distance > q_outer:
            reasons.append("shape_outer")

    return len(reasons) == 0, tuple(reasons)


def _alignment(row: Dict[str, str], gamma: torch.Tensor, eligible: bool, reasons: Tuple[str, ...]) -> PairwiseClassAlignment:
    return PairwiseClassAlignment(
        sample_id=int(row["sample_id"]),
        class_id=int(row["class_id"]),
        gamma=gamma.detach().cpu().double(),
        t_identity_error=_float(row, "t_identity_error"),
        t_registered_error=_float(row, "t_registered_error"),
        t_gain_ratio=_float(row, "t_gain_ratio"),
        pre_common_support_t=float(_float(row, "pre_common_support_t") or 0.0),
        common_support_t=_float(row, "common_support_t"),
        gamma_finite=_bool(row, "gamma_finite"),
        gamma_endpoint_error=_float(row, "gamma_endpoint_error"),
        gamma_strictly_increasing=_bool(row, "gamma_strictly_increasing"),
        gamma_min_increment=_float(row, "gamma_min_increment"),
        gamma_max_local_speed=_float(row, "gamma_max_local_speed"),
        gamma_roughness=_float(row, "gamma_roughness"),
        phase_deviation=_float(row, "phase_deviation"),
        q_shape_distance=_float(row, "q_shape_distance"),
        q_distance_percentile=_float(row, "q_distance_percentile"),
        common_support_shape=_float(row, "common_support_shape"),
        numerically_valid=_bool(row, "numerically_valid"),
        phase_evidence_eligible=eligible,
        reject_reasons=reasons,
        solver_error=(row.get("solver_error") or None),
    )


def _build_scan(
    rows: List[Dict[str, str]],
    gammas: torch.Tensor,
    selected_samples: set[int],
    scenario: Dict[str, object],
    base: Dict[str, object],
) -> TargetHypothesisScanResult:
    alignments = []
    by_sample: Dict[int, List[PairwiseClassAlignment]] = {}
    for index, row in enumerate(rows):
        sample_id = int(row["sample_id"])
        if sample_id not in selected_samples:
            continue
        eligible, reasons = _eligible(row, scenario, base)
        item = _alignment(row, gammas[index], eligible, reasons)
        alignments.append(item)
        by_sample.setdefault(sample_id, []).append(item)

    candidates = []
    hypotheses = []
    zero = one = two = 0
    margin = float(scenario["ambiguity_margin"])
    for sample_id in selected_samples:
        candidate, kept = _select_candidate_and_hypotheses(
            by_sample.get(sample_id, ()), ambiguity_margin=margin
        )
        if candidate is not None:
            candidates.append(candidate)
        hypotheses.extend(kept)
        if len(kept) == 0:
            zero += 1
        elif len(kept) == 1:
            one += 1
        else:
            two += 1

    ready_classes = len({int(row["class_id"]) for row in rows})
    return TargetHypothesisScanResult(
        hypotheses=tuple(hypotheses),
        num_samples=len(selected_samples),
        num_pairwise_attempted=len(alignments),
        num_pre_support_rejected=0,
        num_solver_failed=0,
        num_gamma_rejected=0,
        num_gain_rejected=0,
        num_shape_support_rejected=0,
        num_outer_rejected=0,
        samples_with_zero_hypothesis=zero,
        samples_with_one_hypothesis=one,
        samples_with_two_hypotheses=two,
        pairwise_alignments=tuple(alignments),
        candidate_pseudo_labels=tuple(candidates),
        num_solver_calls=0,
        num_ready_classes=ready_classes,
        num_all_class_pairs=len(alignments),
        scanned_sample_ids=tuple(sorted(selected_samples)),
    )


def _phase_config(base: Dict[str, object], scenario: Dict[str, object]) -> DomainPhaseConfig:
    return DomainPhaseConfig(
        phase_min_samples_per_class=float(base["stage2_phase_min_samples_per_class"]),
        phase_class_dispersion_max=float(base["stage2_phase_class_dispersion_max"]),
        phase_class_diameter_max=float(base["stage2_phase_class_diameter_max"]),
        phase_group_dispersion_max=float(base["stage2_phase_group_dispersion_max"]),
        phase_group_diameter_max=float(base["stage2_phase_group_diameter_max"]),
        phase_group_core_separation=float(base["stage2_phase_group_core_separation"]),
        phase_global_radius=float(base["stage2_phase_global_radius"]),
        phase_confirmation_patience=int(base["stage2_phase_confirmation_patience"]),
        phase_center_drift_max=float(scenario["class_center_drift_max"]),
    )


def _identity_distance(gamma: torch.Tensor) -> float:
    identity = torch.linspace(0.0, 1.0, gamma.numel(), dtype=torch.float64)
    return float(phase_distance(gamma.detach().cpu().double(), identity).item())


def _state_payload(state, scan) -> Dict[str, object]:
    return {
        "budget": int(scan.num_samples),
        "num_hypotheses": len(scan.hypotheses),
        "samples_zero": int(scan.samples_with_zero_hypothesis),
        "samples_one": int(scan.samples_with_one_hypothesis),
        "samples_two": int(scan.samples_with_two_hypotheses),
        "m": int(state.m),
        "decision_status": state.decision_status.value,
        "decision_stability_age": int(state.decision_stability_age),
        "valid_phase_classes": list(state.valid_phase_classes),
        "rejected_classes": list(state.rejected_classes),
        "class_centers": [
            {
                "class_id": int(center.class_id),
                "candidate_count": int(center.candidate_count),
                "effective_evidence_count": float(center.effective_evidence_count),
                "dispersion": float(center.dispersion),
                "diameter": float(center.diameter),
                "median_distance": float(center.median_distance),
                "center_drift": None if center.center_drift is None else float(center.center_drift),
                "valid": bool(center.valid),
                "reject_reason": center.reject_reason,
                "phase_distance_to_identity": _identity_distance(center.center_gamma),
            }
            for center in state.class_centers
        ],
        "groups": [
            {
                "group_id": int(group.group_id),
                "member_classes": list(group.member_classes),
                "within_dispersion": float(group.within_dispersion),
                "diameter": float(group.diameter),
                "core_radius": float(group.core_radius),
                "sample_evidence_count": float(group.sample_evidence_count),
                "class_count": int(group.class_count),
                "center_drift": None if group.center_drift is None else float(group.center_drift),
                "status": group.status.value,
                "confirmation_age": int(group.confirmation_age),
                "phase_distance_to_identity": _identity_distance(group.center_gamma),
            }
            for group in state.groups
        ],
        "residual_evidence_count": int(state.residual_evidence_count),
        "residual_evidence_classes": list(state.residual_evidence_classes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "stage2_pilot_v1.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    calibration_dir = args.calibration_dir.resolve()
    pairwise_path = calibration_dir / "stage2_calibration_pairwise.csv"
    geometry_path = calibration_dir / "stage2_calibration_geometry.pt"
    if not pairwise_path.is_file() or not geometry_path.is_file():
        raise FileNotFoundError("calibration directory must contain pairwise CSV and geometry PT")

    rows = _load_rows(pairwise_path)
    geometry = torch.load(geometry_path, map_location="cpu")
    gammas = geometry["gammas"].detach().cpu().double()
    if len(rows) != int(gammas.shape[0]):
        raise RuntimeError("pairwise CSV row count does not match geometry gamma count")
    for index, row in enumerate(rows):
        if int(row["sample_id"]) != int(geometry["sample_ids"][index].item()):
            raise RuntimeError(f"sample alignment mismatch at row {index}")
        if int(row["class_id"]) != int(geometry["class_ids"][index].item()):
            raise RuntimeError(f"class alignment mismatch at row {index}")

    with args.config.open("r", encoding="utf-8") as handle:
        base = json.load(handle)

    order = _sample_order(rows)
    max_budget = len(order)
    budgets = [value for value in (64, 128, 256, 512) if value <= max_budget]
    if max_budget not in budgets:
        budgets.append(max_budget)

    result = {
        "calibration_dir": str(calibration_dir),
        "config": str(args.config),
        "sample_count": max_budget,
        "pairwise_count": len(rows),
        "scenarios": [],
    }

    for scenario in _scenario_definitions():
        print(f"SCENARIO_START|name={scenario['name']}")
        previous = None
        phase_config = _phase_config(base, scenario)
        stages = []
        for budget in budgets:
            selected = set(order[:budget])
            scan = _build_scan(rows, gammas, selected, scenario, base)
            state = update_domain_phase_state(scan, phase_config, previous_state=previous)
            payload = _state_payload(state, scan)
            stages.append(payload)
            group_text = ";".join(
                f"{g['group_id']}:{','.join(str(c) for c in g['member_classes'])}:{g['status']}"
                for g in payload["groups"]
            ) or "-"
            print(
                "PHASE_SWEEP_STAGE|"
                f"scenario={scenario['name']}|budget={budget}"
                f"|hypotheses={payload['num_hypotheses']}"
                f"|valid_classes={','.join(str(c) for c in payload['valid_phase_classes']) or '-'}"
                f"|m={payload['m']}|decision={payload['decision_status']}"
                f"|decision_age={payload['decision_stability_age']}|groups={group_text}"
            )
            previous = state
        result["scenarios"].append({"settings": scenario, "stages": stages})

    output = args.output or (calibration_dir / "stage2_calibration_offline_phase_sweep.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"PHASE_SWEEP_COMPLETE|output={output}")


if __name__ == "__main__":
    main()
