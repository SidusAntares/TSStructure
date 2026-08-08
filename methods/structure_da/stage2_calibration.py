"""Read-only Stage-2 calibration exports for the Round-C inference chain.

The exporter never changes Phase/Shape state and never performs an optimizer
step.  It serializes the raw all-class pairwise geometry acquired by Round A,
plus the current Phase/Stable/Shape state, so statistical thresholds can be
calibrated from measured scales instead of inferred from training outcomes.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Iterable

import torch
from torch import Tensor

from .domain_phase_state import DomainPhaseState
from .domain_shape_state import DomainShapeState
from .phase_geometry import phase_distance
from .prototype_bank import SourcePrototypeBank
from .stable_target_labels import StableTargetLabelScanResult
from .target_hypothesis_scan import PairwiseClassAlignment, TargetHypothesisScanResult


_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _finite_values(values: Iterable[float | None]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def _distribution(values: Iterable[float | None]) -> dict:
    finite = _finite_values(values)
    if not finite:
        return {"count": 0}
    tensor = torch.tensor(finite, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor,
        torch.tensor(_QUANTILES, dtype=torch.float64),
    )
    result = {
        "count": int(tensor.numel()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }
    for q, value in zip(_QUANTILES, quantiles.tolist()):
        result[f"p{int(round(q * 100)):02d}"] = float(value)
    return result




def _json_number(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None

def _identity_like(gamma: Tensor) -> Tensor:
    return torch.linspace(0.0, 1.0, gamma.numel(), dtype=torch.float64)


def _distance_to_identity(alignment: PairwiseClassAlignment) -> float | None:
    if alignment.gamma is None or not alignment.numerically_valid:
        return None
    gamma = alignment.gamma.detach().cpu().double()
    value = float(phase_distance(gamma, _identity_like(gamma)).item())
    return value if math.isfinite(value) else None


def _source_outer(bank: SourcePrototypeBank, class_id: int, *, fused: bool) -> float | None:
    table = bank.f_quantiles if fused else bank.q_quantiles
    if not 0 <= int(class_id) < table.shape[0] or table.shape[1] < 3:
        return None
    return _json_number(float(table[int(class_id), 2].detach().cpu().item()))


def _pairwise_row(alignment: PairwiseClassAlignment, bank: SourcePrototypeBank) -> dict:
    q_outer = _source_outer(bank, alignment.class_id, fused=False)
    q_distance = _json_number(alignment.q_shape_distance)
    return {
        "sample_id": int(alignment.sample_id),
        "class_id": int(alignment.class_id),
        "t_identity_error": alignment.t_identity_error,
        "t_registered_error": alignment.t_registered_error,
        "t_gain_ratio": alignment.t_gain_ratio,
        "pre_common_support_t": alignment.pre_common_support_t,
        "common_support_t": alignment.common_support_t,
        "gamma_endpoint_error": alignment.gamma_endpoint_error,
        "gamma_min_increment": alignment.gamma_min_increment,
        "gamma_max_local_speed": alignment.gamma_max_local_speed,
        "gamma_roughness": alignment.gamma_roughness,
        "phase_deviation": alignment.phase_deviation,
        "phase_distance_to_identity": _distance_to_identity(alignment),
        "q_shape_distance": alignment.q_shape_distance,
        "source_q_outer": q_outer,
        "q_distance_over_outer": (None if q_outer is None or q_outer <= 0.0 or q_distance is None else q_distance / q_outer),
        "q_distance_percentile": alignment.q_distance_percentile,
        "common_support_shape": alignment.common_support_shape,
        "gamma_finite": bool(alignment.gamma_finite),
        "gamma_strictly_increasing": bool(alignment.gamma_strictly_increasing),
        "numerically_valid": bool(alignment.numerically_valid),
        "phase_evidence_eligible": bool(alignment.phase_evidence_eligible),
        "reject_reasons": "+".join(alignment.reject_reasons),
        "solver_error": alignment.solver_error or "",
    }


def _candidate_rows(result: TargetHypothesisScanResult) -> list[dict]:
    by_sample: dict[int, list[PairwiseClassAlignment]] = defaultdict(list)
    for item in result.pairwise_alignments:
        if item.q_shape_distance is not None and math.isfinite(float(item.q_shape_distance)):
            by_sample[int(item.sample_id)].append(item)
    rows: list[dict] = []
    for sample_id in sorted(by_sample):
        ordered = sorted(
            by_sample[sample_id],
            key=lambda item: (float(item.q_shape_distance), int(item.class_id)),
        )
        best = ordered[0]
        second = ordered[1] if len(ordered) > 1 else None
        gap = (
            float(second.q_shape_distance) - float(best.q_shape_distance)
            if second is not None
            else None
        )
        rows.append(
            {
                "sample_id": sample_id,
                "best_class_id": int(best.class_id),
                "best_q_shape_distance": best.q_shape_distance,
                "best_q_distance_percentile": best.q_distance_percentile,
                "best_gain_ratio": best.t_gain_ratio,
                "best_phase_distance_to_identity": _distance_to_identity(best),
                "best_phase_evidence_eligible": bool(best.phase_evidence_eligible),
                "best_reject_reasons": "+".join(best.reject_reasons),
                "second_class_id": None if second is None else int(second.class_id),
                "second_q_shape_distance": None if second is None else second.q_shape_distance,
                "second_q_distance_percentile": None if second is None else second.q_distance_percentile,
                "raw_q_distance_gap": gap,
                "raw_q_distance_ratio": (
                    None
                    if second is None or float(second.q_shape_distance) <= 0.0
                    else float(best.q_shape_distance) / float(second.q_shape_distance)
                ),
            }
        )
    return rows


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_summary(rows: list[dict]) -> dict:
    metrics = (
        "t_identity_error",
        "t_registered_error",
        "t_gain_ratio",
        "pre_common_support_t",
        "common_support_t",
        "gamma_endpoint_error",
        "gamma_min_increment",
        "gamma_max_local_speed",
        "gamma_roughness",
        "phase_deviation",
        "phase_distance_to_identity",
        "q_shape_distance",
        "source_q_outer",
        "q_distance_over_outer",
        "q_distance_percentile",
        "common_support_shape",
    )
    return {metric: _distribution(row.get(metric) for row in rows) for metric in metrics}


def _phase_state_payload(state: DomainPhaseState) -> dict:
    return {
        "scan_index": int(state.scan_index),
        "m": int(state.m),
        "decision_status": state.decision_status.value,
        "decision_stability_age": int(state.decision_stability_age),
        "valid_phase_classes": list(state.valid_phase_classes),
        "rejected_classes": list(state.rejected_classes),
        "identity_evidence_classes": list(state.identity_evidence_classes),
        "identity_evidence_count": float(state.identity_evidence_count),
        "residual_evidence_classes": list(state.residual_evidence_classes),
        "residual_evidence_count": int(state.residual_evidence_count),
        "class_centers": [
            {
                "class_id": int(center.class_id),
                "candidate_count": int(center.candidate_count),
                "effective_evidence_count": float(center.effective_evidence_count),
                "dispersion": float(center.dispersion),
                "diameter": float(center.diameter),
                "median_distance": float(center.median_distance),
                "center_drift": center.center_drift,
                "valid": bool(center.valid),
                "reject_reason": center.reject_reason,
                "center_distance_to_identity": float(
                    phase_distance(
                        center.center_gamma.detach().cpu().double(),
                        _identity_like(center.center_gamma),
                    ).item()
                ),
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
                "center_drift": group.center_drift,
                "status": group.status.value,
                "confirmation_age": int(group.confirmation_age),
                "center_distance_to_identity": float(
                    phase_distance(
                        group.center_gamma.detach().cpu().double(),
                        _identity_like(group.center_gamma),
                    ).item()
                ),
            }
            for group in state.groups
        ],
    }


def _stable_rows(result: StableTargetLabelScanResult, bank: SourcePrototypeBank) -> list[dict]:
    rows = []
    for item in result.candidates:
        q_outer = _source_outer(bank, item.class_id, fused=False)
        f_outer = _source_outer(bank, item.class_id, fused=True)
        q_distance = _json_number(item.q_distance)
        fused_distance = _json_number(item.fused_distance)
        rows.append(
            {
            "sample_id": int(item.sample_id),
            "class_id": int(item.class_id),
            "group_id": int(item.group_id),
            "phase_compatible": bool(item.phase_compatible),
            "phase_distance_to_group": item.phase_distance_to_group,
            "candidate_q_shape_distance": item.candidate_q_shape_distance,
            "candidate_ambiguous": bool(item.candidate_ambiguous),
            "cls_confidence": item.cls_confidence,
            "cls_margin": item.cls_margin,
            "fused_distance": item.fused_distance,
            "source_f_outer": f_outer,
            "fused_distance_over_outer": (None if f_outer is None or f_outer <= 0.0 or fused_distance is None else fused_distance / f_outer),
            "fused_confidence": item.fused_confidence,
            "fused_margin": item.fused_margin,
            "q_distance": item.q_distance,
            "source_q_outer": q_outer,
            "q_distance_over_outer": (None if q_outer is None or q_outer <= 0.0 or q_distance is None else q_distance / q_outer),
            "q_confidence": item.q_confidence,
            "q_margin": item.q_margin,
            "q_common_support": item.q_common_support,
            "passed_classifier": bool(item.passed_classifier),
            "passed_fused": bool(item.passed_fused),
            "passed_q": bool(item.passed_q),
            "accepted": bool(item.accepted),
            "reject_reason": item.reject_reason or "",
        }
        )
    return rows


def _shape_payload(state: DomainShapeState) -> dict:
    return {
        "scan_index": int(state.scan_index),
        "status": state.status.value,
        "valid_classes": list(state.valid_classes),
        "rho_shape": state.rho_shape,
        "leave_one_out_drift": state.leave_one_out_drift,
        "center_drift": state.center_drift,
        "confirmation_age": int(state.confirmation_age),
        "class_centers": [
            {
                "class_id": int(center.class_id),
                "sample_count": int(center.sample_count),
                "effective_weight": float(center.effective_weight),
                "source_distance": _json_number(center.source_distance),
                "valid": bool(center.valid),
                "reject_reason": center.reject_reason,
            }
            for center in state.class_centers
        ],
    }


def export_stage2_calibration_statistics(
    *,
    output_dir: str,
    hypothesis_result: TargetHypothesisScanResult,
    phase_state: DomainPhaseState,
    stable_result: StableTargetLabelScanResult,
    shape_state: DomainShapeState,
    source_prototype_bank: SourcePrototypeBank,
) -> dict[str, str]:
    """Export deterministic, read-only calibration artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    pairwise_rows = [_pairwise_row(item, source_prototype_bank) for item in hypothesis_result.pairwise_alignments]
    candidate_rows = _candidate_rows(hypothesis_result)
    stable_rows = _stable_rows(stable_result, source_prototype_bank)

    pairwise_path = os.path.join(output_dir, "stage2_calibration_pairwise.csv")
    candidate_path = os.path.join(output_dir, "stage2_calibration_candidates.csv")
    stable_path = os.path.join(output_dir, "stage2_calibration_stable_candidates.csv")
    geometry_path = os.path.join(output_dir, "stage2_calibration_geometry.pt")
    summary_path = os.path.join(output_dir, "stage2_calibration_summary.json")
    _write_csv(pairwise_path, pairwise_rows)
    _write_csv(candidate_path, candidate_rows)
    _write_csv(stable_path, stable_rows)
    gamma_records = [item for item in hypothesis_result.pairwise_alignments if item.gamma is not None]
    torch.save(
        {
            "sample_ids": torch.tensor([item.sample_id for item in gamma_records], dtype=torch.long),
            "class_ids": torch.tensor([item.class_id for item in gamma_records], dtype=torch.long),
            "gammas": (
                torch.stack([item.gamma.detach().cpu().float() for item in gamma_records])
                if gamma_records
                else torch.empty((0, 0), dtype=torch.float32)
            ),
            "numerically_valid": torch.tensor(
                [item.numerically_valid for item in gamma_records], dtype=torch.bool
            ),
            "phase_evidence_eligible": torch.tensor(
                [item.phase_evidence_eligible for item in gamma_records], dtype=torch.bool
            ),
            "candidate_sample_ids": torch.tensor(
                [item.sample_id for item in hypothesis_result.candidate_pseudo_labels], dtype=torch.long
            ),
            "candidate_class_ids": torch.tensor(
                [item.class_id for item in hypothesis_result.candidate_pseudo_labels], dtype=torch.long
            ),
        },
        geometry_path,
    )

    rejection_reason_counts: Counter[str] = Counter()
    rejection_combo_counts: Counter[str] = Counter()
    for row in pairwise_rows:
        combo = str(row["reject_reasons"])
        if combo:
            rejection_combo_counts[combo] += 1
            for reason in combo.split("+"):
                if reason:
                    rejection_reason_counts[reason] += 1

    per_class: dict[str, dict] = {}
    by_class: dict[int, list[dict]] = defaultdict(list)
    for row in pairwise_rows:
        by_class[int(row["class_id"])].append(row)
    for class_id, rows in sorted(by_class.items()):
        per_class[str(class_id)] = {
            "count": len(rows),
            "numerically_valid": sum(bool(row["numerically_valid"]) for row in rows),
            "phase_evidence_eligible": sum(bool(row["phase_evidence_eligible"]) for row in rows),
            "metrics": _metric_summary(rows),
        }

    summary = {
        "num_samples": int(hypothesis_result.num_samples),
        "num_ready_classes": int(hypothesis_result.num_ready_classes),
        "num_pairwise_attempted": int(hypothesis_result.num_pairwise_attempted),
        "num_solver_calls": int(hypothesis_result.num_solver_calls),
        "num_candidate_pseudo_labels": len(hypothesis_result.candidate_pseudo_labels),
        "num_phase_hypotheses": len(hypothesis_result.hypotheses),
        "num_numerically_valid": sum(bool(row["numerically_valid"]) for row in pairwise_rows),
        "num_phase_evidence_eligible": sum(
            bool(row["phase_evidence_eligible"]) for row in pairwise_rows
        ),
        "pairwise_metrics": _metric_summary(pairwise_rows),
        "candidate_metrics": {
            "best_q_shape_distance": _distribution(
                row.get("best_q_shape_distance") for row in candidate_rows
            ),
            "second_q_shape_distance": _distribution(
                row.get("second_q_shape_distance") for row in candidate_rows
            ),
            "raw_q_distance_gap": _distribution(
                row.get("raw_q_distance_gap") for row in candidate_rows
            ),
            "raw_q_distance_ratio": _distribution(
                row.get("raw_q_distance_ratio") for row in candidate_rows
            ),
            "best_gain_ratio": _distribution(
                row.get("best_gain_ratio") for row in candidate_rows
            ),
            "best_phase_distance_to_identity": _distribution(
                row.get("best_phase_distance_to_identity") for row in candidate_rows
            ),
            "best_phase_evidence_eligible_count": sum(
                bool(row["best_phase_evidence_eligible"]) for row in candidate_rows
            ),
        },
        "candidate_class_counts": dict(
            sorted(Counter(int(row["best_class_id"]) for row in candidate_rows).items())
        ),
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "rejection_combination_counts": dict(sorted(rejection_combo_counts.items())),
        "per_class": per_class,
        "phase_state": _phase_state_payload(phase_state),
        "stable": {
            "num_samples": int(stable_result.num_samples),
            "num_candidate_views": int(stable_result.num_candidate_views),
            "num_phase_compatible": int(stable_result.num_phase_compatible),
            "num_phase_incompatible": int(stable_result.num_phase_incompatible),
            "num_classifier_pass": int(stable_result.num_classifier_pass),
            "num_fused_pass": int(stable_result.num_fused_pass),
            "num_q_pass": int(stable_result.num_q_pass),
            "num_stable_labels": int(stable_result.num_stable_labels),
            "num_ambiguous_rejected": int(stable_result.num_ambiguous_rejected),
            "stable_class_counts": list(stable_result.stable_class_counts),
            "phase_distance_to_group": _distribution(
                row.get("phase_distance_to_group") for row in stable_rows
            ),
            "cls_margin": _distribution(row.get("cls_margin") for row in stable_rows),
            "fused_distance": _distribution(row.get("fused_distance") for row in stable_rows),
            "fused_distance_over_outer": _distribution(row.get("fused_distance_over_outer") for row in stable_rows),
            "fused_margin": _distribution(row.get("fused_margin") for row in stable_rows),
            "q_distance": _distribution(row.get("q_distance") for row in stable_rows),
            "q_distance_over_outer": _distribution(row.get("q_distance_over_outer") for row in stable_rows),
            "q_margin": _distribution(row.get("q_margin") for row in stable_rows),
        },
        "shape_state": _shape_payload(shape_state),
        "files": {
            "pairwise_csv": os.path.basename(pairwise_path),
            "candidate_csv": os.path.basename(candidate_path),
            "stable_candidate_csv": os.path.basename(stable_path),
            "geometry_pt": os.path.basename(geometry_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return {
        "summary": summary_path,
        "pairwise": pairwise_path,
        "candidates": candidate_path,
        "stable_candidates": stable_path,
        "geometry": geometry_path,
    }
