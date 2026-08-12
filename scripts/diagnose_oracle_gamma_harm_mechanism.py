#!/usr/bin/env python3
"""08: Oracle true-class gamma harm mechanism diagnostic.

No registration is solved here.  The script replays the numerically-valid
oracle true-class gammas already cached by experiment 06 on the same frozen
Stage-1 model and held-out target-test population.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import diagnose_sample_level_phase_validity as sample06
import visualize_stage2_phase_alignment as phasevis
from methods.structure_da.confirmed_phase_view import align_target_positions_to_source
from methods.structure_da.oracle_gamma_harm_diagnostic import (
    ALPHA_VALUES,
    classifier_margin_and_competitor,
    geometry_margin,
    hard_transition,
    shrink_gamma_toward_identity,
    value_space_residual_diagnostic,
    warp_value_function_gamma,
)
from methods.structure_da.phase_registration import resample_gamma, warp_q_gamma, warp_support_gamma
from methods.structure_da.prototype_bank import support_aware_q_distance
from methods.structure_da.sample_phase_diagnostic import TOnlyPhaseRegistration


CACHE_SCHEMA = 1
FOCUS_CLASSES = (
    "spring_oat",
    "winter_wheat",
    "winter_rye",
    "winter_rapeseed",
    "horsebeans",
    "spring_barley",
    "winter_barley",
    "winter_triticale",
)
SCATTER_CLASSES = ("ALL", "spring_oat", "winter_wheat", "winter_rye")


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
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


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def _rate(values: Iterable[bool]) -> float:
    vals = [bool(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _integration_weights(grid: Tensor) -> Tensor:
    weights = torch.ones_like(grid, dtype=torch.float64)
    if weights.numel() > 1:
        weights[[0, -1]] *= 0.5
    return weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)


def _registration_from_payload(payload: dict) -> TOnlyPhaseRegistration:
    values = dict(payload)
    gamma = values.get("gamma")
    if isinstance(gamma, Tensor):
        values["gamma"] = gamma.detach().cpu().double()
    values["reject_reasons"] = tuple(values.get("reject_reasons", ()))
    return TOnlyPhaseRegistration(**values)


def _load_stage_a_cache(path: Path) -> tuple[list[TOnlyPhaseRegistration], list[int], list[int]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"08 requires experiment-06 Stage-A cache and never recomputes exact-DP: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CACHE_SCHEMA:
        raise ValueError("unsupported experiment-06 Stage-A cache schema")
    records = [_registration_from_payload(row) for row in payload.get("records", ())]
    sample_ids = [int(v) for v in payload.get("sample_ids", ())]
    true_classes = [int(v) for v in payload.get("true_classes", ())]
    if len(records) != len(sample_ids) or len(records) != len(true_classes):
        raise ValueError("experiment-06 Stage-A cache arrays have inconsistent lengths")
    return records, sample_ids, true_classes


def _normalize_cache_identity(
    records: Sequence[TOnlyPhaseRegistration],
    sample_ids: Sequence[int],
    true_classes: Sequence[int],
    dataset_parcels: Sequence[int],
) -> tuple[list[TOnlyPhaseRegistration], list[int], list[int]]:
    """Accept either stable parcel IDs or legacy local dataset indices."""
    parcels = [int(v) for v in dataset_parcels]
    ids = [int(v) for v in sample_ids]
    if set(ids) == set(parcels):
        return list(records), ids, list(map(int, true_classes))
    if ids and all(0 <= value < len(parcels) for value in ids):
        mapped = [parcels[value] for value in ids]
        remapped = [replace(record, sample_id=mapped[index]) for index, record in enumerate(records)]
        return remapped, mapped, list(map(int, true_classes))
    raise ValueError("experiment-06 cache sample IDs are neither parcel IDs nor valid local dataset indices")


def _build_source_value_reference(model, loader, *, num_classes: int, device: torch.device, cache_path: Path) -> dict:
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("schema") == 1 and int(payload.get("num_classes", -1)) == int(num_classes):
            print(f"ORACLE_GAMMA_08_SOURCE_VALUE_CACHE_HIT|path={cache_path}", flush=True)
            return payload

    print("ORACLE_GAMMA_08_SOURCE_VALUE_REFERENCE|status=start", flush=True)
    shape_extractor = model.temporal_module.structure_geometry
    grid = shape_extractor.functional_lift.canonical_grid.detach().cpu().double()
    sum_values = None
    sum_support = None
    counts = torch.zeros(num_classes, dtype=torch.long)
    with torch.no_grad():
        for raw_batch in loader:
            batch = phasevis._move_batch(raw_batch, device)
            backbone = model.forward_backbone(
                batch["pixels"], batch["valid_pixels"], batch["positions"],
                batch.get("extra"), time_mask=batch.get("time_mask"), compute_decomposition=True,
            )
            _trend, structure = model._trend_and_structure(backbone)
            s_out = shape_extractor(structure, backbone.normalized_positions, backbone.time_mask)
            values = s_out.functional.function.detach().cpu().double()
            support = s_out.support_confidence.detach().cpu().double()
            valid = s_out.structure_valid.detach().cpu().bool()
            labels = batch["label"].detach().cpu().long()
            if sum_values is None:
                sum_values = torch.zeros(num_classes, values.shape[1], values.shape[2], dtype=torch.float64)
                sum_support = torch.zeros(num_classes, values.shape[1], dtype=torch.float64)
            for class_id in range(num_classes):
                select = (labels == class_id) & valid
                if not torch.any(select):
                    continue
                class_support = support[select]
                sum_values[class_id] += (values[select] * class_support.unsqueeze(-1)).sum(dim=0)
                sum_support[class_id] += class_support.sum(dim=0)
                counts[class_id] += int(select.sum().item())
    if sum_values is None or sum_support is None:
        raise RuntimeError("source loader produced no valid S value-space observations")
    references = sum_values / sum_support.unsqueeze(-1).clamp_min(1e-8)
    mean_support = sum_support / counts.clamp_min(1).unsqueeze(-1)
    ready = (counts > 0) & torch.isfinite(references).all(dim=(1, 2))
    payload = {
        "schema": 1,
        "num_classes": int(num_classes),
        "grid": grid,
        "reference_values": references,
        "reference_support": mean_support,
        "counts": counts,
        "ready": ready,
        "semantic": "support-weighted source-train class mean of frozen latent S=T+D continuous function",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"ORACLE_GAMMA_08_SOURCE_VALUE_REFERENCE|status=ready|path={cache_path}", flush=True)
    return payload


def _positions_for_alpha(native: Tensor, mask: Tensor, parcels: Tensor, record_by_parcel: dict[int, TOnlyPhaseRegistration], alpha: float) -> tuple[Tensor, Tensor]:
    positions = native.detach().clone()
    available = torch.zeros(native.shape[0], device=native.device, dtype=torch.bool)
    for row, parcel in enumerate(parcels.detach().cpu().tolist()):
        record = record_by_parcel.get(int(parcel))
        if record is None or not record.numerically_valid or not isinstance(record.gamma, Tensor):
            continue
        gamma_alpha = shrink_gamma_toward_identity(record.gamma, alpha)
        positions[row : row + 1] = align_target_positions_to_source(
            native[row : row + 1], mask[row : row + 1], gamma_alpha
        )
        available[row] = True
    return positions, available


def _warp_s_batch_for_alpha(
    q: Tensor,
    support: Tensor,
    parcels: Tensor,
    record_by_parcel: dict[int, TOnlyPhaseRegistration],
    shape_grid: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if float(alpha) == 0.0:
        return q, support, torch.ones(q.shape[0], device=q.device, dtype=torch.bool)
    warped_q = q.detach().clone()
    warped_support = support.detach().clone()
    available = torch.zeros(q.shape[0], device=q.device, dtype=torch.bool)
    reg_grid = None
    for row, parcel in enumerate(parcels.detach().cpu().tolist()):
        record = record_by_parcel.get(int(parcel))
        if record is None or not record.numerically_valid or not isinstance(record.gamma, Tensor):
            continue
        gamma_alpha = shrink_gamma_toward_identity(record.gamma, alpha)
        if reg_grid is None or reg_grid.numel() != gamma_alpha.numel():
            reg_grid = torch.linspace(0.0, 1.0, gamma_alpha.numel(), dtype=torch.float64)
        gamma_shape = resample_gamma(
            gamma_alpha, reg_grid, shape_grid.detach().cpu().double()
        ).to(device=q.device, dtype=q.dtype)
        warped_q[row] = warp_q_gamma(q[row], gamma_shape).squeeze(0)
        warped_support[row] = warp_support_gamma(support[row], gamma_shape, shape_grid)
        available[row] = True
    return warped_q, warped_support, available


def _distance_vectors(q: Tensor, support: Tensor, source_q: Tensor, source_support: Tensor, weights: Tensor) -> Tensor:
    result = support_aware_q_distance(q, source_q, support, source_support, weights)
    distances = result.distance
    return torch.where(result.valid, distances, torch.full_like(distances, float("nan")))


def _value_residual_for_row(
    target_function: Tensor,
    target_support: Tensor,
    record: TOnlyPhaseRegistration,
    source_values: Tensor,
    source_support: Tensor,
    shape_grid: Tensor,
    weights: Tensor,
):
    if not record.numerically_valid or not isinstance(record.gamma, Tensor):
        return None
    reg_grid = torch.linspace(0.0, 1.0, record.gamma.numel(), dtype=torch.float64)
    gamma_shape = resample_gamma(record.gamma, reg_grid, shape_grid.detach().cpu().double())
    warped_values = warp_value_function_gamma(target_function.detach().cpu().double(), gamma_shape)
    warped_support = warp_support_gamma(
        target_support.detach().cpu().double(), gamma_shape, shape_grid.detach().cpu().double()
    )
    return value_space_residual_diagnostic(
        warped_values, source_values.detach().cpu().double(),
        warped_support, source_support.detach().cpu().double(), weights,
    )


def _evaluate(
    *, model, target_loader, records: Sequence[TOnlyPhaseRegistration], classes: Sequence[str],
    source_bank, source_value_ref: dict, device: torch.device,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    record_by_parcel = {int(r.sample_id): r for r in records}
    source_q = source_bank.shape_srvf.to(device=device, dtype=torch.float32)
    source_support = source_bank.shape_support.to(device=device, dtype=torch.float32)
    shape_grid = model.temporal_module.structure_geometry.functional_lift.canonical_grid.to(device=device, dtype=torch.float32)
    weights_device = _integration_weights(shape_grid.detach().cpu()).to(device=device, dtype=torch.float32)
    weights_cpu = _integration_weights(shape_grid.detach().cpu())
    source_values = source_value_ref["reference_values"].double()
    source_value_support = source_value_ref["reference_support"].double()

    rows: list[dict] = []
    sample_ids_all: list[int] = []
    labels_all: list[int] = []
    logits_alpha_all: list[np.ndarray] = []
    s_distance_alpha_all: list[np.ndarray] = []

    with torch.no_grad():
        for raw_batch in target_loader:
            batch = phasevis._move_batch(raw_batch, device)
            backbone = model.forward_backbone(
                batch["pixels"], batch["valid_pixels"], batch["positions"],
                batch.get("extra"), time_mask=batch.get("time_mask"), compute_decomposition=True,
            )
            native = backbone.normalized_positions.detach()
            mask = backbone.time_mask.detach()
            parcels = batch["parcel_index"].detach().cpu().long()
            labels = batch["label"].detach().cpu().long()
            no_out = model.forward_from_backbone(
                backbone, batch["positions"], batch.get("extra"), return_geometry=False
            )
            _trend, structure = model._trend_and_structure(backbone)
            s_out = model.temporal_module.structure_geometry(
                structure, backbone.normalized_positions, backbone.time_mask
            )
            q_native = s_out.srvf
            support_native = s_out.support_confidence
            function_native = s_out.functional.function.detach().cpu().double()
            function_support = s_out.support_confidence.detach().cpu().double()

            alpha_logits: list[Tensor] = []
            alpha_distances: list[Tensor] = []
            alpha_available: list[Tensor] = []
            for alpha in ALPHA_VALUES:
                if alpha == 0.0:
                    logits = no_out.logits
                    available = torch.ones(native.shape[0], device=device, dtype=torch.bool)
                else:
                    positions, available = _positions_for_alpha(native, mask, parcels, record_by_parcel, alpha)
                    out = model.forward_from_backbone(
                        backbone, batch["positions"], batch.get("extra"),
                        temporal_positions_override=positions, return_geometry=False,
                    )
                    logits = out.logits
                q_alpha, support_alpha, s_available = _warp_s_batch_for_alpha(
                    q_native, support_native, parcels, record_by_parcel, shape_grid, alpha
                )
                distances = _distance_vectors(
                    q_alpha, support_alpha, source_q, source_support, weights_device
                )
                combined_available = available & s_available
                if alpha != 0.0:
                    distances = torch.where(
                        combined_available.unsqueeze(-1), distances,
                        torch.full_like(distances, float("nan")),
                    )
                    logits = torch.where(
                        combined_available.unsqueeze(-1), logits,
                        torch.full_like(logits, float("nan")),
                    )
                alpha_logits.append(logits.detach().cpu().float())
                alpha_distances.append(distances.detach().cpu().float())
                alpha_available.append(combined_available.detach().cpu())

            logits_stack = torch.stack(alpha_logits, dim=1)  # [B,A,C]
            distances_stack = torch.stack(alpha_distances, dim=1)
            probs_stack = torch.softmax(logits_stack, dim=-1)
            batch_size = labels.numel()
            for row_index in range(batch_size):
                parcel = int(parcels[row_index].item())
                true_class = int(labels[row_index].item())
                record = record_by_parcel.get(parcel)
                numerical_valid = bool(record is not None and record.numerically_valid and isinstance(record.gamma, Tensor))
                no_logits = logits_stack[row_index, 0]
                no_pred = int(torch.argmax(no_logits).item())
                no_prob = float(probs_stack[row_index, 0, true_class].item())
                no_margin, _no_comp = classifier_margin_and_competitor(no_logits, true_class)
                row = {
                    "sample_id": parcel,
                    "true_class": true_class,
                    "class_name": classes[true_class],
                    "gamma_numerically_valid": numerical_valid,
                    "pred_no": no_pred,
                    "p_true_no": no_prob,
                    "cls_margin_no": no_margin,
                }
                if not numerical_valid:
                    row.update({
                        "pred_gamma": "", "transition_type": "gamma_unavailable",
                        "p_true_gamma": float("nan"), "delta_p_true": float("nan"),
                        "cls_margin_gamma": float("nan"), "delta_cls_margin": float("nan"),
                        "cls_competitor_gamma": "", "t_dist_id": float("nan"),
                        "t_dist_gamma": float("nan"), "delta_t_dist": float("nan"),
                        "s_true_dist_id": float(distances_stack[row_index, 0, true_class].item()),
                        "s_true_dist_gamma": float("nan"), "delta_s_true_dist": float("nan"),
                        "s_margin_id": float("nan"), "s_margin_gamma": float("nan"),
                        "delta_s_margin": float("nan"), "s_competitor_id": "",
                        "s_competitor_gamma": "", "value_residual_gamma": float("nan"),
                        "level_difference": float("nan"), "centered_energy_ratio": float("nan"),
                        "affine_residual_ratio": float("nan"), "affine_residual": float("nan"),
                        "affine_scale": float("nan"), "phase_deviation": float("nan"),
                        "t_gain_ratio": float("nan"), "gamma_speed": float("nan"),
                        "gamma_roughness": float("nan"), "best_alpha_by_true_prob": float("nan"),
                        "intermediate_alpha_beats_full": "",
                    })
                    rows.append(row)
                    continue

                gamma_logits = logits_stack[row_index, -1]
                gamma_probs = probs_stack[row_index, -1]
                gamma_pred = int(torch.argmax(gamma_logits).item())
                gamma_prob = float(gamma_probs[true_class].item())
                gamma_margin, cls_comp = classifier_margin_and_competitor(gamma_logits, true_class)
                no_correct = no_pred == true_class
                gamma_correct = gamma_pred == true_class

                s_id = distances_stack[row_index, 0]
                s_gamma = distances_stack[row_index, -1]
                s_margin_id, s_comp_id = geometry_margin(s_id, true_class)
                s_margin_gamma, s_comp_gamma = geometry_margin(s_gamma, true_class)
                s_true_id = float(s_id[true_class].item())
                s_true_gamma = float(s_gamma[true_class].item())

                value_diag = _value_residual_for_row(
                    function_native[row_index], function_support[row_index], record,
                    source_values[true_class], source_value_support[true_class],
                    shape_grid.detach().cpu().double(), weights_cpu,
                )
                alpha_true_probs = probs_stack[row_index, :, true_class].numpy()
                best_index = int(np.nanargmax(alpha_true_probs))
                best_alpha = float(ALPHA_VALUES[best_index])
                intermediate_best = float(np.nanmax(alpha_true_probs[:-1])) > float(alpha_true_probs[-1])

                row.update({
                    "pred_gamma": gamma_pred,
                    "transition_type": hard_transition(no_correct, gamma_correct),
                    "p_true_gamma": gamma_prob,
                    "delta_p_true": gamma_prob - no_prob,
                    "cls_margin_gamma": gamma_margin,
                    "delta_cls_margin": gamma_margin - no_margin,
                    "cls_competitor_gamma": int(cls_comp),
                    "cls_competitor_gamma_name": classes[int(cls_comp)],
                    "t_dist_id": record.t_identity_error,
                    "t_dist_gamma": record.t_registered_error,
                    "delta_t_dist": (
                        float(record.t_registered_error - record.t_identity_error)
                        if record.t_registered_error is not None and record.t_identity_error is not None else float("nan")
                    ),
                    "s_true_dist_id": s_true_id,
                    "s_true_dist_gamma": s_true_gamma,
                    "delta_s_true_dist": s_true_gamma - s_true_id,
                    "s_margin_id": s_margin_id,
                    "s_margin_gamma": s_margin_gamma,
                    "delta_s_margin": s_margin_gamma - s_margin_id,
                    "s_competitor_id": int(s_comp_id),
                    "s_competitor_id_name": classes[int(s_comp_id)],
                    "s_competitor_gamma": int(s_comp_gamma),
                    "s_competitor_gamma_name": classes[int(s_comp_gamma)],
                    "value_residual_gamma": float("nan") if value_diag is None else value_diag.value_residual,
                    "level_difference": float("nan") if value_diag is None else value_diag.level_difference,
                    "centered_energy_ratio": float("nan") if value_diag is None else value_diag.centered_energy_ratio,
                    "affine_residual": float("nan") if value_diag is None else value_diag.affine_residual,
                    "affine_residual_ratio": float("nan") if value_diag is None else value_diag.affine_residual_ratio,
                    "affine_scale": float("nan") if value_diag is None else value_diag.affine_scale,
                    "value_common_support": float("nan") if value_diag is None else value_diag.common_support,
                    "phase_deviation": record.phase_deviation,
                    "t_gain_ratio": record.t_gain_ratio,
                    "gamma_speed": record.gamma_max_local_speed,
                    "gamma_roughness": record.gamma_roughness,
                    "best_alpha_by_true_prob": best_alpha,
                    "intermediate_alpha_beats_full": bool(intermediate_best),
                    "t_improved": bool(
                        record.t_registered_error is not None
                        and record.t_identity_error is not None
                        and record.t_registered_error < record.t_identity_error
                    ),
                    "s_true_improved": bool(s_true_gamma < s_true_id),
                    "s_margin_improved": bool(s_margin_gamma > s_margin_id),
                })
                for alpha_index, alpha in enumerate(ALPHA_VALUES):
                    suffix = str(alpha).replace(".", "p")
                    p = float(probs_stack[row_index, alpha_index, true_class].item())
                    pred = int(torch.argmax(logits_stack[row_index, alpha_index]).item())
                    margin, _comp = classifier_margin_and_competitor(logits_stack[row_index, alpha_index], true_class)
                    s_margin, _scomp = geometry_margin(distances_stack[row_index, alpha_index], true_class)
                    row[f"p_true_alpha_{suffix}"] = p
                    row[f"pred_alpha_{suffix}"] = pred
                    row[f"cls_margin_alpha_{suffix}"] = margin
                    row[f"s_true_dist_alpha_{suffix}"] = float(distances_stack[row_index, alpha_index, true_class].item())
                    row[f"s_margin_alpha_{suffix}"] = s_margin
                rows.append(row)

            sample_ids_all.extend(int(v) for v in parcels.tolist())
            labels_all.extend(int(v) for v in labels.tolist())
            logits_alpha_all.append(logits_stack.numpy())
            s_distance_alpha_all.append(distances_stack.numpy())
            print(f"ORACLE_GAMMA_08_BATCH|processed={len(rows)}", flush=True)

    vectors = {
        "sample_id": np.asarray(sample_ids_all, dtype=np.int64),
        "true_class": np.asarray(labels_all, dtype=np.int64),
        "alpha_values": np.asarray(ALPHA_VALUES, dtype=np.float32),
        "logits_alpha": np.concatenate(logits_alpha_all, axis=0),
        "s_distance_alpha": np.concatenate(s_distance_alpha_all, axis=0),
    }
    return rows, vectors


def _most_common_name(rows: Sequence[dict], key: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def _per_class_summary(rows: Sequence[dict], classes: Sequence[str]) -> list[dict]:
    out: list[dict] = []
    for class_id, class_name in enumerate(classes):
        scope = [r for r in rows if int(r["true_class"]) == class_id]
        valid = [r for r in scope if bool(r["gamma_numerically_valid"])]
        harmful = [r for r in valid if r["transition_type"] == "harmful_hard"]
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "n_samples": len(scope),
            "n_numerically_valid": len(valid),
            "beneficial_hard_count": sum(r["transition_type"] == "beneficial_hard" for r in valid),
            "harmful_hard_count": len(harmful),
            "stable_correct_count": sum(r["transition_type"] == "stable_correct" for r in valid),
            "stable_wrong_count": sum(r["transition_type"] == "stable_wrong" for r in valid),
            "mean_delta_p_true": _mean(r["delta_p_true"] for r in valid),
            "median_delta_p_true": _median(r["delta_p_true"] for r in valid),
            "mean_delta_t_dist": _mean(r["delta_t_dist"] for r in valid),
            "mean_delta_s_true_dist": _mean(r["delta_s_true_dist"] for r in valid),
            "mean_delta_s_margin": _mean(r["delta_s_margin"] for r in valid),
            "mean_value_residual_gamma": _mean(r["value_residual_gamma"] for r in valid),
            "mean_level_difference": _mean(r["level_difference"] for r in valid),
            "mean_centered_energy_ratio": _mean(r["centered_energy_ratio"] for r in valid),
            "mean_affine_residual_ratio": _mean(r["affine_residual_ratio"] for r in valid),
            "intermediate_alpha_beats_full_rate": _rate(r["intermediate_alpha_beats_full"] for r in valid),
            "harmful_classifier_competitor_mode": _most_common_name(harmful, "cls_competitor_gamma_name"),
            "harmful_s_competitor_mode": _most_common_name(harmful, "s_competitor_gamma_name"),
            "harmful_t_not_improved_count": sum(float(r["delta_t_dist"]) >= 0 for r in harmful if math.isfinite(float(r["delta_t_dist"]))),
            "harmful_t_improved_s_worsened_count": sum(
                float(r["delta_t_dist"]) < 0 and float(r["delta_s_true_dist"]) > 0
                for r in harmful if math.isfinite(float(r["delta_t_dist"])) and math.isfinite(float(r["delta_s_true_dist"]))
            ),
            "harmful_s_true_improved_margin_worsened_count": sum(
                float(r["delta_s_true_dist"]) < 0 and float(r["delta_s_margin"]) < 0
                for r in harmful if math.isfinite(float(r["delta_s_true_dist"])) and math.isfinite(float(r["delta_s_margin"]))
            ),
        }
        for alpha in ALPHA_VALUES:
            suffix = str(alpha).replace(".", "p")
            row[f"best_alpha_{suffix}_count"] = sum(float(r["best_alpha_by_true_prob"]) == float(alpha) for r in valid)
        out.append(row)
    return out


def _alpha_summary(rows: Sequence[dict], classes: Sequence[str]) -> list[dict]:
    output: list[dict] = []
    scopes = [("ALL", None)] + [(name, cid) for cid, name in enumerate(classes)]
    for class_name, class_id in scopes:
        scope = [r for r in rows if bool(r["gamma_numerically_valid"]) and (class_id is None or int(r["true_class"]) == class_id)]
        for alpha in ALPHA_VALUES:
            suffix = str(alpha).replace(".", "p")
            correct = [int(r[f"pred_alpha_{suffix}"]) == int(r["true_class"]) for r in scope]
            output.append({
                "class_id": "ALL" if class_id is None else class_id,
                "class_name": class_name,
                "alpha": alpha,
                "n": len(scope),
                "accuracy_or_recall": _rate(correct),
                "mean_true_probability": _mean(r[f"p_true_alpha_{suffix}"] for r in scope),
                "mean_classifier_margin": _mean(r[f"cls_margin_alpha_{suffix}"] for r in scope),
                "mean_true_class_s_distance": _mean(r[f"s_true_dist_alpha_{suffix}"] for r in scope),
                "mean_s_geometry_margin": _mean(r[f"s_margin_alpha_{suffix}"] for r in scope),
            })
    return output


def _scatter(path: Path, rows: Sequence[dict], x_key: str, y_key: str = "delta_p_true") -> None:
    pairs = [
        (float(r[x_key]), float(r[y_key])) for r in rows
        if bool(r["gamma_numerically_valid"]) and math.isfinite(float(r.get(x_key, float("nan"))))
        and math.isfinite(float(r.get(y_key, float("nan"))))
    ]
    if not pairs:
        return
    x, y = zip(*pairs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=10, alpha=0.45)
    ax.axhline(0.0, linewidth=1)
    if x_key.startswith("delta_"):
        ax.axvline(0.0, linewidth=1)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"{x_key} vs {y_key}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_alpha(path: Path, alpha_rows: Sequence[dict], class_name: str) -> None:
    scope = [r for r in alpha_rows if r["class_name"] == class_name]
    if not scope:
        return
    x = [float(r["alpha"]) for r in scope]
    recall = [float(r["accuracy_or_recall"]) for r in scope]
    prob = [float(r["mean_true_probability"]) for r in scope]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, recall, marker="o", label="Accuracy/Recall")
    ax.plot(x, prob, marker="o", label="Mean true-class probability")
    ax.set_xlabel("alpha")
    ax.set_ylabel("metric")
    ax.set_title(f"Gamma strength sweep: {class_name}")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_readme(path: Path, *, source: str, target: str, stage_a_cache: Path, source_value_cache: Path) -> None:
    text = f"""# 08 Oracle True-Class Gamma Harm Mechanism Diagnostic

## 范围

本实验只回答：为什么一部分 oracle true-class gamma 在 T 配准几何改善后，冻结分类器反而下降。

- source / target：`{source}` → `{target}`
- oracle gamma：直接复用 06 Stage-A cache：`{stage_a_cache}`
- exact-DP 调用数：**0**
- 不使用 production legality 过滤；除无法作为时间变换使用的 numerical-invalid gamma 外，所有缓存 gamma 都进入诊断。
- 不使用 class center / group center / Phase grouping / Stable Label / Teacher / 参数更新 / Domain Shape transport。

## 五层诊断

1. T：比较 `t_dist_id` 与 `t_dist_gamma`；`delta_t_dist < 0` 表示 gamma 在它自己的 T registration geometry 中确实改善。
2. S true class：同一个 gamma 作用于 S-SRVF，不重新求 gamma；比较 `delta_s_true_dist`。
3. S class separation：同时保存 target 与所有 source S-SRVF class prototypes 的距离，使用 `S margin = nearest competitor distance - true-class distance`。
4. latent S value-space residual：直接使用 SRVF 之前、冻结 decomposition 得到的连续 latent `S=T+D` 函数。这里不是原始 NDVI，因此只解释为 vertical/value-space magnitude difference。source reference 从冻结 Stage1 + source-train 确定性扫描得到并缓存：`{source_value_cache}`。
5. over-registration：使用 `gamma_alpha=(1-alpha)id+alpha*gamma`，alpha 固定为 `0,0.25,0.5,0.75,1`；只观察中间强度是否优于 full gamma，不据此修改任何机制。

## affine residual

`affine_residual_ratio` 使用 analysis-only fit：一个 sample-level 正 global scale `a>0` 加 channel-wise constant offset `b`。它只判断 Phase 后的剩余 latent S value-space 差异是否能被简单 scale/offset 大量解释，绝不送入 classifier。

## 主要文件

- `oracle_gamma_harm_sample_level.csv`：逐样本分类、T/S geometry、value residual、gamma deformation、alpha 最优强度。
- `full_vectors.npz`：所有 class logits 与所有 class S-distance，维度包含固定 5 个 alpha。
- `per_class_harm_summary.csv`：逐类 hard transition、T/S 变化、competitor、value residual、best-alpha 分布。
- `alpha_sweep_summary.csv`：每类及 ALL 的 alpha→Recall/Accuracy、mean true probability、classifier margin、S true distance、S margin。
- `scatter/`：要求的五种连续量与 `delta_p_true` 散点图。
- `alpha_sweep/`：重点类别的 gamma strength 曲线。

## 判读边界

脚本不自动修正任何样本，也不根据结果选择新 gamma。`intermediate_alpha_beats_full=true` 仅表示该样本存在比 alpha=1 更高的 true-class probability，是 over-registration 的诊断证据，不是生产规则。
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
    source_train_parcels = np.asarray(sorted(splits[source]["train"]), dtype=np.int64)
    target_test_parcels = np.asarray(sorted(splits[target]["test"]), dtype=np.int64)
    source_train_loader = phasevis._selected_loader(
        data_root, source, classes, source_train_parcels,
        closed_set=closed_set, combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    target_test_loader = phasevis._selected_loader(
        data_root, target, classes, target_test_parcels,
        closed_set=closed_set, combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    target_meta = phasevis._metadata_dataset(
        data_root, target, classes, splits[target]["test"], closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
    )
    label_by_parcel = sample06._label_map(target_meta)

    records, cache_ids, cache_classes = _load_stage_a_cache(args.stage_a_cache.resolve())
    dataset_parcels = target_test_loader.dataset.get_parcel_indices().tolist()
    records, cache_ids, cache_classes = _normalize_cache_identity(
        records, cache_ids, cache_classes, dataset_parcels
    )
    if set(cache_ids) != set(label_by_parcel):
        raise ValueError("06 Stage-A cache does not match reconstructed held-out target-test parcels")
    for sample_id, true_class in zip(cache_ids, cache_classes):
        if int(label_by_parcel[int(sample_id)]) != int(true_class):
            raise ValueError("06 Stage-A cache true class differs from target-test metadata")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    source_value_cache = cache_dir / "source_structure_value_reference.pt"
    source_value_ref = _build_source_value_reference(
        model, source_train_loader, num_classes=len(classes), device=device,
        cache_path=source_value_cache,
    )
    source_bank = sample06._source_bank(calibration)

    print(
        "ORACLE_GAMMA_08_START|"
        f"target_test={len(cache_ids)}|numerically_valid={sum(r.numerically_valid for r in records)}|"
        "exact_dp_calls=0|production_legality_filter=false",
        flush=True,
    )
    sample_rows, vectors = _evaluate(
        model=model, target_loader=target_test_loader, records=records,
        classes=classes, source_bank=source_bank, source_value_ref=source_value_ref,
        device=device,
    )
    print("ORACLE_GAMMA_08_SAMPLE_EVALUATION_READY|status=ready", flush=True)

    class_rows = _per_class_summary(sample_rows, classes)
    alpha_rows = _alpha_summary(sample_rows, classes)
    _write_csv(output_dir / "oracle_gamma_harm_sample_level.csv", sample_rows)
    _write_csv(output_dir / "per_class_harm_summary.csv", class_rows)
    _write_csv(output_dir / "alpha_sweep_summary.csv", alpha_rows)
    np.savez_compressed(output_dir / "full_vectors.npz", **vectors)

    x_keys = (
        "delta_t_dist",
        "delta_s_true_dist",
        "delta_s_margin",
        "phase_deviation",
        "affine_residual_ratio",
    )
    for scope_name in SCATTER_CLASSES:
        scope = sample_rows if scope_name == "ALL" else [r for r in sample_rows if r["class_name"] == scope_name]
        for key in x_keys:
            _scatter(output_dir / "scatter" / scope_name / f"{key}_vs_delta_p_true.png", scope, key)
    for class_name in ("ALL", *FOCUS_CLASSES):
        _plot_alpha(output_dir / "alpha_sweep" / f"{class_name}.png", alpha_rows, class_name)

    valid_rows = [r for r in sample_rows if bool(r["gamma_numerically_valid"])]
    harmful_rows = [r for r in valid_rows if r["transition_type"] == "harmful_hard"]
    result = {
        "protocol": "08_oracle_true_class_gamma_harm_mechanism_diagnostic",
        "source": source,
        "target": target,
        "seed": seed,
        "fold": fold,
        "target_test_samples": len(sample_rows),
        "numerically_valid_gamma": len(valid_rows),
        "exact_dp_calls": 0,
        "production_legality_filter_used": False,
        "class_center_used": False,
        "group_center_used": False,
        "stable_label_used": False,
        "teacher_used": False,
        "training_updates": False,
        "domain_shape_transport_used": False,
        "new_gamma_selection_mechanism_used": False,
        "alpha_values": list(ALPHA_VALUES),
        "harmful_hard_count": len(harmful_rows),
        "harmful_t_not_improved_rate": _rate(
            float(r["delta_t_dist"]) >= 0 for r in harmful_rows if math.isfinite(float(r["delta_t_dist"]))
        ),
        "harmful_t_improved_s_worsened_rate": _rate(
            float(r["delta_t_dist"]) < 0 and float(r["delta_s_true_dist"]) > 0
            for r in harmful_rows if math.isfinite(float(r["delta_t_dist"])) and math.isfinite(float(r["delta_s_true_dist"]))
        ),
        "harmful_s_true_improved_margin_worsened_rate": _rate(
            float(r["delta_s_true_dist"]) < 0 and float(r["delta_s_margin"]) < 0
            for r in harmful_rows if math.isfinite(float(r["delta_s_true_dist"])) and math.isfinite(float(r["delta_s_margin"]))
        ),
        "intermediate_alpha_beats_full_rate_all_valid": _rate(
            bool(r["intermediate_alpha_beats_full"]) for r in valid_rows
        ),
        "class_summary": class_rows,
    }
    _json_dump(output_dir / "summary.json", result)
    manifest = {
        "protocol": result["protocol"],
        "stage_a_cache": str(args.stage_a_cache.resolve()),
        "calibration_checkpoint": str(args.calibration_checkpoint.resolve()),
        "model_checkpoint": str(args.model_checkpoint.resolve()),
        "source_value_reference_cache": str(source_value_cache),
        "target_true_label_use": "oracle-only: selects the cached true-class gamma and audits outcomes",
        "gamma_direction": "source->target gamma; classifier uses inverse gamma to map target positions to source coordinates",
        "s_gamma_use": "same T-derived gamma is applied to S-SRVF; S never solves a new gamma",
        "value_space_semantic": "frozen latent S=T+D continuous function before SRVF; not raw NDVI",
        "affine_fit_semantic": "analysis-only positive global scalar plus channel-wise constant offset; never fed to classifier",
        "exact_dp_calls": 0,
        "forbidden_mechanisms": {
            "production_legality_filter": False,
            "phase_confirmation": False,
            "class_center": False,
            "group_center": False,
            "phase_grouping": False,
            "stable_label": False,
            "teacher_student": False,
            "parameter_update": False,
            "domain_shape_transport": False,
        },
    }
    _json_dump(output_dir / "manifest.json", manifest)
    _write_readme(
        output_dir / "README_中文说明.md", source=source, target=target,
        stage_a_cache=args.stage_a_cache.resolve(), source_value_cache=source_value_cache,
    )
    print(f"ORACLE_GAMMA_08_DONE|output={output_dir}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-cache", type=Path, required=True)
    parser.add_argument("--calibration-checkpoint", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
