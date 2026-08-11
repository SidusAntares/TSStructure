#!/usr/bin/env python3
"""06: held-out sample-level Phase validity diagnostic.

This experiment is diagnostic-only and never changes training, Stable Labels,
Domain Phase groups, or M=1/M=2 selection.

Stage A (oracle-only): target true class y_i is used only to choose the correct
source T prototype, yielding gamma_{i,y_i}.  Gamma generation and admission are
strictly T-only; S-SRVF is evaluated afterwards as independent validation.

Stage B (unsupervised selection baseline): every ready source class proposes a
T-derived Phase and the minimum *raw* aligned S-SRVF distance selects the
(class, Phase) hypothesis.  No classifier, Teacher, pseudo-label history,
class balancing, distance calibration, or new threshold is used for selection.
Target labels are read only after selection for oracle auditing.

The within-class MDS plots are descriptive only.  This script never clusters,
selects a group count, or creates a new Phase grouping mechanism.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import diagnose_class_center_vs_group_phase as classdiag
import visualize_stage2_phase_alignment as phasevis
from methods.structure_da import phase_visualization_protocol as visproto
from methods.structure_da.confirmed_phase_view import align_target_positions_to_source
from methods.structure_da.phase_geometry import phase_distance
from methods.structure_da.phase_registration import (
    build_source_registration_prototypes,
    resample_gamma,
    warp_q_gamma,
    warp_support_gamma,
)
from methods.structure_da.prototype_bank import SourcePrototypeBank, support_aware_q_distance
from methods.structure_da.registration_geometry import SourceRegistrationPrototypeBank
from methods.structure_da.sample_phase_diagnostic import (
    RawShapeValidation,
    TOnlyPhaseRegistration,
    classical_mds,
    evaluate_shape_validation,
    phase_distance_matrix,
    remap_local_sample_ids_to_parcels,
    select_raw_shape_candidate,
    solve_t_only_registrations,
    trend_only_cache,
)
from methods.structure_da.stage2_trainer import (
    DeviceBatchLoader,
    build_stage2_registration_extractor,
)
from methods.structure_da.target_hypothesis_scan import PhaseHypothesisScanConfig
from methods.structure_da import target_hypothesis_scan as targetscan


VARIANTS = ("no_phase", "individual_phase", "class_center", "group_center")
CACHE_SCHEMA = 1


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
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _rate(flags: Iterable[bool]) -> float:
    values = list(bool(value) for value in flags)
    return float(sum(values) / len(values)) if values else float("nan")


def _safe_reduction(before: float, after: float) -> float:
    if not math.isfinite(before) or not math.isfinite(after) or abs(before) <= 1e-12:
        return float("nan")
    return float((before - after) / abs(before))


def _source_bank(checkpoint: dict) -> SourcePrototypeBank:
    payload = checkpoint.get("source_prototype_bank")
    if not isinstance(payload, dict):
        raise ValueError("calibration checkpoint does not contain source_prototype_bank")
    return SourcePrototypeBank(
        trend_srvf=payload["trend_srvf"].detach().cpu(),
        shape_srvf=payload["shape_srvf"].detach().cpu(),
        trend_support=payload["trend_support"].detach().cpu(),
        shape_support=payload["shape_support"].detach().cpu(),
        fused=payload["fused"].detach().cpu(),
        class_counts=payload["class_counts"].detach().cpu(),
        ready=payload["ready"].detach().cpu().bool(),
        q_distance_samples=tuple(item.detach().cpu() for item in payload["q_distance_samples"]),
        f_distance_samples=tuple(item.detach().cpu() for item in payload["f_distance_samples"]),
        q_quantiles=payload["q_quantiles"].detach().cpu(),
        f_quantiles=payload["f_quantiles"].detach().cpu(),
        version=int(payload["version"]),
    )


def _scan_config(runtime: dict, workers: int) -> PhaseHypothesisScanConfig:
    def f(name: str) -> float:
        value = runtime.get(name)
        if value is None:
            raise ValueError(f"runtime_config is missing {name}")
        return float(value)

    return PhaseHypothesisScanConfig(
        registration_lambda=f("stage2_registration_lambda"),
        registration_gain_ratio_max=f("stage2_registration_gain_ratio_max"),
        registration_min_common_support=f("stage2_registration_min_common_support"),
        registration_max_roughness=f("stage2_registration_max_roughness"),
        registration_min_increment=f("stage2_registration_min_increment"),
        registration_max_local_speed=f("stage2_registration_max_local_speed"),
        registration_max_deviation=f("stage2_registration_max_deviation"),
        class_hypothesis_margin=f("stage2_class_hypothesis_margin"),
        k_reg=128,
        registration_workers=int(workers),
    )


def _registration_bank_payload(bank: SourceRegistrationPrototypeBank) -> dict:
    return {
        "trend_srvf": bank.trend_srvf.detach().cpu(),
        "trend_support": bank.trend_support.detach().cpu(),
        "class_counts": bank.class_counts.detach().cpu(),
        "ready": bank.ready.detach().cpu(),
        "registration_grid": bank.registration_grid.detach().cpu(),
    }


def _registration_bank_from_payload(payload: dict) -> SourceRegistrationPrototypeBank:
    return SourceRegistrationPrototypeBank(
        trend_srvf=payload["trend_srvf"].detach().cpu(),
        trend_support=payload["trend_support"].detach().cpu(),
        class_counts=payload["class_counts"].detach().cpu(),
        ready=payload["ready"].detach().cpu().bool(),
        registration_grid=payload["registration_grid"].detach().cpu(),
    )


def _load_or_build_registration_bank(
    cache_path: Path,
    *,
    model,
    source_train_loader,
    num_classes: int,
    device: torch.device,
    reg_extractor,
) -> SourceRegistrationPrototypeBank:
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("schema") == CACHE_SCHEMA:
            print(f"SAMPLE_PHASE_SOURCE_REG_BANK_CACHE_HIT|path={cache_path}", flush=True)
            return _registration_bank_from_payload(payload["bank"])
    print("SAMPLE_PHASE_SOURCE_REG_BANK_BUILD|status=start", flush=True)
    bank = build_source_registration_prototypes(
        model, source_train_loader, num_classes, device=device, reg_extractor=reg_extractor
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": CACHE_SCHEMA, "bank": _registration_bank_payload(bank)}, cache_path)
    print(f"SAMPLE_PHASE_SOURCE_REG_BANK_BUILD|status=ready|path={cache_path}", flush=True)
    return _registration_bank_from_payload(_registration_bank_payload(bank))


def _registration_to_payload(record: TOnlyPhaseRegistration) -> dict:
    payload = asdict(record)
    if isinstance(record.gamma, Tensor):
        payload["gamma"] = record.gamma.detach().cpu()
    return payload


def _registration_from_payload(payload: dict) -> TOnlyPhaseRegistration:
    values = dict(payload)
    gamma = values.get("gamma")
    if isinstance(gamma, Tensor):
        values["gamma"] = gamma.detach().cpu().double()
    values["reject_reasons"] = tuple(values.get("reject_reasons", ()))
    return TOnlyPhaseRegistration(**values)


def _load_registration_cache(path: Path, sample_ids: Sequence[int], true_classes: Sequence[int]):
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CACHE_SCHEMA:
        return None
    if list(payload.get("sample_ids", ())) != list(map(int, sample_ids)):
        return None
    if list(payload.get("true_classes", ())) != list(map(int, true_classes)):
        return None
    return tuple(_registration_from_payload(row) for row in payload.get("records", ()))


def _save_registration_cache(
    path: Path,
    records: Sequence[TOnlyPhaseRegistration],
    sample_ids: Sequence[int],
    true_classes: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": CACHE_SCHEMA,
        "sample_ids": list(map(int, sample_ids)),
        "true_classes": list(map(int, true_classes)),
        "records": [_registration_to_payload(record) for record in records],
    }, path)


def _label_map(meta) -> Dict[int, int]:
    return {
        int(parcel): int(label)
        for parcel, label in zip(meta.get_parcel_indices().tolist(), meta.get_labels().tolist())
    }


def _phase_dist(gamma: Tensor | None, center: Tensor) -> float:
    if gamma is None:
        return float("nan")
    return float(phase_distance(gamma.detach().cpu().double(), center.detach().cpu().double()).item())


def _identity(gamma: Tensor) -> Tensor:
    return torch.linspace(0.0, 1.0, gamma.numel(), dtype=torch.float64)


def _displacement_stats(gamma: Tensor | None) -> dict:
    if gamma is None:
        return {
            "median_displacement_days": float("nan"),
            "p10_displacement_days": float("nan"),
            "p90_displacement_days": float("nan"),
            "max_abs_displacement_days": float("nan"),
        }
    g = gamma.detach().cpu().double()
    displacement = (g - _identity(g)) * 365.0
    return {
        "median_displacement_days": float(torch.quantile(displacement, 0.5).item()),
        "p10_displacement_days": float(torch.quantile(displacement, 0.1).item()),
        "p90_displacement_days": float(torch.quantile(displacement, 0.9).item()),
        "max_abs_displacement_days": float(displacement.abs().max().item()),
    }


def _individual_positions(native: Tensor, mask: Tensor, parcels: Tensor, gamma_by_parcel: Dict[int, Tensor]) -> tuple[Tensor, Tensor]:
    positions = native.detach().clone()
    available = torch.zeros(native.shape[0], device=native.device, dtype=torch.bool)
    for row, parcel in enumerate(parcels.detach().cpu().tolist()):
        gamma = gamma_by_parcel.get(int(parcel))
        if gamma is None:
            continue
        positions[row : row + 1] = align_target_positions_to_source(
            native[row : row + 1], mask[row : row + 1], gamma
        )
        available[row] = True
    return positions.detach(), available


def _individual_warp_geometry(
    q: Tensor,
    support: Tensor,
    parcels: Tensor,
    gamma_by_parcel: Dict[int, Tensor],
    grid: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    warped_q = q.detach().clone()
    warped_support = support.detach().clone()
    available = torch.zeros(q.shape[0], device=q.device, dtype=torch.bool)
    for row, parcel in enumerate(parcels.detach().cpu().tolist()):
        gamma = gamma_by_parcel.get(int(parcel))
        if gamma is None:
            continue
        gamma_grid = classdiag._gamma_grid(gamma, grid)
        warped_q[row] = warp_q_gamma(q[row], gamma_grid).squeeze(0)
        warped_support[row] = warp_support_gamma(support[row], gamma_grid, grid)
        available[row] = True
    return warped_q, warped_support, available


def _stage_a_downstream(
    *,
    model,
    target_loader,
    source_pse,
    checkpoint: dict,
    classes: Sequence[str],
    class_centers: Dict[int, dict],
    group: dict,
    oracle_records: Sequence[TOnlyPhaseRegistration],
    shape_by_key: Dict[tuple[int, int], RawShapeValidation],
    device: torch.device,
    pse_grid_size: int,
) -> tuple[List[dict], List[dict], dict]:
    bank = checkpoint["source_prototype_bank"]
    fused_proto = bank["fused"].to(device=device, dtype=torch.float32)
    trend_proto = bank["trend_srvf"].to(device=device, dtype=torch.float32)
    trend_support = bank["trend_support"].to(device=device, dtype=torch.float32)
    shape_proto = bank["shape_srvf"].to(device=device, dtype=torch.float32)
    shape_support = bank["shape_support"].to(device=device, dtype=torch.float32)
    group_gamma = group["center_gamma"]
    legal_by_parcel = {
        int(record.sample_id): record.gamma
        for record in oracle_records
        if record.t_only_legal and isinstance(record.gamma, Tensor)
    }
    record_by_parcel = {int(record.sample_id): record for record in oracle_records}

    rows: List[dict] = []
    aggregate: Dict[int, List[dict]] = {class_id: [] for class_id in range(len(classes))}

    for raw_batch in target_loader:
        batch = phasevis._move_batch(raw_batch, device)
        backbone = model.forward_backbone(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch.get("extra"), time_mask=batch.get("time_mask")
        )
        native = backbone.normalized_positions.detach()
        mask = backbone.time_mask.detach()
        labels = batch["label"].long()
        parcels = batch["parcel_index"].detach().cpu().long()
        no_phase = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"), return_geometry=True
        )
        if no_phase.geometry is None:
            raise RuntimeError("06 Stage A requires functional geometry")
        individual_pos, individual_available = _individual_positions(
            native, mask, parcels, legal_by_parcel
        )
        class_pos, group_pos = classdiag._variant_positions(
            native, mask, labels, class_centers=class_centers, group_gamma=group_gamma
        )
        individual_out = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"),
            temporal_positions_override=individual_pos, return_geometry=False
        )
        class_out = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"),
            temporal_positions_override=class_pos, return_geometry=False
        )
        group_out = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"),
            temporal_positions_override=group_pos, return_geometry=False
        )
        outputs = {
            "no_phase": no_phase,
            "individual_phase": individual_out,
            "class_center": class_out,
            "group_center": group_out,
        }
        positions = {
            "no_phase": native,
            "individual_phase": individual_pos,
            "class_center": class_pos,
            "group_center": group_pos,
        }

        grid = no_phase.geometry.canonical_grid.detach()
        weights = classdiag._integration_weights(grid.numel(), no_phase.geometry.trend_srvf)
        individual_tq, individual_ts, individual_geom_available = _individual_warp_geometry(
            no_phase.geometry.trend_srvf, no_phase.geometry.trend_support,
            parcels, legal_by_parcel, grid
        )
        individual_sq, individual_ss, _ = _individual_warp_geometry(
            no_phase.geometry.structure_srvf, no_phase.geometry.structure_support,
            parcels, legal_by_parcel, grid
        )
        class_tq, class_ts = classdiag._warp_geometry_by_class(
            no_phase.geometry.trend_srvf, no_phase.geometry.trend_support,
            labels, class_centers, grid
        )
        class_sq, class_ss = classdiag._warp_geometry_by_class(
            no_phase.geometry.structure_srvf, no_phase.geometry.structure_support,
            labels, class_centers, grid
        )
        group_tq, group_ts = classdiag._warp_geometry_group(
            no_phase.geometry.trend_srvf, no_phase.geometry.trend_support, group_gamma, grid
        )
        group_sq, group_ss = classdiag._warp_geometry_group(
            no_phase.geometry.structure_srvf, no_phase.geometry.structure_support, group_gamma, grid
        )
        q_variants = {
            "no_phase": (
                no_phase.geometry.trend_srvf, no_phase.geometry.trend_support,
                no_phase.geometry.structure_srvf, no_phase.geometry.structure_support,
            ),
            "individual_phase": (individual_tq, individual_ts, individual_sq, individual_ss),
            "class_center": (class_tq, class_ts, class_sq, class_ss),
            "group_center": (group_tq, group_ts, group_sq, group_ss),
        }

        metrics = {}
        for variant in VARIANTS:
            logits = outputs[variant].logits
            probs = torch.softmax(logits.float(), dim=-1)
            pred = logits.argmax(dim=-1)
            fused = classdiag._fused_distance(outputs[variant].fused_repr, fused_proto, labels)
            tq, ts, sq, ss = q_variants[variant]
            trend_dist, trend_valid = classdiag._true_class_q_distance(
                tq, ts, trend_proto, trend_support, labels, weights
            )
            shape_dist, shape_valid = classdiag._true_class_q_distance(
                sq, ss, shape_proto, shape_support, labels, weights
            )
            metrics[variant] = {
                "probs": probs, "pred": pred, "fused": fused,
                "trend": trend_dist, "trend_valid": trend_valid,
                "shape": shape_dist, "shape_valid": shape_valid,
            }

        for row_index, parcel in enumerate(parcels.tolist()):
            parcel = int(parcel)
            class_id = int(labels[row_index].item())
            record = record_by_parcel[parcel]
            class_gamma = class_centers[class_id]["center_gamma"]
            sample = {
                "sample_id": parcel,
                "true_class": class_id,
                "class_name": classes[class_id],
                "t_registration_legal": bool(record.t_only_legal),
                "t_registration_reject_reasons": "+".join(record.reject_reasons),
                "t_numerically_valid": bool(record.numerically_valid),
                "t_identity_error": record.t_identity_error,
                "t_registered_error": record.t_registered_error,
                "t_gain_ratio": record.t_gain_ratio,
                "pre_common_support_t": record.pre_common_support_t,
                "common_support_t": record.common_support_t,
                "gamma_roughness": record.gamma_roughness,
                "gamma_min_increment": record.gamma_min_increment,
                "gamma_max_local_speed": record.gamma_max_local_speed,
                "phase_deviation": record.phase_deviation,
                "d_gamma_to_identity": (
                    float("nan") if record.gamma is None else _phase_dist(record.gamma, _identity(record.gamma))
                ),
                "d_gamma_to_class_center": _phase_dist(record.gamma, class_gamma),
                "d_gamma_to_group_center": _phase_dist(record.gamma, group_gamma),
            }
            sample.update(_displacement_stats(record.gamma))
            proposal_shape = shape_by_key.get((parcel, class_id))
            sample["independent_s_distance_from_t_only_gamma"] = (
                float("nan") if proposal_shape is None or proposal_shape.raw_shape_distance is None
                else proposal_shape.raw_shape_distance
            )
            source_center, source_center_support = source_pse.center(class_id)
            for variant in VARIANTS:
                available = variant != "individual_phase" or bool(individual_available[row_index].item())
                if variant == "individual_phase" and bool(individual_available[row_index].item()) != bool(individual_geom_available[row_index].item()):
                    raise RuntimeError("individual Phase position/geometry availability mismatch")
                if not available:
                    for field in (
                        "prediction", "true_probability", "fused_distance", "pse_l2",
                        "trend_srvf_distance", "shape_srvf_distance",
                    ):
                        sample[f"{field}_{variant}"] = float("nan")
                    continue
                m = metrics[variant]
                sample[f"prediction_{variant}"] = int(m["pred"][row_index].item())
                sample[f"true_probability_{variant}"] = float(m["probs"][row_index, class_id].item())
                sample[f"fused_distance_{variant}"] = float(m["fused"][row_index].item())
                sample[f"trend_srvf_distance_{variant}"] = (
                    float(m["trend"][row_index].item()) if bool(m["trend_valid"][row_index].item()) else float("nan")
                )
                sample[f"shape_srvf_distance_{variant}"] = (
                    float(m["shape"][row_index].item()) if bool(m["shape_valid"][row_index].item()) else float("nan")
                )
                trajectory, trajectory_support, _ = visproto.canonicalize_pse_tokens(
                    {
                        "pse_tokens": backbone.tokens[row_index].detach().cpu(),
                        "positions": positions[variant][row_index].detach().cpu(),
                        "mask": mask[row_index].detach().cpu(),
                    },
                    positions_key="positions", grid_size=pse_grid_size,
                )
                _, pse_l2, _ = visproto.pse_integrated_distance(
                    source_center, source_center_support, trajectory, trajectory_support
                )
                sample[f"pse_l2_{variant}"] = pse_l2
            if record.t_only_legal:
                sample["individual_true_prob_gain"] = (
                    sample["true_probability_individual_phase"] - sample["true_probability_no_phase"]
                )
                sample["class_center_true_prob_gain"] = (
                    sample["true_probability_class_center"] - sample["true_probability_no_phase"]
                )
                sample["group_center_true_prob_gain"] = (
                    sample["true_probability_group_center"] - sample["true_probability_no_phase"]
                )
                sample["individual_minus_class_center_gain"] = (
                    sample["individual_true_prob_gain"] - sample["class_center_true_prob_gain"]
                )
            else:
                for field in (
                    "individual_true_prob_gain", "class_center_true_prob_gain",
                    "group_center_true_prob_gain", "individual_minus_class_center_gain",
                ):
                    sample[field] = float("nan")
            rows.append(sample)
            aggregate[class_id].append(sample)

    class_rows: List[dict] = []
    for class_id, class_name in enumerate(classes):
        samples = aggregate[class_id]
        if not samples:
            continue
        legal = [row for row in samples if row["t_registration_legal"]]
        item = {
            "class_id": class_id,
            "class_name": class_name,
            "n_samples": len(samples),
            "legal_phase_count": len(legal),
            "legal_phase_rate": len(legal) / len(samples),
            "median_d_to_identity": float(np.nanmedian([row["d_gamma_to_identity"] for row in samples])),
            "median_d_to_class_center": float(np.nanmedian([row["d_gamma_to_class_center"] for row in samples])),
            "median_d_to_group_center": float(np.nanmedian([row["d_gamma_to_group_center"] for row in samples])),
        }
        if legal:
            item.update({
                "individual_shape_improvement_rate": _rate(
                    row["shape_srvf_distance_individual_phase"] < row["shape_srvf_distance_no_phase"]
                    for row in legal
                ),
                "individual_pse_improvement_rate": _rate(
                    row["pse_l2_individual_phase"] < row["pse_l2_no_phase"] for row in legal
                ),
                "individual_ltae_improvement_rate": _rate(
                    row["fused_distance_individual_phase"] < row["fused_distance_no_phase"] for row in legal
                ),
                "individual_true_prob_improvement_rate": _rate(
                    row["individual_true_prob_gain"] > 0 for row in legal
                ),
                "individual_true_prob_gain_mean": _mean(row["individual_true_prob_gain"] for row in legal),
                "class_center_true_prob_gain_mean_on_individual_legal": _mean(
                    row["class_center_true_prob_gain"] for row in legal
                ),
                "group_center_true_prob_gain_mean_on_individual_legal": _mean(
                    row["group_center_true_prob_gain"] for row in legal
                ),
                "individual_minus_class_center_gain_mean": _mean(
                    row["individual_minus_class_center_gain"] for row in legal
                ),
                "recall_no_phase_on_individual_legal": _rate(
                    int(row["prediction_no_phase"]) == class_id for row in legal
                ),
                "recall_individual_phase_on_legal": _rate(
                    int(row["prediction_individual_phase"]) == class_id for row in legal
                ),
                "recall_class_center_on_individual_legal": _rate(
                    int(row["prediction_class_center"]) == class_id for row in legal
                ),
                "recall_group_center_on_individual_legal": _rate(
                    int(row["prediction_group_center"]) == class_id for row in legal
                ),
            })
        else:
            for name in (
                "individual_shape_improvement_rate", "individual_pse_improvement_rate",
                "individual_ltae_improvement_rate", "individual_true_prob_improvement_rate",
                "individual_true_prob_gain_mean", "class_center_true_prob_gain_mean_on_individual_legal",
                "group_center_true_prob_gain_mean_on_individual_legal",
                "individual_minus_class_center_gain_mean", "recall_no_phase_on_individual_legal",
                "recall_individual_phase_on_legal", "recall_class_center_on_individual_legal",
                "recall_group_center_on_individual_legal",
            ):
                item[name] = float("nan")
        class_rows.append(item)

    legal_all = [row for row in rows if row["t_registration_legal"]]
    summary = {
        "target_test_samples": len(rows),
        "t_only_legal_samples": len(legal_all),
        "t_only_legal_rate": len(legal_all) / len(rows) if rows else float("nan"),
        "individual_true_prob_improvement_rate": _rate(
            row["individual_true_prob_gain"] > 0 for row in legal_all
        ),
        "individual_true_prob_gain_mean": _mean(row["individual_true_prob_gain"] for row in legal_all),
        "individual_minus_class_center_gain_mean": _mean(
            row["individual_minus_class_center_gain"] for row in legal_all
        ),
    }
    return rows, class_rows, summary


def _plot_class_phase_structure(
    root: Path,
    *,
    classes: Sequence[str],
    sample_rows: Sequence[dict],
    oracle_records: Sequence[TOnlyPhaseRegistration],
    class_centers: Dict[int, dict],
    group: dict,
    mds_samples_per_class: int,
    spaghetti_samples_per_class: int,
    seed: int,
    dpi: int,
) -> List[dict]:
    record_by_id = {int(record.sample_id): record for record in oracle_records}
    row_by_id = {int(row["sample_id"]): row for row in sample_rows}
    summary_rows: List[dict] = []
    rng = np.random.default_rng(seed)
    group_gamma = group["center_gamma"].detach().cpu().double()

    for class_id, class_name in enumerate(classes):
        valid_ids = [
            sample_id for sample_id, row in row_by_id.items()
            if int(row["true_class"]) == class_id
            and bool(row["t_registration_legal"])
            and isinstance(record_by_id[sample_id].gamma, Tensor)
        ]
        if not valid_ids:
            continue
        out = root / "per_class" / f"class_{class_id:02d}_{class_name}"
        out.mkdir(parents=True, exist_ok=True)
        gamma_stack = torch.stack([record_by_id[sample_id].gamma for sample_id in valid_ids]).double()
        class_gamma = class_centers[class_id]["center_gamma"].detach().cpu().double()
        grid = torch.linspace(0.0, 1.0, gamma_stack.shape[1], dtype=torch.float64)

        spaghetti_count = min(spaghetti_samples_per_class, len(valid_ids))
        spaghetti_ids = sorted(rng.choice(valid_ids, size=spaghetti_count, replace=False).tolist())
        fig, ax = plt.subplots(figsize=(9, 5))
        for sample_id in spaghetti_ids:
            gamma = record_by_id[sample_id].gamma.detach().cpu().double()
            ax.plot(grid.numpy() * 365.0, ((gamma - grid) * 365.0).numpy(), alpha=0.18, linewidth=0.8)
        ax.plot(grid.numpy() * 365.0, ((class_gamma - grid) * 365.0).numpy(), linewidth=2.5, label="class center")
        ax.plot(grid.numpy() * 365.0, ((group_gamma - grid) * 365.0).numpy(), linewidth=2.0, linestyle="--", label="M=1 group center")
        ax.axhline(0.0, linewidth=1.0, linestyle=":")
        ax.set_xlabel("Canonical day")
        ax.set_ylabel("source→target displacement (days)")
        ax.set_title(f"{class_id} {class_name}: legal individual Phase trajectories")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "sample_phase_displacement_spaghetti.png", dpi=dpi)
        plt.close(fig)

        mds_count = min(mds_samples_per_class, len(valid_ids))
        mds_ids = sorted(rng.choice(valid_ids, size=mds_count, replace=False).tolist())
        mds_gammas = torch.stack([record_by_id[sample_id].gamma for sample_id in mds_ids]).double()
        distance = phase_distance_matrix(mds_gammas)
        coords = classical_mds(distance, dimensions=2).numpy()
        np.savetxt(out / "phase_distance_matrix.csv", distance.numpy(), delimiter=",")
        mds_rows = []
        for index, sample_id in enumerate(mds_ids):
            row = row_by_id[sample_id]
            mds_rows.append({
                "sample_id": sample_id,
                "mds_1": float(coords[index, 0]),
                "mds_2": float(coords[index, 1]),
                "individual_true_prob_gain": row["individual_true_prob_gain"],
                "individual_minus_class_center_gain": row["individual_minus_class_center_gain"],
            })
        _write_csv(out / "mds_subset.csv", mds_rows)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.7)
        ax.set_title(f"{class_id} {class_name}: Phase-geometry MDS (descriptive only)")
        ax.set_xlabel("MDS1")
        ax.set_ylabel("MDS2")
        fig.tight_layout()
        fig.savefig(out / "sample_phase_mds.png", dpi=dpi)
        plt.close(fig)

        for filename, field, title in (
            ("sample_phase_mds_by_true_prob_gain.png", "individual_true_prob_gain", "Individual Phase true-class probability gain"),
            ("sample_phase_mds_by_center_loss.png", "individual_minus_class_center_gain", "Individual minus class-center gain"),
        ):
            values = np.asarray([float(row_by_id[sample_id][field]) for sample_id in mds_ids])
            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=22, alpha=0.8)
            fig.colorbar(scatter, ax=ax, label=title)
            ax.set_title(f"{class_id} {class_name}: {title}")
            ax.set_xlabel("MDS1")
            ax.set_ylabel("MDS2")
            fig.tight_layout()
            fig.savefig(out / filename, dpi=dpi)
            plt.close(fig)

        x = np.asarray([float(row_by_id[sample_id]["d_gamma_to_class_center"]) for sample_id in valid_ids])
        y = np.asarray([float(row_by_id[sample_id]["individual_minus_class_center_gain"]) for sample_id in valid_ids])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, s=14, alpha=0.6)
        ax.axhline(0.0, linewidth=1.0, linestyle=":")
        ax.set_xlabel("dΓ(individual gamma, class center)")
        ax.set_ylabel("individual - class-center true-prob gain")
        ax.set_title(f"{class_id} {class_name}: center distance vs individual advantage")
        fig.tight_layout()
        fig.savefig(out / "distance_to_class_center_vs_individual_advantage.png", dpi=dpi)
        plt.close(fig)

        upper = distance[np.triu_indices(distance.shape[0], k=1)].numpy()
        summary_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "legal_samples": len(valid_ids),
            "mds_subset_size": mds_count,
            "pairwise_phase_distance_median": float(np.median(upper)) if upper.size else 0.0,
            "pairwise_phase_distance_p90": float(np.quantile(upper, 0.9)) if upper.size else 0.0,
            "pairwise_phase_distance_max": float(np.max(upper)) if upper.size else 0.0,
        })
    return summary_rows


def _balanced_subset(meta, per_class: int, seed: int) -> np.ndarray:
    labels = meta.get_labels()
    parcels = meta.get_parcel_indices()
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in sorted(set(int(value) for value in labels.tolist())):
        positions = np.flatnonzero(labels == class_id)
        count = min(int(per_class), len(positions))
        chosen = rng.choice(positions, size=count, replace=False)
        selected.extend(int(parcels[index]) for index in chosen.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _shape_distance_no_phase(target_cache, source_bank: SourcePrototypeBank, sample_index: int, class_id: int) -> float:
    q = target_cache.structure_srvf_shape[sample_index].detach().cpu().float()
    support = target_cache.structure_support_shape[sample_index].detach().cpu().float()
    proto = source_bank.shape_srvf[class_id].detach().cpu().float()
    proto_support = source_bank.shape_support[class_id].detach().cpu().float()
    grid = target_cache.shape_grid.detach().cpu().float()
    weights = torch.ones_like(grid)
    if weights.numel() > 1:
        weights[[0, -1]] *= 0.5
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    distance = support_aware_q_distance(
        q.unsqueeze(0), proto.unsqueeze(0), support.unsqueeze(0),
        proto_support.unsqueeze(0), weights,
    )
    if not bool(distance.valid[0, 0].item()):
        return float("nan")
    return float(distance.distance[0, 0].item())


def _stage_b_selection(
    *,
    target_cache,
    source_bank: SourcePrototypeBank,
    source_reg_bank: SourceRegistrationPrototypeBank,
    scan_config: PhaseHypothesisScanConfig,
    oracle_records: Sequence[TOnlyPhaseRegistration],
    true_class_by_parcel: Dict[int, int],
    subset_parcels: Sequence[int],
    registration_workers: int,
    cache_path: Path,
) -> tuple[List[dict], List[dict], Dict[int, Tensor], dict]:
    cache_index = {int(sample_id): index for index, sample_id in enumerate(target_cache.sample_ids.tolist())}
    oracle_by_id = {int(record.sample_id): record for record in oracle_records}
    ready_classes = source_reg_bank.ready_classes()
    extra_assignments: List[tuple[int, int]] = []
    sample_ids = []
    true_classes = []
    for parcel in subset_parcels:
        parcel = int(parcel)
        sample_ids.append(parcel)
        true_class = int(true_class_by_parcel[parcel])
        true_classes.append(true_class)
        index = cache_index[parcel]
        for class_id in ready_classes:
            if class_id != true_class:
                extra_assignments.append((index, class_id))

    extra_records = _load_registration_cache(cache_path, sample_ids, true_classes)
    # Stage-B cache stores only extra non-true-class records; validate cardinality.
    expected_extra = len(extra_assignments)
    if extra_records is None or len(extra_records) != expected_extra:
        extra_records = solve_t_only_registrations(
            source_reg_bank, trend_only_cache(target_cache), extra_assignments, scan_config,
            workers=registration_workers, progress_label="SAMPLE_PHASE_STAGE_B_DP",
        )
        _save_registration_cache(cache_path, extra_records, sample_ids, true_classes)
    else:
        print(f"SAMPLE_PHASE_STAGE_B_CACHE_HIT|path={cache_path}|records={len(extra_records)}", flush=True)

    candidate_rows: List[dict] = []
    selection_rows: List[dict] = []
    selected_gamma: Dict[int, Tensor] = {}
    all_records_by_sample: Dict[int, List[TOnlyPhaseRegistration]] = {}
    for record in extra_records:
        all_records_by_sample.setdefault(int(record.sample_id), []).append(record)
    for parcel in subset_parcels:
        parcel = int(parcel)
        true_record = oracle_by_id[parcel]
        all_records_by_sample.setdefault(parcel, []).append(true_record)

    shape_by_sample: Dict[int, List[RawShapeValidation]] = {}
    for parcel in subset_parcels:
        parcel = int(parcel)
        for record in all_records_by_sample[parcel]:
            shape = evaluate_shape_validation(
                record, target_cache=target_cache, source_bank=source_bank
            )
            shape_by_sample.setdefault(parcel, []).append(shape)
            gamma_identity_distance = float("nan")
            displacement = _displacement_stats(record.gamma)
            if record.gamma is not None:
                gamma_identity_distance = _phase_dist(record.gamma, _identity(record.gamma))
            candidate_rows.append({
                "sample_id": parcel,
                "true_class": true_class_by_parcel[parcel],
                "candidate_class": record.class_id,
                "t_candidate_legal": record.t_only_legal,
                "t_reject_reasons": "+".join(record.reject_reasons),
                "t_identity_error": record.t_identity_error,
                "t_registered_error": record.t_registered_error,
                "t_gain_ratio": record.t_gain_ratio,
                "pre_common_support_t": record.pre_common_support_t,
                "gamma_roughness": record.gamma_roughness,
                "phase_deviation": record.phase_deviation,
                "d_gamma_to_identity": gamma_identity_distance,
                **displacement,
                "s_distance_before": _shape_distance_no_phase(
                    target_cache, source_bank, record.sample_index, record.class_id
                ),
                "s_distance_after_candidate_gamma": shape.raw_shape_distance,
                "s_common_support": shape.common_support_shape,
                "s_computable": shape.computable,
                "selected": False,
            })

    selection_correct = []
    for parcel in subset_parcels:
        parcel = int(parcel)
        records = all_records_by_sample[parcel]
        shapes = shape_by_sample[parcel]
        selection = select_raw_shape_candidate(records, shapes)
        true_class = int(true_class_by_parcel[parcel])
        selectable = list(selection.selectable_class_ids)
        shape_map = {shape.class_id: shape for shape in shapes if shape.computable and shape.raw_shape_distance is not None}
        ordered = sorted(
            ((float(shape_map[class_id].raw_shape_distance), class_id) for class_id in selectable),
            key=lambda item: (item[0], item[1]),
        )
        rank = None
        for index, (_distance, class_id) in enumerate(ordered, start=1):
            if class_id == true_class:
                rank = index
                break
        true_distance = None
        if true_class in shape_map:
            true_distance = float(shape_map[true_class].raw_shape_distance)
        selected_class = selection.selected_class_id
        correct = selected_class == true_class if selected_class is not None else False
        selection_correct.append(correct)
        if selected_class is not None:
            chosen = next(record for record in records if record.class_id == selected_class)
            if chosen.gamma is not None:
                selected_gamma[parcel] = chosen.gamma
        selected_phase_d_to_id = float("nan")
        if selected_class is not None:
            chosen = next(record for record in records if record.class_id == selected_class)
            if chosen.gamma is not None:
                selected_phase_d_to_id = _phase_dist(chosen.gamma, _identity(chosen.gamma))
        selection_rows.append({
            "sample_id": parcel,
            "true_class": true_class,
            "selected_class": selected_class,
            "selection_available": selected_class is not None,
            "selection_correct": correct,
            "true_class_candidate_rank": rank,
            "selectable_candidate_count": len(selectable),
            "selected_s_distance": selection.selected_distance,
            "true_class_s_distance": true_distance,
            "selection_margin": selection.margin,
            "selected_phase_d_to_identity": selected_phase_d_to_id,
        })
    selected_by_sample = {
        int(row["sample_id"]): row["selected_class"] for row in selection_rows
    }
    for row in candidate_rows:
        row["selected"] = (
            selected_by_sample.get(int(row["sample_id"])) == int(row["candidate_class"])
        )
    summary = {
        "samples": len(subset_parcels),
        "selection_available_rate": _rate(row["selection_available"] for row in selection_rows),
        "raw_s_selection_accuracy": _rate(selection_correct),
    }
    return candidate_rows, selection_rows, selected_gamma, summary


def _stage_b_downstream(
    *,
    model,
    loader,
    selected_gamma: Dict[int, Tensor],
    oracle_gamma: Dict[int, Tensor],
    selection_rows: List[dict],
    classes: Sequence[str],
    device: torch.device,
) -> dict:
    selection_by_id = {int(row["sample_id"]): row for row in selection_rows}
    detailed = []
    for raw_batch in loader:
        batch = phasevis._move_batch(raw_batch, device)
        backbone = model.forward_backbone(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch.get("extra"), time_mask=batch.get("time_mask")
        )
        native = backbone.normalized_positions.detach()
        mask = backbone.time_mask.detach()
        labels = batch["label"].long()
        parcels = batch["parcel_index"].detach().cpu().long()
        no_out = model.forward_from_backbone(backbone, batch["positions"], batch.get("extra"), return_geometry=False)
        selected_pos, selected_available = _individual_positions(native, mask, parcels, selected_gamma)
        oracle_pos, oracle_available = _individual_positions(native, mask, parcels, oracle_gamma)
        selected_out = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"),
            temporal_positions_override=selected_pos, return_geometry=False
        )
        oracle_out = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"),
            temporal_positions_override=oracle_pos, return_geometry=False
        )
        outputs = {"no_phase": no_out, "selected_phase": selected_out, "oracle_phase": oracle_out}
        for row_index, parcel in enumerate(parcels.tolist()):
            parcel = int(parcel)
            true_class = int(labels[row_index].item())
            row = selection_by_id[parcel]
            for variant, output in outputs.items():
                available = True
                if variant == "selected_phase":
                    available = bool(selected_available[row_index].item())
                elif variant == "oracle_phase":
                    available = bool(oracle_available[row_index].item())
                if not available:
                    row[f"prediction_{variant}"] = None
                    row[f"true_probability_{variant}"] = float("nan")
                    continue
                probs = torch.softmax(output.logits[row_index].float(), dim=-1)
                row[f"prediction_{variant}"] = int(output.logits[row_index].argmax().item())
                row[f"true_probability_{variant}"] = float(probs[true_class].item())
            if row["selection_available"]:
                row["selected_phase_true_prob_gain"] = (
                    row["true_probability_selected_phase"] - row["true_probability_no_phase"]
                )
            else:
                row["selected_phase_true_prob_gain"] = float("nan")
            if math.isfinite(float(row["true_probability_oracle_phase"])):
                row["oracle_phase_true_prob_gain"] = (
                    row["true_probability_oracle_phase"] - row["true_probability_no_phase"]
                )
            else:
                row["oracle_phase_true_prob_gain"] = float("nan")
            detailed.append(row)

    correct_rows = [row for row in detailed if row["selection_available"] and row["selection_correct"]]
    wrong_rows = [row for row in detailed if row["selection_available"] and not row["selection_correct"]]
    return {
        "selection_sample_rows": detailed,
        "conditional": {
            "selected_class_correct": {
                "count": len(correct_rows),
                "selected_phase_beneficial_rate": _rate(
                    row["selected_phase_true_prob_gain"] > 0 for row in correct_rows
                ),
                "selected_phase_true_prob_gain_mean": _mean(
                    row["selected_phase_true_prob_gain"] for row in correct_rows
                ),
            },
            "selected_class_wrong": {
                "count": len(wrong_rows),
                "selected_phase_beneficial_rate": _rate(
                    row["selected_phase_true_prob_gain"] > 0 for row in wrong_rows
                ),
                "selected_phase_true_prob_gain_mean": _mean(
                    row["selected_phase_true_prob_gain"] for row in wrong_rows
                ),
            },
        },
    }


def _plot_stage_b(output_dir: Path, rows: Sequence[dict], classes: Sequence[str], dpi: int) -> List[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_count = len(classes)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for row in rows:
        if row["selected_class"] is not None:
            matrix[int(row["true_class"]), int(row["selected_class"])] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, aspect="auto")
    fig.colorbar(image, ax=ax, label="samples")
    ax.set_xticks(range(class_count), labels=[str(i) for i in range(class_count)])
    ax.set_yticks(range(class_count), labels=[str(i) for i in range(class_count)])
    ax.set_xlabel("selected source class by raw S distance")
    ax.set_ylabel("oracle true class")
    ax.set_title("Stage B raw-S candidate selection confusion")
    fig.tight_layout()
    fig.savefig(output_dir / "candidate_selection_confusion_matrix.png", dpi=dpi)
    plt.close(fig)

    ranks = [int(row["true_class_candidate_rank"]) for row in rows if row["true_class_candidate_rank"] is not None]
    if ranks:
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.arange(0.5, max(ranks) + 1.5, 1.0)
        ax.hist(ranks, bins=bins)
        ax.set_xlabel("rank of true-class candidate by raw S distance")
        ax.set_ylabel("samples")
        ax.set_title("True-class candidate rank")
        fig.tight_layout()
        fig.savefig(output_dir / "true_class_candidate_rank_distribution.png", dpi=dpi)
        plt.close(fig)

    margins = [float(row["selection_margin"]) for row in rows if row["selection_margin"] is not None and math.isfinite(float(row["selection_margin"]))]
    if margins:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(margins, bins=40)
        ax.set_xlabel("second-best raw S distance - best raw S distance")
        ax.set_ylabel("samples")
        ax.set_title("Raw-S selection margin")
        fig.tight_layout()
        fig.savefig(output_dir / "selection_margin_distribution.png", dpi=dpi)
        plt.close(fig)

    class_rows = []
    for class_id, class_name in enumerate(classes):
        subset = [row for row in rows if int(row["true_class"]) == class_id]
        if not subset:
            continue
        class_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "support": len(subset),
            "selection_available_rate": _rate(row["selection_available"] for row in subset),
            "selection_recall": _rate(row["selected_class"] == class_id for row in subset),
            "mean_true_class_candidate_rank": _mean(
                row["true_class_candidate_rank"] for row in subset
            ),
            "mean_selection_margin": _mean(row["selection_margin"] for row in subset),
            "selected_phase_true_prob_gain_mean": _mean(
                row.get("selected_phase_true_prob_gain") for row in subset
            ),
        })
    return class_rows


def _plot_joint_stage_a_stage_b(
    output_dir: Path,
    *,
    stage_a_rows: Sequence[dict],
    stage_b_rows: Sequence[dict],
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage_a_rows:
        labels = [str(row["class_id"]) for row in stage_a_rows]
        x = np.arange(len(stage_a_rows), dtype=float)
        individual = np.asarray([float(row.get("individual_true_prob_gain_mean", np.nan)) for row in stage_a_rows])
        center = np.asarray([float(row.get("class_center_true_prob_gain_mean_on_individual_legal", np.nan)) for row in stage_a_rows])
        width = 0.38
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, individual, width=width, label="individual Phase")
        ax.bar(x + width / 2, center, width=width, label="class-center Phase")
        ax.axhline(0.0, linewidth=1.0, linestyle=":")
        ax.set_xticks(x, labels=labels)
        ax.set_xlabel("true class")
        ax.set_ylabel("mean true-class probability gain vs No Phase")
        ax.set_title("Stage A: individual vs class-center Phase")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "sample_vs_class_center_gain_by_class.png", dpi=dpi)
        plt.close(fig)

        sx = np.asarray([float(row.get("individual_shape_improvement_rate", np.nan)) for row in stage_a_rows])
        sy = np.asarray([float(row.get("individual_true_prob_improvement_rate", np.nan)) for row in stage_a_rows])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sx, sy, s=45)
        for index, label in enumerate(labels):
            if math.isfinite(sx[index]) and math.isfinite(sy[index]):
                ax.annotate(label, (sx[index], sy[index]), xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("S-SRVF improvement rate from individual Phase")
        ax.set_ylabel("true-class probability improvement rate")
        ax.set_title("Functional vs classifier validity by class")
        fig.tight_layout()
        fig.savefig(output_dir / "geometry_vs_classifier_validity.png", dpi=dpi)
        plt.close(fig)


def _write_readme(
    path: Path,
    *,
    calibration_checkpoint: Path,
    model_checkpoint: Path,
    source: str,
    target: str,
    stage_b_samples_per_class: int,
    mds_samples_per_class: int,
) -> None:
    text = f"""# 06 Sample-level Phase Validity Diagnostic

## 实验目的

本实验只做 held-out post-hoc diagnosis，不改变训练、Stable Label、Domain Phase 分组或 M=1/M=2 机制。它检查的是 `sample gamma -> class center` 这一层；05 检查的是 `class center -> group center`。

核心问题只有两个：

1. T-SRVF 为单个目标样本产生的 sample-level Phase 本身是否可信；
2. 同一真实类别内部是否存在不能由单一 class-center Phase 保留的、重复出现且有下游价值的 Phase 结构。

本实验**不做聚类、不决定组数、不定义新阈值，也不把 MDS 可视化区域解释成 Domain Phase group**。

## 输入

- source: `{source}`
- target: `{target}`
- frozen model checkpoint: `{model_checkpoint}`
- calibration checkpoint (class centers / M=1 group center): `{calibration_checkpoint}`
- Stage A population: 完整 held-out target-test
- Stage B population: target-test class-balanced 固定子集，每类最多 `{stage_b_samples_per_class}` 个样本
- MDS: 每类合法 Stage-A sample 固定 seed 最多 `{mds_samples_per_class}` 个，仅用于可视化

## Stage A：oracle true-class sample Phase

真实 target label `y_i` **只用于指定正确的 source T prototype**，得到 `gamma_{{i,y_i}}`。因此 Stage A 是 oracle-only diagnostic。

关键独立性约束：sample gamma 的产生和合法性筛选只允许读取：

- T-SRVF registration objective；
- T common support；
- warp finite / endpoint / monotonicity；
- minimum increment；
- local speed；
- roughness；
- phase deviation；
- T registration gain。

**S-SRVF 不参与 gamma 生成，也不参与 Stage-A admission。** S 只在 gamma 固定之后作为独立验证证据。因此 T registration error 的下降只记为 proposal-fit diagnostic，不能自证 Phase validity。

对同一 target-test sample 比较四路：

1. No Phase；
2. Individual Phase `gamma_{{i,y_i}}`；
3. oracle Class-center Phase `gamma_bar_{{y_i}}`；
4. M=1 Group-center Phase `delta`。

PSE、LTAE、classifier 全冻结、`eval()`，无 optimizer、Teacher refresh、Stable Label refresh。

## 类内结构可视化

`sample_phase_displacement_spaghetti.png`、pairwise Fisher-Rao distance 和 classical MDS 只回答：**“一类一个 class center 是否可能过强？”**

MDS 不参与任何 clustering/group-number 决策。完整 target-test 用于 registration 和总体统计；pairwise/MDS 只从合法 samples 中按固定 seed 子采样，避免无价值的 O(N^2) 图形计算。

真正支持未来 sample-level Phase grouping 至少需要同时看到三个核心条件：

1. **Individual Phase 有效**：在没有参与 T 求解的 S-SRVF、冻结 classifier/LTAE/PSE temporal diagnostic 中具有独立收益；
2. **类内存在重复结构**：sample gamma 的差异不是随机散布，而出现重复的 Phase 形态；
3. **Class center 丢失收益**：Individual Phase 系统性优于 class-center Phase。

三者不能同时成立时，本实验不能得出“应该进行 sample-level grouping”。

## Stage B：T 提出 + raw S 选择

Stage B 不使用 target label 做选择。所有 ready source classes 分别由 T 提出 candidate gamma；只保留满足 **T-only legality** 且 S 距离可计算的 candidate，然后用最简单的：

`argmin_c raw aligned S-SRVF distance`

选择 `(selected class, selected Phase)`。

这里**不加入** classifier、Teacher、Stable Label、class balance、距离标准化、source-range normalization 或任何新的校准。raw S 距离只是 06 的最简单基线。如果 confusion 显示大量样本被吸向某一 source class，这本身就是“raw S distance 跨类别不可直接比较”的诊断结果，而不是本实验中要修掉的问题。

真实 label 只在选择完成之后用于：selection accuracy、confusion、true-class candidate rank 和 selected Phase 的 oracle audit。

## 文件说明

- `01_oracle_true_class_phase/oracle_sample_level.csv`: 完整 target-test Stage-A sample-level registration / 四路下游指标。
- `01_oracle_true_class_phase/class_sample_phase_summary.csv`: 每个真实类别的完整统计。
- `01_oracle_true_class_phase/per_class/*`: displacement、MDS、center-distance 与 individual advantage 图。
- `02_candidate_phase_selection/candidate_level.csv`: Stage-B sample × source-class candidate 表。
- `02_candidate_phase_selection/selection_sample_level.csv`: raw-S 选择结果及 selected/oracle Phase 下游结果。
- `02_candidate_phase_selection/selection_class_summary.csv`: 每类 selection recall 等统计。
- `02_candidate_phase_selection/candidate_selection_confusion_matrix.png`: oracle true class vs raw-S selected class。
- `03_joint_diagnosis/joint_class_summary.csv`: Stage A 与 Stage B 类别级摘要合并。
- `cache/`: expensive exact-DP 结果缓存；仅用于相同 held-out population/config 的诊断重跑。
- `summary.json`: 机器可读全局摘要。
- `manifest.json`: 输出协议。

## 结果解释边界

- Stage A 好而 Stage B 差：sample registration 本身可行，主要瓶颈在 `T propose + raw S select` 的类别—Phase hypothesis selection。
- Stage A individual gamma 普遍有害：问题在更上游的 registration-derived sample Phase；聚类不能解决根本问题。
- Individual gamma 有效、类内 Phase geometry 有重复结构、且 class center 明显损失 individual 收益：这才构成进一步研究 sample-level Phase grouping 的强证据。
- S-SRVF 改善但 frozen classifier 下降：只能说明 functional structural validity，不等于 classification usefulness。
- 06 任何结果都不能直接决定最终 Domain Phase 应该分几组；跨类别重复模式必须交回理论设计窗口另行定义。
"""
    path.write_text(text, encoding="utf-8")


def run(args) -> dict:
    checkpoint_path = args.calibration_checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    runtime = checkpoint.get("runtime_config")
    if not isinstance(runtime, dict):
        raise ValueError("calibration checkpoint is missing runtime_config")
    classes = [str(value) for value in runtime["classes"]]
    source = str(runtime["source"])
    target = str(runtime["target"])
    seed = int(runtime["seed"])
    data_root = str(args.data_root or runtime["data_root"])
    closed_set = bool(runtime.get("closed_set", True))
    combine = bool(runtime.get("combine_spring_and_winter", False))
    time_mode = str(runtime.get("time_coordinate_mode", "canonical_day_of_year"))
    val_ratio = float(runtime.get("val_ratio", 0.1))
    test_ratio = float(runtime.get("test_ratio", 0.2))
    fold = int(args.fold)
    class_centers = classdiag._class_center_payloads(checkpoint)
    group = classdiag._final_m1_group(checkpoint)
    source_bank = _source_bank(checkpoint)
    scan_config = _scan_config(runtime, args.registration_workers)

    device = torch.device(args.device)
    model_checkpoint = torch.load(args.model_checkpoint.resolve(), map_location="cpu", weights_only=False)
    model = phasevis._build_model(runtime, checkpoint, device, model_checkpoint=model_checkpoint)
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
    source_test_parcels = np.asarray(sorted(splits[source]["test"]), dtype=np.int64)
    target_test_parcels = np.asarray(sorted(splits[target]["test"]), dtype=np.int64)
    source_test_meta = phasevis._metadata_dataset(
        data_root, source, classes, splits[source]["test"], closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
    )
    target_test_meta = phasevis._metadata_dataset(
        data_root, target, classes, splits[target]["test"], closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
    )
    source_train_loader = phasevis._selected_loader(
        data_root, source, classes, source_train_parcels, closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    source_test_loader = phasevis._selected_loader(
        data_root, source, classes, source_test_parcels, closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    target_test_loader = phasevis._selected_loader(
        data_root, target, classes, target_test_parcels, closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    true_class_by_parcel = _label_map(target_test_meta)

    output_dir = args.output_dir.resolve()
    stage_a_dir = output_dir / "01_oracle_true_class_phase"
    stage_b_dir = output_dir / "02_candidate_phase_selection"
    joint_dir = output_dir / "03_joint_diagnosis"
    cache_dir = output_dir / "cache"
    for directory in (stage_a_dir, stage_b_dir, joint_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    reg_extractor = build_stage2_registration_extractor(
        model, device=device, k_reg=scan_config.k_reg
    )
    source_reg_bank = _load_or_build_registration_bank(
        cache_dir / "source_registration_bank.pt", model=model,
        source_train_loader=DeviceBatchLoader(source_train_loader, device),
        num_classes=len(classes), device=device, reg_extractor=reg_extractor,
    )

    print(
        "SAMPLE_PHASE_STAGE_A_GEOMETRY_CACHE|status=start"
        f"|target_test={len(target_test_parcels)}",
        flush=True,
    )
    target_cache = targetscan._build_target_geometry_cache(
        model, DeviceBatchLoader(target_test_loader, device), device=device,
        shape_grid=model.temporal_module.structure_geometry.functional_lift.canonical_grid.detach().cpu(),
        shape_extractor=model.temporal_module.structure_geometry,
        reg_extractor=reg_extractor,
    )
    dataset = target_test_loader.dataset
    get_parcel_indices = getattr(dataset, "get_parcel_indices", None)
    if not callable(get_parcel_indices):
        raise TypeError("06 target dataset must expose get_parcel_indices()")
    target_cache = replace(
        target_cache,
        sample_ids=remap_local_sample_ids_to_parcels(
            target_cache.sample_ids,
            get_parcel_indices(),
        ),
    )
    print(
        "SAMPLE_PHASE_STAGE_A_GEOMETRY_CACHE|status=ready"
        f"|cached={len(target_cache.sample_ids)}",
        flush=True,
    )
    cache_sample_ids = [int(value) for value in target_cache.sample_ids.tolist()]
    true_classes = [int(true_class_by_parcel[sample_id]) for sample_id in cache_sample_ids]
    assignments = [(index, true_classes[index]) for index in range(len(cache_sample_ids))]
    stage_a_cache = cache_dir / "stage_a_t_only_registrations.pt"
    oracle_records = _load_registration_cache(stage_a_cache, cache_sample_ids, true_classes)
    if oracle_records is None or len(oracle_records) != len(assignments):
        print(
            "SAMPLE_PHASE_STAGE_A_DP_START|"
            f"pairs={len(assignments)}|oracle_true_class_only=true|s_gate=false",
            flush=True,
        )
        oracle_records = solve_t_only_registrations(
            source_reg_bank, trend_only_cache(target_cache), assignments, scan_config,
            workers=args.registration_workers, progress_label="SAMPLE_PHASE_STAGE_A_DP",
            max_target_samples_per_pool=args.dp_target_chunk_size,
        )
        _save_registration_cache(stage_a_cache, oracle_records, cache_sample_ids, true_classes)
    else:
        print(f"SAMPLE_PHASE_STAGE_A_CACHE_HIT|path={stage_a_cache}", flush=True)

    shape_by_key: Dict[tuple[int, int], RawShapeValidation] = {}
    for record in oracle_records:
        if record.t_only_legal:
            shape = evaluate_shape_validation(
                record, target_cache=target_cache, source_bank=source_bank
            )
            shape_by_key[(record.sample_id, record.class_id)] = shape

    source_pse = classdiag._build_source_pse_centers(
        model, source_test_loader, device=device, grid_size=args.pse_grid_size
    )
    stage_a_sample_rows, stage_a_class_rows, stage_a_summary = _stage_a_downstream(
        model=model, target_loader=target_test_loader, source_pse=source_pse,
        checkpoint=checkpoint, classes=classes, class_centers=class_centers,
        group=group, oracle_records=oracle_records, shape_by_key=shape_by_key,
        device=device, pse_grid_size=args.pse_grid_size,
    )
    _write_csv(stage_a_dir / "oracle_sample_level.csv", stage_a_sample_rows)
    _write_csv(stage_a_dir / "class_sample_phase_summary.csv", stage_a_class_rows)
    phase_structure_rows = _plot_class_phase_structure(
        stage_a_dir, classes=classes, sample_rows=stage_a_sample_rows,
        oracle_records=oracle_records, class_centers=class_centers, group=group,
        mds_samples_per_class=args.mds_samples_per_class,
        spaghetti_samples_per_class=args.spaghetti_samples_per_class,
        seed=args.visualization_seed, dpi=args.dpi,
    )
    phase_structure_map = {int(row["class_id"]): row for row in phase_structure_rows}
    for row in stage_a_class_rows:
        row.update({
            key: value for key, value in phase_structure_map.get(int(row["class_id"]), {}).items()
            if key not in {"class_id", "class_name"}
        })
    _write_csv(stage_a_dir / "class_sample_phase_summary.csv", stage_a_class_rows)

    subset_parcels = _balanced_subset(
        target_test_meta, args.stage_b_samples_per_class, args.stage_b_seed
    )
    candidate_rows, selection_rows, selected_gamma, stage_b_selection_summary = _stage_b_selection(
        target_cache=target_cache, source_bank=source_bank, source_reg_bank=source_reg_bank,
        scan_config=scan_config, oracle_records=oracle_records,
        true_class_by_parcel=true_class_by_parcel, subset_parcels=subset_parcels,
        registration_workers=args.registration_workers,
        cache_path=cache_dir / "stage_b_extra_t_only_registrations.pt",
    )
    oracle_gamma = {
        int(record.sample_id): record.gamma
        for record in oracle_records
        if record.t_only_legal and isinstance(record.gamma, Tensor)
    }
    stage_b_loader = phasevis._selected_loader(
        data_root, target, classes, subset_parcels, closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    stage_b_downstream = _stage_b_downstream(
        model=model, loader=stage_b_loader, selected_gamma=selected_gamma,
        oracle_gamma=oracle_gamma, selection_rows=selection_rows,
        classes=classes, device=device,
    )
    selection_rows = stage_b_downstream["selection_sample_rows"]
    _write_csv(stage_b_dir / "candidate_level.csv", candidate_rows)
    _write_csv(stage_b_dir / "selection_sample_level.csv", selection_rows)
    selection_class_rows = _plot_stage_b(stage_b_dir, selection_rows, classes, args.dpi)
    _write_csv(stage_b_dir / "selection_class_summary.csv", selection_class_rows)

    stage_a_by_class = {int(row["class_id"]): row for row in stage_a_class_rows}
    stage_b_by_class = {int(row["class_id"]): row for row in selection_class_rows}
    joint_rows = []
    for class_id, class_name in enumerate(classes):
        row = {"class_id": class_id, "class_name": class_name}
        for prefix, source_rows in (("stage_a", stage_a_by_class), ("stage_b", stage_b_by_class)):
            for key, value in source_rows.get(class_id, {}).items():
                if key not in {"class_id", "class_name"}:
                    row[f"{prefix}_{key}"] = value
        joint_rows.append(row)
    _write_csv(joint_dir / "joint_class_summary.csv", joint_rows)
    _plot_joint_stage_a_stage_b(
        joint_dir, stage_a_rows=stage_a_class_rows, stage_b_rows=selection_class_rows, dpi=args.dpi
    )

    summary = {
        "experiment": "06_sample_level_phase_validity_diagnostic",
        "source": source,
        "target": target,
        "fold": fold,
        "calibration_checkpoint": str(checkpoint_path),
        "model_checkpoint": str(args.model_checkpoint.resolve()),
        "model_updates": 0,
        "teacher_refreshes": 0,
        "stable_label_refreshes": 0,
        "phase_group_decision_modified": False,
        "clustering_performed": False,
        "group_count_selected": False,
        "stage_a": {
            "oracle_true_label_use": "select correct source T prototype only",
            "full_target_test": True,
            "sample_count": len(cache_sample_ids),
            "s_used_for_gamma_generation_or_legality": False,
            "summary": stage_a_summary,
        },
        "stage_b": {
            "selection_rule": "minimum raw aligned S-SRVF distance among T-only-legal candidates",
            "classifier_used_for_selection": False,
            "teacher_used_for_selection": False,
            "stable_label_used_for_selection": False,
            "distance_calibration_used": False,
            "samples_per_class": args.stage_b_samples_per_class,
            "sample_count": len(subset_parcels),
            "selection_summary": stage_b_selection_summary,
            "conditional_phase_validity": stage_b_downstream["conditional"],
        },
        "sample_level_grouping_support_requires": [
            "individual_phase_has_independent_benefit",
            "within_class_phase_differences_show_repeated_structure",
            "class_center_systematically_loses_individual_benefit",
        ],
    }
    _json_dump(output_dir / "summary.json", summary)
    _json_dump(output_dir / "manifest.json", {
        "experiment": summary["experiment"],
        "oracle_only_stage_a": True,
        "stage_b_oracle_labels_used_after_selection_only": True,
        "clustering": "none",
        "files": {
            "01_oracle_true_class_phase": "full target-test T-only oracle registration and independent validation",
            "02_candidate_phase_selection": "T proposals + raw S selection on fixed class-balanced subset",
            "03_joint_diagnosis": "joined class-level Stage A/Stage B summary",
            "cache": "expensive exact-DP diagnostic caches",
            "summary.json": "machine-readable protocol and global results",
            "README_中文说明.md": "full diagnostic semantics and interpretation boundaries",
        },
    })
    _write_readme(
        output_dir / "README_中文说明.md",
        calibration_checkpoint=checkpoint_path,
        model_checkpoint=args.model_checkpoint.resolve(), source=source, target=target,
        stage_b_samples_per_class=args.stage_b_samples_per_class,
        mds_samples_per_class=args.mds_samples_per_class,
    )
    print(
        "SAMPLE_PHASE_VALIDITY_COMPLETE|"
        f"output={output_dir}|stage_a_samples={len(cache_sample_ids)}"
        f"|stage_b_samples={len(subset_parcels)}"
        f"|stage_b_selection_accuracy={stage_b_selection_summary['raw_s_selection_accuracy']:.6f}",
        flush=True,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-checkpoint", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--registration-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pse-grid-size", type=int, default=128)
    parser.add_argument("--stage-b-samples-per-class", type=int, default=128)
    parser.add_argument("--stage-b-seed", type=int, default=106)
    parser.add_argument("--mds-samples-per-class", type=int, default=256)
    parser.add_argument("--spaghetti-samples-per-class", type=int, default=128)
    parser.add_argument("--visualization-seed", type=int, default=206)
    parser.add_argument("--dp-target-chunk-size", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    for name in (
        "registration_workers", "batch_size", "num_workers", "pse_grid_size",
        "stage_b_samples_per_class", "mds_samples_per_class",
        "spaghetti_samples_per_class", "dp_target_chunk_size",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    run(args)


if __name__ == "__main__":
    main()
