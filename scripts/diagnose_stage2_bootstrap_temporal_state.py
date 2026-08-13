#!/usr/bin/env python3
"""12: Stage-2 bootstrap temporal-state diagnostic.

Frozen-model, no-training diagnostic comparing only target LTAE time states:
Identity, the official TimeMatch-style unlabeled scalar bootstrap, and an
oracle nonlinear class-balanced shared Phase estimated exclusively from
TARGET-TRAIN true-class registrations.  Target labels are never used to fit
Identity or TimeMatch.  Target-val/test labels are evaluation-only.

The oracle target-train registration cache is resumable because the first run
may require many exact-DP calls.  Production legality, pseudo labels, Stable
Label, Teacher/Student and any Stage-2 training are intentionally absent.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import compare_stage2_phase_vs_timematch_shift as scalarcmp
import diagnose_sample_level_phase_validity as samplediag
import visualize_stage2_phase_alignment as phasevis
from methods.structure_da.confirmed_phase_view import align_target_positions_to_source
from methods.structure_da.registration_geometry import evaluate_registration_geometry
from methods.structure_da.sample_phase_diagnostic import (
    TOnlyPhaseRegistration,
    TRegistrationGeometryCache,
    solve_t_only_registrations,
)
from methods.structure_da.shared_domain_phase_diagnostic import (
    class_balanced_weights,
    phase_distance_to_identity_for_gamma,
    weighted_frechet_mean_gamma,
)
from methods.structure_da.stage2_bootstrap_diagnostic import (
    add_identity_deltas,
    classwise_semantic_distribution,
    confusion_matrix,
    hard_transition_rows,
    overall_metrics,
    per_class_metrics,
    row_normalize_confusion,
)
from methods.structure_da.stage2_trainer import DeviceBatchLoader, build_stage2_registration_extractor
from methods.structure_da.temporal_registration import invert_monotone_warp

ORACLE_CACHE_SCHEMA = "stage2_bootstrap12_target_train_oracle_gamma_v1"
PROTOCOL = "12_stage2_bootstrap_temporal_state_diagnostic"
CONDITIONS = ("identity", "timematch_scalar", "oracle_shared")


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _record_payload(record: TOnlyPhaseRegistration) -> dict:
    return samplediag._registration_to_payload(record)


def _record_from_payload(payload: dict) -> TOnlyPhaseRegistration:
    return samplediag._registration_from_payload(payload)


def _load_partial_oracle_cache(
    path: Path,
    *,
    expected_ids: Sequence[int],
    true_class_by_parcel: Mapping[int, int],
    source: str,
    target: str,
    seed: int,
    fold: int,
    model_checkpoint: Path,
) -> dict[int, TOnlyPhaseRegistration]:
    if not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != ORACLE_CACHE_SCHEMA:
        raise ValueError(f"unsupported experiment-12 oracle cache schema: {payload.get('schema')!r}")
    for key, expected in (("source", source), ("target", target), ("seed", seed), ("fold", fold)):
        if str(payload.get(key)) != str(expected):
            raise ValueError(f"experiment-12 oracle cache {key} mismatch")
    cached_model = payload.get("model_checkpoint")
    if cached_model is None or Path(str(cached_model)).resolve() != model_checkpoint.resolve():
        raise ValueError("experiment-12 oracle cache Stage-1 checkpoint mismatch")
    expected_set = set(map(int, expected_ids))
    records: dict[int, TOnlyPhaseRegistration] = {}
    for row in payload.get("records", ()):
        record = _record_from_payload(row)
        sid = int(record.sample_id)
        if sid not in expected_set:
            raise ValueError("experiment-12 oracle cache contains parcel outside target-train")
        if int(record.class_id) != int(true_class_by_parcel[sid]):
            raise ValueError("experiment-12 oracle cache true-class identity mismatch")
        if sid in records:
            raise ValueError("duplicate parcel in experiment-12 oracle cache")
        records[sid] = record
    print(
        f"BOOTSTRAP12_ORACLE_TRAIN_CACHE_RESUME|path={path}|completed={len(records)}/{len(expected_ids)}",
        flush=True,
    )
    return records


def _save_partial_oracle_cache(
    path: Path,
    records: Mapping[int, TOnlyPhaseRegistration],
    *,
    expected_ids: Sequence[int],
    true_class_by_parcel: Mapping[int, int],
    source: str,
    target: str,
    seed: int,
    fold: int,
    model_checkpoint: Path,
) -> None:
    ordered = [records[sid] for sid in expected_ids if sid in records]
    payload = {
        "schema": ORACLE_CACHE_SCHEMA,
        "source": source,
        "target": target,
        "seed": int(seed),
        "fold": int(fold),
        "model_checkpoint": str(model_checkpoint.resolve()),
        "expected_count": len(expected_ids),
        "completed_count": len(ordered),
        "sample_ids": [int(record.sample_id) for record in ordered],
        "true_classes": [int(true_class_by_parcel[int(record.sample_id)]) for record in ordered],
        "records": [_record_payload(record) for record in ordered],
        "production_legality_filter": False,
        "target_true_label_use": "oracle-only correct source class for target-train registration",
    }
    _atomic_torch_save(payload, path)


def _flush_registration_chunk(
    *,
    source_reg_bank,
    scan_config,
    chunk_rows: list[tuple[int, int, Tensor, Tensor, Tensor]],
    global_index_by_parcel: Mapping[int, int],
    workers: int,
    dp_target_chunk_size: int,
) -> list[TOnlyPhaseRegistration]:
    if not chunk_rows:
        return []
    registration_grid = source_reg_bank.registration_grid.detach().cpu().double()
    cache = TRegistrationGeometryCache(
        sample_ids=torch.tensor([row[0] for row in chunk_rows], dtype=torch.long),
        trend_srvf_reg=torch.stack([row[2] for row in chunk_rows]),
        trend_support_reg=torch.stack([row[3] for row in chunk_rows]),
        trend_valid=torch.stack([row[4] for row in chunk_rows]),
        registration_grid=registration_grid,
    )
    assignments = [(index, int(row[1])) for index, row in enumerate(chunk_rows)]
    solved = solve_t_only_registrations(
        source_reg_bank,
        cache,
        assignments,
        scan_config,
        workers=workers,
        progress_label="BOOTSTRAP12_ORACLE_TRAIN_DP",
        max_target_samples_per_pool=max(1, min(int(dp_target_chunk_size), len(chunk_rows))),
    )
    return [
        replace(record, sample_index=int(global_index_by_parcel[int(record.sample_id)]))
        for record in solved
    ]


@torch.no_grad()
def _build_target_train_oracle_gamma_cache(
    *,
    model,
    target_loader,
    source_reg_bank,
    reg_extractor,
    scan_config,
    cache_path: Path,
    expected_ids: Sequence[int],
    true_class_by_parcel: Mapping[int, int],
    source: str,
    target: str,
    seed: int,
    fold: int,
    model_checkpoint: Path,
    device: torch.device,
    workers: int,
    geometry_chunk_size: int,
    dp_target_chunk_size: int,
) -> list[TOnlyPhaseRegistration]:
    records = _load_partial_oracle_cache(
        cache_path,
        expected_ids=expected_ids,
        true_class_by_parcel=true_class_by_parcel,
        source=source,
        target=target,
        seed=seed,
        fold=fold,
        model_checkpoint=model_checkpoint,
    )
    expected_set = set(map(int, expected_ids))
    if set(records) == expected_set:
        print(f"BOOTSTRAP12_ORACLE_TRAIN_CACHE_HIT|path={cache_path}|count={len(records)}", flush=True)
        return [records[int(sid)] for sid in expected_ids]

    global_index = {int(sid): index for index, sid in enumerate(expected_ids)}
    chunk_rows: list[tuple[int, int, Tensor, Tensor, Tensor]] = []
    was_training = model.training
    model.eval()
    processed_geometry = 0
    try:
        for raw_batch in target_loader:
            batch = phasevis._move_batch(raw_batch, device)
            parcels = batch.get("parcel_index")
            labels = batch.get("label")
            if not isinstance(parcels, Tensor) or not isinstance(labels, Tensor):
                raise ValueError("target-train batch must contain parcel_index and label")
            unresolved = [
                index for index, sid in enumerate(parcels.detach().cpu().tolist())
                if int(sid) not in records
            ]
            if not unresolved:
                continue
            output = model(
                batch["pixels"], batch["valid_pixels"], batch["positions"],
                batch.get("extra"), return_geometry=True,
            )
            reg = evaluate_registration_geometry(
                output.trend, output.positions, output.mask, reg_extractor
            )
            for index in unresolved:
                sid = int(parcels[index].item())
                label = int(labels[index].item())
                if sid not in expected_set:
                    raise ValueError("target-train loader produced parcel outside reconstructed split")
                if label != int(true_class_by_parcel[sid]):
                    raise ValueError("target-train loader label differs from metadata label")
                chunk_rows.append((
                    sid,
                    label,
                    reg.trend_srvf[index].detach().cpu(),
                    reg.trend_support[index].detach().cpu(),
                    reg.trend_valid[index].detach().cpu(),
                ))
                processed_geometry += 1
            if len(chunk_rows) >= int(geometry_chunk_size):
                solved = _flush_registration_chunk(
                    source_reg_bank=source_reg_bank,
                    scan_config=scan_config,
                    chunk_rows=chunk_rows,
                    global_index_by_parcel=global_index,
                    workers=workers,
                    dp_target_chunk_size=dp_target_chunk_size,
                )
                records.update({int(record.sample_id): record for record in solved})
                _save_partial_oracle_cache(
                    cache_path, records,
                    expected_ids=expected_ids,
                    true_class_by_parcel=true_class_by_parcel,
                    source=source, target=target, seed=seed, fold=fold,
                    model_checkpoint=model_checkpoint,
                )
                print(
                    f"BOOTSTRAP12_ORACLE_TRAIN_CACHE_PROGRESS|completed={len(records)}/{len(expected_ids)}"
                    f"|new_geometry={processed_geometry}", flush=True,
                )
                chunk_rows = []
        if chunk_rows:
            solved = _flush_registration_chunk(
                source_reg_bank=source_reg_bank,
                scan_config=scan_config,
                chunk_rows=chunk_rows,
                global_index_by_parcel=global_index,
                workers=workers,
                dp_target_chunk_size=dp_target_chunk_size,
            )
            records.update({int(record.sample_id): record for record in solved})
            _save_partial_oracle_cache(
                cache_path, records,
                expected_ids=expected_ids,
                true_class_by_parcel=true_class_by_parcel,
                source=source, target=target, seed=seed, fold=fold,
                model_checkpoint=model_checkpoint,
            )
    finally:
        model.train(was_training)

    missing = [sid for sid in expected_ids if int(sid) not in records]
    if missing:
        raise RuntimeError(f"target-train oracle gamma cache incomplete; missing {len(missing)} parcels")
    print(f"BOOTSTRAP12_ORACLE_TRAIN_CACHE_READY|path={cache_path}|count={len(records)}", flush=True)
    return [records[int(sid)] for sid in expected_ids]


def _oracle_shared_center(
    records: Sequence[TOnlyPhaseRegistration],
    true_class_by_parcel: Mapping[int, int],
    num_classes: int,
):
    valid_records = [
        record for record in records
        if bool(record.numerically_valid) and isinstance(record.gamma, Tensor)
    ]
    if not valid_records:
        raise RuntimeError("target-train oracle registration produced no numerically-valid gamma")
    gammas = torch.stack([record.gamma.detach().cpu().double() for record in valid_records])
    labels = np.asarray([true_class_by_parcel[int(record.sample_id)] for record in valid_records], dtype=np.int64)
    if sorted(np.unique(labels).tolist()) != list(range(int(num_classes))):
        raise RuntimeError("oracle shared center requires numerically-valid gamma from every class")
    weights = class_balanced_weights(labels)
    result = weighted_frechet_mean_gamma(gammas, weights)
    per_class_valid = {str(c): int(np.sum(labels == c)) for c in range(int(num_classes))}
    return result, len(valid_records), per_class_valid


def _load_or_estimate_timematch_shift(
    *,
    manifest_path: Path,
    model,
    data_root: str,
    target: str,
    classes: Sequence[str],
    target_train_indices: set[int],
    source: str,
    seed: int,
    fold: int,
    model_checkpoint: Path,
    closed_set: bool,
    combine: bool,
    time_mode: str,
    device: torch.device,
    time_scale_days: float,
    min_shift: int,
    max_shift: int,
    max_batches: int,
    batch_size: int,
    num_pixels: int,
    num_workers: int,
) -> tuple[int, dict]:
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected in (("source", source), ("target", target), ("seed", seed), ("fold", fold)):
            if str(payload.get(key)) != str(expected):
                raise ValueError(f"TimeMatch manifest {key} mismatch")
        manifest_model = payload.get("model_checkpoint")
        if manifest_model is None or Path(str(manifest_model)).name != model_checkpoint.name:
            raise ValueError("TimeMatch manifest Stage-1 checkpoint mismatch")
        result = payload.get("timematch_shift_result")
        if not isinstance(result, dict) or "selected_shift_days" not in result:
            raise ValueError("TimeMatch manifest lacks selected_shift_days")
        shift = int(result["selected_shift_days"])
        return shift, {
            "source": "reused_experiment_04_manifest",
            "manifest": str(manifest_path.resolve()),
            "scalar_shift_days": shift,
            "estimation_split": "target-train",
            "estimation_protocol": "TimeMatch initial Inception-Score scalar shift scan",
            "used_target_labels": False,
            "result": {k: v for k, v in result.items() if k not in {"labels", "predictions", "rows"}},
        }

    loader = scalarcmp._timematch_estimation_loader(
        data_root, target, classes, target_train_indices,
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
        batch_size=batch_size,
        num_workers=num_workers,
        num_pixels=num_pixels,
        seed=seed,
    )
    result = scalarcmp._estimate_timematch_scalar_shift(
        model, loader,
        device=device,
        min_shift=min_shift,
        max_shift=max_shift,
        max_batches=max_batches,
        num_classes=len(classes),
        time_scale_days=time_scale_days,
    )
    shift = int(result["selected_shift_days"])
    return shift, {
        "source": "recomputed_with_existing_experiment_04_TimeMatch_helper",
        "manifest": None,
        "scalar_shift_days": shift,
        "estimation_split": "target-train",
        "estimation_protocol": "TimeMatch initial Inception-Score scalar shift scan",
        "used_target_labels": False,
        "max_shift_days": int(max_shift),
        "estimation_batches": int(max_batches),
        "batch_size": int(batch_size),
        "num_pixels": int(num_pixels),
        "result": {k: v for k, v in result.items() if k not in {"labels", "predictions", "rows"}},
    }


@torch.no_grad()
def _evaluate_split(
    *,
    model,
    loader,
    shared_gamma: Tensor,
    scalar_shift_days: int,
    time_scale_days: float,
    device: torch.device,
) -> dict:
    sample_ids: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    probabilities: dict[str, list[np.ndarray]] = {name: [] for name in CONDITIONS}
    model.eval()
    for raw_batch in loader:
        batch = phasevis._move_batch(raw_batch, device)
        backbone = model.forward_backbone(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch.get("extra"), time_mask=batch.get("time_mask"),
        )
        identity_output = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"), return_geometry=False,
        )
        shared_positions = align_target_positions_to_source(
            backbone.normalized_positions, backbone.time_mask, shared_gamma
        )
        shared_output = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"),
            temporal_positions_override=shared_positions, return_geometry=False,
        )
        scalar_positions = scalarcmp._scalar_positions(backbone, scalar_shift_days, time_scale_days)
        with scalarcmp._timematch_time_extrapolation(model):
            scalar_output = model.forward_from_backbone(
                backbone, batch["positions"], batch.get("extra"),
                temporal_positions_override=scalar_positions, return_geometry=False,
            )
        sample_ids.append(batch["parcel_index"].detach().cpu().numpy().astype(np.int64))
        labels.append(batch["label"].detach().cpu().numpy().astype(np.int64))
        probabilities["identity"].append(torch.softmax(identity_output.logits.float(), dim=-1).cpu().numpy())
        probabilities["timematch_scalar"].append(torch.softmax(scalar_output.logits.float(), dim=-1).cpu().numpy())
        probabilities["oracle_shared"].append(torch.softmax(shared_output.logits.float(), dim=-1).cpu().numpy())
    if not labels:
        raise RuntimeError("target split loader produced no batches")
    ids = np.concatenate(sample_ids)
    y = np.concatenate(labels)
    probs = {name: np.concatenate(parts, axis=0).astype(np.float64) for name, parts in probabilities.items()}
    order = np.argsort(ids, kind="stable")
    return {
        "sample_ids": ids[order],
        "labels": y[order],
        "probabilities": {name: value[order] for name, value in probs.items()},
    }


def _metrics_tables(split_results: Mapping[str, dict], classes: Sequence[str]):
    overall_rows: list[dict] = []
    per_class_rows: list[dict] = []
    count_rows: list[dict] = []
    transition_rows: list[dict] = []
    semantic_rows: list[dict] = []
    raw_candidates: dict[str, dict] = {}

    for split_name, result in split_results.items():
        labels = result["labels"]
        identity_class = per_class_metrics(labels, result["probabilities"]["identity"], classes)
        by_condition_class: dict[str, list[dict]] = {}
        for condition in CONDITIONS:
            probs = result["probabilities"][condition]
            overall_rows.append({"split": split_name, "condition": condition, **overall_metrics(labels, probs, classes)})
            rows = per_class_metrics(labels, probs, classes)
            rows = rows if condition == "identity" else add_identity_deltas(rows, identity_class)
            if condition == "identity":
                rows = [dict(row, delta_recall_vs_identity=0.0, delta_precision_vs_identity=0.0,
                             delta_f1_vs_identity=0.0, delta_true_prob_vs_identity=0.0) for row in rows]
            by_condition_class[condition] = rows
            for row in rows:
                enriched = {"split": split_name, "condition": condition, **row}
                per_class_rows.append(enriched)
                count_rows.append({
                    "split": split_name, "condition": condition,
                    "class_id": row["class_id"], "class_name": row["class_name"],
                    "true_support": row["true_support"], "predicted_count": row["predicted_count"],
                    "predicted_to_true_support_ratio": row["predicted_to_true_support_ratio"],
                })
            if split_name == "target-train":
                for row in classwise_semantic_distribution(labels, probs, classes):
                    semantic_rows.append({"split": split_name, "condition": condition, **row})

        for condition in ("timematch_scalar", "oracle_shared"):
            for row in hard_transition_rows(
                labels, result["probabilities"]["identity"], result["probabilities"][condition], classes
            ):
                transition_rows.append({
                    "split": split_name,
                    "comparison": f"{condition}_vs_identity",
                    **row,
                })
        if split_name == "target-train":
            raw_candidates = {
                condition: {
                    "overall": overall_metrics(labels, result["probabilities"][condition], classes),
                    "per_class": by_condition_class[condition],
                }
                for condition in CONDITIONS
            }
    return overall_rows, per_class_rows, count_rows, transition_rows, semantic_rows, raw_candidates


def _plot_per_class_metric(path: Path, rows: Sequence[dict], split: str, metric: str, ylabel: str) -> None:
    selected = [row for row in rows if row["split"] == split]
    by_condition = {condition: sorted([row for row in selected if row["condition"] == condition], key=lambda r: int(r["class_id"])) for condition in CONDITIONS}
    names = [row["class_name"] for row in by_condition["identity"]]
    x = np.arange(len(names)); width = 0.25
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for offset, condition in enumerate(CONDITIONS):
        values = [float(row[metric]) for row in by_condition[condition]]
        ax.bar(x + (offset - 1) * width, values, width=width, label=condition)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(f"{split}: per-class {ylabel}")
    ax.legend(); ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _plot_train_counts(path: Path, rows: Sequence[dict]) -> None:
    selected = [row for row in rows if row["split"] == "target-train"]
    by_condition = {condition: sorted([row for row in selected if row["condition"] == condition], key=lambda r: int(r["class_id"])) for condition in CONDITIONS}
    names = [row["class_name"] for row in by_condition["identity"]]
    true_support = np.asarray([float(row["true_support"]) for row in by_condition["identity"]])
    x = np.arange(len(names)); width = 0.25
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for offset, condition in enumerate(CONDITIONS):
        values = [float(row["predicted_count"]) for row in by_condition[condition]]
        ax.bar(x + (offset - 1) * width, values, width=width, label=condition)
    ax.plot(x, true_support, marker="o", linewidth=1.5, label="true support (oracle reference)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("count"); ax.set_title("target-train: predicted class counts vs true support")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _plot_confusion(path: Path, matrix: np.ndarray, classes: Sequence[str], title: str, *, normalized: bool) -> None:
    values = row_normalize_confusion(matrix) if normalized else np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    image = ax.imshow(values, aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    ax.set_title(title + (" — row normalized" if normalized else " — raw count"))
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _plot_oracle_phase(path: Path, gamma: Tensor, time_scale_days: float, scalar_shift_days: int) -> dict:
    gamma = gamma.detach().cpu().double()
    grid = torch.linspace(0.0, 1.0, gamma.numel(), dtype=torch.float64)
    inverse = invert_monotone_warp(gamma.unsqueeze(0), grid.unsqueeze(0)).squeeze(0)
    source_to_target = (gamma - grid).numpy() * float(time_scale_days)
    target_to_source = (inverse - grid).numpy() * float(time_scale_days)
    x = grid.numpy() * float(time_scale_days)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, source_to_target, label="oracle shared gamma(t)-t (source→target)")
    ax.plot(x, target_to_source, label="oracle shared gamma^-1(t)-t (target→source)")
    ax.axhline(float(scalar_shift_days), linestyle="--", label=f"TimeMatch target correction {scalar_shift_days:+d} d")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("canonical day"); ax.set_ylabel("displacement (days)")
    ax.set_title("Oracle target-train shared Phase vs TimeMatch scalar bootstrap")
    ax.legend(fontsize=8); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return {
        "distance_to_identity": phase_distance_to_identity_for_gamma(gamma),
        "mean_displacement_days": float(np.mean(source_to_target)),
        "max_abs_displacement_days": float(np.max(np.abs(source_to_target))),
        "mean_target_to_source_correction_days": float(np.mean(target_to_source)),
        "max_abs_target_to_source_correction_days": float(np.max(np.abs(target_to_source))),
    }


def _write_readme(path: Path) -> None:
    text = r"""# 实验 12：Stage 2 冷启动时间状态诊断

## 协议边界

本实验冻结 Stage-1 的 PSE、Time2Vec、LTAE 和 classifier，不进行 Stage-2 训练，不产生或刷新伪标签，不使用 Teacher/Student、Stable Label、class-conditioned Phase 或 Domain Shape。唯一改变是 target 进入 LTAE 的时间位置。

三种条件：

1. `identity`：原始 target 时间位置；
2. `timematch_scalar`：复用/按现有 TimeMatch Inception-Score 无标签流程在 target-train 估计的 scalar shift；
3. `oracle_shared`：只用 target-train true label 指定正确 source class registration，再对 target-train numerically-valid gamma 求类别平衡 Fisher–Rao Fréchet center。它只是一条开发诊断上界，不是正式 UDA 方法。

Target-val/test true label 从不参与 Oracle Shared Phase estimation，只用于事后分类诊断。Identity/TimeMatch 的构建不使用任何 target label。

## 文件说明

- `00_manifest.json`：完整协议、checkpoint、split、TimeMatch estimator、oracle target-train gamma cache、标签使用边界。能证明运行条件；不能证明哪种 bootstrap 应被自动选用。
- `01_bootstrap_overall_metrics.csv`：每个 split×condition 的 Accuracy、Macro-F1、macro precision/recall、true/max probability、margin、entropy、预测类别熵和最大预测类比例。重点看 train/val/test 是否方向一致；不能只凭 Accuracy 选方案。
- `02_bootstrap_per_class_metrics.csv`：三个 split、三条件、全部 10 类的 support、predicted count、precision/recall/F1、confidence/margin/entropy，以及相对 Identity 的变化。是判断类别吞噬/消失和类别牺牲的主表。
- `03_bootstrap_class_count_diagnostic.csv`：逐类 predicted count、true support（oracle reference）与二者比值。比值很大表示该预测类吸入其他类别，比值很小表示该真实类可能在预测空间消失；这个比值不能用于正式 UDA 决策。
- `04_bootstrap_hard_transition_vs_identity.csv`：TimeMatch/Oracle Shared 相对 Identity 的 wrong→correct、correct→wrong、correct→correct、wrong→wrong 和净正确增益，按 split/真实类保存。
- `05_train_per_class_recall.png`：target-train 三条件逐类 Recall。
- `06_train_per_class_precision.png`：target-train 三条件逐类 Precision。
- `07_train_per_class_f1.png`：target-train 三条件逐类 F1。
- `08_train_predicted_class_counts.png`：target-train 三条件 predicted count；折线为真实 support，仅作 oracle diagnosis。
- `09_train_true_class_probability.png`：target-train 三条件逐类 mean true-class probability。
- `10_val_per_class_recall.png` / `11_val_per_class_precision.png` / `12_val_per_class_f1.png`：target-val 三条件逐类指标，只用于事后诊断。
- `13_test_per_class_recall.png` / `14_test_per_class_precision.png` / `15_test_per_class_f1.png`：target-test 三条件逐类指标，只用于事后诊断，不参与参数或 bootstrap 选择。
- `16_confusion_identity_train.png`：target-train Identity raw-count confusion；横轴 predicted，纵轴 true。
- `17_confusion_timematch_train.png`：target-train TimeMatch raw-count confusion。
- `18_confusion_oracle_shared_train.png`：target-train Oracle Shared raw-count confusion。
- `19_confusion_identity_val.png`：target-val Identity raw-count confusion。
- `20_confusion_timematch_val.png`：target-val TimeMatch raw-count confusion。
- `21_confusion_oracle_shared_val.png`：target-val Oracle Shared raw-count confusion。
- `22_confusion_identity_test.png`：target-test Identity raw-count confusion，仅事后诊断。
- `23_confusion_timematch_test.png`：target-test TimeMatch raw-count confusion，仅事后诊断。
- `24_confusion_oracle_shared_test.png`：target-test Oracle Shared raw-count confusion，仅事后诊断。
- `16`–`24` 每个文件另有同 stem 的 `_row_normalized.png`：按真实类别行归一化，观察每个真实类别流向哪些预测类别；它不会改变 raw count 文件。
- `25_classwise_semantic_distribution_summary.csv`：target-train 按真实类别、condition 对 max probability、prediction margin、prediction entropy、true-class probability 的 mean/median/q10/q25/q75/q90。用于判断是否存在强烈类别依赖；本实验不据此设计 threshold。
- `26_oracle_shared_phase_curve.npz`：target-train oracle class-balanced shared gamma、inverse、canonical grid、每类 total/valid registration count。用于审计 Phase 本体；不包含 val/test label 信息。
- `27_oracle_shared_phase_displacement.png`：Oracle Shared source→target 位移、其 target→source 校正位移，以及 TimeMatch scalar target correction 的时间结构对照。方向不同的曲线已显式标注。
- `28_bootstrap_diagnostic_summary.json`：机器可读总摘要和 target-train raw top-1 semantic candidate 诊断。`automatic_bootstrap_choice=null`，脚本不会根据 target 指标自动选择正式方案。
- `cache/target_train_oracle_true_class_registrations.pt`：可断点续跑的 target-train oracle registration cache。只以 true label 指定正确 source prototype；production legality 不用于 Oracle Shared center。首次完整运行可能较昂贵，后续重跑复用该 cache。
- `README_中文说明.md`：本说明。

## 能支持什么

本实验可以判断 Identity 是否已经提供健康的 Stage-2 semantic starting point；TimeMatch scalar 是否带来跨 split、跨类别的低风险改善；以及如果有一个较好的 nonlinear shared Phase，冷启动语义上限相对正式候选还有多大差距。

## 不能支持什么

Oracle Shared 使用 target-train true label，因此不是正式 UDA bootstrap。实验不能据结果直接设计 pseudo-label threshold、Stable Label、Teacher/Student、class-specific Phase、adaptive strength、训练损失或 Phase–semantic iteration。最终判断必须同时考虑 overall、全部 10 类、预测数量、hard transitions 与 confidence/margin/entropy，而不能只选 Accuracy 最高者。
"""
    path.write_text(text, encoding="utf-8")


def run(args) -> dict:
    calibration = torch.load(args.calibration_checkpoint.resolve(), map_location="cpu", weights_only=False)
    runtime = calibration.get("runtime_config")
    if not isinstance(runtime, dict):
        raise ValueError("calibration checkpoint is missing runtime_config")
    classes = [str(v) for v in runtime["classes"]]
    source = str(runtime["source"]); target = str(runtime["target"]); seed = int(runtime["seed"])
    fold = int(args.fold); data_root = str(args.data_root or runtime["data_root"])
    closed_set = bool(runtime.get("closed_set", True)); combine = bool(runtime.get("combine_spring_and_winter", False))
    time_mode = str(runtime.get("time_coordinate_mode", "canonical_day_of_year"))
    val_ratio = float(runtime.get("val_ratio", 0.1)); test_ratio = float(runtime.get("test_ratio", 0.2))
    time_scale_days = float(runtime.get("time_scale", 365.0))
    if len(classes) != 10:
        raise ValueError(f"experiment 12 expects closed-set 10 classes, got {len(classes)}")

    device = torch.device(args.device)
    model_checkpoint_path = args.model_checkpoint.resolve()
    model_checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    model = phasevis._build_model(runtime, calibration, device, model_checkpoint=model_checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    source_all = phasevis._eligible_parcels(data_root, source, classes, closed_set=closed_set, combine_spring_and_winter=combine, time_coordinate_mode=time_mode)
    target_all = phasevis._eligible_parcels(data_root, target, classes, closed_set=closed_set, combine_spring_and_winter=combine, time_coordinate_mode=time_mode)
    splits = phasevis._reconstruct_fold_splits(
        source_all, target_all, source=source, target=target, seed=seed,
        val_ratio=val_ratio, test_ratio=test_ratio, fold=fold,
    )
    target_parcels = {
        name: np.asarray(sorted(splits[target][name]), dtype=np.int64)
        for name in ("train", "val", "test")
    }
    source_train_parcels = np.asarray(sorted(splits[source]["train"]), dtype=np.int64)

    target_meta = {
        name: phasevis._metadata_dataset(
            data_root, target, classes, splits[target][name], closed_set=closed_set,
            combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
        ) for name in ("train", "val", "test")
    }
    true_class_by_train_parcel = {
        int(parcel): int(label)
        for parcel, label in zip(
            target_meta["train"].get_parcel_indices().tolist(),
            target_meta["train"].get_labels().tolist(),
        )
    }
    train_expected_ids = [int(v) for v in target_parcels["train"].tolist()]
    if set(train_expected_ids) != set(true_class_by_train_parcel):
        raise ValueError("target-train metadata does not match reconstructed split")

    loaders = {
        name: phasevis._selected_loader(
            data_root, target, classes, target_parcels[name],
            closed_set=closed_set, combine_spring_and_winter=combine,
            time_coordinate_mode=time_mode, batch_size=args.batch_size,
            num_workers=args.num_workers,
        ) for name in ("train", "val", "test")
    }
    source_train_loader = phasevis._selected_loader(
        data_root, source, classes, source_train_parcels,
        closed_set=closed_set, combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)

    timematch_shift, timematch_meta = _load_or_estimate_timematch_shift(
        manifest_path=args.timematch_manifest.resolve(),
        model=model, data_root=data_root, target=target, classes=classes,
        target_train_indices=splits[target]["train"], source=source, seed=seed, fold=fold,
        model_checkpoint=model_checkpoint_path, closed_set=closed_set, combine=combine,
        time_mode=time_mode, device=device, time_scale_days=time_scale_days,
        min_shift=-int(args.timematch_max_shift), max_shift=int(args.timematch_max_shift),
        max_batches=args.timematch_estimation_batches,
        batch_size=args.timematch_batch_size, num_pixels=args.timematch_num_pixels,
        num_workers=args.num_workers,
    )
    print(
        f"BOOTSTRAP12_TIMEMATCH_READY|shift_days={timematch_shift}|split=target-train|used_target_labels=false"
        f"|source={timematch_meta['source']}", flush=True,
    )

    scan_config = samplediag._scan_config(runtime, args.registration_workers)
    reg_extractor = build_stage2_registration_extractor(model, device=device, k_reg=scan_config.k_reg)
    source_reg_bank = samplediag._load_or_build_registration_bank(
        args.source_registration_bank_cache.resolve(),
        model=model,
        source_train_loader=DeviceBatchLoader(source_train_loader, device),
        num_classes=len(classes), device=device, reg_extractor=reg_extractor,
    )

    oracle_cache_path = cache_dir / "target_train_oracle_true_class_registrations.pt"
    print(
        f"BOOTSTRAP12_ORACLE_TRAIN_START|target_train={len(train_expected_ids)}|"
        "true_label_use=oracle_correct_class_only|production_legality_filter=false",
        flush=True,
    )
    oracle_records = _build_target_train_oracle_gamma_cache(
        model=model, target_loader=loaders["train"], source_reg_bank=source_reg_bank,
        reg_extractor=reg_extractor, scan_config=scan_config, cache_path=oracle_cache_path,
        expected_ids=train_expected_ids, true_class_by_parcel=true_class_by_train_parcel,
        source=source, target=target, seed=seed, fold=fold,
        model_checkpoint=model_checkpoint_path, device=device, workers=args.registration_workers,
        geometry_chunk_size=args.oracle_geometry_chunk_size,
        dp_target_chunk_size=args.dp_target_chunk_size,
    )
    center_result, num_valid, per_class_valid = _oracle_shared_center(
        oracle_records, true_class_by_train_parcel, len(classes)
    )
    shared_gamma = center_result.gamma.detach().cpu().double()
    total_per_class = {
        str(c): int(sum(int(true_class_by_train_parcel[sid]) == c for sid in train_expected_ids))
        for c in range(len(classes))
    }
    print(
        f"BOOTSTRAP12_ORACLE_SHARED_READY|train_total={len(train_expected_ids)}|numerically_valid={num_valid}"
        f"|frechet_converged={str(center_result.converged).lower()}|objective={center_result.objective:.8g}",
        flush=True,
    )

    split_results = {}
    for name in ("train", "val", "test"):
        formal_name = f"target-{name}"
        print(f"BOOTSTRAP12_EVAL_START|split={formal_name}|conditions=identity,timematch_scalar,oracle_shared", flush=True)
        split_results[formal_name] = _evaluate_split(
            model=model, loader=loaders[name], shared_gamma=shared_gamma,
            scalar_shift_days=timematch_shift, time_scale_days=time_scale_days, device=device,
        )
        print(f"BOOTSTRAP12_EVAL_READY|split={formal_name}|n={len(split_results[formal_name]['labels'])}", flush=True)

    overall_rows, per_class_rows, count_rows, transition_rows, semantic_rows, raw_candidates = _metrics_tables(split_results, classes)
    _write_csv(output / "01_bootstrap_overall_metrics.csv", overall_rows)
    _write_csv(output / "02_bootstrap_per_class_metrics.csv", per_class_rows)
    _write_csv(output / "03_bootstrap_class_count_diagnostic.csv", count_rows)
    _write_csv(output / "04_bootstrap_hard_transition_vs_identity.csv", transition_rows)

    _plot_per_class_metric(output / "05_train_per_class_recall.png", per_class_rows, "target-train", "recall", "Recall")
    _plot_per_class_metric(output / "06_train_per_class_precision.png", per_class_rows, "target-train", "precision", "Precision")
    _plot_per_class_metric(output / "07_train_per_class_f1.png", per_class_rows, "target-train", "f1", "F1")
    _plot_train_counts(output / "08_train_predicted_class_counts.png", per_class_rows)
    _plot_per_class_metric(output / "09_train_true_class_probability.png", per_class_rows, "target-train", "mean_true_class_probability", "Mean true-class probability")
    _plot_per_class_metric(output / "10_val_per_class_recall.png", per_class_rows, "target-val", "recall", "Recall")
    _plot_per_class_metric(output / "11_val_per_class_precision.png", per_class_rows, "target-val", "precision", "Precision")
    _plot_per_class_metric(output / "12_val_per_class_f1.png", per_class_rows, "target-val", "f1", "F1")
    _plot_per_class_metric(output / "13_test_per_class_recall.png", per_class_rows, "target-test", "recall", "Recall")
    _plot_per_class_metric(output / "14_test_per_class_precision.png", per_class_rows, "target-test", "precision", "Precision")
    _plot_per_class_metric(output / "15_test_per_class_f1.png", per_class_rows, "target-test", "f1", "F1")

    number = 16
    for split_name, short in (("target-train", "train"), ("target-val", "val"), ("target-test", "test")):
        result = split_results[split_name]
        for condition, stem in (("identity", "identity"), ("timematch_scalar", "timematch"), ("oracle_shared", "oracle_shared")):
            pred = result["probabilities"][condition].argmax(axis=1)
            matrix = confusion_matrix(result["labels"], pred, len(classes))
            base = output / f"{number:02d}_confusion_{stem}_{short}.png"
            _plot_confusion(base, matrix, classes, f"{split_name}: {condition}", normalized=False)
            _plot_confusion(base.with_name(base.stem + "_row_normalized.png"), matrix, classes, f"{split_name}: {condition}", normalized=True)
            number += 1

    _write_csv(output / "25_classwise_semantic_distribution_summary.csv", semantic_rows)
    grid = torch.linspace(0.0, 1.0, shared_gamma.numel(), dtype=torch.float64)
    inverse = invert_monotone_warp(shared_gamma.unsqueeze(0), grid.unsqueeze(0)).squeeze(0)
    np.savez_compressed(
        output / "26_oracle_shared_phase_curve.npz",
        grid=grid.numpy(), gamma=shared_gamma.numpy(), inverse_gamma=inverse.numpy(),
        class_names=np.asarray(classes),
        target_train_total_per_class=np.asarray([total_per_class[str(c)] for c in range(len(classes))]),
        target_train_numerically_valid_per_class=np.asarray([per_class_valid[str(c)] for c in range(len(classes))]),
    )
    phase_summary = _plot_oracle_phase(
        output / "27_oracle_shared_phase_displacement.png", shared_gamma,
        time_scale_days, timematch_shift,
    )

    overall_by_split = {
        split: {condition: next(row for row in overall_rows if row["split"] == split and row["condition"] == condition) for condition in CONDITIONS}
        for split in ("target-train", "target-val", "target-test")
    }
    summary = {
        "protocol": PROTOCOL,
        "source": source, "target": target, "seed": seed, "fold": fold,
        "network_parameters_frozen": True,
        "stage2_training": False,
        "pseudo_label_updates": False,
        "teacher_student": False,
        "stable_label": False,
        "class_conditioned_phase": False,
        "domain_shape_transport": False,
        "conditions": {
            "identity": "native target temporal positions",
            "timematch_scalar": timematch_meta,
            "oracle_shared": {
                "estimation_split": "target-train only",
                "used_target_train_true_labels": True,
                "used_target_val_true_labels": False,
                "used_target_test_true_labels": False,
                "true_label_role": "oracle-only correct source class for sample registration and class-balanced weights",
                "registration_total": len(oracle_records),
                "registration_numerically_valid": num_valid,
                "registration_invalid": len(oracle_records) - num_valid,
                "valid_per_class": per_class_valid,
                "total_per_class": total_per_class,
                "production_legality_filter": False,
                "center_estimator": "class-balanced intrinsic Fisher-Rao Frechet/Karcher mean",
                "frechet_objective": center_result.objective,
                "frechet_iterations": center_result.iterations,
                "frechet_converged": center_result.converged,
                "frechet_final_tangent_norm": center_result.tangent_norm,
                **phase_summary,
            },
        },
        "split_sizes": {split: int(len(result["labels"])) for split, result in split_results.items()},
        "overall": overall_by_split,
        "raw_top1_semantic_candidate_target_train": raw_candidates,
        "focus_classes": ["corn", "spring_barley", "spring_oat", "winter_triticale", "winter_wheat", "winter_rye"],
        "automatic_bootstrap_choice": None,
        "allowed_interpretation_cases": ["A_identity_healthy", "B_timematch_low_risk", "C_oracle_shared_cold_start_dependency", "D_oracle_shared_class_harm", "E_all_semantically_biased"],
    }
    _json_dump(output / "28_bootstrap_diagnostic_summary.json", summary)
    manifest = {
        "protocol": PROTOCOL,
        "calibration_checkpoint": str(args.calibration_checkpoint.resolve()),
        "model_checkpoint": str(model_checkpoint_path),
        "source_registration_bank_cache": str(args.source_registration_bank_cache.resolve()),
        "target_train_oracle_gamma_cache": str(oracle_cache_path),
        "timematch_manifest": str(args.timematch_manifest.resolve()),
        "source": source, "target": target, "seed": seed, "fold": fold,
        "classes": classes,
        "split_sizes": summary["split_sizes"],
        "only_changed_model_input": "target temporal positions passed to Time2Vec/LTAE",
        "H_i_unchanged_across_conditions": True,
        "network_parameters_frozen": True,
        "stage2_training": False,
        "pseudo_label_refresh": False,
        "stable_label": False,
        "teacher_ema": False,
        "target_loss": False,
        "phase_semantic_iteration": False,
        "class_conditioned_phase": False,
        "domain_shape_transport": False,
        "target_test_used_for_parameter_selection": False,
        "identity_label_use": False,
        "timematch_label_use": False,
        "timematch_estimation_split": "target-train",
        "oracle_shared_estimation_split": "target-train",
        "oracle_shared_target_train_label_use": "oracle-only",
        "oracle_shared_target_val_test_label_use": "evaluation-only; never estimation",
        "oracle_shared_registration_filter": "numerical validity only; production legality ignored",
        "gamma_direction": "source_to_target; target LTAE positions use gamma_inverse",
        "automatic_bootstrap_choice": None,
    }
    _json_dump(output / "00_manifest.json", manifest)
    _write_readme(output / "README_中文说明.md")
    print(f"BOOTSTRAP12_DONE|output={output}|stage2_training=false|pseudo_label_updates=false", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-checkpoint", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--source-registration-bank-cache", type=Path, required=True)
    parser.add_argument("--timematch-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--registration-workers", type=int, default=4)
    parser.add_argument("--oracle-geometry-chunk-size", type=int, default=512)
    parser.add_argument("--dp-target-chunk-size", type=int, default=512)
    parser.add_argument("--timematch-max-shift", type=int, default=60)
    parser.add_argument("--timematch-estimation-batches", type=int, default=100)
    parser.add_argument("--timematch-batch-size", type=int, default=128)
    parser.add_argument("--timematch-num-pixels", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.registration_workers < 1 or args.oracle_geometry_chunk_size < 1 or args.dp_target_chunk_size < 1:
        parser.error("registration workers and chunk sizes must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
