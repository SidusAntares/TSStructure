#!/usr/bin/env python3
"""Experiment 13A-2: missing semantic structure and evidence complementarity.

This diagnostic keeps the Stage-1 model, raw candidate assignments, Phase and
functional geometry frozen.  It adds source-distribution conformity, exact
source/target cosine KNN structure, candidate-pool health and source-calibrated
T/S geometry.  Target true labels are joined only after all unlabeled
observables, neighbourhoods, pool summaries and PCA coordinates are saved.
No pseudo-label gate, learned reliability model, threshold search or training
update is implemented.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

import diagnose_stage2_trainable_seed_evidence_13a as a1
import diagnose_sample_level_phase_validity as samplediag
import visualize_stage2_phase_alignment as phasevis
from methods.structure_da.candidate_evidence_diagnostic import (
    confidence_matched_indices,
    confidence_quantile_masks,
    distribution_summary,
    precision_at_coverages,
    standardized_mean_difference,
    summarize_evidence,
)
from methods.structure_da.candidate_structure_diagnostic import (
    DEFAULT_K_VALUES,
    class_centroids,
    complementarity_diagnostics,
    cosine_distance_rows,
    empirical_percentiles_by_class,
    exact_cosine_knn,
    loo_class_centroid_distances,
    pca_project,
    pool_centroid_distances,
    source_knn_observables,
    target_knn_observables,
)
from methods.structure_da.prototype_bank import QUANTILE_LEVELS, SourcePrototypeBank
from methods.structure_da.registration_geometry import (
    SourceRegistrationPrototypeBank,
    evaluate_registration_geometry,
)
from methods.structure_da.stage2_trainer import DeviceBatchLoader, build_stage2_registration_extractor


PROTOCOL = "13A2_missing_semantic_structure_and_evidence_complementarity_diagnostic"
TARGET_FEATURE_SCHEMA = "13A2_target_frozen_ltae_features_v1"
SOURCE_FEATURE_SCHEMA = "13A2_source_frozen_ltae_features_v1"
CROSSFIT_BANK_SCHEMA = "13A2_source_geometry_crossfit_banks_v1"
CROSSFIT_GEOMETRY_SCHEMA = "13A2_source_geometry_crossfit_reference_v1"
K_VALUES = DEFAULT_K_VALUES
PRIMARY_K = 20  # fixed mid-scale diagnostic view; never selected by target oracle labels.
GEOMETRY_RAW_SPECS = (
    ("T_identity_error", False),
    ("T_registered_error", False),
    ("T_gain", True),
    ("T_gain_ratio", False),
    ("S_identity_error", False),
    ("S_registered_error", False),
    ("S_gain", True),
    ("S_gain_ratio", False),
    ("phase_magnitude", False),
)
FORBIDDEN_TARGET_ORACLE_FIELDS = {
    "true_label", "true_class_name", "candidate_correct", "confusion_flow",
    "oat_correct", "barley_absorbed_into_oat", "triticale_correct",
    "rye_absorbed_into_triticale", "wheat_absorbed_into_triticale",
}


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    a1._write_csv(path, rows)


def _json_dump(path: Path, payload) -> None:
    a1._json_dump(path, payload)


def _atomic_torch_save(payload, path: Path) -> None:
    a1._atomic_torch_save(payload, path)


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int(value) -> int:
    return int(float(value))


def _fingerprint_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(value)
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _validate_13a1_input(experiment_dir: Path) -> tuple[list[dict], dict]:
    manifest_path = experiment_dir / "00_manifest.json"
    rows_path = experiment_dir / "01_unlabeled_sample_observables.csv"
    if not manifest_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError("13A-2 requires 13A-1 manifest and unlabeled sample observables")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = manifest.get("bootstrap_state")
    if isinstance(state, dict):
        state = state.get("state")
    if str(state) != "identity":
        raise ValueError("13A-2 is frozen to delta_boot=identity and requires matching 13A-1 artifacts")
    if bool(manifest.get("raw_observable_file_contains_true_label", False)):
        raise ValueError("13A-1 observable artifact is not label-free")
    if str(manifest.get("candidate_class_source")) != "raw classifier top-1 only":
        raise ValueError("13A-1 candidate definition is incompatible with 13A-2")
    rows = _load_csv(rows_path)
    if not rows:
        raise ValueError("13A-1 unlabeled observable file is empty")
    if any(FORBIDDEN_TARGET_ORACLE_FIELDS.intersection(row) for row in rows):
        raise RuntimeError("oracle field leaked into 13A-1 label-free observable input")
    return rows, manifest


@torch.no_grad()
def _collect_frozen_features(model, loader, *, device: torch.device, source_labels_allowed: bool) -> dict:
    sample_ids: list[np.ndarray] = []
    features: list[np.ndarray] = []
    posteriors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    for raw_batch in loader:
        if not source_labels_allowed and "label" in raw_batch:
            raise RuntimeError("target true label leaked into 13A-2 frozen feature extraction")
        batch = phasevis._move_batch(raw_batch, device)
        backbone = model.forward_backbone(
            batch["pixels"], batch["valid_pixels"], batch["positions"], batch.get("extra"),
            time_mask=batch.get("time_mask"), compute_decomposition=False,
        )
        output = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"), return_geometry=False,
        )
        sample_ids.append(batch["parcel_index"].detach().cpu().numpy().astype(np.int64))
        features.append(output.fused_repr.detach().cpu().float().numpy())
        posteriors.append(torch.softmax(output.logits.float(), dim=-1).detach().cpu().numpy())
        if source_labels_allowed:
            labels.append(batch["label"].detach().cpu().numpy().astype(np.int64))
    if not sample_ids:
        raise RuntimeError("frozen feature loader produced no samples")
    ids = np.concatenate(sample_ids)
    feat = np.concatenate(features).astype(np.float32)
    post = np.concatenate(posteriors).astype(np.float32)
    order = np.argsort(ids, kind="stable")
    result = {
        "sample_ids": ids[order],
        "features": feat[order],
        "raw_posterior": post[order],
        "raw_pred": post[order].argmax(axis=1).astype(np.int64),
    }
    if source_labels_allowed:
        result["labels"] = np.concatenate(labels)[order].astype(np.int64)
    if np.unique(result["sample_ids"]).size != result["sample_ids"].size:
        raise ValueError("parcel identities are not unique in frozen feature cache")
    return result


def _load_or_generate_feature_cache(
    path: Path, *, schema: str, model, loader, device: torch.device,
    source_labels_allowed: bool, source: str, target: str, seed: int, fold: int,
    model_checkpoint: Path,
) -> dict:
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != schema:
            raise ValueError(f"unsupported feature cache schema at {path}")
        for key, expected in (("source", source), ("target", target), ("seed", seed), ("fold", fold)):
            if str(payload.get(key)) != str(expected):
                raise ValueError(f"13A-2 feature cache {key} mismatch")
        if Path(str(payload.get("model_checkpoint"))).resolve() != model_checkpoint.resolve():
            raise ValueError("13A-2 feature cache Stage-1 checkpoint mismatch")
        if bool(payload.get("source_labels_allowed")) != bool(source_labels_allowed):
            raise ValueError("13A-2 feature cache label-boundary mismatch")
        print(f"SEED13A2_FEATURE_CACHE_HIT|path={path}", flush=True)
        return payload["observables"]
    print(f"SEED13A2_FEATURE_CACHE_BUILD|path={path}|source_labels_allowed={str(source_labels_allowed).lower()}", flush=True)
    obs = _collect_frozen_features(model, loader, device=device, source_labels_allowed=source_labels_allowed)
    _atomic_torch_save({
        "schema": schema,
        "source": source,
        "target": target,
        "seed": int(seed),
        "fold": int(fold),
        "model_checkpoint": str(model_checkpoint.resolve()),
        "source_labels_allowed": bool(source_labels_allowed),
        "contains_target_true_labels": False,
        "observables": obs,
    }, path)
    return obs


def _stratified_source_fold_assignment(sample_ids: np.ndarray, labels: np.ndarray, folds: int) -> np.ndarray:
    if int(folds) < 2:
        raise ValueError("source cross-fit requires at least two folds")
    assignment = np.full(sample_ids.shape, -1, dtype=np.int64)
    for cid in np.unique(labels):
        idx = np.flatnonzero(labels == cid)
        ordered = idx[np.argsort(sample_ids[idx], kind="stable")]
        assignment[ordered] = np.arange(ordered.size, dtype=np.int64) % int(folds)
    if np.any(assignment < 0):
        raise RuntimeError("source cross-fit assignment incomplete")
    return assignment


def _cpu_source_bank(bank: SourcePrototypeBank) -> SourcePrototypeBank:
    return SourcePrototypeBank(
        trend_srvf=bank.trend_srvf.detach().cpu(),
        shape_srvf=bank.shape_srvf.detach().cpu(),
        trend_support=bank.trend_support.detach().cpu(),
        shape_support=bank.shape_support.detach().cpu(),
        fused=bank.fused.detach().cpu(),
        class_counts=bank.class_counts.detach().cpu(),
        ready=bank.ready.detach().cpu(),
        q_distance_samples=tuple(item.detach().cpu() for item in bank.q_distance_samples),
        f_distance_samples=tuple(item.detach().cpu() for item in bank.f_distance_samples),
        q_quantiles=bank.q_quantiles.detach().cpu(),
        f_quantiles=bank.f_quantiles.detach().cpu(),
        version=int(bank.version),
    )


def _cpu_registration_bank(bank: SourceRegistrationPrototypeBank) -> SourceRegistrationPrototypeBank:
    return SourceRegistrationPrototypeBank(
        trend_srvf=bank.trend_srvf.detach().cpu(),
        trend_support=bank.trend_support.detach().cpu(),
        class_counts=bank.class_counts.detach().cpu(),
        ready=bank.ready.detach().cpu(),
        registration_grid=bank.registration_grid.detach().cpu(),
    )


def _build_crossfit_reference_banks(
    cache_path: Path, *, model, source_loader, source_ids: np.ndarray, source_labels: np.ndarray,
    fold_assignment: np.ndarray, num_classes: int, folds: int, device: torch.device,
    reg_extractor, model_checkpoint: Path, geometry_signature: str,
) -> tuple[list[SourcePrototypeBank], list[SourceRegistrationPrototypeBank]]:
    fingerprint = _fingerprint_arrays(source_ids.astype(np.int64), source_labels.astype(np.int64), fold_assignment.astype(np.int64))
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("schema") == CROSSFIT_BANK_SCHEMA and payload.get("fingerprint") == fingerprint:
            if str(payload.get("geometry_signature")) != str(geometry_signature):
                raise ValueError("13A-2 source cross-fit bank geometry-config mismatch")
            if Path(str(payload.get("model_checkpoint"))).resolve() != model_checkpoint.resolve():
                raise ValueError("13A-2 source cross-fit bank checkpoint mismatch")
            print(f"SEED13A2_SOURCE_CROSSFIT_BANK_CACHE_HIT|path={cache_path}", flush=True)
            return list(payload["shape_banks"]), list(payload["registration_banks"])
    print("SEED13A2_SOURCE_CROSSFIT_BANK_BUILD|status=start", flush=True)
    id_to_fold = {int(sid): int(f) for sid, f in zip(source_ids.tolist(), fold_assignment.tolist())}
    shape_sum = shape_sup_sum = trend_sum = trend_sup_sum = None
    reg_sum = reg_sup_sum = None
    shape_count = reg_count = None
    shape_grid = reg_grid = None
    model.eval()
    with torch.inference_mode():
        for raw_batch in source_loader:
            batch = phasevis._move_batch(raw_batch, device)
            output = model(
                batch["pixels"], batch["valid_pixels"], batch["positions"], batch.get("extra"),
                return_geometry=True,
            )
            if output.geometry is None:
                raise RuntimeError("source cross-fit geometry requires functional geometry")
            reg = evaluate_registration_geometry(output.trend, output.positions, output.mask, reg_extractor)
            labels = batch["label"].long()
            parcel = batch["parcel_index"].detach().cpu().long().tolist()
            fold_t = torch.tensor([id_to_fold[int(sid)] for sid in parcel], device=device, dtype=torch.long)
            shape_q = output.geometry.structure_srvf
            shape_sup = output.geometry.structure_support
            trend_q = output.geometry.trend_srvf
            trend_sup = output.geometry.trend_support
            shape_valid = output.geometry.structure_valid
            reg_q = reg.trend_srvf
            reg_sup = reg.trend_support
            reg_valid = reg.trend_valid
            if shape_sum is None:
                shape_grid = output.geometry.canonical_grid.detach().cpu()
                reg_grid = reg.registration_grid.detach().cpu()
                fdim = int(shape_q.shape[-1]); ks = int(shape_q.shape[1]); kr = int(reg_q.shape[1])
                shape_sum = torch.zeros(folds, num_classes, ks, fdim, device=device, dtype=shape_q.dtype)
                shape_sup_sum = torch.zeros(folds, num_classes, ks, device=device, dtype=shape_sup.dtype)
                trend_sum = torch.zeros_like(shape_sum)
                trend_sup_sum = torch.zeros_like(shape_sup_sum)
                reg_sum = torch.zeros(folds, num_classes, kr, fdim, device=device, dtype=reg_q.dtype)
                reg_sup_sum = torch.zeros(folds, num_classes, kr, device=device, dtype=reg_sup.dtype)
                shape_count = torch.zeros(folds, num_classes, device=device, dtype=torch.long)
                reg_count = torch.zeros(folds, num_classes, device=device, dtype=torch.long)
            mean_support = shape_sup.mean(dim=1)
            for f in range(int(folds)):
                for cid in range(int(num_classes)):
                    smask = (fold_t == f) & (labels == cid) & shape_valid & (mean_support > 0)
                    if torch.any(smask).item():
                        sq = shape_q[smask]; ss = shape_sup[smask]
                        tq = trend_q[smask]; ts = trend_sup[smask]
                        shape_sum[f, cid] += (sq * ss.unsqueeze(-1)).sum(dim=0)
                        shape_sup_sum[f, cid] += ss.sum(dim=0)
                        trend_sum[f, cid] += (tq * ts.unsqueeze(-1)).sum(dim=0)
                        trend_sup_sum[f, cid] += ts.sum(dim=0)
                        shape_count[f, cid] += int(smask.sum().item())
                    rmask = (fold_t == f) & (labels == cid) & reg_valid
                    if torch.any(rmask).item():
                        rq = reg_q[rmask]; rs = reg_sup[rmask]
                        reg_sum[f, cid] += (rq * rs.unsqueeze(-1)).sum(dim=0)
                        reg_sup_sum[f, cid] += rs.sum(dim=0)
                        reg_count[f, cid] += int(rmask.sum().item())
    if shape_sum is None or reg_sum is None or shape_grid is None or reg_grid is None:
        raise RuntimeError("source loader produced no geometry for cross-fit reference")
    total_shape_sum = shape_sum.sum(dim=0); total_shape_sup = shape_sup_sum.sum(dim=0)
    total_trend_sum = trend_sum.sum(dim=0); total_trend_sup = trend_sup_sum.sum(dim=0)
    total_shape_count = shape_count.sum(dim=0)
    total_reg_sum = reg_sum.sum(dim=0); total_reg_sup = reg_sup_sum.sum(dim=0); total_reg_count = reg_count.sum(dim=0)
    shape_banks: list[SourcePrototypeBank] = []
    reg_banks: list[SourceRegistrationPrototypeBank] = []
    eps = 1e-8
    for query_fold in range(int(folds)):
        c_shape_sum = total_shape_sum - shape_sum[query_fold]
        c_shape_sup = total_shape_sup - shape_sup_sum[query_fold]
        c_trend_sum = total_trend_sum - trend_sum[query_fold]
        c_trend_sup = total_trend_sup - trend_sup_sum[query_fold]
        c_count = total_shape_count - shape_count[query_fold]
        c_reg_sum = total_reg_sum - reg_sum[query_fold]
        c_reg_sup = total_reg_sup - reg_sup_sum[query_fold]
        c_reg_count = total_reg_count - reg_count[query_fold]
        if torch.any(c_count <= 0).item() or torch.any(c_reg_count <= 0).item():
            raise RuntimeError("source cross-fit reference has an empty class")
        shape_proto = c_shape_sum / (c_shape_sup.unsqueeze(-1) + eps)
        shape_support = c_shape_sup / c_count.unsqueeze(-1).to(c_shape_sup.dtype)
        trend_proto = c_trend_sum / (c_trend_sup.unsqueeze(-1) + eps)
        trend_support = c_trend_sup / c_count.unsqueeze(-1).to(c_trend_sup.dtype)
        ready = c_count > 0
        zeros_fused = torch.zeros(num_classes, shape_proto.shape[-1], device=device, dtype=shape_proto.dtype)
        empty_samples = tuple(torch.zeros(0, device=device, dtype=shape_proto.dtype) for _ in range(num_classes))
        zeros_q = torch.zeros(num_classes, len(QUANTILE_LEVELS), device=device, dtype=shape_proto.dtype)
        shape_banks.append(_cpu_source_bank(SourcePrototypeBank(
            trend_srvf=trend_proto,
            shape_srvf=shape_proto,
            trend_support=trend_support,
            shape_support=shape_support,
            fused=zeros_fused,
            class_counts=c_count,
            ready=ready,
            q_distance_samples=empty_samples,
            f_distance_samples=empty_samples,
            q_quantiles=zeros_q,
            f_quantiles=zeros_q.clone(),
            version=0,
        )))
        reg_ready = c_reg_count > 0
        reg_banks.append(_cpu_registration_bank(SourceRegistrationPrototypeBank(
            trend_srvf=c_reg_sum / (c_reg_sup.unsqueeze(-1) + eps),
            trend_support=c_reg_sup / c_reg_count.unsqueeze(-1).to(c_reg_sup.dtype),
            class_counts=c_reg_count,
            ready=reg_ready,
            registration_grid=reg_grid.to(device=device, dtype=shape_proto.dtype),
        )))
    _atomic_torch_save({
        "schema": CROSSFIT_BANK_SCHEMA,
        "fingerprint": fingerprint,
        "model_checkpoint": str(model_checkpoint.resolve()),
        "geometry_signature": str(geometry_signature),
        "folds": int(folds),
        "method": "deterministic source-true-class-stratified K-fold; query fold excluded from every reference prototype",
        "shape_banks": shape_banks,
        "registration_banks": reg_banks,
    }, cache_path)
    print(f"SEED13A2_SOURCE_CROSSFIT_BANK_BUILD|status=ready|path={cache_path}", flush=True)
    return shape_banks, reg_banks


def _load_source_crossfit_geometry_cache(
    path: Path, *, fingerprint: str, model_checkpoint: Path, folds: int, geometry_signature: str,
) -> dict[int, dict]:
    if not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CROSSFIT_GEOMETRY_SCHEMA:
        raise ValueError("unsupported 13A-2 source cross-fit geometry schema")
    if str(payload.get("fingerprint")) != str(fingerprint):
        raise ValueError("13A-2 source cross-fit geometry assignment mismatch")
    if Path(str(payload.get("model_checkpoint"))).resolve() != model_checkpoint.resolve():
        raise ValueError("13A-2 source cross-fit geometry checkpoint mismatch")
    if int(payload.get("folds")) != int(folds):
        raise ValueError("13A-2 source cross-fit geometry fold-count mismatch")
    if str(payload.get("geometry_signature")) != str(geometry_signature):
        raise ValueError("13A-2 source cross-fit geometry config mismatch")
    rows = {int(row["sample_id"]): dict(row) for row in payload.get("records", ())}
    print(f"SEED13A2_SOURCE_CROSSFIT_GEOMETRY_RESUME|completed={len(rows)}", flush=True)
    return rows


def _save_source_crossfit_geometry_cache(
    path: Path, records: Mapping[int, dict], *, fingerprint: str, model_checkpoint: Path, folds: int, geometry_signature: str,
) -> None:
    _atomic_torch_save({
        "schema": CROSSFIT_GEOMETRY_SCHEMA,
        "fingerprint": fingerprint,
        "model_checkpoint": str(model_checkpoint.resolve()),
        "geometry_signature": str(geometry_signature),
        "folds": int(folds),
        "source_true_label_used_only_for_source_reference": True,
        "contains_target_true_labels": False,
        "records": [records[key] for key in sorted(records)],
    }, path)


@torch.no_grad()
def _load_or_generate_source_crossfit_geometry(
    cache_path: Path, *, model, source_loader, source_ids: np.ndarray, source_labels: np.ndarray,
    fold_assignment: np.ndarray, shape_banks: Sequence[SourcePrototypeBank],
    reg_banks: Sequence[SourceRegistrationPrototypeBank], reg_extractor, scan_config,
    device: torch.device, model_checkpoint: Path, workers: int, geometry_chunk_size: int,
    dp_chunk_size: int, geometry_signature: str,
) -> dict[int, dict]:
    fingerprint = _fingerprint_arrays(source_ids.astype(np.int64), source_labels.astype(np.int64), fold_assignment.astype(np.int64))
    records = _load_source_crossfit_geometry_cache(
        cache_path, fingerprint=fingerprint, model_checkpoint=model_checkpoint, folds=len(shape_banks), geometry_signature=geometry_signature,
    )
    expected = set(map(int, source_ids.tolist()))
    if set(records) == expected:
        print(f"SEED13A2_SOURCE_CROSSFIT_GEOMETRY_CACHE_HIT|count={len(records)}", flush=True)
        return records
    label_by_id = {int(sid): int(label) for sid, label in zip(source_ids.tolist(), source_labels.tolist())}
    fold_by_id = {int(sid): int(fold) for sid, fold in zip(source_ids.tolist(), fold_assignment.tolist())}
    chunks: dict[int, list[dict]] = {fold: [] for fold in range(len(shape_banks))}

    def flush(fold: int) -> None:
        chunk = chunks[fold]
        if not chunk:
            return
        solved = a1._flush_geometry_chunk(
            chunk=chunk,
            source_reg_bank=reg_banks[fold],
            source_bank=shape_banks[fold],
            scan_config=scan_config,
            workers=workers,
            dp_chunk_size=dp_chunk_size,
        )
        for row in solved:
            sid = int(row["sample_id"])
            row["source_true_class"] = int(label_by_id[sid])
            row["source_crossfit_fold"] = int(fold_by_id[sid])
            records[sid] = row
        chunks[fold] = []
        _save_source_crossfit_geometry_cache(
            cache_path, records, fingerprint=fingerprint, model_checkpoint=model_checkpoint, folds=len(shape_banks), geometry_signature=geometry_signature,
        )
        print(f"SEED13A2_SOURCE_CROSSFIT_GEOMETRY_PROGRESS|completed={len(records)}/{len(expected)}", flush=True)

    model.eval()
    for raw_batch in source_loader:
        parcel_cpu = raw_batch["parcel_index"].detach().cpu().long().tolist()
        unresolved = [idx for idx, sid in enumerate(parcel_cpu) if int(sid) not in records]
        if not unresolved:
            continue
        batch = phasevis._move_batch(raw_batch, device)
        output = model(
            batch["pixels"], batch["valid_pixels"], batch["positions"], batch.get("extra"), return_geometry=True,
        )
        if output.geometry is None:
            raise RuntimeError("source cross-fit geometry requires functional geometry")
        reg = evaluate_registration_geometry(output.trend, output.positions, output.mask, reg_extractor)
        for idx in unresolved:
            sid = int(parcel_cpu[idx]); cid = int(label_by_id[sid]); fold = int(fold_by_id[sid])
            chunks[fold].append({
                "sample_id": sid,
                "raw_pred": cid,  # source true class is the legal reference class for source calibration.
                "trend_srvf": reg.trend_srvf[idx].detach().cpu(),
                "trend_support": reg.trend_support[idx].detach().cpu(),
                "trend_valid": reg.trend_valid[idx].detach().cpu(),
                "structure_srvf": output.geometry.structure_srvf[idx].detach().cpu(),
                "structure_support": output.geometry.structure_support[idx].detach().cpu(),
                "structure_valid": output.geometry.structure_valid[idx].detach().cpu(),
                "registration_grid": reg.registration_grid.detach().cpu(),
                "shape_grid": output.geometry.canonical_grid.detach().cpu(),
            })
            if len(chunks[fold]) >= int(geometry_chunk_size):
                flush(fold)
    for fold in range(len(shape_banks)):
        flush(fold)
    missing = expected - set(records)
    if missing:
        raise RuntimeError(f"13A-2 source cross-fit geometry incomplete: missing {len(missing)} source parcels")
    return records


def _geometry_reference_arrays(records: Mapping[int, dict], source_ids: np.ndarray, source_labels: np.ndarray):
    rows = [records[int(sid)] for sid in source_ids.tolist()]
    labels = np.asarray(source_labels, dtype=np.int64)
    values: dict[str, np.ndarray] = {}
    for key, _higher in GEOMETRY_RAW_SPECS:
        values[key] = np.asarray([_float_or_nan(row.get(key)) for row in rows], dtype=np.float64)
    return labels, values


def _evidence_specs() -> list[tuple[str, str, bool]]:
    specs: list[tuple[str, str, bool]] = [
        ("raw_prob", "A_classifier_baseline", True),
        ("raw_margin", "A_classifier_baseline", True),
        ("raw_entropy", "A_classifier_baseline", False),
        ("semantic_raw_candidate_similarity", "A1_source_centroid_baseline", True),
        ("semantic_raw_candidate_margin", "A1_source_centroid_baseline", True),
        ("source_class_distance", "B_source_distribution", False),
        ("source_class_distance_percentile", "B_source_distribution", False),
        ("source_nearest_candidate_distance", "B_source_distribution", False),
        ("source_nearest_other_distance", "B_source_distribution", True),
        ("source_nearest_candidate_vs_other_margin", "B_source_distribution", True),
        ("target_nearest_same_candidate_distance", "C_target_neighborhood", False),
        ("target_nearest_other_candidate_distance", "C_target_neighborhood", True),
        ("target_neighborhood_margin", "C_target_neighborhood", True),
        ("target_neighborhood_distance_ratio", "C_target_neighborhood", True),
    ]
    for k in K_VALUES:
        specs.extend([
            (f"source_knn{k}_candidate_fraction", "B_source_distribution", True),
            (f"source_knn{k}_label_entropy", "B_source_distribution", False),
            (f"source_knn{k}_candidate_vs_other_distance_margin", "B_source_distribution", True),
            (f"target_knn{k}_mean_distance", "C_target_neighborhood", False),
            (f"target_knn{k}_median_distance", "C_target_neighborhood", False),
            (f"target_knn{k}_nearest_distance", "C_target_neighborhood", False),
            (f"target_knn{k}_mean_cosine_similarity", "C_target_neighborhood", True),
            (f"target_knn{k}_candidate_fraction", "C_target_neighborhood", True),
            (f"target_knn{k}_candidate_posterior_mean", "C_target_neighborhood", True),
            (f"target_knn{k}_label_entropy", "C_target_neighborhood", False),
            (f"target_knn{k}_mutual_rate", "C_target_neighborhood", True),
        ])
    for key, higher in GEOMETRY_RAW_SPECS:
        specs.append((key, "E_geometry_raw_baseline", higher))
        specs.append((f"{key}_source_percentile", "E_geometry_source_calibrated", higher))
    return specs


def _array(rows: Sequence[dict], key: str) -> np.ndarray:
    return np.asarray([_float_or_nan(row.get(key)) for row in rows], dtype=np.float64)


def _oracle_join(unlabeled_rows: Sequence[dict], label_by_parcel: Mapping[int, int], classes: Sequence[str]) -> list[dict]:
    out: list[dict] = []
    for row in unlabeled_rows:
        item = dict(row); sid = int(item["sample_id"]); pred = int(item["raw_pred"])
        if sid not in label_by_parcel:
            raise ValueError("oracle evaluator could not find target true label")
        true = int(label_by_parcel[sid])
        item.update({
            "true_label": true,
            "true_class_name": str(classes[true]),
            "candidate_correct": bool(true == pred),
            "confusion_flow": f"{classes[true]} -> {classes[pred]}",
            "oat_correct": bool(classes[true] == "spring_oat" and classes[pred] == "spring_oat"),
            "barley_absorbed_into_oat": bool(classes[true] == "spring_barley" and classes[pred] == "spring_oat"),
            "triticale_correct": bool(classes[true] == "winter_triticale" and classes[pred] == "winter_triticale"),
            "rye_absorbed_into_triticale": bool(classes[true] == "winter_rye" and classes[pred] == "winter_triticale"),
            "wheat_absorbed_into_triticale": bool(classes[true] == "winter_wheat" and classes[pred] == "winter_triticale"),
        })
        out.append(item)
    return out


def _matching_masks(rows: Sequence[dict], classes: Sequence[str], *, seed: int):
    pred = np.asarray([int(row["raw_pred"]) for row in rows], dtype=np.int64)
    correct = np.asarray([bool(row["candidate_correct"]) for row in rows], dtype=bool)
    prob = _array(rows, "raw_prob"); margin = _array(rows, "raw_margin")
    matched_by_class: dict[int, np.ndarray] = {}
    balance_rows: list[dict] = []
    for cid, cname in enumerate(classes):
        pool = pred == cid
        local, diag = confidence_matched_indices(
            correct[pool], prob[pool], margin[pool], seed=seed + cid * 1009, bins=10,
        )
        mask = np.zeros(len(rows), dtype=bool); mask[np.flatnonzero(pool)[local]] = True
        matched_by_class[cid] = mask
        balance_rows.append({"candidate_class": cid, "candidate_class_name": cname, **diag})
    return matched_by_class, balance_rows


def _single_evidence_analysis(rows: Sequence[dict], classes: Sequence[str], *, seed: int, bootstrap_reps: int):
    pred = np.asarray([int(row["raw_pred"]) for row in rows], dtype=np.int64)
    correct = np.asarray([bool(row["candidate_correct"]) for row in rows], dtype=bool)
    raw_prob = _array(rows, "raw_prob")
    matched_by_class, balance_rows = _matching_masks(rows, classes, seed=seed)
    summary_rows: list[dict] = []
    coverage_rows: list[dict] = []
    veto_rows: list[dict] = []
    for cid, cname in enumerate(classes):
        pool = pred == cid
        global_idx = np.flatnonzero(pool)
        conf = confidence_quantile_masks(raw_prob[pool])
        slices: dict[str, np.ndarray] = {}
        for name in ("all", "top25", "top10"):
            mask = np.zeros(len(rows), dtype=bool); mask[global_idx[conf[name]]] = True; slices[name] = mask
        slices["confidence_matched"] = matched_by_class[cid]
        for evidence, group, higher in _evidence_specs():
            values = _array(rows, evidence)
            if not np.any(np.isfinite(values[pool])):
                continue
            for slice_name, mask in slices.items():
                reps = bootstrap_reps if slice_name in {"all", "confidence_matched"} else 0
                stats = summarize_evidence(
                    values[mask], correct[mask], higher_is_reliable=higher,
                    seed=seed + cid * 3001 + sum(map(ord, evidence + slice_name)), bootstrap_reps=reps,
                )
                summary_rows.append({
                    "candidate_class": cid, "candidate_class_name": cname,
                    "evidence_group": group, "evidence": evidence,
                    "analysis_slice": slice_name, **stats,
                })
            scores = values[pool] if higher else -values[pool]
            for item in precision_at_coverages(correct[pool], scores):
                coverage_rows.append({
                    "candidate_class": cid, "candidate_class_name": cname,
                    "evidence_group": group, "evidence": evidence,
                    "base_precision": float(np.mean(correct[pool])) if np.any(pool) else float("nan"), **item,
                })
            if group == "E_geometry_source_calibrated":
                for slice_name in ("all", "top25", "top10", "confidence_matched"):
                    hit = next(r for r in summary_rows if r["candidate_class"] == cid and r["evidence"] == evidence and r["analysis_slice"] == slice_name)
                    veto_rows.append({
                        "candidate_class": cid, "candidate_class_name": cname,
                        "evidence": evidence, "analysis_slice": slice_name,
                        "n": hit["n"], "n_correct": hit["n_correct"], "n_wrong": hit["n_wrong"],
                        "wrong_reject_at_correct_retention_95": hit["wrong_reject_at_correct_retention_95"],
                        "achieved_correct_retention_95": hit["achieved_correct_retention_95"],
                        "wrong_reject_at_correct_retention_90": hit["wrong_reject_at_correct_retention_90"],
                        "achieved_correct_retention_90": hit["achieved_correct_retention_90"],
                    })
    return summary_rows, coverage_rows, balance_rows, veto_rows, matched_by_class


def _critical_flow_analysis(rows: Sequence[dict], classes: Sequence[str], *, seed: int, bootstrap_reps: int):
    name_to_id = {str(name): idx for idx, name in enumerate(classes)}
    flow_specs = [
        ("spring_barley_correct", "spring_barley", "spring_barley"),
        ("spring_oat_correct", "spring_oat", "spring_oat"),
        ("spring_barley_to_spring_oat", "spring_barley", "spring_oat"),
        ("winter_triticale_correct", "winter_triticale", "winter_triticale"),
        ("winter_rye_to_winter_triticale", "winter_rye", "winter_triticale"),
        ("winter_wheat_to_winter_triticale", "winter_wheat", "winter_triticale"),
        ("winter_rye_correct", "winter_rye", "winter_rye"),
        ("winter_wheat_correct", "winter_wheat", "winter_wheat"),
    ]
    flow_rows: list[dict] = []
    for flow_name, true_name, pred_name in flow_specs:
        if true_name not in name_to_id or pred_name not in name_to_id:
            continue
        subset = [r for r in rows if int(r["true_label"]) == name_to_id[true_name] and int(r["raw_pred"]) == name_to_id[pred_name]]
        for evidence, group, higher in _evidence_specs():
            vals = _array(subset, evidence)
            if not np.any(np.isfinite(vals)):
                continue
            flow_rows.append({
                "flow": flow_name, "true_class": true_name, "candidate_class": pred_name,
                "evidence_group": group, "evidence": evidence, "higher_is_reliable": higher,
                "n": len(subset), **distribution_summary(vals),
            })
    comparisons = [
        ("oat_correct_vs_barley_absorbed", "spring_oat", ["spring_barley"]),
        ("triticale_correct_vs_rye_wheat_absorbed", "winter_triticale", ["winter_rye", "winter_wheat"]),
    ]
    matched_rows: list[dict] = []; balance_rows: list[dict] = []
    for comp_idx, (comp, candidate_name, wrong_names) in enumerate(comparisons):
        if candidate_name not in name_to_id or any(name not in name_to_id for name in wrong_names):
            continue
        cid = name_to_id[candidate_name]; wrong_ids = {name_to_id[n] for n in wrong_names}
        selected = [r for r in rows if int(r["raw_pred"]) == cid and (int(r["true_label"]) == cid or int(r["true_label"]) in wrong_ids)]
        is_correct = np.asarray([int(r["true_label"]) == cid for r in selected], dtype=bool)
        matched, balance = confidence_matched_indices(
            is_correct, _array(selected, "raw_prob"), _array(selected, "raw_margin"),
            seed=seed + comp_idx * 131, bins=10,
        )
        balance_rows.append({"comparison": comp, "candidate_class": candidate_name, "wrong_true_classes": ",".join(wrong_names), **balance})
        for evidence, group, higher in _evidence_specs():
            vals = _array(selected, evidence)
            if not np.any(np.isfinite(vals[matched])):
                continue
            stats = summarize_evidence(
                vals[matched], is_correct[matched], higher_is_reliable=higher,
                seed=seed + comp_idx * 5003 + sum(map(ord, evidence)), bootstrap_reps=bootstrap_reps,
            )
            matched_rows.append({
                "comparison": comp, "candidate_class": candidate_name,
                "wrong_true_classes": ",".join(wrong_names), "evidence_group": group,
                "evidence": evidence, **stats,
            })
    return flow_rows, matched_rows, balance_rows


def _complementarity_pairs() -> list[tuple[str, str, str, bool, str, bool]]:
    geom = [f"{name}_source_percentile" for name, _ in GEOMETRY_RAW_SPECS]
    geom_orientation = {f"{name}_source_percentile": higher for name, higher in GEOMETRY_RAW_SPECS}
    pairs = [
        ("classifier_x_source_distribution", "raw_prob_x_source_distance_percentile", "raw_prob", True, "source_class_distance_percentile", False),
        ("classifier_x_source_distribution", "raw_prob_x_source_knn20", "raw_prob", True, f"source_knn{PRIMARY_K}_candidate_fraction", True),
        ("classifier_x_target_neighborhood", "raw_prob_x_target_knn20", "raw_prob", True, f"target_knn{PRIMARY_K}_candidate_fraction", True),
        ("classifier_x_target_neighborhood", "raw_prob_x_target_boundary_margin", "raw_prob", True, "target_neighborhood_margin", True),
        ("source_distribution_x_target_neighborhood", "source_distance_percentile_x_target_knn20", "source_class_distance_percentile", False, f"target_knn{PRIMARY_K}_candidate_fraction", True),
        ("source_distribution_x_target_neighborhood", "source_knn20_x_target_knn20", f"source_knn{PRIMARY_K}_candidate_fraction", True, f"target_knn{PRIMARY_K}_candidate_fraction", True),
    ]
    for g in geom:
        pairs.append(("target_neighborhood_x_geometry", f"target_knn20_x_{g}", f"target_knn{PRIMARY_K}_candidate_fraction", True, g, geom_orientation[g]))
        pairs.append(("source_distribution_x_geometry", f"source_distance_percentile_x_{g}", "source_class_distance_percentile", False, g, geom_orientation[g]))
    return pairs


def _complementarity_analysis(rows: Sequence[dict], classes: Sequence[str], matched_by_class: Mapping[int, np.ndarray]):
    pred = np.asarray([int(r["raw_pred"]) for r in rows], dtype=np.int64)
    correct = np.asarray([bool(r["candidate_correct"]) for r in rows], dtype=bool)
    raw_prob = _array(rows, "raw_prob")
    summary: list[dict] = []; grids: list[dict] = []; joints: list[dict] = []
    for cid, cname in enumerate(classes):
        pool = pred == cid; idx = np.flatnonzero(pool); conf = confidence_quantile_masks(raw_prob[pool])
        slices = {"all": pool, "confidence_matched": matched_by_class[cid]}
        for name in ("top25", "top10"):
            mask = np.zeros(len(rows), dtype=bool); mask[idx[conf[name]]] = True; slices[name] = mask
        for family, pair, a_name, a_high, b_name, b_high in _complementarity_pairs():
            a = _array(rows, a_name); b = _array(rows, b_name)
            for slice_name, mask in slices.items():
                valid = mask & np.isfinite(a) & np.isfinite(b)
                if valid.sum() < 3:
                    continue
                rho, grid_rows, joint_rows = complementarity_diagnostics(
                    a[valid], b[valid], correct[valid],
                    a_higher_is_support=a_high, b_higher_is_support=b_high,
                )
                summary.append({
                    "candidate_class": cid, "candidate_class_name": cname,
                    "analysis_slice": slice_name, "pair_family": family, "pair": pair,
                    "a_evidence": a_name, "a_higher_is_support": a_high,
                    "b_evidence": b_name, "b_higher_is_support": b_high,
                    "n": int(valid.sum()), "base_precision": float(np.mean(correct[valid])),
                    "spearman_support_oriented": rho,
                })
                for item in grid_rows:
                    grids.append({
                        "candidate_class": cid, "candidate_class_name": cname,
                        "analysis_slice": slice_name, "pair_family": family, "pair": pair, **item,
                    })
                for item in joint_rows:
                    joints.append({
                        "candidate_class": cid, "candidate_class_name": cname,
                        "analysis_slice": slice_name, "pair_family": family, "pair": pair, **item,
                    })
    return summary, grids, joints


def _pool_health_rows(
    *, classes: Sequence[str], target_features: np.ndarray, source_labels: np.ndarray,
    source_loo_distance: np.ndarray, raw_pred: np.ndarray, raw_posterior: np.ndarray,
    target_obs: Mapping[str, np.ndarray], pool_distance: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    n = raw_pred.size
    for cid, cname in enumerate(classes):
        pool = raw_pred == cid
        source_c = source_labels == cid
        row = {
            "candidate_class": cid,
            "candidate_class_name": cname,
            "N_candidate": int(pool.sum()),
            "candidate_fraction": float(pool.mean()),
            "posterior_mass_all_target": float(raw_posterior[:, cid].sum()),
            "candidate_pool_centroid_distance_mean": float(np.mean(pool_distance[pool])) if np.any(pool) else float("nan"),
            "candidate_pool_centroid_distance_median": float(np.median(pool_distance[pool])) if np.any(pool) else float("nan"),
            "source_class_loo_dispersion_mean": float(np.mean(source_loo_distance[source_c])) if np.any(source_c) else float("nan"),
            "source_class_loo_dispersion_median": float(np.median(source_loo_distance[source_c])) if np.any(source_c) else float("nan"),
        }
        denom = row["source_class_loo_dispersion_median"]
        row["target_source_dispersion_ratio_median"] = (
            float(row["candidate_pool_centroid_distance_median"] / denom)
            if math.isfinite(denom) and abs(denom) > 1e-12 else float("nan")
        )
        for k in K_VALUES:
            for key in ("mean_distance", "candidate_fraction", "label_entropy", "mutual_rate"):
                values = np.asarray(target_obs[f"target_knn{k}_{key}"], dtype=np.float64)[pool]
                row[f"target_knn{k}_{key}_mean"] = float(np.mean(values)) if values.size else float("nan")
                row[f"target_knn{k}_{key}_median"] = float(np.median(values)) if values.size else float("nan")
                row[f"target_knn{k}_{key}_q75"] = float(np.quantile(values, 0.75)) if values.size else float("nan")
        rows.append(row)
    return rows


def _pool_oracle_context(pool_rows: Sequence[dict], audited: Sequence[dict], classes: Sequence[str]) -> list[dict]:
    out: list[dict] = []
    for base in pool_rows:
        cid = int(base["candidate_class"])
        subset = [r for r in audited if int(r["raw_pred"]) == cid]
        counts = {str(name): 0 for name in classes}
        for r in subset:
            counts[str(classes[int(r["true_label"])])] += 1
        item = dict(base)
        item["oracle_candidate_precision"] = float(np.mean([bool(r["candidate_correct"]) for r in subset])) if subset else float("nan")
        for name in classes:
            item[f"oracle_true_count_{name}"] = counts[str(name)]
        out.append(item)
    return out


def _precompute_pca_payloads(
    source_features: np.ndarray, source_labels: np.ndarray, target_features: np.ndarray,
    raw_pred: np.ndarray, classes: Sequence[str], *, max_points: int, seed: int,
) -> dict[str, dict]:
    rng = np.random.default_rng(int(seed)); name_to_id = {str(name): i for i, name in enumerate(classes)}
    specs = {
        "spring_source_plus_oat_candidate": (["spring_oat", "spring_barley"], "spring_oat"),
        "winter_source_plus_triticale_candidate": (["winter_triticale", "winter_rye", "winter_wheat"], "winter_triticale"),
    }
    payloads: dict[str, dict] = {}
    for key, (source_names, candidate_name) in specs.items():
        if candidate_name not in name_to_id or any(name not in name_to_id for name in source_names):
            continue
        source_mask = np.isin(source_labels, [name_to_id[name] for name in source_names])
        target_mask = raw_pred == name_to_id[candidate_name]
        source_idx = np.flatnonzero(source_mask); target_idx = np.flatnonzero(target_mask)
        if source_idx.size > max_points:
            source_idx = np.sort(rng.choice(source_idx, max_points, replace=False))
        if target_idx.size > max_points:
            target_idx = np.sort(rng.choice(target_idx, max_points, replace=False))
        merged = np.concatenate([source_features[source_idx], target_features[target_idx]], axis=0)
        coords, ratio = pca_project(merged, 2)
        payloads[key] = {
            "source_indices": source_idx,
            "target_indices": target_idx,
            "source_coords": coords[:source_idx.size],
            "target_coords": coords[source_idx.size:],
            "explained_variance_ratio": ratio,
            "source_true_classes": source_labels[source_idx],
            "target_candidate_class": int(name_to_id[candidate_name]),
            "source_class_names": source_names,
        }
    for candidate_name in ("spring_oat", "winter_triticale"):
        if candidate_name not in name_to_id:
            continue
        target_idx = np.flatnonzero(raw_pred == name_to_id[candidate_name])
        if target_idx.size > max_points * 2:
            target_idx = np.sort(rng.choice(target_idx, max_points * 2, replace=False))
        coords, ratio = pca_project(target_features[target_idx], 2)
        payloads[f"target_pool_{candidate_name}"] = {
            "target_indices": target_idx,
            "target_coords": coords,
            "explained_variance_ratio": ratio,
            "target_candidate_class": int(name_to_id[candidate_name]),
        }
    return payloads


def _save_pca_npz(path: Path, payloads: Mapping[str, dict]) -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    for key, payload in payloads.items():
        metadata[key] = {}
        for name, value in payload.items():
            if isinstance(value, np.ndarray):
                arrays[f"{key}__{name}"] = value
            else:
                metadata[key][name] = value
    arrays["metadata_json_utf8"] = np.frombuffer(json.dumps(metadata, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


def _plot_pca_payloads(output: Path, payloads: Mapping[str, dict], audited: Sequence[dict], source_labels: np.ndarray, classes: Sequence[str]) -> None:
    true_by_index = np.asarray([int(r["true_label"]) for r in audited], dtype=np.int64)
    for key, payload in payloads.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        if "source_coords" in payload:
            sc = payload["source_coords"]; src_cls = np.asarray(payload["source_true_classes"], dtype=np.int64)
            for cid in np.unique(src_cls):
                mask = src_cls == cid
                ax.scatter(sc[mask, 0], sc[mask, 1], s=8, alpha=0.30, label=f"source true {classes[int(cid)]}")
            tc = payload["target_coords"]; tidx = np.asarray(payload["target_indices"], dtype=np.int64); true = true_by_index[tidx]
            for cid in np.unique(true):
                mask = true == cid
                ax.scatter(tc[mask, 0], tc[mask, 1], s=10, alpha=0.55, label=f"target oracle true {classes[int(cid)]}")
        else:
            tc = payload["target_coords"]; tidx = np.asarray(payload["target_indices"], dtype=np.int64); true = true_by_index[tidx]
            for cid in np.unique(true):
                mask = true == cid
                ax.scatter(tc[mask, 0], tc[mask, 1], s=9, alpha=0.55, label=f"oracle true {classes[int(cid)]}")
        ratio = np.asarray(payload["explained_variance_ratio"], dtype=np.float64)
        ax.set_xlabel(f"PCA1 ({100*ratio[0]:.1f}% var)"); ax.set_ylabel(f"PCA2 ({100*ratio[1]:.1f}% var)")
        ax.set_title(f"13A-2 frozen LTAE PCA: {key}\ncoordinates built without target true labels; oracle labels only color points")
        ax.legend(fontsize=8, markerscale=1.5); ax.grid(alpha=0.2)
        path = output / "plots" / "pca" / f"{key}.png"; path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_pair_scatter(output: Path, rows: Sequence[dict], classes: Sequence[str], *, a_name: str, b_name: str, filename: str) -> None:
    pred = np.asarray([int(r["raw_pred"]) for r in rows], dtype=np.int64)
    correct = np.asarray([bool(r["candidate_correct"]) for r in rows], dtype=bool)
    a = _array(rows, a_name); b = _array(rows, b_name)
    for candidate_name in ("spring_oat", "winter_triticale"):
        if candidate_name not in classes:
            continue
        cid = classes.index(candidate_name); mask = (pred == cid) & np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(a[mask & correct], b[mask & correct], s=10, alpha=0.50, label="oracle correct candidate")
        ax.scatter(a[mask & ~correct], b[mask & ~correct], s=10, alpha=0.35, label="oracle wrong candidate")
        ax.set_xlabel(a_name); ax.set_ylabel(b_name)
        ax.set_title(f"{candidate_name}: {a_name} × {b_name}\noracle color is post-hoc only")
        ax.legend(); ax.grid(alpha=0.2)
        path = output / "plots" / "pair_scatter" / f"{candidate_name}_{filename}.png"; path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_precision_coverage_by_evidence(output: Path, coverage_rows: Sequence[dict], classes: Sequence[str]) -> None:
    key_classes = {name for name in ("spring_barley", "spring_oat", "winter_rye", "winter_triticale", "winter_wheat") if name in classes}
    evidence_names = sorted({str(r["evidence"]) for r in coverage_rows if str(r["evidence_group"]) not in {"A_classifier_baseline", "A1_source_centroid_baseline", "E_geometry_raw_baseline"}})
    for evidence in evidence_names:
        subset = [r for r in coverage_rows if r["evidence"] == evidence and r["candidate_class_name"] in key_classes]
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for cname in sorted(key_classes):
            rows = sorted([r for r in subset if r["candidate_class_name"] == cname], key=lambda r: float(r["coverage"]))
            if rows:
                ax.plot([100*float(r["coverage"]) for r in rows], [float(r["precision"]) for r in rows], marker="o", label=cname)
        ax.set_xlabel("candidate coverage (%)"); ax.set_ylabel("oracle precision")
        ax.set_ylim(0, 1.02); ax.set_title(f"13A-2 Precision–Coverage: {evidence}")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
        path = output / "plots" / "precision_coverage" / f"{evidence}.png"; path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)



def _fmt(value) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(x) else f"{x:.4f}"


def _write_metric_interpretation_report(
    path: Path, *, critical_matched: Sequence[dict], critical_balance: Sequence[dict],
    veto_rows: Sequence[dict], complementarity_rows: Sequence[dict], classes: Sequence[str],
) -> None:
    """Write a fixed, non-selective numerical interpretation report.

    Evidence names are protocol-fixed below; nothing is chosen by oracle AUROC.
    The report deliberately stops short of an automatic 13A-2 verdict.
    """
    fixed_evidence = (
        "source_class_distance_percentile",
        f"source_knn{PRIMARY_K}_candidate_fraction",
        f"target_knn{PRIMARY_K}_candidate_fraction",
        "target_neighborhood_margin",
        "T_registered_error_source_percentile",
        "S_registered_error_source_percentile",
    )
    lines = [
        "# 13A-2 关键数值解释（oracle-only diagnostic）",
        "",
        "> 本文件在全部无标签 observable 保存完成后生成。target true label 只用于事后 correctness/flow 评价；这里的任何数值都不能反馈到 K、距离、阈值、特征选择或 `C(c)→T(c)`。",
        "",
        "## Confidence matching 是否成立",
        "",
    ]
    for row in critical_balance:
        lines.append(
            f"- **{row['comparison']}**：matched correct={row.get('n_correct_matched', 0)}，"
            f"wrong={row.get('n_wrong_matched', 0)}；raw probability SMD={_fmt(row.get('raw_prob_smd_after'))}，"
            f"raw margin SMD={_fmt(row.get('raw_margin_smd_after'))}。"
        )
        p_smd = _float_or_nan(row.get("raw_prob_smd_after")); m_smd = _float_or_nan(row.get("raw_margin_smd_after"))
        if math.isfinite(p_smd) and math.isfinite(m_smd) and abs(p_smd) < 0.1 and abs(m_smd) < 0.1:
            lines.append("  - 两个控制变量的 `|SMD|<0.1`，说明正确/错误组在 classifier confidence 与 margin 上已经较接近；随后 evidence 的差异更不容易被解释为单纯 confidence 重复。")
        else:
            lines.append("  - 至少一个控制变量的 `|SMD|>=0.1` 或不可计算；该 comparison 的 matched evidence 需要谨慎解释，不能宣称已经完全控制 classifier confidence。")
    lines.extend(["", "## 核心高置信错误流：固定 evidence", ""])
    for comparison in ("oat_correct_vs_barley_absorbed", "triticale_correct_vs_rye_wheat_absorbed"):
        lines.append(f"### {comparison}")
        lines.append("")
        subset = {str(r["evidence"]): r for r in critical_matched if str(r["comparison"]) == comparison}
        lines.append("| evidence | N | base precision | AUROC | AUPRC | correct mean | wrong mean |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for evidence in fixed_evidence:
            row = subset.get(evidence)
            if row is None:
                lines.append(f"| {evidence} | 0 | NA | NA | NA | NA | NA |")
                continue
            lines.append(
                f"| {evidence} | {row.get('n', 0)} | {_fmt(row.get('base_precision'))} | {_fmt(row.get('auroc'))} | "
                f"{_fmt(row.get('auprc'))} | {_fmt(row.get('correct_mean'))} | {_fmt(row.get('wrong_mean'))} |"
            )
        lines.extend([
            "",
            "解释：AUROC 约为 0.5 表示对 matched correct/wrong 基本没有排序能力；例如 AUROC=0.64 可读作随机抽一对 correct/wrong 时约有 64% 概率把 correct 排在预先规定的可靠方向。它只证明存在一定信息，不能单独成为训练 gate。AUPRC 必须与该 comparison 的 base precision 一起看。",
            "",
        ])
    lines.extend(["## Geometry veto：固定 T/S registered-error percentile", ""])
    for cname in ("spring_oat", "winter_triticale"):
        if cname not in classes:
            continue
        rows = [r for r in veto_rows if r.get("candidate_class_name") == cname and r.get("analysis_slice") == "confidence_matched" and r.get("evidence") in {"T_registered_error_source_percentile", "S_registered_error_source_percentile"}]
        for row in rows:
            w95 = _float_or_nan(row.get("wrong_reject_at_correct_retention_95")); w90 = _float_or_nan(row.get("wrong_reject_at_correct_retention_90"))
            lines.append(
                f"- **{cname} / {row['evidence']}**：WRR@CR95={_fmt(w95)}，WRR@CR90={_fmt(w90)}。"
            )
            if math.isfinite(w95):
                lines.append(f"  - WRR@CR95={w95:.3f} 表示在仍保留约 95% 正确 candidate 的条件下，该 source-calibrated geometry evidence 可排除约 {100*w95:.1f}% 错误 candidate；这只表示 veto potential，不是正向类别确认。")
    lines.extend(["", "## 五组互补性：固定 K=20 代表关系", ""])
    fixed_pairs = (
        "raw_prob_x_source_distance_percentile",
        "raw_prob_x_target_knn20",
        "source_distance_percentile_x_target_knn20",
        "target_knn20_x_T_registered_error_source_percentile",
        "source_distance_percentile_x_T_registered_error_source_percentile",
    )
    lines.append("| candidate | slice | pair | support-oriented Spearman | N | base precision |")
    lines.append("|---|---|---|---:|---:|---:|")
    for cname in ("spring_oat", "winter_triticale"):
        for pair in fixed_pairs:
            hits = [r for r in complementarity_rows if r.get("candidate_class_name") == cname and r.get("analysis_slice") == "confidence_matched" and r.get("pair") == pair]
            if not hits:
                continue
            row = hits[0]
            lines.append(f"| {cname} | confidence_matched | {pair} | {_fmt(row.get('spearman_support_oriented'))} | {row.get('n', 0)} | {_fmt(row.get('base_precision'))} |")
    lines.extend([
        "",
        "解释：Spearman 接近 1 或 -1 表示两项 support-oriented 排序高度重复；相关较低本身也不等于互补，必须同时结合单项可分性、`10b_fixed_quantile_4x4.csv` 的二维富集和 `10c_fixed_top_intersections.csv` 的固定 top25%/top10% 联合区域。",
        "",
        "## 本报告明确不做的事",
        "",
        "- 不按 AUROC 排名后自动挑 best evidence；",
        "- 不根据 4×4 oracle precision 选择阈值；",
        "- 不学习 Logistic/MLP/XGBoost 等 reliability model；",
        "- 不生成 class-specific rule；",
        "- 不自动判定 13A-2 为完成/部分完成/失败；最终理论判定必须结合全部固定输出。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")

def _write_readme(path: Path) -> None:
    text = f"""# 实验 13A-2：缺失语义结构与多证据互补性诊断

## 实验目的

本实验在与 13A-1 完全相同的冻结 Stage-1 状态下，补测四类静态无标签信息：source 类别分布符合度、target frozen-LTAE 局部邻域、candidate-pool 群体健康度、source-calibrated T/S 几何异常。`delta_boot = identity`，不人为制造第二个 Phase 视图。实验不训练模型、不更新 PSE/LTAE/classifier/Phase、不运行 Teacher–Student，也不定义 `C(c)→T(c)` 或多证据加权公式。

raw candidate 永远来自 frozen classifier raw top-1。Source KNN、target KNN、geometry 和 pool health 只能支持/质疑这个 candidate，不能重新分类。

## 标签边界

`01/02/03/03b` 在读取 target true label **之前**写出。target KNN 的邻居选择、PCA 坐标、source conformity、geometry percentile 和 pool health 均不使用 target true label。`04_oracle_sample_audit.csv` 开始才进行 oracle-only join；所有 oracle 字段仅用于事后评价和绘图着色，不参与无监督决策。

## 固定实现选择

- LTAE feature：Stage-1 `fused_repr`。
- 距离：cosine distance，与 13A-1 的 source semantic cosine reference 保持同一度量语义。
- KNN：exact chunked cosine KNN；K 固定为 `{K_VALUES}`，不是用 oracle 调出的超参数。
- 互补性主图使用固定中尺度 `K={PRIMARY_K}`；其它 K 仍全部输出用于尺度稳定性检查。
- source feature distance calibration：source class centroid 使用 leave-one-out 距离形成 reference distribution；target query 使用完整 source class centroid。
- source geometry calibration：固定 source-true-class-stratified 5-fold cross-fit。每个 source sample 的 T/S registration reference 都由不包含该 sample 所在 fold 的 4/5 source data 构成。fold 只由 source parcel id + source true class 决定，与 target label 无关。
- optional perturbation uncertainty F：本轮关闭。13A-1/13A-2 的 selected evaluation loader 使用完整 parcel pixels + Identity transform；为了实验 F 临时切换 RandomSamplePixels 会改变 A1 输入协议，因此不新增随机视图。

## 输出文件

- `00_manifest.json`：checkpoint、split、距离、K、source cross-fit、label boundary、明确禁止项。
- `01_unlabeled_sample_observables.csv`：全部逐样本 label-free scalar。保留 13A-1 baseline/raw geometry，并加入 source distance percentile、source KNN、target KNN/mutual/boundary 以及每项 geometry source percentile。**不含 target true label。**
- `02_unlabeled_dense_vectors.npz`：target frozen LTAE feature、posterior、source/target KNN indices/distances。**不含 target true label。**
- `03_unlabeled_candidate_pool_health.csv`：每个 raw candidate pool 的 count、posterior mass、feature dispersion、source/target dispersion ratio、local density/neighbor consistency/entropy 摘要。它是 pool-level context，不进入 sample score。
- `03b_unlabeled_pca_coordinates.npz`：仅由 source labels（合法）+ target raw candidate + frozen features 决定的 PCA 坐标；target true label 不参与 PCA。
- `04_oracle_sample_audit.csv`：在 01–03b 落盘后才 join target true label / candidate correctness / confusion flow。
- `05_single_evidence_summary.csv`：per candidate class × evidence × all/top25/top10/confidence-matched 的 correct/wrong 分布、bootstrap 95% CI、AUROC/AUPRC、WRR@CR95/90。
- `06_precision_coverage.csv`：5/10/20/30/50/100% fixed coverage precision，并带 candidate base precision。
- `07_confidence_matched_balance.csv`：同 candidate class 内 raw probability × raw margin 10×10 coarsened exact matching 的匹配前后 SMD。通常 `|SMD|<0.1` 表示控制变量差异已经很小。
- `08_geometry_veto_summary.csv`：source-calibrated geometry 的 `WrongRejectRate@CorrectRetention=95/90%`。geometry 只解释为 veto potential。
- `09_critical_confusion_flows.csv`：barley/oat、rye/wheat/triticale 指定 flow 的逐 evidence 分布。
- `09b_critical_confidence_matched.csv`：`oat→oat vs barley→oat`、`triticale→triticale vs rye/wheat→triticale` 在 confidence matching 后的新增 evidence AUROC/AUPRC。
- `09c_critical_match_balance.csv`：上述关键 flow 的 raw confidence/margin SMD，必须先看 balance 再解释 09b。
- `10_complementarity_summary.csv`：理论要求五组关系的 support-oriented Spearman。相关高表示信息重复度可能较高；低相关且两项各自有可分性才支持互补可能。
- `10b_fixed_quantile_4x4.csv`：每项 evidence 按 candidate pool 自身固定四分位切成 4×4，无 target-label threshold search。
- `10c_fixed_top_intersections.csv`：固定 top25% / top10% 两证据交集的 N/coverage/oracle precision，不搜索其它 coverage。
- `11_candidate_pool_oracle_context.csv`：03 的 oracle-only 后验解释，展示 candidate pool 真实组成；不能反馈给正式机制。
- `12_13a2_diagnostic_summary.json`：协议与关键 pool/flow 摘要，不自动判“任务完成/部分完成/失败”，最终理论判定仍由实验结果解释后给出。
- `13_metric_interpretation.md`：对预先固定的关键 matched evidence、SMD、geometry WRR 和五类代表性 Spearman 写入实际数值与中文解释；不按 oracle 结果自动挑 best evidence。

## 指标解释

- **AUROC**：例如 0.64 表示随机抽一个正确 candidate 和一个错误 candidate，该 evidence 约有 64% 概率把正确样本排在更可靠一侧；存在信息但远不足以自动成为训练 gate。
- **AUPRC**：必须和同 candidate pool 的 base precision 一起看。对 spring_oat 这类低 base-precision pool，20% precision 与健康类别的 95% precision 不是同一个含义。
- **Precision–Coverage**：例如 top10% precision 表示按该 evidence 的预先规定方向只保留最可靠 10% candidate 时的 oracle 正确比例。这里只诊断“小而纯核心”，不冻结 threshold。
- **SMD**：匹配后 `|SMD|<0.1` 通常表示 raw probability/margin 已较接近；若这时新 evidence 仍能分离 correct/wrong，才更能说明它不是 confidence 的重复表达。
- **WrongReject@95%CorrectRetention**：例如 0.17 表示仍保留至少约 95% 正确 candidate 时，该 veto evidence 能排掉约 17% 错误 candidate；说明有限但真实的否决价值。
- **Target neighbor candidate fraction**：只表示邻域当前预测一致，**不等于 candidate 正确**。大量 absorbed barley 若自己形成紧密群，也可能得到接近 1 的 oat-neighbor fraction。
- **Local density / mutual-neighbor**：只描述局部结构稳定度，不能单独解释为正确性。
- **Source percentile**：0.95 表示该值约高于 95% source true-class reference samples。对 error/phase magnitude 等“越小越可靠”量，高 percentile 更像异常；对 gain 等“越大越支持”量，方向在 CSV 的 `higher_is_reliable` 中明确保存。

## 必看问题

1. `oat→oat` 与 `barley→oat` 在 source distribution / target neighborhood 上是否出现明显差异？
2. `triticale→triticale` 与 `rye/wheat→triticale` 是否也有可解释增量结构？
3. confidence-matched 后新 evidence 是否仍有 AUROC/AUPRC 增益？
4. source distribution × target neighborhood 是否在固定 4×4 / top25/top10 联合区域出现更高纯度，而 Spearman 又没有接近完全相关？
5. geometry source percentile 在 target local support 已较高时还能否额外 veto 错误？
6. spring_oat / winter_triticale pool 是否同时表现为数量膨胀、feature dispersion 增大、邻域多结构/混杂？

## 不能支持的结论

本实验不能直接给出 TRAINABLE gate、class-specific threshold、可靠性权重、target prior、最终 pseudo-label、Teacher/Student 规则，也不能根据 oracle 最优格子或最高 AUROC 自动挑 evidence。PCA/散点图中的 target true label 只用于事后着色；UMAP 未作为主要数值依据。
"""
    path.write_text(text, encoding="utf-8")


def run(args) -> dict:
    a1_rows, a1_manifest = _validate_13a1_input(args.experiment13a1_dir.resolve())
    calibration = torch.load(args.calibration_checkpoint.resolve(), map_location="cpu", weights_only=False)
    runtime = calibration.get("runtime_config")
    if not isinstance(runtime, dict):
        raise ValueError("calibration checkpoint is missing runtime_config")
    classes = [str(v) for v in runtime["classes"]]
    source = str(runtime["source"]); target = str(runtime["target"]); seed = int(runtime["seed"]); fold = int(args.fold)
    expected_task = ("austria/33UVP/2017", "denmark/32VNH/2017", 1, 0)
    actual_task = (source, target, seed, fold)
    if actual_task != expected_task:
        raise ValueError(
            "first 13A-2 run is frozen to AT1->DK1 seed=1 fold=0 "
            f"(expected runtime={expected_task}, got runtime={actual_task})"
        )
    data_root = str(args.data_root or runtime["data_root"])
    closed_set = bool(runtime.get("closed_set", True)); combine = bool(runtime.get("combine_spring_and_winter", False))
    time_mode = str(runtime.get("time_coordinate_mode", "canonical_day_of_year"))
    val_ratio = float(runtime.get("val_ratio", 0.1)); test_ratio = float(runtime.get("test_ratio", 0.2))
    device = torch.device(args.device); knn_device = torch.device(args.knn_device)
    model_checkpoint_path = args.model_checkpoint.resolve()
    model_checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    model = phasevis._build_model(runtime, calibration, device, model_checkpoint=model_checkpoint)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    source_all = phasevis._eligible_parcels(data_root, source, classes, closed_set=closed_set, combine_spring_and_winter=combine, time_coordinate_mode=time_mode)
    target_all = phasevis._eligible_parcels(data_root, target, classes, closed_set=closed_set, combine_spring_and_winter=combine, time_coordinate_mode=time_mode)
    splits = phasevis._reconstruct_fold_splits(source_all, target_all, source=source, target=target, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio, fold=fold)
    source_train_parcels = np.asarray(sorted(splits[source]["train"]), dtype=np.int64)
    target_train_parcels = np.asarray(sorted(splits[target]["train"]), dtype=np.int64)
    source_loader = phasevis._selected_loader(data_root, source, classes, source_train_parcels, closed_set=closed_set, combine_spring_and_winter=combine, time_coordinate_mode=time_mode, batch_size=args.batch_size, num_workers=args.num_workers)
    target_loader_raw = phasevis._selected_loader(data_root, target, classes, target_train_parcels, closed_set=closed_set, combine_spring_and_winter=combine, time_coordinate_mode=time_mode, batch_size=args.batch_size, num_workers=args.num_workers)
    target_loader = a1.LabelStrippedLoader(target_loader_raw)

    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"; cache.mkdir(parents=True, exist_ok=True)
    source_feat = _load_or_generate_feature_cache(
        cache / "source_frozen_ltae_features.pt", schema=SOURCE_FEATURE_SCHEMA, model=model,
        loader=source_loader, device=device, source_labels_allowed=True, source=source, target=target,
        seed=seed, fold=fold, model_checkpoint=model_checkpoint_path,
    )
    target_feat = _load_or_generate_feature_cache(
        cache / "target_frozen_ltae_features.pt", schema=TARGET_FEATURE_SCHEMA, model=model,
        loader=target_loader, device=device, source_labels_allowed=False, source=source, target=target,
        seed=seed, fold=fold, model_checkpoint=model_checkpoint_path,
    )
    if set(map(int, source_feat["sample_ids"])) != set(map(int, source_train_parcels)):
        raise ValueError("source frozen feature cache does not cover exactly source-train")
    if set(map(int, target_feat["sample_ids"])) != set(map(int, target_train_parcels)):
        raise ValueError("target frozen feature cache does not cover exactly target-train")

    # Align the immutable 13A-1 candidate rows to the new frozen-feature cache.
    a1_by_id = {int(r["sample_id"]): r for r in a1_rows}
    target_ids = np.asarray(target_feat["sample_ids"], dtype=np.int64)
    if set(a1_by_id) != set(map(int, target_ids.tolist())):
        raise ValueError("13A-1 candidate rows and 13A-2 target feature cache cover different parcels")
    aligned_a1 = [a1_by_id[int(sid)] for sid in target_ids.tolist()]
    candidates = np.asarray([_int(r["raw_pred"]) for r in aligned_a1], dtype=np.int64)
    if not np.array_equal(candidates, np.asarray(target_feat["raw_pred"], dtype=np.int64)):
        raise ValueError("13A-2 frozen inference changed 13A-1 raw top-1 candidate assignment")
    a1_prob = np.asarray([_float_or_nan(r["raw_prob"]) for r in aligned_a1], dtype=np.float64)
    target_post = np.asarray(target_feat["raw_posterior"], dtype=np.float64)
    if np.max(np.abs(a1_prob - target_post[np.arange(target_post.shape[0]), candidates])) > 2e-5:
        raise ValueError("13A-2 frozen inference probability mismatch vs 13A-1")

    source_features = np.asarray(source_feat["features"], dtype=np.float32)
    source_ids = np.asarray(source_feat["sample_ids"], dtype=np.int64)
    source_labels = np.asarray(source_feat["labels"], dtype=np.int64)
    target_features = np.asarray(target_feat["features"], dtype=np.float32)

    # B1: source class distribution conformity with LOO source calibration.
    source_loo_dist, source_centroids = loo_class_centroid_distances(source_features, source_labels, len(classes))
    target_source_dist = cosine_distance_rows(target_features, source_centroids[candidates])
    target_source_percentile = empirical_percentiles_by_class(
        source_loo_dist, source_labels, target_source_dist, candidates, len(classes),
    )

    # B2/C: exact source/target cosine KNN at pre-fixed scales.
    print(f"SEED13A2_SOURCE_KNN_START|query={len(target_features)}|reference={len(source_features)}|kmax={max(K_VALUES)}|device={knn_device}", flush=True)
    source_knn = exact_cosine_knn(
        target_features, source_features, k_max=max(K_VALUES), device=knn_device,
        chunk_size=args.knn_chunk_size, query_groups=candidates, reference_groups=source_labels,
    )
    source_obs = source_knn_observables(source_knn, source_labels, candidates, num_classes=len(classes), k_values=K_VALUES)
    print(f"SEED13A2_TARGET_KNN_START|n={len(target_features)}|kmax={max(K_VALUES)}|device={knn_device}", flush=True)
    target_knn = exact_cosine_knn(
        target_features, target_features, k_max=max(K_VALUES), device=knn_device,
        chunk_size=args.knn_chunk_size, exclude_self=True,
        query_reference_indices=np.arange(len(target_features), dtype=np.int64),
        query_groups=candidates, reference_groups=candidates,
    )
    target_obs = target_knn_observables(target_knn, candidates, target_post, num_classes=len(classes), k_values=K_VALUES)
    pool_distance = pool_centroid_distances(target_features, candidates, len(classes))

    # E: source-calibrated T/S geometry using source-only 5-fold cross-fit.
    source_fold = _stratified_source_fold_assignment(source_ids, source_labels, args.source_crossfit_folds)
    scan_config = samplediag._scan_config(runtime, args.registration_workers)
    geometry_signature = repr(scan_config)
    reg_extractor = build_stage2_registration_extractor(model, device=device, k_reg=scan_config.k_reg)
    shape_banks, reg_banks = _build_crossfit_reference_banks(
        cache / "source_crossfit_reference_banks.pt", model=model, source_loader=source_loader,
        source_ids=source_ids, source_labels=source_labels, fold_assignment=source_fold,
        num_classes=len(classes), folds=args.source_crossfit_folds, device=device,
        reg_extractor=reg_extractor, model_checkpoint=model_checkpoint_path, geometry_signature=geometry_signature,
    )
    source_geo = _load_or_generate_source_crossfit_geometry(
        cache / "source_crossfit_geometry.pt", model=model, source_loader=source_loader,
        source_ids=source_ids, source_labels=source_labels, fold_assignment=source_fold,
        shape_banks=shape_banks, reg_banks=reg_banks, reg_extractor=reg_extractor,
        scan_config=scan_config, device=device, model_checkpoint=model_checkpoint_path,
        workers=args.registration_workers, geometry_chunk_size=args.geometry_chunk_size,
        dp_chunk_size=args.dp_target_chunk_size, geometry_signature=geometry_signature,
    )
    source_geo_labels, source_geo_values = _geometry_reference_arrays(source_geo, source_ids, source_labels)

    unlabeled_rows: list[dict] = []
    for idx, base in enumerate(aligned_a1):
        row = dict(base)
        row["target_index"] = int(idx)
        row["candidate_class"] = int(candidates[idx])
        row["candidate_class_name"] = str(classes[candidates[idx]])
        row["source_class_distance"] = float(target_source_dist[idx])
        row["source_class_distance_percentile"] = float(target_source_percentile[idx])
        row["candidate_pool_centroid_distance"] = float(pool_distance[idx])
        for name, values in source_obs.items():
            row[name] = float(values[idx])
        for name, values in target_obs.items():
            row[name] = float(values[idx])
        unlabeled_rows.append(row)
    for key, _higher in GEOMETRY_RAW_SPECS:
        query = np.asarray([_float_or_nan(row.get(key)) for row in unlabeled_rows], dtype=np.float64)
        percentile = empirical_percentiles_by_class(
            source_geo_values[key], source_geo_labels, query, candidates, len(classes),
        )
        for idx, row in enumerate(unlabeled_rows):
            row[f"{key}_source_percentile"] = float(percentile[idx])

    pool_rows = _pool_health_rows(
        classes=classes, target_features=target_features, source_labels=source_labels,
        source_loo_distance=source_loo_dist, raw_pred=candidates, raw_posterior=target_post,
        target_obs=target_obs, pool_distance=pool_distance,
    )
    pca_payloads = _precompute_pca_payloads(
        source_features, source_labels, target_features, candidates, classes,
        max_points=args.pca_max_points, seed=args.analysis_seed,
    )

    # Hard label-free boundary: all primary observables are materialized first.
    if any(FORBIDDEN_TARGET_ORACLE_FIELDS.intersection(row) for row in unlabeled_rows):
        raise RuntimeError("oracle field leaked into 13A-2 label-free sample artifact")
    _write_csv(output / "01_unlabeled_sample_observables.csv", unlabeled_rows)
    np.savez_compressed(
        output / "02_unlabeled_dense_vectors.npz",
        sample_id=target_ids,
        frozen_ltae_feature=target_features,
        raw_posterior=target_post.astype(np.float32),
        source_knn_indices=source_knn.indices.astype(np.int32),
        source_knn_distances=source_knn.distances.astype(np.float32),
        target_knn_indices=target_knn.indices.astype(np.int32),
        target_knn_distances=target_knn.distances.astype(np.float32),
    )
    _write_csv(output / "03_unlabeled_candidate_pool_health.csv", pool_rows)
    _save_pca_npz(output / "03b_unlabeled_pca_coordinates.npz", pca_payloads)
    print(f"SEED13A2_UNLABELED_OBSERVABLES_SAVED|n={len(unlabeled_rows)}|contains_target_true_label=false", flush=True)

    # Oracle-only phase starts here.
    target_meta = phasevis._metadata_dataset(
        data_root, target, classes, splits[target]["train"], closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
    )
    label_by_parcel = {int(p): int(y) for p, y in zip(target_meta.get_parcel_indices().tolist(), target_meta.get_labels().tolist())}
    audited = _oracle_join(unlabeled_rows, label_by_parcel, classes)
    _write_csv(output / "04_oracle_sample_audit.csv", audited)
    summary_rows, coverage_rows, balance_rows, veto_rows, matched_by_class = _single_evidence_analysis(
        audited, classes, seed=args.analysis_seed, bootstrap_reps=args.bootstrap_reps,
    )
    _write_csv(output / "05_single_evidence_summary.csv", summary_rows)
    _write_csv(output / "06_precision_coverage.csv", coverage_rows)
    _write_csv(output / "07_confidence_matched_balance.csv", balance_rows)
    _write_csv(output / "08_geometry_veto_summary.csv", veto_rows)
    critical, critical_matched, critical_balance = _critical_flow_analysis(
        audited, classes, seed=args.analysis_seed + 5000, bootstrap_reps=args.bootstrap_reps,
    )
    _write_csv(output / "09_critical_confusion_flows.csv", critical)
    _write_csv(output / "09b_critical_confidence_matched.csv", critical_matched)
    _write_csv(output / "09c_critical_match_balance.csv", critical_balance)
    comp_summary, comp_grid, comp_joint = _complementarity_analysis(audited, classes, matched_by_class)
    _write_csv(output / "10_complementarity_summary.csv", comp_summary)
    _write_csv(output / "10b_fixed_quantile_4x4.csv", comp_grid)
    _write_csv(output / "10c_fixed_top_intersections.csv", comp_joint)
    pool_oracle = _pool_oracle_context(pool_rows, audited, classes)
    _write_csv(output / "11_candidate_pool_oracle_context.csv", pool_oracle)
    _write_metric_interpretation_report(
        output / "13_metric_interpretation.md", critical_matched=critical_matched,
        critical_balance=critical_balance, veto_rows=veto_rows,
        complementarity_rows=comp_summary, classes=classes,
    )

    _plot_pca_payloads(output, pca_payloads, audited, source_labels, classes)
    _plot_pair_scatter(output, audited, classes, a_name="source_class_distance_percentile", b_name=f"target_knn{PRIMARY_K}_candidate_fraction", filename="source_conformity_x_target_knn20")
    _plot_pair_scatter(output, audited, classes, a_name=f"target_knn{PRIMARY_K}_candidate_fraction", b_name="T_registered_error_source_percentile", filename="target_knn20_x_T_registered_error_percentile")
    _plot_pair_scatter(output, audited, classes, a_name=f"target_knn{PRIMARY_K}_candidate_fraction", b_name="S_registered_error_source_percentile", filename="target_knn20_x_S_registered_error_percentile")
    _plot_precision_coverage_by_evidence(output, coverage_rows, classes)

    candidate_precision = {}
    for cid, cname in enumerate(classes):
        subset = [r for r in audited if int(r["raw_pred"]) == cid]
        candidate_precision[cname] = {
            "n_candidate": len(subset),
            "oracle_precision": float(np.mean([bool(r["candidate_correct"]) for r in subset])) if subset else float("nan"),
        }
    summary = {
        "protocol": PROTOCOL,
        "source": source, "target": target, "seed": seed, "fold": fold,
        "analysis_split": "target-train",
        "delta_boot": "identity",
        "candidate_definition": "frozen raw Stage-1 classifier top-1; immutable throughout 13A-2",
        "n_target": len(audited), "n_source": len(source_ids),
        "k_values": list(map(int, K_VALUES)), "primary_visualization_k": int(PRIMARY_K),
        "feature_metric": "cosine distance on frozen Stage-1 LTAE fused_repr",
        "source_feature_calibration": "leave-one-out class centroid distance empirical CDF",
        "source_geometry_calibration": f"deterministic source-true-class-stratified {args.source_crossfit_folds}-fold cross-fit; query fold excluded from prototype",
        "optional_perturbation_uncertainty_F": False,
        "optional_F_reason": "diagnostic loader is deterministic full-parcel Identity sampling; no new stochastic view introduced",
        "candidate_precision_oracle_only": candidate_precision,
        "automatic_trainable_gate": None,
        "learned_reliability_model": False,
        "threshold_grid_search": False,
        "class_specific_rules": False,
        "target_training_updates": False,
        "teacher_student": False,
        "phase_updates": False,
        "candidate_reclassification": False,
        "automatic_13A2_verdict": None,
    }
    _json_dump(output / "12_13a2_diagnostic_summary.json", summary)
    manifest = {
        "protocol": PROTOCOL,
        "calibration_checkpoint": str(args.calibration_checkpoint.resolve()),
        "model_checkpoint": str(model_checkpoint_path),
        "experiment13a1_dir": str(args.experiment13a1_dir.resolve()),
        "experiment13a1_protocol": a1_manifest.get("protocol"),
        "source": source, "target": target, "seed": seed, "fold": fold,
        "analysis_split": "target-train",
        "classes": classes,
        "delta_boot": "identity",
        "network_parameters_frozen": True,
        "candidate_class_source": "13A-1 frozen raw classifier top-1 only",
        "candidate_reclassification_allowed": False,
        "target_true_label_read_after_files": [
            "01_unlabeled_sample_observables.csv", "02_unlabeled_dense_vectors.npz",
            "03_unlabeled_candidate_pool_health.csv", "03b_unlabeled_pca_coordinates.npz",
        ],
        "target_true_label_used_for_knn": False,
        "target_true_label_used_for_distance": False,
        "target_true_label_used_for_pca": False,
        "target_true_label_used_for_feature_selection": False,
        "target_true_label_used_for_threshold_selection": False,
        "feature_space": "frozen Stage-1 LTAE fused_repr",
        "distance": "cosine distance",
        "knn": {"exact": True, "k_values": list(map(int, K_VALUES)), "chunk_size": int(args.knn_chunk_size), "device": str(knn_device)},
        "source_feature_reference": {"method": "leave-one-out centroid distances", "self_containment": False},
        "source_geometry_reference": {
            "method": "deterministic stratified K-fold cross-fit",
            "folds": int(args.source_crossfit_folds),
            "stratification": "source true class only",
            "assignment": "within each source true class, sort parcel_id then round-robin fold",
            "query_sample_in_reference": False,
            "cached_exact_dp": True,
            "cache_path": str((cache / "source_crossfit_geometry.pt").resolve()),
        },
        "pool_health_is_sample_score": False,
        "geometry_role": "negative structural veto diagnostic; never class reassignment",
        "single_evidence_before_complementarity": True,
        "complementarity_pair_families": sorted({row[0] for row in _complementarity_pairs()}),
        "fixed_quantile_grid": "4x4 quartiles within each raw candidate class/slice",
        "fixed_joint_coverages": [0.25, 0.10],
        "learned_combination": False,
        "logistic_regression": False, "mlp": False, "random_forest": False, "xgboost": False,
        "target_label_supervised_clustering": False,
        "threshold_grid_search": False,
        "class_specific_threshold": False,
        "target_prior_assumption": False,
        "optional_perturbation_uncertainty_F": False,
        "optional_F_reason": "not activated because the A1/A2 selected loader is deterministic and changing to stochastic pixel sampling would change the frozen input protocol",
    }
    _json_dump(output / "00_manifest.json", manifest)
    _write_readme(output / "README_中文说明.md")
    print(f"SEED13A2_DONE|output={output}|n={len(audited)}|training_updates=false|trainable_gate=false", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-checkpoint", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--experiment13a1-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--knn-device", type=str, default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--knn-chunk-size", type=int, default=512)
    parser.add_argument("--source-crossfit-folds", type=int, default=5)
    parser.add_argument("--registration-workers", type=int, default=4)
    parser.add_argument("--geometry-chunk-size", type=int, default=512)
    parser.add_argument("--dp-target-chunk-size", type=int, default=512)
    parser.add_argument("--analysis-seed", type=int, default=20260814)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--pca-max-points", type=int, default=6000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.source_crossfit_folds != 5:
        parser.error("13A-2 source geometry calibration is protocol-fixed to 5-fold cross-fit")
    if min(args.knn_chunk_size, args.registration_workers, args.geometry_chunk_size, args.dp_target_chunk_size, args.pca_max_points) < 1:
        parser.error("chunk/worker/PCA sizes must be positive")
    if args.bootstrap_reps < 50:
        parser.error("bootstrap-reps must be at least 50")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
