#!/usr/bin/env python3
"""Experiment 13A: unlabeled evidence separability for initial trainable seeds.

The raw frozen Stage-1 classifier top-1 defines the candidate class.  Phase,
semantic-reference and frozen T/S-SRVF geometry are *candidate support/conflict
observables only*: none may relabel a sample.  Label-free observables are
materialized before target labels are joined by the oracle-only evaluator.
No TRAINABLE gate, threshold search, model update, EMA or long Stage-2 training
is implemented here.
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

import compare_stage2_phase_vs_timematch_shift as scalarcmp
import diagnose_sample_level_phase_validity as samplediag
import visualize_stage2_phase_alignment as phasevis
from methods.structure_da.candidate_evidence_diagnostic import (
    auprc,
    auroc,
    bootstrap_mean_ci,
    candidate_margin,
    confidence_matched_indices,
    confidence_quantile_masks,
    cosine_similarity_matrix,
    distribution_summary,
    jensen_shannon_rows,
    precision_at_coverages,
    prediction_entropy,
    prediction_margin,
    spearman_correlation,
    summarize_evidence,
    two_by_two_precision,
)
from methods.structure_da.prototype_bank import SourcePrototypeBank, support_aware_q_distance
from methods.structure_da.registration_geometry import (
    TargetGeometryCache,
    evaluate_registration_geometry,
)
from methods.structure_da.sample_phase_diagnostic import (
    TOnlyPhaseRegistration,
    TRegistrationGeometryCache,
    evaluate_shape_validation,
    solve_t_only_registrations,
)
from methods.structure_da.stage2_trainer import DeviceBatchLoader, build_stage2_registration_extractor

PROTOCOL = "13A_trainable_seed_unlabeled_evidence_separability_diagnostic"
SEMANTIC_CACHE_SCHEMA = "13A_unlabeled_semantic_observables_v1"
GEOMETRY_CACHE_SCHEMA = "13A_raw_candidate_geometry_v1"
FORMAL_BOOTSTRAP_STATES = ("identity", "timematch_scalar")


class LabelStrippedLoader:
    """Hard boundary: observable generation cannot receive target true labels."""

    def __init__(self, loader) -> None:
        self.loader = loader
        self.dataset = getattr(loader, "dataset", None)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield {key: value for key, value in batch.items() if key != "label"}


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
                fields.append(key); seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _fingerprint_candidates(sample_ids: np.ndarray, raw_pred: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(sample_ids, dtype=np.int64).tobytes())
    h.update(np.asarray(raw_pred, dtype=np.int64).tobytes())
    return h.hexdigest()


def _source_bank(checkpoint: dict) -> SourcePrototypeBank:
    return samplediag._source_bank(checkpoint)


def _read_bootstrap_state(summary_path: Path, state: str) -> dict:
    if state not in FORMAL_BOOTSTRAP_STATES:
        raise ValueError(f"13A formal bootstrap state must be one of {FORMAL_BOOTSTRAP_STATES}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    conditions = payload.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("experiment-12 summary lacks conditions")
    if state == "identity":
        return {
            "state": "identity",
            "formal_unlabeled": True,
            "scalar_shift_days": 0,
            "source": "identity",
        }
    tm = conditions.get("timematch_scalar")
    if not isinstance(tm, dict):
        raise ValueError("experiment-12 summary lacks timematch_scalar metadata")
    shift = tm.get("scalar_shift_days")
    if shift is None:
        raise ValueError("experiment-12 TimeMatch metadata lacks scalar_shift_days")
    if bool(tm.get("used_target_labels", False)):
        raise ValueError("experiment-12 TimeMatch state unexpectedly used target labels")
    return {
        "state": "timematch_scalar",
        "formal_unlabeled": True,
        "scalar_shift_days": int(shift),
        "source": "experiment12_timematch_scalar",
        "experiment12_metadata": tm,
    }


@torch.no_grad()
def _generate_unlabeled_semantic_observables(
    *, model, loader, source_semantic_references: np.ndarray,
    bootstrap: dict, time_scale_days: float, device: torch.device,
) -> dict:
    sample_ids: list[np.ndarray] = []
    raw_post: list[np.ndarray] = []
    phase_post: list[np.ndarray] = []
    raw_feat: list[np.ndarray] = []
    phase_feat: list[np.ndarray] = []
    model.eval()
    state = str(bootstrap["state"])
    shift = int(bootstrap["scalar_shift_days"])
    for raw_batch in loader:
        # The LabelStrippedLoader has already removed target true labels.
        if "label" in raw_batch:
            raise RuntimeError("label leaked into 13A unlabeled observable generation")
        batch = phasevis._move_batch(raw_batch, device)
        backbone = model.forward_backbone(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch.get("extra"), time_mask=batch.get("time_mask"), compute_decomposition=False,
        )
        raw = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"), return_geometry=False,
        )
        if state == "identity":
            phase = raw
        else:
            corrected_positions = scalarcmp._scalar_positions(backbone, shift, time_scale_days)
            with scalarcmp._timematch_time_extrapolation(model):
                phase = model.forward_from_backbone(
                    backbone, batch["positions"], batch.get("extra"),
                    temporal_positions_override=corrected_positions,
                    return_geometry=False,
                )
        sample_ids.append(batch["parcel_index"].detach().cpu().numpy().astype(np.int64))
        raw_post.append(torch.softmax(raw.logits.float(), dim=-1).detach().cpu().numpy())
        phase_post.append(torch.softmax(phase.logits.float(), dim=-1).detach().cpu().numpy())
        raw_feat.append(raw.fused_repr.detach().cpu().float().numpy())
        phase_feat.append(phase.fused_repr.detach().cpu().float().numpy())
    if not sample_ids:
        raise RuntimeError("target-train loader produced no samples")
    ids = np.concatenate(sample_ids)
    raw_p = np.concatenate(raw_post).astype(np.float64)
    phase_p = np.concatenate(phase_post).astype(np.float64)
    raw_f = np.concatenate(raw_feat).astype(np.float64)
    phase_f = np.concatenate(phase_feat).astype(np.float64)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]; raw_p = raw_p[order]; phase_p = phase_p[order]
    raw_f = raw_f[order]; phase_f = phase_f[order]
    if np.unique(ids).size != ids.size:
        raise ValueError("target-train parcel identities are not unique")

    raw_pred = raw_p.argmax(axis=1).astype(np.int64)
    phase_pred = phase_p.argmax(axis=1).astype(np.int64)
    row = np.arange(ids.size)
    raw_prob = raw_p[row, raw_pred]
    phase_candidate_prob = phase_p[row, raw_pred]
    raw_margin = prediction_margin(raw_p)
    raw_entropy = prediction_entropy(raw_p)
    phase_candidate_margin = candidate_margin(phase_p, raw_pred)
    raw_phase_js = jensen_shannon_rows(raw_p, phase_p)
    raw_sim = cosine_similarity_matrix(raw_f, source_semantic_references)
    phase_sim = cosine_similarity_matrix(phase_f, source_semantic_references)

    return {
        "sample_ids": ids,
        "raw_posterior": raw_p,
        "phase_posterior": phase_p,
        "raw_pred": raw_pred,
        "phase_pred": phase_pred,
        "raw_prob": raw_prob,
        "raw_margin": raw_margin,
        "raw_entropy": raw_entropy,
        "phase_candidate_prob": phase_candidate_prob,
        "phase_candidate_margin": phase_candidate_margin,
        "raw_phase_agree": (phase_pred == raw_pred).astype(np.int8),
        "delta_candidate_prob": phase_candidate_prob - raw_prob,
        "raw_phase_js": raw_phase_js,
        "semantic_raw_similarity_all": raw_sim,
        "semantic_phase_similarity_all": phase_sim,
        "semantic_raw_candidate_similarity": raw_sim[row, raw_pred],
        "semantic_raw_candidate_margin": candidate_margin(raw_sim, raw_pred),
        "semantic_phase_candidate_similarity": phase_sim[row, raw_pred],
        "semantic_phase_candidate_margin": candidate_margin(phase_sim, raw_pred),
    }


def _load_or_generate_semantic_cache(
    cache_path: Path, *, model, loader, source_refs: np.ndarray, bootstrap: dict,
    time_scale_days: float, device: torch.device, source: str, target: str,
    seed: int, fold: int, model_checkpoint: Path,
) -> dict:
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != SEMANTIC_CACHE_SCHEMA:
            raise ValueError("unsupported 13A semantic cache schema")
        for key, expected in (("source", source), ("target", target), ("seed", seed), ("fold", fold)):
            if str(payload.get(key)) != str(expected):
                raise ValueError(f"13A semantic cache {key} mismatch")
        if Path(str(payload.get("model_checkpoint"))).resolve() != model_checkpoint.resolve():
            raise ValueError("13A semantic cache Stage-1 checkpoint mismatch")
        if str(payload.get("bootstrap_state")) != str(bootstrap["state"]):
            raise ValueError("13A semantic cache bootstrap-state mismatch")
        if int(payload.get("scalar_shift_days", 0)) != int(bootstrap["scalar_shift_days"]):
            raise ValueError("13A semantic cache scalar-shift mismatch")
        print(f"SEED13A_SEMANTIC_CACHE_HIT|path={cache_path}", flush=True)
        return payload["observables"]
    print("SEED13A_SEMANTIC_OBSERVABLE_START|labels_visible=false", flush=True)
    obs = _generate_unlabeled_semantic_observables(
        model=model, loader=loader, source_semantic_references=source_refs,
        bootstrap=bootstrap, time_scale_days=time_scale_days, device=device,
    )
    _atomic_torch_save({
        "schema": SEMANTIC_CACHE_SCHEMA,
        "source": source, "target": target, "seed": int(seed), "fold": int(fold),
        "model_checkpoint": str(model_checkpoint.resolve()),
        "bootstrap_state": str(bootstrap["state"]),
        "scalar_shift_days": int(bootstrap["scalar_shift_days"]),
        "contains_target_true_labels": False,
        "observables": obs,
    }, cache_path)
    print(f"SEED13A_SEMANTIC_OBSERVABLE_READY|n={len(obs['sample_ids'])}|labels_visible=false", flush=True)
    return obs


def _s_identity_distance(target_cache: TargetGeometryCache, source_bank: SourcePrototypeBank, sample_index: int, class_id: int) -> tuple[float, float]:
    q = target_cache.structure_srvf_shape[sample_index].detach().cpu().float()
    support = target_cache.structure_support_shape[sample_index].detach().cpu().float()
    proto = source_bank.shape_srvf[class_id].detach().cpu().float()
    proto_support = source_bank.shape_support[class_id].detach().cpu().float()
    grid = target_cache.shape_grid.detach().cpu().float()
    weights = torch.ones_like(grid)
    if weights.numel() > 1:
        weights[[0, -1]] *= 0.5
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    out = support_aware_q_distance(
        q.unsqueeze(0), proto.unsqueeze(0), support.unsqueeze(0), proto_support.unsqueeze(0), weights
    )
    return float(out.distance[0, 0].item()), float(out.common_support[0, 0].item())


def _geometry_row(reg: TOnlyPhaseRegistration, target_cache: TargetGeometryCache, source_bank: SourcePrototypeBank) -> dict:
    s_id, s_id_support = _s_identity_distance(target_cache, source_bank, reg.sample_index, reg.class_id)
    shape = evaluate_shape_validation(reg, target_cache=target_cache, source_bank=source_bank)
    s_reg = float(shape.raw_shape_distance) if shape.raw_shape_distance is not None else float("nan")
    t_id = float(reg.t_identity_error) if reg.t_identity_error is not None else float("nan")
    t_reg = float(reg.t_registered_error) if reg.t_registered_error is not None else float("nan")
    return {
        "sample_id": int(reg.sample_id),
        "raw_pred": int(reg.class_id),
        "gamma": None if reg.gamma is None else reg.gamma.detach().cpu().double(),
        "registration_numerically_valid": bool(reg.numerically_valid),
        "solver_error": reg.solver_error,
        "target_trend_valid": bool(reg.target_trend_valid),
        "T_pre_common_support": float(reg.pre_common_support_t),
        "T_common_support": float(reg.common_support_t) if reg.common_support_t is not None else float("nan"),
        "T_identity_error": t_id,
        "T_registered_error": t_reg,
        "T_gain": float(t_id - t_reg) if math.isfinite(t_id) and math.isfinite(t_reg) else float("nan"),
        "T_gain_ratio": float(reg.t_gain_ratio) if reg.t_gain_ratio is not None else float("nan"),
        "T_registration_improvement": float(t_id - t_reg) if math.isfinite(t_id) and math.isfinite(t_reg) else float("nan"),
        "S_identity_error": s_id,
        "S_identity_common_support": s_id_support,
        "S_registered_error": s_reg,
        "S_registered_common_support": float(shape.common_support_shape) if shape.common_support_shape is not None else float("nan"),
        "S_gain": float(s_id - s_reg) if math.isfinite(s_reg) else float("nan"),
        "S_gain_ratio": float(s_reg / (s_id + 1e-8)) if math.isfinite(s_reg) else float("nan"),
        "S_registration_improvement": float(s_id - s_reg) if math.isfinite(s_reg) else float("nan"),
        "S_distance_percentile": float(shape.q_distance_percentile) if shape.q_distance_percentile is not None else float("nan"),
        "phase_magnitude": float(reg.phase_deviation) if reg.phase_deviation is not None else float("nan"),
        "gamma_endpoint_error": float(reg.gamma_endpoint_error) if reg.gamma_endpoint_error is not None else float("nan"),
        "gamma_min_increment": float(reg.gamma_min_increment) if reg.gamma_min_increment is not None else float("nan"),
        "gamma_max_local_speed": float(reg.gamma_max_local_speed) if reg.gamma_max_local_speed is not None else float("nan"),
        "gamma_roughness": float(reg.gamma_roughness) if reg.gamma_roughness is not None else float("nan"),
        "phase_deviation": float(reg.phase_deviation) if reg.phase_deviation is not None else float("nan"),
    }


def _flush_geometry_chunk(
    *, chunk: list[dict], source_reg_bank, source_bank: SourcePrototypeBank,
    scan_config, workers: int, dp_chunk_size: int,
) -> list[dict]:
    if not chunk:
        return []
    reg_grid = chunk[0]["registration_grid"].detach().cpu().double()
    shape_grid = chunk[0]["shape_grid"].detach().cpu().double()
    t_cache = TRegistrationGeometryCache(
        sample_ids=torch.tensor([int(row["sample_id"]) for row in chunk], dtype=torch.long),
        trend_srvf_reg=torch.stack([row["trend_srvf"] for row in chunk]),
        trend_support_reg=torch.stack([row["trend_support"] for row in chunk]),
        trend_valid=torch.stack([row["trend_valid"] for row in chunk]),
        registration_grid=reg_grid,
    )
    target_cache = TargetGeometryCache(
        sample_ids=t_cache.sample_ids,
        trend_srvf_reg=t_cache.trend_srvf_reg,
        trend_support_reg=t_cache.trend_support_reg,
        trend_valid=t_cache.trend_valid,
        structure_srvf_shape=torch.stack([row["structure_srvf"] for row in chunk]),
        structure_support_shape=torch.stack([row["structure_support"] for row in chunk]),
        structure_valid=torch.stack([row["structure_valid"] for row in chunk]),
        registration_grid=reg_grid,
        shape_grid=shape_grid,
    )
    assignments = [(idx, int(row["raw_pred"])) for idx, row in enumerate(chunk)]
    regs = solve_t_only_registrations(
        source_reg_bank, t_cache, assignments, scan_config,
        workers=workers, progress_label="SEED13A_RAW_CANDIDATE_DP",
        max_target_samples_per_pool=max(1, min(int(dp_chunk_size), len(chunk))),
    )
    return [_geometry_row(reg, target_cache, source_bank) for reg in regs]


def _load_partial_geometry_cache(
    path: Path, *, source: str, target: str, seed: int, fold: int,
    model_checkpoint: Path, candidate_fingerprint: str,
) -> dict[int, dict]:
    if not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != GEOMETRY_CACHE_SCHEMA:
        raise ValueError("unsupported 13A candidate-geometry cache schema")
    for key, expected in (("source", source), ("target", target), ("seed", seed), ("fold", fold)):
        if str(payload.get(key)) != str(expected):
            raise ValueError(f"13A geometry cache {key} mismatch")
    if Path(str(payload.get("model_checkpoint"))).resolve() != model_checkpoint.resolve():
        raise ValueError("13A geometry cache Stage-1 checkpoint mismatch")
    if str(payload.get("candidate_fingerprint")) != candidate_fingerprint:
        raise ValueError("13A geometry cache raw-candidate assignment mismatch")
    rows = {}
    for row in payload.get("records", ()):
        item = dict(row)
        gamma = item.get("gamma")
        if isinstance(gamma, Tensor):
            item["gamma"] = gamma.detach().cpu().double()
        sid = int(item["sample_id"])
        if sid in rows:
            raise ValueError("duplicate parcel in 13A geometry cache")
        rows[sid] = item
    print(f"SEED13A_GEOMETRY_CACHE_RESUME|completed={len(rows)}", flush=True)
    return rows


def _save_partial_geometry_cache(
    path: Path, records: Mapping[int, dict], *, source: str, target: str, seed: int, fold: int,
    model_checkpoint: Path, candidate_fingerprint: str,
) -> None:
    ordered = [records[key] for key in sorted(records)]
    _atomic_torch_save({
        "schema": GEOMETRY_CACHE_SCHEMA,
        "source": source, "target": target, "seed": int(seed), "fold": int(fold),
        "model_checkpoint": str(model_checkpoint.resolve()),
        "candidate_fingerprint": candidate_fingerprint,
        "contains_target_true_labels": False,
        "candidate_source": "frozen raw classifier top-1 only",
        "geometry_relabels_candidate": False,
        "records": ordered,
    }, path)


@torch.no_grad()
def _load_or_generate_candidate_geometry(
    cache_path: Path, *, model, loader, semantic_obs: dict, source_bank: SourcePrototypeBank,
    source_reg_bank, reg_extractor, scan_config, device: torch.device,
    source: str, target: str, seed: int, fold: int, model_checkpoint: Path,
    workers: int, geometry_chunk_size: int, dp_chunk_size: int,
) -> dict[int, dict]:
    ids = np.asarray(semantic_obs["sample_ids"], dtype=np.int64)
    pred = np.asarray(semantic_obs["raw_pred"], dtype=np.int64)
    candidate_by_id = {int(sid): int(c) for sid, c in zip(ids.tolist(), pred.tolist())}
    fingerprint = _fingerprint_candidates(ids, pred)
    records = _load_partial_geometry_cache(
        cache_path, source=source, target=target, seed=seed, fold=fold,
        model_checkpoint=model_checkpoint, candidate_fingerprint=fingerprint,
    )
    expected = set(candidate_by_id)
    if set(records) == expected:
        print(f"SEED13A_GEOMETRY_CACHE_HIT|path={cache_path}|count={len(records)}", flush=True)
        return records

    chunk: list[dict] = []
    model.eval()
    for raw_batch in loader:
        if "label" in raw_batch:
            raise RuntimeError("label leaked into 13A geometry observable generation")
        parcels_cpu = raw_batch["parcel_index"].detach().cpu().long().tolist()
        unresolved_indices = [idx for idx, sid in enumerate(parcels_cpu) if int(sid) not in records]
        if not unresolved_indices:
            continue
        batch = phasevis._move_batch(raw_batch, device)
        output = model(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch.get("extra"), return_geometry=True,
        )
        if output.geometry is None:
            raise RuntimeError("13A geometry requires functional geometry")
        reg = evaluate_registration_geometry(output.trend, output.positions, output.mask, reg_extractor)
        for idx in unresolved_indices:
            sid = int(parcels_cpu[idx])
            if sid not in candidate_by_id:
                raise ValueError("geometry loader parcel not present in semantic observable cache")
            chunk.append({
                "sample_id": sid,
                "raw_pred": candidate_by_id[sid],
                "trend_srvf": reg.trend_srvf[idx].detach().cpu(),
                "trend_support": reg.trend_support[idx].detach().cpu(),
                "trend_valid": reg.trend_valid[idx].detach().cpu(),
                "structure_srvf": output.geometry.structure_srvf[idx].detach().cpu(),
                "structure_support": output.geometry.structure_support[idx].detach().cpu(),
                "structure_valid": output.geometry.structure_valid[idx].detach().cpu(),
                "registration_grid": reg.registration_grid.detach().cpu(),
                "shape_grid": output.geometry.canonical_grid.detach().cpu(),
            })
        if len(chunk) >= int(geometry_chunk_size):
            solved = _flush_geometry_chunk(
                chunk=chunk, source_reg_bank=source_reg_bank, source_bank=source_bank,
                scan_config=scan_config, workers=workers, dp_chunk_size=dp_chunk_size,
            )
            records.update({int(row["sample_id"]): row for row in solved})
            _save_partial_geometry_cache(
                cache_path, records, source=source, target=target, seed=seed, fold=fold,
                model_checkpoint=model_checkpoint, candidate_fingerprint=fingerprint,
            )
            print(f"SEED13A_GEOMETRY_CACHE_PROGRESS|completed={len(records)}/{len(expected)}", flush=True)
            chunk = []
    if chunk:
        solved = _flush_geometry_chunk(
            chunk=chunk, source_reg_bank=source_reg_bank, source_bank=source_bank,
            scan_config=scan_config, workers=workers, dp_chunk_size=dp_chunk_size,
        )
        records.update({int(row["sample_id"]): row for row in solved})
        _save_partial_geometry_cache(
            cache_path, records, source=source, target=target, seed=seed, fold=fold,
            model_checkpoint=model_checkpoint, candidate_fingerprint=fingerprint,
        )
    missing = expected - set(records)
    if missing:
        raise RuntimeError(f"13A candidate geometry cache incomplete: missing {len(missing)} target-train parcels")
    print(f"SEED13A_GEOMETRY_CACHE_READY|count={len(records)}|labels_visible=false", flush=True)
    return records


def _build_unlabeled_rows(obs: dict, geometry: Mapping[int, dict]) -> tuple[list[dict], np.ndarray]:
    ids = np.asarray(obs["sample_ids"], dtype=np.int64)
    rows: list[dict] = []
    gammas: list[np.ndarray] = []
    for idx, sid in enumerate(ids.tolist()):
        geo = geometry[int(sid)]
        if int(geo["raw_pred"]) != int(obs["raw_pred"][idx]):
            raise ValueError("13A geometry candidate differs from frozen raw top-1")
        row = {
            "sample_id": int(sid), "target_index": int(idx),
            "raw_pred": int(obs["raw_pred"][idx]),
            "raw_prob": float(obs["raw_prob"][idx]),
            "raw_margin": float(obs["raw_margin"][idx]),
            "raw_entropy": float(obs["raw_entropy"][idx]),
            "phase_pred": int(obs["phase_pred"][idx]),
            "phase_candidate_prob": float(obs["phase_candidate_prob"][idx]),
            "phase_candidate_margin": float(obs["phase_candidate_margin"][idx]),
            "raw_phase_agree": int(obs["raw_phase_agree"][idx]),
            "delta_candidate_prob": float(obs["delta_candidate_prob"][idx]),
            "raw_phase_js": float(obs["raw_phase_js"][idx]),
            "semantic_raw_candidate_similarity": float(obs["semantic_raw_candidate_similarity"][idx]),
            "semantic_raw_candidate_margin": float(obs["semantic_raw_candidate_margin"][idx]),
            "semantic_phase_candidate_similarity": float(obs["semantic_phase_candidate_similarity"][idx]),
            "semantic_phase_candidate_margin": float(obs["semantic_phase_candidate_margin"][idx]),
        }
        row.update({key: value for key, value in geo.items() if key not in {"sample_id", "raw_pred", "gamma"}})
        rows.append(row)
        gamma = geo.get("gamma")
        if isinstance(gamma, Tensor):
            gammas.append(gamma.detach().cpu().double().numpy())
        else:
            # K is known from any successful gamma later; temporarily empty.
            gammas.append(np.asarray([], dtype=np.float64))
    k = next((arr.size for arr in gammas if arr.size), 0)
    if k == 0:
        gamma_matrix = np.empty((len(gammas), 0), dtype=np.float64)
    else:
        gamma_matrix = np.full((len(gammas), k), np.nan, dtype=np.float64)
        for idx, arr in enumerate(gammas):
            if arr.size:
                if arr.size != k:
                    raise ValueError("inconsistent candidate gamma grid length")
                gamma_matrix[idx] = arr
    return rows, gamma_matrix


EVIDENCE_SPECS = (
    ("raw_prob", "A_classifier", True),
    ("raw_margin", "A_classifier", True),
    ("raw_entropy", "A_classifier", False),
    ("raw_phase_agree", "B_view", True),
    ("phase_candidate_prob", "B_view", True),
    ("phase_candidate_margin", "B_view", True),
    ("delta_candidate_prob", "B_view", True),
    ("raw_phase_js", "B_view", False),
    ("semantic_raw_candidate_similarity", "C_semantic", True),
    ("semantic_raw_candidate_margin", "C_semantic", True),
    ("semantic_phase_candidate_similarity", "C_semantic", True),
    ("semantic_phase_candidate_margin", "C_semantic", True),
    ("T_identity_error", "D_geometry", False),
    ("T_registered_error", "D_geometry", False),
    ("T_gain_ratio", "D_geometry", False),
    ("T_registration_improvement", "D_geometry", True),
    ("T_pre_common_support", "D_geometry", True),
    ("T_common_support", "D_geometry", True),
    ("S_identity_error", "D_geometry", False),
    ("S_registered_error", "D_geometry", False),
    ("S_gain_ratio", "D_geometry", False),
    ("S_registration_improvement", "D_geometry", True),
    ("S_registered_common_support", "D_geometry", True),
    ("phase_deviation", "D_geometry", False),
    ("gamma_roughness", "D_geometry", False),
    ("gamma_max_local_speed", "D_geometry", False),
)


def _oracle_join(unlabeled_rows: Sequence[dict], label_by_parcel: Mapping[int, int], classes: Sequence[str]) -> list[dict]:
    audited = []
    for row in unlabeled_rows:
        sid = int(row["sample_id"])
        if sid not in label_by_parcel:
            raise ValueError("oracle evaluator could not find target true label for parcel")
        truth = int(label_by_parcel[sid]); pred = int(row["raw_pred"])
        audited.append({
            **row,
            "true_label": truth,
            "true_class_name": str(classes[truth]),
            "candidate_correct": bool(truth == pred),
            "candidate_class_name": str(classes[pred]),
            "confusion_flow": f"{classes[truth]} -> {classes[pred]}",
        })
    return audited


def _array(rows: Sequence[dict], key: str) -> np.ndarray:
    return np.asarray([row.get(key, float("nan")) for row in rows], dtype=np.float64)


def _oracle_analysis(rows: Sequence[dict], classes: Sequence[str], *, seed: int, bootstrap_reps: int):
    correctness = np.asarray([bool(row["candidate_correct"]) for row in rows], dtype=bool)
    pred = np.asarray([int(row["raw_pred"]) for row in rows], dtype=np.int64)
    raw_prob = _array(rows, "raw_prob"); raw_margin = _array(rows, "raw_margin")

    matched_global = np.zeros(len(rows), dtype=bool)
    balance_rows: list[dict] = []
    matched_by_class: dict[int, np.ndarray] = {}
    for class_id, class_name in enumerate(classes):
        pool = pred == class_id
        local_match, balance = confidence_matched_indices(
            correctness[pool], raw_prob[pool], raw_margin[pool], seed=seed + class_id * 97,
        )
        global_indices = np.flatnonzero(pool)
        mask = np.zeros(len(rows), dtype=bool); mask[global_indices[local_match]] = True
        matched_by_class[class_id] = mask; matched_global |= mask
        balance_rows.append({
            "candidate_class": int(class_id), "candidate_class_name": str(class_name), **balance
        })

    summary_rows: list[dict] = []
    coverage_rows: list[dict] = []
    veto_rows: list[dict] = []
    for class_id, class_name in enumerate(classes):
        pool = pred == class_id
        local_quantiles = confidence_quantile_masks(raw_prob[pool])
        global_indices = np.flatnonzero(pool)
        slices: dict[str, np.ndarray] = {}
        for name, local in local_quantiles.items():
            mask = np.zeros(len(rows), dtype=bool); mask[global_indices[local]] = True; slices[name] = mask
        slices["confidence_matched"] = matched_by_class[class_id]
        for evidence_name, group, higher in EVIDENCE_SPECS:
            values = _array(rows, evidence_name)
            for slice_name, mask in slices.items():
                # Bootstrap CIs are required for the principal correct/wrong
                # comparison and confidence-matched incremental audit.  The
                # auxiliary confidence quantile slices retain full descriptive
                # statistics/AUC without multiplying bootstrap cost.
                slice_bootstrap_reps = bootstrap_reps if slice_name in {"all", "confidence_matched"} else 0
                stats = summarize_evidence(
                    values[mask], correctness[mask], higher_is_reliable=higher,
                    seed=seed + class_id * 1009 + sum(map(ord, evidence_name + slice_name)),
                    bootstrap_reps=slice_bootstrap_reps,
                )
                summary_rows.append({
                    "candidate_class": int(class_id), "candidate_class_name": str(class_name),
                    "evidence_group": group, "evidence": evidence_name,
                    "analysis_slice": slice_name, **stats,
                })
            # Precision-coverage is always defined on the full candidate pool.
            values_pool = values[pool]
            scores = values_pool if higher else -values_pool
            for item in precision_at_coverages(correctness[pool], scores):
                coverage_rows.append({
                    "candidate_class": int(class_id), "candidate_class_name": str(class_name),
                    "evidence_group": group, "evidence": evidence_name,
                    "base_precision": float(np.mean(correctness[pool])) if np.any(pool) else float("nan"),
                    **item,
                })
            if group == "D_geometry":
                for slice_name in ("all", "top25", "top10", "confidence_matched"):
                    hit = next(row for row in summary_rows if row["candidate_class"] == class_id and row["evidence"] == evidence_name and row["analysis_slice"] == slice_name)
                    veto_rows.append({
                        "candidate_class": int(class_id), "candidate_class_name": str(class_name),
                        "evidence": evidence_name, "analysis_slice": slice_name,
                        "n": hit["n"], "n_correct": hit["n_correct"], "n_wrong": hit["n_wrong"],
                        "wrong_reject_at_correct_retention_95": hit["wrong_reject_at_correct_retention_95"],
                        "achieved_correct_retention_95": hit["achieved_correct_retention_95"],
                        "wrong_reject_at_correct_retention_90": hit["wrong_reject_at_correct_retention_90"],
                        "achieved_correct_retention_90": hit["achieved_correct_retention_90"],
                    })

    overall_rows: list[dict] = []
    for evidence_name, group, higher in EVIDENCE_SPECS:
        values = _array(rows, evidence_name)
        pooled = summarize_evidence(values, correctness, higher_is_reliable=higher, seed=seed + sum(map(ord, evidence_name)), bootstrap_reps=bootstrap_reps)
        class_all = [row for row in summary_rows if row["evidence"] == evidence_name and row["analysis_slice"] == "all"]
        aucs = [float(row["auroc"]) for row in class_all if math.isfinite(float(row["auroc"]))]
        aps = [float(row["auprc"]) for row in class_all if math.isfinite(float(row["auprc"]))]
        matched = summarize_evidence(values[matched_global], correctness[matched_global], higher_is_reliable=higher, seed=seed + 17 + sum(map(ord, evidence_name)), bootstrap_reps=bootstrap_reps)
        overall_rows.append({
            "evidence_group": group, "evidence": evidence_name,
            "pooled_auroc": pooled["auroc"], "pooled_auprc": pooled["auprc"],
            "macro_per_class_auroc": float(np.mean(aucs)) if aucs else float("nan"),
            "macro_per_class_auprc": float(np.mean(aps)) if aps else float("nan"),
            "confidence_matched_pooled_auroc": matched["auroc"],
            "confidence_matched_pooled_auprc": matched["auprc"],
            "matched_n": matched["n"],
        })
    return summary_rows, coverage_rows, balance_rows, veto_rows, overall_rows, matched_global


def _critical_flow_rows(rows: Sequence[dict], classes: Sequence[str]) -> list[dict]:
    flow_specs = [
        ("spring_barley_correct", "spring_barley", "spring_barley"),
        ("spring_barley_to_spring_oat", "spring_barley", "spring_oat"),
        ("spring_oat_correct", "spring_oat", "spring_oat"),
        ("winter_triticale_correct", "winter_triticale", "winter_triticale"),
        ("winter_rye_to_winter_triticale", "winter_rye", "winter_triticale"),
        ("winter_wheat_to_winter_triticale", "winter_wheat", "winter_triticale"),
        ("winter_rye_correct", "winter_rye", "winter_rye"),
        ("winter_wheat_correct", "winter_wheat", "winter_wheat"),
    ]
    name_to_id = {str(name): idx for idx, name in enumerate(classes)}
    out = []
    for flow_name, true_name, pred_name in flow_specs:
        if true_name not in name_to_id or pred_name not in name_to_id:
            continue
        subset = [row for row in rows if int(row["true_label"]) == name_to_id[true_name] and int(row["raw_pred"]) == name_to_id[pred_name]]
        for evidence_name, group, higher in EVIDENCE_SPECS:
            stats = distribution_summary(_array(subset, evidence_name))
            out.append({
                "flow": flow_name, "true_class": true_name, "candidate_class": pred_name,
                "evidence_group": group, "evidence": evidence_name, "higher_is_reliable": higher,
                "n": len(subset), **stats,
            })
    return out



def _critical_matched_rows(rows: Sequence[dict], classes: Sequence[str], *, seed: int, bootstrap_reps: int):
    name_to_id = {str(name): idx for idx, name in enumerate(classes)}
    specs = [
        ("oat_correct_vs_barley_absorbed", "spring_oat", ["spring_barley"]),
        ("triticale_correct_vs_rye_wheat_absorbed", "winter_triticale", ["winter_rye", "winter_wheat"]),
    ]
    detail_rows: list[dict] = []
    balance_rows: list[dict] = []
    for comp_index, (comparison, candidate_name, wrong_true_names) in enumerate(specs):
        if candidate_name not in name_to_id or any(name not in name_to_id for name in wrong_true_names):
            continue
        cid = name_to_id[candidate_name]
        selected = [
            row for row in rows
            if int(row["raw_pred"]) == cid
            and (int(row["true_label"]) == cid or int(row["true_label"]) in {name_to_id[name] for name in wrong_true_names})
        ]
        if not selected:
            continue
        is_correct_flow = np.asarray([int(row["true_label"]) == cid for row in selected], dtype=bool)
        prob = _array(selected, "raw_prob"); margin = _array(selected, "raw_margin")
        matched, balance = confidence_matched_indices(
            is_correct_flow, prob, margin, seed=seed + comp_index * 131, bins=10,
        )
        balance_rows.append({
            "comparison": comparison, "candidate_class": candidate_name,
            "wrong_true_classes": ",".join(wrong_true_names), **balance,
        })
        for evidence_name, group, higher in EVIDENCE_SPECS:
            values = _array(selected, evidence_name)
            stats = summarize_evidence(
                values[matched], is_correct_flow[matched], higher_is_reliable=higher,
                seed=seed + comp_index * 2003 + sum(map(ord, evidence_name)),
                bootstrap_reps=bootstrap_reps,
            )
            detail_rows.append({
                "comparison": comparison, "candidate_class": candidate_name,
                "wrong_true_classes": ",".join(wrong_true_names),
                "evidence_group": group, "evidence": evidence_name, **stats,
            })
    return detail_rows, balance_rows

def _complementarity_rows(rows: Sequence[dict], classes: Sequence[str], matched_global: np.ndarray) -> list[dict]:
    pred = np.asarray([int(row["raw_pred"]) for row in rows], dtype=np.int64)
    correct = np.asarray([bool(row["candidate_correct"]) for row in rows], dtype=bool)
    sem = _array(rows, "semantic_phase_candidate_margin")
    view = -_array(rows, "raw_phase_js")  # higher = stronger raw/Phase agreement
    geo = -_array(rows, "S_gain_ratio")    # higher = stronger registered S improvement
    pairs = (("semantic_x_view", sem, view), ("semantic_x_geometry", sem, geo))
    output = []
    scopes = [(None, "ALL")] + [(idx, str(name)) for idx, name in enumerate(classes)]
    for class_id, class_name in scopes:
        base = np.ones(len(rows), dtype=bool) if class_id is None else pred == int(class_id)
        for analysis_slice, slice_mask in (("all", base), ("confidence_matched", base & matched_global)):
            for pair_name, a, b in pairs:
                mask = slice_mask & np.isfinite(a) & np.isfinite(b)
                rho = spearman_correlation(a[mask], b[mask])
                cells = two_by_two_precision(a[mask], b[mask], correct[mask])
                for cell in cells:
                    output.append({
                        "candidate_class": "ALL" if class_id is None else int(class_id),
                        "candidate_class_name": class_name,
                        "analysis_slice": analysis_slice,
                        "pair": pair_name,
                        "spearman": rho,
                        **cell,
                    })
    return output


def _plot_distribution_panels(path: Path, rows: Sequence[dict], classes: Sequence[str], evidence: str) -> None:
    pred = np.asarray([int(row["raw_pred"]) for row in rows], dtype=np.int64)
    correct = np.asarray([bool(row["candidate_correct"]) for row in rows], dtype=bool)
    values = _array(rows, evidence)
    fig, ax = plt.subplots(figsize=(14, 5))
    positions = []; data = []; labels = []
    for cid, name in enumerate(classes):
        for offset, flag in ((-0.18, True), (0.18, False)):
            arr = values[(pred == cid) & (correct == flag) & np.isfinite(values)]
            if arr.size:
                positions.append(cid + offset); data.append(arr); labels.append((name, flag))
    if data:
        bp = ax.boxplot(data, positions=positions, widths=0.3, showfliers=False, patch_artist=False)
        del bp
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel(evidence); ax.set_title(f"13A candidate-class correct/wrong distribution: {evidence}\nleft=correct, right=wrong")
    ax.grid(axis="y", alpha=0.2); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_precision_coverage(path: Path, coverage_rows: Sequence[dict], class_id: int, class_name: str) -> None:
    preferred = ["raw_prob", "raw_phase_js", "semantic_phase_candidate_margin", "T_registration_improvement", "S_registration_improvement"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for evidence in preferred:
        subset = [row for row in coverage_rows if int(row["candidate_class"]) == class_id and row["evidence"] == evidence]
        if not subset:
            continue
        subset = sorted(subset, key=lambda row: float(row["coverage"]))
        ax.plot([100 * float(row["coverage"]) for row in subset], [float(row["precision"]) for row in subset], marker="o", label=evidence)
    base = next((float(row["base_precision"]) for row in coverage_rows if int(row["candidate_class"]) == class_id), float("nan"))
    if math.isfinite(base):
        ax.axhline(base, linestyle="--", linewidth=1, label="raw candidate precision")
    ax.set_xlabel("candidate coverage (%)"); ax.set_ylabel("oracle precision")
    ax.set_title(f"Precision–Coverage: candidate {class_name}")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.2); ax.legend(fontsize=7); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_confidence_scatter(path: Path, rows: Sequence[dict], evidence: str, *, seed: int, limit: int = 6000) -> None:
    x = _array(rows, "raw_prob"); y = _array(rows, evidence)
    correct = np.asarray([bool(row["candidate_correct"]) for row in rows])
    keep = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if keep.size > limit:
        keep = np.random.default_rng(seed).choice(keep, size=limit, replace=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    if keep.size:
        ax.scatter(x[keep], y[keep], c=correct[keep].astype(int), s=8, alpha=0.35)
    ax.set_xlabel("raw candidate probability"); ax.set_ylabel(evidence)
    ax.set_title(f"Raw confidence vs {evidence}; color=oracle candidate correctness")
    ax.grid(alpha=0.2); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _write_readme(path: Path, bootstrap_state: str) -> None:
    text = f"""# 实验 13A：可训练种子的无标签证据可分性诊断

## 协议边界

本实验是 Stage 2 / Part I-A 的冻结诊断。raw top-1 classifier prediction 只定义 `candidate class`；它不等于 trainable pseudo-label。Phase view、source LTAE semantic reference 和 T/S-SRVF geometry 都只能评价同一个 raw candidate 的支持、冲突和不确定性，禁止重新 argmax 产生新类别。

当前固定 bootstrap state：`{bootstrap_state}`。Observable 生成阶段通过 `LabelStrippedLoader` 移除 target true label，并先写出 `01/02`；随后独立 oracle evaluator 才从 metadata join true label。没有 optimizer、EMA、Teacher/Student、PSE/LTAE/classifier adaptation、shared Phase refresh、Stable Label 或 TRAINABLE gate。

## 文件

- `00_manifest.json`：冻结对象、bootstrap state、label-use boundary、registration cache、禁止机制。用于审计协议，不能给出最终 gate。
- `01_unlabeled_sample_observables.csv`：逐 target-train sample 的 label-free scalar observables。包括 raw posterior 标量、raw/Phase 双视图、source semantic-reference support、candidate-specific T/S/Phase 原始几何量。**不含 true label/candidate_correct/confusion flow。**
- `02_unlabeled_dense_vectors.npz`：与 01 同顺序的完整 raw/Phase posterior、全部 source-class semantic cosine similarity、candidate gamma。用于复核标量计算；不含 target true label。
- `03_oracle_sample_audit.csv`：在 01 完全落盘以后才 join `true_label`、`candidate_correct` 和 `confusion_flow`。只允许离线分析，不能反馈 observable 或 candidate 构造。
- `04_per_candidate_evidence_summary.csv`：每个 raw candidate class × 每项 evidence × `all/q0_25/q25_50/q50_75/q75_90/q90_100/top25/top10/confidence_matched` 的 correct/wrong distribution、bootstrap CI、AUROC/AUPRC、base precision、geometry-veto retention 指标。
- `05_precision_coverage.csv`：每类每 evidence 的 Precision@5/10/20/30/50/100% candidate coverage 和 selected N。低 coverage 的高 precision 只说明 seed 潜力，不是正式 threshold。
- `06_confidence_matched_balance.csv`：同一 candidate class 内用 raw probability × raw margin 2-D quantile cells 做 coarsened exact matching 的样本数与匹配前后 SMD。先检查 balance，再解释 matched evidence。
- `07_geometry_veto_summary.csv`：geometry evidence 在 CorrectRetention=95%/90% 时可拒绝多少 wrong candidate。geometry 被解释为 negative veto potential，而不是 positive label confirmation。
- `08_critical_confusion_flows.csv`：spring barley/oat 和 winter rye/wheat/triticale 指定错误流的逐 evidence distribution。
- `08b_critical_confusion_flow_matched.csv`：专门对 `true oat→oat vs barley→oat`、`true triticale→triticale vs rye/wheat→triticale` 做同 candidate-class 的 raw probability × margin matching 后，再报告每项 evidence 的 AUROC/AUPRC/分布与 bootstrap CI。
- `08c_critical_confusion_flow_match_balance.csv`：上述两个关键错误流匹配前后 raw confidence/margin balance；必须先检查 balance 才解释 08b。
- `09_evidence_complementarity.csv`：预先固定的 `semantic_phase_margin × raw/Phase agreement(-JS)` 与 `semantic_phase_margin × S registration reliability(-S gain ratio)` 的 Spearman 与 2×2 median-cell precision，只看重复/互补，不学习权重。
- `10_overall_evidence_summary.csv`：pooled 与 macro-per-candidate AUROC/AUPRC、confidence-matched pooled 结果。只能补充 04，不能覆盖逐类结论。
- `11_13a_diagnostic_summary.json`：关键错误池规模、raw candidate precision、matching 概况、禁止机制和 `automatic_trainable_gate=null` / `automatic_13a_verdict=null`。
- `plots/evidence_distributions/`：若干主要 evidence 的 candidate-class correct/wrong boxplot。
- `plots/precision_coverage/`：每个 raw candidate class 一张 precision–coverage 图。
- `plots/confidence_vs_evidence/`：raw confidence 与 semantic/view/geometry evidence 的 oracle-colored scatter；颜色只用于离线解释。
- `cache/unlabeled_semantic_observables.pt`：label-free frozen inference cache。
- `cache/raw_candidate_geometry.pt`：按 frozen raw top-1 candidate 求得的 candidate registration/geometry cache。其 class assignment 不来自 target label，可断点续跑。
- `README_中文说明.md`：本说明。

## Evidence 职责

- A / classifier posterior：基础 certainty，不等价 correctness。
- B / raw–Phase view：固定 bootstrap temporal state 下的 temporal semantic agreement/conflict；Phase view 不改 candidate。
- C / source LTAE semantic reference：对 classifier candidate 的 positive semantic support；不得用 cosine argmax 重分类。
- D / frozen T/S-SRVF geometry：negative structural veto potential；旧 legality flag 不作为可靠性结论。
- E / stochastic input consistency：本实现不临时新增 augmentation，因此默认不运行。

## 能证明什么

可以判断每项正式无标签 observable 在 raw candidate pool 内是否能区分 correct/wrong，降低 coverage 是否提升 seed precision，高-confidence wrong pool 是否仍可识别，以及 confidence-matched 后是否还有 posterior 之外的增量信息。

## 不能证明什么

本实验不定义 TRAINABLE gate、不搜索 threshold、不训练联合 classifier/Logistic/MLP/SVM、不做 class-specific rule table、不更新任何模型或 Phase，也不能因为某项 oracle AUROC 高就把它直接写入正式 Stage 2。
"""
    path.write_text(text, encoding="utf-8")


def run(args) -> dict:
    calibration = torch.load(args.calibration_checkpoint.resolve(), map_location="cpu", weights_only=False)
    runtime = calibration.get("runtime_config")
    if not isinstance(runtime, dict):
        raise ValueError("calibration checkpoint is missing runtime_config")
    classes = [str(v) for v in runtime["classes"]]
    source = str(runtime["source"]); target = str(runtime["target"]); seed = int(runtime["seed"]); fold = int(args.fold)
    if len(classes) != 10:
        raise ValueError("experiment 13A expects 10 closed-set classes")
    data_root = str(args.data_root or runtime["data_root"])
    closed_set = bool(runtime.get("closed_set", True)); combine = bool(runtime.get("combine_spring_and_winter", False))
    time_mode = str(runtime.get("time_coordinate_mode", "canonical_day_of_year"))
    val_ratio = float(runtime.get("val_ratio", 0.1)); test_ratio = float(runtime.get("test_ratio", 0.2))
    time_scale_days = float(runtime.get("time_scale", 365.0))

    bootstrap = _read_bootstrap_state(args.experiment12_summary.resolve(), args.bootstrap_state)
    device = torch.device(args.device)
    model_checkpoint_path = args.model_checkpoint.resolve()
    model_checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    model = phasevis._build_model(runtime, calibration, device, model_checkpoint=model_checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    source_all = phasevis._eligible_parcels(data_root, source, classes, closed_set=closed_set, combine_spring_and_winter=combine, time_coordinate_mode=time_mode)
    target_all = phasevis._eligible_parcels(data_root, target, classes, closed_set=closed_set, combine_spring_and_winter=combine, time_coordinate_mode=time_mode)
    splits = phasevis._reconstruct_fold_splits(source_all, target_all, source=source, target=target, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio, fold=fold)
    target_train_parcels = np.asarray(sorted(splits[target]["train"]), dtype=np.int64)
    source_train_parcels = np.asarray(sorted(splits[source]["train"]), dtype=np.int64)

    target_loader_raw = phasevis._selected_loader(
        data_root, target, classes, target_train_parcels,
        closed_set=closed_set, combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    unlabeled_target_loader = LabelStrippedLoader(target_loader_raw)
    source_train_loader = phasevis._selected_loader(
        data_root, source, classes, source_train_parcels,
        closed_set=closed_set, combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    source_bank = _source_bank(calibration)
    if not bool(source_bank.ready.all().item()):
        raise RuntimeError("13A requires all source semantic/geometry classes ready")
    source_refs = source_bank.fused.detach().cpu().float().numpy()

    semantic_obs = _load_or_generate_semantic_cache(
        cache_dir / "unlabeled_semantic_observables.pt",
        model=model, loader=unlabeled_target_loader, source_refs=source_refs,
        bootstrap=bootstrap, time_scale_days=time_scale_days, device=device,
        source=source, target=target, seed=seed, fold=fold, model_checkpoint=model_checkpoint_path,
    )
    if set(map(int, semantic_obs["sample_ids"])) != set(map(int, target_train_parcels.tolist())):
        raise ValueError("13A semantic observable cache does not cover exactly target-train")

    scan_config = samplediag._scan_config(runtime, args.registration_workers)
    reg_extractor = build_stage2_registration_extractor(model, device=device, k_reg=scan_config.k_reg)
    source_reg_bank = samplediag._load_or_build_registration_bank(
        args.source_registration_bank_cache.resolve(), model=model,
        source_train_loader=DeviceBatchLoader(source_train_loader, device), num_classes=len(classes),
        device=device, reg_extractor=reg_extractor,
    )
    geometry = _load_or_generate_candidate_geometry(
        cache_dir / "raw_candidate_geometry.pt",
        model=model, loader=unlabeled_target_loader, semantic_obs=semantic_obs,
        source_bank=source_bank, source_reg_bank=source_reg_bank, reg_extractor=reg_extractor,
        scan_config=scan_config, device=device, source=source, target=target, seed=seed, fold=fold,
        model_checkpoint=model_checkpoint_path, workers=args.registration_workers,
        geometry_chunk_size=args.geometry_chunk_size, dp_chunk_size=args.dp_target_chunk_size,
    )

    unlabeled_rows, gamma_matrix = _build_unlabeled_rows(semantic_obs, geometry)
    # Hard separation: write label-free artifacts *before* any target metadata label is read.
    _write_csv(output / "01_unlabeled_sample_observables.csv", unlabeled_rows)
    np.savez_compressed(
        output / "02_unlabeled_dense_vectors.npz",
        sample_id=np.asarray(semantic_obs["sample_ids"], dtype=np.int64),
        raw_posterior=np.asarray(semantic_obs["raw_posterior"], dtype=np.float32),
        phase_posterior=np.asarray(semantic_obs["phase_posterior"], dtype=np.float32),
        semantic_raw_similarity_all_classes=np.asarray(semantic_obs["semantic_raw_similarity_all"], dtype=np.float32),
        semantic_phase_similarity_all_classes=np.asarray(semantic_obs["semantic_phase_similarity_all"], dtype=np.float32),
        candidate_gamma=gamma_matrix,
    )
    forbidden = {"true_label", "candidate_correct", "confusion_flow", "true_class_name"}
    if any(forbidden.intersection(row) for row in unlabeled_rows):
        raise RuntimeError("oracle field leaked into label-free observable artifact")
    print(f"SEED13A_UNLABELED_OBSERVABLES_SAVED|n={len(unlabeled_rows)}|contains_true_label=false", flush=True)

    # Oracle-only evaluator starts here. Target true label first becomes visible now.
    target_meta = phasevis._metadata_dataset(
        data_root, target, classes, splits[target]["train"], closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
    )
    label_by_parcel = {
        int(parcel): int(label) for parcel, label in zip(target_meta.get_parcel_indices().tolist(), target_meta.get_labels().tolist())
    }
    audited = _oracle_join(unlabeled_rows, label_by_parcel, classes)
    _write_csv(output / "03_oracle_sample_audit.csv", audited)
    summary_rows, coverage_rows, balance_rows, veto_rows, overall_rows, matched_global = _oracle_analysis(
        audited, classes, seed=args.analysis_seed, bootstrap_reps=args.bootstrap_reps,
    )
    _write_csv(output / "04_per_candidate_evidence_summary.csv", summary_rows)
    _write_csv(output / "05_precision_coverage.csv", coverage_rows)
    _write_csv(output / "06_confidence_matched_balance.csv", balance_rows)
    _write_csv(output / "07_geometry_veto_summary.csv", veto_rows)
    critical_rows = _critical_flow_rows(audited, classes)
    _write_csv(output / "08_critical_confusion_flows.csv", critical_rows)
    critical_matched, critical_match_balance = _critical_matched_rows(
        audited, classes, seed=args.analysis_seed + 5000, bootstrap_reps=args.bootstrap_reps,
    )
    _write_csv(output / "08b_critical_confusion_flow_matched.csv", critical_matched)
    _write_csv(output / "08c_critical_confusion_flow_match_balance.csv", critical_match_balance)
    complementarity = _complementarity_rows(audited, classes, matched_global)
    _write_csv(output / "09_evidence_complementarity.csv", complementarity)
    _write_csv(output / "10_overall_evidence_summary.csv", overall_rows)

    selected_plot_evidence = ["raw_prob", "raw_phase_js", "semantic_phase_candidate_margin", "T_gain_ratio", "S_gain_ratio"]
    for evidence in selected_plot_evidence:
        _plot_distribution_panels(output / "plots" / "evidence_distributions" / f"{evidence}.png", audited, classes, evidence)
    for cid, cname in enumerate(classes):
        _plot_precision_coverage(output / "plots" / "precision_coverage" / f"{cid:02d}_{cname}.png", coverage_rows, cid, cname)
    for idx, evidence in enumerate(("semantic_phase_candidate_margin", "raw_phase_js", "S_gain_ratio")):
        _plot_confidence_scatter(output / "plots" / "confidence_vs_evidence" / f"{evidence}.png", audited, evidence, seed=args.analysis_seed + idx)

    pred = np.asarray([int(row["raw_pred"]) for row in audited], dtype=np.int64)
    correct = np.asarray([bool(row["candidate_correct"]) for row in audited], dtype=bool)
    candidate_precision = {
        str(classes[cid]): {
            "n_candidate": int(np.sum(pred == cid)),
            "n_correct": int(np.sum((pred == cid) & correct)),
            "n_wrong": int(np.sum((pred == cid) & ~correct)),
            "raw_candidate_precision": float(np.mean(correct[pred == cid])) if np.any(pred == cid) else float("nan"),
        } for cid in range(len(classes))
    }
    flow_counts = {}
    for flow in (
        "spring_oat -> spring_oat", "spring_barley -> spring_oat",
        "winter_triticale -> winter_triticale", "winter_rye -> winter_triticale",
        "winter_wheat -> winter_triticale", "winter_rye -> winter_rye", "winter_wheat -> winter_wheat",
    ):
        flow_counts[flow] = int(sum(row["confusion_flow"] == flow for row in audited))

    diagnostic_summary = {
        "protocol": PROTOCOL,
        "source": source, "target": target, "seed": seed, "fold": fold,
        "analysis_split": "target-train",
        "n_samples": len(audited),
        "bootstrap_state": bootstrap,
        "candidate_definition": "frozen raw Stage-1 classifier top-1 on native target time",
        "candidate_is_trainable_label": False,
        "phase_view_relabels_candidate": False,
        "geometry_relabels_candidate": False,
        "semantic_reference_relabels_candidate": False,
        "observable_generation_contains_target_true_label": False,
        "oracle_evaluator_join_after_observable_save": True,
        "candidate_precision": candidate_precision,
        "critical_flow_counts": flow_counts,
        "confidence_matching": balance_rows,
        "evidence_groups": {
            "A": "classifier certainty baseline",
            "B": "raw/Phase temporal semantic agreement/conflict",
            "C": "positive source semantic-reference support",
            "D": "negative frozen structural veto potential",
            "E": "not run; no new stochastic augmentation introduced",
        },
        "automatic_trainable_gate": None,
        "automatic_threshold": None,
        "automatic_13A_verdict": None,
        "teacher_student_training": False,
        "optimizer_steps": 0,
        "ema_teacher": False,
        "pse_adaptation": False,
        "ltae_adaptation": False,
        "classifier_adaptation": False,
        "shared_phase_refresh": False,
        "class_conditioned_phase": False,
        "domain_shape_transport": False,
        "learned_evidence_combination": False,
        "threshold_grid_search": False,
        "class_specific_thresholds": False,
    }
    _json_dump(output / "11_13a_diagnostic_summary.json", diagnostic_summary)
    manifest = {
        "protocol": PROTOCOL,
        "calibration_checkpoint": str(args.calibration_checkpoint.resolve()),
        "model_checkpoint": str(model_checkpoint_path),
        "experiment12_summary": str(args.experiment12_summary.resolve()),
        "source_registration_bank_cache": str(args.source_registration_bank_cache.resolve()),
        "raw_candidate_geometry_cache": str((cache_dir / "raw_candidate_geometry.pt").resolve()),
        "source": source, "target": target, "seed": seed, "fold": fold,
        "classes": classes, "target_train_size": len(audited),
        "bootstrap_state": bootstrap,
        "raw_observable_file_contains_true_label": False,
        "dense_vector_file_contains_true_label": False,
        "oracle_join_is_separate_phase": True,
        "candidate_class_source": "raw classifier top-1 only",
        "candidate_registration_assignment_source": "raw classifier top-1 only",
        "candidate_registration_target_label_use": False,
        "production_legality_used_as_reliability_conclusion": False,
        "confidence_threshold_preselection": False,
        "confidence_matching_method": "within-candidate-class 10x10 2D quantile coarsened exact matching on raw probability and raw margin",
        "confidence_matching_seed": int(args.analysis_seed),
        "bootstrap_ci_replicates": int(args.bootstrap_reps),
        "optional_input_consistency_evidence_E": False,
        "optional_E_reason": "no new augmentation is introduced solely for experiment 13A",
        "network_parameters_frozen": True,
        "training_updates": False,
        "teacher_student": False,
        "stable_label": False,
        "pseudo_label_training": False,
        "trainable_gate_constructed": False,
        "geometry_argmin_reclassification": False,
        "learned_gate": False,
    }
    _json_dump(output / "00_manifest.json", manifest)
    _write_readme(output / "README_中文说明.md", str(bootstrap["state"]))
    print(f"SEED13A_DONE|output={output}|n={len(audited)}|training_updates=false|trainable_gate=false", flush=True)
    return diagnostic_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-checkpoint", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--experiment12-summary", type=Path, required=True)
    parser.add_argument("--bootstrap-state", choices=FORMAL_BOOTSTRAP_STATES, required=True,
                        help="Formal unlabeled delta_boot chosen after experiment 12; oracle_shared is intentionally disallowed.")
    parser.add_argument("--source-registration-bank-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--registration-workers", type=int, default=4)
    parser.add_argument("--geometry-chunk-size", type=int, default=512)
    parser.add_argument("--dp-target-chunk-size", type=int, default=512)
    parser.add_argument("--analysis-seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.registration_workers < 1 or args.geometry_chunk_size < 1 or args.dp_target_chunk_size < 1:
        parser.error("registration workers and geometry/DP chunk sizes must be positive")
    if args.bootstrap_reps < 50:
        parser.error("bootstrap-reps must be at least 50")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
