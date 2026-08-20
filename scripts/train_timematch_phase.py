#!/usr/bin/env python3
"""Unified TimeMatch adaptation with optional nonlinear residual Domain Phase.

``--alpha-candidates 0`` is exactly scalar TimeMatch and skips geometry.
Providing any nonzero alpha builds one frozen bootstrap residual warp and,
after each ordinary TimeMatch scalar AM search, refines alpha by AM.

Target-train truth is physically stripped from the main training and candidate
selection paths.  Registration is Stage-1-frozen, bootstrap-only and no-grad.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import Tensor

from methods.structure_da.phase_evidence import compute_gamma_diagnostics
from methods.structure_da.phase_registration import (
    FdasrsfCurveRegistrationAdapter,
    build_source_registration_prototypes,
    check_gamma_legality,
)
from methods.structure_da.registration_geometry import evaluate_registration_geometry
from methods.structure_da.original_timematch import (
    FrozenGeometryCopy,
    build_original_timematch_model,
    module_state_hash,
)
from methods.structure_da.temporal_srvf import TemporalSRVFExtractor
from methods.structure_da.timematch_nonlinear_phase import (
    ALPHA_BANK,
    aggregate_class_residual_phases,
    candidate_phase_grid,
    canonical_grid,
    evaluate_phase_grid,
    inverse_phase_on_canonical_grid,
    requires_nonlinear_phase,
    select_alpha_from_probabilities,
    translate_grid_function,
)
from methods.structure_da.timematch_utils import (
    class_distribution,
    official_timematch_ema_update,
    sha256_file,
    write_json,
)
from utils.focal_loss import FocalLoss

PROTOCOL = "UNIFIED_TIMEMATCH_OPTIONAL_NONLINEAR_PHASE_v1"
FORBIDDEN_TARGET_KEYS = frozenset({"label", "labels", "target", "targets", "y", "true_label", "true_labels"})


def _capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _state_hash(model) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _integration_weights(grid: Tensor) -> Tensor:
    w = torch.ones_like(grid, dtype=torch.float64, device="cpu")
    w[[0, -1]] *= 0.5
    return w / w.sum()


def _metric(labels: np.ndarray, pred: np.ndarray, num_classes: int) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("metric labels/pred must be matching vectors")
    if y.size == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "weighted_f1": float("nan")}
    f1, support = [], []
    for c in range(int(num_classes)):
        tp = int(np.sum((y == c) & (p == c)))
        fp = int(np.sum((y != c) & (p == c)))
        fn = int(np.sum((y == c) & (p != c)))
        denom = 2 * tp + fp + fn
        f1.append(0.0 if denom == 0 else (2.0 * tp / denom))
        support.append(int(np.sum(y == c)))
    return {
        "accuracy": float(np.mean(y == p)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)) if sum(support) else float("nan"),
    }


def _phase_positions(backbone, phase_grid: Tensor, *, time_scale_days: float, augmentation_shift_days=None) -> Tensor:
    pos = evaluate_phase_grid(backbone.normalized_positions, phase_grid, time_mask=backbone.time_mask)
    if augmentation_shift_days is not None:
        aug = augmentation_shift_days
        if not isinstance(aug, Tensor):
            aug = torch.as_tensor(aug, device=pos.device)
        aug = aug.to(device=pos.device, dtype=pos.dtype)
        if aug.ndim == 1:
            aug = aug[:, None]
        pos = torch.where(backbone.time_mask, pos + aug / float(time_scale_days), torch.zeros_like(pos))
    return pos


def forward_phase(model, batch: dict, *, phase_grid: Tensor, augmentation_shift_days=None):
    bb = model.forward_backbone(
        batch["pixels"], batch["valid_pixels"], batch["positions"], batch.get("extra"),
        time_mask=batch.get("time_mask"), compute_decomposition=False,
    )
    pos = _phase_positions(
        bb, phase_grid, time_scale_days=float(model.backbone.time_scale),
        augmentation_shift_days=augmentation_shift_days,
    )
    return model.forward_from_backbone(
        bb, batch["positions"], batch.get("extra"), temporal_positions_override=pos, return_geometry=False
    )


def _scalar_phase_grid(delta_days: float, k: int = 128) -> Tensor:
    return candidate_phase_grid(delta_days=delta_days, alpha=0.0, rho_dom=canonical_grid(k), time_scale_days=365.0)


def _teacher_forward(model, batch: dict, *, mode: str, delta_days: int, alpha: float, rho_dom: Tensor):
    grid = candidate_phase_grid(
        delta_days=delta_days, alpha=alpha, rho_dom=rho_dom, time_scale_days=float(model.backbone.time_scale)
    )
    return forward_phase(model, batch, phase_grid=grid)


def _source_forward(model, batch: dict, *, mode: str, source_to_target_days: int, source_phase: Tensor | None):
    aug = batch.get("temporal_aug_shift_days")
    if source_phase is None:
        raise RuntimeError("TimeMatch requires a fixed source inverse phase")
    return forward_phase(model, batch, phase_grid=source_phase, augmentation_shift_days=aug)


def alpha_probability_cube(
    model, loader, device: torch.device, *, delta_days: int, alphas: Sequence[float], rho_dom: Tensor,
    max_batches: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate only alpha at a fixed scalar delta; never searches delta×alpha."""
    model.eval(); parts = []; ids = []
    phase_grids = [candidate_phase_grid(delta_days=delta_days, alpha=a, rho_dom=rho_dom,
                                        time_scale_days=float(model.backbone.time_scale)) for a in alphas]
    with torch.no_grad():
        for bi, raw in enumerate(loader):
            if bi >= int(max_batches):
                break
            if FORBIDDEN_TARGET_KEYS.intersection(raw.keys()):
                raise RuntimeError("target truth leaked into alpha candidate evaluation")
            batch = tm.move(raw, device)
            bb = model.forward_backbone(
                batch["pixels"], batch["valid_pixels"], batch["positions"], batch.get("extra"),
                time_mask=batch.get("time_mask"), compute_decomposition=False,
            )
            b, l, d = bb.tokens.shape; na = len(phase_grids)
            latent = bb.tokens[:, None].expand(b, na, l, d).reshape(b * na, l, d)
            mask = bb.time_mask[:, None].expand(b, na, l).reshape(b * na, l)
            pos_by_alpha = [evaluate_phase_grid(bb.normalized_positions, g, time_mask=bb.time_mask) for g in phase_grids]
            pos = torch.stack(pos_by_alpha, dim=1).reshape(b * na, l)
            raw_enc = model.temporal_module.raw_encoder(latent=latent, positions=pos, mask=mask)
            logits = model.classifier(raw_enc.fused_repr)
            parts.append(torch.softmax(logits.float(), dim=1).reshape(b, na, -1).cpu())
            ids.append(batch["parcel_index"].detach().cpu().long())
    if not parts:
        raise RuntimeError("alpha candidate evaluation received no target batches")
    return torch.cat(parts, dim=0).numpy().astype(np.float64), torch.cat(ids).numpy().astype(np.int64)


def _save_alpha_candidates(path: Path, *, ids: np.ndarray, cube: np.ndarray, alphas: Sequence[float],
                           delta_days: int, selected_alpha: float, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = cube.argmax(axis=2).astype(np.int64)
    conf = cube.max(axis=2).astype(np.float32)
    np.savez_compressed(
        path, sample_id=ids.astype(np.int64), posterior=cube.astype(np.float32),
        alphas=np.asarray(alphas, dtype=np.float32), predictions=pred,
        confidence=conf, confidence_mask=(conf > float(threshold)),
        delta_days=np.asarray([int(delta_days)], dtype=np.int64),
        selected_alpha=np.asarray([float(selected_alpha)], dtype=np.float32),
        contains_target_true_labels=np.asarray([False], dtype=np.bool_),
    )


def _alpha_select(model, loader, device, *, delta_days: int, alphas: Sequence[float], rho_dom: Tensor,
                  class_distr: np.ndarray, sample_size: int, output: Path, epoch_tag: str,
                  pseudo_threshold: float) -> tuple[float, list[dict], dict]:
    # DataLoader iterator construction itself can consume the process torch RNG.
    # Alpha evaluation is a no-grad selection diagnostic and must not perturb
    # the TimeMatch sampler/augmentation random stream.
    rng = _capture_rng()
    try:
        cube, ids = alpha_probability_cube(
            model, loader, device, delta_days=delta_days, alphas=alphas, rho_dom=rho_dom, max_batches=sample_size
        )
    finally:
        _restore_rng(rng)
    selected = select_alpha_from_probabilities(cube, alphas, class_distr)
    is_best = int(np.argmax([float(r["inception_score"]) for r in selected.rows]))
    rows = []
    for j, row in enumerate(selected.rows):
        r = dict(row); r.update({
            "epoch_tag": epoch_tag, "delta_days_fixed": int(delta_days), "search_dimension": "alpha_only",
            "is_score": float(row["inception_score"]), "is_selected_diagnostic": bool(j == is_best),
        })
        rows.append(r)
    tm.write_csv(output / "alpha_scans" / f"{epoch_tag}.csv", rows)
    _save_alpha_candidates(
        output / "alpha_candidates" / f"{epoch_tag}.npz", ids=ids, cube=cube, alphas=alphas,
        delta_days=delta_days, selected_alpha=selected.alpha, threshold=pseudo_threshold,
    )
    print(
        f"TMNP_ALPHA|epoch_tag={epoch_tag}|delta_fixed={delta_days:+d}|selected={selected.alpha:.2f}|"
        f"best_am={selected.best_am:.8f}|second_am={selected.second_best_am:.8f}|margin={selected.margin:.8f}|"
        f"candidates={','.join(f'{x:.2f}' for x in alphas)}|samples={cube.shape[0]}", flush=True,
    )
    summary = {
        "best_am": selected.best_am, "second_best_am": selected.second_best_am, "margin": selected.margin,
        "is_diagnostic_alpha": float(alphas[is_best]),
    }
    return selected.alpha, rows, summary


def _target_geometry_prototypes(
    model, loader, device: torch.device, *, reg_extractor, pseudo_by_id: dict[int, tuple[int, float]],
    num_classes: int, threshold: float,
) -> tuple[list[Tensor | None], list[Tensor | None], list[int], Tensor]:
    q_sum = [None] * num_classes; sup_sum = [None] * num_classes; counts = [0] * num_classes; grid = None
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            if FORBIDDEN_TARGET_KEYS.intersection(raw.keys()):
                raise RuntimeError("target truth leaked into nonlinear residual bootstrap")
            batch = tm.move(raw, device)
            output = model(
                batch["pixels"], batch["valid_pixels"], batch["positions"], batch.get("extra"), return_geometry=True
            )
            reg = evaluate_registration_geometry(output.trend, output.positions, output.mask, reg_extractor)
            if grid is None:
                grid = reg.registration_grid.detach().cpu().double()
            ids = batch["parcel_index"].detach().cpu().long().tolist()
            for i, sid in enumerate(ids):
                assignment = pseudo_by_id.get(int(sid))
                if assignment is None:
                    continue
                cid, conf = assignment
                if float(conf) <= float(threshold) or not bool(reg.trend_valid[i].item()):
                    continue
                q = reg.trend_srvf[i].detach().cpu().double(); sup = reg.trend_support[i].detach().cpu().double()
                weighted = q * sup[:, None]
                if q_sum[cid] is None:
                    q_sum[cid] = weighted.clone(); sup_sum[cid] = sup.clone()
                else:
                    q_sum[cid] += weighted; sup_sum[cid] += sup
                counts[cid] += 1
    if grid is None:
        raise RuntimeError("target geometry bootstrap received no batches")
    q_out, sup_out = [], []
    for cid in range(num_classes):
        if q_sum[cid] is None or counts[cid] == 0:
            q_out.append(None); sup_out.append(None); continue
        q_out.append(q_sum[cid] / (sup_sum[cid][:, None] + 1e-8))
        sup_out.append(sup_sum[cid] / float(counts[cid]))
    return q_out, sup_out, counts, grid


def _build_frozen_residual(
    *, runtime: dict, semantic_model, source_train: Sequence[int], target_train: Sequence[int],
    pseudo_scan: dict, delta0: int, threshold: float, device: torch.device, batch_size: int, num_workers: int,
    registration_workers: int, output: Path, stage1_checkpoint_sha256: str,
) -> tuple[Tensor, dict, FrozenGeometryCopy]:
    """One-time O(C_valid) prototype registration; returns frozen target->source residual rho_dom."""
    rng = _capture_rng()
    try:
        classes = [str(x) for x in runtime["classes"]]; num_classes = len(classes)
        geometry_model = FrozenGeometryCopy(semantic_model).to(device)
        geometry_model.assert_frozen()
        geometry_hash_before = module_state_hash(geometry_model.geometry_pse)
        scan_config = tm.registration_config(runtime, registration_workers)
        reg_extractor = TemporalSRVFExtractor(
            feature_dim=128, canonical_grid_size=scan_config.k_reg,
            time_reference=0.0, time_scale=1.0,
            min_mean_support=0.0, min_dynamic_energy=0.0,
        ).to(device)
        source_loader = tm.selected_loader(
            runtime["data_root"], runtime["source"], classes, np.asarray(source_train, dtype=np.int64),
            runtime,
            batch_size=batch_size, num_workers=num_workers,
        )
        source_bank = build_source_registration_prototypes(
            geometry_model, tm.DeviceBatchLoader(source_loader, device), num_classes, device=device, reg_extractor=reg_extractor
        )
        target_loader = tm.scan_loader(
            runtime["data_root"], runtime["target"], classes, target_train, runtime,
            batch_size=batch_size, num_workers=num_workers, strip_label=True,
        )
        ids = pseudo_scan["sample_id"].detach().cpu().long().tolist()
        post = pseudo_scan["posterior"].detach().cpu().float()
        pred = post.argmax(dim=1).tolist(); conf = post.max(dim=1).values.tolist()
        pseudo_by_id = {int(s): (int(c), float(q)) for s, c, q in zip(ids, pred, conf)}
        target_q, target_sup, counts, grid = _target_geometry_prototypes(
            geometry_model, target_loader, device, reg_extractor=reg_extractor, pseudo_by_id=pseudo_by_id,
            num_classes=num_classes, threshold=threshold,
        )
        min_samples = int(float(runtime.get("stage2_phase_min_samples_per_class", 3.0)))
        adapter = FdasrsfCurveRegistrationAdapter(scan_config.registration_lambda)
        weights = _integration_weights(grid)
        class_rhos = []; class_rows = []; rho_payload = {}
        for cid in range(num_classes):
            row = {"class_id": cid, "class_name": classes[cid], "target_support_samples": int(counts[cid]),
                   "minimum_samples": min_samples, "valid": False, "reject_reason": ""}
            if counts[cid] < min_samples:
                row["reject_reason"] = "insufficient_target_support"; class_rows.append(row); continue
            if not bool(source_bank.ready[cid].item()) or target_q[cid] is None:
                row["reject_reason"] = "prototype_unavailable"; class_rows.append(row); continue
            source_q = source_bank.trend_srvf[cid].detach().cpu().double()
            source_sup = source_bank.trend_support[cid].detach().cpu().double()
            corrected_q, corrected_sup = translate_grid_function(
                target_q[cid], delta_days=delta0, time_scale_days=float(runtime.get("time_scale", 365.0)),
                support=target_sup[cid],
            )
            try:
                gamma_s2t = adapter.register(source_q, corrected_q)
                legal = check_gamma_legality(
                    gamma_s2t, grid,
                    registration_min_increment=scan_config.registration_min_increment,
                    registration_max_local_speed=scan_config.registration_max_local_speed,
                    registration_max_roughness=scan_config.registration_max_roughness,
                    registration_max_deviation=scan_config.registration_max_deviation,
                )
                diag = compute_gamma_diagnostics(
                    sample_id=-1, class_id=cid, gamma=gamma_s2t, source_trend_srvf=source_q,
                    target_trend_srvf=corrected_q, source_support=source_sup, target_support=corrected_sup,
                    integration_weights=weights, registration_grid=grid,
                )
                residual_ok = bool(legal.legal and diag.common_support >= scan_config.registration_min_common_support
                                   and diag.gain_ratio <= scan_config.registration_gain_ratio_max)
                row.update({
                    "solver_success": True, "gamma_legal": bool(legal.legal), "common_support": diag.common_support,
                    "identity_error": diag.e_id, "registered_error": diag.e_reg, "gain_ratio": diag.gain_ratio,
                    "roughness": legal.roughness, "min_increment": legal.min_increment,
                    "max_local_speed": legal.max_local_speed, "phase_deviation": legal.phase_deviation,
                })
                if not residual_ok:
                    reasons = []
                    if not legal.legal: reasons.append("gamma_illegal")
                    if diag.common_support < scan_config.registration_min_common_support: reasons.append("low_common_support")
                    if diag.gain_ratio > scan_config.registration_gain_ratio_max: reasons.append("gain_gate")
                    row["reject_reason"] = "+".join(reasons); class_rows.append(row); continue
                rho = inverse_phase_on_canonical_grid(gamma_s2t).detach().cpu().double()
                class_rhos.append(rho); rho_payload[str(cid)] = rho
                row.update({"valid": True, "reject_reason": ""}); class_rows.append(row)
            except Exception as exc:  # numerical failure is a legal bootstrap exclusion
                row.update({"solver_success": False, "reject_reason": f"solver_error:{type(exc).__name__}"})
                class_rows.append(row)
        fallback_reason = None
        if not class_rhos:
            rho_dom = canonical_grid(int(scan_config.k_reg)); fallback_reason = "no_valid_classes"
        else:
            try:
                rho_dom = aggregate_class_residual_phases(class_rhos)
                for alpha in ALPHA_BANK:
                    candidate_phase_grid(delta_days=delta0, alpha=alpha, rho_dom=rho_dom,
                                         time_scale_days=float(runtime.get("time_scale", 365.0)))
            except Exception as exc:
                rho_dom = canonical_grid(int(scan_config.k_reg)); fallback_reason = f"invalid_domain_residual:{type(exc).__name__}"
        tm.write_csv(output / "bootstrap_registration_by_class.csv", class_rows)
        torch.save({
            "schema_version": "timematch_nonlinear_domain_phase_v1",
            "task": f"{runtime['source']}->{runtime['target']}",
            "seed": int(runtime["seed"]), "fold": 0,
            "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
            "rho_by_class": rho_payload, "rho_dom": rho_dom, "residual": rho_dom - canonical_grid(rho_dom.numel()),
            "delta0_days": int(delta0), "valid_class_ids": [int(r["class_id"]) for r in class_rows if r["valid"]],
            "fallback_reason": fallback_reason, "target_truth_used": False,
            "geometry_state_hash": geometry_hash_before,
        }, output / "bootstrap_domain_residual.pt")
        geometry_model.assert_frozen()
        if module_state_hash(geometry_model.geometry_pse) != geometry_hash_before:
            raise RuntimeError("frozen Stage-1 geometry model changed during residual construction")
        info = {"valid_classes": [int(r["class_id"]) for r in class_rows if r["valid"]],
                "target_support_by_class": counts, "fallback_reason": fallback_reason,
                "geometry_state_hash": geometry_hash_before}
        print(
            f"TMNP_RESIDUAL|delta0={delta0:+d}|valid_classes={','.join(map(str, info['valid_classes'])) or 'none'}|"
            f"fallback={fallback_reason or 'none'}|registration_calls={len(class_rhos)}", flush=True,
        )
        return rho_dom, info, geometry_model
    finally:
        _restore_rng(rng)


def _scan_phase(model, loader, device, *, mode: str, delta_days: int, alpha: float, rho_dom: Tensor,
                allow_labels: bool) -> dict:
    model.eval(); ids = []; posts = []; labels = []
    with torch.no_grad():
        for raw in loader:
            if not allow_labels and FORBIDDEN_TARGET_KEYS.intersection(raw.keys()):
                raise RuntimeError("target truth leaked into label-free scan")
            batch = tm.move(raw, device)
            out = _teacher_forward(model, batch, mode=mode, delta_days=delta_days, alpha=alpha, rho_dom=rho_dom)
            ids.append(batch["parcel_index"].detach().cpu().long()); posts.append(torch.softmax(out.logits.float(), dim=1).cpu())
            if allow_labels:
                labels.append(batch["label"].detach().cpu().long())
    sid = torch.cat(ids); post = torch.cat(posts); order = torch.argsort(sid)
    result = {"sample_id": sid[order], "posterior": post[order]}
    if allow_labels:
        result["label"] = torch.cat(labels)[order]
    return result


@torch.no_grad()
def _scan_fixed_phase(model, loader, device, *, phase_grid: Tensor) -> dict:
    model.eval(); ids = []; posts = []; labels = []
    for raw in loader:
        batch = tm.move(raw, device)
        out = forward_phase(model, batch, phase_grid=phase_grid)
        ids.append(batch["parcel_index"].detach().cpu().long())
        posts.append(torch.softmax(out.logits.float(), dim=1).cpu())
        labels.append(batch["label"].detach().cpu().long())
    sample_id = torch.cat(ids); posterior = torch.cat(posts); label = torch.cat(labels)
    order = torch.argsort(sample_id)
    return {"sample_id": sample_id[order], "posterior": posterior[order], "label": label[order]}


def _save_and_print_validation(output: Path, rows: list[dict], *, epoch: int, mode: str, student, teacher,
                               val_loader, device, delta_days: int, alpha: float, source_phase: Tensor,
                               pseudo_threshold: float, num_classes: int) -> float:
    s = _scan_fixed_phase(student, val_loader, device, phase_grid=source_phase)
    t = _scan_fixed_phase(teacher, val_loader, device, phase_grid=source_phase)
    if not torch.equal(s["sample_id"], t["sample_id"]):
        raise RuntimeError("Student/Teacher source-val sample ordering mismatch")
    truth = s["label"].numpy().astype(np.int64)
    row_base = {"epoch": int(epoch), "mode": mode, "delta_days": int(delta_days), "alpha": float(alpha)}
    metrics = {}
    for role, scan in (("student", s), ("teacher", t)):
        post = scan["posterior"].numpy(); pred = post.argmax(1); conf = post.max(1)
        m = _metric(truth, pred, num_classes); coverage = float(np.mean(conf > pseudo_threshold))
        row = {**row_base, "role": role, **m, "confidence_coverage": coverage, "mean_max_probability": float(conf.mean())}
        rows.append(row); metrics[role] = row
    tm.write_csv(output / "source_val_metrics.csv", rows)
    ss = metrics["student"]; tt = metrics["teacher"]
    print(
        f"TMNP_SOURCE_VAL|mode={mode}|epoch={epoch}|delta={delta_days:+d}|alpha={alpha:.2f}|"
        f"student_acc={ss['accuracy']:.6f}|student_macro_f1={ss['macro_f1']:.6f}|student_weighted_f1={ss['weighted_f1']:.6f}|"
        f"teacher_acc={tt['accuracy']:.6f}|teacher_macro_f1={tt['macro_f1']:.6f}|teacher_weighted_f1={tt['weighted_f1']:.6f}|"
        f"teacher_conf_coverage={tt['confidence_coverage']:.6f}", flush=True,
    )
    return float(ss["macro_f1"])


def _evaluate_final_target_test(output: Path, *, student, runtime: dict, target_test: Sequence[int],
                                classes: Sequence[str], device: torch.device, num_workers: int) -> dict:
    # Held-out target-test labels become visible only after all 20 training epochs.
    test_loader = tm.scan_loader(
        runtime["data_root"], runtime["target"], list(classes), target_test, runtime,
        batch_size=64, num_workers=num_workers, strip_label=False,
    )
    scan = tm.scan_model(student, test_loader, device, allow_labels=True)
    truth = scan["label"].numpy().astype(np.int64)
    post = scan["posterior"].numpy().astype(np.float32)
    pred = post.argmax(1).astype(np.int64)
    metrics = _metric(truth, pred, len(classes))
    payload = {
        "role": "student",
        "checkpoint_semantics": "final_student_after_epoch_20__TimeMatch_output_student_true",
        "n": int(truth.size),
        **metrics,
        "target_test_used_for_training": False,
    }
    write_json(output / "test_metrics.json", payload)
    np.savez_compressed(
        output / "test_predictions.npz",
        sample_id=scan["sample_id"].numpy().astype(np.int64), label=truth,
        posterior=post, prediction=pred,
    )
    print(
        f"TMNP_TEST|mode={output.name}|role=student|epoch=20|n={truth.size}|"
        f"accuracy={metrics['accuracy']:.6f}|macro_f1={metrics['macro_f1']:.6f}|"
        f"weighted_f1={metrics['weighted_f1']:.6f}", flush=True,
    )
    return payload


def _parse_alpha_candidates(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("alpha candidates must be comma-separated numbers") from error
    try:
        requires_nonlinear_phase(values)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("alpha candidates must be unique")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-candidates", type=_parse_alpha_candidates, default=(0.0,), help="comma-separated values in [0,1]; use 0 for exact scalar TimeMatch")
    p.add_argument("--model-checkpoint", type=Path, required=True)
    p.add_argument("--geometry-config", type=Path, default=None, help="JSON registration thresholds; required only when a nonzero alpha is enabled")
    p.add_argument("--source", default=None, help="optional runtime source-domain override for P1")
    p.add_argument("--target", default=None, help="optional runtime target-domain override for P1")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--registration-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--geometry-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pseudo-threshold", type=float, default=.9)
    p.add_argument("--ema-decay", type=float, default=.9999)
    p.add_argument("--trade-off", type=float, default=2.0)
    p.add_argument("--focal-gamma", type=float, default=1.0)
    p.add_argument("--seq-length", type=int, default=30)
    p.add_argument("--num-pixels", type=int, default=64)
    p.add_argument("--max-temporal-shift", type=int, default=60)
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--shift-chunk", type=int, default=8)
    p.add_argument("--with-shift-aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shift-aug-p", type=float, default=1.0)
    p.add_argument("--max-shift-aug", type=int, default=60)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main() -> None:
    global tm
    a = parse_args()
    from scripts import timematch_runtime as tm
    alpha_bank = tuple(a.alpha_candidates)
    nonlinear_enabled = requires_nonlinear_phase(alpha_bank)
    mode = "NONLINEAR_PHASE" if nonlinear_enabled else "ALPHA_ZERO"
    if a.epochs != 20 or a.steps_per_epoch != 500:
        raise ValueError("this experiment is frozen to 20 epochs x 500 steps")
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    seed = 1
    tm.set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(a.device)
    checkpoint = torch.load(a.model_checkpoint.resolve(), map_location="cpu", weights_only=False)
    runtime = dict(checkpoint.get("runtime_config") or {})
    if not runtime:
        raise ValueError("Stage-1 checkpoint is missing runtime_config; rerun the unified source stage")
    if nonlinear_enabled:
        if a.geometry_config is None:
            raise ValueError("--geometry-config is required when alpha candidates contain a nonzero value")
        runtime.update(json.loads(a.geometry_config.resolve().read_text(encoding="utf-8")))
    if a.source is not None:
        runtime["source"] = str(a.source)
    if a.target is not None:
        runtime["target"] = str(a.target)
    runtime["data_root"] = str(a.data_root.resolve())
    if int(runtime.get("seed", -1)) != seed:
        raise ValueError("P0/P1 is frozen to seed=1")
    classes = [str(x) for x in runtime["classes"]]; num_classes = len(classes)
    student = build_original_timematch_model(runtime, checkpoint, device)
    policy = SimpleNamespace(
        trainable_parameter_names=tuple(name for name, parameter in student.named_parameters() if parameter.requires_grad),
        frozen_parameter_names=tuple(name for name, parameter in student.named_parameters() if not parameter.requires_grad),
    )
    teacher = deepcopy(student); teacher.eval(); [p.requires_grad_(False) for p in teacher.parameters()]

    source_all = tm.eligible_parcels(runtime["data_root"], runtime["source"], classes, runtime)
    target_all = tm.eligible_parcels(runtime["data_root"], runtime["target"], classes, runtime)
    splits = tm.reconstruct_fold_splits(
        source_all, target_all, source=runtime["source"], target=runtime["target"], seed=seed,
        val_ratio=float(runtime.get("val_ratio", .1)), test_ratio=float(runtime.get("test_ratio", .2)), fold=0,
    )
    source_train = sorted(splits[runtime["source"]]["train"]); target_train = sorted(splits[runtime["target"]]["train"])
    source_val = sorted(splits[runtime["source"]]["val"])
    target_test = sorted(splits[runtime["target"]]["test"])
    source_loader, scalar_shift_loader, target_loader = tm.make_loaders(
        runtime, splits, batch_size=a.batch_size, num_workers=a.num_workers, seed=seed,
        seq_length=a.seq_length, num_pixels=a.num_pixels, max_shift_aug=a.max_shift_aug,
        shift_aug_p=a.shift_aug_p, with_shift_aug=a.with_shift_aug,
    )
    alpha_loader = None; train_scan = None
    if nonlinear_enabled:
        # The extra loader and scan are never constructed in exact alpha-zero mode.
        loader_rng = _capture_rng()
        try:
            _unused_source, alpha_loader, _unused_target = tm.make_loaders(
                runtime, splits, batch_size=a.batch_size, num_workers=a.num_workers, seed=seed,
                seq_length=a.seq_length, num_pixels=a.num_pixels, max_shift_aug=a.max_shift_aug,
                shift_aug_p=a.shift_aug_p, with_shift_aug=a.with_shift_aug,
            )
        finally:
            _restore_rng(loader_rng)
        train_scan = tm.scan_loader(
            runtime["data_root"], runtime["target"], classes, target_train, runtime,
            batch_size=a.geometry_batch_size, num_workers=a.num_workers, strip_label=True,
        )
    val_loader = tm.scan_loader(
        runtime["data_root"], runtime["source"], classes, source_val, runtime,
        batch_size=a.geometry_batch_size, num_workers=a.num_workers, strip_label=False,
    )

    named = dict(student.named_parameters()); params = [named[n] for n in policy.trainable_parameter_names]
    optimizer = torch.optim.Adam(params, lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=a.epochs * a.steps_per_epoch, eta_min=0.0
    )
    criterion = FocalLoss(gamma=a.focal_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=(a.amp and device.type == "cuda"))
    (out / "README_中文说明.md").write_text(
        "# TimeMatch 锚定的 Nonlinear Phase Search\n\n"
        f"模式：`{mode}`；任务：`{runtime['source']} -> {runtime['target']}`；seed=1，fold=0。\n\n"
        "训练主路径不读取 target-train true labels。scalar shift 始终由原 TimeMatch IS/AM 路径决定；"
        "非线性模式只在固定当前 delta 后通过 AM 搜索 alpha。rho_dom 由 Stage-1 冻结趋势几何在 bootstrap 一次性估计，之后不更新、不反向传播。"
        "checkpoint selection 仅使用 source-val labels；target labels 只在全部训练完成后的最终 target-test 中读取。\n",
        encoding="utf-8",
    )
    write_json(out / "00_config.json", {
        "protocol": PROTOCOL, "mode": mode, "source": runtime["source"], "target": runtime["target"], "seed": seed, "fold": 0,
        "stage1_checkpoint": str(a.model_checkpoint.resolve()), "stage1_checkpoint_sha256": sha256_file(a.model_checkpoint),
        "epochs": a.epochs, "steps_per_epoch": a.steps_per_epoch, "batch_size": a.batch_size,
        "optimizer": {"name": "Adam", "lr": a.lr, "weight_decay": a.weight_decay},
        "scheduler": {"name": "CosineAnnealingLR", "T_max": a.epochs * a.steps_per_epoch},
        "loss": {"source": "FocalLoss", "target": "FocalLoss", "gamma": a.focal_gamma, "trade_off": a.trade_off},
        "ema": {"implementation": "official-TimeMatch-compatible state_dict-wide EMA", "decay": a.ema_decay, "teacher_eval": True},
        "pseudo_threshold": a.pseudo_threshold, "alpha_candidates": list(alpha_bank),
        "phase_definition": "u + delta/365 + alpha*(rho_dom(u)-u)",
        "alpha_zero_semantics": "exact scalar TimeMatch temporal translation",
        "scalar_search": "TimeMatch IS bootstrap then AM each epoch; alpha fixed to zero during delta search",
        "alpha_search": "AM only at fixed current delta; no Cartesian delta x alpha search",
        "rho_dom": "bootstrap-only frozen Stage-1 trend prototype registration" if nonlinear_enabled else "identity",
        "source_phase": "fixed inverse of initial target-to-source data Phase",
        "target_student_time": "native/strong augmentation only", "target_truth_train_used": False,
        "trainable_parameter_names": list(policy.trainable_parameter_names),
    })

    # Match official TimeMatch loader/RNG ordering: instantiate source/target
    # training iterators before the shift bootstrap.  Extra nonlinear scans
    # below restore parent RNG and therefore cannot change these streams.
    source_iter = iter(source_loader); target_iter = iter(target_loader)

    # Original TimeMatch bootstrap: IS -> initial pseudo class distribution -> first AM delta0.
    delta_is = tm.estimate_shift(
        teacher, scalar_shift_loader, device, min_shift=-a.max_temporal_shift, max_shift=a.max_temporal_shift,
        sample_size=a.sample_size, estimator="IS", class_distr=None, shift_chunk=a.shift_chunk,
        output=out, epoch_tag="initial_IS",
    )
    if delta_is >= 0: min_shift, max_shift = 0, a.max_temporal_shift
    else: min_shift, max_shift = -a.max_temporal_shift, 0
    initial_pseudo = tm.pseudo_labels_from_weak_loader(teacher, scalar_shift_loader, device, delta_is)
    initial_class_distr = class_distribution(initial_pseudo, num_classes)
    delta0 = tm.estimate_shift(
        teacher, scalar_shift_loader, device, min_shift=min_shift, max_shift=max_shift,
        sample_size=a.sample_size, estimator="AM", class_distr=initial_class_distr, shift_chunk=a.shift_chunk,
        output=out, epoch_tag="initial_AM_delta0",
    )
    min_shift, max_shift = min(delta0, 0), max(0, delta0)
    pseudo_delta0 = None
    if nonlinear_enabled:
        scan_rng = _capture_rng()
        try:
            pseudo_delta0 = _scan_phase(
                teacher, train_scan, device, mode="ALPHA_ZERO", delta_days=delta0, alpha=0.0,
                rho_dom=canonical_grid(128), allow_labels=False,
            )
        finally:
            _restore_rng(scan_rng)
        rho_dom, residual_info, geometry_guard = _build_frozen_residual(
            runtime=runtime, semantic_model=student, source_train=source_train,
            target_train=target_train, pseudo_scan=pseudo_delta0, delta0=delta0, threshold=a.pseudo_threshold,
            device=device, batch_size=a.geometry_batch_size, num_workers=a.num_workers,
            registration_workers=a.registration_workers, output=out,
            stage1_checkpoint_sha256=sha256_file(a.model_checkpoint),
        )
    else:
        rho_dom = canonical_grid(128); residual_info = {"valid_classes": [], "fallback_reason": "identity_by_mode"}
        geometry_guard = None
        torch.save({"rho_dom": rho_dom, "residual": rho_dom-rho_dom, "fallback_reason": "identity_by_mode",
                    "target_truth_used": False}, out / "bootstrap_domain_residual.pt")

    if nonlinear_enabled:
        alpha0, alpha0_rows, alpha0_summary = _alpha_select(
            teacher, alpha_loader, device, delta_days=delta0, alphas=alpha_bank, rho_dom=rho_dom,
            class_distr=initial_class_distr, sample_size=a.sample_size, output=out,
            epoch_tag="initial_alpha0", pseudo_threshold=a.pseudo_threshold,
        )
    else:
        alpha0 = 0.0; alpha0_rows = []
        alpha0_summary = {"best_am": None, "second_best_am": None, "margin": None}

    source_to_target_days = -int(delta0)
    initial_data_phase = candidate_phase_grid(
        delta_days=delta0, alpha=alpha0, rho_dom=rho_dom, time_scale_days=float(runtime.get("time_scale", 365.0))
    )
    source_phase = inverse_phase_on_canonical_grid(initial_data_phase)
    if not nonlinear_enabled:
        expected = canonical_grid(source_phase.numel()) + source_to_target_days / float(runtime.get("time_scale", 365.0))
        if not torch.allclose(source_phase, expected, atol=1e-12, rtol=0):
            raise RuntimeError("alpha=0 fixed source inverse is not the exact TimeMatch opposite scalar shift")
    torch.save({
        "delta_is_days": int(delta_is), "delta0_days": int(delta0), "initial_class_distribution": initial_class_distr,
        "alpha0": float(alpha0), "rho_dom": rho_dom, "source_phase": source_phase,
        "source_to_target_days": int(source_to_target_days), "residual_info": residual_info,
        "source_phase_frozen": True, "rho_dom_frozen": True, "target_truth_used": False,
    }, out / "bootstrap_state.pt")
    print(
        f"TMNP_BOOTSTRAP|mode={mode}|delta_IS={delta_is:+d}|delta0={delta0:+d}|alpha0={alpha0:.2f}|"
        f"source_scalar={source_to_target_days:+d}|rho_fallback={residual_info.get('fallback_reason') or 'none'}", flush=True,
    )

    # Epoch 1 reuses the already-computed delta0/alpha0.  Epoch 2+ refresh scalar delta first, then alpha.
    all_pseudo_labels = initial_pseudo.copy()
    current_delta = int(delta0); current_alpha = float(alpha0)
    epoch_rows = []; val_rows = []; best_f1 = -float("inf")
    for epoch in range(1, a.epochs + 1):
        estimated = class_distribution(all_pseudo_labels, num_classes)
        if epoch > 1:
            current_delta = tm.estimate_shift(
                teacher, scalar_shift_loader, device, min_shift=min_shift, max_shift=max_shift,
                sample_size=a.sample_size, estimator="AM", class_distr=estimated, shift_chunk=a.shift_chunk,
                output=out, epoch_tag=f"epoch_{epoch:03d}_delta_AM",
            )
            if nonlinear_enabled:
                current_alpha, _, alpha_summary = _alpha_select(
                    teacher, alpha_loader, device, delta_days=current_delta, alphas=alpha_bank, rho_dom=rho_dom,
                    class_distr=estimated, sample_size=a.sample_size, output=out,
                    epoch_tag=f"epoch_{epoch:03d}_alpha_AM", pseudo_threshold=a.pseudo_threshold,
                )
            else:
                current_alpha = 0.0
                alpha_summary = {"best_am": None, "second_best_am": None, "margin": None}
        else:
            alpha_summary = alpha0_summary
        student.train(); teacher.eval(); epoch_pseudo = []
        meters = defaultdict(float); meters["steps"] = 0
        for _step in range(a.steps_per_epoch):
            try: sraw = next(source_iter)
            except StopIteration: source_iter = iter(source_loader); sraw = next(source_iter)
            try: weak_raw, strong_raw = next(target_iter)
            except StopIteration: target_iter = iter(target_loader); weak_raw, strong_raw = next(target_iter)
            if FORBIDDEN_TARGET_KEYS.intersection(weak_raw.keys()) or FORBIDDEN_TARGET_KEYS.intersection(strong_raw.keys()):
                raise RuntimeError("target truth leaked into TimeMatch nonlinear Phase training")
            s = tm.move(sraw, device); weak = tm.move(weak_raw, device); strong = tm.move(strong_raw, device)
            amp_on = bool(a.amp and device.type == "cuda")
            teacher.eval()
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_on):
                tout = _teacher_forward(
                    teacher, weak, mode=mode, delta_days=current_delta, alpha=current_alpha, rho_dom=rho_dom
                )
                tpost = torch.softmax(tout.logits.float(), dim=1); conf, pseudo = tpost.max(1); mask = conf > a.pseudo_threshold
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_on):
                sout = _source_forward(
                    student, s, mode=mode, source_to_target_days=source_to_target_days, source_phase=source_phase
                )
                source_loss = criterion(sout.logits, s["label"].long()); target_loss = source_loss.sum() * 0.0
                if int(mask.sum().item()) >= 2:
                    subset = {k: (v[mask] if isinstance(v, Tensor) and v.ndim > 0 and v.shape[0] == mask.shape[0] else v)
                              for k, v in strong.items()}
                    uout = tm.forward_scalar(
                        student, subset, shift_days=0.0, augmentation_shift_days=subset.get("temporal_aug_shift_days")
                    )
                    target_loss = criterion(uout.logits, pseudo[mask])
                total = source_loss + a.trade_off * target_loss
            scaler.scale(total).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
            official_timematch_ema_update(student, teacher, a.ema_decay)
            pseudo_cpu = pseudo.detach().cpu()
            conf_cpu = conf.detach().cpu()
            epoch_pseudo.extend(pseudo_cpu.tolist())
            meters["source_loss"] += float(source_loss.detach()); meters["target_loss"] += float(target_loss.detach())
            meters["total_loss"] += float(total.detach()); meters["confident"] += int(mask.sum().item()); meters["steps"] += 1
            meters["pseudo_count"] += int(pseudo_cpu.numel()); meters["confidence_sum"] += float(conf_cpu.sum().item())
            for cid, count in enumerate(torch.bincount(pseudo_cpu, minlength=num_classes).tolist()):
                meters[f"pred_count_{cid}"] += int(count)
        all_pseudo_labels = np.asarray(epoch_pseudo, dtype=np.int64)
        nsteps = int(meters["steps"])
        row = {
            "epoch": epoch, "mode": mode, "delta_days": current_delta, "alpha": current_alpha,
            "source_loss": meters["source_loss"] / nsteps, "target_loss": meters["target_loss"] / nsteps,
            "total_loss": meters["total_loss"] / nsteps, "confident_seen": int(meters["confident"]),
            "mean_confident_per_step": meters["confident"] / nsteps,
            "teacher_pseudo_confidence_mean": meters["confidence_sum"] / max(1, meters["pseudo_count"]),
            "teacher_confidence_coverage": meters["confident"] / max(1, meters["pseudo_count"]),
            "predicted_class_counts": ",".join(str(int(meters[f"pred_count_{cid}"])) for cid in range(num_classes)),
            "alpha_best_am": alpha_summary.get("best_am"), "alpha_second_best_am": alpha_summary.get("second_best_am"),
            "alpha_am_margin": alpha_summary.get("margin"),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        epoch_rows.append(row); tm.write_csv(out / "epoch_training.csv", epoch_rows)
        if geometry_guard is not None:
            geometry_guard.assert_frozen()
            if module_state_hash(geometry_guard.geometry_pse) != residual_info["geometry_state_hash"]:
                raise RuntimeError("geometry_pse state changed across a Stage2 epoch")
        student_f1 = _save_and_print_validation(
            out, val_rows, epoch=epoch, mode=mode, student=student, teacher=teacher, val_loader=val_loader,
            device=device, delta_days=current_delta, alpha=current_alpha, source_phase=source_phase,
            pseudo_threshold=a.pseudo_threshold, num_classes=num_classes,
        )
        print(
            f"TMNP_EPOCH|mode={mode}|epoch={epoch}/{a.epochs}|delta={current_delta:+d}|alpha={current_alpha:.2f}|"
            f"source_loss={row['source_loss']:.6f}|target_loss={row['target_loss']:.6f}|total_loss={row['total_loss']:.6f}|"
            f"pseudo_conf_mean={row['teacher_pseudo_confidence_mean']:.6f}|coverage={row['teacher_confidence_coverage']:.6f}|"
            f"class_counts={row['predicted_class_counts']}|alpha_am_margin={row['alpha_am_margin']}|lr={row['lr']:.8g}", flush=True,
        )
        state = {
            "protocol": PROTOCOL, "mode": mode, "epoch": epoch, "student_state_dict": {k:v.detach().cpu() for k,v in student.state_dict().items()},
            "teacher_state_dict": {k:v.detach().cpu() for k,v in teacher.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "current_delta_days": current_delta, "current_alpha": current_alpha,
            "source_to_target_days": source_to_target_days, "source_phase": source_phase, "rho_dom": rho_dom,
            "target_truth_used_for_training": False,
        }
        if student_f1 > best_f1:
            best_f1 = student_f1; torch.save(state, out / "best_source_val.pt")
        torch.save(state, out / "last.pt")
    try:
        import matplotlib.pyplot as plt
        figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
        xs = [int(r["epoch"]) for r in epoch_rows]
        plt.figure(); plt.plot(xs, [int(r["delta_days"]) for r in epoch_rows], marker="o"); plt.xlabel("epoch"); plt.ylabel("TimeMatch delta (days)"); plt.tight_layout(); plt.savefig(figdir / "delta_trajectory.png", dpi=180); plt.close()
        plt.figure(); plt.plot(xs, [float(r["alpha"]) for r in epoch_rows], marker="o"); plt.xlabel("epoch"); plt.ylabel("selected alpha"); plt.ylim(-0.05, 1.05); plt.tight_layout(); plt.savefig(figdir / "alpha_trajectory.png", dpi=180); plt.close()
        student_val = [r for r in val_rows if r["role"] == "student"]
        plt.figure(); plt.plot([int(r["epoch"]) for r in student_val], [float(r["macro_f1"]) for r in student_val], marker="o"); plt.xlabel("epoch"); plt.ylabel("source-val Macro-F1"); plt.tight_layout(); plt.savefig(figdir / "source_val_macro_f1.png", dpi=180); plt.close()
    except Exception as exc:
        print(f"TMNP_PLOT_WARNING|type={type(exc).__name__}|message={exc}", flush=True)
    # Match upstream TimeMatch output_student=True semantics for the primary
    # 20-epoch result: save and test the final Student, not a target-test-selected model.
    torch.save({"state_dict": {k:v.detach().cpu() for k,v in student.state_dict().items()}}, out / "model.pt")
    final_state = torch.load(out / "model.pt", map_location="cpu", weights_only=False)["state_dict"]
    student.load_state_dict(final_state); student.to(device)
    test_metrics = _evaluate_final_target_test(
        out, student=student, runtime=runtime, target_test=target_test, classes=classes,
        device=device, num_workers=a.num_workers,
    )
    write_json(out / "run_summary.json", {
        "protocol": PROTOCOL, "mode": mode, "completed_epochs": a.epochs, "best_source_val_macro_f1": best_f1,
        "final_delta_days": current_delta, "final_alpha": current_alpha, "rho_dom_frozen": True,
        "registration_refresh_count": 1 if nonlinear_enabled else 0,
        "primary_checkpoint_semantics": "final_student_after_epoch_20__TimeMatch_output_student_true",
        "test": test_metrics, "target_truth_used_for_training": False,
    })
    print(
        f"TMNP_COMPLETE|mode={mode}|epochs={a.epochs}|best_source_val_macro_f1={best_f1:.6f}|"
        f"test_macro_f1={test_metrics['macro_f1']:.6f}|final_delta={current_delta:+d}|"
        f"final_alpha={current_alpha:.2f}|output={out}", flush=True,
    )


if __name__ == "__main__":
    main()
