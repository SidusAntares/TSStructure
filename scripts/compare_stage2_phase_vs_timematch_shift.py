#!/usr/bin/env python3
"""First-layer diagnostic: no shift vs TimeMatch-style scalar shift vs Domain Phase.

This script is a no-gradient, no-training diagnostic.  It answers one narrow
question before any Stage-2 optimization changes are considered:

    Does the current Stage-1 TSStructure representation benefit more from
    (a) the TimeMatch-style global integer calendar shift or
    (b) the confirmed nonlinear Domain Phase gamma?

The TimeMatch-style scalar shift follows the official TimeMatch *initial shift
estimation* semantics:

* candidate target-to-source shifts are integer days in [-60, 60] by default;
* the shift is added directly to target observation dates before the temporal
  encoder;
* the source-trained model is evaluated on 100 target batches by default;
* the initial shift is the candidate maximizing Inception Score.
* TimeMatch allocates a temporal-shift buffer around its positional table; the
  Scalar audit therefore permits Time2Vec extrapolation beyond [0,1] without
  clamping, but only inside this diagnostic script.

In TSStructure the PSE/decomposition backbone is computed once for each target
batch.  A global additive shift does not change pairwise temporal differences,
so the scalar comparison changes only the normalized observation positions
passed to Time2Vec/LTAE.  This also avoids the backbone's physical-date [0,365]
validation from incorrectly clipping a TimeMatch calendar translation.

Target true labels are NEVER used to choose the TimeMatch-style shift.  They are
used only for post-hoc oracle diagnostics (accuracy/F1 curves and per-class
alignment summaries).

This script depends on ``scripts/visualize_stage2_phase_alignment.py`` from the
previous Phase-alignment diagnostic so that model loading, fold reconstruction,
Domain Phase application and source-prototype distance semantics remain exactly
consistent across the two audits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, REPOSITORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import visualize_stage2_phase_alignment as phasevis
from dataset import PixelSetData, worker_init_fn
from methods.structure_da import phase_visualization_protocol as visproto
from transforms import Normalize, RandomSamplePixels, ToTensor


TIMEMATCH_REPOSITORY = "https://github.com/jnyborg/timematch"
TIMEMATCH_PAPER = "https://doi.org/10.1016/j.isprsjprs.2022.04.018"


def _resolve_unbounded_time2vec_inputs(
    encoder,
    positions: Tensor,
    time_mask: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Resolve Time2Vec inputs without the production [0,1] range gate."""
    if not isinstance(positions, Tensor):
        raise ValueError("positions must be a torch.Tensor")
    if not positions.is_floating_point() or positions.is_complex():
        raise ValueError("positions must be a real floating-point tensor")
    if positions.ndim == 1:
        sequence_length = positions.shape[0]
        if isinstance(time_mask, Tensor) and time_mask.ndim == 2:
            batch_size = time_mask.shape[0]
        else:
            batch_size = 1
        resolved_positions = positions.unsqueeze(0).expand(batch_size, -1)
    elif positions.ndim == 2:
        batch_size, sequence_length = positions.shape
        resolved_positions = positions
    else:
        raise ValueError("positions must have shape [L] or [B, L]")

    if time_mask is None:
        resolved_mask = torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=positions.device,
        )
    else:
        if not isinstance(time_mask, Tensor):
            raise ValueError("time_mask must be a torch.Tensor or None")
        if time_mask.ndim == 1:
            if time_mask.shape != (sequence_length,):
                raise ValueError("time_mask must have shape [L] or [B, L]")
            resolved_mask = time_mask.unsqueeze(0).expand(batch_size, -1)
        elif time_mask.ndim == 2:
            if time_mask.shape != (batch_size, sequence_length):
                raise ValueError("time_mask must have shape [L] or [B, L]")
            resolved_mask = time_mask
        else:
            raise ValueError("time_mask must have shape [L] or [B, L]")
        if time_mask.is_complex() or (
            time_mask.dtype != torch.bool
            and (
                not torch.isfinite(time_mask).all().item()
                or not torch.all((time_mask == 0) | (time_mask == 1)).item()
            )
        ):
            raise ValueError("time_mask must contain only finite 0/1 values")
        resolved_mask = resolved_mask.to(device=positions.device, dtype=torch.bool)

    if resolved_positions.device != encoder.linear_weight.device:
        raise ValueError("positions device must match module parameters")
    if resolved_positions.dtype != encoder.linear_weight.dtype:
        raise ValueError("positions dtype must match module parameters")
    valid_positions = resolved_positions[resolved_mask]
    if not torch.isfinite(valid_positions).all().item():
        raise ValueError("valid positions must be finite")
    return resolved_positions, resolved_mask


def _unbounded_time2vec_forward(
    encoder,
    positions: Tensor,
    *,
    time_mask: Optional[Tensor] = None,
) -> Tensor:
    """Evaluate the learned Time2Vec formula beyond [0,1] for this audit only."""
    resolved_positions, resolved_mask = _resolve_unbounded_time2vec_inputs(
        encoder, positions, time_mask
    )
    safe_positions = torch.where(
        resolved_mask,
        resolved_positions,
        torch.zeros_like(resolved_positions),
    )
    normalized = (safe_positions - encoder.time_reference) / encoder.time_scale
    linear = encoder.linear_weight * normalized + encoder.linear_bias
    periodic = torch.sin(
        normalized.unsqueeze(-1) * encoder.frequencies + encoder.phase
    )
    encoding = torch.cat([linear.unsqueeze(-1), periodic], dim=-1)
    encoding = torch.where(
        resolved_mask.unsqueeze(-1),
        encoding,
        torch.zeros_like(encoding),
    )
    if not torch.isfinite(encoding).all().item():
        raise ValueError("time encoding must contain only finite values")
    return encoding


@contextmanager
def _timematch_time_extrapolation(model):
    """Temporarily enable TimeMatch-like out-of-year timestamps.

    Official TimeMatch adds integer shifts directly to day indices and its
    positional table contains an explicit temporal-shift buffer. Production
    TSStructure instead rejects and clamps normalized positions outside [0,1].
    Clamping would not represent a global translation, so only the Scalar audit
    view temporarily evaluates the same learned Time2Vec formula without that
    boundary restriction. Model parameters are untouched.
    """
    encoder = visproto.phase_only_time_encoder(model)
    required = (
        "linear_weight",
        "linear_bias",
        "frequencies",
        "phase",
        "time_reference",
        "time_scale",
    )
    if not all(hasattr(encoder, name) for name in required):
        # Fixed analytic encoders already support out-of-year positions and need
        # no monkey patch.  This keeps the same scalar-shift scanner reusable
        # for the TimeMatch fixed-PE Stage-1 ablation.
        yield
        return

    had_instance_forward = "forward" in encoder.__dict__
    previous_forward = encoder.__dict__.get("forward")

    def _forward(this, positions: Tensor, *, time_mask: Optional[Tensor] = None) -> Tensor:
        return _unbounded_time2vec_forward(this, positions, time_mask=time_mask)

    encoder.forward = types.MethodType(_forward, encoder)
    try:
        yield
    finally:
        if had_instance_forward:
            encoder.forward = previous_forward
        else:
            del encoder.forward


def _scalar_positions(backbone, shift_days: float, time_scale_days: float) -> Tensor:
    """Add a TimeMatch-style calendar shift to valid normalized positions.

    ``backbone.normalized_positions`` are already in the model's [physical
    day]/time_scale coordinate.  TimeMatch adds ``shift_days`` directly to raw
    day-of-year positions before its temporal encoder, hence the equivalent
    TSStructure temporal override is ``t_norm + shift_days / time_scale``.

    Do not clamp to [0,1]: the original TimeMatch shift is an unrestricted
    calendar translation and can move dates outside the original annual range.
    Padding positions remain zero and are masked by LTAE.
    """
    delta = float(shift_days) / float(time_scale_days)
    shifted = backbone.normalized_positions + delta
    return torch.where(backbone.time_mask, shifted, torch.zeros_like(shifted))


def _macro_f1(labels: np.ndarray, predictions: np.ndarray, class_ids: Sequence[int]) -> float:
    scores: List[float] = []
    for class_id in class_ids:
        y_true = labels == int(class_id)
        y_pred = predictions == int(class_id)
        tp = float(np.logical_and(y_true, y_pred).sum())
        fp = float(np.logical_and(~y_true, y_pred).sum())
        fn = float(np.logical_and(y_true, ~y_pred).sum())
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
    return float(np.mean(scores)) if scores else float("nan")


def _random_unlabeled_parcels(dataset: PixelSetData, count: int, seed: int) -> np.ndarray:
    parcels = np.asarray(dataset.get_parcel_indices(), dtype=np.int64)
    if parcels.ndim != 1 or parcels.size == 0:
        raise ValueError("target train dataset contains no parcel indices")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(parcels.size)
    return parcels[order[: min(int(count), parcels.size)]]


def _timematch_estimation_loader(
    data_root: str,
    domain: str,
    classes: Sequence[str],
    train_indices: set,
    *,
    closed_set: bool,
    combine_spring_and_winter: bool,
    time_coordinate_mode: str,
    batch_size: int,
    num_workers: int,
    num_pixels: int,
    seed: int,
) -> DataLoader:
    """Build the weak-view loader used only for TimeMatch-style shift scanning.

    Official TimeMatch uses RandomSamplePixels -> Normalize -> ToTensor for the
    no-augmentation shift-estimation loader.  We reproduce that transform and a
    shuffled target-train loader here.  A seeded generator makes the diagnostic
    archive reproducible.
    """
    transform = transforms.Compose([
        RandomSamplePixels(int(num_pixels)),
        Normalize(),
        ToTensor(),
    ])
    dataset = PixelSetData(
        data_root=data_root,
        dataset_name=domain,
        classes=list(classes),
        transform=transform,
        indices=train_indices,
        with_extra=False,
        closed_set=closed_set,
        combine_spring_and_winter=combine_spring_and_winter,
        time_coordinate_mode=time_coordinate_mode,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


@torch.no_grad()
def _estimate_timematch_scalar_shift(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    min_shift: int,
    max_shift: int,
    max_batches: int,
    num_classes: int,
    time_scale_days: float,
) -> dict:
    """Reproduce TimeMatch's initial IS shift selection on the TSStructure model."""
    shifts = list(range(int(min_shift), int(max_shift) + 1))
    if not shifts:
        raise ValueError("empty scalar-shift search range")

    model.eval()
    probability_batches: List[Tensor] = []
    label_batches: List[Tensor] = []
    processed_batches = 0

    for raw_batch in loader:
        if processed_batches >= int(max_batches):
            break
        batch = phasevis._move_batch(raw_batch, device)
        backbone = model.forward_backbone(
            batch["pixels"],
            batch["valid_pixels"],
            batch["positions"],
            batch.get("extra"),
            time_mask=batch.get("time_mask"),
        )
        shift_probs: List[Tensor] = []
        for shift_days in shifts:
            shifted_positions = _scalar_positions(
                backbone, shift_days, time_scale_days
            )
            with _timematch_time_extrapolation(model):
                output = model.forward_from_backbone(
                    backbone,
                    batch["positions"],
                    batch.get("extra"),
                    temporal_positions_override=shifted_positions,
                    return_geometry=False,
                )
            shift_probs.append(torch.softmax(output.logits.float(), dim=-1).cpu())
        probability_batches.append(torch.stack(shift_probs, dim=1))  # [B,S,C]
        label_batches.append(batch["label"].detach().cpu().long())
        processed_batches += 1
        print(
            "TIMEMATCH_SHIFT_SCAN_PROGRESS|"
            f"batch={processed_batches}/{max_batches}|"
            f"samples={sum(int(v.shape[0]) for v in label_batches)}",
            flush=True,
        )

    if not probability_batches:
        raise RuntimeError("TimeMatch-style shift scan processed zero target batches")

    probabilities = torch.cat(probability_batches, dim=0).numpy().astype(np.float64)
    labels = torch.cat(label_batches, dim=0).numpy().astype(np.int64)
    # Official TimeMatch notation: p(y|x,shift), then p(y|shift)=mean_x p(y|x,shift).
    p_yx = probabilities
    p_y = probabilities.mean(axis=0)
    eps = 1e-5
    inception_scores = np.mean(
        np.sum(p_yx * (np.log(p_yx + eps) - np.log(p_y[None, :, :] + eps)), axis=2),
        axis=0,
    )
    entropy_scores = -np.mean(
        np.sum(p_yx * np.log(p_yx + eps), axis=2), axis=0
    )
    predictions = probabilities.argmax(axis=2)

    all_class_ids = tuple(range(int(num_classes)))
    accuracy_scores = np.asarray(
        [(labels == predictions[:, index]).mean() for index in range(len(shifts))],
        dtype=np.float64,
    )
    macro_f1_scores = np.asarray(
        [
            _macro_f1(labels, predictions[:, index], all_class_ids)
            for index in range(len(shifts))
        ],
        dtype=np.float64,
    )

    best_is_index = int(np.argmax(inception_scores))
    best_acc_index = int(np.argmax(accuracy_scores))
    best_f1_index = int(np.argmax(macro_f1_scores))
    rows = [
        {
            "shift_days": int(shift),
            "inception_score": float(inception_scores[index]),
            "entropy_score": float(entropy_scores[index]),
            "oracle_accuracy": float(accuracy_scores[index]),
            "oracle_macro_f1": float(macro_f1_scores[index]),
            "selected_by_inception_score": bool(index == best_is_index),
            "oracle_best_accuracy": bool(index == best_acc_index),
            "oracle_best_macro_f1": bool(index == best_f1_index),
        }
        for index, shift in enumerate(shifts)
    ]
    return {
        "rows": rows,
        "selected_shift_days": int(shifts[best_is_index]),
        "selected_inception_score": float(inception_scores[best_is_index]),
        "selected_oracle_accuracy": float(accuracy_scores[best_is_index]),
        "selected_oracle_macro_f1": float(macro_f1_scores[best_is_index]),
        "oracle_best_accuracy_shift_days": int(shifts[best_acc_index]),
        "oracle_best_accuracy": float(accuracy_scores[best_acc_index]),
        "oracle_best_macro_f1_shift_days": int(shifts[best_f1_index]),
        "oracle_best_macro_f1": float(macro_f1_scores[best_f1_index]),
        "num_samples": int(labels.size),
        "num_batches": int(processed_batches),
        "labels": labels,
        "predictions": predictions,
    }


def _plot_shift_scan(path: Path, result: dict, dpi: int) -> None:
    rows = result["rows"]
    shifts = np.asarray([row["shift_days"] for row in rows], dtype=np.float64)
    inception = np.asarray([row["inception_score"] for row in rows], dtype=np.float64)
    accuracy = np.asarray([row["oracle_accuracy"] for row in rows], dtype=np.float64)
    macro_f1 = np.asarray([row["oracle_macro_f1"] for row in rows], dtype=np.float64)
    selected = float(result["selected_shift_days"])
    oracle_acc = float(result["oracle_best_accuracy_shift_days"])
    oracle_f1 = float(result["oracle_best_macro_f1_shift_days"])

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    axes[0].plot(shifts, inception, linewidth=1.8)
    axes[0].axvline(selected, linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Inception Score")
    axes[0].set_title(
        f"TimeMatch-style scalar shift scan — selected by IS: {int(selected):+d} days"
    )
    axes[0].grid(alpha=0.25)

    axes[1].plot(shifts, accuracy, label="Oracle accuracy", linewidth=1.6)
    axes[1].plot(shifts, macro_f1, label="Oracle Macro-F1", linewidth=1.6)
    axes[1].axvline(selected, linestyle="--", linewidth=1.2, label="IS-selected")
    axes[1].axvline(oracle_acc, linestyle=":", linewidth=1.1, label="Oracle-Acc best")
    axes[1].axvline(oracle_f1, linestyle="-.", linewidth=1.1, label="Oracle-F1 best")
    axes[1].set_xlabel("Target-to-source scalar shift (days; added to target dates)")
    axes[1].set_ylabel("Oracle metric")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _attach_scalar_view(
    model,
    loader: DataLoader,
    records: Dict[int, List[dict]],
    *,
    device: torch.device,
    shift_days: int,
    time_scale_days: float,
) -> None:
    lookup: Dict[Tuple[int, int], dict] = {}
    for class_id, class_records in records.items():
        for record in class_records:
            lookup[(int(class_id), int(record["parcel_index"]))] = record

    attached = 0
    for raw_batch in loader:
        batch = phasevis._move_batch(raw_batch, device)
        backbone = model.forward_backbone(
            batch["pixels"],
            batch["valid_pixels"],
            batch["positions"],
            batch.get("extra"),
            time_mask=batch.get("time_mask"),
        )
        scalar_positions = _scalar_positions(backbone, shift_days, time_scale_days)
        with _timematch_time_extrapolation(model):
            output = model.forward_from_backbone(
                backbone,
                batch["positions"],
                batch.get("extra"),
                temporal_positions_override=scalar_positions,
                return_geometry=False,
            )
        labels = batch["label"].detach().cpu().long()
        parcels = batch["parcel_index"].detach().cpu().long()
        for row in range(len(labels)):
            key = (int(labels[row].item()), int(parcels[row].item()))
            record = lookup.get(key)
            if record is None:
                continue
            record.update(
                {
                    "positions_scalar": scalar_positions[row].detach().cpu(),
                    "fused_repr_scalar": output.fused_repr[row].detach().cpu(),
                    "logits_scalar": output.logits[row].detach().cpu(),
                }
            )
            attached += 1

    expected = sum(len(class_records) for class_records in records.values())
    if attached != expected:
        missing = [
            (class_id, int(record["parcel_index"]))
            for class_id, class_records in records.items()
            for record in class_records
            if "fused_repr_scalar" not in record
        ]
        raise RuntimeError(
            f"scalar view attached {attached}/{expected} records; missing={missing[:10]}"
        )


def _center(records: Sequence[dict], key: str) -> Tensor:
    return torch.stack([record[key].float() for record in records], dim=0).mean(dim=0)


def _three_view_distances(
    records: Sequence[dict],
    *,
    prototype: Tensor,
    base_key: str,
    scalar_key: str,
    phase_key: str,
) -> Tuple[Tensor, Tensor, Tensor, dict]:
    prototype = prototype.detach().cpu().float()
    base = torch.stack([record[base_key].float() for record in records])
    scalar = torch.stack([record[scalar_key].float() for record in records])
    phase = torch.stack([record[phase_key].float() for record in records])
    d0 = torch.linalg.vector_norm(base - prototype, dim=1)
    ds = torch.linalg.vector_norm(scalar - prototype, dim=1)
    dp = torch.linalg.vector_norm(phase - prototype, dim=1)
    base_mean = float(d0.mean().item())
    eps = 1e-12
    stats = {
        "no_shift_mean": base_mean,
        "scalar_mean": float(ds.mean().item()),
        "phase_mean": float(dp.mean().item()),
        "scalar_relative_reduction": float((d0.mean() - ds.mean()).item() / max(base_mean, eps)),
        "phase_relative_reduction": float((d0.mean() - dp.mean()).item() / max(base_mean, eps)),
        "scalar_improvement_rate": float((ds < d0).float().mean().item()),
        "phase_improvement_rate": float((dp < d0).float().mean().item()),
    }
    return d0, ds, dp, stats


def _three_view_pse_metrics(
    source_records: Sequence[dict],
    target_records: Sequence[dict],
    *,
    grid_size: int,
) -> Tuple[dict, List[dict]]:
    source_center, source_support, _ = phasevis._canonical_pse_center(
        source_records, positions_key="positions", grid_size=grid_size
    )
    class_distances = {}
    centers = {}
    for name, position_key in (
        ("no_shift", "positions"),
        ("timematch_scalar", "positions_scalar"),
        ("domain_phase", "positions_after"),
    ):
        center, support, _ = phasevis._canonical_pse_center(
            target_records, positions_key=position_key, grid_size=grid_size
        )
        mse, l2, common = phasevis._pse_integrated_distance(
            source_center, source_support, center, support
        )
        class_distances[name] = {"mse": mse, "l2": l2, "common_grid": common}
        centers[name] = (center, support)

    rows: List[dict] = []
    for record in target_records:
        row = {"parcel_index": int(record["parcel_index"])}
        for name, position_key in (
            ("no_shift", "positions"),
            ("timematch_scalar", "positions_scalar"),
            ("domain_phase", "positions_after"),
        ):
            trajectory, support, _ = phasevis._canonicalize_pse_tokens(
                record, positions_key=position_key, grid_size=grid_size
            )
            mse, l2, common = phasevis._pse_integrated_distance(
                source_center, source_support, trajectory, support
            )
            row[f"pse_mse_{name}"] = mse
            row[f"pse_l2_{name}"] = l2
            row[f"pse_common_grid_{name}"] = common
        rows.append(row)
    return class_distances, rows


def _three_view_classification(records: Sequence[dict], class_id: int) -> dict:
    logits0 = torch.stack([record["logits_before"].float() for record in records])
    logits_s = torch.stack([record["logits_scalar"].float() for record in records])
    logits_p = torch.stack([record["logits_after"].float() for record in records])
    probs0 = torch.softmax(logits0, dim=-1)
    probs_s = torch.softmax(logits_s, dim=-1)
    probs_p = torch.softmax(logits_p, dim=-1)
    pred0 = logits0.argmax(dim=-1)
    pred_s = logits_s.argmax(dim=-1)
    pred_p = logits_p.argmax(dim=-1)
    return {
        "prob_no": probs0[:, class_id],
        "prob_scalar": probs_s[:, class_id],
        "prob_phase": probs_p[:, class_id],
        "pred_no": pred0,
        "pred_scalar": pred_s,
        "pred_phase": pred_p,
        "true_probability_no_shift_mean": float(probs0[:, class_id].mean().item()),
        "true_probability_scalar_mean": float(probs_s[:, class_id].mean().item()),
        "true_probability_phase_mean": float(probs_p[:, class_id].mean().item()),
        "oracle_accuracy_no_shift": float((pred0 == class_id).float().mean().item()),
        "oracle_accuracy_scalar": float((pred_s == class_id).float().mean().item()),
        "oracle_accuracy_phase": float((pred_p == class_id).float().mean().item()),
    }


def _plot_distance_comparison(
    path: Path,
    *,
    title: str,
    metrics: Sequence[Tuple[str, Tensor, Tensor, Tensor]],
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(metrics), 2, figsize=(10.8, 4.0 * len(metrics)), squeeze=False)
    for row, (name, base, scalar, phase) in enumerate(metrics):
        for col, (view_name, after) in enumerate((("TimeMatch scalar", scalar), ("Domain Phase", phase))):
            ax = axes[row, col]
            x = base.detach().cpu().numpy()
            y = after.detach().cpu().numpy()
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            pad = max((hi - lo) * 0.08, 1e-6)
            ax.scatter(x, y, s=22, alpha=0.75)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.0)
            rate = float((after < base).float().mean().item())
            ax.set_title(
                f"{name} — {view_name}\n"
                f"mean {float(base.mean()):.3f} → {float(after.mean()):.3f}, improve={rate:.1%}"
            )
            ax.set_xlabel("No shift distance")
            ax.set_ylabel(f"{view_name} distance")
            ax.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_probability_comparison(
    path: Path,
    *,
    title: str,
    base: Tensor,
    scalar: Tensor,
    phase: Tensor,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, (view_name, after) in zip(
        axes, (("TimeMatch scalar", scalar), ("Domain Phase", phase))
    ):
        x = base.detach().cpu().numpy()
        y = after.detach().cpu().numpy()
        ax.scatter(x, y, s=24, alpha=0.75)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
        rate = float((after > base).float().mean().item())
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("No shift true-class probability")
        ax.set_ylabel(f"{view_name} true-class probability")
        ax.set_title(
            f"{view_name}\nmean {float(base.mean()):.3f} → {float(after.mean()):.3f}, increased={rate:.1%}"
        )
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _global_summary(sample_rows: Sequence[dict], requested_classes: Sequence[int]) -> dict:
    labels = np.asarray([row["class_id"] for row in sample_rows], dtype=np.int64)
    pred0 = np.asarray([row["prediction_no_shift"] for row in sample_rows], dtype=np.int64)
    preds = np.asarray([row["prediction_scalar"] for row in sample_rows], dtype=np.int64)
    predp = np.asarray([row["prediction_phase"] for row in sample_rows], dtype=np.int64)

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in sample_rows]))

    return {
        "num_samples": int(len(sample_rows)),
        "visualized_classes": [int(v) for v in requested_classes],
        "pse_l2_mean": {
            "no_shift": mean("pse_l2_no_shift"),
            "timematch_scalar": mean("pse_l2_scalar"),
            "domain_phase": mean("pse_l2_phase"),
        },
        "fused_distance_mean": {
            "no_shift": mean("fused_distance_no_shift"),
            "timematch_scalar": mean("fused_distance_scalar"),
            "domain_phase": mean("fused_distance_phase"),
        },
        "true_class_probability_mean": {
            "no_shift": mean("true_probability_no_shift"),
            "timematch_scalar": mean("true_probability_scalar"),
            "domain_phase": mean("true_probability_phase"),
        },
        "oracle_accuracy": {
            "no_shift": float((pred0 == labels).mean()),
            "timematch_scalar": float((preds == labels).mean()),
            "domain_phase": float((predp == labels).mean()),
        },
        "oracle_macro_f1_on_visualized_classes": {
            "no_shift": _macro_f1(labels, pred0, requested_classes),
            "timematch_scalar": _macro_f1(labels, preds, requested_classes),
            "domain_phase": _macro_f1(labels, predp, requested_classes),
        },
    }


def _write_readme(
    path: Path,
    *,
    checkpoint: Path,
    source: str,
    target: str,
    shift_result: dict,
    global_summary: dict,
    requested_classes: Sequence[int],
) -> None:
    selected = int(shift_result["selected_shift_days"])
    lines = [
        "# 第一层时间校正对照实验：No Shift vs TimeMatch Scalar vs Domain Phase",
        "",
        "## 1. 这个目录在回答什么",
        "",
        "本实验只做前向诊断，不训练模型、不更新参数、不重新估计 Domain Phase。TimeMatch scalar shift 只在 target-train 上无监督选择；正式 No/Scalar/Phase 对照使用 held-out source-test / target-test。目的只有一个：在同一个 Stage-1 TSStructure 模型上，比较最基础的 TimeMatch 式整段时间平移和当前 nonlinear Domain Phase，谁更能改善 target→source 的同类表示对齐。",
        "",
        "三种 target 视图：",
        "",
        "1. `No shift`：使用目标域原始观测日期。",
        f"2. `TimeMatch scalar`：所有目标观测日期统一加 `{selected:+d}` 天。该 shift 由 Inception Score 自动选择，不使用 target label。",
        "3. `Domain Phase`：使用 checkpoint 中已经 confirmed 的 Domain Phase，目标 LTAE 时间位置使用 `gamma^{-1}(t_target)`。",
        "",
        "## 2. TimeMatch scalar 是怎样得到的",
        "",
        "这里复现 TimeMatch 官方代码的**初始 temporal-shift estimation**语义：",
        "",
        "- 枚举整数天 shift；默认范围 `[-60, 60]`；",
        "- 对每个候选 shift，将同一个 target batch 的 observation dates 整体加上该 shift，再送入 temporal encoder；",
        "- 默认扫描 100 个 target batch；",
        "- 用 Inception Score 最大的 shift 作为无监督选择结果；",
        "- target true label 只用于本目录中的 oracle accuracy / Macro-F1 曲线，绝不参与 shift 选择。",
        "- 实现边界：TimeMatch 的位置编码显式预留 temporal-shift buffer，因此平移后的日期可以越过原年度边界；TSStructure 正式 ContinuousTime2Vec 只允许 `[0,1]`。为避免错误 clamp，本诊断仅在 Scalar 分支临时使用同一组已学习 Time2Vec 参数做区间外外推；No-shift 与 Domain Phase 仍走正式路径，模型参数不变。",
        "",
        f"本次 IS 选择：`{selected:+d}` 天。",
        f"本次 oracle-accuracy 最优 shift：`{int(shift_result['oracle_best_accuracy_shift_days']):+d}` 天。",
        f"本次 oracle-Macro-F1 最优 shift：`{int(shift_result['oracle_best_macro_f1_shift_days']):+d}` 天。",
        "",
        "官方参考：",
        f"- TimeMatch 代码：{TIMEMATCH_REPOSITORY}",
        f"- TimeMatch 论文：{TIMEMATCH_PAPER}",
        "",
        "## 3. 每个文件/文件夹是什么",
        "",
        "| 文件/目录 | 含义 | 怎么看 |",
        "|---|---|---|",
        "| `README_中文说明.md` | 当前说明文件 | 归档时与整个目录一起保留 |",
        "| `timematch_scalar_shift/shift_scan.csv` | -60~60 每个整数 shift 的 IS、entropy、oracle accuracy、oracle Macro-F1 | `selected_by_inception_score=true` 是正式无监督 scalar shift |",
        "| `timematch_scalar_shift/shift_scan.png` | scalar shift 扫描曲线 | 看 IS 选中的 shift 是否接近 oracle 最优；差很大说明 scalar shift **估计器**本身在当前模型上有问题 |",
        "| `class_comparison_summary.csv` | 每类 No/Scalar/Phase 的 direct PSE distance、LTAE 距离、概率、准确率和 Phase 函数几何结果 | 第一层最主要汇总表 |",
        "| `sample_comparison.csv` | 每个 oracle target 样本的三视图详细结果 | 可计算分布、失败样本和类别差异 |",
        "| `ltae_representation_comparison/` | 每类 Phase-only 单路 LTAE fused 距离散点 | 左列 Scalar、右列 Domain Phase；点在 y=x 下方表示比 No shift 更接近 source |",
        "| `classifier_probability_comparison/` | 每类真实类别概率 before/after 散点 | 点在 y=x 上方表示该时间校正提高 true-class probability |",
        "| `phase_groups/` | 保存的 confirmed nonlinear gamma 及其日期偏移 | 用于对照 nonlinear Phase 和 scalar shift 的尺度 |",
        "| `comparison_manifest.json` | 完整运行配置与全局汇总 | 机器可读归档 |",
        "",
        "## 4. 为什么没有强行画 Scalar 的 SRVF Shape/Trend 几何",
        "",
        "TimeMatch 的 scalar shift 是对日历日期做不受端点约束的整体平移，例如 `t -> t-20 days`。当前 SRVF Domain Phase 则定义在固定 `[0,1]` 区间并满足端点约束。把 TimeMatch translation 人为转换成一个固定端点的 gamma 会额外引入边界形变，不再是 TimeMatch 原操作。",
        "",
        "因此本实验在双方真正共同的接口——**进入 Time2Vec/LTAE 的 observation positions**——进行公平比较。Shape/Trend SRVF 只继续记录 Domain Phase 自身 before→after 的几何效果。",
        "",
        "## 5. 最重要的判断顺序",
        "",
        "1. 先看 `shift_scan.png`：IS 选择的 scalar shift 是否接近 oracle 最优 shift。",
        "2. 先看全局 `pse_l2_mean`：Scalar 与 Domain Phase 谁在完整 PSE latent temporal process 上更接近 source；再看 `fused_distance_mean` 判断这种改善是否传到 LTAE。",
        "3. 再看 `oracle_accuracy` / `true_class_probability_mean`：几何改善是否传到 classifier。",
        "4. 若 oracle scalar shift 很有效而 IS scalar 不行，问题主要是 shift estimator；若 IS scalar 也明显优于 Domain Phase，问题主要在 Domain Phase；若 Scalar 和 Domain Phase 都几乎不改变 LTAE，则应优先检查 Time2Vec/LTAE 对时间位置的敏感性。",
        "",
        "## 6. 本次全局结果（运行后自动写入）",
        "",
        f"- checkpoint：`{checkpoint}`",
        f"- source：`{source}`",
        f"- target：`{target}`",
        f"- visualized classes：`{','.join(str(v) for v in requested_classes)}`",
        f"- TimeMatch IS scalar shift：`{selected:+d}` days",
        f"- Fused distance：`{global_summary['fused_distance_mean']}`",
        f"- Oracle accuracy：`{global_summary['oracle_accuracy']}`",
        f"- Oracle Macro-F1（仅可视化类别）：`{global_summary['oracle_macro_f1_on_visualized_classes']}`",
        f"- Mean true-class probability：`{global_summary['true_class_probability_mean']}`",
        "",
        "## 7. 标签使用边界",
        "",
        "target true labels 仅用于 post-hoc oracle 诊断：选取同类 source/target 样本、计算 oracle accuracy/F1、真实类别概率和逐类距离。它们不进入 TimeMatch IS shift 选择、不进入 Domain Phase 估计、不训练模型。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    runtime = checkpoint.get("runtime_config")
    if not isinstance(runtime, dict):
        raise ValueError("checkpoint must contain runtime_config")

    source = str(runtime["source"])
    target = str(runtime["target"])
    classes = [str(value) for value in runtime["classes"]]
    data_root = str(args.data_root or runtime["data_root"])
    seed = int(runtime["seed"])
    fold = int(args.fold)
    closed_set = bool(phasevis._runtime_value(runtime, "closed_set", True))
    combine = bool(phasevis._runtime_value(runtime, "combine_spring_and_winter", False))
    time_mode = str(
        phasevis._runtime_value(runtime, "time_coordinate_mode", "canonical_day_of_year")
    )
    val_ratio = float(phasevis._runtime_value(runtime, "val_ratio", 0.1))
    test_ratio = float(phasevis._runtime_value(runtime, "test_ratio", 0.2))
    time_scale_days = float(phasevis._runtime_value(runtime, "time_scale", 365.0))

    groups = phasevis._checkpoint_group_payloads(checkpoint)
    class_to_group = phasevis._class_to_group(
        groups,
        phase_routes=checkpoint.get("phase_routes"),
        num_classes=len(classes),
    )
    available_classes = tuple(sorted(class_to_group))
    requested_classes = available_classes if args.classes is None else tuple(args.classes)
    unknown = sorted(set(requested_classes) - set(available_classes))
    if unknown:
        raise ValueError(
            "first-layer comparison currently uses confirmed Phase member classes only; "
            "non-member generalization belongs to the second-layer diagnostic. Unknown: "
            + ",".join(str(v) for v in unknown)
        )

    device = torch.device(args.device)
    model_checkpoint_path = None if args.model_checkpoint is None else args.model_checkpoint.resolve()
    model_checkpoint = (
        None
        if model_checkpoint_path is None
        else torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    )
    model = phasevis._build_model(
        runtime, checkpoint, device, model_checkpoint=model_checkpoint
    )

    source_all = phasevis._eligible_parcels(
        data_root,
        source,
        classes,
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
    )
    target_all = phasevis._eligible_parcels(
        data_root,
        target,
        classes,
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
    )
    splits = phasevis._reconstruct_fold_splits(
        source_all,
        target_all,
        source=source,
        target=target,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        fold=fold,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # A. TimeMatch-style scalar shift scan over target train batches.
    # ------------------------------------------------------------------
    shift_loader = _timematch_estimation_loader(
        data_root,
        target,
        classes,
        splits[target]["train"],
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
        batch_size=args.timematch_batch_size,
        num_workers=args.num_workers,
        num_pixels=args.timematch_num_pixels,
        seed=seed,
    )
    print(
        "FIRST_LAYER_SHIFT_COMPARE_START|"
        f"checkpoint={checkpoint_path}|source={source}|target={target}|"
        f"shift_range=[{-args.timematch_max_shift},{args.timematch_max_shift}]|"
        f"shift_batches={args.timematch_estimation_batches}",
        flush=True,
    )
    shift_result = _estimate_timematch_scalar_shift(
        model,
        shift_loader,
        device=device,
        min_shift=-args.timematch_max_shift,
        max_shift=args.timematch_max_shift,
        max_batches=args.timematch_estimation_batches,
        num_classes=len(classes),
        time_scale_days=time_scale_days,
    )
    shift_rows = shift_result["rows"]
    selected_shift = int(shift_result["selected_shift_days"])
    _write_csv(output_dir / "timematch_scalar_shift" / "shift_scan.csv", shift_rows)
    _plot_shift_scan(
        output_dir / "timematch_scalar_shift" / "shift_scan.png",
        shift_result,
        args.dpi,
    )
    print(
        "TIMEMATCH_SHIFT_SELECTED|"
        f"is_shift={selected_shift:+d}|"
        f"oracle_acc_shift={int(shift_result['oracle_best_accuracy_shift_days']):+d}|"
        f"oracle_f1_shift={int(shift_result['oracle_best_macro_f1_shift_days']):+d}|"
        f"samples={shift_result['num_samples']}|batches={shift_result['num_batches']}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # B. Fixed oracle visualization subset shared by No/Scalar/Phase views.
    # ------------------------------------------------------------------
    source_meta = phasevis._metadata_dataset(
        data_root,
        source,
        classes,
        splits[source]["test"],
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
    )
    target_meta = phasevis._metadata_dataset(
        data_root,
        target,
        classes,
        splits[target]["test"],
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
    )
    source_parcels = phasevis._uniform_selected_parcels(
        source_meta, requested_classes, args.samples_per_class
    )
    target_parcels = phasevis._uniform_selected_parcels(
        target_meta, requested_classes, args.samples_per_class
    )
    source_loader = phasevis._selected_loader(
        data_root,
        source,
        classes,
        source_parcels,
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    target_loader = phasevis._selected_loader(
        data_root,
        target,
        classes,
        target_parcels,
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    source_records = phasevis._collect_geometry(
        model,
        source_loader,
        device=device,
        target=False,
        class_to_group=class_to_group,
    )
    target_records = phasevis._collect_geometry(
        model,
        target_loader,
        device=device,
        target=True,
        class_to_group=class_to_group,
    )
    _attach_scalar_view(
        model,
        target_loader,
        target_records,
        device=device,
        shift_days=selected_shift,
        time_scale_days=time_scale_days,
    )

    bank = checkpoint.get("source_prototype_bank")
    if not isinstance(bank, dict):
        raise ValueError("checkpoint source_prototype_bank is missing")
    source_fused_proto = bank["fused"].detach().cpu()
    source_shape_proto = bank["shape_srvf"].detach().cpu()
    source_shape_support = bank["shape_support"].detach().cpu()
    source_trend_proto = bank["trend_srvf"].detach().cpu()
    source_trend_support = bank["trend_support"].detach().cpu()

    class_rows: List[dict] = []
    sample_rows: List[dict] = []
    for class_id in requested_classes:
        source_class = source_records.get(class_id, [])
        target_class = target_records.get(class_id, [])
        if not source_class or not target_class:
            print(
                f"FIRST_LAYER_CLASS_SKIPPED|class={class_id}|"
                f"source={len(source_class)}|target={len(target_class)}",
                flush=True,
            )
            continue
        name = classes[class_id]
        group_id = int(class_to_group[class_id]["group_id"])
        stem = phasevis._class_stem(class_id, name)

        fused0, fuseds, fusedp, fused_stats = _three_view_distances(
            target_class,
            prototype=source_fused_proto[class_id],
            base_key="fused_repr_before",
            scalar_key="fused_repr_scalar",
            phase_key="fused_repr_after",
        )
        cls = _three_view_classification(target_class, class_id)
        pse_views, pse_rows = _three_view_pse_metrics(
            source_class, target_class, grid_size=args.pse_grid_size
        )

        # Domain Phase's own function-geometry effect is retained for context.
        shape_stats = phasevis._distance_stats(
            target_class,
            q_before_key="shape_q_before",
            support_before_key="shape_support_before",
            q_after_key="shape_q_after",
            support_after_key="shape_support_after",
            prototype_q=source_shape_proto[class_id],
            prototype_support=source_shape_support[class_id],
        )
        geometry_trend_stats = phasevis._distance_stats(
            target_class,
            q_before_key="trend_q_before",
            support_before_key="trend_support_before",
            q_after_key="trend_q_after",
            support_after_key="trend_support_after",
            prototype_q=source_trend_proto[class_id],
            prototype_support=source_trend_support[class_id],
        )

        class_rows.append(
            {
                "class_id": int(class_id),
                "class_name": name,
                "phase_group_id": group_id,
                "target_samples": len(target_class),
                "timematch_scalar_shift_days": selected_shift,
                "pse_l2_no_shift": pse_views["no_shift"]["l2"],
                "pse_l2_scalar": pse_views["timematch_scalar"]["l2"],
                "pse_l2_phase": pse_views["domain_phase"]["l2"],
                "pse_mse_no_shift": pse_views["no_shift"]["mse"],
                "pse_mse_scalar": pse_views["timematch_scalar"]["mse"],
                "pse_mse_phase": pse_views["domain_phase"]["mse"],
                "fused_no_shift_mean": fused_stats["no_shift_mean"],
                "fused_scalar_mean": fused_stats["scalar_mean"],
                "fused_phase_mean": fused_stats["phase_mean"],
                "fused_scalar_relative_reduction": fused_stats["scalar_relative_reduction"],
                "fused_phase_relative_reduction": fused_stats["phase_relative_reduction"],
                "fused_scalar_improvement_rate": fused_stats["scalar_improvement_rate"],
                "fused_phase_improvement_rate": fused_stats["phase_improvement_rate"],
                "true_probability_no_shift_mean": cls["true_probability_no_shift_mean"],
                "true_probability_scalar_mean": cls["true_probability_scalar_mean"],
                "true_probability_phase_mean": cls["true_probability_phase_mean"],
                "oracle_accuracy_no_shift": cls["oracle_accuracy_no_shift"],
                "oracle_accuracy_scalar": cls["oracle_accuracy_scalar"],
                "oracle_accuracy_phase": cls["oracle_accuracy_phase"],
                "shape_geometry_no_shift_mean": shape_stats["before_mean"],
                "shape_geometry_domain_phase_mean": shape_stats["after_mean"],
                "shape_geometry_domain_phase_improvement_rate": shape_stats["improvement_rate"],
                "trend_geometry_no_shift_mean": geometry_trend_stats["before_mean"],
                "trend_geometry_domain_phase_mean": geometry_trend_stats["after_mean"],
                "trend_geometry_domain_phase_improvement_rate": geometry_trend_stats["improvement_rate"],
            }
        )

        for index, record in enumerate(target_class):
            pse_sample = pse_rows[index]
            logits0 = record["logits_before"].float()
            logitss = record["logits_scalar"].float()
            logitsp = record["logits_after"].float()
            probs0 = torch.softmax(logits0, dim=-1)
            probss = torch.softmax(logitss, dim=-1)
            probsp = torch.softmax(logitsp, dim=-1)
            phase_delta = (
                record["positions_after"][record["mask"].bool()]
                - record["positions"][record["mask"].bool()]
            ).float() * time_scale_days
            sample_rows.append(
                {
                    "class_id": int(class_id),
                    "class_name": name,
                    "parcel_index": int(record["parcel_index"]),
                    "phase_group_id": group_id,
                    "scalar_shift_days": selected_shift,
                    "phase_position_shift_signed_mean_days": float(phase_delta.mean().item()) if phase_delta.numel() else None,
                    "phase_position_shift_abs_mean_days": float(phase_delta.abs().mean().item()) if phase_delta.numel() else None,
                    "pse_l2_no_shift": pse_sample["pse_l2_no_shift"],
                    "pse_l2_scalar": pse_sample["pse_l2_timematch_scalar"],
                    "pse_l2_phase": pse_sample["pse_l2_domain_phase"],
                    "pse_mse_no_shift": pse_sample["pse_mse_no_shift"],
                    "pse_mse_scalar": pse_sample["pse_mse_timematch_scalar"],
                    "pse_mse_phase": pse_sample["pse_mse_domain_phase"],
                    "fused_distance_no_shift": float(fused0[index].item()),
                    "fused_distance_scalar": float(fuseds[index].item()),
                    "fused_distance_phase": float(fusedp[index].item()),
                    "true_probability_no_shift": float(probs0[class_id].item()),
                    "true_probability_scalar": float(probss[class_id].item()),
                    "true_probability_phase": float(probsp[class_id].item()),
                    "prediction_no_shift": int(logits0.argmax().item()),
                    "prediction_scalar": int(logitss.argmax().item()),
                    "prediction_phase": int(logitsp.argmax().item()),
                    "correct_no_shift": bool(logits0.argmax().item() == class_id),
                    "correct_scalar": bool(logitss.argmax().item() == class_id),
                    "correct_phase": bool(logitsp.argmax().item() == class_id),
                }
            )

        _plot_distance_comparison(
            output_dir / "ltae_representation_comparison" / f"{stem}.png",
            title=f"No shift vs TimeMatch scalar vs Domain Phase — class {class_id}: {name}",
            metrics=(
                ("Fused / formal source prototype", fused0, fuseds, fusedp),
            ),
            dpi=args.dpi,
        )
        _plot_probability_comparison(
            output_dir / "classifier_probability_comparison" / f"{stem}.png",
            title=f"True-class probability — class {class_id}: {name}",
            base=cls["prob_no"],
            scalar=cls["prob_scalar"],
            phase=cls["prob_phase"],
            dpi=args.dpi,
        )

        print(
            "FIRST_LAYER_CLASS|"
            f"class={class_id}:{name}|group={group_id}|"
            f"fused=no:{fused_stats['no_shift_mean']:.6g},"
            f"scalar:{fused_stats['scalar_mean']:.6g},"
            f"phase:{fused_stats['phase_mean']:.6g}|"
            f"acc=no:{cls['oracle_accuracy_no_shift']:.4f},"
            f"scalar:{cls['oracle_accuracy_scalar']:.4f},"
            f"phase:{cls['oracle_accuracy_phase']:.4f}",
            flush=True,
        )

    if not class_rows or not sample_rows:
        raise RuntimeError("no class produced first-layer comparison diagnostics")

    _write_csv(output_dir / "class_comparison_summary.csv", class_rows)
    _write_csv(output_dir / "sample_comparison.csv", sample_rows)
    group_summaries = phasevis._plot_phase_groups(output_dir, groups, args.dpi)
    global_summary = _global_summary(sample_rows, requested_classes)

    manifest = {
        "purpose": "first-layer no-shift vs TimeMatch-style scalar shift vs confirmed Domain Phase comparison",
        "phase_checkpoint": str(checkpoint_path),
        "model_checkpoint": str(model_checkpoint_path or checkpoint_path),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "successful_optimizer_steps": checkpoint.get("successful_optimizer_steps"),
        "source": source,
        "target": target,
        "seed": seed,
        "fold": fold,
        "classes": list(classes),
        "visualized_class_ids": [int(v) for v in requested_classes],
        "shift_estimation_partition": "target train only; labels ignored for Inception-Score selection",
        "oracle_comparison_partition": "held-out source test + held-out target test",
        "target_label_usage": "oracle post-hoc diagnostics only; never used to select scalar shift or estimate Domain Phase",
        "timematch_reference": {
            "repository": TIMEMATCH_REPOSITORY,
            "paper": TIMEMATCH_PAPER,
            "official_initial_estimator_semantics": "integer target-to-source dates + shift; choose maximum Inception Score",
            "max_shift_days": int(args.timematch_max_shift),
            "estimation_batches_requested": int(args.timematch_estimation_batches),
            "batch_size": int(args.timematch_batch_size),
            "num_pixels": int(args.timematch_num_pixels),
        },
        "timematch_shift_result": {
            key: value
            for key, value in shift_result.items()
            if key not in {"rows", "labels", "predictions"}
        },
        "domain_phase_direction": {
            "saved_gamma": "source_to_target",
            "target_ltae_positions": "gamma_inverse(target_positions)",
        },
        "scalar_geometry_note": "TimeMatch scalar calendar translation is not forced into a fixed-endpoint SRVF gamma; scalar and Domain Phase are compared at their common LTAE-position interface.",
        "phase_groups": group_summaries,
        "global_summary": global_summary,
        "class_summary": class_rows,
    }
    _json_dump(output_dir / "comparison_manifest.json", manifest)
    _json_dump(output_dir / "manifest.json", manifest)
    _json_dump(output_dir / "summary.json", global_summary)
    _write_readme(
        output_dir / "README_中文说明.md",
        checkpoint=checkpoint_path,
        source=source,
        target=target,
        shift_result=shift_result,
        global_summary=global_summary,
        requested_classes=requested_classes,
    )

    print(
        "FIRST_LAYER_SHIFT_COMPARE_COMPLETE|"
        f"output={output_dir}|"
        f"is_shift={selected_shift:+d}|"
        f"fused_no={global_summary['fused_distance_mean']['no_shift']:.6g}|"
        f"fused_scalar={global_summary['fused_distance_mean']['timematch_scalar']:.6g}|"
        f"fused_phase={global_summary['fused_distance_mean']['domain_phase']:.6g}|"
        f"acc_no={global_summary['oracle_accuracy']['no_shift']:.4f}|"
        f"acc_scalar={global_summary['oracle_accuracy']['timematch_scalar']:.4f}|"
        f"acc_phase={global_summary['oracle_accuracy']['domain_phase']:.4f}",
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="First-layer diagnostic: No shift vs TimeMatch scalar vs Domain Phase."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage-2 checkpoint providing confirmed Phase state and source statistics")
    parser.add_argument("--model-checkpoint", type=Path, default=None, help="Optional Stage-1 checkpoint providing zero-step model weights while reusing saved Phase state")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--classes", type=phasevis._parse_int_list, default=None)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pse-grid-size", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--timematch-max-shift", type=int, default=60)
    parser.add_argument("--timematch-estimation-batches", type=int, default=100)
    parser.add_argument("--timematch-batch-size", type=int, default=128)
    parser.add_argument("--timematch-num-pixels", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples_per_class <= 0 or args.batch_size <= 0:
        raise ValueError("sample/batch sizes must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be nonnegative")
    if args.pse_grid_size < 2:
        raise ValueError("pse-grid-size must be at least 2")
    if args.timematch_max_shift < 0:
        raise ValueError("timematch-max-shift must be nonnegative")
    if args.timematch_estimation_batches <= 0 or args.timematch_batch_size <= 0:
        raise ValueError("TimeMatch estimation batch settings must be positive")
    if args.timematch_num_pixels <= 0 or args.dpi <= 0:
        raise ValueError("TimeMatch num-pixels and dpi must be positive")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run(args)


if __name__ == "__main__":
    main()
