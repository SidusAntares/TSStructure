#!/usr/bin/env python3
"""Oracle-only M=1 class-center vs group-center Domain Phase diagnostic.

This script does not train or update any model state.  It compares, on exactly
one held-out target-test population and one frozen Stage-1 model:

    A. No Phase (identity)
    B. oracle class-center Phase gamma_bar_{y_i}
    C. final M=1 group-center Phase delta

The target true label is used only in branch B to select the corresponding
class-level Phase center.  That branch is therefore permanently oracle-only and
must never be used as an unsupervised inference rule.

Interpretation is intentionally split by final M=1 estimation membership:
classes in C_est can diagnose M=1 aggregation/compression; classes outside
C_est can only diagnose whether confirmed Phase was over-extended to a class
that did not estimate the shared center.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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

import visualize_stage2_phase_alignment as phasevis
from methods.structure_da import phase_visualization_protocol as visproto
from methods.structure_da.confirmed_phase_view import align_target_positions_to_source
from methods.structure_da.phase_geometry import phase_distance
from methods.structure_da.phase_registration import (
    resample_gamma,
    warp_q_gamma,
    warp_support_gamma,
)
from methods.structure_da.prototype_bank import support_aware_q_distance


VARIANTS = ("no_phase", "class_center", "group_center")


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _status(value) -> str:
    if hasattr(value, "value"):
        value = value.value
    return str(value).lower()


def _phase_payload(checkpoint: dict) -> dict:
    payload = checkpoint.get("phase_state")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint does not contain phase_state")
    return payload


def _final_m1_group(checkpoint: dict) -> dict:
    phase = _phase_payload(checkpoint)
    if int(phase.get("m", -1)) != 1:
        raise ValueError("05 diagnostic requires a final M=1 Phase state")
    groups = phase.get("groups")
    if not isinstance(groups, (tuple, list)):
        raise ValueError("phase_state.groups is missing")
    confirmed = [item for item in groups if _status(item.get("status")) == "confirmed"]
    if len(confirmed) != 1:
        raise ValueError("05 diagnostic requires exactly one confirmed M=1 group")
    if not isinstance(confirmed[0].get("center_gamma"), Tensor):
        raise ValueError("confirmed group center_gamma is missing")
    return confirmed[0]


def _class_center_payloads(checkpoint: dict) -> Dict[int, dict]:
    centers = _phase_payload(checkpoint).get("class_centers")
    if not isinstance(centers, (tuple, list)) or not centers:
        raise ValueError(
            "checkpoint has no serialized class centers; rerun Stage-2 calibration "
            "with the Phase-center persistence patch before running experiment 05"
        )
    result: Dict[int, dict] = {}
    for payload in centers:
        if not isinstance(payload, dict) or not isinstance(payload.get("center_gamma"), Tensor):
            continue
        class_id = int(payload["class_id"])
        result[class_id] = payload
    if not result:
        raise ValueError("checkpoint class_centers contain no valid gamma tensors")
    return result


def _identity_like(gamma: Tensor) -> Tensor:
    return torch.linspace(0.0, 1.0, gamma.numel(), dtype=torch.float64)


def _gamma_grid(gamma: Tensor, target_grid: Tensor) -> Tensor:
    gamma_cpu = gamma.detach().cpu().double()
    reg_grid = torch.linspace(0.0, 1.0, gamma_cpu.numel(), dtype=torch.float64)
    return resample_gamma(gamma_cpu, reg_grid, target_grid.detach().cpu().double()).to(
        device=target_grid.device, dtype=target_grid.dtype
    )


def _integration_weights(grid_size: int, reference: Tensor) -> Tensor:
    weights = torch.ones(grid_size, device=reference.device, dtype=reference.dtype)
    if grid_size > 1:
        weights[[0, -1]] *= 0.5
    return weights / weights.sum().clamp_min(torch.finfo(reference.dtype).eps)


def _variant_positions(
    native: Tensor,
    mask: Tensor,
    labels: Tensor,
    *,
    class_centers: Dict[int, dict],
    group_gamma: Tensor,
) -> Tuple[Tensor, Tensor]:
    class_positions = native.detach().clone()
    for class_id in sorted({int(value) for value in labels.detach().cpu().tolist()}):
        payload = class_centers.get(class_id)
        if payload is None:
            continue
        rows = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        class_positions[rows] = align_target_positions_to_source(
            native[rows], mask[rows], payload["center_gamma"]
        )
    group_positions = align_target_positions_to_source(native, mask, group_gamma)
    return class_positions.detach(), group_positions.detach()


def _warp_geometry_by_class(
    q: Tensor,
    support: Tensor,
    labels: Tensor,
    class_centers: Dict[int, dict],
    grid: Tensor,
) -> Tuple[Tensor, Tensor]:
    warped_q = q.detach().clone()
    warped_support = support.detach().clone()
    for class_id in sorted({int(value) for value in labels.detach().cpu().tolist()}):
        payload = class_centers.get(class_id)
        if payload is None:
            continue
        rows = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        gamma = _gamma_grid(payload["center_gamma"], grid)
        warped_q[rows] = warp_q_gamma(q[rows], gamma)
        for row in rows.detach().cpu().tolist():
            warped_support[row] = warp_support_gamma(support[row], gamma, grid)
    return warped_q, warped_support


def _warp_geometry_group(
    q: Tensor,
    support: Tensor,
    group_gamma: Tensor,
    grid: Tensor,
) -> Tuple[Tensor, Tensor]:
    gamma = _gamma_grid(group_gamma, grid)
    warped_q = warp_q_gamma(q, gamma)
    warped_support = torch.stack(
        [warp_support_gamma(item, gamma, grid) for item in support], dim=0
    )
    return warped_q, warped_support


def _true_class_q_distance(
    q: Tensor,
    support: Tensor,
    prototypes: Tensor,
    prototype_support: Tensor,
    labels: Tensor,
    weights: Tensor,
) -> Tuple[Tensor, Tensor]:
    result = support_aware_q_distance(
        q,
        prototypes,
        support,
        prototype_support,
        weights,
    )
    rows = torch.arange(labels.shape[0], device=labels.device)
    return result.distance[rows, labels], result.valid[rows, labels]


def _fused_distance(fused: Tensor, prototypes: Tensor, labels: Tensor) -> Tensor:
    selected = prototypes.index_select(0, labels)
    return torch.linalg.vector_norm(fused - selected, dim=-1)


class _PSECenterAccumulator:
    def __init__(self, *, grid_size: int) -> None:
        self.grid_size = int(grid_size)
        self.sum_by_class: Dict[int, Tensor] = {}
        self.count_by_class: Dict[int, Tensor] = {}
        self.samples_by_class: Dict[int, int] = {}

    def add(self, *, class_id: int, tokens: Tensor, positions: Tensor, mask: Tensor) -> None:
        trajectory, support, _ = visproto.canonicalize_pse_tokens(
            {"pse_tokens": tokens, "positions": positions, "mask": mask},
            positions_key="positions",
            grid_size=self.grid_size,
        )
        if class_id not in self.sum_by_class:
            self.sum_by_class[class_id] = torch.zeros_like(trajectory)
            self.count_by_class[class_id] = torch.zeros(
                trajectory.shape[0], dtype=torch.float32
            )
            self.samples_by_class[class_id] = 0
        self.sum_by_class[class_id] += trajectory * support.unsqueeze(-1)
        self.count_by_class[class_id] += support.float()
        self.samples_by_class[class_id] += 1

    def center(self, class_id: int) -> Tuple[Tensor, Tensor]:
        counts = self.count_by_class[class_id]
        center = self.sum_by_class[class_id] / counts.clamp_min(1.0).unsqueeze(-1)
        return center, counts > 0


@torch.no_grad()
def _build_source_pse_centers(model, loader, *, device: torch.device, grid_size: int):
    accumulator = _PSECenterAccumulator(grid_size=grid_size)
    for raw_batch in loader:
        batch = phasevis._move_batch(raw_batch, device)
        backbone = model.forward_backbone(
            batch["pixels"],
            batch["valid_pixels"],
            batch["positions"],
            batch.get("extra"),
            time_mask=batch.get("time_mask"),
            compute_decomposition=False,
        )
        labels = batch["label"].detach().cpu().long()
        for row, class_id in enumerate(labels.tolist()):
            accumulator.add(
                class_id=int(class_id),
                tokens=backbone.tokens[row].detach().cpu(),
                positions=backbone.normalized_positions[row].detach().cpu(),
                mask=backbone.time_mask[row].detach().cpu(),
            )
    return accumulator


def _safe_relative_reduction(before: float, after: float) -> float:
    if not math.isfinite(before) or not math.isfinite(after) or abs(before) <= 1e-12:
        return float("nan")
    return (before - after) / before


def _binary_class_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_id: int) -> dict:
    tp = sum(int(t == class_id and p == class_id) for t, p in zip(y_true, y_pred))
    fp = sum(int(t != class_id and p == class_id) for t, p in zip(y_true, y_pred))
    fn = sum(int(t == class_id and p != class_id) for t, p in zip(y_true, y_pred))
    support = sum(int(t == class_id) for t in y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / support if support else float("nan")
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _selected_test_parcels(meta, classes: Sequence[int], samples_per_class: int) -> np.ndarray:
    if samples_per_class <= 0:
        return np.asarray(sorted(int(value) for value in meta.get_parcel_indices().tolist()))
    return phasevis._uniform_selected_parcels(meta, classes, samples_per_class)


def _phase_geometry_outputs(
    output_dir: Path,
    *,
    checkpoint: dict,
    classes: Sequence[str],
    group: dict,
    class_centers: Dict[int, dict],
    dpi: int,
) -> Tuple[List[dict], List[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    group_gamma = group["center_gamma"].detach().cpu().double()
    identity = _identity_like(group_gamma)
    members = {int(value) for value in group.get("member_classes", ())}
    geometry_rows: List[dict] = []
    prepared: List[Tuple[int, Tensor]] = []
    for class_id in sorted(class_centers):
        payload = class_centers[class_id]
        gamma = payload["center_gamma"].detach().cpu().double()
        if gamma.shape != group_gamma.shape:
            raise ValueError("class and group gamma must share the same grid")
        displacement = (gamma - identity) * 365.0
        residual = (gamma - group_gamma) * 365.0
        interior = (identity >= 0.10) & (identity <= 0.90)
        geometry_rows.append({
            "class_id": class_id,
            "class_name": classes[class_id],
            "estimation_member": class_id in members,
            "comparison_role": visproto.phase_diagnostic_role(class_id, tuple(members)),
            "class_center_valid": bool(payload.get("valid", False)),
            "reject_reason": payload.get("reject_reason"),
            "candidate_count": payload.get("candidate_count"),
            "effective_evidence_count": payload.get("effective_evidence_count"),
            "within_class_dispersion": payload.get("dispersion"),
            "within_class_diameter": payload.get("diameter"),
            "center_drift": payload.get("center_drift"),
            "d_to_identity": float(phase_distance(gamma, identity).item()),
            "d_to_group_center": float(phase_distance(gamma, group_gamma).item()),
            "median_displacement_days": float(displacement[interior].median().item()),
            "p10_displacement_days": float(torch.quantile(displacement[interior], 0.10).item()),
            "p90_displacement_days": float(torch.quantile(displacement[interior], 0.90).item()),
            "max_abs_displacement_days": float(displacement.abs().max().item()),
            "max_abs_class_vs_group_residual_days": float(residual.abs().max().item()),
        })
        prepared.append((class_id, gamma))
    _write_csv(output_dir / "class_phase_geometry.csv", geometry_rows)

    pairwise_rows: List[dict] = []
    matrix = np.zeros((len(prepared), len(prepared)), dtype=np.float64)
    for i, (left_id, left_gamma) in enumerate(prepared):
        for j, (right_id, right_gamma) in enumerate(prepared):
            value = float(phase_distance(left_gamma, right_gamma).item())
            matrix[i, j] = value
            pairwise_rows.append({
                "class_id_i": left_id,
                "class_name_i": classes[left_id],
                "class_id_j": right_id,
                "class_name_j": classes[right_id],
                "phase_distance": value,
            })
    _write_csv(output_dir / "pairwise_class_center_distance.csv", pairwise_rows)

    x = identity.numpy() * 365.0
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    ax.plot(x, x, linestyle=":", linewidth=1.4, label="identity")
    ax.plot(x, group_gamma.numpy() * 365.0, linewidth=3.0, label="M=1 group center")
    for class_id, gamma in prepared:
        marker = "*" if class_id in members else ""
        ax.plot(x, gamma.numpy() * 365.0, linewidth=1.2, alpha=0.8,
                label=f"{class_id}:{classes[class_id]}{marker}")
    ax.set_xlabel("source canonical day")
    ax.set_ylabel("target canonical day")
    ax.set_title("Class Phase centers vs M=1 group center (* = estimation member)")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(output_dir / "class_center_gamma_overlay.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    ax.axhline(0.0, linestyle=":", linewidth=1.2, label="identity displacement")
    ax.plot(x, (group_gamma - identity).numpy() * 365.0, linewidth=3.0,
            label="M=1 group center")
    for class_id, gamma in prepared:
        ax.plot(x, (gamma - identity).numpy() * 365.0, linewidth=1.2, alpha=0.8,
                label=f"{class_id}:{classes[class_id]}")
    ax.set_xlabel("canonical day")
    ax.set_ylabel("gamma(t) - t (days)")
    ax.set_title("Class-center Phase displacement")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(output_dir / "class_center_displacement_days.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    ax.axhline(0.0, linestyle=":", linewidth=1.2)
    for class_id, gamma in prepared:
        ax.plot(x, (gamma - group_gamma).numpy() * 365.0, linewidth=1.2, alpha=0.85,
                label=f"{class_id}:{classes[class_id]}")
    ax.set_xlabel("canonical day")
    ax.set_ylabel("class center - group center (days)")
    ax.set_title("Local residual hidden by M=1 aggregation")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(output_dir / "class_vs_group_residual_days.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    labels = [f"{class_id}:{classes[class_id]}" for class_id, _ in prepared]
    d_id = [next(row["d_to_identity"] for row in geometry_rows if row["class_id"] == c)
            for c, _ in prepared]
    d_group = [next(row["d_to_group_center"] for row in geometry_rows if row["class_id"] == c)
               for c, _ in prepared]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)
    axes[0].bar(np.arange(len(labels)), d_id)
    axes[0].set_title("d_gamma(class center, identity)\n(no identity threshold inferred)")
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    axes[0].set_ylabel("Fisher-Rao Phase distance")
    axes[0].grid(axis="y", alpha=0.18)
    axes[1].bar(np.arange(len(labels)), d_group)
    radius = checkpoint.get("runtime_config", {}).get("stage2_phase_global_radius")
    if radius is not None:
        axes[1].axhline(float(radius), linestyle="--", linewidth=1.2,
                        label="group radius threshold (applies only here)")
        axes[1].legend(fontsize=8)
    axes[1].set_title("d_gamma(class center, M=1 group center)")
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    axes[1].set_ylabel("Fisher-Rao Phase distance")
    axes[1].grid(axis="y", alpha=0.18)
    fig.savefig(output_dir / "distance_to_identity_and_group.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    ax.set_title("Pairwise class-center Phase distance")
    fig.colorbar(image, ax=ax, label="d_gamma")
    fig.savefig(output_dir / "pairwise_class_center_distance_heatmap.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)

    progressive_rows: List[dict] = []
    prior_by_class: Dict[int, Tensor] = {}
    progressive = checkpoint.get("phase_state_progressive", ())
    if isinstance(progressive, (tuple, list)):
        for stage in progressive:
            if not isinstance(stage, dict):
                continue
            budget = int(stage.get("evidence_budget", -1))
            state = stage.get("phase_state")
            if not isinstance(state, dict):
                continue
            for payload in state.get("class_centers", ()):
                gamma = payload.get("center_gamma") if isinstance(payload, dict) else None
                if not isinstance(gamma, Tensor):
                    continue
                class_id = int(payload["class_id"])
                gamma = gamma.detach().cpu().double()
                previous = prior_by_class.get(class_id)
                recomputed_drift = (
                    None if previous is None else float(phase_distance(gamma, previous).item())
                )
                progressive_rows.append({
                    "evidence_budget": budget,
                    "scan_index": int(state.get("scan_index", -1)),
                    "phase_m": int(state.get("m", -1)),
                    "phase_decision": state.get("decision_status"),
                    "class_id": class_id,
                    "class_name": classes[class_id],
                    "valid": bool(payload.get("valid", False)),
                    "reject_reason": payload.get("reject_reason"),
                    "candidate_count": payload.get("candidate_count"),
                    "effective_evidence_count": payload.get("effective_evidence_count"),
                    "dispersion": payload.get("dispersion"),
                    "diameter": payload.get("diameter"),
                    "serialized_center_drift": payload.get("center_drift"),
                    "recomputed_d_to_previous_center": recomputed_drift,
                    "d_to_identity": float(phase_distance(gamma, _identity_like(gamma)).item()),
                    "d_to_final_m1_group": float(phase_distance(gamma, group_gamma).item()),
                })
                prior_by_class[class_id] = gamma
    if progressive_rows:
        _write_csv(output_dir / "class_center_progressive_stability.csv", progressive_rows)

    return geometry_rows, progressive_rows


@torch.no_grad()
def _downstream_three_way(
    *,
    model,
    target_loader,
    source_pse: _PSECenterAccumulator,
    checkpoint: dict,
    classes: Sequence[str],
    class_centers: Dict[int, dict],
    group: dict,
    device: torch.device,
    pse_grid_size: int,
) -> Tuple[List[dict], List[dict], dict]:
    bank = checkpoint.get("source_prototype_bank")
    if not isinstance(bank, dict):
        raise ValueError("checkpoint does not contain source_prototype_bank")
    fused_proto = bank["fused"].to(device=device, dtype=torch.float32)
    trend_proto = bank["trend_srvf"].to(device=device, dtype=torch.float32)
    trend_proto_support = bank["trend_support"].to(device=device, dtype=torch.float32)
    shape_proto = bank["shape_srvf"].to(device=device, dtype=torch.float32)
    shape_proto_support = bank["shape_support"].to(device=device, dtype=torch.float32)
    group_gamma = group["center_gamma"]
    estimation_members = {int(value) for value in group.get("member_classes", ())}

    y_true: List[int] = []
    y_pred: Dict[str, List[int]] = {variant: [] for variant in VARIANTS}
    true_prob: Dict[str, Dict[int, List[float]]] = {
        variant: {class_id: [] for class_id in range(len(classes))}
        for variant in VARIANTS
    }
    scalar_metrics: Dict[str, Dict[int, Dict[str, List[float]]]] = {
        variant: {
            class_id: {
                "fused": [], "pse": [], "trend": [], "shape": [],
            }
            for class_id in range(len(classes))
        }
        for variant in VARIANTS
    }
    target_pse = {
        variant: _PSECenterAccumulator(grid_size=pse_grid_size) for variant in VARIANTS
    }
    sample_rows: List[dict] = []

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
        raw = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"), return_geometry=True
        )
        if raw.geometry is None:
            raise RuntimeError("05 diagnostic requires functional geometry")
        class_positions, group_positions = _variant_positions(
            native, mask, labels, class_centers=class_centers, group_gamma=group_gamma
        )
        class_out = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"),
            temporal_positions_override=class_positions, return_geometry=False
        )
        group_out = model.forward_from_backbone(
            backbone, batch["positions"], batch.get("extra"),
            temporal_positions_override=group_positions, return_geometry=False
        )
        outputs = {
            "no_phase": raw,
            "class_center": class_out,
            "group_center": group_out,
        }
        positions = {
            "no_phase": native,
            "class_center": class_positions,
            "group_center": group_positions,
        }

        grid = raw.geometry.canonical_grid.detach()
        weights = _integration_weights(grid.numel(), raw.geometry.trend_srvf)
        class_trend_q, class_trend_support = _warp_geometry_by_class(
            raw.geometry.trend_srvf, raw.geometry.trend_support, labels, class_centers, grid
        )
        class_shape_q, class_shape_support = _warp_geometry_by_class(
            raw.geometry.structure_srvf, raw.geometry.structure_support, labels, class_centers, grid
        )
        group_trend_q, group_trend_support = _warp_geometry_group(
            raw.geometry.trend_srvf, raw.geometry.trend_support, group_gamma, grid
        )
        group_shape_q, group_shape_support = _warp_geometry_group(
            raw.geometry.structure_srvf, raw.geometry.structure_support, group_gamma, grid
        )
        q_variants = {
            "no_phase": (
                raw.geometry.trend_srvf, raw.geometry.trend_support,
                raw.geometry.structure_srvf, raw.geometry.structure_support,
            ),
            "class_center": (
                class_trend_q, class_trend_support, class_shape_q, class_shape_support,
            ),
            "group_center": (
                group_trend_q, group_trend_support, group_shape_q, group_shape_support,
            ),
        }

        batch_metrics: Dict[str, dict] = {}
        for variant in VARIANTS:
            logits = outputs[variant].logits
            probs = torch.softmax(logits.float(), dim=-1)
            predictions = logits.argmax(dim=-1)
            fused_distance = _fused_distance(outputs[variant].fused_repr, fused_proto, labels)
            tq, ts, sq, ss = q_variants[variant]
            trend_distance, trend_valid = _true_class_q_distance(
                tq, ts, trend_proto, trend_proto_support, labels, weights
            )
            shape_distance, shape_valid = _true_class_q_distance(
                sq, ss, shape_proto, shape_proto_support, labels, weights
            )
            batch_metrics[variant] = {
                "probs": probs,
                "pred": predictions,
                "fused": fused_distance,
                "trend": trend_distance,
                "trend_valid": trend_valid,
                "shape": shape_distance,
                "shape_valid": shape_valid,
            }

        labels_cpu = labels.detach().cpu().tolist()
        y_true.extend(int(value) for value in labels_cpu)
        for variant in VARIANTS:
            y_pred[variant].extend(int(value) for value in batch_metrics[variant]["pred"].detach().cpu().tolist())

        for row, class_id_raw in enumerate(labels_cpu):
            class_id = int(class_id_raw)
            source_center, source_support = source_pse.center(class_id)
            sample = {
                "parcel_index": int(parcels[row].item()),
                "class_id": class_id,
                "class_name": classes[class_id],
                "estimation_member": class_id in estimation_members,
                "comparison_role": visproto.phase_diagnostic_role(
                    class_id, tuple(estimation_members)
                ),
                "class_center_valid": bool(class_centers[class_id].get("valid", False)),
                "class_center_reject_reason": class_centers[class_id].get("reject_reason"),
            }
            tokens_cpu = backbone.tokens[row].detach().cpu()
            mask_cpu = mask[row].detach().cpu()
            for variant in VARIANTS:
                metrics = batch_metrics[variant]
                pred = int(metrics["pred"][row].item())
                prob = float(metrics["probs"][row, class_id].item())
                fused = float(metrics["fused"][row].item())
                trend = float(metrics["trend"][row].item()) if bool(metrics["trend_valid"][row].item()) else float("nan")
                shape = float(metrics["shape"][row].item()) if bool(metrics["shape_valid"][row].item()) else float("nan")
                pos_cpu = positions[variant][row].detach().cpu()
                trajectory, trajectory_support, _ = visproto.canonicalize_pse_tokens(
                    {"pse_tokens": tokens_cpu, "positions": pos_cpu, "mask": mask_cpu},
                    positions_key="positions", grid_size=pse_grid_size,
                )
                _, pse_l2, _ = visproto.pse_integrated_distance(
                    source_center, source_support, trajectory, trajectory_support
                )
                true_prob[variant][class_id].append(prob)
                scalar_metrics[variant][class_id]["fused"].append(fused)
                scalar_metrics[variant][class_id]["pse"].append(pse_l2)
                scalar_metrics[variant][class_id]["trend"].append(trend)
                scalar_metrics[variant][class_id]["shape"].append(shape)
                target_pse[variant].add(
                    class_id=class_id, tokens=tokens_cpu, positions=pos_cpu, mask=mask_cpu
                )
                sample[f"prediction_{variant}"] = pred
                sample[f"true_probability_{variant}"] = prob
                sample[f"fused_distance_{variant}"] = fused
                sample[f"pse_l2_{variant}"] = pse_l2
                sample[f"trend_srvf_distance_{variant}"] = trend
                sample[f"shape_srvf_distance_{variant}"] = shape
            sample_rows.append(sample)

    class_rows: List[dict] = []
    for class_id, class_name in enumerate(classes):
        if class_id not in class_centers or class_id not in source_pse.sum_by_class:
            continue
        support = sum(int(value == class_id) for value in y_true)
        if support == 0:
            continue
        member = class_id in estimation_members
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "support": support,
            "estimation_member": member,
            "comparison_role": visproto.phase_diagnostic_role(class_id, tuple(estimation_members)),
            "class_center_valid": bool(class_centers[class_id].get("valid", False)),
            "class_center_reject_reason": class_centers[class_id].get("reject_reason"),
        }
        source_center, source_support = source_pse.center(class_id)
        for variant in VARIANTS:
            cls = _binary_class_metrics(y_true, y_pred[variant], class_id)
            row[f"precision_{variant}"] = cls["precision"]
            row[f"recall_{variant}"] = cls["recall"]
            row[f"f1_{variant}"] = cls["f1"]
            row[f"true_probability_{variant}"] = _mean(true_prob[variant][class_id])
            for metric in ("fused", "pse", "trend", "shape"):
                row[f"{metric}_distance_{variant}"] = _mean(
                    scalar_metrics[variant][class_id][metric]
                )
            target_center, target_support = target_pse[variant].center(class_id)
            _, class_mean_l2, _ = visproto.pse_integrated_distance(
                source_center, source_support, target_center, target_support
            )
            row[f"pse_class_mean_l2_{variant}"] = class_mean_l2

        class_recall_gain = row["recall_class_center"] - row["recall_no_phase"]
        group_recall_gain = row["recall_group_center"] - row["recall_no_phase"]
        row["recall_gain_class_center"] = class_recall_gain
        row["recall_gain_group_center"] = group_recall_gain
        row.update(visproto.phase_gain_gap_fields(
            class_gain=class_recall_gain,
            group_gain=group_recall_gain,
            estimation_member=member,
        ))
        for metric in ("fused", "pse", "trend", "shape"):
            base = row[f"{metric}_distance_no_phase"]
            class_after = row[f"{metric}_distance_class_center"]
            group_after = row[f"{metric}_distance_group_center"]
            row[f"{metric}_reduction_class_center"] = _safe_relative_reduction(base, class_after)
            row[f"{metric}_reduction_group_center"] = _safe_relative_reduction(base, group_after)
        base = row["pse_class_mean_l2_no_phase"]
        row["pse_class_mean_reduction_class_center"] = _safe_relative_reduction(
            base, row["pse_class_mean_l2_class_center"]
        )
        row["pse_class_mean_reduction_group_center"] = _safe_relative_reduction(
            base, row["pse_class_mean_l2_group_center"]
        )
        class_rows.append(row)

    global_summary = {
        "num_target_samples": len(y_true),
        "variants": {},
    }
    for variant in VARIANTS:
        correct = sum(int(t == p) for t, p in zip(y_true, y_pred[variant]))
        recalls = [_binary_class_metrics(y_true, y_pred[variant], c)["recall"]
                   for c in range(len(classes))]
        f1s = [_binary_class_metrics(y_true, y_pred[variant], c)["f1"]
               for c in range(len(classes))]
        global_summary["variants"][variant] = {
            "accuracy": correct / len(y_true) if y_true else float("nan"),
            "macro_recall": _mean(recalls),
            "macro_f1": _mean(f1s),
            "mean_true_probability": _mean(
                value for class_values in true_prob[variant].values() for value in class_values
            ),
        }
    return class_rows, sample_rows, global_summary


def _plot_gain_by_class(path: Path, rows: Sequence[dict], *, metric: str, ylabel: str, dpi: int) -> None:
    labels = [f"{row['class_id']}:{row['class_name']}" for row in rows]
    class_values = [float(row[f"{metric}_class_center"]) for row in rows]
    group_values = [float(row[f"{metric}_group_center"]) for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.0, 5.8), constrained_layout=True)
    ax.bar(x - width / 2, class_values, width, label="class-center Phase (oracle)")
    ax.bar(x + width / 2, group_values, width, label="M=1 group-center Phase")
    ax.axhline(0.0, linestyle=":", linewidth=1.0)
    ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _joint_diagnosis_outputs(
    output_dir: Path,
    *,
    geometry_rows: Sequence[dict],
    class_rows: Sequence[dict],
    dpi: int,
) -> List[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = {int(row["class_id"]): row for row in geometry_rows}
    rows: List[dict] = []
    for downstream in class_rows:
        class_id = int(downstream["class_id"])
        phase = geometry[class_id]
        row = {
            "class_id": class_id,
            "class_name": downstream["class_name"],
            "estimation_member": downstream["estimation_member"],
            "comparison_role": downstream["comparison_role"],
            "class_center_valid": downstream["class_center_valid"],
            "reject_reason": downstream["class_center_reject_reason"],
            "d_to_identity": phase["d_to_identity"],
            "d_to_group_center": phase["d_to_group_center"],
            "recall_gain_class_center": downstream["recall_gain_class_center"],
            "recall_gain_group_center": downstream["recall_gain_group_center"],
            "compression_loss": downstream.get("compression_loss"),
            "application_gap": downstream.get("application_gap"),
            "pse_reduction_class_center": downstream["pse_reduction_class_center"],
            "pse_reduction_group_center": downstream["pse_reduction_group_center"],
            "fused_reduction_class_center": downstream["fused_reduction_class_center"],
            "fused_reduction_group_center": downstream["fused_reduction_group_center"],
            "trend_reduction_class_center": downstream["trend_reduction_class_center"],
            "trend_reduction_group_center": downstream["trend_reduction_group_center"],
            "shape_reduction_class_center": downstream["shape_reduction_class_center"],
            "shape_reduction_group_center": downstream["shape_reduction_group_center"],
        }
        rows.append(row)
    _write_csv(output_dir / "compression_and_application_gap_by_class.csv", rows)

    keys = [
        "d_to_identity",
        "d_to_group_center",
        "recall_gain_class_center",
        "recall_gain_group_center",
        "fused_reduction_class_center",
        "fused_reduction_group_center",
        "shape_reduction_class_center",
        "shape_reduction_group_center",
    ]
    raw = np.asarray([[float(row[key]) for key in keys] for row in rows], dtype=np.float64)
    normalized = raw.copy()
    for column in range(normalized.shape[1]):
        values = normalized[:, column]
        finite = np.isfinite(values)
        if not finite.any():
            normalized[:, column] = 0.5
            continue
        low = float(np.min(values[finite]))
        high = float(np.max(values[finite]))
        if high - low <= 1e-12:
            normalized[:, column] = 0.5
        else:
            normalized[:, column] = (values - low) / (high - low)
        normalized[~finite, column] = 0.5
    labels = [f"{row['class_id']}:{row['class_name']}" for row in rows]
    fig, ax = plt.subplots(figsize=(13.5, 7.0), constrained_layout=True)
    image = ax.imshow(normalized, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(rows)), labels, fontsize=8)
    ax.set_xticks(np.arange(len(keys)), keys, rotation=55, ha="right", fontsize=8)
    ax.set_title("Phase-class diagnostic matrix (column-wise normalized color; raw values annotated)")
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            value = raw[i, j]
            text = "nan" if not math.isfinite(value) else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, label="column-wise normalized value")
    fig.savefig(output_dir / "phase_class_diagnostic_matrix.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return rows


def _write_readme(
    path: Path,
    *,
    checkpoint: Path,
    model_checkpoint: Path,
    source: str,
    target: str,
    samples_per_class: int,
) -> None:
    population = (
        "完整 held-out source-test / target-test"
        if samples_per_class <= 0
        else f"held-out test 每类最多 {samples_per_class} 个样本的固定均匀子集"
    )
    text = f"""# 05 Class-center vs M=1 group-center Phase diagnostic

## 实验目的

判断当前 M=1 是否真的是跨类别共享、且对分类有意义的 Domain Phase。实验不修改 M=1/M=2 决策，不训练模型，不刷新 Teacher/Stable Label。

## 输入

- Phase calibration checkpoint: `{checkpoint}`
- Frozen Stage1 model checkpoint: `{model_checkpoint}`
- source: `{source}`
- target: `{target}`
- population: {population}

PSE、LTAE、classifier 全部固定并处于 `eval()`。三路使用完全相同的 target-test 样本。

## 三路对照

1. `no_phase`: 原始 target 时间位置。
2. `class_center`: 使用 target true class 选择 `gamma_bar_c`，再用 `gamma_bar_c^-1(t_target)`。**这是 oracle-only diagnostic，不属于 UDA inference。**
3. `group_center`: 对所有类别使用最终确认的 M=1 group center `delta`，再用 `delta^-1(t_target)`。

## 最重要的语义区分

- `estimation_member=true`: 该类属于最终 `C_est`，`gamma_bar_c -> delta` 确实参与了 M=1 聚合。这里 `compression_loss` 才能解释为 **M=1 聚合压缩损失**。
- `estimation_member=false`: 该类没有参与 `delta` 的估计。这里不允许使用“压缩损失”表述；`application_gap` 只表示 **confirmed Domain Phase 向非估计类别扩展使用时的差异**。

## 目录

- `01_phase_geometry/`: class-center、identity、M=1 center 的 Phase 几何；包括 pairwise center distance 与 progressive stability。
- `02_downstream_three_way/`: No Phase / class-center / group-center 的 Recall、F1、true-class probability、PSE/LTAE(TSStructure fused)/T-SRVF/S-SRVF 距离。
- `03_joint_diagnosis/`: 将 Phase 几何和下游收益放在同一张诊断矩阵中。
- `sample_level.csv`: target-test 样本级三路预测和距离，仅用于 oracle diagnosis。
- `summary.json`: 全局协议和指标。
- `manifest.json`: 文件与实验元数据。

## identity 的解释边界

`d_gamma(gamma_bar_c, id)` 很小只支持“该类别 Phase 较弱/接近无需校正”的证据。**本实验没有为单类别定义 identity-confirmation threshold，也不会据此确认 M=0 或 identity group。**

`distance_to_identity_and_group.png` 中，class diameter threshold 不会被画成参考线，因为它约束的是同一类别 sample-level candidate warps 的 diameter，统计对象与 `d(gamma_bar_c,id)` / `d(gamma_bar_c,delta)` 不同。若 checkpoint 中存在 `phase_global_radius`，它只在 `d(gamma_bar_c,delta)` 子图中作为 M=1 group radius 参考线。

## 结果判读

- `c in C_est`，class-center 有益而 group-center 有害：直接支持 **M=1 聚合造成 Phase 信息损失**。
- `c not in C_est`，class-center 有益而 group-center 有害：支持 **共享 Phase 向该类别扩展使用不合理**，不属于 M=1 压缩问题。
- class-center 与 group-center 都有害：registration-derived Phase 对该类别本身缺乏分类价值；单纯增加组数未必能解决。
- `gamma_bar_c` 接近 identity，No Phase≈class-center，而 group-center 有害：支持弱 Phase / 近恒等结构的证据，但不能单类确认 identity group。
- 多个可靠 class centers 呈两个稳定、内部紧密且彼此分离的块：支持下一步正式比较 M=1 与 M=2。
- 所有可靠 class centers 都紧密围绕 delta，但分类收益仍正负相反：分组不是主要问题，应转向“函数配准意义上的共享 Phase 是否应该应用于所有分类类别”。
"""
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    runtime = checkpoint.get("runtime_config")
    if not isinstance(runtime, dict):
        raise ValueError("calibration checkpoint must contain runtime_config")
    classes = [str(value) for value in runtime["classes"]]
    source = str(runtime["source"])
    target = str(runtime["target"])
    seed = int(runtime["seed"])
    data_root = str(args.data_root or runtime["data_root"])
    fold = int(args.fold)
    val_ratio = float(runtime.get("val_ratio", 0.1))
    test_ratio = float(runtime.get("test_ratio", 0.2))
    closed_set = bool(runtime.get("closed_set", True))
    combine = bool(runtime.get("combine_spring_and_winter", False))
    time_mode = str(runtime.get("time_coordinate_mode", "canonical_day_of_year"))

    group = _final_m1_group(checkpoint)
    class_centers = _class_center_payloads(checkpoint)
    missing = sorted(set(range(len(classes))) - set(class_centers))
    if missing:
        raise ValueError("calibration checkpoint is missing class centers: " + ",".join(map(str, missing)))

    reference_distance = None
    if args.reference_phase_checkpoint is not None:
        reference = torch.load(
            args.reference_phase_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        reference_group = _final_m1_group(reference)
        reference_distance = float(phase_distance(
            group["center_gamma"], reference_group["center_gamma"]
        ).item())

    device = torch.device(args.device)
    model_checkpoint_path = args.model_checkpoint.resolve()
    model_checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    model = phasevis._build_model(
        runtime, checkpoint, device, model_checkpoint=model_checkpoint
    )
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
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
    source_meta = phasevis._metadata_dataset(
        data_root, source, classes, splits[source]["test"], closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
    )
    target_meta = phasevis._metadata_dataset(
        data_root, target, classes, splits[target]["test"], closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
    )
    class_ids = tuple(range(len(classes)))
    source_parcels = _selected_test_parcels(source_meta, class_ids, args.samples_per_class)
    target_parcels = _selected_test_parcels(target_meta, class_ids, args.samples_per_class)
    source_loader = phasevis._selected_loader(
        data_root, source, classes, source_parcels, closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    target_loader = phasevis._selected_loader(
        data_root, target, classes, target_parcels, closed_set=closed_set,
        combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    output_dir = args.output_dir.resolve()
    phase_dir = output_dir / "01_phase_geometry"
    downstream_dir = output_dir / "02_downstream_three_way"
    joint_dir = output_dir / "03_joint_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "CLASS_CENTER_PHASE_DIAG_START|"
        f"checkpoint={checkpoint_path}|source={source}|target={target}|fold={fold}|"
        f"source_test={len(source_parcels)}|target_test={len(target_parcels)}|"
        f"samples_per_class={args.samples_per_class}|oracle_class_center=true",
        flush=True,
    )

    geometry_rows, progressive_rows = _phase_geometry_outputs(
        phase_dir, checkpoint=checkpoint, classes=classes, group=group,
        class_centers=class_centers, dpi=args.dpi,
    )
    source_pse = _build_source_pse_centers(
        model, source_loader, device=device, grid_size=args.pse_grid_size
    )
    class_rows, sample_rows, global_summary = _downstream_three_way(
        model=model, target_loader=target_loader, source_pse=source_pse,
        checkpoint=checkpoint, classes=classes, class_centers=class_centers,
        group=group, device=device, pse_grid_size=args.pse_grid_size,
    )
    _write_csv(downstream_dir / "class_comparison.csv", class_rows)
    _write_csv(output_dir / "sample_level.csv", sample_rows)

    _plot_gain_by_class(
        downstream_dir / "classification_gain_by_class.png", class_rows,
        metric="recall_gain", ylabel="Recall gain vs No Phase", dpi=args.dpi,
    )
    for metric, label in (
        ("pse_reduction", "PSE sample-distance relative reduction vs No Phase"),
        ("fused_reduction", "LTAE fused-distance relative reduction vs No Phase"),
        ("trend_reduction", "T-SRVF distance relative reduction vs No Phase"),
        ("shape_reduction", "S-SRVF distance relative reduction vs No Phase"),
    ):
        _plot_gain_by_class(
            downstream_dir / f"{metric}_by_class.png", class_rows,
            metric=metric, ylabel=label, dpi=args.dpi,
        )

    joint_rows = _joint_diagnosis_outputs(
        joint_dir, geometry_rows=geometry_rows, class_rows=class_rows, dpi=args.dpi
    )
    summary = {
        "experiment": "05_class_center_vs_group_center_phase_diagnostic",
        "checkpoint": str(checkpoint_path),
        "model_checkpoint": str(model_checkpoint_path),
        "reference_phase_checkpoint": (
            None if args.reference_phase_checkpoint is None
            else str(args.reference_phase_checkpoint.resolve())
        ),
        "recalibrated_group_to_reference_phase_distance": reference_distance,
        "source": source,
        "target": target,
        "fold": fold,
        "samples_per_class": int(args.samples_per_class),
        "full_held_out_test": bool(args.samples_per_class <= 0),
        "source_test_samples": int(len(source_parcels)),
        "target_test_samples": int(len(target_parcels)),
        "oracle_only_class_center_selection": True,
        "model_updates": 0,
        "teacher_refreshes": 0,
        "stable_label_refreshes": 0,
        "phase_decision_modified": False,
        "m": int(_phase_payload(checkpoint)["m"]),
        "estimation_members": [int(value) for value in group.get("member_classes", ())],
        "progressive_stage_count": len(checkpoint.get("phase_state_progressive", ())),
        "global_downstream": global_summary,
        "class_results": joint_rows,
    }
    _json_dump(output_dir / "summary.json", summary)
    manifest = {
        "experiment": summary["experiment"],
        "oracle_only": True,
        "files": {
            "01_phase_geometry": "Phase-space class/group/identity and progressive diagnostics",
            "02_downstream_three_way": "No Phase vs oracle class-center vs M=1 group-center",
            "03_joint_diagnosis": "joint Phase/downstream diagnostic matrix",
            "sample_level.csv": "same held-out target-test population, sample-level three-way metrics",
            "summary.json": "machine-readable experiment summary",
            "README_中文说明.md": "protocol, semantics, and result interpretation",
        },
    }
    _json_dump(output_dir / "manifest.json", manifest)
    _write_readme(
        output_dir / "README_中文说明.md", checkpoint=checkpoint_path,
        model_checkpoint=model_checkpoint_path, source=source, target=target,
        samples_per_class=args.samples_per_class,
    )
    print(
        "CLASS_CENTER_PHASE_DIAG_COMPLETE|"
        f"output={output_dir}|target_test={len(target_parcels)}|"
        f"reference_group_distance={reference_distance}",
        flush=True,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="new calibration checkpoint containing class centers and progressive states")
    parser.add_argument("--model-checkpoint", type=Path, required=True,
                        help="frozen Stage1 checkpoint used for all three downstream branches")
    parser.add_argument("--reference-phase-checkpoint", type=Path, default=None,
                        help="optional old Stage2 checkpoint; only checks final M=1 group-center reproducibility")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--samples-per-class", type=int, default=0,
                        help="0 means complete held-out test; positive values select a deterministic per-class subset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pse-grid-size", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.samples_per_class < 0:
        raise ValueError("samples-per-class must be >= 0")
    run(args)


if __name__ == "__main__":
    main()
