#!/usr/bin/env python3
"""Oracle visual audit of Stage-2 Domain Phase alignment.

This is a post-hoc diagnostic only. Target true labels are used exclusively to
pair the same semantic class across source and target. They never enter
training, pseudo-label construction, Domain Phase estimation, or checkpoint
selection.

The script adapts the repository's existing fixed-PCA spaghetti-plot machinery
for a saved Stage-2 checkpoint. For every class covered by a confirmed
non-identity Domain Phase group it visualizes:

1. source / target-before / target-after Shape-SRVF spaghetti curves;
2. source / target-before / target-after Trend-SRVF spaghetti curves;
3. source / target-before / target-after raw Structure-token trajectories,
   where the target-after panel changes only the observation positions passed
   to Time2Vec/LTAE (values are unchanged);
4. class-level before/after support-aware distances to the frozen source
   prototype; and
5. the confirmed Domain Phase gamma itself and its displacement in days.

PCA bases are fitted from SOURCE rows only, then frozen for all target views.
This prevents the alignment under examination from rotating the visualization
basis itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import GroupByShapesBatchSampler, PixelSetData, worker_init_fn
from methods.structure_da.feature_snapshots import fit_deterministic_pca, project_features
from methods.structure_da.full_model import TSStructureModel
from methods.structure_da.confirmed_phase_view import align_target_positions_to_source
from methods.structure_da.phase_registration import (
    resample_gamma,
    warp_q_gamma,
    warp_support_gamma,
)
from methods.structure_da.prototype_bank import support_aware_q_distance
from transforms import Identity, Normalize, ToTensor




def _parse_int_list(value: str) -> Tuple[int, ...]:
    items = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated class list")
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("class list must not contain duplicates")
    return items


def _status_text(value) -> str:
    if hasattr(value, "value"):
        value = value.value
    return str(value).lower()


def _checkpoint_group_payloads(checkpoint: dict) -> List[dict]:
    phase = checkpoint.get("phase_state")
    if not isinstance(phase, dict):
        raise ValueError("checkpoint does not contain a serialized phase_state dictionary")
    groups = phase.get("groups")
    if not isinstance(groups, (tuple, list)):
        raise ValueError("checkpoint phase_state.groups is missing or invalid")
    confirmed = [group for group in groups if _status_text(group.get("status")) == "confirmed"]
    if not confirmed:
        raise ValueError("checkpoint contains no confirmed non-identity Domain Phase group")
    return confirmed


def _class_to_group(groups: Sequence[dict]) -> Dict[int, dict]:
    mapping: Dict[int, dict] = {}
    for group in groups:
        for class_id in group.get("member_classes", ()):
            class_id = int(class_id)
            if class_id in mapping:
                raise ValueError("a class belongs to more than one confirmed phase group")
            mapping[class_id] = group
    return mapping


def _runtime_value(runtime: dict, name: str, default=None):
    value = runtime.get(name, default)
    if value is None:
        return default
    return value


def _build_model(runtime: dict, checkpoint: dict, device: torch.device) -> TSStructureModel:
    model = TSStructureModel(
        num_classes=int(runtime["num_classes"]),
        input_dim=int(_runtime_value(runtime, "input_dim", 10)),
        with_extra=bool(_runtime_value(runtime, "with_extra", False)),
        time_reference=float(_runtime_value(runtime, "time_reference", 0.0)),
        time_scale=float(_runtime_value(runtime, "time_scale", 365.0)),
        tau_fast_init=float(_runtime_value(runtime, "tau_fast_init", 0.05)),
        tau_slow_init=float(_runtime_value(runtime, "tau_slow_init", 0.20)),
        tau_min=float(_runtime_value(runtime, "tau_min", 1e-4)),
        delta_tau_min=float(_runtime_value(runtime, "delta_tau_min", 1e-4)),
        trend_num_basis=int(_runtime_value(runtime, "trend_num_basis", 12)),
        structure_num_basis=int(_runtime_value(runtime, "structure_num_basis", 12)),
        canonical_grid_size=int(_runtime_value(runtime, "canonical_grid_size", 64)),
        roughness_grid_size=int(_runtime_value(runtime, "roughness_grid_size", 256)),
        trend_smoothing=float(_runtime_value(runtime, "trend_smoothing", 1e-2)),
        structure_smoothing=float(_runtime_value(runtime, "structure_smoothing", 1e-3)),
        n_head=int(_runtime_value(runtime, "n_head", 16)),
        d_k=int(_runtime_value(runtime, "d_k", 8)),
        d_model=int(_runtime_value(runtime, "d_model", 256)),
        ltae_mlp=tuple(int(v) for v in _runtime_value(runtime, "ltae_mlp", [256, 128])),
        dropout=float(_runtime_value(runtime, "dropout", 0.2)),
        classifier_hidden=tuple(
            int(v) for v in _runtime_value(runtime, "classifier_hidden", [64, 32])
        ),
        max_initial_frequency=float(
            _runtime_value(runtime, "time2vec_max_frequency", 16.0)
        ),
    )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain state_dict")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _eligible_parcels(
    data_root: str,
    domain: str,
    classes: Sequence[str],
    *,
    closed_set: bool,
    combine_spring_and_winter: bool,
    time_coordinate_mode: str,
) -> np.ndarray:
    dataset = PixelSetData(
        data_root=data_root,
        dataset_name=domain,
        classes=list(classes),
        transform=None,
        indices=None,
        with_extra=False,
        closed_set=closed_set,
        combine_spring_and_winter=combine_spring_and_winter,
        time_coordinate_mode=time_coordinate_mode,
    )
    return dataset.get_parcel_indices()


def _reconstruct_fold_train_indices(
    source_parcels: np.ndarray,
    target_parcels: np.ndarray,
    *,
    source: str,
    target: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    fold: int,
) -> Dict[str, set]:
    """Reproduce create_train_val_test_folds exactly for the requested fold."""
    rng = random.Random(seed)
    requested = None
    for fold_index in range(fold + 1):
        splits: Dict[str, set] = {}
        for name, raw in ((source, source_parcels), (target, target_parcels)):
            indices = [int(value) for value in raw.tolist()]
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val
            rng.shuffle(indices)
            splits[name] = set(indices[:n_train])
        requested = splits
    if requested is None:
        raise RuntimeError("failed to reconstruct requested fold")
    return requested


def _metadata_train_dataset(
    data_root: str,
    domain: str,
    classes: Sequence[str],
    train_indices: set,
    *,
    closed_set: bool,
    combine_spring_and_winter: bool,
    time_coordinate_mode: str,
) -> PixelSetData:
    return PixelSetData(
        data_root=data_root,
        dataset_name=domain,
        classes=list(classes),
        transform=None,
        indices=train_indices,
        with_extra=False,
        closed_set=closed_set,
        combine_spring_and_winter=combine_spring_and_winter,
        time_coordinate_mode=time_coordinate_mode,
    )


def _uniform_selected_parcels(
    dataset: PixelSetData,
    classes: Sequence[int],
    samples_per_class: int,
) -> np.ndarray:
    labels = dataset.get_labels()
    parcels = dataset.get_parcel_indices()
    selected: List[np.ndarray] = []
    for class_id in classes:
        positions = np.flatnonzero(labels == int(class_id))
        if not len(positions):
            continue
        positions = positions[np.argsort(parcels[positions], kind="stable")]
        count = min(samples_per_class, len(positions))
        pick = np.linspace(0, len(positions) - 1, num=count).round().astype(np.int64)
        selected.append(parcels[positions[pick]])
    if not selected:
        raise ValueError("no selected samples for requested classes")
    return np.concatenate(selected).astype(np.int64, copy=False)


def _selected_loader(
    data_root: str,
    domain: str,
    classes: Sequence[str],
    parcel_indices: np.ndarray,
    *,
    closed_set: bool,
    combine_spring_and_winter: bool,
    time_coordinate_mode: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    transform = transforms.Compose([Identity(), Normalize(), ToTensor()])
    dataset = PixelSetData(
        data_root=data_root,
        dataset_name=domain,
        classes=list(classes),
        transform=transform,
        indices=set(int(value) for value in parcel_indices.tolist()),
        with_extra=False,
        closed_set=closed_set,
        combine_spring_and_winter=combine_spring_and_winter,
        time_coordinate_mode=time_coordinate_mode,
    )
    return DataLoader(
        dataset=dataset,
        batch_sampler=GroupByShapesBatchSampler(
            dataset, batch_size, by_time=True, by_pixel_dim=True
        ),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )


def _move_batch(batch: dict, device: torch.device) -> dict:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device=device) if isinstance(value, Tensor) else value
    return result


def _integration_weights(grid_size: int, reference: Tensor) -> Tensor:
    weights = torch.ones(grid_size, device=reference.device, dtype=reference.dtype)
    if grid_size > 1:
        weights[[0, -1]] *= 0.5
    return weights / weights.sum()


def _resampled_group_gamma(group: dict, grid: Tensor) -> Tensor:
    gamma = group.get("center_gamma")
    if not isinstance(gamma, Tensor):
        raise ValueError("confirmed group center_gamma is missing")
    gamma = gamma.detach().to(device=grid.device, dtype=grid.dtype)
    registration_grid = torch.linspace(
        0.0, 1.0, gamma.numel(), device=grid.device, dtype=grid.dtype
    )
    return resample_gamma(gamma, registration_grid, grid)


def _direct_resample_gamma(group: dict, grid: Tensor) -> Tensor:
    """Evaluate saved gamma(u) on ``grid`` without inverting it.

    This is intentionally implemented locally for audit comparison. The current
    repository ``resample_gamma`` is also evaluated separately so the figure can
    reveal a direction/resampling mismatch without modifying training code.
    """
    gamma = group.get("center_gamma")
    if not isinstance(gamma, Tensor):
        raise ValueError("confirmed group center_gamma is missing")
    gamma = gamma.detach().to(device=grid.device, dtype=grid.dtype)
    source_grid = torch.linspace(
        0.0, 1.0, gamma.numel(), device=grid.device, dtype=grid.dtype
    )
    query = grid.to(device=gamma.device, dtype=gamma.dtype)
    upper = torch.searchsorted(source_grid, query, right=True).clamp(
        min=1, max=source_grid.numel() - 1
    )
    lower = upper - 1
    x0 = source_grid[lower]
    x1 = source_grid[upper]
    y0 = gamma[lower]
    y1 = gamma[upper]
    fraction = (query - x0) / (x1 - x0).clamp_min(1e-12)
    output = y0 + fraction * (y1 - y0)
    output = torch.where(query == 0, torch.zeros_like(output), output)
    output = torch.where(query == 1, torch.ones_like(output), output)
    return output.clamp(0.0, 1.0)


def _append_record(storage: Dict[int, List[dict]], class_id: int, record: dict) -> None:
    storage.setdefault(class_id, []).append(record)


@torch.no_grad()
def _collect_geometry(
    model: TSStructureModel,
    loader: DataLoader,
    *,
    device: torch.device,
    target: bool,
    class_to_group: Dict[int, dict],
) -> Dict[int, List[dict]]:
    records: Dict[int, List[dict]] = {}
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        pixels = batch["pixels"]
        valid_pixels = batch["valid_pixels"]
        positions = batch["positions"]
        extra = batch.get("extra")
        backbone = model.forward_backbone(
            pixels, valid_pixels, positions, extra, time_mask=batch.get("time_mask")
        )
        output = model.forward_from_backbone(
            backbone, positions, extra, return_geometry=True
        )
        if output.geometry is None:
            raise RuntimeError("phase visualization requires functional geometry")
        trend_tokens, structure_tokens = model._trend_and_structure(backbone)
        labels = batch["label"].detach().cpu().long()
        parcels = batch["parcel_index"].detach().cpu().long()
        grid = output.geometry.canonical_grid.detach()
        for row in range(len(labels)):
            class_id = int(labels[row].item())
            record = {
                "parcel_index": int(parcels[row].item()),
                "positions": backbone.normalized_positions[row].detach().cpu(),
                "mask": backbone.time_mask[row].detach().cpu(),
                "trend_tokens": trend_tokens[row].detach().cpu(),
                "structure_tokens": structure_tokens[row].detach().cpu(),
                "trend_q_before": output.geometry.trend_srvf[row].detach().cpu(),
                "trend_support_before": output.geometry.trend_support[row].detach().cpu(),
                "shape_q_before": output.geometry.structure_srvf[row].detach().cpu(),
                "shape_support_before": output.geometry.structure_support[row].detach().cpu(),
                "trend_valid": bool(output.geometry.trend_valid[row].item()),
                "shape_valid": bool(output.geometry.structure_valid[row].item()),
                "grid": grid.detach().cpu(),
            }
            if target:
                group = class_to_group.get(class_id)
                if group is None:
                    continue
                gamma_grid = _resampled_group_gamma(group, grid)
                gamma_grid_direct = _direct_resample_gamma(group, grid)
                aligned_positions = align_target_positions_to_source(
                    backbone.normalized_positions[row : row + 1],
                    backbone.time_mask[row : row + 1],
                    group["center_gamma"],
                )[0]
                record.update({
                    "group_id": int(group["group_id"]),
                    "positions_after": aligned_positions.detach().cpu(),
                    # CURRENT IMPLEMENTATION: uses repository resample_gamma.
                    "trend_q_after": warp_q_gamma(
                        output.geometry.trend_srvf[row], gamma_grid
                    ).squeeze(0).detach().cpu(),
                    "trend_support_after": warp_support_gamma(
                        output.geometry.trend_support[row], gamma_grid, grid
                    ).detach().cpu(),
                    "shape_q_after": warp_q_gamma(
                        output.geometry.structure_srvf[row], gamma_grid
                    ).squeeze(0).detach().cpu(),
                    "shape_support_after": warp_support_gamma(
                        output.geometry.structure_support[row], gamma_grid, grid
                    ).detach().cpu(),
                    # AUDIT REFERENCE: directly evaluates gamma(u) on the Shape grid.
                    "trend_q_after_direct": warp_q_gamma(
                        output.geometry.trend_srvf[row], gamma_grid_direct
                    ).squeeze(0).detach().cpu(),
                    "trend_support_after_direct": warp_support_gamma(
                        output.geometry.trend_support[row], gamma_grid_direct, grid
                    ).detach().cpu(),
                    "shape_q_after_direct": warp_q_gamma(
                        output.geometry.structure_srvf[row], gamma_grid_direct
                    ).squeeze(0).detach().cpu(),
                    "shape_support_after_direct": warp_support_gamma(
                        output.geometry.structure_support[row], gamma_grid_direct, grid
                    ).detach().cpu(),
                    "gamma_grid_current": gamma_grid.detach().cpu(),
                    "gamma_grid_direct": gamma_grid_direct.detach().cpu(),
                })
            _append_record(records, class_id, record)
    for class_id in records:
        records[class_id].sort(key=lambda item: item["parcel_index"])
    return records


def _source_pca(
    source_records: Dict[int, List[dict]],
    *,
    feature_key: str,
    support_key: Optional[str],
    token_mask: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    for class_records in source_records.values():
        for record in class_records:
            values = record[feature_key].numpy()
            if token_mask:
                mask = record["mask"].numpy().astype(bool)
                values = values[mask]
                weight = np.full(len(values), 1.0 / max(len(values), 1), dtype=np.float64)
            else:
                if support_key is None:
                    weight = np.ones(len(values), dtype=np.float64)
                else:
                    support = record[support_key].numpy().astype(np.float64)
                    weight = np.maximum(support, 1e-6)
            if len(values):
                rows.append(values.astype(np.float32))
                weights.append(weight)
    if not rows:
        raise ValueError("source PCA has no rows")
    fit = fit_deterministic_pca(
        np.concatenate(rows, axis=0),
        num_components=2,
        weights=np.concatenate(weights, axis=0),
    )
    return fit.mean, fit.components, fit.explained_variance_ratio


def _project(values: Tensor, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return project_features(values.numpy().astype(np.float32), mean, components)


def _support_weighted_center(records: Sequence[dict], q_key: str, support_key: str) -> Tuple[Tensor, Tensor]:
    q = torch.stack([record[q_key].double() for record in records])
    support = torch.stack([record[support_key].double() for record in records])
    denominator = support.sum(dim=0)
    center = (q * support.unsqueeze(-1)).sum(dim=0) / (denominator.unsqueeze(-1) + 1e-8)
    mean_support = support.mean(dim=0)
    return center.float(), mean_support.float()


def _distance_stats(
    records: Sequence[dict],
    *,
    q_before_key: str,
    support_before_key: str,
    q_after_key: str,
    support_after_key: str,
    prototype_q: Tensor,
    prototype_support: Tensor,
) -> dict:
    q_before = torch.stack([record[q_before_key].float() for record in records])
    s_before = torch.stack([record[support_before_key].float() for record in records])
    q_after = torch.stack([record[q_after_key].float() for record in records])
    s_after = torch.stack([record[support_after_key].float() for record in records])
    prototype_q = prototype_q.float().unsqueeze(0)
    prototype_support = prototype_support.float().unsqueeze(0)
    weights = _integration_weights(q_before.shape[1], q_before)
    before_output = support_aware_q_distance(
        q_before, prototype_q, s_before, prototype_support, weights
    )
    after_output = support_aware_q_distance(
        q_after, prototype_q, s_after, prototype_support, weights
    )
    valid = before_output.valid[:, 0] & after_output.valid[:, 0]
    before = before_output.distance[:, 0][valid]
    after = after_output.distance[:, 0][valid]
    if not len(before):
        return {
            "valid_samples": 0,
            "before_mean": None,
            "before_median": None,
            "after_mean": None,
            "after_median": None,
            "mean_relative_reduction": None,
            "improvement_rate": None,
            "class_center_before": None,
            "class_center_after": None,
        }
    before_center, before_support = _support_weighted_center(
        records, q_before_key, support_before_key
    )
    after_center, after_support = _support_weighted_center(
        records, q_after_key, support_after_key
    )
    center_before = support_aware_q_distance(
        before_center.unsqueeze(0), prototype_q,
        before_support.unsqueeze(0), prototype_support,
        weights,
    ).distance[0, 0]
    center_after = support_aware_q_distance(
        after_center.unsqueeze(0), prototype_q,
        after_support.unsqueeze(0), prototype_support,
        weights,
    ).distance[0, 0]
    return {
        "valid_samples": int(valid.sum().item()),
        "before_mean": float(before.mean().item()),
        "before_median": float(before.median().item()),
        "after_mean": float(after.mean().item()),
        "after_median": float(after.median().item()),
        "mean_relative_reduction": float(
            (before.mean() - after.mean()).item() / max(float(before.mean().item()), 1e-12)
        ),
        "improvement_rate": float((after < before).float().mean().item()),
        "class_center_before": float(center_before.item()),
        "class_center_after": float(center_after.item()),
    }


def _robust_limits(arrays: Iterable[np.ndarray], lower: float, upper: float) -> Tuple[float, float]:
    values = np.concatenate([
        np.asarray(value)[np.isfinite(value)] for value in arrays
        if np.isfinite(np.asarray(value)).any()
    ])
    low, high = np.percentile(values, (lower, upper))
    low, high = float(low), float(high)
    if low == high:
        pad = max(abs(low) * 0.05, 1e-6)
        low -= pad
        high += pad
    return low, high


def _class_stem(class_id: int, name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    return f"class_{class_id:02d}_{safe}" if safe else f"class_{class_id:02d}"


def _select_display(records: Sequence[dict], maximum: int) -> List[dict]:
    if len(records) <= maximum:
        return list(records)
    positions = np.linspace(0, len(records) - 1, num=maximum).round().astype(np.int64)
    return [records[int(index)] for index in positions]


def _plot_q_spaghetti(
    output_path: Path,
    *,
    title: str,
    source: Sequence[dict],
    target: Sequence[dict],
    q_prefix: str,
    support_prefix: str,
    source_prototype: Tensor,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
    display_samples: int,
    robust_lower: float,
    robust_upper: float,
    dpi: int,
) -> None:
    source_display = _select_display(source, display_samples)
    target_display = _select_display(target, display_samples)
    grid = source_display[0]["grid"].numpy() * 365.0
    proto_pc = _project(source_prototype.cpu(), pca_mean, pca_components)
    source_pc = [_project(item[f"{q_prefix}_before"], pca_mean, pca_components) for item in source_display]
    before_pc = [_project(item[f"{q_prefix}_before"], pca_mean, pca_components) for item in target_display]
    after_pc = [_project(item[f"{q_prefix}_after"], pca_mean, pca_components) for item in target_display]
    direct_pc = [_project(item[f"{q_prefix}_after_direct"], pca_mean, pca_components) for item in target_display]

    for mode in ("full", "robust"):
        figure, axes = plt.subplots(
            2, 4, figsize=(17.0, 7.0), sharex=True, sharey="row", constrained_layout=True
        )
        for component in range(2):
            all_values = [value[:, component] for value in source_pc + before_pc + after_pc + direct_pc]
            if mode == "robust":
                ylim = _robust_limits(all_values, robust_lower, robust_upper)
            else:
                finite = np.concatenate([v[np.isfinite(v)] for v in all_values])
                ylim = (float(finite.min()), float(finite.max()))
                if ylim[0] == ylim[1]:
                    ylim = (ylim[0] - 1e-6, ylim[1] + 1e-6)
            panels = (
                ("Source", source_display, source_pc, f"{support_prefix}_before"),
                ("Target before Phase", target_display, before_pc, f"{support_prefix}_before"),
                ("Target after CURRENT code", target_display, after_pc, f"{support_prefix}_after"),
                ("Target after direct gamma(u) resample", target_display, direct_pc, f"{support_prefix}_after_direct"),
            )
            for column, (panel_title, records, projected, support_key) in enumerate(panels):
                axis = axes[component, column]
                for record, values in zip(records, projected):
                    support = record[support_key].numpy()
                    plotted = values[:, component].copy()
                    plotted[support < 0.05] = np.nan
                    axis.plot(grid, plotted, linewidth=0.8, alpha=0.45)
                axis.plot(
                    grid, proto_pc[:, component],
                    linestyle="--", linewidth=2.0,
                    label="source prototype",
                )
                axis.set_ylim(*ylim)
                axis.set_title(panel_title)
                axis.grid(alpha=0.18)
                if component == 1:
                    axis.set_xlabel("canonical day of year")
                if column == 0:
                    axis.set_ylabel(f"PC{component + 1}")
                if component == 0 and column == 3:
                    axis.legend(loc="best")
        figure.suptitle(title)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path.with_name(output_path.stem + f"_{mode}.png"), dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _plot_q_mean_overlay(
    output_path: Path,
    *,
    title: str,
    target: Sequence[dict],
    q_prefix: str,
    support_prefix: str,
    source_prototype: Tensor,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
    dpi: int,
) -> None:
    grid = target[0]["grid"].numpy() * 365.0
    before_center, _ = _support_weighted_center(
        target, f"{q_prefix}_before", f"{support_prefix}_before"
    )
    after_center, _ = _support_weighted_center(
        target, f"{q_prefix}_after", f"{support_prefix}_after"
    )
    direct_center, _ = _support_weighted_center(
        target, f"{q_prefix}_after_direct", f"{support_prefix}_after_direct"
    )
    source_pc = _project(source_prototype.cpu(), pca_mean, pca_components)
    before_pc = _project(before_center, pca_mean, pca_components)
    after_pc = _project(after_center, pca_mean, pca_components)
    direct_pc = _project(direct_center, pca_mean, pca_components)
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True, constrained_layout=True)
    for component, axis in enumerate(axes):
        axis.plot(grid, source_pc[:, component], linewidth=2.0, label="source prototype")
        axis.plot(grid, before_pc[:, component], linewidth=2.0, label="target before")
        axis.plot(grid, after_pc[:, component], linewidth=2.0, label="target after CURRENT code")
        axis.plot(grid, direct_pc[:, component], linewidth=2.0, label="target after direct gamma(u)")
        axis.set_ylabel(f"PC{component + 1}")
        axis.grid(alpha=0.18)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("canonical day of year")
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_structure_position_spaghetti(
    output_path: Path,
    *,
    title: str,
    source: Sequence[dict],
    target: Sequence[dict],
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
    display_samples: int,
    robust_lower: float,
    robust_upper: float,
    dpi: int,
) -> None:
    source_display = _select_display(source, display_samples)
    target_display = _select_display(target, display_samples)

    def projected(record: dict) -> np.ndarray:
        return _project(record["structure_tokens"], pca_mean, pca_components)

    source_pc = [projected(record) for record in source_display]
    target_pc = [projected(record) for record in target_display]
    figure, axes = plt.subplots(
        2, 3, figsize=(13.2, 7.0), sharex=True, sharey="row", constrained_layout=True
    )
    for component in range(2):
        values = []
        for record, pc in list(zip(source_display, source_pc)) + list(zip(target_display, target_pc)):
            values.append(pc[record["mask"].numpy().astype(bool), component])
        ylim = _robust_limits(values, robust_lower, robust_upper)
        panels = (
            ("Source positions", source_display, source_pc, "positions"),
            ("Target original positions", target_display, target_pc, "positions"),
            ("Target positions after confirmed Phase", target_display, target_pc, "positions_after"),
        )
        for column, (panel_title, records, pcs, position_key) in enumerate(panels):
            axis = axes[component, column]
            for record, pc in zip(records, pcs):
                mask = record["mask"].numpy().astype(bool)
                x = record[position_key].numpy()[mask] * 365.0
                y = pc[mask, component]
                axis.plot(x, y, linewidth=0.8, alpha=0.45, marker=".", markersize=2.0)
            axis.set_ylim(*ylim)
            axis.set_title(panel_title)
            axis.grid(alpha=0.18)
            if component == 1:
                axis.set_xlabel("day of year used by Time2Vec/LTAE")
            if column == 0:
                axis.set_ylabel(f"Structure PC{component + 1}")
    figure.suptitle(title + " — target values unchanged; only x positions move")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _invert_gamma_numpy(gamma: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, gamma, grid)


def _plot_phase_groups(output_dir: Path, groups: Sequence[dict], dpi: int) -> List[dict]:
    summaries = []
    for group in groups:
        gamma = group["center_gamma"].detach().cpu().double().numpy()
        grid = np.linspace(0.0, 1.0, len(gamma), dtype=np.float64)
        inverse = _invert_gamma_numpy(gamma, grid)
        displacement = (gamma - grid) * 365.0
        inverse_displacement = (inverse - grid) * 365.0
        interior = (grid >= 0.10) & (grid <= 0.90)
        interior_shift = displacement[interior]
        median_shift = float(np.median(interior_shift))
        residual = interior_shift - median_shift
        summary = {
            "group_id": int(group["group_id"]),
            "member_classes": [int(v) for v in group.get("member_classes", ())],
            "gamma_points": int(len(gamma)),
            "source_to_target_shift_days_interior_median": median_shift,
            "source_to_target_shift_days_interior_p10": float(np.percentile(interior_shift, 10)),
            "source_to_target_shift_days_interior_p90": float(np.percentile(interior_shift, 90)),
            "source_to_target_translation_residual_std_days": float(np.std(residual)),
            "source_to_target_max_abs_shift_days": float(np.max(np.abs(displacement))),
            "target_to_source_max_abs_shift_days": float(np.max(np.abs(inverse_displacement))),
        }
        summaries.append(summary)
        figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.6), constrained_layout=True)
        axes[0].plot(grid * 365.0, gamma * 365.0, linewidth=2.0, label="confirmed gamma: source→target")
        axes[0].plot(grid * 365.0, grid * 365.0, linestyle="--", linewidth=1.2, label="identity")
        axes[0].set_xlabel("source canonical day")
        axes[0].set_ylabel("target canonical day")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.18)
        axes[1].plot(grid * 365.0, displacement, linewidth=2.0, label="source→target gamma(t)-t")
        axes[1].plot(grid * 365.0, inverse_displacement, linewidth=1.6, label="target→source gamma⁻¹(t)-t")
        axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
        axes[1].set_xlabel("canonical day")
        axes[1].set_ylabel("displacement (days)")
        axes[1].legend(loc="best")
        axes[1].grid(alpha=0.18)
        figure.suptitle(
            f"Confirmed Domain Phase group {int(group['group_id'])}; "
            f"classes={','.join(str(v) for v in group.get('member_classes', ()))}"
        )
        path = output_dir / "phase_groups" / f"group_{int(group['group_id']):02d}_gamma.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
    return summaries


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("summary rows are empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    runtime = checkpoint.get("runtime_config")
    if not isinstance(runtime, dict):
        raise ValueError("Stage-2 checkpoint must contain runtime_config")
    source = str(runtime["source"])
    target = str(runtime["target"])
    classes = [str(value) for value in runtime["classes"]]
    data_root = str(args.data_root or runtime["data_root"])
    seed = int(runtime["seed"])
    fold = int(args.fold)
    num_folds = int(_runtime_value(runtime, "num_folds", 1))
    if fold < 0 or fold >= num_folds:
        raise ValueError(f"fold must lie in [0,{num_folds - 1}]")
    closed_set = bool(_runtime_value(runtime, "closed_set", True))
    combine = bool(_runtime_value(runtime, "combine_spring_and_winter", False))
    time_mode = str(_runtime_value(runtime, "time_coordinate_mode", "canonical_day_of_year"))
    val_ratio = float(_runtime_value(runtime, "val_ratio", 0.1))
    test_ratio = float(_runtime_value(runtime, "test_ratio", 0.2))

    groups = _checkpoint_group_payloads(checkpoint)
    class_to_group = _class_to_group(groups)
    available_classes = tuple(sorted(class_to_group))
    requested_classes = available_classes if args.classes is None else tuple(args.classes)
    unknown = sorted(set(requested_classes) - set(available_classes))
    if unknown:
        raise ValueError(
            "requested classes do not belong to a confirmed Domain Phase group: "
            + ",".join(str(v) for v in unknown)
        )
    if any(class_id < 0 or class_id >= len(classes) for class_id in requested_classes):
        raise ValueError("requested class id is outside checkpoint class range")

    device = torch.device(args.device)
    model = _build_model(runtime, checkpoint, device)

    source_all = _eligible_parcels(
        data_root, source, classes,
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
    )
    target_all = _eligible_parcels(
        data_root, target, classes,
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
    )
    train_indices = _reconstruct_fold_train_indices(
        source_all, target_all,
        source=source,
        target=target,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        fold=fold,
    )

    source_meta = _metadata_train_dataset(
        data_root, source, classes, train_indices[source],
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
    )
    target_meta = _metadata_train_dataset(
        data_root, target, classes, train_indices[target],
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
    )
    source_parcels = _uniform_selected_parcels(
        source_meta, requested_classes, args.samples_per_class
    )
    target_parcels = _uniform_selected_parcels(
        target_meta, requested_classes, args.samples_per_class
    )
    source_loader = _selected_loader(
        data_root, source, classes, source_parcels,
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    target_loader = _selected_loader(
        data_root, target, classes, target_parcels,
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(
        "PHASE_ALIGNMENT_VIS_START|"
        f"checkpoint={checkpoint_path}|source={source}|target={target}|fold={fold}|"
        f"classes={','.join(str(v) for v in requested_classes)}|"
        f"samples_per_class={args.samples_per_class}",
        flush=True,
    )
    source_records = _collect_geometry(
        model, source_loader, device=device, target=False, class_to_group=class_to_group
    )
    target_records = _collect_geometry(
        model, target_loader, device=device, target=True, class_to_group=class_to_group
    )

    shape_mean, shape_components, shape_ratio = _source_pca(
        source_records,
        feature_key="shape_q_before",
        support_key="shape_support_before",
        token_mask=False,
    )
    trend_mean, trend_components, trend_ratio = _source_pca(
        source_records,
        feature_key="trend_q_before",
        support_key="trend_support_before",
        token_mask=False,
    )
    structure_mean, structure_components, structure_ratio = _source_pca(
        source_records,
        feature_key="structure_tokens",
        support_key=None,
        token_mask=True,
    )

    bank = checkpoint.get("source_prototype_bank")
    if not isinstance(bank, dict):
        raise ValueError("checkpoint source_prototype_bank is missing")
    source_shape_proto = bank["shape_srvf"].detach().cpu()
    source_shape_support = bank["shape_support"].detach().cpu()
    source_trend_proto = bank["trend_srvf"].detach().cpu()
    source_trend_support = bank["trend_support"].detach().cpu()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "source_only_projection_basis.npz",
        shape_mean=shape_mean,
        shape_components=shape_components,
        shape_explained_variance_ratio=shape_ratio,
        trend_mean=trend_mean,
        trend_components=trend_components,
        trend_explained_variance_ratio=trend_ratio,
        structure_mean=structure_mean,
        structure_components=structure_components,
        structure_explained_variance_ratio=structure_ratio,
    )

    summary_rows: List[dict] = []
    for class_id in requested_classes:
        source_class = source_records.get(class_id, [])
        target_class = target_records.get(class_id, [])
        if not source_class or not target_class:
            print(
                f"PHASE_ALIGNMENT_CLASS_SKIPPED|class_id={class_id}|"
                f"source={len(source_class)}|target={len(target_class)}",
                flush=True,
            )
            continue
        name = classes[class_id]
        stem = _class_stem(class_id, name)
        group_id = int(class_to_group[class_id]["group_id"])
        shape_stats = _distance_stats(
            target_class,
            q_before_key="shape_q_before",
            support_before_key="shape_support_before",
            q_after_key="shape_q_after",
            support_after_key="shape_support_after",
            prototype_q=source_shape_proto[class_id],
            prototype_support=source_shape_support[class_id],
        )
        shape_direct_stats = _distance_stats(
            target_class,
            q_before_key="shape_q_before",
            support_before_key="shape_support_before",
            q_after_key="shape_q_after_direct",
            support_after_key="shape_support_after_direct",
            prototype_q=source_shape_proto[class_id],
            prototype_support=source_shape_support[class_id],
        )
        trend_stats = _distance_stats(
            target_class,
            q_before_key="trend_q_before",
            support_before_key="trend_support_before",
            q_after_key="trend_q_after",
            support_after_key="trend_support_after",
            prototype_q=source_trend_proto[class_id],
            prototype_support=source_trend_support[class_id],
        )
        trend_direct_stats = _distance_stats(
            target_class,
            q_before_key="trend_q_before",
            support_before_key="trend_support_before",
            q_after_key="trend_q_after_direct",
            support_after_key="trend_support_after_direct",
            prototype_q=source_trend_proto[class_id],
            prototype_support=source_trend_support[class_id],
        )
        gamma_current = target_class[0]["gamma_grid_current"]
        gamma_direct = target_class[0]["gamma_grid_direct"]
        gamma_resample_max_abs_difference = float((gamma_current - gamma_direct).abs().max().item())
        row = {
            "class_id": class_id,
            "class_name": name,
            "group_id": group_id,
            "source_samples": len(source_class),
            "target_samples": len(target_class),
            "shape_before_mean": shape_stats["before_mean"],
            "shape_after_mean": shape_stats["after_mean"],
            "shape_before_median": shape_stats["before_median"],
            "shape_after_median": shape_stats["after_median"],
            "shape_mean_relative_reduction": shape_stats["mean_relative_reduction"],
            "shape_improvement_rate": shape_stats["improvement_rate"],
            "shape_class_center_before": shape_stats["class_center_before"],
            "shape_class_center_after": shape_stats["class_center_after"],
            "shape_direct_after_mean": shape_direct_stats["after_mean"],
            "shape_direct_after_median": shape_direct_stats["after_median"],
            "shape_direct_mean_relative_reduction": shape_direct_stats["mean_relative_reduction"],
            "shape_direct_improvement_rate": shape_direct_stats["improvement_rate"],
            "shape_direct_class_center_after": shape_direct_stats["class_center_after"],
            "trend_before_mean": trend_stats["before_mean"],
            "trend_after_mean": trend_stats["after_mean"],
            "trend_before_median": trend_stats["before_median"],
            "trend_after_median": trend_stats["after_median"],
            "trend_mean_relative_reduction": trend_stats["mean_relative_reduction"],
            "trend_improvement_rate": trend_stats["improvement_rate"],
            "trend_class_center_before": trend_stats["class_center_before"],
            "trend_class_center_after": trend_stats["class_center_after"],
            "trend_direct_after_mean": trend_direct_stats["after_mean"],
            "trend_direct_after_median": trend_direct_stats["after_median"],
            "trend_direct_mean_relative_reduction": trend_direct_stats["mean_relative_reduction"],
            "trend_direct_improvement_rate": trend_direct_stats["improvement_rate"],
            "trend_direct_class_center_after": trend_direct_stats["class_center_after"],
            "gamma_resample_max_abs_difference": gamma_resample_max_abs_difference,
        }
        summary_rows.append(row)
        print(
            "PHASE_ALIGNMENT_CLASS|"
            f"class={class_id}:{name}|group={group_id}|"
            f"shape_current={shape_stats['before_mean']:.6g}->{shape_stats['after_mean']:.6g}|"
            f"shape_direct={shape_stats['before_mean']:.6g}->{shape_direct_stats['after_mean']:.6g}|"
            f"trend_current={trend_stats['before_mean']:.6g}->{trend_stats['after_mean']:.6g}|"
            f"trend_direct={trend_stats['before_mean']:.6g}->{trend_direct_stats['after_mean']:.6g}|"
            f"gamma_resample_diff={gamma_resample_max_abs_difference:.6g}",
            flush=True,
        )
        _plot_q_spaghetti(
            output_dir / "shape_spaghetti" / f"{stem}.png",
            title=f"Shape-SRVF Phase alignment — class {class_id}: {name}",
            source=source_class,
            target=target_class,
            q_prefix="shape_q",
            support_prefix="shape_support",
            source_prototype=source_shape_proto[class_id],
            pca_mean=shape_mean,
            pca_components=shape_components,
            display_samples=args.display_samples,
            robust_lower=args.robust_lower,
            robust_upper=args.robust_upper,
            dpi=args.dpi,
        )
        _plot_q_mean_overlay(
            output_dir / "shape_mean_overlay" / f"{stem}.png",
            title=f"Shape-SRVF class center — class {class_id}: {name}",
            target=target_class,
            q_prefix="shape_q",
            support_prefix="shape_support",
            source_prototype=source_shape_proto[class_id],
            pca_mean=shape_mean,
            pca_components=shape_components,
            dpi=args.dpi,
        )
        _plot_q_spaghetti(
            output_dir / "trend_spaghetti" / f"{stem}.png",
            title=f"Trend-SRVF Phase alignment — class {class_id}: {name}",
            source=source_class,
            target=target_class,
            q_prefix="trend_q",
            support_prefix="trend_support",
            source_prototype=source_trend_proto[class_id],
            pca_mean=trend_mean,
            pca_components=trend_components,
            display_samples=args.display_samples,
            robust_lower=args.robust_lower,
            robust_upper=args.robust_upper,
            dpi=args.dpi,
        )
        _plot_structure_position_spaghetti(
            output_dir / "ltae_position_spaghetti" / f"{stem}.png",
            title=f"Raw Structure tokens and LTAE positions — class {class_id}: {name}",
            source=source_class,
            target=target_class,
            pca_mean=structure_mean,
            pca_components=structure_components,
            display_samples=args.display_samples,
            robust_lower=args.robust_lower,
            robust_upper=args.robust_upper,
            dpi=args.dpi,
        )

    if not summary_rows:
        raise RuntimeError("no class produced a Phase-alignment diagnostic")
    _write_csv(output_dir / "phase_alignment_summary.csv", summary_rows)
    group_summaries = _plot_phase_groups(output_dir, groups, args.dpi)
    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "source": source,
        "target": target,
        "seed": seed,
        "fold": fold,
        "classes": list(classes),
        "visualized_class_ids": list(requested_classes),
        "samples_per_class": args.samples_per_class,
        "display_samples": args.display_samples,
        "target_label_usage": "oracle post-hoc visualization only; never used by training or Domain Phase estimation",
        "phase_direction": {
            "group_gamma": "source_to_target",
            "target_alignment": "target positions use gamma inverse; target q should use warp_q_gamma(q, gamma)",
        },
        "resample_audit": {
            "current": "repository resample_gamma output used by current Stage-2 code",
            "direct": "piecewise-linear evaluation of saved gamma(u) on the 64-point Shape grid; no inversion",
            "purpose": "diagnose whether current gamma resampling preserves the registration direction",
        },
        "pca_fit_scope": "source-only selected true-class rows; basis frozen before projecting target before/after views",
        "shape_pca_explained_variance_ratio": shape_ratio.tolist(),
        "trend_pca_explained_variance_ratio": trend_ratio.tolist(),
        "structure_pca_explained_variance_ratio": structure_ratio.tolist(),
        "phase_groups": group_summaries,
        "class_summary": summary_rows,
    }
    _json_dump(output_dir / "phase_alignment_manifest.json", manifest)
    print(
        "PHASE_ALIGNMENT_VIS_COMPLETE|"
        f"output={output_dir}|classes={len(summary_rows)}",
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize true-class source/target Shape before and after saved Domain Phase."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--classes", type=_parse_int_list, default=None)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--display-samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--robust-lower", type=float, default=1.0)
    parser.add_argument("--robust-upper", type=float, default=99.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples_per_class <= 0 or args.display_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.display_samples > args.samples_per_class:
        raise ValueError("display-samples cannot exceed samples-per-class")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    if not 0.0 <= args.robust_lower < args.robust_upper <= 100.0:
        raise ValueError("robust percentile range is invalid")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    run(args)


if __name__ == "__main__":
    main()
