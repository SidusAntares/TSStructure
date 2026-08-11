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
   prototype;
5. the single-stream Time2Vec/LTAE fused representation distance before and
   after applying gamma^{-1} to target observation positions; and
6. the confirmed Domain Phase gamma itself and its displacement in days.

The preferred input is ``stage2_calibration_state.pt`` written by a
diagnostic-only Stage-2 calibration run, so the model weights are still the
Stage-1 checkpoint and only the no-grad Phase/Shape statistics have been
added. PCA bases are fitted from SOURCE rows only, then frozen for all target views.
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
from methods.structure_da.phase_geometry import phase_distance
from methods.structure_da import phase_visualization_protocol as visproto
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


def _class_to_group(
    groups: Sequence[dict],
    *,
    phase_routes: Optional[Sequence[Optional[int]]] = None,
    num_classes: Optional[int] = None,
) -> Dict[int, dict]:
    """Resolve Phase *usage* routing, not only estimation membership.

    ``member_classes`` are the class centers that estimated a group.  Stage-2
    usage can be broader: M=1 is available to every class and M=2 can expand
    through Stable-Label routing.  Prefer the serialized ``phase_routes`` when
    present so held-out diagnostics test the same C_use semantics as training.
    """
    group_by_id = {int(group["group_id"]): group for group in groups}
    mapping: Dict[int, dict] = {}
    if phase_routes is not None:
        for class_id, route in enumerate(phase_routes):
            if route is None:
                continue
            group = group_by_id.get(int(route))
            if group is not None:
                mapping[int(class_id)] = group
    if mapping:
        return mapping

    for group in groups:
        for class_id in group.get("member_classes", ()):
            class_id = int(class_id)
            if class_id in mapping:
                raise ValueError("a class belongs to more than one confirmed phase group")
            mapping[class_id] = group
    if len(groups) == 1 and num_classes is not None:
        group = groups[0]
        for class_id in range(int(num_classes)):
            mapping.setdefault(class_id, group)
    return mapping


def _runtime_value(runtime: dict, name: str, default=None):
    value = runtime.get(name, default)
    if value is None:
        return default
    return value


def _checkpoint_model_state_dict(checkpoint: dict) -> dict:
    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, dict):
        return state_dict
    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, dict):
        return state_dict
    raise ValueError("checkpoint contains neither model_state_dict nor state_dict")


def _build_model(
    runtime: dict,
    checkpoint: dict,
    device: torch.device,
    *,
    model_checkpoint: Optional[dict] = None,
) -> TSStructureModel:
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
    state_source = checkpoint if model_checkpoint is None else model_checkpoint
    model.load_state_dict(_checkpoint_model_state_dict(state_source), strict=True)
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


def _reconstruct_fold_splits(
    source_parcels: np.ndarray,
    target_parcels: np.ndarray,
    *,
    source: str,
    target: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    fold: int,
) -> Dict[str, Dict[str, set]]:
    """Reproduce ``train.create_train_val_test_folds`` for one fold.

    Keeping all three partitions here is important: Phase is estimated from
    target-train, whereas the formal post-hoc visualization must use held-out
    source-test/target-test parcels.
    """
    rng = random.Random(seed)
    requested = None
    for _fold_index in range(fold + 1):
        splits: Dict[str, Dict[str, set]] = {}
        for name, raw in ((source, source_parcels), (target, target_parcels)):
            indices = [int(value) for value in raw.tolist()]
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val
            rng.shuffle(indices)
            splits[name] = {
                "train": set(indices[:n_train]),
                "val": set(indices[n_train:n_train + n_val]),
                "test": set(indices[n_train + n_val:]),
            }
        requested = splits
    if requested is None:
        raise RuntimeError("failed to reconstruct requested fold")
    return requested


def _metadata_dataset(
    data_root: str,
    domain: str,
    classes: Sequence[str],
    indices: set,
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
        indices=indices,
        with_extra=False,
        closed_set=closed_set,
        combine_spring_and_winter=combine_spring_and_winter,
        time_coordinate_mode=time_coordinate_mode,
    )


# Backward-compatible aliases for older diagnostic imports.  New code should
# use the full split dictionary and ``_metadata_dataset`` explicitly.
def _reconstruct_fold_train_indices(*args, **kwargs) -> Dict[str, set]:
    splits = _reconstruct_fold_splits(*args, **kwargs)
    return {domain: parts["train"] for domain, parts in splits.items()}


def _metadata_train_dataset(*args, **kwargs) -> PixelSetData:
    return _metadata_dataset(*args, **kwargs)


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

        labels = batch["label"].detach().cpu().long()
        parcels = batch["parcel_index"].detach().cpu().long()
        trend_tokens, structure_tokens = model._trend_and_structure(backbone)
        grid = output.geometry.canonical_grid.detach()

        aligned_positions_batch: Optional[Tensor] = None
        aligned_output = None
        if target:
            aligned_positions_batch = backbone.normalized_positions.clone()
            grouped_rows: Dict[int, List[int]] = {}
            group_payloads: Dict[int, dict] = {}
            for row in range(len(labels)):
                class_id = int(labels[row].item())
                group = class_to_group.get(class_id)
                if group is None:
                    continue
                group_id = int(group["group_id"])
                grouped_rows.setdefault(group_id, []).append(row)
                group_payloads[group_id] = group
            for group_id, rows in grouped_rows.items():
                index = torch.tensor(rows, device=device, dtype=torch.long)
                group = group_payloads[group_id]
                aligned_positions_batch[index] = align_target_positions_to_source(
                    backbone.normalized_positions[index],
                    backbone.time_mask[index],
                    group["center_gamma"],
                )
            aligned_output = model.forward_from_backbone(
                backbone,
                positions,
                extra,
                temporal_positions_override=aligned_positions_batch,
                return_geometry=False,
            )

        for row in range(len(labels)):
            class_id = int(labels[row].item())
            record = {
                "parcel_index": int(parcels[row].item()),
                "positions": backbone.normalized_positions[row].detach().cpu(),
                "mask": backbone.time_mask[row].detach().cpu(),
                "pse_tokens": backbone.tokens[row].detach().cpu(),
                "trend_tokens": trend_tokens[row].detach().cpu(),
                "structure_tokens": structure_tokens[row].detach().cpu(),
                "trend_q_before": output.geometry.trend_srvf[row].detach().cpu(),
                "trend_support_before": output.geometry.trend_support[row].detach().cpu(),
                "shape_q_before": output.geometry.structure_srvf[row].detach().cpu(),
                "shape_support_before": output.geometry.structure_support[row].detach().cpu(),
                "trend_valid": bool(output.geometry.trend_valid[row].item()),
                "shape_valid": bool(output.geometry.structure_valid[row].item()),
                "grid": grid.detach().cpu(),
                "fused_repr_before": output.fused_repr[row].detach().cpu(),
                "logits_before": output.logits[row].detach().cpu(),
            }
            if target:
                group = class_to_group.get(class_id)
                if group is None or aligned_output is None or aligned_positions_batch is None:
                    continue
                gamma_grid = _resampled_group_gamma(group, grid)
                gamma_grid_reference = _direct_resample_gamma(group, grid)
                record.update({
                    "group_id": int(group["group_id"]),
                    "positions_after": aligned_positions_batch[row].detach().cpu(),
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
                    "fused_repr_after": aligned_output.fused_repr[row].detach().cpu(),
                    "logits_after": aligned_output.logits[row].detach().cpu(),
                    "gamma_grid": gamma_grid.detach().cpu(),
                    "gamma_grid_reference": gamma_grid_reference.detach().cpu(),
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


def _canonicalize_pse_tokens(
    record: dict,
    *,
    positions_key: str,
    grid_size: int = 128,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Piecewise-linearly place PSE latent tokens on a fixed canonical grid.

    Domain Phase does not alter PSE values; it changes the time coordinates
    attached to those values.  This helper therefore uses the same latent
    tokens with either native positions or the saved gamma-inverse corrected
    positions.  No target information is used to fit parameters.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    mask = record["mask"].bool()
    positions = record[positions_key][mask].detach().cpu().double()
    values = record["pse_tokens"][mask].detach().cpu().double()
    grid = torch.linspace(0.0, 1.0, int(grid_size), dtype=torch.float64)
    output = torch.zeros((grid.numel(), values.shape[-1]), dtype=torch.float64)
    support = torch.zeros(grid.numel(), dtype=torch.bool)
    if positions.numel() < 2:
        return output.float(), support, grid.float()
    if not torch.all(positions[1:] > positions[:-1]).item():
        raise ValueError(f"{positions_key} must be strictly increasing on valid tokens")
    support = (grid >= positions[0]) & (grid <= positions[-1])
    query = grid[support]
    if not query.numel():
        return output.float(), support, grid.float()
    upper = torch.searchsorted(positions, query, right=True).clamp(
        min=1, max=positions.numel() - 1
    )
    lower = upper - 1
    x0 = positions[lower]
    x1 = positions[upper]
    fraction = ((query - x0) / (x1 - x0).clamp_min(1e-12)).unsqueeze(-1)
    interpolated = values[lower] + fraction * (values[upper] - values[lower])
    output[support] = interpolated
    return output.float(), support, grid.float()


def _canonical_pse_center(
    records: Sequence[dict],
    *,
    positions_key: str,
    grid_size: int = 128,
) -> Tuple[Tensor, Tensor, Tensor]:
    trajectories: List[Tensor] = []
    supports: List[Tensor] = []
    grid: Optional[Tensor] = None
    for record in records:
        trajectory, support, current_grid = _canonicalize_pse_tokens(
            record, positions_key=positions_key, grid_size=grid_size
        )
        trajectories.append(trajectory)
        supports.append(support)
        grid = current_grid
    if not trajectories or grid is None:
        raise ValueError("PSE center requires at least one record")
    values = torch.stack(trajectories, dim=0)
    support = torch.stack(supports, dim=0)
    counts = support.sum(dim=0)
    center = (values * support.unsqueeze(-1)).sum(dim=0) / counts.clamp_min(1).unsqueeze(-1)
    return center, counts > 0, grid


def _pse_integrated_distance(
    left: Tensor,
    left_support: Tensor,
    right: Tensor,
    right_support: Tensor,
) -> Tuple[float, float, int]:
    common = left_support.bool() & right_support.bool()
    count = int(common.sum().item())
    if count < 2:
        return float("nan"), float("nan"), count
    squared_feature_error = (left[common].float() - right[common].float()).square().mean(dim=-1)
    weights = torch.ones_like(squared_feature_error)
    if weights.numel() > 1:
        weights[[0, -1]] *= 0.5
    integrated_mse = float((squared_feature_error * weights).sum().item() / weights.sum().item())
    return integrated_mse, float(np.sqrt(max(integrated_mse, 0.0))), count


def _pse_class_metrics(
    source_records: Sequence[dict],
    target_records: Sequence[dict],
    *,
    grid_size: int = 128,
) -> Tuple[dict, List[dict], dict]:
    source_center, source_support, grid = _canonical_pse_center(
        source_records, positions_key="positions", grid_size=grid_size
    )
    target_before_center, target_before_support, _ = _canonical_pse_center(
        target_records, positions_key="positions", grid_size=grid_size
    )
    target_after_center, target_after_support, _ = _canonical_pse_center(
        target_records, positions_key="positions_after", grid_size=grid_size
    )
    before_mse, before_l2, before_common = _pse_integrated_distance(
        source_center, source_support, target_before_center, target_before_support
    )
    after_mse, after_l2, after_common = _pse_integrated_distance(
        source_center, source_support, target_after_center, target_after_support
    )

    sample_rows: List[dict] = []
    improved = 0
    valid_samples = 0
    for record in target_records:
        before, before_support, _ = _canonicalize_pse_tokens(
            record, positions_key="positions", grid_size=grid_size
        )
        after, after_support, _ = _canonicalize_pse_tokens(
            record, positions_key="positions_after", grid_size=grid_size
        )
        sample_before_mse, sample_before_l2, sample_before_common = _pse_integrated_distance(
            source_center, source_support, before, before_support
        )
        sample_after_mse, sample_after_l2, sample_after_common = _pse_integrated_distance(
            source_center, source_support, after, after_support
        )
        is_valid = np.isfinite(sample_before_l2) and np.isfinite(sample_after_l2)
        if is_valid:
            valid_samples += 1
            improved += int(sample_after_l2 < sample_before_l2)
        sample_rows.append(
            {
                "parcel_index": int(record["parcel_index"]),
                "pse_mse_before": sample_before_mse,
                "pse_mse_after": sample_after_mse,
                "pse_l2_before": sample_before_l2,
                "pse_l2_after": sample_after_l2,
                "pse_common_grid_before": sample_before_common,
                "pse_common_grid_after": sample_after_common,
                "pse_distance_improved": bool(is_valid and sample_after_l2 < sample_before_l2),
            }
        )
    relative = (before_l2 - after_l2) / max(before_l2, 1e-12) if np.isfinite(before_l2) else float("nan")
    summary = {
        "pse_class_mean_mse_before": before_mse,
        "pse_class_mean_mse_after": after_mse,
        "pse_class_mean_l2_before": before_l2,
        "pse_class_mean_l2_after": after_l2,
        "pse_class_mean_relative_reduction": relative,
        "pse_sample_improvement_rate": improved / valid_samples if valid_samples else float("nan"),
        "pse_valid_samples": valid_samples,
        "pse_common_grid_before": before_common,
        "pse_common_grid_after": after_common,
    }
    curves = {
        "grid": grid,
        "source_center": source_center,
        "source_support": source_support,
        "target_before_center": target_before_center,
        "target_before_support": target_before_support,
        "target_after_center": target_after_center,
        "target_after_support": target_after_support,
    }
    return summary, sample_rows, curves


def _plot_pse_class_mean_alignment(
    path: Path,
    *,
    title: str,
    curves: dict,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
    dpi: int,
) -> None:
    grid = curves["grid"].numpy() * 365.0
    projected = {
        name: _project(curves[name], pca_mean, pca_components)
        for name in ("source_center", "target_before_center", "target_after_center")
    }
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True, constrained_layout=True)
    labels = (
        ("source_center", "source_support", "Source test class mean"),
        ("target_before_center", "target_before_support", "Target test native"),
        ("target_after_center", "target_after_support", "Target test after Domain Phase"),
    )
    for component, axis in enumerate(axes):
        for value_key, support_key, label in labels:
            support = curves[support_key].numpy().astype(bool)
            axis.plot(grid[support], projected[value_key][support, component], label=label, linewidth=1.8)
        axis.set_ylabel(f"source-fit PCA PC{component + 1}")
        axis.grid(alpha=0.18)
        axis.legend(loc="best")
    axes[-1].set_xlabel("canonical day")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


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


def _representation_center(records: Sequence[dict], key: str) -> Tensor:
    if not records:
        raise ValueError("representation center requires at least one record")
    return torch.stack([record[key].float() for record in records], dim=0).mean(dim=0)


def _representation_distances(
    records: Sequence[dict],
    *,
    before_key: str,
    after_key: str,
    prototype: Tensor,
) -> Tuple[Tensor, Tensor, dict]:
    if not records:
        raise ValueError("representation distances require at least one record")
    prototype = prototype.detach().cpu().float()
    before_repr = torch.stack([record[before_key].float() for record in records], dim=0)
    after_repr = torch.stack([record[after_key].float() for record in records], dim=0)
    before = torch.linalg.vector_norm(before_repr - prototype.unsqueeze(0), dim=-1)
    after = torch.linalg.vector_norm(after_repr - prototype.unsqueeze(0), dim=-1)
    before_mean = float(before.mean().item())
    after_mean = float(after.mean().item())
    stats = {
        "before_mean": before_mean,
        "before_median": float(before.median().item()),
        "after_mean": after_mean,
        "after_median": float(after.median().item()),
        "mean_relative_reduction": (before_mean - after_mean) / max(before_mean, 1e-12),
        "improvement_rate": float((after < before).float().mean().item()),
    }
    return before, after, stats


def _classification_stats(records: Sequence[dict], class_id: int) -> dict:
    logits_before = torch.stack([record["logits_before"].float() for record in records])
    logits_after = torch.stack([record["logits_after"].float() for record in records])
    probs_before = torch.softmax(logits_before, dim=-1)
    probs_after = torch.softmax(logits_after, dim=-1)
    pred_before = logits_before.argmax(dim=-1)
    pred_after = logits_after.argmax(dim=-1)
    return {
        "true_probability_before_mean": float(probs_before[:, class_id].mean().item()),
        "true_probability_after_mean": float(probs_after[:, class_id].mean().item()),
        "accuracy_before": float((pred_before == class_id).float().mean().item()),
        "accuracy_after": float((pred_after == class_id).float().mean().item()),
    }


def _position_shift_stats(records: Sequence[dict]) -> dict:
    signed: List[Tensor] = []
    absolute: List[Tensor] = []
    for record in records:
        mask = record["mask"].bool()
        delta = (record["positions_after"][mask] - record["positions"][mask]).float() * 365.0
        if delta.numel():
            signed.append(delta)
            absolute.append(delta.abs())
    if not signed:
        return {
            "signed_mean_days": None,
            "signed_median_days": None,
            "absolute_mean_days": None,
            "absolute_median_days": None,
            "max_absolute_days": None,
        }
    signed_values = torch.cat(signed)
    absolute_values = torch.cat(absolute)
    return {
        "signed_mean_days": float(signed_values.mean().item()),
        "signed_median_days": float(signed_values.median().item()),
        "absolute_mean_days": float(absolute_values.mean().item()),
        "absolute_median_days": float(absolute_values.median().item()),
        "max_absolute_days": float(absolute_values.max().item()),
    }


def _plot_representation_distance(
    output_path: Path,
    *,
    title: str,
    metrics: Sequence[Tuple[str, Tensor, Tensor]],
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), constrained_layout=True)
    for axis, (label, before, after) in zip(axes.flat, metrics):
        x = before.detach().cpu().numpy()
        y = after.detach().cpu().numpy()
        low = float(min(np.min(x), np.min(y)))
        high = float(max(np.max(x), np.max(y)))
        if low == high:
            pad = max(abs(low) * 0.05, 1e-6)
            low -= pad
            high += pad
        axis.scatter(x, y, s=20, alpha=0.7)
        axis.plot([low, high], [low, high], linestyle="--", linewidth=1.2, label="no change")
        improvement = float(np.mean(y < x))
        axis.set_title(f"{label} — improved {improvement * 100:.1f}%")
        axis.set_xlabel("distance before Phase")
        axis.set_ylabel("distance after Phase")
        axis.grid(alpha=0.18)
        axis.legend(loc="best")
    figure.suptitle(title + " — points below diagonal are closer to source")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


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

    for mode in ("full", "robust"):
        figure, axes = plt.subplots(
            2, 3, figsize=(13.2, 7.0), sharex=True, sharey="row", constrained_layout=True
        )
        for component in range(2):
            all_values = [value[:, component] for value in source_pc + before_pc + after_pc]
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
                ("Target after confirmed Phase", target_display, after_pc, f"{support_prefix}_after"),
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
                if component == 0 and column == 2:
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
    source_pc = _project(source_prototype.cpu(), pca_mean, pca_components)
    before_pc = _project(before_center, pca_mean, pca_components)
    after_pc = _project(after_center, pca_mean, pca_components)
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True, constrained_layout=True)
    for component, axis in enumerate(axes):
        axis.plot(grid, source_pc[:, component], linewidth=2.0, label="source prototype")
        axis.plot(grid, before_pc[:, component], linewidth=2.0, label="target before")
        axis.plot(grid, after_pc[:, component], linewidth=2.0, label="target after Phase")
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
            "within_dispersion": group.get("within_dispersion"),
            "diameter": group.get("diameter"),
            "core_radius": group.get("core_radius"),
            "center_drift": group.get("center_drift"),
            "sample_evidence_count": group.get("sample_evidence_count"),
            "class_count": group.get("class_count"),
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


def _phase_consistency_outputs(
    output_dir: Path,
    *,
    checkpoint: dict,
    groups: Sequence[dict],
    classes: Sequence[str],
    dpi: int,
) -> dict:
    """Write experiment-03 class/group Phase consistency diagnostics.

    Older Stage-2 checkpoints saved group centers but not class centers.  Those
    checkpoints remain usable for experiments 01/02/04; experiment 03 records
    the missing serialization explicitly instead of reconstructing it from new
    DP calls or pretending the data exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    group_summary = _plot_phase_groups(output_dir, groups, dpi)
    phase = checkpoint.get("phase_state")
    class_centers = phase.get("class_centers") if isinstance(phase, dict) else None
    available = isinstance(class_centers, (list, tuple)) and len(class_centers) > 0
    availability = {
        "class_center_gamma_available": bool(available),
        "legacy_checkpoint_degraded_mode": not bool(available),
        "reason": None if available else (
            "checkpoint predates full Phase-state serialization; group center gamma is available "
            "but class center gammas cannot be recovered without rerunning registration"
        ),
    }
    _json_dump(output_dir / "availability.json", availability)
    _json_dump(output_dir / "group_summary.json", {"groups": group_summary})

    if not available:
        _json_dump(output_dir / "manifest.json", {
            "experiment": "03_domain_phase_consistency",
            **availability,
            "groups": group_summary,
        })
        (output_dir / "README_中文说明.md").write_text(
            "# 03 Domain-level Phase consistency\n\n"
            "本实验用于判断已确认的 Phase 是否由多个类别共同支持，而不是类别特异 Phase。\n\n"
            "当前 checkpoint 属于旧序列化格式：保存了 confirmed group center gamma，"
            "但没有保存 class-center gamma。因此本目录仍给出 `phase_groups/` 与 "
            "`group_summary.json`，但无法可靠计算 class-center pairwise distance。"
            "不会重新运行 exact-DP，也不会从日志或 group center 伪造 class center。\n\n"
            "后续由本 patch 产生的新 checkpoint 会完整保存 class centers；届时同一脚本"
            "会自动输出 `class_center_gamma.png`、`displacement_days_by_class.png`、"
            "`class_phase_summary.csv` 和 `pairwise_center_distance.csv`。\n",
            encoding="utf-8",
        )
        return {"availability": availability, "groups": group_summary, "class_centers": []}

    group_by_id = {int(group["group_id"]): group for group in groups}
    class_rows: List[dict] = []
    pairwise_rows: List[dict] = []
    prepared: List[Tuple[int, Tensor, dict]] = []
    for payload in class_centers:
        gamma = payload.get("center_gamma")
        if not isinstance(gamma, Tensor):
            continue
        class_id = int(payload["class_id"])
        gamma = gamma.detach().cpu().double()
        nearest_group_id = None
        nearest_distance = None
        for group_id, group in group_by_id.items():
            distance = float(phase_distance(gamma, group["center_gamma"].detach().cpu().double()).item())
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_group_id = group_id
        grid = torch.linspace(0.0, 1.0, gamma.numel(), dtype=torch.float64)
        displacement = (gamma - grid) * 365.0
        interior = displacement[(grid >= 0.10) & (grid <= 0.90)]
        class_rows.append({
            "class_id": class_id,
            "class_name": classes[class_id] if 0 <= class_id < len(classes) else str(class_id),
            "valid": bool(payload.get("valid", False)),
            "reject_reason": payload.get("reject_reason"),
            "candidate_count": payload.get("candidate_count"),
            "effective_evidence_count": payload.get("effective_evidence_count"),
            "dispersion": payload.get("dispersion"),
            "diameter": payload.get("diameter"),
            "median_distance": payload.get("median_distance"),
            "center_drift": payload.get("center_drift"),
            "nearest_group_id": nearest_group_id,
            "distance_to_nearest_group": nearest_distance,
            "source_to_target_shift_days_interior_median": float(interior.median().item()) if interior.numel() else float("nan"),
            "source_to_target_shift_days_interior_p10": float(torch.quantile(interior, 0.10).item()) if interior.numel() else float("nan"),
            "source_to_target_shift_days_interior_p90": float(torch.quantile(interior, 0.90).item()) if interior.numel() else float("nan"),
        })
        prepared.append((class_id, gamma, payload))

    for left_id, left_gamma, _ in prepared:
        for right_id, right_gamma, _ in prepared:
            pairwise_rows.append({
                "class_id_i": left_id,
                "class_id_j": right_id,
                "phase_distance": float(phase_distance(left_gamma, right_gamma).item()),
            })
    if class_rows:
        _write_csv(output_dir / "class_phase_summary.csv", class_rows)
    if pairwise_rows:
        _write_csv(output_dir / "pairwise_center_distance.csv", pairwise_rows)

    fig, ax = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    for class_id, gamma, payload in prepared:
        grid = np.linspace(0.0, 365.0, gamma.numel())
        ax.plot(grid, gamma.numpy() * 365.0, linewidth=1.2, alpha=0.75, label=f"{class_id}:{classes[class_id]}")
    for group in groups:
        gamma = group["center_gamma"].detach().cpu().double().numpy()
        grid = np.linspace(0.0, 365.0, len(gamma))
        ax.plot(grid, gamma * 365.0, linewidth=3.0, linestyle="--", label=f"group {int(group['group_id'])}")
    ax.plot([0.0, 365.0], [0.0, 365.0], linestyle=":", linewidth=1.2, label="identity")
    ax.set_xlabel("source canonical day")
    ax.set_ylabel("target canonical day")
    ax.set_title("Class Phase centers and confirmed Domain Phase group centers")
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.savefig(output_dir / "class_center_gamma.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    for class_id, gamma, _ in prepared:
        grid = torch.linspace(0.0, 1.0, gamma.numel(), dtype=torch.float64)
        ax.plot(grid.numpy() * 365.0, ((gamma - grid) * 365.0).numpy(), linewidth=1.2, label=f"{class_id}:{classes[class_id]}")
    ax.axhline(0.0, linestyle=":", linewidth=1.0)
    ax.set_xlabel("canonical day")
    ax.set_ylabel("gamma(t)-t (days)")
    ax.set_title("Source-to-target Phase displacement by class")
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.savefig(output_dir / "displacement_days_by_class.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    _json_dump(output_dir / "manifest.json", {
        "experiment": "03_domain_phase_consistency",
        **availability,
        "groups": group_summary,
        "num_class_centers": len(class_rows),
    })
    (output_dir / "README_中文说明.md").write_text(
        "# 03 Domain-level Phase consistency\n\n"
        "目标：区分共享的 domain-level Phase 与 class-intrinsic Phase。target true label 不参与本实验；"
        "这里使用的是 calibration 时保存的 class-level Phase centers。\n\n"
        "重点查看 `class_center_gamma.png`、`displacement_days_by_class.png`、"
        "`pairwise_center_distance.csv` 和 `class_phase_summary.csv`。若可靠类别中心紧密围绕"
        "同一个 group center，支持共享 Domain Phase；若类别中心明显分裂而 M=1 仍被确认，"
        "应重新审计 group-level criteria。\n",
        encoding="utf-8",
    )
    return {"availability": availability, "groups": group_summary, "class_centers": class_rows}


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


# Bind the script to the dependency-light, unit-tested protocol helpers.  The
# local definitions above are kept for backward source compatibility with old
# notebooks that may import them by line/function, but runtime calls use this
# shared implementation.
_checkpoint_model_state_dict = visproto.checkpoint_model_state_dict
_class_to_group = visproto.class_to_group
_reconstruct_fold_splits = visproto.reconstruct_fold_splits
_canonicalize_pse_tokens = visproto.canonicalize_pse_tokens
_canonical_pse_center = visproto.canonical_pse_center
_pse_integrated_distance = visproto.pse_integrated_distance
_pse_class_metrics = visproto.pse_class_metrics


def _write_chinese_readme(
    path: Path,
    *,
    checkpoint_path: Path,
    source: str,
    target: str,
    seed: int,
    fold: int,
    summary_rows: Sequence[dict],
    group_summaries: Sequence[dict],
) -> None:
    lines: List[str] = []
    lines.append("# Domain Phase 对齐可视化结果说明")
    lines.append("")
    lines.append("本目录使用 held-out source-test / target-test 检查已确认的 Domain Phase 是否真正改善 source-target 同类对齐。目标域真实标签只用于离线 oracle 可视化配对，不参与训练、伪标签、Domain Phase、阈值选择或模型选择。")
    lines.append("")
    lines.append("## 本次运行")
    lines.append("")
    lines.append(f"- source: `{source}`")
    lines.append(f"- target: `{target}`")
    lines.append(f"- seed: `{seed}`")
    lines.append(f"- fold: `{fold}`")
    lines.append(f"- checkpoint: `{checkpoint_path}`")
    lines.append("")
    lines.append("## Phase 方向")
    lines.append("")
    lines.append("保存的 `gamma` 定义为 source→target，即 `gamma(u_source)=u_target`。因此：")
    lines.append("")
    lines.append("- target 送入 Time2Vec/LTAE 的时间位置使用 `gamma^{-1}(t_target)`；")
    lines.append("- target SRVF/Shape 回到 source 公共时间坐标时使用 `(q_target, gamma)`；")
    lines.append("- `gamma_resample_max_abs_difference` 是正式 `resample_gamma` 与独立线性插值参考实现的差值，修复后应接近 0。")
    lines.append("")
    lines.append("## 文件夹说明")
    lines.append("")
    lines.append("| 路径 | 含义 | 主要看什么 |")
    lines.append("|---|---|---|")
    lines.append("| `phase_groups/` | 已确认 Domain Phase 的 `gamma` 与位移（天） | `gamma(t)-t` 是否像合理的域级时间偏移 |")
    lines.append("| `shape_spaghetti/` | source、target-before、target-after 的 Shape-SRVF PCA spaghetti | Phase 后 target Shape 是否整体向 source 靠近 |")
    lines.append("| `shape_mean_overlay/` | source prototype 与 target 类中心 before/after 叠加 | 类中心层面是否改善 |")
    lines.append("| `trend_spaghetti/` | Trend-SRVF 的同类 before/after | 作为 Phase 估计来源，Trend 本身是否被正确对齐 |")
    lines.append("| `ltae_position_spaghetti/` | Structure token 值不变，只显示 LTAE 使用的时间横坐标 | `gamma^{-1}(t_target)` 是否真正移动了观测时间位置 |")
    lines.append("| `ltae_representation_distance/` | LTAE 表示 before/after 到 source 同类中心的距离散点 | 点在对角线下方表示 Phase 后表示更接近 source |")
    lines.append("")
    lines.append("## 数据文件说明")
    lines.append("")
    lines.append("| 文件 | 含义 |")
    lines.append("|---|---|")
    lines.append("| `summary.csv` | 每个类别的 Shape、Trend、LTAE 表示 before/after 汇总指标 |")
    lines.append("| `sample_metrics.csv` | 每个 target-test 样本的 LTAE 距离、位置移动、分类概率 before/after |")
    lines.append("| `manifest.json` | 本次运行参数、held-out split、Phase 方向、PCA 信息和完整汇总 |")
    lines.append("| `source_only_projection_basis.npz` | 只用 source 拟合并冻结的 PCA basis，保证 target before/after 共用同一投影 |")
    lines.append("")
    if group_summaries:
        lines.append("## Domain Phase 摘要")
        lines.append("")
        lines.append("| group | classes | 内部中位 source→target shift / 天 | P10 | P90 |")
        lines.append("|---:|---|---:|---:|---:|")
        for item in group_summaries:
            classes = ",".join(str(v) for v in item["member_classes"])
            lines.append(
                f"| {item['group_id']} | {classes} | "
                f"{item['source_to_target_shift_days_interior_median']:.2f} | "
                f"{item['source_to_target_shift_days_interior_p10']:.2f} | "
                f"{item['source_to_target_shift_days_interior_p90']:.2f} |"
            )
        lines.append("")
    lines.append("## 每类关键结果")
    lines.append("")
    lines.append("`reduction > 0` 表示距离下降；`improve rate` 表示样本中 after 距离小于 before 的比例。")
    lines.append("")
    lines.append("| class | Shape mean | Trend mean | LTAE fused mean | fused improve rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['class_id']} {row['class_name']} | "
            f"{row['shape_before_mean']:.3f}→{row['shape_after_mean']:.3f} | "
            f"{row['trend_before_mean']:.3f}→{row['trend_after_mean']:.3f} | "
            f"{row['ltae_fused_before_mean']:.3f}→{row['ltae_fused_after_mean']:.3f} | "
            f"{100.0 * row['ltae_fused_improvement_rate']:.1f}% |"
        )
    lines.append("")
    lines.append("## 如何判断")
    lines.append("")
    lines.append("1. `gamma_resample_max_abs_difference` 应接近 0；否则说明 Shape-grid 上的 gamma 重采样仍有方向/实现不一致。")
    lines.append("2. `Shape/Trend mean before→after` 若下降且 improvement rate 较高，说明函数几何对齐有效。")
    lines.append("3. 最关键的是 `LTAE fused mean before→after`：若下降且多数样本位于 `ltae_representation_distance/` 对角线下方，说明时间位置校正确实改善了分类器输入表示的 source-target 同类距离。")
    lines.append("4. 若 Shape/Trend 改善但 LTAE fused 不改善，说明 Phase 几何正确，但 Time2Vec/LTAE 没有把位置校正转化为更好的表示。")
    lines.append("5. 若三者都不改善，应继续检查 Domain Phase 的估计本身，而不是优先调训练超参数。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    class_to_group = _class_to_group(
        groups,
        phase_routes=checkpoint.get("phase_routes"),
        num_classes=len(classes),
    )
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
    model_checkpoint_path = None if args.model_checkpoint is None else args.model_checkpoint.resolve()
    model_checkpoint = (
        None
        if model_checkpoint_path is None
        else torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    )
    model = _build_model(
        runtime, checkpoint, device, model_checkpoint=model_checkpoint
    )

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
    splits = _reconstruct_fold_splits(
        source_all, target_all,
        source=source,
        target=target,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        fold=fold,
    )

    source_meta = _metadata_dataset(
        data_root, source, classes, splits[source]["test"],
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
    )
    target_meta = _metadata_dataset(
        data_root, target, classes, splits[target]["test"],
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
    pse_mean, pse_components, pse_ratio = _source_pca(
        source_records,
        feature_key="pse_tokens",
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
    source_fused_proto = bank["fused"].detach().cpu()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir = output_dir / "01_domain_phase_geometry_alignment"
    pse_dir = output_dir / "02_pse_latent_phase_alignment"
    phase_consistency_dir = output_dir / "03_domain_phase_consistency"
    for directory in (geometry_dir, pse_dir, phase_consistency_dir):
        directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        geometry_dir / "source_only_projection_basis.npz",
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
    np.savez_compressed(
        pse_dir / "pse_pca_source_fit.npz",
        pse_mean=pse_mean,
        pse_components=pse_components,
        pse_explained_variance_ratio=pse_ratio,
    )

    summary_rows: List[dict] = []
    sample_rows: List[dict] = []
    pse_summary_rows: List[dict] = []
    pse_sample_rows: List[dict] = []
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
        trend_stats = _distance_stats(
            target_class,
            q_before_key="trend_q_before",
            support_before_key="trend_support_before",
            q_after_key="trend_q_after",
            support_after_key="trend_support_after",
            prototype_q=source_trend_proto[class_id],
            prototype_support=source_trend_support[class_id],
        )
        pse_stats, class_pse_samples, pse_curves = _pse_class_metrics(
            source_class, target_class, grid_size=args.pse_grid_size
        )
        pse_summary_rows.append({
            "class_id": int(class_id),
            "class_name": name,
            "phase_group_id": group_id,
            "source_test_samples": len(source_class),
            "target_test_samples": len(target_class),
            **pse_stats,
        })
        for item in class_pse_samples:
            pse_sample_rows.append({
                "class_id": int(class_id),
                "class_name": name,
                "phase_group_id": group_id,
                **item,
            })
        _plot_pse_class_mean_alignment(
            pse_dir / f"class_{class_id:02d}_{name}" / "pse_pc_mean_before_after.png",
            title=f"PSE latent Phase alignment — class {class_id}: {name}",
            curves=pse_curves,
            pca_mean=pse_mean,
            pca_components=pse_components,
            dpi=args.dpi,
        )

        source_fused_sampled = _representation_center(source_class, "fused_repr_before")
        fused_before, fused_after, fused_stats = _representation_distances(
            target_class,
            before_key="fused_repr_before",
            after_key="fused_repr_after",
            prototype=source_fused_proto[class_id],
        )
        fused_sampled_before, fused_sampled_after, fused_sampled_stats = _representation_distances(
            target_class,
            before_key="fused_repr_before",
            after_key="fused_repr_after",
            prototype=source_fused_sampled,
        )
        cls_stats = _classification_stats(target_class, class_id)
        position_stats = _position_shift_stats(target_class)

        gamma_current = target_class[0]["gamma_grid"]
        gamma_reference = target_class[0]["gamma_grid_reference"]
        gamma_resample_max_abs_difference = float(
            (gamma_current - gamma_reference).abs().max().item()
        )

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
            "trend_before_mean": trend_stats["before_mean"],
            "trend_after_mean": trend_stats["after_mean"],
            "trend_before_median": trend_stats["before_median"],
            "trend_after_median": trend_stats["after_median"],
            "trend_mean_relative_reduction": trend_stats["mean_relative_reduction"],
            "trend_improvement_rate": trend_stats["improvement_rate"],
            "trend_class_center_before": trend_stats["class_center_before"],
            "trend_class_center_after": trend_stats["class_center_after"],
            "ltae_fused_before_mean": fused_stats["before_mean"],
            "ltae_fused_after_mean": fused_stats["after_mean"],
            "ltae_fused_mean_relative_reduction": fused_stats["mean_relative_reduction"],
            "ltae_fused_improvement_rate": fused_stats["improvement_rate"],
            "ltae_fused_sampled_before_mean": fused_sampled_stats["before_mean"],
            "ltae_fused_sampled_after_mean": fused_sampled_stats["after_mean"],
            "ltae_fused_sampled_improvement_rate": fused_sampled_stats["improvement_rate"],
            "true_probability_before_mean": cls_stats["true_probability_before_mean"],
            "true_probability_after_mean": cls_stats["true_probability_after_mean"],
            "oracle_accuracy_before": cls_stats["accuracy_before"],
            "oracle_accuracy_after": cls_stats["accuracy_after"],
            "position_shift_signed_mean_days": position_stats["signed_mean_days"],
            "position_shift_absolute_mean_days": position_stats["absolute_mean_days"],
            "position_shift_max_absolute_days": position_stats["max_absolute_days"],
            "gamma_resample_max_abs_difference": gamma_resample_max_abs_difference,
        }
        summary_rows.append(row)

        for index, record in enumerate(target_class):
            mask = record["mask"].bool()
            position_delta = (
                record["positions_after"][mask] - record["positions"][mask]
            ).float() * 365.0
            logits_before = record["logits_before"].float()
            logits_after = record["logits_after"].float()
            probs_before = torch.softmax(logits_before, dim=-1)
            probs_after = torch.softmax(logits_after, dim=-1)
            sample_rows.append(
                {
                    "class_id": class_id,
                    "class_name": name,
                    "parcel_index": int(record["parcel_index"]),
                    "group_id": group_id,
                    "position_shift_signed_mean_days": float(position_delta.mean().item()) if position_delta.numel() else None,
                    "position_shift_absolute_mean_days": float(position_delta.abs().mean().item()) if position_delta.numel() else None,
                    "position_shift_max_absolute_days": float(position_delta.abs().max().item()) if position_delta.numel() else None,
                    "fused_distance_before": float(fused_before[index].item()),
                    "fused_distance_after": float(fused_after[index].item()),
                    "fused_distance_improved": bool(fused_after[index] < fused_before[index]),
                    "fused_sampled_distance_before": float(fused_sampled_before[index].item()),
                    "fused_sampled_distance_after": float(fused_sampled_after[index].item()),
                    "true_probability_before": float(probs_before[class_id].item()),
                    "true_probability_after": float(probs_after[class_id].item()),
                    "prediction_before": int(logits_before.argmax().item()),
                    "prediction_after": int(logits_after.argmax().item()),
                    "correct_before": bool(logits_before.argmax().item() == class_id),
                    "correct_after": bool(logits_after.argmax().item() == class_id),
                }
            )

        print(
            "PHASE_ALIGNMENT_CLASS|"
            f"class={class_id}:{name}|group={group_id}|"
            f"shape={shape_stats['before_mean']:.6g}->{shape_stats['after_mean']:.6g}|"
            f"trend={trend_stats['before_mean']:.6g}->{trend_stats['after_mean']:.6g}|"
            f"ltae_fused={fused_stats['before_mean']:.6g}->{fused_stats['after_mean']:.6g}|"
            f"ltae_improve_rate={fused_stats['improvement_rate']:.4f}|"
            f"gamma_resample_diff={gamma_resample_max_abs_difference:.6g}",
            flush=True,
        )

        _plot_q_spaghetti(
            geometry_dir / "shape_spaghetti" / f"{stem}.png",
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
            geometry_dir / "shape_mean_overlay" / f"{stem}.png",
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
            geometry_dir / "trend_spaghetti" / f"{stem}.png",
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
            geometry_dir / "ltae_position_spaghetti" / f"{stem}.png",
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
        _plot_representation_distance(
            geometry_dir / "ltae_representation_distance" / f"{stem}.png",
            title=f"LTAE representation Phase alignment — class {class_id}: {name}",
            metrics=(
                ("Fused / formal source prototype", fused_before, fused_after),
                ("Fused / sampled source center", fused_sampled_before, fused_sampled_after),
            ),
            dpi=args.dpi,
        )

    if not summary_rows:
        raise RuntimeError("no class produced a Phase-alignment diagnostic")
    _write_csv(geometry_dir / "summary.csv", summary_rows)
    _write_csv(geometry_dir / "sample_metrics.csv", sample_rows)
    if not pse_summary_rows:
        raise RuntimeError("no class produced a PSE latent Phase diagnostic")
    geometry_macro = {
        "num_classes": len(summary_rows),
        "shape_before_class_equal_mean": float(np.mean([row["shape_before_mean"] for row in summary_rows])),
        "shape_after_class_equal_mean": float(np.mean([row["shape_after_mean"] for row in summary_rows])),
        "trend_before_class_equal_mean": float(np.mean([row["trend_before_mean"] for row in summary_rows])),
        "trend_after_class_equal_mean": float(np.mean([row["trend_after_mean"] for row in summary_rows])),
        "ltae_fused_before_class_equal_mean": float(np.mean([row["ltae_fused_before_mean"] for row in summary_rows])),
        "ltae_fused_after_class_equal_mean": float(np.mean([row["ltae_fused_after_mean"] for row in summary_rows])),
    }
    _json_dump(geometry_dir / "summary.json", geometry_macro)

    _write_csv(pse_dir / "pse_distance_summary.csv", pse_summary_rows)
    _write_csv(pse_dir / "sample_level.csv", pse_sample_rows)

    def _macro_mean(key: str) -> float:
        values = [float(row[key]) for row in pse_summary_rows if np.isfinite(float(row[key]))]
        return float(np.mean(values)) if values else float("nan")

    pse_macro = {
        "num_classes": len(pse_summary_rows),
        "class_equal_pse_l2_before": _macro_mean("pse_class_mean_l2_before"),
        "class_equal_pse_l2_after": _macro_mean("pse_class_mean_l2_after"),
        "class_equal_relative_reduction": _macro_mean("pse_class_mean_relative_reduction"),
        "class_equal_sample_improvement_rate": _macro_mean("pse_sample_improvement_rate"),
    }
    _json_dump(pse_dir / "summary.json", pse_macro)

    phase_consistency = _phase_consistency_outputs(
        phase_consistency_dir,
        checkpoint=checkpoint,
        groups=groups,
        classes=classes,
        dpi=args.dpi,
    )
    group_summaries = phase_consistency["groups"]

    common_manifest = {
        "phase_checkpoint": str(checkpoint_path),
        "model_checkpoint": str(model_checkpoint_path or checkpoint_path),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "successful_optimizer_steps": checkpoint.get("successful_optimizer_steps"),
        "source": source,
        "target": target,
        "seed": seed,
        "fold": fold,
        "data_partition": "held-out source test + held-out target test",
        "classes": list(classes),
        "visualized_class_ids": list(requested_classes),
        "samples_per_class": args.samples_per_class,
        "target_label_usage": "oracle-only diagnostic grouping; never used by training, Phase estimation, threshold selection or model selection",
        "phase_direction": {
            "group_gamma": "source_to_target",
            "target_ltae_alignment": "target positions use gamma inverse",
            "target_srvf_alignment": "target q uses warp_q_gamma(q, gamma)",
        },
    }
    geometry_manifest = {
        **common_manifest,
        "experiment": "01_domain_phase_geometry_alignment",
        "resample_self_check": {
            "reference": "independent piecewise-linear evaluation of saved gamma(u) on the Shape grid",
            "expected": "gamma_resample_max_abs_difference approximately zero",
        },
        "pca_fit_scope": "source-test selected rows only; basis frozen before target projection",
        "shape_pca_explained_variance_ratio": shape_ratio.tolist(),
        "trend_pca_explained_variance_ratio": trend_ratio.tolist(),
        "structure_pca_explained_variance_ratio": structure_ratio.tolist(),
        "class_summary": summary_rows,
    }
    _json_dump(geometry_dir / "manifest.json", geometry_manifest)
    _write_chinese_readme(
        geometry_dir / "README_中文说明.md",
        checkpoint_path=checkpoint_path,
        source=source,
        target=target,
        seed=seed,
        fold=fold,
        summary_rows=summary_rows,
        group_summaries=group_summaries,
    )

    pse_manifest = {
        **common_manifest,
        "experiment": "02_pse_latent_phase_alignment",
        "pse_grid_size": int(args.pse_grid_size),
        "metric": "piecewise-linear canonical interpolation of unchanged PSE latent values; class-equal support-aware integrated feature-MSE/L2 before vs gamma^-1-corrected target positions",
        "pca_fit_scope": "PCA fitted only from held-out source PSE tokens; target before/after only transformed",
        "pse_pca_explained_variance_ratio": pse_ratio.tolist(),
        "macro_summary": pse_macro,
        "class_summary": pse_summary_rows,
    }
    _json_dump(pse_dir / "manifest.json", pse_manifest)
    (pse_dir / "README_中文说明.md").write_text(
        "# 02 PSE latent Phase alignment\n\n"
        "目的：直接检验完整 PSE latent temporal process H 中是否存在可测的共享 Domain Phase。"
        "本实验使用 held-out source-test / target-test；target true label 仅用于 oracle 分组。\n\n"
        "Phase 不修改 PSE latent value，只把 target token 的时间位置从 native t 改为 "
        "gamma^{-1}(t)。脚本在固定 128-D PSE 空间做 canonical 线性插值，在共同 support 上"
        "计算 class-mean integrated MSE/L2，并报告每个 target 样本到 source class mean 的"
        "before/after 距离。`pse_distance_summary.csv` 是正式数值结果；PCA 仅用于作图，且只"
        "由 source tokens 拟合。\n\n"
        "若大多数类别以及 class-equal macro 的 after distance 明显下降，支持 PSE latent 中存在"
        "共享 domain phase；若 SRVF geometry 改善而本目录 direct PSE distance 不改善，则当前"
        "Phase 主要是分解/SRVF 几何内部效应。\n",
        encoding="utf-8",
    )

    suite_manifest = {
        **common_manifest,
        "experiments": {
            "01_domain_phase_geometry_alignment": "01_domain_phase_geometry_alignment/manifest.json",
            "02_pse_latent_phase_alignment": "02_pse_latent_phase_alignment/manifest.json",
            "03_domain_phase_consistency": "03_domain_phase_consistency/manifest.json",
        },
        "phase_consistency": phase_consistency["availability"],
    }
    _json_dump(output_dir / "manifest.json", suite_manifest)
    (output_dir / "README_中文说明.md").write_text(
        "# Phase-only Domain Phase 独立验证套件\n\n"
        "本目录包含三个互相独立的 held-out 诊断：\n\n"
        "- `01_domain_phase_geometry_alignment/`：Trend / Structure-SRVF 与 LTAE 表示 before/after；\n"
        "- `02_pse_latent_phase_alignment/`：直接 128-D PSE latent support-aware distance；\n"
        "- `03_domain_phase_consistency/`：class-level Phase centers 与 group center 一致性。\n\n"
        "所有 target true labels 都是 oracle-only diagnostic，不参与训练、Phase estimation 或无监督选择。\n",
        encoding="utf-8",
    )
    print(
        "PHASE_ALIGNMENT_VIS_COMPLETE|"
        f"output={output_dir}|classes={len(summary_rows)}|samples={len(sample_rows)}",
        flush=True,
    )
    return suite_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize true-class source/target Shape before and after saved Domain Phase."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage-2 checkpoint providing confirmed Phase state and source statistics")
    parser.add_argument("--model-checkpoint", type=Path, default=None, help="Optional Stage-1 checkpoint providing model weights for a zero-step audit while reusing saved Phase state")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--classes", type=_parse_int_list, default=None)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--display-samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pse-grid-size", type=int, default=128)
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
    if args.pse_grid_size < 2:
        raise ValueError("pse-grid-size must be at least 2")
    if not 0.0 <= args.robust_lower < args.robust_upper <= 100.0:
        raise ValueError("robust percentile range is invalid")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    run(args)


if __name__ == "__main__":
    main()
