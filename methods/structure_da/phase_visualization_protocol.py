"""Pure helpers shared by post-hoc Domain Phase visualization scripts.

This module deliberately avoids dataset/plotting imports so protocol tests can
validate split reconstruction, checkpoint compatibility, C_use routing and PSE
latent distance semantics without requiring the full data stack.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


def checkpoint_model_state_dict(checkpoint: dict) -> dict:
    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, dict):
        return state_dict
    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, dict):
        return state_dict
    raise ValueError("checkpoint contains neither model_state_dict nor state_dict")


def phase_only_time_encoder(model):
    """Return the single-LTAE temporal encoder from the Phase-only model.

    Phase-only TSStructure exposes exactly one classification temporal path:
    ``temporal_module.raw_encoder.time_encoder``.  Keeping this lookup in the
    protocol module prevents visualization scripts from silently falling back
    to the removed dual-stream ``shared_ltae/shared_time_encoder`` hierarchy.
    """
    try:
        encoder = model.temporal_module.raw_encoder.time_encoder
    except AttributeError as error:
        raise RuntimeError(
            "Phase-only model must expose "
            "temporal_module.raw_encoder.time_encoder"
        ) from error
    if encoder is None:
        raise RuntimeError(
            "Phase-only model temporal_module.raw_encoder.time_encoder is None"
        )
    return encoder


def class_to_group(
    groups: Sequence[dict],
    *,
    phase_routes: Optional[Sequence[Optional[int]]] = None,
    num_classes: Optional[int] = None,
) -> Dict[int, dict]:
    """Resolve Phase usage routing (C_use), not only estimation membership."""
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


def reconstruct_fold_splits(
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
    """Reproduce ``train.create_train_val_test_folds`` for one requested fold."""
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


def canonicalize_pse_tokens(
    record: dict,
    *,
    positions_key: str,
    grid_size: int = 128,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Interpolate unchanged PSE token values on a fixed canonical time grid."""
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
    output[support] = values[lower] + fraction * (values[upper] - values[lower])
    return output.float(), support, grid.float()


def canonical_pse_center(
    records: Sequence[dict],
    *,
    positions_key: str,
    grid_size: int = 128,
) -> Tuple[Tensor, Tensor, Tensor]:
    trajectories: List[Tensor] = []
    supports: List[Tensor] = []
    grid: Optional[Tensor] = None
    for record in records:
        trajectory, support, current_grid = canonicalize_pse_tokens(
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


def pse_integrated_distance(
    left: Tensor,
    left_support: Tensor,
    right: Tensor,
    right_support: Tensor,
) -> Tuple[float, float, int]:
    common = left_support.bool() & right_support.bool()
    count = int(common.sum().item())
    if count < 2:
        return float("nan"), float("nan"), count
    squared_feature_error = (
        left[common].float() - right[common].float()
    ).square().mean(dim=-1)
    weights = torch.ones_like(squared_feature_error)
    if weights.numel() > 1:
        weights[[0, -1]] *= 0.5
    integrated_mse = float(
        (squared_feature_error * weights).sum().item() / weights.sum().item()
    )
    return integrated_mse, float(np.sqrt(max(integrated_mse, 0.0))), count


def pse_class_metrics(
    source_records: Sequence[dict],
    target_records: Sequence[dict],
    *,
    grid_size: int = 128,
) -> Tuple[dict, List[dict], dict]:
    source_center, source_support, grid = canonical_pse_center(
        source_records, positions_key="positions", grid_size=grid_size
    )
    target_before_center, target_before_support, _ = canonical_pse_center(
        target_records, positions_key="positions", grid_size=grid_size
    )
    target_after_center, target_after_support, _ = canonical_pse_center(
        target_records, positions_key="positions_after", grid_size=grid_size
    )
    before_mse, before_l2, before_common = pse_integrated_distance(
        source_center, source_support, target_before_center, target_before_support
    )
    after_mse, after_l2, after_common = pse_integrated_distance(
        source_center, source_support, target_after_center, target_after_support
    )

    sample_rows: List[dict] = []
    improved = 0
    valid_samples = 0
    for record in target_records:
        before, before_support, _ = canonicalize_pse_tokens(
            record, positions_key="positions", grid_size=grid_size
        )
        after, after_support, _ = canonicalize_pse_tokens(
            record, positions_key="positions_after", grid_size=grid_size
        )
        sample_before_mse, sample_before_l2, sample_before_common = pse_integrated_distance(
            source_center, source_support, before, before_support
        )
        sample_after_mse, sample_after_l2, sample_after_common = pse_integrated_distance(
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
    relative = (
        (before_l2 - after_l2) / max(before_l2, 1e-12)
        if np.isfinite(before_l2)
        else float("nan")
    )
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
