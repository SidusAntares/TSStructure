#!/usr/bin/env python3
"""Offline oracle audit of class-conditioned FreDN structural candidates."""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fredn.structural_probe import (
    TopologyMismatchError,
    align_curve_to_landmark_template,
    build_landmark_prototype,
    compute_topology_comparison,
    contrastive_feasibility_by_class,
    curve_correlation,
    detect_structural_landmarks,
    extract_fourier_conditions,
    fit_source_class_projections,
    median_absolute_deviation,
    modal_signature,
    normalized_l2,
    pointwise_intra_class_variance,
    pareto_modes,
    parse_fredn_checkpoint_specs,
    project_curves,
    robust_signal_scale,
    segment_shape_descriptors,
    summarize_topology,
    topology_signature,
)
from models.fredn.model import PseFreDNLTae
from models.stclassifier import PseLTae


DOMAIN_PATHS = {
    "AT1": "austria/33UVP/2017",
    "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
}
PROMINENCE_SENSITIVITY = (0.10, 0.15, 0.20)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_state_dict(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    return {name.removeprefix("module."): value for name, value in state.items()}


def _validate_source_checkpoint_config(path, num_modes, args):
    path = Path(path)
    config_path = path.parent.parent / "train_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"checkpoint config not found: {config_path}; refusing unverified reuse"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "model": "psefrednltae",
        "fredn_num_modes": num_modes,
        "fredn_fourier_solver": "dense_direct",
        "fredn_nufft_reg": args.fourier_reg,
        "fredn_period_days": args.period_days,
        "source": DOMAIN_PATHS[args.source],
        "target": DOMAIN_PATHS[args.source],
        "seed": args.seed,
        "num_folds": 1,
        "seq_length": 30,
        "num_pixels": args.num_pixels,
        "batch_size": args.batch_size,
        "input_dim": args.input_dim,
        "with_extra": args.with_extra,
        "closed_set": True,
        "combine_spring_and_winter": False,
        "with_shift_aug": False,
        "epochs": 100,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if path.parent.name != "fold_0":
        mismatches["fold"] = (path.parent.name, "fold_0")
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f"checkpoint config mismatch for mode {num_modes}: {details}")
    return config


def load_fredn_checkpoint(path, num_modes, num_classes, args, device):
    config = _validate_source_checkpoint_config(path, num_modes, args)
    state = _load_state_dict(path, device)
    mask_name = "frequency_disentangler.nonnegative_logits"
    if mask_name not in state:
        raise ValueError(f"checkpoint lacks {mask_name}")
    expected_mask_shape = ((num_modes // 2) + 1, 128)
    if tuple(state[mask_name].shape) != expected_mask_shape:
        raise ValueError(
            f"checkpoint mask shape {tuple(state[mask_name].shape)} does not match "
            f"num_modes={num_modes}, PSE_dim=128 ({expected_mask_shape})"
        )
    model = PseFreDNLTae(
        input_dim=args.input_dim,
        with_extra=args.with_extra,
        num_classes=num_classes,
        fredn_num_modes=num_modes,
        fredn_nufft_reg=args.fourier_reg,
        fredn_period_days=args.period_days,
        fredn_fourier_solver=args.fourier_solver,
    )
    model.load_state_dict(state, strict=True)
    if model.frequency_disentangler.channels != 128:
        raise ValueError("probe requires a 128-dimensional PSE output")
    if model.fredn_num_modes != num_modes:
        raise ValueError("constructed model num_modes does not match requested mode")
    metadata = {
        "mode": num_modes,
        "checkpoint_path": str(Path(path).resolve()),
        "checkpoint_model_class": type(model).__name__,
        "checkpoint_num_modes": model.fredn_num_modes,
        "source": config["source"],
        "seed": config["seed"],
        "fold": 0,
    }
    return model.to(device).eval().requires_grad_(False), metadata


def load_plain_checkpoint(path, num_classes, args, device):
    model = PseLTae(
        input_dim=args.input_dim,
        with_extra=args.with_extra,
        num_classes=num_classes,
    )
    model.load_state_dict(_load_state_dict(path, device), strict=True)
    return model.to(device).eval().requires_grad_(False)


def _build_protocol(data_root, source_path, target_path):
    from dataset import PixelSetData
    from utils import label_utils

    candidate_classes = [
        name
        for name in label_utils.get_classes(source_path.split("/")[0])
        if name != "unknown"
    ]
    candidate_source = PixelSetData(
        data_root,
        source_path,
        candidate_classes,
        closed_set=True,
        combine_spring_and_winter=False,
    )
    labels, counts = np.unique(candidate_source.get_labels(), return_counts=True)
    classes = [
        candidate_classes[int(label)]
        for label, count in zip(labels, counts)
        if count >= 200
    ]
    if not classes:
        raise ValueError("no source class has at least 200 eligible samples")
    return classes


def _build_dataset(data_root, domain_path, classes, num_pixels):
    from torchvision.transforms import transforms

    from dataset import PixelSetData
    from transforms import Normalize, RandomSamplePixels, ToTensor

    transform = transforms.Compose(
        [RandomSamplePixels(num_pixels), Normalize(), ToTensor()]
    )
    return PixelSetData(
        data_root,
        domain_path,
        classes,
        transform=transform,
        closed_set=True,
        combine_spring_and_winter=False,
    )


def _loader(dataset, batch_size, num_workers):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _to_device(sample, device, with_extra):
    pixels = sample["pixels"].to(device)
    mask = sample["valid_pixels"].to(device)
    positions = sample["positions"].to(device)
    labels = sample["label"].to(device)
    extra = sample["extra"].to(device) if with_extra else None
    return pixels, mask, positions, extra, labels


@torch.inference_mode()
def fit_source_projections(model, dataset, args, device):
    _seed_everything(args.seed)
    features = []
    labels = []
    for sample in _loader(dataset, args.batch_size, args.num_workers):
        pixels, mask, _, extra, batch_labels = _to_device(
            sample, device, args.with_extra
        )
        features.append(model.spatial_encoder(pixels, mask, extra).cpu())
        labels.append(batch_labels.cpu())
    return fit_source_class_projections(torch.cat(features), torch.cat(labels))


def _interpolate_raw(positions, projected, grid):
    result = []
    for times, values in zip(positions, projected):
        if grid[0] < times[0] or grid[-1] > times[-1]:
            raise ValueError("dense grid would require raw feature extrapolation")
        result.append(np.interp(grid, times, values))
    return np.stack(result)


def _select_oracle_group_projection(projected_by_class, labels):
    """Select already-computed projections only for offline oracle grouping."""
    selected = []
    for index, class_id in enumerate(labels.detach().cpu().tolist()):
        class_id = int(class_id)
        if class_id not in projected_by_class:
            raise KeyError(f"missing source projection for class {class_id}")
        selected.append(projected_by_class[class_id][index])
    return torch.stack(selected)


@torch.inference_mode()
def extract_condition_curves(model, dataset, projections, grid, args, device):
    _seed_everything(args.seed)
    condition_curves = {
        "raw": [],
        "fourier_recon": [],
        "fredn_trend": [],
    }
    labels_out = []
    grid_tensor = torch.as_tensor(grid, device=device, dtype=torch.float32)
    for sample in _loader(dataset, args.batch_size, args.num_workers):
        pixels, mask, positions, extra, labels = _to_device(
            sample, device, args.with_extra
        )
        spatial = model.spatial_encoder(pixels, mask, extra)
        projected_raw = _select_oracle_group_projection(
            project_curves(spatial, projections), labels
        )
        condition_curves["raw"].append(
            _interpolate_raw(
                positions.detach().cpu().numpy(),
                projected_raw.detach().cpu().numpy(),
                grid,
            )
        )
        if hasattr(model, "fourier_analyzer"):
            dense_positions = grid_tensor.unsqueeze(0).expand(spatial.shape[0], -1)
            dense_conditions = extract_fourier_conditions(
                model, spatial, positions, dense_positions
            )
            for condition, dense_features in dense_conditions.items():
                projected = _select_oracle_group_projection(
                    project_curves(dense_features, projections), labels
                )
                condition_curves[condition].append(
                    projected.detach().cpu().numpy()
                )
        labels_out.append(labels.detach().cpu().numpy())
    result = {
        "raw": np.concatenate(condition_curves["raw"]),
        "labels": np.concatenate(labels_out),
    }
    for condition in ("fourier_recon", "fredn_trend"):
        if condition_curves[condition]:
            result[condition] = np.concatenate(condition_curves[condition])
    return result


def _support(source_dataset, target_dataset, grid_step):
    source_starts = np.repeat(min(source_dataset.date_positions), len(source_dataset))
    source_ends = np.repeat(max(source_dataset.date_positions), len(source_dataset))
    target_starts = np.repeat(min(target_dataset.date_positions), len(target_dataset))
    target_ends = np.repeat(max(target_dataset.date_positions), len(target_dataset))
    start = max(np.quantile(source_starts, 0.05), np.quantile(target_starts, 0.05))
    end = min(np.quantile(source_ends, 0.95), np.quantile(target_ends, 0.95))
    if end <= start:
        raise ValueError("source and target temporal supports do not overlap")
    grid = np.arange(np.ceil(start), np.floor(end) + grid_step * 0.5, grid_step)
    if len(grid) < 3:
        raise ValueError("common temporal support is too short")
    return float(grid[0]), float(grid[-1]), grid


def _group_indices(labels):
    grouped = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        grouped[int(label)].append(index)
    return grouped


def _detect_by_class(curves, labels, thresholds, grid, min_distance):
    return [
        detect_structural_landmarks(
            grid,
            curve,
            min_distance_days=min_distance,
            prominence_threshold=thresholds[int(label)],
        )
        for curve, label in zip(curves, labels)
    ]


def _macro_mean(rows, key):
    values = [row[key] for row in rows if np.isfinite(row.get(key, np.nan))]
    return float(np.mean(values)) if values else float("nan")


def _landmark_comparison(source_signatures, target_signatures):
    values = compute_topology_comparison(source_signatures, target_signatures)
    return {
        "same_class_landmark_match": values["same_class_topology_match"],
        "target_modal_match_rate": values[
            "target_matches_source_modal_signature_rate"
        ],
        "different_class_landmark_collision": values[
            "different_class_topology_collision"
        ],
        "landmark_discrimination_margin": values[
            "topology_discrimination_margin"
        ],
    }


def _phase_and_shape(
    mode_count,
    condition,
    grid,
    source_curves,
    source_labels,
    source_landmarks,
    target_curves,
    target_labels,
    target_landmarks,
    classes,
):
    source_groups = _group_indices(source_labels)
    target_groups = _group_indices(target_labels)
    per_class = []
    landmark_rows = []
    aligned_sources = {}
    aligned_targets = {}
    source_prototypes = {}
    common_classes = sorted(set(source_groups) & set(target_groups))
    target_total_by_class = {
        class_id: len(target_groups[class_id]) for class_id in common_classes
    }
    for class_id in common_classes:
        source_indices = source_groups[class_id]
        target_indices = target_groups[class_id]
        source_sets = [source_landmarks[index] for index in source_indices]
        mode, prototype, _ = build_landmark_prototype(source_sets)
        accepted_source = [
            index
            for index in source_indices
            if topology_signature(source_landmarks[index]) == mode
        ]
        accepted_target = [
            index
            for index in target_indices
            if topology_signature(target_landmarks[index]) == mode
        ]
        base_row = {
            "mode": mode_count,
            "condition": condition,
            "class_id": class_id,
            "class_name": classes[class_id],
            "source_modal_signature": f"{mode[0]}:{mode[1]}",
            "source_accepted": len(accepted_source),
            "target_accepted": len(accepted_target),
            "target_total": len(target_indices),
            "phase_probe_coverage": len(accepted_target) / len(target_indices),
        }
        if not prototype or not accepted_source or not accepted_target:
            per_class.append(base_row)
            continue
        source_aligned = [
            align_curve_to_landmark_template(
                grid, source_curves[index], source_landmarks[index], prototype
            )
            for index in accepted_source
        ]
        target_aligned = [
            align_curve_to_landmark_template(
                grid, target_curves[index], target_landmarks[index], prototype
            )
            for index in accepted_target
        ]
        source_before = source_curves[accepted_source]
        target_before = target_curves[accepted_target]
        source_aligned = np.stack(source_aligned)
        target_aligned = np.stack(target_aligned)
        combined_before = np.concatenate([source_before, target_before])
        combined_after = np.concatenate([source_aligned, target_aligned])
        variance_before = pointwise_intra_class_variance(combined_before)
        variance_after = pointwise_intra_class_variance(combined_after)
        source_median_before = np.median(source_before, axis=0)
        target_median_before = np.median(target_before, axis=0)
        source_median_after = np.median(source_aligned, axis=0)
        target_median_after = np.median(target_aligned, axis=0)
        source_prototypes[class_id] = source_median_after
        aligned_sources[class_id] = source_aligned
        aligned_targets[class_id] = target_aligned
        per_class.append(
            {
                **base_row,
                "variance_before": variance_before,
                "variance_after": variance_after,
                "variance_reduction": 1.0 - variance_after / max(variance_before, 1e-12),
                "prototype_l2_before": normalized_l2(
                    source_median_before, target_median_before
                ),
                "prototype_l2_after": normalized_l2(
                    source_median_after, target_median_after
                ),
                "prototype_correlation_before": curve_correlation(
                    source_median_before, target_median_before
                ),
                "prototype_correlation_after": curve_correlation(
                    source_median_after, target_median_after
                ),
            }
        )
        for order, canonical in enumerate(prototype):
            source_items = [source_landmarks[index][order] for index in accepted_source]
            target_items = [target_landmarks[index][order] for index in accepted_target]
            source_times = [item.time for item in source_items]
            target_times = [item.time for item in target_items]
            source_amp = np.asarray([item.amplitude for item in source_items])
            target_amp = np.asarray([item.amplitude for item in target_items])
            source_prom = np.asarray([item.prominence for item in source_items])
            target_prom = np.asarray([item.prominence for item in target_items])
            pooled_amp = max(np.sqrt((source_amp.var() + target_amp.var()) / 2), 1e-12)
            pooled_prom = max(np.sqrt((source_prom.var() + target_prom.var()) / 2), 1e-12)
            landmark_rows.append(
                {
                    "record_type": "landmark",
                    "mode": mode_count,
                    "condition": condition,
                    "class_id": class_id,
                    "class_name": classes[class_id],
                    "landmark_order": order,
                    "kind": canonical.kind,
                    "canonical_time": canonical.time,
                    "source_time_std": float(np.std(source_times)),
                    "source_time_mad": median_absolute_deviation(source_times),
                    "target_time_std": float(np.std(target_times)),
                    "target_time_mad": median_absolute_deviation(target_times),
                    "source_target_median_time_offset": float(
                        np.median(target_times) - np.median(source_times)
                    ),
                    "source_amplitude_mean": float(source_amp.mean()),
                    "source_amplitude_std": float(source_amp.std()),
                    "source_amplitude_median": float(np.median(source_amp)),
                    "source_amplitude_mad": median_absolute_deviation(source_amp),
                    "target_amplitude_mean": float(target_amp.mean()),
                    "target_amplitude_std": float(target_amp.std()),
                    "target_amplitude_median": float(np.median(target_amp)),
                    "target_amplitude_mad": median_absolute_deviation(target_amp),
                    "standardized_amplitude_gap": float(
                        abs(source_amp.mean() - target_amp.mean()) / pooled_amp
                    ),
                    "source_prominence_mean": float(source_prom.mean()),
                    "source_prominence_std": float(source_prom.std()),
                    "source_prominence_median": float(np.median(source_prom)),
                    "source_prominence_mad": median_absolute_deviation(source_prom),
                    "target_prominence_mean": float(target_prom.mean()),
                    "target_prominence_std": float(target_prom.std()),
                    "target_prominence_median": float(np.median(target_prom)),
                    "target_prominence_mad": median_absolute_deviation(target_prom),
                    "standardized_prominence_gap": float(
                        abs(source_prom.mean() - target_prom.mean()) / pooled_prom
                    ),
                }
            )
        source_segments = [
            segment_shape_descriptors(grid, curve, prototype)
            for curve in source_aligned
        ]
        target_segments = [
            segment_shape_descriptors(grid, curve, prototype)
            for curve in target_aligned
        ]
        for order in range(max(mode[0] - 1, 0)):
            source_entries = [items[order] for items in source_segments if len(items) > order]
            target_entries = [items[order] for items in target_segments if len(items) > order]
            if not source_entries or not target_entries:
                continue
            row = {
                "record_type": "segment",
                "mode": mode_count,
                "condition": condition,
                "class_id": class_id,
                "class_name": classes[class_id],
                "segment_order": order,
                "left_kind": source_entries[0]["left_kind"],
                "right_kind": source_entries[0]["right_kind"],
            }
            for domain, entries in (
                ("source", source_entries),
                ("target", target_entries),
            ):
                for metric in ("duration", "slope", "area"):
                    metric_values = np.asarray([item[metric] for item in entries])
                    row[f"{domain}_{metric}_mean"] = float(metric_values.mean())
                    row[f"{domain}_{metric}_std"] = float(metric_values.std())
            landmark_rows.append(row)
    contrastive, contrastive_rows = contrastive_feasibility_by_class(
        source_prototypes,
        aligned_targets,
        target_total_by_class,
    )
    for row in contrastive_rows:
        row.update(
            {
                "mode": mode_count,
                "condition": condition,
                "class": classes[row["class_id"]],
                "class_name": classes[row["class_id"]],
            }
        )
    total_target = sum(row["target_total"] for row in per_class)
    total_accepted = sum(row["target_accepted"] for row in per_class)
    l2_before = _macro_mean(per_class, "prototype_l2_before")
    l2_after = _macro_mean(per_class, "prototype_l2_after")
    corr_before = _macro_mean(per_class, "prototype_correlation_before")
    corr_after = _macro_mean(per_class, "prototype_correlation_after")
    summary = {
        "mode": mode_count,
        "condition": condition,
        "num_classes": len(per_class),
        "phase_coverage_macro": _macro_mean(per_class, "phase_probe_coverage"),
        "phase_coverage_micro": (
            total_accepted / total_target if total_target else float("nan")
        ),
        "pointwise_variance_reduction": _macro_mean(per_class, "variance_reduction"),
        "prototype_l2_before": l2_before,
        "prototype_l2_after": l2_after,
        "prototype_l2_reduction": 1.0 - l2_after / max(l2_before, 1e-12),
        "prototype_corr_before": corr_before,
        "prototype_corr_after": corr_after,
        "prototype_corr_gain": corr_after - corr_before,
    }
    summary.update(contrastive)
    return (
        summary,
        per_class,
        landmark_rows,
        contrastive_rows,
        aligned_sources,
        aligned_targets,
    )


def _plot_condition(
    output_dir,
    condition,
    classes,
    grid,
    source_curves,
    source_labels,
    target_curves,
    target_labels,
    source_landmarks,
    aligned_sources=None,
    aligned_targets=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_groups = _group_indices(source_labels)
    target_groups = _group_indices(target_labels)
    for class_id in sorted(set(source_groups) & set(target_groups)):
        class_dir = Path(output_dir) / "figures" / classes[class_id]
        class_dir.mkdir(parents=True, exist_ok=True)
        source_values = source_curves[source_groups[class_id]]
        target_values = target_curves[target_groups[class_id]]
        _, prototype, _ = build_landmark_prototype(
            [source_landmarks[index] for index in source_groups[class_id]]
        )
        figure, axis = plt.subplots(figsize=(10, 4))
        for values, label, color in (
            (source_values, "source", "tab:blue"),
            (target_values, "target", "tab:orange"),
        ):
            median = np.median(values, axis=0)
            lower, upper = np.quantile(values, [0.25, 0.75], axis=0)
            axis.plot(grid, median, color=color, label=label)
            axis.fill_between(grid, lower, upper, color=color, alpha=0.2)
        for item in prototype:
            marker = "^" if item.kind == "peak" else "v"
            axis.scatter(item.time, item.amplitude, marker=marker, color="black")
        axis.set(title=f"{classes[class_id]} — {condition}", xlabel="day")
        axis.legend()
        figure.tight_layout()
        figure.savefig(class_dir / f"{condition}_before.png", dpi=150)
        plt.close(figure)

        if aligned_sources and class_id in aligned_sources and class_id in aligned_targets:
            figure, axis = plt.subplots(figsize=(10, 4))
            for values, label, color in (
                (aligned_sources[class_id], "source", "tab:blue"),
                (aligned_targets[class_id], "target", "tab:orange"),
            ):
                median = np.median(values, axis=0)
                lower, upper = np.quantile(values, [0.25, 0.75], axis=0)
                axis.plot(grid, median, color=color, label=label)
                axis.fill_between(grid, lower, upper, color=color, alpha=0.2)
            for item in prototype:
                axis.axvline(item.time, color="black", alpha=0.25)
            axis.set(title=f"{classes[class_id]} — oracle phase", xlabel="day")
            axis.legend()
            figure.tight_layout()
            figure.savefig(class_dir / f"{condition}_after_oracle_phase.png", dpi=150)
            plt.close(figure)

        gallery = class_dir / "sample_gallery"
        gallery.mkdir(exist_ok=True)
        figure, axis = plt.subplots(figsize=(10, 4))
        for index in source_groups[class_id][:3]:
            axis.plot(grid, source_curves[index], color="tab:blue", alpha=0.65)
        for index in target_groups[class_id][:3]:
            axis.plot(grid, target_curves[index], color="tab:orange", alpha=0.65)
        axis.set(title=f"{classes[class_id]} — {condition} samples", xlabel="day")
        figure.tight_layout()
        figure.savefig(gallery / f"{condition}.png", dpi=150)
        plt.close(figure)


def analyze_condition(
    mode_count,
    condition,
    grid,
    source_curves,
    source_labels,
    target_curves,
    target_labels,
    classes,
    args,
    output_dir,
):
    source_groups = _group_indices(source_labels)
    thresholds = {
        class_id: robust_signal_scale(source_curves[indices]) * args.prominence_rel
        for class_id, indices in source_groups.items()
    }
    source_landmarks = _detect_by_class(
        source_curves, source_labels, thresholds, grid, args.min_distance_days
    )
    target_landmarks = _detect_by_class(
        target_curves, target_labels, thresholds, grid, args.min_distance_days
    )
    source_signatures = defaultdict(list)
    target_signatures = defaultdict(list)
    for label, landmarks in zip(source_labels, source_landmarks):
        source_signatures[int(label)].append(topology_signature(landmarks))
    for label, landmarks in zip(target_labels, target_landmarks):
        target_signatures[int(label)].append(topology_signature(landmarks))

    topology_rows = []
    per_class_rows = []
    for domain, labels, landmarks in (
        ("source", source_labels, source_landmarks),
        ("target", target_labels, target_landmarks),
    ):
        for class_id, indices in _group_indices(labels).items():
            row = {
                "mode": mode_count,
                "condition": condition,
                "domain": domain,
                "class_id": class_id,
                "class_name": classes[class_id],
            }
            row.update(summarize_topology([landmarks[index] for index in indices]))
            per_class_rows.append(row)
    for class_id in sorted(set(source_signatures) & set(target_signatures)):
        row = {
            "mode": mode_count,
            "condition": condition,
            "class_id": class_id,
            "class_name": classes[class_id],
        }
        row.update(
            _landmark_comparison(
                {class_id: source_signatures[class_id]},
                {class_id: target_signatures[class_id]},
            )
        )
        topology_rows.append(row)
    modal_coverage = np.mean(
        [row["modal_signature_rate"] for row in per_class_rows if row["domain"] == "source"]
    )
    aggregate = {
        "mode": mode_count,
        "condition": condition,
        **_landmark_comparison(source_signatures, target_signatures),
        "source_modal_signature_rate": float(modal_coverage),
    }

    (
        phase_summary,
        phase_rows,
        landmark_rows,
        contrastive_rows,
        aligned_source,
        aligned_target,
    ) = (
        _phase_and_shape(
            mode_count,
            condition,
            grid,
            source_curves,
            source_labels,
            source_landmarks,
            target_curves,
            target_labels,
            target_landmarks,
            classes,
        )
    )
    _plot_condition(
        output_dir,
        f"m{mode_count}_{condition}",
        classes,
        grid,
        source_curves,
        source_labels,
        target_curves,
        target_labels,
        source_landmarks,
        aligned_source,
        aligned_target,
    )
    return {
        "aggregate": aggregate,
        "per_class": per_class_rows + topology_rows,
        "phase_summary": phase_summary,
        "phase_per_class": phase_rows,
        "landmarks": landmark_rows,
        "contrastive_per_class": contrastive_rows,
        "source_signatures": source_signatures,
        "target_signatures": target_signatures,
    }


def _sensitivity_rows(mode_count, condition, source_curves, source_labels, target_curves, target_labels, grid, args):
    rows = []
    source_groups = _group_indices(source_labels)
    for prominence_rel in PROMINENCE_SENSITIVITY:
        thresholds = {
            class_id: robust_signal_scale(source_curves[indices]) * prominence_rel
            for class_id, indices in source_groups.items()
        }
        source_landmarks = _detect_by_class(
            source_curves, source_labels, thresholds, grid, args.min_distance_days
        )
        target_landmarks = _detect_by_class(
            target_curves, target_labels, thresholds, grid, args.min_distance_days
        )
        source_signatures = defaultdict(list)
        target_signatures = defaultdict(list)
        for label, values in zip(source_labels, source_landmarks):
            source_signatures[int(label)].append(topology_signature(values))
        for label, values in zip(target_labels, target_landmarks):
            target_signatures[int(label)].append(topology_signature(values))
        rows.append(
            {
                "mode": mode_count,
                "condition": condition,
                "prominence_rel": prominence_rel,
                **_landmark_comparison(source_signatures, target_signatures),
            }
        )
    return rows


def _mode_sweep_row(aggregate, phase):
    return {
        "mode": aggregate["mode"],
        "condition": aggregate["condition"],
        "same_class_landmark_match": aggregate["same_class_landmark_match"],
        "different_class_landmark_collision": aggregate[
            "different_class_landmark_collision"
        ],
        "landmark_discrimination_margin": aggregate[
            "landmark_discrimination_margin"
        ],
        "source_modal_signature_rate": aggregate["source_modal_signature_rate"],
        "target_modal_match_rate": aggregate["target_modal_match_rate"],
        "phase_coverage_macro": phase["phase_coverage_macro"],
        "phase_coverage_micro": phase["phase_coverage_micro"],
        "pointwise_variance_reduction": phase["pointwise_variance_reduction"],
        "prototype_l2_reduction": phase["prototype_l2_reduction"],
        "prototype_corr_gain": phase["prototype_corr_gain"],
        "contrastive_positive_margin_micro": phase[
            "micro_positive_margin_rate"
        ],
        "contrastive_positive_margin_macro": phase[
            "macro_positive_margin_rate"
        ],
        "contrastive_margin_mean": phase["contrastive_margin_mean"],
        "eligible_class_count": phase["eligible_class_count"],
        "dominant_class_fraction": phase["dominant_class_fraction"],
    }


def _mask_effect_rows(mode_sweep_rows):
    by_key = {
        (row["mode"], row["condition"]): row for row in mode_sweep_rows
    }
    rows = []
    for mode_count in sorted({row["mode"] for row in mode_sweep_rows}):
        recon = by_key[(mode_count, "fourier_recon")]
        trend = by_key[(mode_count, "fredn_trend")]
        rows.append(
            {
                "mode": mode_count,
                "delta_same_class_match": trend["same_class_landmark_match"]
                - recon["same_class_landmark_match"],
                "delta_different_class_collision": trend[
                    "different_class_landmark_collision"
                ]
                - recon["different_class_landmark_collision"],
                "improvement_different_class_collision": recon[
                    "different_class_landmark_collision"
                ]
                - trend["different_class_landmark_collision"],
                "delta_landmark_margin": trend["landmark_discrimination_margin"]
                - recon["landmark_discrimination_margin"],
                "delta_modal_match_rate": trend["target_modal_match_rate"]
                - recon["target_modal_match_rate"],
                "delta_phase_coverage_macro": trend["phase_coverage_macro"]
                - recon["phase_coverage_macro"],
                "delta_phase_coverage_micro": trend["phase_coverage_micro"]
                - recon["phase_coverage_micro"],
                "delta_variance_reduction": trend[
                    "pointwise_variance_reduction"
                ]
                - recon["pointwise_variance_reduction"],
                "delta_prototype_l2_reduction": trend[
                    "prototype_l2_reduction"
                ]
                - recon["prototype_l2_reduction"],
                "delta_contrastive_positive_margin_macro": trend[
                    "contrastive_positive_margin_macro"
                ]
                - recon["contrastive_positive_margin_macro"],
                "delta_contrastive_margin_mean": trend[
                    "contrastive_margin_mean"
                ]
                - recon["contrastive_margin_mean"],
            }
        )
    return rows


def _plot_mode_summaries(output_dir, mode_sweep_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir) / "figures_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    trend = sorted(
        [row for row in mode_sweep_rows if row["condition"] == "fredn_trend"],
        key=lambda row: row["mode"],
    )
    modes = [row["mode"] for row in trend]
    for key, filename, ylabel in (
        ("landmark_discrimination_margin", "mode_vs_landmark_margin.png", "landmark discrimination margin"),
        ("phase_coverage_macro", "mode_vs_phase_coverage.png", "phase coverage (macro)"),
        ("contrastive_positive_margin_macro", "mode_vs_contrastive_positive_rate.png", "positive margin rate (macro)"),
    ):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(modes, [row[key] for row in trend], marker="o")
        axis.set(xlabel="Fourier modes", ylabel=ylabel)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    for row in trend:
        x_value = row["phase_coverage_macro"]
        y_value = row["contrastive_positive_margin_macro"]
        axis.scatter(x_value, y_value)
        axis.annotate(str(row["mode"]), (x_value, y_value))
    axis.set(
        xlabel="phase coverage (macro)",
        ylabel="contrastive positive margin (macro)",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "coverage_vs_discriminability.png", dpi=150)
    plt.close(figure)

    by_condition = {
        condition: sorted(
            [row for row in mode_sweep_rows if row["condition"] == condition],
            key=lambda row: row["mode"],
        )
        for condition in ("fourier_recon", "fredn_trend")
    }
    for key, filename, ylabel in (
        ("same_class_landmark_match", "mode_vs_same_class_match_recon_vs_trend.png", "same-class landmark match"),
        ("landmark_discrimination_margin", "mode_vs_landmark_margin_recon_vs_trend.png", "landmark discrimination margin"),
        ("contrastive_margin_mean", "mode_vs_contrastive_margin_recon_vs_trend.png", "contrastive margin mean"),
    ):
        figure, axis = plt.subplots(figsize=(7, 4))
        for condition, rows in by_condition.items():
            axis.plot(
                [row["mode"] for row in rows],
                [row[key] for row in rows],
                marker="o",
                label=condition,
            )
        axis.set(xlabel="Fourier modes", ylabel=ylabel)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)


def _write_round2_summary(path, source, target, mode_sweep_rows, mask_rows, pareto_rows):
    trend = [row for row in mode_sweep_rows if row["condition"] == "fredn_trend"]

    def maximizing(key):
        finite = [row for row in trend if np.isfinite(row[key])]
        if not finite:
            return []
        best = max(row[key] for row in finite)
        return [row["mode"] for row in finite if row[key] == best]

    leaders = {
        "landmark stability": maximizing("target_modal_match_rate"),
        "gate coverage": maximizing("phase_coverage_macro"),
        "class discriminability": maximizing("contrastive_positive_margin_macro"),
    }
    common_leaders = (
        set.intersection(*(set(values) for values in leaders.values()))
        if all(leaders.values())
        else set()
    )
    lines = [
        "# FreDN Structural Probe Mode Resolution Sweep",
        "",
        f"Source: `{source}`  ",
        f"Target: `{target}`  ",
        "ORACLE_ANALYSIS_ONLY=true  ",
        "TARGET_LABEL_USED_FOR_TRAINING=false",
        "",
        "## Resolution effect",
        "",
        "The raw, FourierRecon, and FreDN trend curves for each mode use the same "
        "source-raw class projection and paired samples. Direct comparisons across "
        "modes remain secondary because each checkpoint has separately trained PSE weights.",
        "",
    ]
    for criterion, modes in leaders.items():
        lines.append(f"- Modes maximizing {criterion}: {', '.join(map(str, modes))}.")
    lines.append("")
    if common_leaders:
        lines.append(
            "The same resolution leads all reported structural criteria: "
            + ", ".join(map(str, sorted(common_leaders)))
            + "."
        )
    else:
        lines.append("No single resolution dominates all structural criteria.")
    lines.extend(
        [
            "",
            "Pareto modes on macro phase coverage and macro contrastive positive-margin rate: "
            + ", ".join(str(row["mode"]) for row in pareto_rows)
            + ".",
            "",
            "## Learned-mask effect",
            "",
            "Positive deltas mean FreDN trend exceeds the same-mode FourierRecon value, "
            "except collision, for which `improvement_different_class_collision` is the "
            "higher-is-better form.",
            "",
            "| mode | delta same-class match | collision improvement | delta landmark margin | delta macro coverage | delta macro positive rate | delta contrastive margin |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mask_rows:
        lines.append(
            "| {mode} | {delta_same_class_match:.6g} | "
            "{improvement_different_class_collision:.6g} | "
            "{delta_landmark_margin:.6g} | {delta_phase_coverage_macro:.6g} | "
            "{delta_contrastive_positive_margin_macro:.6g} | "
            "{delta_contrastive_margin_mean:.6g} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpret learned-mask usefulness per mode from the joint movement of "
            "same-class match, different-class collision, discrimination margin, coverage, "
            "and contrastive diagnostics. Increased collision alone is not classified as "
            "over-smoothing when the same-class match and discrimination margin also improve.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source", choices=DOMAIN_PATHS, required=True)
    parser.add_argument("--target", choices=DOMAIN_PATHS, required=True)
    parser.add_argument("--fredn-checkpoint", action="append", default=[])
    parser.add_argument("--fredn9-checkpoint")
    parser.add_argument("--fredn17-checkpoint")
    parser.add_argument("--raw-checkpoint")
    parser.add_argument("--git-commit", default="unavailable")
    parser.add_argument("--git-branch", default="unavailable")
    parser.add_argument(
        "--git-dirty",
        choices=("clean", "dirty", "unavailable"),
        default="unavailable",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-pixels", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--with-extra", action="store_true")
    parser.add_argument("--grid-step-days", type=float, default=1.0)
    parser.add_argument("--prominence-rel", type=float, default=0.15)
    parser.add_argument("--min-distance-days", type=float, default=14.0)
    parser.add_argument("--period-days", type=float, default=365.0)
    parser.add_argument("--fourier-reg", type=float, default=1e-3)
    parser.add_argument(
        "--fourier-solver", choices=("dense_direct", "nufft_cg"), default="dense_direct"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source == args.target:
        raise SystemExit("source and target must differ")
    if args.grid_step_days <= 0:
        raise SystemExit("--grid-step-days must be positive")
    if args.num_workers != 0:
        raise SystemExit("structural probe requires --num-workers 0 for deterministic pixel sampling")
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise SystemExit(f"dataset root not found: {data_root}")
    try:
        checkpoint_paths = parse_fredn_checkpoint_specs(
            args.fredn_checkpoint,
            args.fredn9_checkpoint,
            args.fredn17_checkpoint,
        )
    except ValueError as error:
        raise SystemExit(str(error))
    for mode_count, checkpoint_path in checkpoint_paths.items():
        if not Path(checkpoint_path).is_file():
            raise SystemExit(
                f"checkpoint missing for mode {mode_count}: {checkpoint_path}"
            )
    if args.raw_checkpoint and not Path(args.raw_checkpoint).is_file():
        raise SystemExit(f"raw checkpoint missing: {args.raw_checkpoint}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    print("ORACLE_ANALYSIS_ONLY=true")
    print("TARGET_LABEL_USED_FOR_TRAINING=false")
    _seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = _build_protocol(
        args.data_root, DOMAIN_PATHS[args.source], DOMAIN_PATHS[args.target]
    )
    source_dataset = _build_dataset(
        args.data_root, DOMAIN_PATHS[args.source], classes, args.num_pixels
    )
    target_dataset = _build_dataset(
        args.data_root, DOMAIN_PATHS[args.target], classes, args.num_pixels
    )
    support_start, support_end, grid = _support(
        source_dataset, target_dataset, args.grid_step_days
    )
    topology_summary = []
    topology_per_class = []
    topology_sensitivity = []
    phase_summary = []
    phase_per_class = []
    landmark_statistics = []
    contrastive_per_class = []
    mode_sweep_rows = []
    checkpoint_manifest = []
    for mode_count, checkpoint_path in checkpoint_paths.items():
        print(f"PROBE_MODE_START|mode={mode_count}|checkpoint={checkpoint_path}")
        model, checkpoint_metadata = load_fredn_checkpoint(
            checkpoint_path,
            mode_count,
            len(classes),
            args,
            device,
        )
        checkpoint_manifest.append(checkpoint_metadata)
        projections = fit_source_projections(model, source_dataset, args, device)
        source = extract_condition_curves(
            model, source_dataset, projections, grid, args, device
        )
        target = extract_condition_curves(
            model, target_dataset, projections, grid, args, device
        )
        for condition in ("raw", "fourier_recon", "fredn_trend"):
            result = analyze_condition(
                mode_count,
                condition,
                grid,
                source[condition],
                source["labels"],
                target[condition],
                target["labels"],
                classes,
                args,
                output_dir,
            )
            topology_summary.append(result["aggregate"])
            topology_per_class.extend(result["per_class"])
            phase_summary.append(result["phase_summary"])
            phase_per_class.extend(result["phase_per_class"])
            landmark_statistics.extend(result["landmarks"])
            contrastive_per_class.extend(result["contrastive_per_class"])
            mode_sweep_rows.append(
                _mode_sweep_row(result["aggregate"], result["phase_summary"])
            )
            topology_sensitivity.extend(
                _sensitivity_rows(
                    mode_count,
                    condition,
                    source[condition],
                    source["labels"],
                    target[condition],
                    target["labels"],
                    grid,
                    args,
                )
            )
        del source, target, projections, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"PROBE_MODE_FINISHED|mode={mode_count}")

    if args.raw_checkpoint:
        raw_model = load_plain_checkpoint(
            args.raw_checkpoint, len(classes), args, device
        )
        projections = fit_source_projections(raw_model, source_dataset, args, device)
        source = extract_condition_curves(
            raw_model, source_dataset, projections, grid, args, device
        )
        target = extract_condition_curves(
            raw_model, target_dataset, projections, grid, args, device
        )
        result = analyze_condition(
            0,
            "plain_raw",
            grid,
            source["raw"],
            source["labels"],
            target["raw"],
            target["labels"],
            classes,
            args,
            output_dir,
        )
        topology_summary.append(result["aggregate"])
        topology_per_class.extend(result["per_class"])
        phase_summary.append(result["phase_summary"])
        phase_per_class.extend(result["phase_per_class"])
        landmark_statistics.extend(result["landmarks"])
        contrastive_per_class.extend(result["contrastive_per_class"])
        del source, target, projections, raw_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    resolution_tradeoff = [
        row for row in mode_sweep_rows if row["condition"] == "fredn_trend"
    ]
    mask_effect_rows = _mask_effect_rows(mode_sweep_rows)
    pareto_rows = pareto_modes(
        resolution_tradeoff,
        x_key="phase_coverage_macro",
        y_key="contrastive_positive_margin_macro",
    )

    manifest = {
        "git_commit": args.git_commit,
        "git_branch": args.git_branch,
        "git_dirty": args.git_dirty,
        "source": args.source,
        "target": args.target,
        "source_path": DOMAIN_PATHS[args.source],
        "target_path": DOMAIN_PATHS[args.target],
        "classes": classes,
        "seed": args.seed,
        "checkpoints": checkpoint_manifest,
        "raw_checkpoint": str(Path(args.raw_checkpoint).resolve()) if args.raw_checkpoint else None,
        "num_modes": list(checkpoint_paths),
        "period_days": args.period_days,
        "fourier_solver": args.fourier_solver,
        "temporal_support": {
            "support_start": support_start,
            "support_end": support_end,
            "support_days": support_end - support_start,
        },
        "grid_resolution_days": args.grid_step_days,
        "prominence_rel": args.prominence_rel,
        "min_distance_days": args.min_distance_days,
        "ORACLE_ANALYSIS_ONLY": True,
        "TARGET_LABEL_USED_FOR_TRAINING": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "topology_summary.csv", topology_summary)
    _write_csv(output_dir / "topology_per_class.csv", topology_per_class)
    _write_csv(output_dir / "topology_sensitivity.csv", topology_sensitivity)
    _write_csv(output_dir / "landmark_statistics.csv", landmark_statistics)
    _write_csv(output_dir / "phase_alignment_summary.csv", phase_summary)
    _write_csv(output_dir / "phase_alignment_per_class.csv", phase_per_class)
    _write_csv(output_dir / "contrastive_per_class.csv", contrastive_per_class)
    _write_csv(output_dir / "mode_sweep_summary.csv", mode_sweep_rows)
    _write_csv(output_dir / "resolution_tradeoff.csv", resolution_tradeoff)
    _write_csv(output_dir / "mask_effect_summary.csv", mask_effect_rows)
    _write_csv(output_dir / "pareto_modes.csv", pareto_rows)
    _plot_mode_summaries(output_dir, mode_sweep_rows)
    _write_round2_summary(
        output_dir / "summary.md",
        args.source,
        args.target,
        mode_sweep_rows,
        mask_effect_rows,
        pareto_rows,
    )
    print(f"STRUCTURAL_PROBE_COMPLETE|output_dir={output_dir}")


if __name__ == "__main__":
    main()
