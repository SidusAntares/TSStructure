#!/usr/bin/env python3
"""Offline oracle Mode-13 phase/shape diagnostic; never trains a model."""

import argparse
import csv
import hashlib
import json
import random
import sys
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.phase_shape_diagnostic import (
    _periodic_shift,
    amplitude_metrics,
    build_pointwise_median_prototypes,
    estimate_nonlinear_phase,
    estimate_scalar_phase,
    landmark_alignment_metrics,
    registration_metrics,
    robust_normalize,
    shape_margin,
    warp_curve,
)
from models.fredn.structural_probe import (
    build_direct_fourier_views,
    fit_source_class_projections,
)
from models.stclassifier import PseLTae


DOMAINS = {
    "AT1": "austria/33UVP/2017",
    "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
}
EPS = np.finfo(np.float64).eps


def write_csv(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def load_model(checkpoint, classes, device, source_path, seed):
    checkpoint = Path(checkpoint)
    config_path = checkpoint.parent.parent / "train_config.json"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"source checkpoint/config missing: {checkpoint}, {config_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "model": "pseltae",
        "source": source_path,
        "target": source_path,
        "seed": seed,
        "num_folds": 1,
    }
    mismatch = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if checkpoint.parent.name != "fold_0":
        mismatch["fold"] = (checkpoint.parent.name, "fold_0")
    if mismatch:
        raise ValueError(f"source checkpoint metadata mismatch: {mismatch}")
    model = PseLTae(
        input_dim=config.get("input_dim", 10),
        with_extra=config.get("with_extra", False),
        num_classes=len(classes),
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(
        {key.removeprefix("module."): value for key, value in state_dict.items()}
    )
    return model.to(device).eval().requires_grad_(False), config


def read_checkpoint_config(checkpoint):
    checkpoint = Path(checkpoint)
    config_path = checkpoint.parent.parent / "train_config.json"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"source checkpoint/config missing: {checkpoint}, {config_path}"
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def replay_fold_zero(args):
    """Replay create_train_val_test_folds once for [source, target]."""
    from dataset import PixelSetData
    from train import create_train_val_test_folds
    from utils import label_utils

    source_path, target_path = DOMAINS[args.source], DOMAINS[args.target]
    candidates = [
        value
        for value in label_utils.get_classes(source_path.split("/")[0])
        if value != "unknown"
    ]
    candidate_source = PixelSetData(
        args.data_root, source_path, candidates, closed_set=True
    )
    labels, counts = np.unique(candidate_source.get_labels(), return_counts=True)
    classes = [
        candidates[int(label)]
        for label, count in zip(labels, counts)
        if count >= 200
    ]
    source = PixelSetData(args.data_root, source_path, classes, closed_set=True)
    target = PixelSetData(args.data_root, target_path, classes, closed_set=True)
    eligible = {
        source_path: source.get_parcel_indices().tolist(),
        target_path: target.get_parcel_indices().tolist(),
    }
    random.seed(args.seed)
    split = create_train_val_test_folds(
        [source_path, target_path],
        1,
        eligible,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )[0]
    return classes, split


def dataset_for(path, classes, indices, args):
    """Full-pixel, full-temporal-sequence dataset with no augmentation."""
    from dataset import PixelSetData
    from torchvision.transforms import transforms
    from transforms import Normalize, ToTensor

    transform = transforms.Compose([Normalize(), ToTensor()])
    return PixelSetData(
        args.data_root,
        path,
        classes,
        transform=transform,
        indices=indices,
        with_extra=args.with_extra,
        closed_set=True,
        combine_spring_and_winter=False,
    )


def cache_loader(dataset, batch_size):
    from dataset import GroupByShapesBatchSampler

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_sampler=GroupByShapesBatchSampler(dataset, batch_size),
    )


def redact_dataset_labels(dataset):
    """Replace target labels before __getitem__/collation without touching source data."""
    redacted = copy(dataset)
    redacted.samples = [
        (path, parcel_index, 0, extra)
        for path, parcel_index, _label, extra in dataset.samples
    ]
    return redacted


def shift_loader(dataset, args):
    """Match TimeMatch's epoch-0 weak-view shift loader semantics."""
    from torchvision.transforms import transforms
    from transforms import Normalize, RandomSamplePixels, ToTensor

    weak = deepcopy(dataset)
    weak.transform = transforms.Compose(
        [RandomSamplePixels(args.num_pixels), Normalize(), ToTensor()]
    )
    return torch.utils.data.DataLoader(
        weak,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )


@torch.inference_mode()
def epoch0_shift(model, target_dataset, args, device):
    from timematch import (
        estimate_class_distribution,
        estimate_temporal_shift,
        get_pseudo_labels,
    )

    target_loader = shift_loader(target_dataset, args)
    if len(target_loader) < args.shift_sample_size:
        raise ValueError(
            "TimeMatch epoch-0 shift requires "
            f"{args.shift_sample_size} batches, but target train provides "
            f"only {len(target_loader)}"
        )
    initial = estimate_temporal_shift(
        model,
        target_loader,
        device,
        min_shift=-60,
        max_shift=60,
        sample_size=args.shift_sample_size,
        shift_estimator="IS",
        progress_bar="off",
    )
    pseudo = get_pseudo_labels(
        model, target_loader, device, initial, n=None, progress_bar="off"
    )
    distribution = estimate_class_distribution(
        torch.argmax(pseudo, dim=1), len(args.classes)
    )
    low, high = (0, 60) if initial >= 0 else (-60, 0)
    final = estimate_temporal_shift(
        model,
        target_loader,
        device,
        class_distribution=distribution,
        min_shift=low,
        max_shift=high,
        sample_size=args.shift_sample_size,
        shift_estimator="AM",
        progress_bar="off",
    )
    return int(final), int(initial)


@torch.inference_mode()
def extract_cache(spatial_encoder, dataset, shift, args, device):
    grid = torch.linspace(
        0.0, 365.0, 65, dtype=torch.float32, device=device
    )[:-1]
    views, positions, parcel_ids = [], [], []
    pse_forward_count = 0
    for sample in cache_loader(dataset, args.batch_size):
        pixels = sample["pixels"].to(device)
        valid = sample["valid_pixels"].to(device)
        extra = sample["extra"].to(device) if args.with_extra else None
        spatial = spatial_encoder(pixels, valid, extra)
        pse_forward_count += 1
        original_positions = sample["positions"].to(device)
        dense_positions = grid.unsqueeze(0).expand(len(spatial), -1)
        view = build_direct_fourier_views(
            spatial,
            original_positions + shift,
            dense_positions,
            mode_counts=(13,),
            period_days=365.0,
            reg=1e-3,
        )[13]
        views.append(view.cpu())
        positions.append(sample["positions"].cpu())
        parcel_ids.append(sample["parcel_index"].cpu())
    return {
        "sample_id": torch.cat(parcel_ids),
        "positions": torch.cat(positions),
        "mode13_features": torch.cat(views),
        "grid_days": grid.cpu(),
        "pse_batch_forward_count": pse_forward_count,
    }


def _by_kind(landmarks):
    return {
        kind: [landmark for landmark in landmarks if landmark.kind == kind]
        for kind in ("peak", "valley")
    }


def _ratio(target, source):
    denominator = source if abs(source) > EPS else np.copysign(EPS, source or 1.0)
    return float(target / denominator)


def _finite_mean(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def _peak_valley_range(landmarks, fallback_curve):
    peaks = [item.amplitude for item in landmarks if item.kind == "peak"]
    valleys = [item.amplitude for item in landmarks if item.kind == "valley"]
    if peaks and valleys:
        return float(max(peaks) - min(valleys))
    return float(np.ptp(fallback_curve))


def _split_hash(indices):
    values = ",".join(str(value) for value in sorted(indices)).encode("utf-8")
    return hashlib.sha256(values).hexdigest()


def labels_for_cache(dataset, cache):
    """Join labels by parcel id after extraction, preserving cache order."""
    label_by_parcel = dict(
        zip(dataset.get_parcel_indices().tolist(), dataset.get_labels().tolist())
    )
    return np.asarray(
        [label_by_parcel[int(parcel)] for parcel in cache["sample_id"].tolist()]
    )


def analyze(args, source_cache, target_cache, classes):
    source_values = source_cache["mode13_features"].numpy()
    target_values = target_cache["mode13_features"].numpy()
    source_labels = np.asarray(source_cache["labels"])
    target_labels = np.asarray(target_cache["labels"])
    source_prototypes = build_pointwise_median_prototypes(
        source_values, source_labels
    )
    target_prototypes = build_pointwise_median_prototypes(
        target_values, target_labels
    )
    projections = fit_source_class_projections(
        torch.from_numpy(source_values), torch.from_numpy(source_labels)
    )
    common = sorted(set(source_prototypes) & set(target_prototypes))
    grid = np.linspace(0.0, 365.0, 64, endpoint=False)
    phase_rows, shape_rows, landmark_rows, segment_rows = [], [], [], []
    artifacts = {
        "source_raw_prototypes": {},
        "target_raw_prototypes": {},
        "source_norm_prototypes": {},
        "target_norm_prototypes": {},
        "scalar_delta_by_class": {},
        "gamma_by_class": {},
        "source_pca_direction_by_class": {},
    }
    summaries = []

    for class_id in common:
        source_raw = source_prototypes[class_id]
        target_global = target_prototypes[class_id]
        source_norm = robust_normalize(source_raw)
        target_norm = robust_normalize(target_global)
        scalar_delta, _ = estimate_scalar_phase(source_norm, target_norm)
        target_scalar = _periodic_shift(target_global, scalar_delta, 365.0)
        nonlinear = estimate_nonlinear_phase(source_norm, target_norm)
        target_nonlinear = warp_curve(target_global, nonlinear.gamma)
        aligned = {
            "global": target_global,
            "scalar": target_scalar,
            "nonlinear": target_nonlinear,
        }
        loading = projections[class_id].numpy()
        source_curve = source_raw @ loading
        curves = {name: value @ loading for name, value in aligned.items()}
        prominence = args.prominence_rel * max(np.ptp(source_curve), EPS)
        registrations, landmark_metrics = {}, {}
        for phase_type, target_curve in curves.items():
            registrations[phase_type] = registration_metrics(
                source_norm, robust_normalize(aligned[phase_type])
            )
            landmark_metrics[phase_type] = landmark_alignment_metrics(
                source_curve,
                target_curve,
                grid,
                prominence,
                args.min_distance_days,
            )

        global_error = registrations["global"]["normalized_l2"]
        scalar_error = registrations["scalar"]["normalized_l2"]
        nonlinear_error = registrations["nonlinear"]["normalized_l2"]
        improvements = {
            "scalar_improvement_vs_global": global_error - scalar_error,
            "nonlinear_improvement_vs_global": global_error - nonlinear_error,
            "nonlinear_improvement_vs_scalar": scalar_error - nonlinear_error,
        }
        for phase_type in ("global", "scalar", "nonlinear"):
            if phase_type == "nonlinear":
                mean_disp = nonlinear.mean_displacement
                max_disp = nonlinear.max_displacement
                p95_disp = nonlinear.p95_displacement
                min_derivative = nonlinear.min_derivative
                max_derivative = nonlinear.max_derivative
                roughness = nonlinear.roughness
                valid = nonlinear.valid
                reason = nonlinear.failure_reason
            elif phase_type == "scalar":
                mean_disp = max_disp = p95_disp = abs(scalar_delta) / 365.0
                min_derivative = max_derivative = 1.0
                roughness, valid, reason = 0.0, True, ""
            else:
                mean_disp = max_disp = p95_disp = 0.0
                min_derivative = max_derivative = 1.0
                roughness, valid, reason = 0.0, True, ""
            registration = registrations[phase_type]
            landmarks = landmark_metrics[phase_type]
            phase_rows.append(
                {
                    "task": args.task,
                    "class_index": class_id,
                    "class_name": classes[class_id],
                    "source_count": int(np.sum(source_labels == class_id)),
                    "target_count": int(np.sum(target_labels == class_id)),
                    "phase_type": phase_type,
                    "registration_l2": registration["l2"],
                    "registration_normalized_l2": registration["normalized_l2"],
                    "registration_error": registration["normalized_l2"],
                    "registration_corr": registration["correlation"],
                    **improvements,
                    "gamma_mean_displacement": mean_disp,
                    "gamma_max_displacement": max_disp,
                    "gamma_p95_displacement": p95_disp,
                    "gamma_mean_displacement_days": mean_disp * 365.0,
                    "gamma_max_displacement_days": max_disp * 365.0,
                    "gamma_p95_displacement_days": p95_disp * 365.0,
                    "gamma_deviation": mean_disp,
                    "min_gamma_derivative": min_derivative,
                    "max_gamma_derivative": max_derivative,
                    "gamma_roughness": roughness,
                    "matched_peak_count": landmarks["matched_peak_count"],
                    "matched_valley_count": landmarks["matched_valley_count"],
                    "unmatched_source_count": landmarks["unmatched_source_count"],
                    "unmatched_target_count": landmarks["unmatched_target_count"],
                    "landmark_time_error_mean": landmarks["mean_time_error"],
                    "landmark_time_error_median": landmarks["median_time_error"],
                    "landmark_time_error_p95": landmarks["p95_time_error"],
                    "phase_valid": valid,
                    "phase_failure_reason": reason,
                    "failure_reason": reason,
                }
            )

        paired_by_phase = {
            phase_type: {
                (pair[0].kind, pair[0].time): pair[1]
                for pair in values["matched_pairs"]
            }
            for phase_type, values in landmark_metrics.items()
        }
        nonlinear_pairs = list(landmark_metrics["nonlinear"]["matched_pairs"])
        peak_differences, peak_ratios = [], []
        valley_differences, valley_ratios = [], []
        prominence_differences, prominence_ratios = [], []
        landmark_index = 0
        for source_mark, target_mark in nonlinear_pairs:
            kind = source_mark.kind
            difference = target_mark.amplitude - source_mark.amplitude
            ratio = _ratio(target_mark.amplitude, source_mark.amplitude)
            if kind == "peak":
                peak_differences.append(difference)
                peak_ratios.append(ratio)
                prominence_differences.append(
                    target_mark.prominence - source_mark.prominence
                )
                prominence_ratios.append(
                    _ratio(target_mark.prominence, source_mark.prominence)
                )
            else:
                valley_differences.append(difference)
                valley_ratios.append(ratio)
            phase_times = {}
            key = (source_mark.kind, source_mark.time)
            for phase_type in ("global", "scalar", "nonlinear"):
                match = paired_by_phase[phase_type].get(key)
                phase_times[phase_type] = (
                    match.time if match is not None else np.nan
                )
            landmark_rows.append(
                {
                    "task": args.task,
                    "class": classes[class_id],
                    "landmark_index": landmark_index,
                    "landmark_type": kind,
                    "source_time": source_mark.time,
                    "target_time_global": phase_times["global"],
                    "target_time_scalar": phase_times["scalar"],
                    "target_time_nonlinear": phase_times["nonlinear"],
                    "global_time_error": abs(
                        phase_times["global"] - source_mark.time
                    ),
                    "scalar_time_error": abs(
                        phase_times["scalar"] - source_mark.time
                    ),
                    "nonlinear_time_error": abs(
                        phase_times["nonlinear"] - source_mark.time
                    ),
                    "source_value": source_mark.amplitude,
                    "target_value_aligned": target_mark.amplitude,
                    "height_difference": difference,
                    "height_absolute_difference": abs(difference),
                    "height_relative_difference": difference
                    / max(abs(source_mark.amplitude), EPS),
                    "height_ratio": ratio,
                    "source_prominence": source_mark.prominence,
                    "target_prominence": target_mark.prominence,
                    "prominence_difference": target_mark.prominence
                    - source_mark.prominence,
                    "prominence_ratio": _ratio(
                        target_mark.prominence, source_mark.prominence
                    ),
                }
            )
            landmark_index += 1

        nonlinear_pairs.sort(key=lambda pair: pair[0].time)
        peak_valley_differences, peak_valley_ratios = [], []
        for (source_left, target_left), (source_right, target_right) in zip(
            nonlinear_pairs, nonlinear_pairs[1:]
        ):
            if source_left.kind == source_right.kind:
                continue
            source_amplitude = abs(source_right.amplitude - source_left.amplitude)
            target_amplitude = abs(target_right.amplitude - target_left.amplitude)
            peak_valley_differences.append(target_amplitude - source_amplitude)
            peak_valley_ratios.append(_ratio(target_amplitude, source_amplitude))

        class_segments = []
        for segment_index, ((source_a, target_a), (source_b, target_b)) in enumerate(
            zip(nonlinear_pairs, nonlinear_pairs[1:])
        ):
            source_i, source_j = np.searchsorted(grid, [source_a.time, source_b.time])
            target_i, target_j = np.searchsorted(grid, [target_a.time, target_b.time])
            if source_j <= source_i or target_j <= target_i:
                continue
            source_segment = source_curve[source_i : source_j + 1]
            target_segment = curves["nonlinear"][target_i : target_j + 1]
            target_segment = np.interp(
                np.linspace(0.0, 1.0, len(source_segment)),
                np.linspace(0.0, 1.0, len(target_segment)),
                target_segment,
            )
            source_auc = float(np.trapz(source_segment))
            target_auc = float(np.trapz(target_segment))
            source_diff, target_diff = np.diff(source_segment), np.diff(target_segment)
            row = {
                "task": args.task,
                "class": classes[class_id],
                "segment_index": segment_index,
                "start_landmark_type": source_a.kind,
                "end_landmark_type": source_b.kind,
                "source_length_days": source_b.time - source_a.time,
                "target_length_days": target_b.time - target_a.time,
                "source_auc": source_auc,
                "target_auc": target_auc,
                "auc_ratio": _ratio(abs(target_auc), abs(source_auc)),
                "source_mean": float(np.mean(source_segment)),
                "target_mean": float(np.mean(target_segment)),
                "source_range": float(np.ptp(source_segment)),
                "target_range": float(np.ptp(target_segment)),
                "range_ratio": _ratio(np.ptp(target_segment), np.ptp(source_segment)),
                "source_std": float(np.std(source_segment)),
                "target_std": float(np.std(target_segment)),
                "std_ratio": _ratio(np.std(target_segment), np.std(source_segment)),
                "source_total_rise": float(source_diff[source_diff > 0].sum()),
                "target_total_rise": float(target_diff[target_diff > 0].sum()),
                "source_total_fall": float(-source_diff[source_diff < 0].sum()),
                "target_total_fall": float(-target_diff[target_diff < 0].sum()),
                "segment_l1": float(np.mean(np.abs(target_segment - source_segment))),
                "segment_l2": float(np.sqrt(np.mean((target_segment - source_segment) ** 2))),
                "segment_corr": float(np.corrcoef(source_segment, target_segment)[0, 1])
                if np.std(source_segment) * np.std(target_segment) > EPS
                else 0.0,
            }
            segment_rows.append(row)
            class_segments.append(row)

        margin = shape_margin(class_id, target_nonlinear, source_prototypes)
        amplitude = amplitude_metrics(source_curve, curves["nonlinear"])
        source_pv = _peak_valley_range(
            landmark_metrics["nonlinear"]["source_landmarks"], source_curve
        )
        target_pv = _peak_valley_range(
            landmark_metrics["nonlinear"]["target_landmarks"], curves["nonlinear"]
        )
        shape_rows.append(
            {
                "task": args.task,
                "class_index": class_id,
                "class_name": classes[class_id],
                **margin,
                "global_range_source": amplitude["range_source"],
                "global_range_target": amplitude["range_target"],
                "global_range_ratio": amplitude["range_ratio"],
                "global_std_source": amplitude["std_source"],
                "global_std_target": amplitude["std_target"],
                "global_std_ratio": amplitude["std_ratio"],
                "global_iqr_source": amplitude["iqr_source"],
                "global_iqr_target": amplitude["iqr_target"],
                "global_iqr_ratio": amplitude["iqr_ratio"],
                "peak_to_valley_source": source_pv,
                "peak_to_valley_target": target_pv,
                "peak_to_valley_ratio": _ratio(target_pv, source_pv),
                "peak_height_diff_mean": _finite_mean(peak_differences),
                "peak_height_ratio_mean": _finite_mean(peak_ratios),
                "valley_height_diff_mean": _finite_mean(valley_differences),
                "valley_height_ratio_mean": _finite_mean(valley_ratios),
                "prominence_diff_mean": _finite_mean(prominence_differences),
                "prominence_ratio_mean": _finite_mean(prominence_ratios),
                "peak_valley_amplitude_diff_mean": _finite_mean(peak_valley_differences),
                "peak_valley_amplitude_ratio_mean": _finite_mean(peak_valley_ratios),
                "segment_auc_ratio_mean": _finite_mean(
                    row["auc_ratio"] for row in class_segments
                ),
                "segment_range_ratio_mean": _finite_mean(
                    row["range_ratio"] for row in class_segments
                ),
                "segment_l2_mean": _finite_mean(
                    row["segment_l2"] for row in class_segments
                ),
            }
        )
        artifacts["source_raw_prototypes"][class_id] = source_raw
        artifacts["target_raw_prototypes"][class_id] = target_global
        artifacts["source_norm_prototypes"][class_id] = source_norm
        artifacts["target_norm_prototypes"][class_id] = target_norm
        artifacts["scalar_delta_by_class"][class_id] = scalar_delta
        artifacts["gamma_by_class"][class_id] = nonlinear.gamma
        artifacts["source_pca_direction_by_class"][class_id] = loading
        summaries.append((registrations, landmark_metrics))
        plot_class(
            args.output_dir / f"class_{class_id}_phase_shape.png",
            grid,
            source_curve,
            curves,
            nonlinear.gamma,
            classes[class_id],
        )

    plot_gammas(args.output_dir / "gamma_by_class.png", artifacts, classes)
    summary = task_summary(args.task, summaries, shape_rows, segment_rows)
    return phase_rows, shape_rows, landmark_rows, segment_rows, artifacts, summary


def task_summary(task, summaries, shape_rows, segment_rows):
    def registration(phase_type):
        return _finite_mean(
            value[0][phase_type]["normalized_l2"] for value in summaries
        )

    def landmark(phase_type):
        return _finite_mean(
            value[1][phase_type]["mean_time_error"] for value in summaries
        )

    return {
        "task": task,
        "num_common_classes": len(summaries),
        "global_registration_error_mean": registration("global"),
        "scalar_registration_error_mean": registration("scalar"),
        "nonlinear_registration_error_mean": registration("nonlinear"),
        "global_landmark_error_mean": landmark("global"),
        "scalar_landmark_error_mean": landmark("scalar"),
        "nonlinear_landmark_error_mean": landmark("nonlinear"),
        "nonlinear_better_than_scalar_rate": _finite_mean(
            value[0]["nonlinear"]["normalized_l2"]
            < value[0]["scalar"]["normalized_l2"]
            for value in summaries
        ),
        "shape_same_class_distance_mean": _finite_mean(
            row["same_class_shape_distance"] for row in shape_rows
        ),
        "shape_wrong_class_distance_mean": _finite_mean(
            row["nearest_wrong_distance"] for row in shape_rows
        ),
        "shape_positive_margin_rate": _finite_mean(
            row["margin_positive"] for row in shape_rows
        ),
        "amplitude_ratio_mean": _finite_mean(
            row["global_range_ratio"] for row in shape_rows
        ),
        "amplitude_ratio_std": float(
            np.nanstd([row["global_range_ratio"] for row in shape_rows])
        ),
        "prominence_ratio_mean": _finite_mean(
            row["prominence_ratio_mean"] for row in shape_rows
        ),
        "prominence_ratio_std": float(
            np.nanstd([row["prominence_ratio_mean"] for row in shape_rows])
        ),
        "segment_auc_ratio_mean": _finite_mean(
            row["auc_ratio"] for row in segment_rows
        ),
        "segment_auc_ratio_std": float(
            np.nanstd([row["auc_ratio"] for row in segment_rows])
        )
        if segment_rows
        else np.nan,
    }


def plot_class(path, grid, source, curves, gamma, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    for axis, phase_type in zip(axes[:3], ("global", "scalar", "nonlinear")):
        axis.plot(grid, source, label="source")
        axis.plot(grid, curves[phase_type], label=f"target {phase_type}")
        axis.legend()
    axes[3].plot(grid, curves["nonlinear"] - source, label="aligned residual")
    axes[3].legend()
    figure.suptitle(name)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_gammas(path, artifacts, classes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.linspace(0.0, 1.0, 128)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(time, time, "k--", label="identity")
    for class_id, gamma in artifacts["gamma_by_class"].items():
        axis.plot(time, gamma, label=classes[class_id])
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source", choices=DOMAINS, required=True)
    parser.add_argument("--target", choices=DOMAINS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-pixels", type=int, default=64)
    parser.add_argument("--global-shift", type=int)
    parser.add_argument("--shift-sample-size", type=int, default=100)
    parser.add_argument("--prominence-rel", type=float, default=0.15)
    parser.add_argument("--min-distance-days", type=float, default=14.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.task = f"{args.source}_{args.target}"
    if args.source == args.target:
        raise ValueError("source and target must differ")
    if not Path(args.data_root).is_dir():
        raise FileNotFoundError(f"data root missing: {args.data_root}")
    try:
        import fdasrsf
    except ImportError as error:
        raise RuntimeError(
            "required local dependency fdasrsf==2.6.1 is not installed"
        ) from error
    if getattr(fdasrsf, "__version__", None) != "2.6.1":
        raise RuntimeError(
            f"fdasrsf==2.6.1 required, found {getattr(fdasrsf, '__version__', 'unknown')}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    checkpoint_config = read_checkpoint_config(args.checkpoint)
    args.val_ratio = float(checkpoint_config.get("val_ratio", 0.1))
    args.test_ratio = float(checkpoint_config.get("test_ratio", 0.2))
    classes, split = replay_fold_zero(args)
    args.classes = classes
    device = torch.device(args.device)
    source_path, target_path = DOMAINS[args.source], DOMAINS[args.target]
    model, config = load_model(
        args.checkpoint, classes, device, source_path, args.seed
    )
    args.with_extra = bool(config.get("with_extra", False))
    source_indices = split[source_path]["train"]
    target_indices = split[target_path]["train"]
    source_dataset = dataset_for(source_path, classes, source_indices, args)
    target_dataset = dataset_for(target_path, classes, target_indices, args)
    target_features_only = redact_dataset_labels(target_dataset)

    if args.global_shift is None:
        global_shift, initial_shift = epoch0_shift(
            model, target_features_only, args, device
        )
        global_shift_source = "epoch0_source_checkpoint_IS_then_AM"
    else:
        global_shift, initial_shift = args.global_shift, None
        global_shift_source = "explicit_override"

    source_cache = extract_cache(model.spatial_encoder, source_dataset, 0, args, device)
    target_cache = extract_cache(
        model.spatial_encoder, target_features_only, global_shift, args, device
    )
    # First target-label access: both label-free Mode-13 caches are complete.
    source_cache["labels"] = labels_for_cache(source_dataset, source_cache)
    target_cache["labels"] = labels_for_cache(target_dataset, target_cache)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(source_cache, args.output_dir / "source_mode13_cache.pt")
    torch.save(target_cache, args.output_dir / "target_mode13_cache.pt")
    phase, shape, landmarks, segments, artifacts, summary = analyze(
        args, source_cache, target_cache, classes
    )
    write_csv(args.output_dir / "phase_class_metrics.csv", phase)
    write_csv(args.output_dir / "shape_class_metrics.csv", shape)
    write_csv(args.output_dir / "landmark_metrics.csv", landmarks)
    write_csv(args.output_dir / "segment_metrics.csv", segments)
    write_csv(args.output_dir / "task_summary.csv", [summary])
    torch.save(artifacts, args.output_dir / "diagnostic_artifacts.pt")
    metadata = {
        "task": args.task,
        "oracle_target_labels": True,
        "training_performed": False,
        "target_label_first_use": "after_source_and_target_mode13_cache",
        "split_strategy": "replay_existing_protocol",
        "split_replay": [source_path, target_path],
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "fold": 0,
        "seed": args.seed,
        "source_partition": "train",
        "target_partition": "train",
        "source_count": len(source_dataset),
        "target_count": len(target_dataset),
        "source_train_indices_sha256": _split_hash(source_indices),
        "target_train_indices_sha256": _split_hash(target_indices),
        "full_temporal_sequence": True,
        "augmentation": False,
        "cache_shuffle": False,
        "shift_loader_shuffle": True,
        "num_workers": 0,
        "mode": 13,
        "grid_points": 64,
        "global_shift": global_shift,
        "initial_is_shift": initial_shift,
        "global_shift_source": global_shift_source,
        "default_global_shift_semantics": (
            "TimeMatch epoch-0 shift under source checkpoint"
        ),
        "not_final_timematch_shift": True,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "classes": classes,
        "fdasrsf_version": fdasrsf.__version__,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print("ORACLE_TARGET_LABELS=true")
    print("TRAINING_PERFORMED=false")


if __name__ == "__main__":
    main()
