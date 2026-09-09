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
    estimate_scalar_phase,
    landmark_alignment_metrics,
    registration_metrics,
    robust_normalize,
    shape_margin,
    warp_curve,
    constrained_residual_phase,
    discover_phase_support,
    constrained_partial_phase,
    classify_phase_applicability,
    _crop_support,
    VisualizationUnavailable,
    fit_visualization_shared_pca,
    prepare_visualization_group,
    visualization_distance_matrices,
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
    channel_rows, lambda_rows, selections, candidate_gammas = [], [], [], {}
    artifacts["valid_channel_masks"] = {}
    artifacts["selected_lambda"] = {}
    applicability_rows = []
    for key in ("phase_applicability_by_class", "common_support_by_class", "partial_gamma_by_class",
                "partial_selected_lambda_by_class", "partial_valid_channel_masks"):
        artifacts[key] = {}

    for class_id in common:
        source_raw = source_prototypes[class_id]
        target_global = target_prototypes[class_id]
        source_norm = robust_normalize(source_raw)
        target_norm = robust_normalize(target_global)
        scalar_delta, _ = estimate_scalar_phase(source_norm, target_norm)
        target_scalar = _periodic_shift(target_global, scalar_delta, 365.0)
        loading = projections[class_id].numpy()
        result = constrained_residual_phase(
            source_raw, target_scalar, loading, grid,
            lambdas=getattr(args, "phase_lambdas", (0, .01, .1, 1, 10)),
            floor_ratio=getattr(args, "phase_iqr_floor_ratio", 1e-3),
            max_warp_days=getattr(args, "max_residual_warp_days", 60),
            prominence=args.prominence_rel * max(np.ptp(source_raw @ loading), EPS),
            min_distance_days=args.min_distance_days)
        support = discover_phase_support(
            source_raw @ loading, target_scalar @ loading, grid,
            prominence=args.prominence_rel * max(np.ptp(source_raw @ loading), EPS),
            min_distance_days=args.min_distance_days,
            edge_points=getattr(args, "phase_edge_points", 8),
            edge_monotonicity=getattr(args, "phase_edge_monotonicity", .75),
            edge_range_ratio=getattr(args, "phase_edge_range_ratio", .15),
            full_landmark_coverage=getattr(args, "full_landmark_coverage", .8),
            partial_min_landmarks=getattr(args, "partial_min_landmarks", 2),
            partial_min_time_coverage=getattr(args, "partial_min_time_coverage", .2))
        partial = constrained_partial_phase(
            source_raw, target_scalar, grid, support,
            lambdas=getattr(args, "phase_lambdas", (0, .01, .1, 1, 10)),
            floor_ratio=getattr(args, "phase_iqr_floor_ratio", 1e-3),
            max_warp_days=getattr(args, "max_residual_warp_days", 60))
        applicability = classify_phase_applicability(support, result, partial)
        state = applicability["phase_applicability"]
        support_scalars = {k: v for k, v in support.items() if k not in (
            "source_landmarks", "target_landmarks", "matched_pairs", "common_chain")}
        applicability_rows.append(dict(task=args.task, class_index=class_id, class_name=classes[class_id],
            **support_scalars, **applicability, full_selected_phase=result["selected_phase"],
            full_landmark_error=result["selected_landmark_error"],
            partial_selected_phase=partial["selected_phase"],
            partial_landmark_error_before=partial["scalar_landmark_error"],
            partial_landmark_error_after=partial["selected_landmark_error"],
            partial_landmark_gain=partial["scalar_landmark_error"]-partial["selected_landmark_error"],
            partial_nonlinear_accepted=partial["nonlinear_accepted"],
            partial_solution_valid=partial["solution_valid"], partial_failure_reason=partial["failure_reason"]))
        artifacts["phase_applicability_by_class"][class_id] = state
        artifacts["common_support_by_class"][class_id] = dict(support_scalars,
            chain=[((a.kind, a.time), (b.kind, b.time)) for a, b in support["common_chain"]])
        if partial["solution_valid"]:
            artifacts["partial_gamma_by_class"][class_id] = partial["phase"].gamma
            artifacts["partial_selected_lambda_by_class"][class_id] = partial["selected_lambda"]
        if "normalized" in partial:
            artifacts["partial_valid_channel_masks"][class_id] = partial["normalized"]["valid_channels"]
        nonlinear = result["phase"]
        target_nonlinear = result["raw_aligned"]
        norm = result["normalized"]
        selected_info = {key: result[key] for key in (
            "selected_phase", "selected_lambda", "nonlinear_accepted")}
        selection = dict(selected_info, scalar_delta_days=scalar_delta,
                         scalar_landmark_error=result["scalar_landmark_error"],
                         selected_landmark_error=result["selected_landmark_error"],
                         landmark_gain=result["scalar_landmark_error"]-result["selected_landmark_error"],
                         valid_phase_channel_count=int(norm["valid_channels"].sum()),
                         total_channel_count=len(norm["valid_channels"]),
                         normalized_global_max_abs=norm["normalized_global_max_abs"],
                         residual_gamma_mean_days=nonlinear.mean_displacement*365,
                         residual_gamma_max_days=nonlinear.max_displacement*365,
                         residual_gamma_p95_days=nonlinear.p95_displacement*365,
                         selection_failure_reason=result["failure_reason"])
        selections.append(selection)
        candidate_gammas[class_id] = result["candidate_gammas"]
        for row in result["candidates"]:
            lambda_rows.append(dict(task=args.task, registration_scope="FULL_CONSTRAINED", **{"class": classes[class_id]},
                                    **row, **{"lambda": row["lambda_value"]}))
        for row in partial["candidates"]:
            lambda_rows.append(dict(task=args.task, registration_scope="PARTIAL_CONSTRAINED",
                                    **{"class": classes[class_id]}, **row, **{"lambda": row["lambda_value"]}))
        for channel in range(len(norm["valid_channels"])):
            channel_rows.append(dict(task=args.task, registration_scope="FULL_CONSTRAINED", **{"class": classes[class_id]}, channel=channel,
                source_iqr=norm["source_iqr"][channel], target_iqr=norm["target_iqr"][channel],
                iqr_floor=norm["iqr_floor"], phase_channel_valid=bool(norm["valid_channels"][channel]),
                source_norm_max_abs=float(np.max(np.abs(norm["source"][:, channel]))),
                target_norm_max_abs=float(np.max(np.abs(norm["target"][:, channel]))),
                source_norm_std=float(np.std(norm["source"][:, channel])),
                target_norm_std=float(np.std(norm["target"][:, channel]))))
        if "normalized" in partial:
            pn = partial["normalized"]
            for channel in range(len(pn["valid_channels"])):
                channel_rows.append(dict(task=args.task, registration_scope="PARTIAL_CONSTRAINED",
                    **{"class": classes[class_id]}, channel=channel, source_iqr=pn["source_iqr"][channel],
                    target_iqr=pn["target_iqr"][channel], iqr_floor=pn["iqr_floor"],
                    phase_channel_valid=bool(pn["valid_channels"][channel]),
                    source_norm_max_abs=float(np.max(np.abs(pn["source"][:, channel]))),
                    target_norm_max_abs=float(np.max(np.abs(pn["target"][:, channel])))))
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
        # Keep the scalar-frame validity mask and scales fixed across all phases.
        # These are also the exact tensors used for candidate registration audits.
        global_phase_norm = np.zeros_like(target_global, dtype=np.float64)
        valid_channels = norm["valid_channels"]
        global_phase_norm[:, valid_channels] = (
            target_global[:, valid_channels]
            - np.median(target_scalar[:, valid_channels], axis=0)
        ) / np.maximum(norm["target_iqr"][valid_channels], norm["iqr_floor"])
        phase_representations = {
            "global": global_phase_norm,
            "scalar": norm["target"],
            "nonlinear": warp_curve(norm["target"], nonlinear.gamma)
            if result["nonlinear_accepted"] else norm["target"],
        }
        for phase_type, target_curve in curves.items():
            registrations[phase_type] = registration_metrics(
                norm["source"], phase_representations[phase_type]
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
        selection.update(scalar_registration_error=scalar_error,
                         selected_registration_error=nonlinear_error)
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
                    "registration_scope": "FULL_CONSTRAINED",
                    **applicability,
                    **selection,
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
                    "landmark_metric_valid": bool(landmarks["matched_pairs"])
                    and np.isfinite(landmarks["mean_time_error"]),
                    "phase_failure_reason": reason,
                    "failure_reason": reason,
                }
            )

        # Retain full reference artifacts even when structural applicability rejects full.
        artifacts["source_raw_prototypes"][class_id] = source_raw
        artifacts["target_raw_prototypes"][class_id] = target_global
        artifacts["source_norm_prototypes"][class_id] = norm["source"]
        artifacts["target_norm_prototypes"][class_id] = norm["target"]
        artifacts["scalar_delta_by_class"][class_id] = scalar_delta
        artifacts["gamma_by_class"][class_id] = nonlinear.gamma
        artifacts["source_pca_direction_by_class"][class_id] = loading
        artifacts["valid_channel_masks"][class_id] = norm["valid_channels"]
        artifacts["selected_lambda"][class_id] = result["selected_lambda"]
        summaries.append((registrations, landmark_metrics))
        plot_class(args.output_dir / f"class_{class_id}_phase_shape.png", grid, source_curve,
                   curves, support, partial, loading, applicability, classes[class_id])
        if state != "FULL_PHASE":
            shape, marks, segments = support_shape_metrics(
                args.task, class_id, classes[class_id], source_raw, target_scalar,
                grid, loading, support, partial, applicability)
            shape_rows.append(shape)
            landmark_rows.extend(marks)
            segment_rows.extend(segments)
            continue

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
                "source_start_day": source_a.time,
                "source_end_day": source_b.time,
                "target_start_day": target_a.time,
                "target_end_day": target_b.time,
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

        margin = shape_margin(class_id, target_nonlinear, source_prototypes) if len(source_prototypes) > 1 else dict(
            same_class_shape_distance=np.nan, nearest_wrong_class=np.nan, nearest_wrong_distance=np.nan,
            shape_margin=np.nan, margin_positive=np.nan)
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
        shape_rows[-1].update(**applicability, shape_phase_conditioned_valid=True,
            full_shape_metrics_valid=True, partial_shape_metrics_valid=False,
            shape_metric_scope="full_year",
            observable_landmark_count=len(nonlinear_pairs), observable_segment_count=len(class_segments),
            observable_time_coverage=1.)
        for rows in (shape_rows, landmark_rows, segment_rows):
            for row in rows:
                if row.get("class", row.get("class_name")) == classes[class_id]:
                    row.update(selected_info)
                    if rows is not shape_rows:
                        row.update(**applicability, in_common_support=True, observable_in_both_domains=True)
        # Full coverage may be >=0.8 yet have unmatched boundary structure. Report it as NA too.
        missing_support = dict(support, source_landmarks=landmark_metrics["nonlinear"]["source_landmarks"],
                               target_landmarks=landmark_metrics["nonlinear"]["target_landmarks"])
        _, all_marks, all_segments = support_shape_metrics(args.task, class_id, classes[class_id],
            source_raw, target_scalar, grid, loading, missing_support, {},
            dict(phase_applicability="PHASE_NOT_APPLICABLE", phase_applicability_reason="unmatched_structure"))
        matched_source = {(a.kind, a.time) for a, _ in nonlinear_pairs}
        matched_target = {(b.kind, b.time) for _, b in nonlinear_pairs}
        for row in all_marks:
            key = (row["landmark_type"], row["source_time"])
            matched = key in matched_source if np.isfinite(row["source_time"]) else (
                row["landmark_type"], row["target_time_scalar"]) in matched_target
            if not matched:
                row.update(**applicability, **selected_info)
                landmark_rows.append(row)
        for row in all_segments:
            domain = row["segment_domain"]
            represented = any(r.get(domain + "_start_day") == row.get(domain + "_start_day")
                              and r.get(domain + "_end_day") == row.get(domain + "_end_day")
                              for r in class_segments)
            if not represented:
                row.update(**applicability, **selected_info)
                segment_rows.append(row)

    plot_gammas(args.output_dir / "gamma_by_class.png", artifacts, classes)
    summary = task_summary(args.task, summaries, shape_rows, segment_rows)
    write_csv(args.output_dir / "phase_applicability_metrics.csv", applicability_rows)
    for state, prefix in (("FULL_PHASE", "full_phase"), ("PARTIAL_PHASE", "partial_phase"),
                          ("PHASE_NOT_APPLICABLE", "phase_not_applicable")):
        count = sum(r["phase_applicability"] == state for r in applicability_rows)
        summary[prefix + "_count"] = count
        summary[prefix + "_rate"] = count/max(1, len(applicability_rows))
    truncation_count = sum(any(r[s + "_truncation_evidence"] == "strong" for s in ("left", "right"))
                           for r in applicability_rows)
    coverages = [r["common_time_coverage_min"] for r in applicability_rows]
    summary.update(boundary_truncation_class_count=truncation_count,
        boundary_truncation_class_rate=truncation_count/max(1, len(applicability_rows)),
        common_support_coverage_mean=_finite_mean(coverages),
        common_support_coverage_median=float(np.median(coverages)) if coverages else np.nan,
        partial_landmark_gain_mean=_finite_mean(r["partial_landmark_gain"] for r in applicability_rows
                                               if r["phase_applicability"] == "PARTIAL_PHASE"))
    write_csv(args.output_dir / "phase_channel_metrics.csv", channel_rows)
    write_csv(args.output_dir / "phase_lambda_metrics.csv", lambda_rows)
    write_csv(args.output_dir / "phase_lambda_summary.csv", summarize_lambdas(args.task, lambda_rows))
    plot_candidate_gammas(args.output_dir / "gamma_candidates_by_class.png", candidate_gammas, classes)
    accepted = [s for s in selections if s["nonlinear_accepted"]]
    accepted_gammas = [artifacts["gamma_by_class"][c] for c in common
                       if artifacts["selected_lambda"][c] is not None]
    summary.update(
        selected_phase_landmark_error_mean=_finite_mean(s["selected_landmark_error"] for s in selections),
        landmark_improvement_mean=_finite_mean(s["landmark_gain"] for s in selections),
        landmark_improvement_rate=_finite_mean(s["landmark_gain"] > 0 for s in selections),
        nonlinear_accepted_count=len(accepted), nonlinear_accepted_rate=len(accepted)/max(1, len(selections)),
        accepted_nonlinear_class_count=len(accepted), accepted_nonlinear_class_rate=len(accepted)/max(1, len(selections)),
        residual_mean_days_mean=_finite_mean(s["residual_gamma_mean_days"] for s in accepted),
        residual_max_days_mean=_finite_mean(s["residual_gamma_max_days"] for s in accepted),
        residual_mean_displacement_days_mean=_finite_mean(s["residual_gamma_mean_days"] for s in accepted),
        residual_mean_displacement_days_std=float(np.std([s["residual_gamma_mean_days"] for s in accepted])) if accepted else np.nan,
        residual_max_displacement_days_mean=_finite_mean(s["residual_gamma_max_days"] for s in accepted),
        cross_class_gamma_dispersion=float(np.var(accepted_gammas, axis=0).mean()) if accepted_gammas else np.nan,
        cross_class_gamma_mean=json.dumps(np.mean(accepted_gammas, axis=0).tolist()) if accepted_gammas else "[]",
        cross_class_gamma_variance=json.dumps(np.var(accepted_gammas, axis=0).tolist()) if accepted_gammas else "[]",
        valid_phase_channel_rate=sum(s["valid_phase_channel_count"] for s in selections)/max(1, sum(s["total_channel_count"] for s in selections)),
        numeric_failure_count=sum(not r["gamma_valid"] for r in lambda_rows),
        overwarp_rejection_count=sum(r["rejection_reason"] == "residual_warp_too_large" for r in lambda_rows),
        no_landmark_rejection_count=sum(r["rejection_reason"] == "no_matched_landmarks" for r in lambda_rows),
        no_improvement_rejection_count=sum(s["selection_failure_reason"] == "no_landmark_improvement" for s in selections))
    return phase_rows, shape_rows, landmark_rows, segment_rows, artifacts, summary


def support_shape_metrics(task, class_id, name, source, target_scalar, grid, loading,
                          support, partial, applicability):
    """No whole-year shape calculation on a partial or rejected correspondence."""
    valid = applicability["phase_applicability"] == "PARTIAL_PHASE"
    info = dict(**applicability, selected_phase=partial["selected_phase"] if valid else "none",
                selected_lambda=partial["selected_lambda"] if valid else None,
                nonlinear_accepted=bool(valid and partial["nonlinear_accepted"]))
    shape = dict(task=task, class_index=class_id, class_name=name, **info,
                 shape_metric_scope="common_support" if valid else "none",
                 shape_phase_conditioned_valid=valid, full_shape_metrics_valid=False,
                 partial_shape_metrics_valid=valid, observable_landmark_count=0,
                 observable_segment_count=0,
                 observable_time_coverage=support["common_time_coverage_min"] if valid else 0.)
    for key in ("same_class_shape_distance", "nearest_wrong_class", "nearest_wrong_distance", "shape_margin",
                "margin_positive", "peak_to_valley_source", "peak_to_valley_target", "peak_to_valley_ratio",
                "peak_height_diff_mean", "peak_height_ratio_mean", "valley_height_diff_mean", "valley_height_ratio_mean",
                "prominence_diff_mean", "prominence_ratio_mean", "peak_valley_amplitude_diff_mean",
                "peak_valley_amplitude_ratio_mean", "segment_auc_ratio_mean", "segment_range_ratio_mean", "segment_l2_mean"):
        shape[key] = np.nan
    for metric in ("range", "std", "iqr"):
        for suffix in ("source", "target", "ratio"):
            shape["global_" + metric + "_" + suffix] = np.nan
    pairs = {(a.kind, a.time): b for a, b in support["common_chain"]} if valid else {}
    if valid:
        source_curve = partial["raw_source"] @ loading
        target_curve = partial["raw_aligned"] @ loading
        local_grid = partial["source_grid"]
        amplitude = amplitude_metrics(source_curve, target_curve)
        shape.update({"global_" + key: value for key, value in amplitude.items()})
        for domain, values in (("source", source), ("target", target_scalar)):
            times, cropped = _crop_support(values, grid, support[domain + "_common_start_day"],
                                           support[domain + "_common_end_day"])
            # Prominence bases must be within the support, never the full-year detector's bases.
            if domain == "source":
                source_times, source_local = times, cropped @ loading
            else:
                target_times, target_local = times, cropped @ loading

    def local_prominence(mark, times, curve):
        index = int(np.argmin(np.abs(times-mark.time)))
        if index == 0 or index == len(times)-1:
            return np.nan  # Endpoint extrema have no two-sided observable prominence.
        signed = curve if mark.kind == "peak" else -curve
        from scipy.signal import peak_prominences
        return float(peak_prominences(signed, [index])[0][0])

    landmarks = []
    for index, mark in enumerate(support["source_landmarks"]):
        mate = pairs.get((mark.kind, mark.time))
        observable = mate is not None
        row = dict(task=task, **{"class": name}, **info, landmark_index=index, landmark_type=mark.kind,
                   source_time=mark.time, source_value=mark.amplitude,
                   in_common_support=observable, observable_in_both_domains=observable)
        for key in ("target_time_global", "target_time_scalar", "target_time_nonlinear", "global_time_error",
                    "scalar_time_error", "nonlinear_time_error", "target_value_aligned", "height_difference",
                    "height_absolute_difference", "height_relative_difference", "height_ratio",
                    "source_prominence", "target_prominence", "prominence_difference", "prominence_ratio"):
            row[key] = np.nan
        if observable:
            u = (mark.time-local_grid[0])/(local_grid[-1]-local_grid[0])
            mapped = float(np.interp(u, np.linspace(0, 1, 128), partial["mapped_target_grid"]))
            sp = local_prominence(mark, source_times, source_local)
            tp = local_prominence(mate, target_times, target_local)
            difference = mate.amplitude-mark.amplitude
            row.update(target_time_scalar=mate.time, mapped_target_time=mapped,
                       nonlinear_time_error=abs(mapped-mate.time),
                       target_value_aligned=mate.amplitude, height_difference=difference,
                       height_absolute_difference=abs(difference),
                       height_relative_difference=difference/max(abs(mark.amplitude), EPS),
                       height_ratio=_ratio(mate.amplitude, mark.amplitude), source_prominence=sp,
                       target_prominence=tp, prominence_difference=tp-sp,
                       prominence_ratio=_ratio(tp, sp), prominence_observable=bool(np.isfinite(sp) and np.isfinite(tp)))
        landmarks.append(row)
    # Target-only landmarks are also explicitly unobservable, rather than disappearing.
    matched_targets = {(b.kind, b.time) for b in pairs.values()}
    for mark in support["target_landmarks"]:
        if (mark.kind, mark.time) not in matched_targets:
            landmarks.append(dict(task=task, **{"class": name}, **info, landmark_type=mark.kind,
                source_time=np.nan, target_time_scalar=mark.time, in_common_support=False,
                observable_in_both_domains=False, height_ratio=np.nan, prominence_ratio=np.nan))
    segments = []
    for index, (left, right) in enumerate(zip(support["source_landmarks"], support["source_landmarks"][1:])):
        ta, tb = pairs.get((left.kind, left.time)), pairs.get((right.kind, right.time))
        observable = ta is not None and tb is not None
        row = dict(task=task, **{"class": name}, **info, segment_index=index,
                   segment_domain="source",
                   start_landmark_type=left.kind, end_landmark_type=right.kind,
                   source_start_day=left.time, source_end_day=right.time,
                   in_common_support=observable, observable_in_both_domains=observable)
        for key in ("source_length_days", "target_length_days", "source_auc", "target_auc", "auc_ratio",
                    "source_range", "target_range", "range_ratio", "source_std", "target_std", "std_ratio",
                    "source_mean", "target_mean", "segment_l1", "segment_l2", "segment_corr"):
            row[key] = np.nan
        if observable:
            # Slice BOTH selected registered curves on the same source-time interval.
            # Do not add another segment-level time warp.
            times, s = _crop_support(source_curve[:, None], local_grid, left.time, right.time)
            _, t = _crop_support(target_curve[:, None], local_grid, left.time, right.time)
            s, t = s[:, 0], t[:, 0]
            sa, tt = float(np.trapz(s, times)), float(np.trapz(t, times))
            row.update(source_length_days=right.time-left.time, target_length_days=tb.time-ta.time,
                       source_auc=sa, target_auc=tt, auc_ratio=_ratio(abs(tt), abs(sa)),
                       source_range=float(np.ptp(s)), target_range=float(np.ptp(t)),
                       range_ratio=_ratio(np.ptp(t), np.ptp(s)), source_std=float(np.std(s)),
                       target_std=float(np.std(t)), std_ratio=_ratio(np.std(t), np.std(s)),
                       source_mean=float(np.mean(s)), target_mean=float(np.mean(t)),
                       segment_l1=float(np.mean(np.abs(t-s))), segment_l2=float(np.sqrt(np.mean((t-s)**2))),
                       segment_corr=float(np.corrcoef(s, t)[0, 1]) if np.std(s)*np.std(t) > EPS else 0.)
            row.update(target_start_day=ta.time, target_end_day=tb.time)
        segments.append(row)
    represented_target_edges = {(r.get("target_start_day"), r.get("target_end_day")) for r in segments
                                if r["observable_in_both_domains"]}
    for left, right in zip(support["target_landmarks"], support["target_landmarks"][1:]):
        if (left.time, right.time) not in represented_target_edges:
            segments.append(dict(task=task, **{"class": name}, **info, segment_domain="target",
                start_landmark_type=left.kind, end_landmark_type=right.kind,
                target_start_day=left.time, target_end_day=right.time,
                source_start_day=np.nan, source_end_day=np.nan,
                in_common_support=False, observable_in_both_domains=False,
                auc_ratio=np.nan, range_ratio=np.nan, std_ratio=np.nan, segment_l1=np.nan,
                segment_l2=np.nan, segment_corr=np.nan))
    observed = [r for r in landmarks if r["observable_in_both_domains"]]
    observed_segments = [r for r in segments if r["observable_in_both_domains"]]
    shape.update(observable_landmark_count=len(observed), observable_segment_count=len(observed_segments))
    if valid:
        for kind, label in (("peak", "peak"), ("valley", "valley")):
            shape[label + "_height_diff_mean"] = _finite_mean(r["height_difference"] for r in observed if r["landmark_type"] == kind)
            shape[label + "_height_ratio_mean"] = _finite_mean(r["height_ratio"] for r in observed if r["landmark_type"] == kind)
        shape["prominence_diff_mean"] = _finite_mean(r["prominence_difference"] for r in observed)
        shape["prominence_ratio_mean"] = _finite_mean(r["prominence_ratio"] for r in observed)
        for label, key in (("segment_auc_ratio_mean", "auc_ratio"), ("segment_range_ratio_mean", "range_ratio"),
                           ("segment_l2_mean", "segment_l2")):
            shape[label] = _finite_mean(r[key] for r in observed_segments)
        differences, ratios = [], []
        for (a, b), (c, d) in zip(support["common_chain"], support["common_chain"][1:]):
            if a.kind != c.kind:
                s, t = abs(c.amplitude-a.amplitude), abs(d.amplitude-b.amplitude)
                differences.append(t-s); ratios.append(_ratio(t, s))
        shape["peak_valley_amplitude_diff_mean"] = _finite_mean(differences)
        shape["peak_valley_amplitude_ratio_mean"] = _finite_mean(ratios)
    return shape, landmarks, segments


def summarize_lambdas(task, rows):
    result = []
    for scope, lam in sorted({(r["registration_scope"], r["lambda_value"]) for r in rows}):
        group = [r for r in rows if r["lambda_value"] == lam and r["registration_scope"] == scope]
        result.append(dict(task=task, registration_scope=scope, **{"lambda": lam},
            valid_rate=_finite_mean(r["gamma_valid"] for r in group),
            admissible_rate=_finite_mean(r["candidate_admissible"] for r in group),
            registration_error_mean=_finite_mean(r["registration_error"] for r in group),
            landmark_error_mean=_finite_mean(r["landmark_error_mean"] for r in group),
            mean_residual_displacement_days=_finite_mean(r["gamma_mean_displacement_days"] for r in group),
            max_residual_displacement_days_mean=_finite_mean(r["gamma_max_displacement_days"] for r in group),
            overwarp_rate=_finite_mean(r["rejection_reason"] == "residual_warp_too_large" for r in group)))
    return result


def plot_candidate_gammas(path, candidates, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(max(1, len(candidates)), 1, figsize=(8, max(3, 3*len(candidates))), squeeze=False)
    time = np.linspace(0, 1, 128)
    for ax, (class_id, gammas) in zip(axes[:, 0], candidates.items()):
        ax.plot(time, time, "k--", label="identity")
        for lam, gamma in gammas.items():
            ax.plot(time, gamma, label=f'lambda={lam}' + (' reference' if lam == 0 else ''))
        ax.set_title(classes[class_id]); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def task_summary(task, summaries, shape_rows, segment_rows):
    partial_shapes = [r for r in shape_rows if r["phase_applicability"] == "PARTIAL_PHASE"]
    partial_segments = [r for r in segment_rows if r["phase_applicability"] == "PARTIAL_PHASE"
                        and r["observable_in_both_domains"]]
    shape_rows = [r for r in shape_rows if r["phase_applicability"] == "FULL_PHASE"]
    segment_rows = [r for r in segment_rows if r["phase_applicability"] == "FULL_PHASE"
                    and r["observable_in_both_domains"]]
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
        "legacy_shape_summary_scope": "FULL_PHASE_only",
        "partial_amplitude_ratio_mean": _finite_mean(r["global_range_ratio"] for r in partial_shapes),
        "partial_prominence_ratio_mean": _finite_mean(r["prominence_ratio_mean"] for r in partial_shapes),
        "partial_segment_auc_ratio_mean": _finite_mean(r["auc_ratio"] for r in partial_segments),
        "partial_segment_l2_mean": _finite_mean(r["segment_l2"] for r in partial_segments),
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
            np.nanstd([row["global_range_ratio"] for row in shape_rows]) if shape_rows else np.nan
        ),
        "prominence_ratio_mean": _finite_mean(
            row["prominence_ratio_mean"] for row in shape_rows
        ),
        "prominence_ratio_std": float(
            np.nanstd([row["prominence_ratio_mean"] for row in shape_rows]) if shape_rows else np.nan
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


def plot_class(path, grid, source, curves, support, partial, loading, applicability, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    for axis in axes[:3]:
        axis.plot(grid, source, label="source")
        axis.plot(grid, curves["scalar"], label="target scalar")
        axis.legend()
    axes[0].set_title("Global + Scalar frame")
    axes[1].set_title("Salient landmarks and contiguous correspondence chain")
    for domain, color in (("source", "C0"), ("target", "C1")):
        for mark in support[domain + "_landmarks"]:
            axes[1].scatter(mark.time, mark.amplitude, color=color, marker="^" if mark.kind == "peak" else "v")
    for a, b in support["common_chain"]:
        axes[1].plot([a.time, b.time], [a.amplitude, b.amplitude], "k--", alpha=.5)
    axes[2].set_title("Common support (no padding)")
    for domain, color in (("source", "C0"), ("target", "C1")):
        a, b = support[domain + "_common_start_day"], support[domain + "_common_end_day"]
        if np.isfinite(a) and b > a:
            axes[2].axvspan(a, b, color=color, alpha=.15, label=domain + " support")
    for side, x in (("left", grid[0]), ("right", grid[-1])):
        if any(support[side + "_boundary_active_" + d] for d in ("source", "target")):
            axes[0].text(x, axes[0].get_ylim()[1], side + " truncation candidate\n" +
                         support[side + "_truncation_evidence"], va="top",
                         ha="left" if side == "left" else "right", fontsize=8)
    state = applicability["phase_applicability"]
    axes[3].set_title("Selected registration" if state != "PHASE_NOT_APPLICABLE" else "No phase-conditioned comparison")
    if state == "FULL_PHASE":
        axes[3].plot(grid, source, label="source")
        axes[3].plot(grid, curves["nonlinear"], label="target full selected")
    elif state == "PARTIAL_PHASE":
        sg = partial["source_grid"]
        axes[3].plot(sg, partial["raw_source"] @ loading, label="source common support")
        axes[3].plot(sg, partial["raw_aligned"] @ loading, label="target partial selected")
        axes[3].axvspan(grid[0], sg[0], color="grey", alpha=.3)
        axes[3].axvspan(sg[-1], grid[-1], color="grey", alpha=.3)
    if state != "PHASE_NOT_APPLICABLE":
        axes[3].legend()
    figure.suptitle(name + " | " + state + "\n" + applicability["phase_applicability_reason"], fontsize=9)
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
        if artifacts["selected_lambda"].get(class_id) is None:
            continue
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
    parser.add_argument("--phase-iqr-floor-ratio", type=float, default=1e-3)
    parser.add_argument("--phase-lambdas", type=lambda value: tuple(float(x) for x in value.split(',')), default=(0, .01, .1, 1, 10))
    parser.add_argument("--max-residual-warp-days", type=float, default=60)
    parser.add_argument("--save-feature-cache", action="store_true")
    parser.add_argument("--phase-edge-points", type=int, default=8)
    parser.add_argument("--phase-edge-monotonicity", type=float, default=.75)
    parser.add_argument("--phase-edge-range-ratio", type=float, default=.15)
    parser.add_argument("--full-landmark-coverage", type=float, default=.80)
    parser.add_argument("--partial-min-landmarks", type=int, default=2)
    parser.add_argument("--partial-min-time-coverage", type=float, default=.20)
    parser.add_argument("--viz-max-curves-per-group", type=int, default=40)
    parser.add_argument("--viz-dpi", type=int, default=160)
    return parser.parse_args()


def main():
    if "--aggregate-phase-applicability-root" in sys.argv:
        parser = argparse.ArgumentParser(description="Aggregate completed phase applicability visualizations")
        parser.add_argument("--aggregate-phase-applicability-root", type=Path, required=True)
        parser.add_argument("--viz-dpi", type=int, default=160)
        aggregate_args = parser.parse_args()
        generate_global_phase_applicability_figure(
            aggregate_args.aggregate_phase_applicability_root, dpi=aggregate_args.viz_dpi)
        return
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
    save_feature_caches(args.output_dir, source_cache, target_cache, args.save_feature_cache)
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
        "oracle_phase_selection": True,
        "nonlinear_semantics": "scalar_plus_constrained_residual_nonlinear",
        "phase_iqr_floor_ratio": args.phase_iqr_floor_ratio,
        "phase_lambdas": args.phase_lambdas,
        "max_residual_warp_days": args.max_residual_warp_days,
        "save_feature_cache": args.save_feature_cache,
        "normalization_audit_frame": "source_and_scalar_aligned_target",
        "full_registration_scope": "FULL_CONSTRAINED; unchanged whole-year normalization and solver",
        "full_support_assumption": "major structure completely observable within both annual windows",
        "partial_normalization": "crop first; native interval IQR and validity; normalize then resample to 128",
        "partial_mapping": "target_day = a_t + gamma(u)*(b_t-a_t)",
        "partial_residual_days": "(gamma(u)-u)*(b_t-a_t)",
        "partial_landmark_error": "mean abs(mapped source chain landmark - target chain landmark), target days",
        "partial_shape_scope": "common support only; endpoint prominence NA; no extrapolation",
        "partial_auc_units": "raw PCA amplitude * source-frame days; full legacy AUC unchanged",
        "partial_support_padding_days": 0,
        "contiguous_chain_rule": "both full landmark indices increment by 1; longest; earliest tie",
        "truncation_rule": "active edge + compatible extremum flank + unmatched boundary extremum in other domain",
        "phase_applicability_order": ["FULL_PHASE", "PARTIAL_PHASE", "PHASE_NOT_APPLICABLE"],
        "phase_edge_points": args.phase_edge_points,
        "phase_edge_monotonicity": args.phase_edge_monotonicity,
        "phase_edge_range_ratio": args.phase_edge_range_ratio,
        "full_landmark_coverage": args.full_landmark_coverage,
        "partial_min_landmarks": args.partial_min_landmarks,
        "partial_min_time_coverage": args.partial_min_time_coverage,
        "invalid_phase_channels": "excluded_from_SRVF_and_zero_in_normalization_audit",
        "phase_selection_rule": "landmark_mean_then_mean_displacement_then_larger_lambda; strictly_improve_scalar",
        "numeric_failure_count_unit": "class_lambda_candidates",
        "overwarp_rejection_count_unit": "class_lambda_candidates_including_reference",
        "no_landmark_rejection_count_unit": "class_lambda_candidates",
        "no_improvement_rejection_count_unit": "classes",
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
    # All existing numerical CSV/JSON/artifacts are complete before plotting begins.
    with (args.output_dir / "phase_applicability_metrics.csv").open(encoding="utf-8", newline="") as stream:
        visualization_states = list(csv.DictReader(stream))
    generate_diagnostic_visualizations(args.output_dir, args.task, classes, source_cache, target_cache,
                                       artifacts, visualization_states,
                                       max_curves=args.viz_max_curves_per_group, dpi=args.viz_dpi)


def save_feature_caches(directory, source, target, enabled=False):
    if enabled:
        torch.save(source, directory / "source_mode13_cache.pt")
        torch.save(target, directory / "target_mode13_cache.pt")


def _visualization_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _visualization_ylim(bundles):
    low, high = 0., 0.
    for bundle in bundles:
        for key in ("samples", "prototype", "quantiles"):
            values = np.asarray(bundle[key])
            finite = values[np.isfinite(values)]
            if finite.size:
                low, high = min(low, float(finite.min())), max(high, float(finite.max()))
        if "bounds" in bundle:
            low, high = min(low, bundle["bounds"][0]), max(high, bundle["bounds"][1])
    padding = .05 * max(high-low, 1e-6)
    return low-padding, high+padding


def _plot_visualization_bundle(axis, grid, bundle, color, label, samples=True):
    if samples:
        axis.plot(grid, bundle["samples"].T, color=color, alpha=.12, linewidth=.6)
    axis.fill_between(grid, bundle["quantiles"][0], bundle["quantiles"][2], color=color, alpha=.16)
    axis.plot(grid, bundle["prototype"], color=color, linewidth=2.2, label=label + " prototype")
    if samples:
        axis.plot(grid, bundle["quantiles"][1], color=color, linestyle="--", linewidth=1.2,
                  label=label + " sample median")


def _shade_visualization_support(axis, data):
    if data["state"] == "PARTIAL_PHASE":
        axis.axvspan(0, data["support"]["source_common_start_day"], color="grey", alpha=.2)
        axis.axvspan(data["support"]["source_common_end_day"], 365, color="grey", alpha=.2)


def plot_visualization_spaghetti(data, task, name, status):
    plt = _visualization_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    grid, state = data["grid"], data["state"]
    for axis, source_key, target_key, title in (
            (axes[0], "source", "target_global", "A: Global"),
            (axes[1], "source", "target_scalar", "B: Global + Scalar"),
            (axes[2], "source_selected", "target_selected", "C: Selected phase")):
        _plot_visualization_bundle(axis, grid, data[source_key], "C0", "source")
        _plot_visualization_bundle(axis, grid, data[target_key], "C1", "target")
        axis.set_title(title)
        axis.legend(fontsize=7, loc="best")
    if state == "PHASE_NOT_APPLICABLE":
        axes[2].set_title("C: PHASE NOT APPLICABLE (Scalar reference)")
        axes[3].set_title("D: Aligned residual unavailable")
        axes[3].text(.5, .5, "No phase-conditioned residual", transform=axes[3].transAxes, ha="center")
    else:
        _plot_visualization_bundle(axes[3], grid, data["residual"], "C3", "target - source prototype")
        axes[3].axhline(0, color="black", linewidth=.7)
        axes[3].set_title("D: Aligned shape residual (raw amplitude)")
        axes[3].legend(fontsize=7)
    _shade_visualization_support(axes[2], data)
    _shade_visualization_support(axes[3], data)
    ylim = _visualization_ylim(data[k] for k in (
        "source", "target_global", "target_scalar", "source_selected", "target_selected", "residual"))
    for axis in axes:
        axis.set(xlim=(0,365), ylim=ylim, xlabel="Day of Year", ylabel="Raw Mode13 projection")
    if state == "PARTIAL_PHASE":
        s = data["support"]
        detail = (f'source support=[{s["source_common_start_day"]:.1f}, {s["source_common_end_day"]:.1f}], '
                  f'target support=[{s["target_common_start_day"]:.1f}, {s["target_common_end_day"]:.1f}]\n'
                  f'coverage={s.get("common_time_coverage_min", np.nan):.3f}, '
                  f'lambda={status.get("partial_selected_lambda", data.get("selected_lambda"))}, '
                  f'landmark {status.get("partial_landmark_error_before", "NA")} -> '
                  f'{status.get("partial_landmark_error_after", "NA")} days')
    elif state == "FULL_PHASE":
        detail = f'selected={status.get("full_selected_phase", "scalar")}, lambda={data.get("selected_lambda")}'
    else:
        detail = status.get("phase_applicability_reason", "no admissible phase correspondence")
    figure.suptitle(f"{task} | {name}\n{state}\n{detail}", fontsize=10)
    figure.tight_layout(rect=(0,0,1,.88))
    return figure


def _plot_visualization_gallery(groups, classes, key, title, ylim):
    plt = _visualization_pyplot()
    columns = min(3, len(groups))
    figure, axes = plt.subplots(int(np.ceil(len(groups)/columns)), columns,
                               figsize=(5*columns, 3.2*np.ceil(len(groups)/columns)),
                               sharex=True, sharey=True, squeeze=False)
    for axis, (class_id, data) in zip(axes.ravel(), groups.items()):
        _plot_visualization_bundle(axis, data["grid"], data[key], "C0" if key == "source" else "C1", "class")
        label = classes[class_id]
        if key == "target_selected":
            label += "\n" + data["state"]
            if data["state"] == "PHASE_NOT_APPLICABLE":
                label += " (Scalar reference)"
            _shade_visualization_support(axis, data)
        axis.set(title=label, xlim=(0,365), ylim=ylim, xlabel="Day of Year")
    for axis in axes.ravel()[len(groups):]:
        axis.set_visible(False)
    figure.suptitle(title + "\nShared source-only PCA axis; solid=prototype, dashed=sample median", fontsize=10)
    figure.tight_layout(rect=(0,0,1,.91))
    return figure


def _plot_visualization_overlay(groups, classes, key, title, ylim):
    plt = _visualization_pyplot()
    figure, axis = plt.subplots(figsize=(10,5))
    for class_id, data in groups.items():
        axis.plot(data["grid"], data[key]["prototype"], label=classes[class_id])
    axis.set(xlim=(0,365), ylim=ylim, xlabel="Day of Year", ylabel="Raw shared PCA projection", title=title)
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def _plot_visualization_residuals(groups, classes, task):
    plt = _visualization_pyplot()
    columns = min(3, len(groups))
    figure, axes = plt.subplots(int(np.ceil(len(groups)/columns)), columns,
        figsize=(5*columns, 3.2*np.ceil(len(groups)/columns)), sharex=True, sharey=False, squeeze=False)
    for axis, (class_id, data) in zip(axes.ravel(), groups.items()):
        if data["state"] == "PHASE_NOT_APPLICABLE":
            axis.text(.5,.5,"Residual unavailable",transform=axis.transAxes,ha="center")
        else:
            _plot_visualization_bundle(axis, data["grid"], data["residual"], "C3", "residual", samples=False)
            axis.axhline(0,color="black",linewidth=.7)
            _shade_visualization_support(axis,data)
        axis.set(title=classes[class_id] + "\n" + data["state"], xlim=(0,365), xlabel="Day of Year")
    for axis in axes.ravel()[len(groups):]:
        axis.set_visible(False)
    figure.suptitle(task + " | Each panel uses its own source-class PCA axis.\n"
                   "Compare temporal residual pattern within each class; do not compare absolute y magnitude across classes.", fontsize=10)
    figure.tight_layout(rect=(0,0,1,.89))
    return figure


def _plot_visualization_matrix(matrix, labels, title, vmax):
    from matplotlib.patches import Rectangle
    plt = _visualization_pyplot()
    figure, axis = plt.subplots(figsize=(max(6,len(labels)*.8), max(5,len(labels)*.65)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgrey")
    artist = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=vmax)
    for row in range(len(labels)):
        axis.add_patch(Rectangle((row-.5,row-.5),1,1,fill=False,edgecolor="red",linewidth=1.5))
        for column in range(len(labels)):
            text = f'{matrix[row,column]:.2f}' if np.isfinite(matrix[row,column]) else "NA"
            axis.text(column,row,text,ha="center",va="center",fontsize=7, color="black" if text == "NA" else "white")
    axis.set(xticks=np.arange(len(labels)),yticks=np.arange(len(labels)),yticklabels=labels,
             xlabel="Target class",ylabel="Source class",title=title)
    axis.set_xticklabels(labels,rotation=45,ha="right")
    figure.colorbar(artist,ax=axis,label="Multidimensional normalized L2")
    figure.tight_layout()
    return figure


_PHASE_APPLICABILITY_ORDER = ("FULL_PHASE", "PARTIAL_PHASE", "PHASE_NOT_APPLICABLE")


def _finite_number(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _truthy(value):
    return value is True or str(value).strip().lower() in ("true", "1", "yes")


def select_phase_applicability_representatives(records):
    """Select auditable cases with the frozen state-specific quality ordering."""
    chosen = {}
    for state in _PHASE_APPLICABILITY_ORDER:
        candidates = [record for record in records if record.get("state") == state]
        if state == "FULL_PHASE":
            key = lambda r: (-_finite_number(r.get("common_coverage"), -np.inf),
                             _finite_number(r.get("landmark_error"), np.inf),
                             _finite_number(r.get("warp_days"), np.inf),
                             str(r.get("task", "")), int(r.get("class_index", -1)))
        elif state == "PARTIAL_PHASE":
            key = lambda r: (not _truthy(r.get("nonlinear_accepted")),
                             -_finite_number(r.get("landmark_gain"), -np.inf),
                             -_finite_number(r.get("common_coverage"), -np.inf),
                             str(r.get("task", "")), int(r.get("class_index", -1)))
        else:
            key = lambda r: (not _truthy(r.get("strong_truncation")),
                             _finite_number(r.get("common_coverage"), np.inf),
                             -_finite_number(r.get("topology_mismatch"), -np.inf),
                             str(r.get("task", "")), int(r.get("class_index", -1)))
        chosen[state] = min(candidates, key=key) if candidates else None
    return chosen


def _phase_applicability_record(task, class_id, class_name, data, status, artifacts):
    state = data["state"]
    raw_support = data.get("support") or None
    support_keys = ("source_common_start_day", "source_common_end_day",
                    "target_common_start_day", "target_common_end_day",
                    "common_time_coverage_min")
    support = ({key: float(raw_support[key]) for key in support_keys if key in raw_support}
               if raw_support else None)
    gamma = None
    phase_x = phase_y = phase_identity = None
    if state == "FULL_PHASE":
        gamma = (artifacts["gamma_by_class"][class_id]
                 if artifacts["selected_lambda"][class_id] is not None else np.linspace(0, 1, 128))
        phase_x = np.linspace(0, 365, len(gamma))
        phase_y = np.asarray(gamma) * 365
        phase_identity = phase_x.copy()
    elif state == "PARTIAL_PHASE":
        gamma = artifacts["partial_gamma_by_class"].get(class_id)
        if artifacts["partial_selected_lambda_by_class"].get(class_id) is None:
            gamma = np.linspace(0, 1, 128)
        if gamma is not None and support:
            source_start, source_end = (float(support["source_common_" + suffix + "_day"])
                                        for suffix in ("start", "end"))
            target_start, target_end = (float(support["target_common_" + suffix + "_day"])
                                        for suffix in ("start", "end"))
            unit = np.linspace(0, 1, len(gamma))
            phase_x = source_start + unit * (source_end-source_start)
            phase_y = target_start + np.asarray(gamma) * (target_end-target_start)
            phase_identity = target_start + unit * (target_end-target_start)
    reason = status.get("phase_applicability_reason", "")
    strong_truncation = any(status.get(side + "_truncation_evidence") == "strong"
                            for side in ("left", "right"))
    mismatch_words = ("mismatch", "unmatched", "insufficient", "invalid")
    return dict(task=task, class_index=int(class_id), class_name=class_name, state=state,
        common_coverage=_finite_number(status.get("common_time_coverage_min"), np.nan),
        landmark_error=_finite_number(status.get("full_landmark_error"), np.nan),
        warp_days=float(np.max(np.abs(np.asarray(gamma)-np.linspace(0, 1, len(gamma)))) *
                        (365 if state == "FULL_PHASE" else
                         (float(support["target_common_end_day"])-float(support["target_common_start_day"])))
                        if gamma is not None else 0.0),
        nonlinear_accepted=_truthy(status.get("partial_nonlinear_accepted")),
        landmark_gain=_finite_number(status.get("partial_landmark_gain"), np.nan),
        strong_truncation=strong_truncation,
        topology_mismatch=sum(word in reason.lower() for word in mismatch_words), reason=reason,
        grid=np.asarray(data["grid"]).tolist(), source=np.asarray(data["source"]["prototype"]).tolist(),
        target_before=np.asarray(data["target_scalar"]["prototype"]).tolist(),
        source_selected=np.asarray(data["source_selected"]["prototype"]).tolist(),
        target_selected=np.asarray(data["target_selected"]["prototype"]).tolist(),
        support=support, phase_x=None if phase_x is None else phase_x.tolist(),
        phase_y=None if phase_y is None else phase_y.tolist(),
        phase_identity=None if phase_identity is None else phase_identity.tolist())


def _plot_phase_applicability_row(axes, record, expected_state):
    for axis, title in zip(axes, ("Before registration", "Shape after selected registration", "Phase")):
        axis.set_title(title, fontsize=9)
    if record is None:
        for axis in axes:
            axis.text(.5, .5, "No case in this task", transform=axis.transAxes, ha="center", va="center")
            axis.set_xticks([]); axis.set_yticks([])
        axes[0].set_ylabel(expected_state)
        return
    grid = np.asarray(record["grid"])
    source, before = np.asarray(record["source"]), np.asarray(record["target_before"])
    axes[0].plot(grid, source, color="C0", linewidth=2.2, label="source")
    axes[0].plot(grid, before, color="C1", linewidth=2.2, label="target Scalar")
    axes[0].legend(fontsize=7)
    if expected_state == "PHASE_NOT_APPLICABLE":
        axes[1].text(.5, .54, "No reliable shape correspondence", transform=axes[1].transAxes,
                     ha="center", va="center", fontweight="bold")
        axes[1].text(.5, .40, record.get("reason", ""), transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=7, wrap=True)
        axes[2].text(.5, .56, "PHASE REJECTED", transform=axes[2].transAxes,
                     ha="center", va="center", fontweight="bold", color="0.35")
        axes[2].text(.5, .40, record.get("reason", ""), transform=axes[2].transAxes,
                     ha="center", va="center", fontsize=7, wrap=True)
    else:
        axes[1].plot(grid, np.asarray(record["source_selected"]), color="C0", linewidth=2.2, label="source")
        axes[1].plot(grid, np.asarray(record["target_selected"]), color="C1", linewidth=2.2,
                     label="target selected")
        axes[1].legend(fontsize=7)
        axes[2].plot(record["phase_x"], record["phase_y"], color="C3", linewidth=2.2, label="selected gamma")
        axes[2].plot(record["phase_x"], record["phase_identity"], color="0.45", linestyle="--",
                     linewidth=1.2, label="identity")
        axes[2].legend(fontsize=7)
        axes[2].set_ylabel("Target day")
    if expected_state == "PARTIAL_PHASE" and record.get("support"):
        support = record["support"]
        for axis in axes[:2]:
            axis.axvspan(0, float(support["source_common_start_day"]), color="grey", alpha=.2)
            axis.axvspan(float(support["source_common_end_day"]), 365, color="grey", alpha=.2)
        source_range = f'[{float(support["source_common_start_day"]):.1f}, {float(support["source_common_end_day"]):.1f}]'
        target_range = f'[{float(support["target_common_start_day"]):.1f}, {float(support["target_common_end_day"]):.1f}]'
        axes[0].set_title(f"Before registration\nsource={source_range}, target={target_range}", fontsize=9)
        axes[1].set_title(f"Shape after selected registration\nsource={source_range}, target={target_range}", fontsize=9)
    low = min(np.nanmin(source), np.nanmin(before))
    high = max(np.nanmax(source), np.nanmax(before))
    if expected_state != "PHASE_NOT_APPLICABLE":
        for values in (record["source_selected"], record["target_selected"]):
            values = np.asarray(values); finite = values[np.isfinite(values)]
            if finite.size:
                low, high = min(low, finite.min()), max(high, finite.max())
    padding = .05 * max(float(high-low), 1e-6)
    for axis in axes[:2]:
        axis.set(xlim=(0,365), ylim=(low-padding, high+padding), xlabel="Day of Year")
    axes[0].set_ylabel(expected_state + "\nsource-class PC1")
    axes[0].text(.01, .98, f'{record["task"]} | {record["class_name"]}', transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=7)
    if record.get("phase_x") is not None:
        axes[2].set_xlabel("Source day")


def plot_phase_applicability_grid(representatives, title):
    plt = _visualization_pyplot()
    figure, axes = plt.subplots(3, 3, figsize=(15, 11), squeeze=False)
    for row, state in enumerate(_PHASE_APPLICABILITY_ORDER):
        _plot_phase_applicability_row(axes[row], representatives.get(state), state)
    figure.suptitle(title + "\nregistration_space = multivariate Mode13 | "
                    "visualization_space = source_class_PC1 | phase_level = class_level_cross_domain",
                    fontsize=11)
    figure.tight_layout(rect=(0,0,1,.94))
    return figure


def plot_phase_applicability_triptych(record):
    plt = _visualization_pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(15, 3.8), squeeze=False)
    _plot_phase_applicability_row(axes[0], record, record["state"])
    figure.suptitle(f'{record["task"]} | {record["class_name"]} | {record["state"]}\n'
                    "registration_space = multivariate Mode13 | visualization_space = source_class_PC1 | "
                    "phase_level = class_level_cross_domain", fontsize=10)
    figure.tight_layout(rect=(0,0,1,.84))
    return figure


def generate_global_phase_applicability_figure(output_root, dpi=160):
    if dpi < 1:
        raise ValueError("visualization DPI must be positive")
    output_root = Path(output_root)
    candidates = []
    manifests = sorted(output_root.glob("*/visualizations/visualization_manifest.json"))
    if len(manifests) != 4:
        raise ValueError(f"exactly four completed task visualization manifests required; found {len(manifests)}")
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        candidates.extend(record for record in manifest.get("phase_applicability_representatives", {}).values()
                          if record is not None)
    chosen = select_phase_applicability_representatives(candidates)
    destination = output_root / "visualizations/phase_applicability"
    destination.mkdir(parents=True, exist_ok=True)
    figure = plot_phase_applicability_grid(chosen, "Four-task representative phase applicability")
    try:
        figure.savefig(destination / "phase_applicability_representative_global.png", dpi=dpi)
    finally:
        _visualization_pyplot().close(figure)
    (destination / "phase_applicability_representative_global.json").write_text(json.dumps(dict(
        registration_space="multivariate Mode13", visualization_space="source_class_PC1",
        phase_level="class_level_cross_domain", representatives=chosen), indent=2), encoding="utf-8")
    return chosen


def generate_diagnostic_visualizations(output_dir, task, classes, source_cache, target_cache,
                                       artifacts, states, max_curves=40, dpi=160):
    """Post-save plotting consumer; inputs contain data/results, never models or loaders."""
    import re
    import warnings
    if max_curves < 1 or dpi < 1:
        raise ValueError("visualization maximum curves and DPI must be positive")
    root = Path(output_dir) / "visualizations"
    root.mkdir(parents=True,exist_ok=True)
    plt = _visualization_pyplot()
    common = sorted(artifacts["phase_applicability_by_class"])
    manifest = dict(task=task, mode=13, same_class_projection="source_class_pca",
        registration_space="multivariate Mode13", visualization_space="source_class_PC1",
        phase_level="class_level_cross_domain",
        cross_class_projection="source_shared_pca", shared_pca_fit_source_only=True,
        max_curves_per_group=max_curves, dpi=dpi, sample_selection="deterministic_linspace",
        prototype="projection_of_existing_multidimensional_pointwise_median",
        sample_median="full_group_projection_median_dashed", band="full_group_q25_q75",
        target_gallery_phase="scalar", target_selected_gallery_phase="selected",
        partial_outside_support="NaN", before_matrix_phase="global_plus_scalar",
        selected_matrix_non_full_target_columns="NA", sample_ids={"source":{},"target":{}},
        shared_pca_classes=common, shared_pca_sign="largest_absolute_loading_positive",
        gallery_ylims={}, files=[], warnings=[], same_class_projection_directions={})
    def warn(message):
        manifest["warnings"].append(message)
        warnings.warn("visualization: " + message, RuntimeWarning)
    def save(figure, relative):
        path = root / relative
        path.parent.mkdir(parents=True,exist_ok=True)
        try:
            figure.savefig(path,dpi=dpi)
        finally:
            plt.close(figure)
        manifest["files"].append(relative)
    state_by_class = {int(row["class_index"]):row for row in states}
    try:
        shared = fit_visualization_shared_pca({c:artifacts["source_raw_prototypes"][c] for c in common})
        manifest["shared_pca_direction"] = shared.tolist()
    except VisualizationUnavailable as error:
        shared = None
        warn(str(error))
    same_groups, shared_groups, applicability_records = {}, {}, []
    grid = np.linspace(0,365,64,endpoint=False)
    for c in common:
        # Class slicing is transient. Only compact 1-D plotting bundles survive this loop.
        sc = np.asarray(source_cache["labels"]) == c
        tc = np.asarray(target_cache["labels"]) == c
        if "sample_id" not in source_cache or "sample_id" not in target_cache:
            warn(f"class {c}: missing stable sample IDs; sample plots skipped")
            continue
        kwargs = dict(source=source_cache["mode13_features"].numpy()[sc],
            target=target_cache["mode13_features"].numpy()[tc],
            source_ids=np.asarray(source_cache["sample_id"])[sc],target_ids=np.asarray(target_cache["sample_id"])[tc],
            source_prototype=artifacts["source_raw_prototypes"][c],target_prototype=artifacts["target_raw_prototypes"][c],
            grid=grid,scalar_delta=artifacts["scalar_delta_by_class"][c],
            state=artifacts["phase_applicability_by_class"][c],
            gamma=artifacts["gamma_by_class"][c] if artifacts["selected_lambda"][c] is not None else None,
            support=artifacts["common_support_by_class"].get(c,{}),
            partial_gamma=artifacts["partial_gamma_by_class"].get(c),max_curves=max_curves)
        try:
            same = prepare_visualization_group(**kwargs,direction=artifacts["source_pca_direction_by_class"][c])
            same["selected_lambda"] = (artifacts["partial_selected_lambda_by_class"].get(c)
                if same["state"] == "PARTIAL_PHASE" else artifacts["selected_lambda"][c])
            same_groups[c] = same
            manifest["sample_ids"]["source"][str(c)] = same["source_ids"].tolist()
            manifest["sample_ids"]["target"][str(c)] = same["target_ids"].tolist()
            manifest["same_class_projection_directions"][str(c)] = artifacts["source_pca_direction_by_class"][c].tolist()
            slug = re.sub(r"[^\w.-]+", "_", classes[c]).strip("._") or f"class_{c}"
            # Prefixing the class index avoids sanitized-name collisions and accidental overwrite.
            save(plot_visualization_spaghetti(same,task,classes[c],state_by_class.get(c,{})),
                 f"cross_domain_same_class/{c}_{slug}_phase_shape_spaghetti.png")
            record = _phase_applicability_record(task, c, classes[c], same, state_by_class.get(c,{}), artifacts)
            applicability_records.append(record)
            save(plot_phase_applicability_triptych(record),
                 f"phase_applicability/paper_style_triptychs/{c}_{slug}.png")
            if shared is not None:
                shared_groups[c] = prepare_visualization_group(**kwargs,direction=shared)
        except VisualizationUnavailable as error:
            warn(f"class {c}: {error}")
        del kwargs
    representatives = select_phase_applicability_representatives(applicability_records)
    manifest["phase_applicability_representatives"] = representatives
    save(plot_phase_applicability_grid(representatives, task + " | Phase applicability cases"),
         "phase_applicability/phase_applicability_cases.png")
    if shared_groups:
        ylim = _visualization_ylim(g[k] for g in shared_groups.values() for k in ("source","target_scalar","target_selected"))
        for key, name, title in (("source","source_class_gallery","Source"),
                ("target_scalar","target_class_gallery","Target Global + Scalar"),
                ("target_selected","target_selected_phase_gallery","Target selected phase")):
            save(_plot_visualization_gallery(shared_groups,classes,key,task + " | " + title,ylim),
                 f"within_domain_class_gallery/{name}.png")
            manifest["gallery_ylims"][name] = list(ylim)
        for key, name in (("source","source"),("target_scalar","target")):
            save(_plot_visualization_overlay(shared_groups,classes,key,task + " | " + name + " shared PCA prototypes",ylim),
                 f"within_domain_class_gallery/{name}_class_prototypes_overlay.png")
    if same_groups:
        save(_plot_visualization_residuals(same_groups,classes,task),"cross_domain_residual/shape_residual_by_class.png")
    if common:
        before, selected = visualization_distance_matrices(artifacts,common)
        finite = np.r_[before[np.isfinite(before)],selected[np.isfinite(selected)]]
        if finite.size:
            vmax = max(float(finite.max()),EPS)
            for matrix, stage in ((before,"before"),(selected,"selected")):
                save(_plot_visualization_matrix(matrix,[classes[c] for c in common],
                    task + " | " + ("Before: Global + Scalar" if stage == "before" else "Selected: FULL_PHASE columns only"),vmax),
                    f"cross_class_matrix/source_target_shape_distance_{stage}.png")
        else:
            warn("no finite multidimensional distances; matrices skipped")
    manifest["residual_axis_caveat"] = "Each panel uses its own source-class PCA axis; do not compare absolute y magnitude across classes."
    (root / "visualization_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()
