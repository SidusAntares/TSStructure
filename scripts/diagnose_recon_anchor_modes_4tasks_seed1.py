#!/usr/bin/env python3
"""Run the offline low-mode salient anchor audit from local artifacts only."""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.recon_anchor_diagnostic import (
    classify_source_anchor,
    detect_salient_extrema,
    diagnostic_eligibility,
    finite_median,
    local_multivariate_correlation,
    local_scalar_correlation,
    match_sample_anchor,
    project_fourier_coefficients,
    read_authoritative_global_shift,
    shifted_calendar,
    summarize_matches,
    write_csv,
)
from models.fourier_reconstruction import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
)
from scripts.visualize_shift_configs_4tasks import (
    DOMAINS,
    _key_values,
    _loader,
    build_split_datasets,
    fit_class_projections,
    load_classes,
    load_raw_spatial_encoder,
    reconstruct_class_prototype,
    resolve_source_checkpoint,
)


MODES = (7, 9, 11, 13)
TASKS = (("AT1", "DK1"), ("DK1", "FR1"), ("FR1", "FR2"), ("FR2", "AT1"))
SUMMARY_FIELDS = (
    "task", "class_id", "class_name", "mode", "source_count", "target_count",
    "prototype_anchor_exists", "prototype_anchor_type", "prototype_anchor_day",
    "prototype_anchor_prominence", "prototype_normalized_prominence", "prototype_width_days",
    "strongest_peak_day", "strongest_peak_prominence", "strongest_peak_normalized_prominence",
    "strongest_valley_day", "strongest_valley_prominence", "strongest_valley_normalized_prominence",
    "num_prominent_peaks", "num_prominent_valleys", "source_match_count", "source_occurrence_rate",
    "source_anchor_timing_median", "source_anchor_timing_std", "source_anchor_timing_mad",
    "source_anchor_error_median", "source_anchor_error_p90", "source_anchor_status",
    "target_match_rate_raw", "target_match_rate_global", "target_timing_error_raw_median",
    "target_timing_error_global_median", "target_timing_error_reduction",
    "target_within_10d_global", "target_within_20d_global", "target_within_30d_global",
    "source_norm_prominence_median", "target_norm_prominence_median",
    "prominence_abs_diff_median", "local_pc1_corr_median", "local_multivar_corr_median",
    "source_extrema_count_median", "target_extrema_count_median",
    "extrema_count_abs_diff_median", "eligibility", "reason",
)


@torch.inference_mode()
def extract_mode_coefficients(
    spatial_encoder, dataset, class_ids, modes, batch_size, device, with_extra
):
    grouped = {mode: defaultdict(list) for mode in modes}
    analyzers = {
        mode: BatchedDirectFourierAnalyzer(mode, period_days=365.0, reg=0.001).to(device)
        for mode in modes
    }
    for sample in _loader(dataset, batch_size):
        pixels = sample["pixels"].to(device)
        mask = sample["valid_pixels"].to(device)
        extra = sample["extra"].to(device) if with_extra else None
        features = spatial_encoder(pixels, mask, extra)
        positions = sample["positions"].to(device=device, dtype=features.dtype)
        labels = sample["label"].numpy()
        # Oracle target labels first enter here, after the PSE/Fourier cache is complete.
        for mode, analyzer in analyzers.items():
            coefficients, _ = analyzer(features, positions, collect_diagnostics=False)
            values = coefficients.cpu().numpy()
            for class_id in np.unique(labels):
                class_id = int(class_id)
                if class_id in class_ids:
                    grouped[mode][class_id].append(values[labels == class_id])
    return {
        mode: {
            class_id: np.concatenate(parts, axis=0)
            for class_id, parts in class_groups.items()
        }
        for mode, class_groups in grouped.items()
    }


@torch.inference_mode()
def reconstruct_projected_curves(coefficients, projection, mode, device, batch_size=256):
    projected, offset = project_fourier_coefficients(coefficients, projection)
    synthesizer = BatchedDirectFourierSynthesizer(mode, period_days=365.0).to(device)
    curves = []
    days = torch.arange(365, device=device, dtype=torch.float32)
    for start in range(0, len(projected), batch_size):
        batch = torch.as_tensor(projected[start : start + batch_size], device=device)
        positions = days.unsqueeze(0).expand(len(batch), -1)
        curves.append(
            synthesizer(batch.unsqueeze(-1), positions).squeeze(-1).cpu().numpy() - offset
        )
    return np.concatenate(curves, axis=0)


@torch.inference_mode()
def reconstruct_target_windows(coefficients, centers, mode, device, radius=30, batch_size=256):
    coefficients = np.asarray(coefficients)
    centers = np.asarray(centers, dtype=np.float32)
    synthesizer = BatchedDirectFourierSynthesizer(mode, period_days=365.0).to(device)
    result = []
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    for start in range(0, len(coefficients), batch_size):
        stop = min(start + batch_size, len(coefficients))
        coeff = torch.as_tensor(coefficients[start:stop], device=device)
        positions = torch.as_tensor(centers[start:stop], device=device).unsqueeze(1) + offsets
        result.append(synthesizer(coeff, positions).cpu().numpy())
    return np.concatenate(result, axis=0) if result else np.empty((0, 2 * radius + 1, 0))


def _safe_name(class_id, class_name):
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", class_name).strip("_")
    return f"class_{class_id:02d}_{name}"


def _plot_source(path, grid, curves, prototype, detection, status, title, max_samples, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    chosen = np.arange(len(curves))
    if len(chosen) > max_samples:
        chosen = np.sort(rng.choice(chosen, max_samples, replace=False))
    figure, axis = plt.subplots(figsize=(10, 4.8))
    for curve in curves[chosen]:
        axis.plot(grid, curve, color="#9ECAE1", alpha=0.18, lw=0.6)
    q25, q75 = np.quantile(curves, (0.25, 0.75), axis=0)
    axis.fill_between(grid, q25, q75, color="#4C78A8", alpha=0.2, label="source IQR")
    axis.plot(grid, prototype, color="#1F4E79", lw=2.4, label="source median")
    for item in detection.extrema:
        index = int(np.argmin(abs(grid - item.day)))
        axis.scatter(item.day, prototype[index], marker="^" if item.kind == "peak" else "v", s=34)
    if detection.principal_anchor is not None:
        axis.axvline(detection.principal_anchor.day, color="#D62728", ls="--", lw=1.5)
        axis.axvspan(detection.principal_anchor.day - 30, detection.principal_anchor.day + 30, color="#D62728", alpha=0.07)
    else:
        axis.text(0.5, 0.9, "NO STABLE SALIENT ANCHOR", transform=axis.transAxes, ha="center", weight="bold")
    axis.set(xlim=(0, 364), xlabel="Day", ylabel="source-class PC1", title=f"{title}\n{status}")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_cross_domain(path, grid, source, target, shift, anchor, raw_matches, global_matches, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_median = np.median(source, axis=0)
    target_median = np.median(target, axis=0)
    shifted, valid = shifted_calendar(grid, shift)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    axes[0].plot(grid, source_median, lw=2.2, label="source")
    axes[0].plot(grid, target_median, lw=2.2, label="target")
    axes[0].set_title("Raw calendar")
    axes[1].plot(grid, source_median, lw=2.2, label="source")
    axes[1].plot(shifted[valid], target_median[valid], lw=2.2, label="target + global shift")
    axes[1].set_title(f"Global aligned (delta={shift:+g} d)")
    if anchor is not None:
        for axis in axes:
            axis.axvline(anchor.day, color="#D62728", ls="--")
            axis.axvspan(anchor.day - 30, anchor.day + 30, color="#D62728", alpha=0.06)
        for axis, matches in zip(axes, (raw_matches, global_matches)):
            days = [item.day for item in matches if item.matched]
            if days:
                axis.axvline(np.median(days), color="#F28E2B", ls=":", lw=2)
    for axis in axes:
        axis.set_xlim(0, 364)
        axis.set_xlabel("Day in source frame")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("source-class PC1")
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_mode_comparison(path, class_name, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_values = np.concatenate([np.ravel(record["source_prototype"]) for record in records] + [np.ravel(record["target_prototype"]) for record in records])
    low, high = np.nanmin(all_values), np.nanmax(all_values)
    padding = max((high - low) * 0.05, 0.1)
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharey=True)
    for axis, record in zip(axes, records):
        grid = record["grid"]
        shifted, valid = shifted_calendar(grid, record["shift"])
        axis.fill_between(grid, record["q25"], record["q75"], alpha=0.18)
        axis.plot(grid, record["source_prototype"], lw=2.0, label="source")
        axis.plot(shifted[valid], record["target_prototype"][valid], lw=2.0, label="target global")
        anchor = record["anchor"]
        if anchor is not None:
            axis.axvline(anchor.day, color="#D62728", ls="--")
            axis.axvspan(anchor.day - 30, anchor.day + 30, color="#D62728", alpha=0.06)
            matched_days = [item.day for item in record["global_matches"] if item.matched]
            if matched_days:
                axis.axvline(np.median(matched_days), color="#F28E2B", ls=":", lw=1.8)
        else:
            axis.text(0.5, 0.85, "NO STABLE\nSALIENT ANCHOR", transform=axis.transAxes, ha="center")
        for item in record["detection"].extrema:
            index = int(np.argmin(abs(grid - item.day)))
            axis.scatter(
                item.day,
                record["source_prototype"][index],
                marker="^" if item.kind == "peak" else "v",
                s=24,
            )
        row = record["row"]
        axis.set_title(
            f"Mode {record['mode']}\nocc={row['source_occurrence_rate']:.2f}, MAD={row['source_anchor_timing_mad']:.1f}\n"
            f"target match={row['target_match_rate_global']:.2f}, err={row['target_timing_error_global_median']:.1f}\n"
            f"local multi={row['local_multivar_corr_median']:.3f}"
        )
        axis.set(xlim=(0, 364), ylim=(low - padding, high + padding), xlabel="Day")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("source-class PC1")
    axes[0].legend(frameon=False)
    figure.suptitle(f"{class_name}: independent low-mode anchor views; fixed source-class PC1")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_task_heatmaps(output_dir, classes, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    occurrence = np.full((len(classes), len(MODES)), np.nan)
    timing = np.full_like(occurrence, np.nan)
    for row in rows:
        i, j = int(row["class_id"]), MODES.index(int(row["mode"]))
        occurrence[i, j] = row["source_occurrence_rate"]
        timing[i, j] = row["target_timing_error_global_median"]
    figure, axes = plt.subplots(1, 2, figsize=(12, max(5, 0.45 * len(classes))))
    for axis, matrix, title, cmap in (
        (axes[0], occurrence, "Source anchor occurrence", "viridis"),
        (axes[1], timing, "Global-aligned target median error (days)", "magma_r"),
    ):
        image = axis.imshow(matrix, aspect="auto", cmap=cmap)
        axis.set_xticks(range(len(MODES)), MODES)
        axis.set_yticks(range(len(classes)), classes)
        axis.set_xlabel("Independent Fourier mode count")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(output_dir / "anchor_timing_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    for matrix, title, filename, cmap in (
        (occurrence, "Source anchor occurrence", "source_anchor_stability_heatmap.png", "viridis"),
        (timing, "Global-aligned target median timing error (days)", "global_target_timing_heatmap.png", "magma_r"),
    ):
        figure, axis = plt.subplots(figsize=(6.5, max(5, 0.45 * len(classes))))
        image = axis.imshow(matrix, aspect="auto", cmap=cmap)
        axis.set_xticks(range(len(MODES)), MODES)
        axis.set_yticks(range(len(classes)), classes)
        axis.set(xlabel="Independent Fourier mode count", title=title)
        figure.colorbar(image, ax=axis, fraction=0.046)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)


def _plot_task_mode_overview(output_dir, mode_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [row["mode"] for row in mode_rows]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].bar(modes, [row["source_stable_class_rate"] for row in mode_rows])
    axes[0].set_title("Source stable-class rate")
    axes[1].bar(modes, [row["median_target_within_20d"] for row in mode_rows])
    axes[1].set_title("Target anchors within 20 days")
    axes[2].bar(modes, [row["median_local_multivar_corr"] for row in mode_rows])
    axes[2].set_title("Local multivariate correlation")
    for axis in axes:
        axis.set_xticks(modes)
        axis.set_xlabel("Independent Fourier mode count")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Diagnostic overview (no automatic best-mode selection)")
    figure.tight_layout()
    figure.savefig(output_dir / "task_mode_overview.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze_class_mode(args, task_name, class_id, class_name, mode, projection, source_coeff, target_coeff, source_prototype_multi, global_shift, device):
    grid = np.arange(365, dtype=np.float64)
    source_curves = reconstruct_projected_curves(source_coeff, projection, mode, device, args.batch_size)
    target_curves = reconstruct_projected_curves(target_coeff, projection, mode, device, args.batch_size)
    source_prototype = np.median(source_curves, axis=0)
    target_prototype = np.median(target_curves, axis=0)
    detection = detect_salient_extrema(source_prototype, grid, args.min_extrema_distance_days, args.min_width_days, args.min_normalized_prominence)
    anchor = detection.principal_anchor
    source_matches = [] if anchor is None else [
        match_sample_anchor(curve, grid, anchor, args.anchor_search_radius_days, args.min_width_days, args.min_normalized_prominence, 0, args.min_extrema_distance_days)
        for curve in source_curves
    ]
    raw_matches = [] if anchor is None else [
        match_sample_anchor(curve, grid, anchor, args.anchor_search_radius_days, args.min_width_days, args.min_normalized_prominence, 0, args.min_extrema_distance_days)
        for curve in target_curves
    ]
    global_matches = [] if anchor is None else [
        match_sample_anchor(curve, grid, anchor, args.anchor_search_radius_days, args.min_width_days, args.min_normalized_prominence, global_shift, args.min_extrema_distance_days)
        for curve in target_curves
    ]
    source_stats = summarize_matches(source_matches)
    raw_stats = summarize_matches(raw_matches)
    global_stats = summarize_matches(global_matches)
    source_status = classify_source_anchor(anchor, source_stats["occurrence_rate"], source_stats["timing_mad"], args.min_normalized_prominence)
    local_scalar, local_multi, source_counts, target_counts = [], [], [], []
    matched_indices = [index for index, item in enumerate(global_matches) if item.matched]
    if source_status == "SOURCE_ANCHOR_STABLE" and matched_indices and anchor.day >= 30 and anchor.day <= 334:
        original_centers = [global_matches[index].day - global_shift for index in matched_indices]
        in_bounds = [
            (center >= 30 and center <= 334) for center in original_centers
        ]
        valid_indices = [index for index, keep in zip(matched_indices, in_bounds) if keep]
        valid_centers = [center for center, keep in zip(original_centers, in_bounds) if keep]
        target_windows = reconstruct_target_windows(target_coeff[valid_indices], valid_centers, mode, device, 30, args.batch_size)
        source_window_multi = source_prototype_multi[int(anchor.day) - 30 : int(anchor.day) + 31]
        source_window_scalar = source_prototype[int(anchor.day) - 30 : int(anchor.day) + 31]
        source_window_detection = detect_salient_extrema(source_window_scalar, np.arange(-30, 31), 3, 2, 0.05)
        for index, center, target_window in zip(valid_indices, valid_centers, target_windows):
            local_scalar.append(local_scalar_correlation(source_prototype, grid, anchor.day, target_curves[index], grid, center, 30))
            value, _ = local_multivariate_correlation(source_window_multi, target_window)
            local_multi.append(value)
            target_window_scalar = target_curves[index][int(center) - 30 : int(center) + 31]
            target_detection = detect_salient_extrema(target_window_scalar, np.arange(-30, 31), 3, 2, 0.05)
            source_counts.append(len(source_window_detection.extrema))
            target_counts.append(len(target_detection.extrema))
    nan = float("nan")
    prominence_differences = [abs(item.normalized_prominence - anchor.normalized_prominence) for item in global_matches if item.matched] if anchor else []
    global_errors = [item.absolute_error for item in global_matches if item.matched]
    row = {
        "task": task_name, "class_id": class_id, "class_name": class_name, "mode": mode,
        "source_count": len(source_curves), "target_count": len(target_curves),
        "prototype_anchor_exists": anchor is not None,
        "prototype_anchor_type": "" if anchor is None else anchor.kind,
        "prototype_anchor_day": nan if anchor is None else anchor.day,
        "prototype_anchor_prominence": nan if anchor is None else anchor.prominence,
        "prototype_normalized_prominence": nan if anchor is None else anchor.normalized_prominence,
        "prototype_width_days": nan if anchor is None else anchor.width_days,
        "strongest_peak_day": nan if detection.strongest_peak is None else detection.strongest_peak.day,
        "strongest_peak_prominence": nan if detection.strongest_peak is None else detection.strongest_peak.prominence,
        "strongest_peak_normalized_prominence": nan if detection.strongest_peak is None else detection.strongest_peak.normalized_prominence,
        "strongest_valley_day": nan if detection.strongest_valley is None else detection.strongest_valley.day,
        "strongest_valley_prominence": nan if detection.strongest_valley is None else detection.strongest_valley.prominence,
        "strongest_valley_normalized_prominence": nan if detection.strongest_valley is None else detection.strongest_valley.normalized_prominence,
        "num_prominent_peaks": sum(item.kind == "peak" for item in detection.extrema),
        "num_prominent_valleys": sum(item.kind == "valley" for item in detection.extrema),
        "source_match_count": source_stats["match_count"], "source_occurrence_rate": source_stats["occurrence_rate"],
        "source_anchor_timing_median": source_stats["timing_median"], "source_anchor_timing_std": source_stats["timing_std"],
        "source_anchor_timing_mad": source_stats["timing_mad"], "source_anchor_error_median": source_stats["timing_error_median"],
        "source_anchor_error_p90": source_stats["timing_error_p90"], "source_anchor_status": source_status,
        "target_match_rate_raw": raw_stats["occurrence_rate"], "target_match_rate_global": global_stats["occurrence_rate"],
        "target_timing_error_raw_median": raw_stats["timing_error_median"], "target_timing_error_global_median": global_stats["timing_error_median"],
        "target_timing_error_reduction": raw_stats["timing_error_median"] - global_stats["timing_error_median"],
        "target_within_10d_global": np.mean(np.asarray(global_errors) <= 10) if global_errors else nan,
        "target_within_20d_global": np.mean(np.asarray(global_errors) <= 20) if global_errors else nan,
        "target_within_30d_global": np.mean(np.asarray(global_errors) <= 30) if global_errors else nan,
        "source_norm_prominence_median": source_stats["prominence_median"], "target_norm_prominence_median": global_stats["prominence_median"],
        "prominence_abs_diff_median": finite_median(prominence_differences), "local_pc1_corr_median": finite_median(local_scalar),
        "local_multivar_corr_median": finite_median(local_multi), "source_extrema_count_median": finite_median(source_counts),
        "target_extrema_count_median": finite_median(target_counts),
        "extrema_count_abs_diff_median": finite_median(abs(np.asarray(source_counts) - np.asarray(target_counts))) if source_counts else nan,
        "eligibility": diagnostic_eligibility(source_status, global_stats["occurrence_rate"], global_stats["timing_error_median"]),
        "reason": "" if anchor is not None else "no_salient_prototype_extremum",
    }
    return row, {
        "grid": grid, "mode": mode, "source_curves": source_curves, "target_curves": target_curves,
        "source_prototype": source_prototype, "target_prototype": target_prototype,
        "q25": np.quantile(source_curves, 0.25, axis=0), "q75": np.quantile(source_curves, 0.75, axis=0),
        "detection": detection, "anchor": anchor, "source_matches": source_matches,
        "raw_matches": raw_matches, "global_matches": global_matches, "shift": global_shift, "row": row,
    }


def run_task(args, source_alias, target_alias, checkpoint_overrides):
    source_path, target_path = DOMAINS[source_alias], DOMAINS[target_alias]
    checkpoint, config = resolve_source_checkpoint(args.source_checkpoint_root, source_alias, source_path, args.seed, args.fold, checkpoint_overrides)
    classes = load_classes(checkpoint, source_path, args.data_root)
    task_name = f"{source_alias}_{target_alias}"
    shift = read_authoritative_global_shift(args.shift_visualization_root, task_name)
    source_dataset, target_dataset = build_split_datasets(args.data_root, source_path, target_path, classes, args.seed, config)
    device = torch.device(args.device)
    spatial_encoder = load_raw_spatial_encoder(checkpoint, config, classes, device)
    projections, latent_dim = fit_class_projections(spatial_encoder, source_dataset, len(classes), np.arange(365), args.batch_size, device, bool(config.get("with_extra", False)))
    class_ids = set(projections)
    source_coefficients = extract_mode_coefficients(spatial_encoder, source_dataset, class_ids, MODES, args.batch_size, device, bool(config.get("with_extra", False)))
    target_coefficients = extract_mode_coefficients(spatial_encoder, target_dataset, class_ids, MODES, args.batch_size, device, bool(config.get("with_extra", False)))
    output_dir = args.output_root / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, class_records = [], defaultdict(list)
    for class_id, class_name in enumerate(classes):
        if class_id not in projections:
            continue
        class_dir = output_dir / _safe_name(class_id, class_name)
        for mode in MODES:
            source_coeff = source_coefficients[mode].get(class_id)
            target_coeff = target_coefficients[mode].get(class_id)
            if source_coeff is None or target_coeff is None:
                continue
            synthesizer = BatchedDirectFourierSynthesizer(mode, period_days=365.0).to(device)
            source_prototype_multi = reconstruct_class_prototype(source_coeff, synthesizer, device, args.batch_size)
            row, record = analyze_class_mode(args, task_name, class_id, class_name, mode, projections[class_id], source_coeff, target_coeff, source_prototype_multi, shift.shift_days, device)
            all_rows.append(row)
            class_records[class_id].append(record)
            mode_dir = class_dir / f"mode{mode:02d}"
            _plot_source(mode_dir / "source_anchor.png", record["grid"], record["source_curves"], record["source_prototype"], record["detection"], row["source_anchor_status"], f"{task_name} | {class_name} | mode {mode}", args.max_spaghetti, args.seed + class_id + mode)
            _plot_cross_domain(mode_dir / "cross_domain_anchor.png", record["grid"], record["source_curves"], record["target_curves"], shift.shift_days, record["anchor"], record["raw_matches"], record["global_matches"], f"{task_name} | {class_name} | mode {mode}")
            sample_rows = []
            for index in range(max(len(record["source_matches"]), len(record["global_matches"]))):
                source_match = record["source_matches"][index] if index < len(record["source_matches"]) else None
                raw_match = record["raw_matches"][index] if index < len(record["raw_matches"]) else None
                global_match = record["global_matches"][index] if index < len(record["global_matches"]) else None
                sample_rows.append({
                    "sample_index": index,
                    "source_matched": "" if source_match is None else source_match.matched,
                    "source_day": "" if source_match is None else source_match.day,
                    "source_normalized_prominence": "" if source_match is None else source_match.normalized_prominence,
                    "target_raw_matched": "" if raw_match is None else raw_match.matched,
                    "target_raw_day": "" if raw_match is None else raw_match.day,
                    "target_raw_normalized_prominence": "" if raw_match is None else raw_match.normalized_prominence,
                    "target_global_matched": "" if global_match is None else global_match.matched,
                    "target_global_day": "" if global_match is None else global_match.day,
                    "target_global_error": "" if global_match is None else global_match.absolute_error,
                    "target_global_normalized_prominence": "" if global_match is None else global_match.normalized_prominence,
                })
            write_csv(mode_dir / "sample_metrics.csv", sample_rows, ("sample_index", "source_matched", "source_day", "source_normalized_prominence", "target_raw_matched", "target_raw_day", "target_raw_normalized_prominence", "target_global_matched", "target_global_day", "target_global_error", "target_global_normalized_prominence"))
        if class_records[class_id]:
            _plot_mode_comparison(class_dir / "mode_comparison.png", class_name, class_records[class_id])
    write_csv(output_dir / "class_mode_summary.csv", all_rows, SUMMARY_FIELDS)
    mode_rows = []
    for mode in MODES:
        rows = [row for row in all_rows if row["mode"] == mode]
        stable = [row for row in rows if row["source_anchor_status"] == "SOURCE_ANCHOR_STABLE"]
        mode_rows.append({
            "mode": mode, "num_classes": len(rows),
            "num_classes_with_anchor": sum(bool(row["prototype_anchor_exists"]) for row in rows),
            "num_source_stable_classes": len(stable), "source_stable_class_rate": len(stable) / len(rows) if rows else np.nan,
            "median_source_occurrence": finite_median(row["source_occurrence_rate"] for row in rows),
            "median_source_timing_mad": finite_median(row["source_anchor_timing_mad"] for row in rows),
            "median_target_global_match_rate": finite_median(row["target_match_rate_global"] for row in rows),
            "median_target_global_timing_error": finite_median(row["target_timing_error_global_median"] for row in rows),
            "median_target_within_20d": finite_median(row["target_within_20d_global"] for row in rows),
            "median_prominence_difference": finite_median(row["prominence_abs_diff_median"] for row in rows),
            "median_local_pc1_corr": finite_median(row["local_pc1_corr_median"] for row in rows),
            "median_local_multivar_corr": finite_median(row["local_multivar_corr_median"] for row in rows),
            "median_extrema_count_difference": finite_median(row["extrema_count_abs_diff_median"] for row in rows),
        })
    write_csv(output_dir / "mode_summary.csv", mode_rows, tuple(mode_rows[0]))
    _plot_task_heatmaps(output_dir, classes, all_rows)
    _plot_task_mode_overview(output_dir, mode_rows)
    manifest = {
        "audit_type": "offline_low_mode_salient_anchor_diagnostic",
        "training": False, "parameter_update": False, "target_pseudo_label": False,
        "target_label_usage": "oracle_offline_grouping_only",
        "task": task_name, "source": source_path, "target": target_path,
        "seed": args.seed, "fold": args.fold, "source_checkpoint": str(checkpoint),
        "raw_pse_checkpoint_semantics": "source-only Raw PseLTae",
        "global_shift_source": shift.source_path, "global_shift_days": shift.shift_days,
        "global_shift_semantics": shift.semantics,
        "fourier": {"modes": list(MODES), "fit": "independent dense-direct ridge fit per mode", "period_days": 365, "reg": 0.001, "grid_days": [0, 364]},
        "pca": "one source-class PC1 fitted once from fixed Raw PSE observations and reused across all modes; target transform-only; largest-absolute loading positive",
        "anchor_detector": {"min_extrema_distance_days": args.min_extrema_distance_days, "min_width_days": args.min_width_days, "min_normalized_prominence": args.min_normalized_prominence, "threshold_role": "diagnostic only"},
        "matching": {"radius_days": args.anchor_search_radius_days, "same_type": True, "circular_wrap": False},
        "local_window": {"radius_days": 30, "warp": False},
        "source_count": len(source_dataset), "target_count": len(target_dataset), "latent_dim": latent_dim,
        "classes": classes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[FINISHED] {task_name}: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--source-checkpoint-root", type=Path, default=Path("outputs"))
    parser.add_argument("--source-checkpoint", action="append", default=[])
    parser.add_argument("--shift-visualization-root", type=Path, default=Path("outputs/shift_visualizations_seed1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/recon_anchor_diagnostic_4tasks_seed1"))
    parser.add_argument("--source-domain", choices=DOMAINS)
    parser.add_argument("--target-domain", choices=DOMAINS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-extrema-distance-days", type=float, default=7)
    parser.add_argument("--min-width-days", type=float, default=3)
    parser.add_argument("--min-normalized-prominence", type=float, default=0.20)
    parser.add_argument("--anchor-search-radius-days", type=float, default=30)
    parser.add_argument("--max-spaghetti", type=int, default=40)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fold != 0:
        raise SystemExit("ERROR: this diagnostic reproduces fold 0 only")
    if (args.source_domain is None) != (args.target_domain is None):
        raise SystemExit("ERROR: source and target domains must be provided together")
    for path, label in ((args.data_root, "data root"), (args.source_checkpoint_root, "source checkpoint root"), (args.shift_visualization_root, "shift visualization root")):
        if not path.is_dir():
            raise SystemExit(f"ERROR: {label} not found: {path}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tasks = ((args.source_domain, args.target_domain),) if args.source_domain else TASKS
    overrides = _key_values(args.source_checkpoint)
    for source_alias, target_alias in tasks:
        run_task(args, source_alias, target_alias, overrides)


if __name__ == "__main__":
    main()
