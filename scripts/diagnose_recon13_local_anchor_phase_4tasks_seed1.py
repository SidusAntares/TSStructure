#!/usr/bin/env python3
"""Offline oracle Mode13 local registration: batched shape fitting or SRVF-DP."""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.phase_shape_diagnostic import normalized_l2
from analysis.shift_visualization import SourceClassPC1
from analysis.recon_anchor_diagnostic import (
    anchor_domain_elevation,
    apply_local_forward_to_timestamps,
    build_anchor_time_maps,
    compose_local_query,
    detect_salient_extrema,
    domain_projection_baseline,
    estimate_anchor_fixed_local_phases,
    estimate_batched_window_phase,
    finite_median,
    match_sample_anchor,
    read_authoritative_global_shift,
    select_gated_anchor,
    summarize_matches,
    write_csv,
    window_metrics_batch,
)
from models.fourier_reconstruction import (
    BatchedDirectFourierAnalyzer, BatchedDirectFourierSynthesizer,
    positions_to_periodic_points, centered_modes,
)
from scripts.visualize_shift_configs_4tasks import (
    DOMAINS, _key_values, _loader, build_split_datasets,
    load_classes, load_raw_spatial_encoder,
    resolve_source_checkpoint,
)

NUM_MODES = 13
PERIOD_DAYS = 365.0
REG = 0.001
SRVF_LAMBDA = 0
TASKS = (("AT1", "DK1"), ("DK1", "FR1"), ("FR1", "FR2"), ("FR2", "AT1"))

SAMPLE_FIELDS = (
    "sample_id", "true_class", "anchor_found", "target_anchor_day",
    "anchor_residual_days", "anchor_relative_prominence",
    "anchor_domain_relative_elevation", "anchor_warp_valid",
    "anchor_warp_max_displacement", "anchor_warp_left_scale",
    "anchor_warp_right_scale", "anchor_warp_extreme", "nonlinear_valid",
    "nonlinear_max_displacement", "nonlinear_min_derivative",
    "nonlinear_median_derivative", "nonlinear_max_derivative",
    "nonlinear_extreme", "corr_global", "corr_anchor", "corr_nonlinear",
    "srvf_dist_global", "srvf_dist_anchor", "srvf_dist_nonlinear",
    "local_distance_global", "local_distance_anchor", "local_distance_nonlinear",
    "anchor_gain", "nonlinear_gain", "anchor_only_sufficient",
    "nonlinear_helpful", "raw_time_mapping_monotone", "fallback_reason", "phase_solver",
)


def interpolation_statistics(features, positions, grid):
    """Exact linear-interpolation PCA statistics via W.T W, without [365,D]."""
    features = np.asarray(features, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    unique, inverse, counts = np.unique(positions, return_inverse=True, return_counts=True)
    if len(unique) < 2:
        raise ValueError("at least two distinct source timestamps are required")
    right = np.searchsorted(unique, grid, side="right").clip(1, len(unique)-1)
    left = right-1
    fraction = ((grid-unique[left])/(unique[right]-unique[left])).clip(0, 1)
    weights = np.zeros((len(grid), len(unique)))
    weights[np.arange(len(grid)), left] = 1-fraction
    weights[np.arange(len(grid)), right] += fraction
    weights = weights[:, inverse]/counts[inverse]
    return weights.sum(0) @ features, features.T @ (weights.T @ weights) @ features


@torch.inference_mode()
def _shared_basis(days, device, dtype=torch.complex64):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    positions = torch.as_tensor(days, device=device, dtype=real_dtype)
    phase = positions_to_periodic_points(positions, PERIOD_DAYS)[:, None] * centered_modes(NUM_MODES, device, real_dtype)
    return torch.exp(1j*phase).to(dtype)


def reconstruct_projected_fast(coefficients, projection):
    """All-domain scalar projection with a shared basis and bounded BLAS blocks."""
    coefficients = np.asarray(coefficients)
    modes = np.arange(-(NUM_MODES//2), NUM_MODES//2+1)
    points = (2*np.pi*np.arange(365)/PERIOD_DAYS + np.pi) % (2*np.pi) - np.pi
    basis = np.exp(1j*points[:, None]*modes)
    result = np.empty((len(coefficients), 365), dtype=np.float64)
    offset = float(projection.center @ projection.axis)
    # Avoid a whole-domain complex64 -> complex128 temporary for the PC1 dot.
    for start in range(0, len(coefficients), 2048):
        projected = np.einsum("nfd,d->nf", coefficients[start:start+2048], projection.axis, optimize=True)
        result[start:start+2048] = (projected @ basis.T).real-offset
    return result


@torch.inference_mode()
def reconstruct_projected_queries(
    coefficients, projection, query_days, device, batch_size=256,
):
    """Evaluate source-class PC1 curves on per-sample source-frame queries."""
    coefficients = np.asarray(coefficients)
    query_days = np.asarray(query_days)
    if coefficients.ndim != 3 or query_days.ndim != 2 or len(coefficients) != len(query_days):
        raise ValueError("expected coefficients [N,F,D] and query days [N,T]")
    projected = np.einsum(
        "nfd,d->nf", coefficients, np.asarray(projection.axis), optimize=True
    )
    complex_dtype = torch.complex128 if projected.dtype == np.complex128 else torch.complex64
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    modes = centered_modes(NUM_MODES, device=device, dtype=real_dtype)
    curves = []
    for start in range(0, len(projected), int(batch_size)):
        stop = min(start + int(batch_size), len(projected))
        points = positions_to_periodic_points(
            torch.as_tensor(query_days[start:stop], device=device, dtype=real_dtype),
            PERIOD_DAYS,
        )
        matrix = torch.exp(1j * points.unsqueeze(-1) * modes).to(complex_dtype)
        coefficients_batch = torch.as_tensor(
            projected[start:stop], device=device, dtype=complex_dtype
        )
        curves.append(
            torch.matmul(matrix, coefficients_batch.unsqueeze(-1))
            .squeeze(-1).real.cpu().numpy()
        )
    if not curves:
        return np.empty((0, query_days.shape[1]), dtype=np.float32)
    return np.concatenate(curves) - float(projection.center @ projection.axis)


def full_year_queries(
    grid, global_shift, anchor_maps=None, window_days=None,
    residual_query_days=None,
):
    """Build A/B/C inverse queries on one shared source-calendar grid."""
    grid = np.asarray(grid, dtype=np.float64)
    global_query = grid - float(global_shift)
    anchor_query = global_query.copy()
    nonlinear_query = global_query.copy()
    if anchor_maps is None:
        return global_query, anchor_query, nonlinear_query
    anchor_query = anchor_maps.query(grid) - float(global_shift)
    nonlinear_query = anchor_query.copy()
    if window_days is not None and residual_query_days is not None:
        window_days = np.asarray(window_days, dtype=np.float64)
        inside = (grid >= window_days[0]) & (grid <= window_days[-1])
        nonlinear_query[inside] = (
            compose_local_query(window_days, anchor_maps, residual_query_days)
            - float(global_shift)
        )
    return global_query, anchor_query, nonlinear_query


def _profile_metrics(source_curves, target_curves):
    source_median = np.median(np.asarray(source_curves), axis=0)
    target_median = np.median(np.asarray(target_curves), axis=0)
    denominator = np.linalg.norm(source_median - source_median.mean()) * np.linalg.norm(
        target_median - target_median.mean()
    )
    correlation = (
        float(np.dot(source_median - source_median.mean(), target_median - target_median.mean()) / denominator)
        if denominator > 1e-15 else float("nan")
    )
    return correlation, normalized_l2(source_median, target_median)


def _stable_curve_sample(curves, maximum, seed):
    curves = np.asarray(curves)
    if len(curves) <= maximum:
        return curves
    rng = np.random.default_rng(seed)
    return curves[np.sort(rng.choice(len(curves), maximum, replace=False))]


def _render_full_year_distribution(
    path, task_name, class_name, stage_name, grid, source_curves,
    target_curves, global_shift, ylim, max_spaghetti, seed, metrics,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_sample = _stable_curve_sample(source_curves, max_spaghetti, seed)
    target_sample = _stable_curve_sample(target_curves, max_spaghetti, seed + 7919)
    source_median = np.median(source_curves, axis=0)
    target_median = np.median(target_curves, axis=0)
    source_q25, source_q75 = np.quantile(source_curves, (.25, .75), axis=0)
    target_q25, target_q75 = np.quantile(target_curves, (.25, .75), axis=0)
    labels = {
        "global_only": "Mode13 global shift (before local alignment)",
        "anchor_aligned": "Mode13 main peak/valley aligned",
        "local_nonlinear_aligned": "Mode13 anchor + local nonlinear aligned",
    }
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for curve in source_sample:
        axes[0].plot(grid, curve, color="#4C78A8", alpha=.13, lw=.7)
    for curve in target_sample:
        axes[0].plot(grid, curve, color="#F28E2B", alpha=.13, lw=.7)
    axes[0].plot(grid, source_median, color="#1F4E79", lw=2.4, label="source median")
    axes[0].plot(grid, target_median, color="#B85C00", lw=2.4, label="target median")
    axes[0].legend(loc="best", frameon=False, ncol=2)
    axes[0].set_ylabel("Source-class PC1 score")
    axes[1].fill_between(grid, source_q25, source_q75, color="#4C78A8", alpha=.22)
    axes[1].fill_between(grid, target_q25, target_q75, color="#F28E2B", alpha=.22)
    axes[1].plot(grid, source_median, color="#1F4E79", lw=2.4, label="source median + IQR")
    axes[1].plot(grid, target_median, color="#B85C00", lw=2.4, label="target median + IQR")
    axes[1].legend(loc="best", frameon=False, ncol=2)
    axes[1].set_xlabel("Day in source temporal frame")
    axes[1].set_ylabel("Source-class PC1 score")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.set_xlim(0, 365)
        axis.set_ylim(*ylim)
    correlation, distance = metrics
    fig.suptitle(
        f"{task_name.replace('_', '→')} | {class_name} | {labels[stage_name]}\n"
        f"source n={len(source_curves)} | target n={len(target_curves)} | "
        f"global shift={global_shift:+g} days | median corr={correlation:.4f} | "
        f"normalized L2={distance:.4f}"
    )
    fig.tight_layout(rect=(0, 0, 1, .92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_full_year_alignment_figures(
    output_dir, task_name, class_id, class_name, grid, source_curves,
    stage_curves, global_shift, raw_ylim=None, max_spaghetti=40, seed=1,
    main_final_dir=None,
):
    """Render shift-visualization-compatible full-year A/B/C distributions."""
    folders = {
        "global_only": "01_global_only",
        "anchor_aligned": "02_anchor_aligned",
        "local_nonlinear_aligned": "03_local_nonlinear_aligned",
    }
    expected = set(folders)
    if set(stage_curves) != expected:
        raise ValueError(f"expected full-year stages {sorted(expected)}")
    all_values = np.concatenate(
        [np.asarray(source_curves).reshape(-1)]
        + [np.asarray(stage_curves[key]).reshape(-1) for key in folders]
    )
    finite = all_values[np.isfinite(all_values)]
    if finite.size:
        low, high = map(float, (finite.min(), finite.max()))
        padding = .05 * max(high - low, .1)
        ylim = (low - padding, high + padding)
    else:
        ylim = (-1., 1.)
    if raw_ylim is not None:
        # Exact reuse makes 01 Raw PSE and 04 directly comparable by eye.
        ylim = (float(raw_ylim[0]), float(raw_ylim[1]))
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", class_name).strip("_")
    filename = f"{int(class_id):02d}_{safe_name}.png"
    metadata = {}
    for stage, folder in folders.items():
        curves = np.asarray(stage_curves[stage])
        metrics = _profile_metrics(source_curves, curves)
        path = Path(output_dir) / "full_year_alignment" / folder / filename
        _render_full_year_distribution(
            path, task_name, class_name, stage, np.asarray(grid),
            np.asarray(source_curves), curves, global_shift, ylim,
            max_spaghetti, seed + int(class_id), metrics,
        )
        metadata[stage] = {
            "path": str(path), "ylim": [float(ylim[0]), float(ylim[1])],
            "source_count": int(len(source_curves)),
            "target_count": int(len(curves)),
            "median_profile_correlation": float(metrics[0]),
            "median_profile_normalized_l2": float(metrics[1]),
        }
        if stage == "local_nonlinear_aligned" and main_final_dir is not None:
            main_path = Path(main_final_dir) / filename
            _render_full_year_distribution(
                main_path, task_name, class_name, stage, np.asarray(grid),
                np.asarray(source_curves), curves, global_shift, ylim,
                max_spaghetti, seed + int(class_id), metrics,
            )
            metadata[stage]["main_path"] = str(main_path)
    return metadata


@torch.inference_mode()
def reconstruct_prototype_fast(coefficients, device, memory_mb=256):
    """Exact pointwise median with resident class coefficients and bounded blocks."""
    count, modes, channels = coefficients.shape
    if count == 0 or modes != NUM_MODES or memory_mb <= 0:
        raise ValueError("nonempty Mode13 coefficients and a positive memory budget are required")
    coefficient_tensor = torch.as_tensor(coefficients, device=device).permute(1,0,2).reshape(modes,-1)
    basis = _shared_basis(np.arange(365), device, coefficient_tensor.dtype)
    # Complex product, real host array and median workspace all count towards budget.
    chunk = max(1, min(365, int(memory_mb*1024**2/(count*channels*24))))
    prototype = np.empty((365, channels), dtype=np.float32)
    for start in range(0,365,chunk):
        stop = min(365,start+chunk)
        values = (basis[start:stop] @ coefficient_tensor).real.reshape(stop-start,count,channels).cpu().numpy().copy()
        prototype[start:stop] = np.median(values, axis=1, overwrite_input=True)
    return prototype


@torch.inference_mode()
def extract_mode13_cache(spatial_encoder, dataset, batch_size, device, with_extra,
                         fit_source_pca=False):
    """Finish the feature/Fourier cache before labels are used for oracle grouping."""
    analyzer = BatchedDirectFourierAnalyzer(NUM_MODES, period_days=PERIOD_DAYS, reg=REG).to(device)
    coefficients, labels, sample_ids, positions = [], [], [], []
    sequential, statistics = 0, {}
    started = time.perf_counter()
    for batch_index, sample in enumerate(_loader(dataset, batch_size)):
        pixels = sample["pixels"].to(device)
        mask = sample["valid_pixels"].to(device)
        extra = sample["extra"].to(device) if with_extra else None
        features = spatial_encoder(pixels, mask, extra)
        pos = sample["positions"].to(device=device, dtype=features.dtype)
        coeff, _ = analyzer(features, pos, collect_diagnostics=False)
        if fit_source_pca:
            raw_features = features.cpu().numpy()
            for feature, position, label in zip(raw_features, sample["positions"].numpy(), sample["label"].numpy()):
                total, cross = interpolation_statistics(feature, position, np.arange(365.))
                key = int(label)
                if key not in statistics:
                    statistics[key] = [0, np.zeros_like(total), np.zeros_like(cross)]
                statistics[key][0] += 365
                statistics[key][1] += total
                statistics[key][2] += cross
        coefficients.append(coeff.cpu().numpy())
        labels.append(sample["label"].numpy().copy())
        positions.extend([np.asarray(row) for row in sample["positions"].numpy()])
        ids = sample.get("parcel_index")
        if ids is None:
            sample_ids.extend(range(sequential, sequential + len(features)))
        else:
            sample_ids.extend(np.asarray(ids).reshape(-1).tolist())
        sequential += len(features)
        if batch_index % 10 == 0:
            print(f"PROGRESS|stage={'source_pca_cache' if fit_source_pca else 'target_cache'}|batches={batch_index+1}|samples={sequential}|seconds={time.perf_counter()-started:.1f}", flush=True)
    if not coefficients:
        raise ValueError("empty dataset: no features available")
    projections = {}
    for key, (count, total, cross) in statistics.items():
        center = total/count
        _, vectors = np.linalg.eigh(cross-count*np.outer(center, center))
        axis = vectors[:, -1]
        if axis[np.argmax(abs(axis))] < 0:
            axis = -axis
        projections[key] = SourceClassPC1(center, axis)
    return {
        "coefficients": np.concatenate(coefficients),
        "labels": np.concatenate(labels),
        "sample_ids": np.asarray(sample_ids),
        "positions": positions,
        "projections": projections,
    }


@torch.inference_mode()
def reconstruct_multivariate_batch(
    coefficients, query_days, synthesizer, device, batch_size=256,
):
    coefficients = np.asarray(coefficients)
    query_days = np.asarray(query_days, dtype=np.float32)
    if coefficients.ndim != 3 or query_days.ndim != 2 or len(coefficients) != len(query_days):
        raise ValueError("batch reconstruction needs coefficients [N,F,D] and query days [N,T]")
    chunks = []
    for start in range(0, len(coefficients), int(batch_size)):
        stop = min(start + int(batch_size), len(coefficients))
        coeff = torch.as_tensor(coefficients[start:stop], device=device)
        days = torch.as_tensor(query_days[start:stop], device=device)
        chunks.append(synthesizer(coeff, days).cpu().numpy())
    if not chunks:
        return np.empty((0, query_days.shape[1], coefficients.shape[2]), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _safe_name(class_id, name):
    return f"class_{class_id:02d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_')}"


def _plot_anchor_reference(path, grid, scalar, anchor, elevation, reason):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(grid, scalar, lw=2.2)
    if anchor is not None:
        ax.axvline(anchor.day, color="tab:red", ls="--")
        ax.axvspan(anchor.day - 30, anchor.day + 30, color="tab:red", alpha=.08)
    ax.set(title=f"Mode13 salient anchor | elevation={elevation:.3f}\n{reason or 'eligible'}",
           xlabel="Source/reference day", ylabel="source-class PC1", xlim=(0, 364))
    ax.grid(alpha=.2); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _plot_sample(path, days, source_pc1, global_pc1, anchor_pc1, nonlinear_pc1, row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, target, title in zip(axes, (global_pc1, anchor_pc1, nonlinear_pc1),
                                 ("A. Global", "B. + Anchor", "C. + Local nonlinear")):
        ax.plot(days, source_pc1, lw=2.1, label="source reference")
        ax.plot(days, target, lw=1.8, label="target")
        ax.axvline(days[len(days)//2], color="tab:red", ls="--", lw=1)
        ax.set_title(title); ax.grid(alpha=.2); ax.set_xlabel("Day")
    axes[0].set_ylabel("source-class PC1"); axes[0].legend(frameon=False)
    fig.suptitle(f"anchor residual={row['anchor_residual_days']:.1f} | corr={row['corr_global']:.3f}/{row['corr_anchor']:.3f}/{row['corr_nonlinear']:.3f}")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _channel_figure(days, source, curves, sample_id, max_channels=8):
    """Compare identical latent channels and scales across A/B/C, plus overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    channels = np.argsort(-np.var(source, axis=0), kind="stable")[:max_channels]
    fig, axes = plt.subplots(len(channels), 4, figsize=(17, 2.1*len(channels)), squeeze=False)
    titles = ("A: global shift", "B: anchor alignment", "C: local nonlinear", "Target A / B / C overlay")
    colors = ("tab:orange", "tab:green", "tab:red")
    for row_index, channel in enumerate(channels):
        all_values = np.concatenate([source[:,channel]]+[curve[:,channel] for curve in curves])
        low, high = float(all_values.min()), float(all_values.max())
        margin = max((high-low)*.08, 1e-6)
        for column, ax in enumerate(axes[row_index]):
            ax.plot(days, source[:,channel], color="black", lw=1.7, label="source prototype")
            for index in (range(3) if column == 3 else (column,)):
                ax.plot(days, curves[index][:,channel], color=colors[index], lw=1.4, label="ABC"[index])
            ax.set_ylim(low-margin, high+margin)
            ax.axvline(days[len(days)//2], color=".6", ls=":")
            ax.grid(alpha=.15)
            if row_index == 0:
                ax.set_title(titles[column]); ax.legend(fontsize=7)
            if column == 0:
                ax.set_ylabel(f"channel {channel}\nPSE feature value")
            if row_index == len(channels)-1:
                ax.set_xlabel("Source/reference calendar day")
    fig.suptitle(f"Sample {sample_id} | channels ranked by source-window variance | same scale across each row")
    fig.tight_layout(rect=(0,0,1,.98))
    return fig, channels


def _plot_feature_examples(directory, stem, days, source, curves):
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    fig, channels = _channel_figure(days, source, curves, stem)
    fig.savefig(directory / f"{stem}_channels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # All channels remain available; a source-only scale is shared across stages.
    center = source.mean(0)
    scale = np.maximum(source.std(0), 1e-6)
    images = [(value-center)/scale for value in (source,)+tuple(curves)]
    limit = max(1., float(np.percentile(abs(np.stack(images)), 98)))
    fig, axes = plt.subplots(1,4,figsize=(17,5),sharex=True,sharey=True,layout="constrained")
    for ax, values, title in zip(axes, images, ("Source", "A: global", "B: anchor", "C: nonlinear")):
        artist = ax.imshow(values.T, origin="lower", aspect="auto", extent=(days[0],days[-1],-.5,source.shape[1]-.5),
                           cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(title); ax.set_xlabel("Source/reference day")
    axes[0].set_ylabel("PSE latent channel")
    fig.colorbar(artist, ax=list(axes), label="Source-window standardized feature (colors clipped at shared 98th percentile)")
    fig.savefig(directory / f"{stem}_heatmap.png", dpi=150)
    plt.close(fig)
    np.savez_compressed(directory / f"{stem}_curves.npz", days=days, source=source,
                        global_curve=curves[0], anchor_curve=curves[1], nonlinear_curve=curves[2],
                        displayed_channels=channels)


def _plot_warp(path, days, anchor_maps, residual, composite):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(days, days, color=".6", ls="--", label="identity")
    ax.plot(days, anchor_maps.query(days), label="Q_B")
    ax.plot(days, residual, label="gamma")
    ax.plot(days, composite, label="Q_C = Q_B o gamma", lw=2)
    ax.set(xlabel="source output day", ylabel="target input day"); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _plot_raw_preview(path, raw_global_days, final_days, a, b, anchor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(raw_global_days, raw_global_days, "o-", ms=3, label="global-only")
    ax.plot(raw_global_days, final_days, "o-", ms=3, label="final forward map")
    ax.axvspan(a, b, color="tab:red", alpha=.07); ax.axvline(anchor, color="tab:red", ls="--")
    ax.set(xlabel="global-aligned target timestamp", ylabel="source/reference timestamp")
    ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _plot_overviews(output_dir, classes, class_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [row["class_name"] for row in class_rows]
    elevations = [row["domain_relative_elevation"] for row in class_rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), 4.5))
    ax.bar(labels, elevations); ax.axhline(.75, color="tab:red", ls="--", label="diagnostic gate")
    ax.tick_params(axis="x", rotation=35); ax.set(ylabel="Domain-relative elevation", title="Mode13 anchor elevation")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output_dir / "anchor_elevation_overview.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), 4.5))
    x = np.arange(len(labels)); width=.25
    for offset, key, label in ((-width, "corr_global_median", "Global"), (0, "corr_anchor_median", "+Anchor"), (width, "corr_nonlinear_median", "+Nonlinear")):
        ax.bar(x + offset, [row[key] for row in class_rows], width, label=label)
    ax.set_xticks(x, labels, rotation=35); ax.set_ylabel("Median multivariate correlation"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output_dir / "task_gain_overview.png", dpi=180); plt.close(fig)


def run_task(args, source_alias, target_alias, overrides):
    source_path, target_path = DOMAINS[source_alias], DOMAINS[target_alias]
    checkpoint, config = resolve_source_checkpoint(args.source_checkpoint_root, source_alias, source_path, args.seed, args.fold, overrides)
    classes = load_classes(checkpoint, source_path, args.data_root)
    source_set, target_set = build_split_datasets(args.data_root, source_path, target_path, classes, args.seed, config)
    shift = read_authoritative_global_shift(args.shift_visualization_root, f"{source_alias}_{target_alias}")
    shift_manifest_path = args.shift_visualization_root / f"{source_alias}_{target_alias}" / "manifest.json"
    shift_manifest = json.loads(shift_manifest_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    spatial = load_raw_spatial_encoder(checkpoint, config, classes, device)
    task_started = time.perf_counter()
    stage_times = {}
    print(f"PROGRESS|task={source_alias}_{target_alias}|stage=source_mode13_cache", flush=True)
    source = extract_mode13_cache(spatial, source_set, args.batch_size, device, bool(config.get("with_extra", False)), fit_source_pca=True)
    projections, latent_dim = source["projections"], source["coefficients"].shape[-1]
    stage_times["source_pca_and_cache_seconds"] = time.perf_counter()-task_started
    target_started = time.perf_counter()
    print(f"PROGRESS|task={source_alias}_{target_alias}|stage=target_mode13_cache", flush=True)
    target = extract_mode13_cache(spatial, target_set, args.batch_size, device, bool(config.get("with_extra", False)))
    stage_times["target_cache_seconds"] = time.perf_counter()-target_started
    # Target true labels enter only after the complete target Mode13 cache above.
    synth = BatchedDirectFourierSynthesizer(NUM_MODES, period_days=PERIOD_DAYS).to(device)
    grid = np.arange(365, dtype=np.float64)
    task_output = args.output_root / f"{source_alias}_{target_alias}"
    output = task_output / "04_oracle_samplewise_anchor_nonlinear"
    output.mkdir(parents=True, exist_ok=True)
    sample_rows, class_rows = [], []
    full_year_outputs = {}
    for class_id, class_name in enumerate(classes):
        if class_id not in projections:
            continue
        print(
            f"PROGRESS|task={source_alias}_{target_alias}|class={class_id}|stage=domain_projection",
            flush=True,
        )
        projection = projections[class_id]
        projection_started = time.perf_counter()
        source_all_scalar = reconstruct_projected_fast(source["coefficients"], projection)
        target_all_scalar = reconstruct_projected_fast(target["coefficients"], projection)
        source_baseline = domain_projection_baseline(source_all_scalar[..., None], np.ones(1), np.zeros(1))
        target_baseline = domain_projection_baseline(target_all_scalar[..., None], np.ones(1), np.zeros(1))
        source_indices = np.flatnonzero(source["labels"] == class_id)
        target_indices = np.flatnonzero(target["labels"] == class_id)
        if not len(source_indices) or not len(target_indices):
            continue
        global_query = np.arange(365, dtype=np.float64) - shift.shift_days
        full_year_queries_by_stage = {
            "global_only": np.broadcast_to(global_query, (len(target_indices), 365)).copy(),
            "anchor_aligned": np.broadcast_to(global_query, (len(target_indices), 365)).copy(),
            "local_nonlinear_aligned": np.broadcast_to(global_query, (len(target_indices), 365)).copy(),
        }
        target_local_row = {int(index): row for row, index in enumerate(target_indices)}
        source_coeff = source["coefficients"][source_indices]
        print(f"PROGRESS|class={class_id}|stage=source_prototype|seconds={time.perf_counter()-projection_started:.1f}", flush=True)
        prototype_started = time.perf_counter()
        source_reference = reconstruct_prototype_fast(source_coeff, device, args.prototype_memory_mb)
        print(f"PROGRESS|class={class_id}|stage=source_prototype_done|seconds={time.perf_counter()-prototype_started:.1f}", flush=True)
        source_scalar = projection.transform(source_reference[None])[0]
        detection = detect_salient_extrema(source_scalar, grid, args.min_extrema_distance_days, args.min_width_days, args.min_normalized_prominence)
        candidate = max(detection.extrema, key=lambda item: item.normalized_prominence, default=None)
        anchor, reason = select_gated_anchor(detection.extrema, source_scalar, grid, source_baseline,
                                             args.min_normalized_prominence, args.min_domain_relative_elevation)
        gated_anchor = anchor
        source_matches = [] if anchor is None else [match_sample_anchor(source_all_scalar[i], grid, anchor, 30, args.min_width_days, args.min_normalized_prominence) for i in source_indices]
        source_stats = summarize_matches(source_matches)
        if anchor is not None and source_stats["occurrence_rate"] < .70:
            anchor, reason = None, "low_source_occurrence"
        if anchor is not None and (not np.isfinite(source_stats["timing_mad"]) or source_stats["timing_mad"] > 20):
            anchor, reason = None, "high_source_timing_mad"
        if anchor is not None and not 30 <= anchor.day <= 334:
            anchor, reason = None, "anchor_window_out_of_support"
        reported_anchor = anchor or gated_anchor or candidate
        elevation = float("nan") if reported_anchor is None else anchor_domain_elevation(source_scalar, grid, reported_anchor, source_baseline)
        class_dir = output / "diagnostics" / _safe_name(class_id, class_name)
        if args.max_examples > 0:
            _plot_anchor_reference(class_dir / "anchor_reference.png", grid, source_scalar, anchor, elevation, reason)
        current = []
        if anchor is not None:
            class_started = time.perf_counter()
            a, b = anchor.day - 30, anchor.day + 30
            days = np.arange(a, b + 1, dtype=np.float64)
            source_window = source_reference[int(a):int(b) + 1]
            source_pc1 = projection.transform(source_window[None])[0]
            eligible = []
            for index in target_indices:
                scalar_curve = target_all_scalar[index]
                match = match_sample_anchor(scalar_curve, grid, anchor, 30, args.min_width_days,
                                            args.min_normalized_prominence, shift.shift_days,
                                            args.min_extrema_distance_days)
                row = {key: np.nan for key in SAMPLE_FIELDS}
                row.update(sample_id=target["sample_ids"][index], true_class=class_id,
                           anchor_found=False, nonlinear_valid=False, fallback_reason="target_anchor_not_found")
                if match.matched:
                    raw_anchor_day = match.day - shift.shift_days
                    target_elevation = (float(np.interp(raw_anchor_day, grid, scalar_curve)) - target_baseline.median) / (target_baseline.iqr + 1e-15)
                    if anchor.kind == "valley":
                        target_elevation = -target_elevation
                    if target_elevation >= args.min_domain_relative_elevation:
                        maps = build_anchor_time_maps(a, b, match.day, anchor.day)
                        support_ok = maps.valid and a - shift.shift_days >= 0 and b - shift.shift_days <= 364
                        if support_ok:
                            row.update(anchor_found=True, target_anchor_day=match.day,
                                anchor_residual_days=anchor.day-match.day,
                                anchor_relative_prominence=match.normalized_prominence,
                                anchor_domain_relative_elevation=target_elevation,
                                anchor_warp_valid=maps.valid, anchor_warp_max_displacement=maps.max_displacement,
                                anchor_warp_left_scale=maps.left_scale, anchor_warp_right_scale=maps.right_scale,
                                anchor_warp_extreme=maps.extreme)
                            eligible.append((index, row, maps))
                            local_row = target_local_row[int(index)]
                            _, anchor_full_query, _ = full_year_queries(
                                grid, shift.shift_days, anchor_maps=maps
                            )
                            full_year_queries_by_stage["anchor_aligned"][local_row] = anchor_full_query
                            full_year_queries_by_stage["local_nonlinear_aligned"][local_row] = anchor_full_query
                        else:
                            row["fallback_reason"] = "insufficient_common_support" if maps.valid else maps.failure_reason
                    else:
                        row["fallback_reason"] = "low_target_domain_relative_elevation"
                if not row["anchor_found"]:
                    sample_rows.append(row); current.append(row)

            if eligible:
                example_indices = set(np.linspace(0,len(eligible)-1,min(args.max_examples,len(eligible)),dtype=int))
                phase_seconds = 0.
                # Memory and latency are bounded by one batch, regardless of class size.
                for batch_start in range(0, len(eligible), args.batch_size):
                    batch_started = time.perf_counter()
                    batch = eligible[batch_start:batch_start+args.batch_size]
                    indices = np.asarray([item[0] for item in batch], dtype=np.int64)
                    coefficients = target["coefficients"][indices]
                    global_queries = np.broadcast_to(days-shift.shift_days, (len(batch),len(days))).copy()
                    anchor_queries = np.stack([item[2].query(days)-shift.shift_days for item in batch])
                    target_global = reconstruct_multivariate_batch(coefficients, global_queries, synth, device, args.batch_size)
                    target_anchor = reconstruct_multivariate_batch(coefficients, anchor_queries, synth, device, args.batch_size)
                    phase_started = time.perf_counter()
                    if args.phase_solver == "batched_monotone":
                        phases = estimate_batched_window_phase(source_window, target_anchor, days, anchor.day,
                                                               device=device, steps=args.phase_steps, segments=args.phase_segments)
                    else:
                        phases = estimate_anchor_fixed_local_phases(source_window, target_anchor, days, anchor.day, args.srvf_workers)
                    phase_seconds += time.perf_counter()-phase_started
                    residuals = [phase.residual_query_days if phase.valid else days for phase in phases]
                    composite_queries = np.stack([compose_local_query(days,item[2],residual) for item,residual in zip(batch,residuals)])
                    target_nonlinear = reconstruct_multivariate_batch(coefficients, composite_queries-shift.shift_days, synth, device, args.batch_size)
                    metrics = [window_metrics_batch(source_window,values) for values in (target_global,target_anchor,target_nonlinear)]
                    # The fitted objective uses interpolated B curves. Verify gain on the
                    # actual Fourier-synthesized C before accepting the resulting map.
                    rejected = np.zeros(len(batch), dtype=bool)
                    if args.phase_solver == "batched_monotone":
                        rejected = ~np.isfinite(metrics[2][0]) | (metrics[2][0] < metrics[1][0]-1e-7)
                        target_nonlinear[rejected] = target_anchor[rejected]
                        for metric_b, metric_c in zip(metrics[1],metrics[2]):
                            metric_c[rejected] = metric_b[rejected]
                    for offset, (item,phase) in enumerate(zip(batch,phases)):
                        index, row, maps = item
                        residual = days if rejected[offset] else residuals[offset]
                        q_composite = compose_local_query(days,maps,residual)
                        curve_a,curve_b,curve_c = target_global[offset],target_anchor[offset],target_nonlinear[offset]
                        corr_a,corr_b,corr_c = (metric[0][offset] for metric in metrics)
                        srvf_a,srvf_b,srvf_c = (metric[1][offset] for metric in metrics)
                        dist_a,dist_b,dist_c = (metric[2][offset] for metric in metrics)
                        raw_global = np.asarray(target["positions"][index],dtype=np.float64)+shift.shift_days
                        raw_final = apply_local_forward_to_timestamps(raw_global,maps,days,residual)
                        local_row = target_local_row[int(index)]
                        if phase.valid and not rejected[offset]:
                            _, _, nonlinear_full_query = full_year_queries(
                                grid, shift.shift_days, maps, days, residual
                            )
                            full_year_queries_by_stage["local_nonlinear_aligned"][local_row] = nonlinear_full_query
                        row.update(
                            nonlinear_valid=phase.valid and not rejected[offset],
                            nonlinear_max_displacement=0. if rejected[offset] else phase.max_displacement_days,
                            nonlinear_min_derivative=1. if rejected[offset] else phase.min_derivative,
                            nonlinear_median_derivative=1. if rejected[offset] else phase.median_derivative,
                            nonlinear_max_derivative=1. if rejected[offset] else phase.max_derivative,
                            nonlinear_extreme=False if rejected[offset] else phase.extreme,
                            corr_global=corr_a,corr_anchor=corr_b,corr_nonlinear=corr_c,
                            srvf_dist_global=srvf_a,srvf_dist_anchor=srvf_b,srvf_dist_nonlinear=srvf_c,
                            local_distance_global=dist_a,local_distance_anchor=dist_b,local_distance_nonlinear=dist_c,
                            anchor_gain=corr_b-corr_a,nonlinear_gain=corr_c-corr_b,
                            anchor_only_sufficient=bool(corr_b>corr_a and corr_c-corr_b<.01),
                            nonlinear_helpful=bool(corr_c-corr_b>=.01),
                            raw_time_mapping_monotone=bool(np.all(np.diff(raw_final)>0)),
                            fallback_reason="fourier_recheck_no_gain" if rejected[offset] else phase.failure_reason,
                            phase_solver=args.phase_solver,
                        )
                        sample_rows.append(row); current.append(row)
                        if batch_start+offset in example_indices:
                            projected = [projection.transform(value[None])[0] for value in (curve_a,curve_b,curve_c)]
                            stem = f"sample_{target['sample_ids'][index]}"
                            _plot_sample(class_dir/"local_alignment_examples"/f"{stem}.png",days,source_pc1,*projected,row)
                            _plot_feature_examples(class_dir/"feature_examples",stem,days,source_window,(curve_a,curve_b,curve_c))
                            _plot_warp(class_dir/"warp_examples"/f"{stem}.png",days,maps,residual,q_composite)
                            if batch_start+offset == 0:
                                _plot_raw_preview(class_dir/"raw_time_mapping_preview.png",raw_global,raw_final,a,b,anchor.day)
                    print(f"PROGRESS|task={source_alias}_{target_alias}|class={class_id}|aligned={min(batch_start+len(batch),len(eligible))}/{len(eligible)}|phase_solver={args.phase_solver}|batch_seconds={time.perf_counter()-batch_started:.1f}|phase_seconds={phase_seconds:.1f}",flush=True)
                print(
                    f"PROGRESS|task={source_alias}_{target_alias}|class={class_id}|"
                    f"eligible={len(eligible)}|phase_seconds={phase_seconds:.1f}|"
                    f"class_seconds={time.perf_counter()-class_started:.1f}",
                    flush=True,
                )
        else:
            for index in target_indices:
                row = {key: np.nan for key in SAMPLE_FIELDS}
                row.update(
                    sample_id=target["sample_ids"][index], true_class=class_id,
                    anchor_found=False, nonlinear_valid=False,
                    raw_time_mapping_monotone=True, fallback_reason=reason,
                )
                sample_rows.append(row); current.append(row)
        target_class_coefficients = target["coefficients"][target_indices]
        full_year_curves = {
            stage: reconstruct_projected_queries(
                target_class_coefficients, projection, queries, device,
                args.batch_size,
            )
            for stage, queries in full_year_queries_by_stage.items()
        }
        raw_record = (
            shift_manifest.get("class_outputs", {})
            .get(str(class_id), {})
            .get("raw", {})
        )
        raw_ylim = raw_record.get("ylim")
        full_year_metadata = render_full_year_alignment_figures(
            output, f"{source_alias}_{target_alias}", class_id, class_name,
            grid, source_all_scalar[source_indices], full_year_curves,
            shift.shift_days, raw_ylim=raw_ylim,
            max_spaghetti=args.max_spaghetti, seed=args.seed,
            main_final_dir=output,
        )
        full_year_outputs[str(class_id)] = full_year_metadata
        successful = [row for row in current if row["anchor_found"]]
        for row in current:
            row["phase_solver"] = args.phase_solver
        nonlinear_valid = [row for row in successful if row["nonlinear_valid"]]
        nonlinear_rate = np.mean([row["nonlinear_helpful"] for row in successful]) if successful else 0
        diagnostic = "GLOBAL_ONLY" if anchor is None or not successful else ("ANCHOR_PLUS_NONLINEAR" if nonlinear_rate >= .5 else "ANCHOR_ONLY")
        class_rows.append({
            "task": f"{source_alias}_{target_alias}", "class_id": class_id, "class_name": class_name,
            "phase_solver": args.phase_solver,
            "full_year_corr_global": full_year_metadata["global_only"]["median_profile_correlation"],
            "full_year_corr_anchor": full_year_metadata["anchor_aligned"]["median_profile_correlation"],
            "full_year_corr_nonlinear": full_year_metadata["local_nonlinear_aligned"]["median_profile_correlation"],
            "full_year_l2_global": full_year_metadata["global_only"]["median_profile_normalized_l2"],
            "full_year_l2_anchor": full_year_metadata["anchor_aligned"]["median_profile_normalized_l2"],
            "full_year_l2_nonlinear": full_year_metadata["local_nonlinear_aligned"]["median_profile_normalized_l2"],
            "anchor_type": "" if anchor is None else anchor.kind, "anchor_day": np.nan if anchor is None else anchor.day,
            "relative_prominence": np.nan if reported_anchor is None else reported_anchor.normalized_prominence,
            "domain_relative_elevation": elevation, "source_occurrence_rate": source_stats["occurrence_rate"],
            "source_timing_mad": source_stats["timing_mad"], "num_target_samples": len(target_indices),
            "num_target_anchor_found": len(successful), "target_anchor_found_rate": len(successful)/len(target_indices) if len(target_indices) else np.nan,
            "anchor_residual_abs_median": finite_median(abs(row["anchor_residual_days"]) for row in successful),
            "corr_global_median": finite_median(row["corr_global"] for row in successful),
            "corr_anchor_median": finite_median(row["corr_anchor"] for row in successful),
            "corr_nonlinear_median": finite_median(row["corr_nonlinear"] for row in successful),
            "anchor_gain_median": finite_median(row["anchor_gain"] for row in successful),
            "nonlinear_gain_median": finite_median(row["nonlinear_gain"] for row in successful),
            "srvf_distance_global_median": finite_median(row["srvf_dist_global"] for row in successful),
            "srvf_distance_anchor_median": finite_median(row["srvf_dist_anchor"] for row in successful),
            "srvf_distance_nonlinear_median": finite_median(row["srvf_dist_nonlinear"] for row in successful),
            "max_nonlinear_displacement_median": finite_median(row["nonlinear_max_displacement"] for row in successful),
            "nonlinear_extreme_rate": np.mean([row["nonlinear_extreme"] for row in nonlinear_valid]) if nonlinear_valid else np.nan,
            "anchor_only_sufficient_rate": np.mean([row["anchor_only_sufficient"] for row in successful]) if successful else np.nan,
            "nonlinear_helpful_rate": nonlinear_rate, "diagnostic_class": diagnostic, "fallback_reason": reason,
        })
        write_csv(output / "sample_summary.csv", sample_rows, SAMPLE_FIELDS)
        write_csv(output / "class_summary.csv", class_rows, tuple(class_rows[0]))
    write_csv(output / "sample_summary.csv", sample_rows, SAMPLE_FIELDS)
    write_csv(output / "class_summary.csv", class_rows, tuple(class_rows[0]) if class_rows else ())
    task_row = {"task": f"{source_alias}_{target_alias}", "num_classes": len(class_rows),
                "num_samples": len(sample_rows), "nonlinear_helpful_rate": finite_median(row["nonlinear_helpful_rate"] for row in class_rows)}
    write_csv(output / "task_summary.csv", [task_row], tuple(task_row))
    if args.max_examples > 0:
        _plot_overviews(output / "diagnostics", classes, class_rows)
    manifest = {
        "audit_type": "offline_oracle_mode13_anchor_fixed_local_phase",
        "training": False, "parameter_update": False,
        "per_sample_time_map_fitting": True, "model_parameter_update": False,
        "target_label_usage": "oracle_offline_grouping_only",
        "oracle_assumption": "target true labels simulate 100-percent-correct pseudo labels",
        "task": f"{source_alias}_{target_alias}", "source": source_path, "target": target_path,
        "source_checkpoint": str(checkpoint), "seed": args.seed, "fold": args.fold,
        "fourier": {"num_modes": NUM_MODES, "period_days": PERIOD_DAYS, "reg": REG, "solver": "dense_direct"},
        "global_shift_days": shift.shift_days, "global_shift_source": shift.source_path,
        "source_anchor_gate": {"relative_prominence": args.min_normalized_prominence,
                               "domain_relative_elevation": args.min_domain_relative_elevation,
                               "occurrence_rate": .70, "timing_mad_days": 20},
        "local_window_radius_days": 30,
        "phase_solver": args.phase_solver,
        "phase_objective": "mean_channel_pearson_correlation" if args.phase_solver == "batched_monotone" else "multivariate_srvf_dp",
        "phase_steps": args.phase_steps if args.phase_solver == "batched_monotone" else None,
        "phase_segments_per_half": args.phase_segments if args.phase_solver == "batched_monotone" else None,
        "minimum_relative_segment_duration": .05 if args.phase_solver == "batched_monotone" else None,
        "srvf_lambda": SRVF_LAMBDA if args.phase_solver == "srvf_dp" else None,
        "example_selection": "uniformly_spaced_eligible_sample_indices",
        "registration_space": "multivariate_Mode13",
        "visualization_space": "source_class_PC1_and_individual_latent_channels",
        "primary_visualization": "shift_visualization-compatible full-year class distributions",
        "primary_output_directory": str(output),
        "full_year_class_outputs": full_year_outputs,
        "timing_seconds": dict(stage_times,total=time.perf_counter()-task_started),
        "query_map_contract": "source/reference output day -> target input day",
        "composition": "Q_C = Q_B o gamma; F_C = inverse(Q_C) = inverse(gamma) o F_B",
        "outside_window": "strict identity", "padding": False, "circular_wrap": False,
        "latent_dim": latent_dim, "source_count": len(source_set), "target_count": len(target_set),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[FINISHED] {source_alias}_{target_alias}: {output}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--source-checkpoint-root", type=Path, default=Path("outputs"))
    parser.add_argument("--source-checkpoint", action="append", default=[])
    parser.add_argument("--shift-visualization-root", type=Path, default=Path("outputs/fredn/shift_visualizations_seed1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fredn/shift_visualizations_seed1"))
    parser.add_argument("--source-domain", choices=DOMAINS); parser.add_argument("--target-domain", choices=DOMAINS)
    parser.add_argument("--seed", type=int, default=1); parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--device", default="cuda")
    parser.add_argument("--srvf-workers", type=int, default=2)
    parser.add_argument("--phase-solver", choices=("batched_monotone", "srvf_dp"), default="batched_monotone")
    parser.add_argument("--phase-steps", type=int, default=80)
    parser.add_argument("--phase-segments", type=int, default=8)
    parser.add_argument("--prototype-memory-mb", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--min-extrema-distance-days", type=float, default=7)
    parser.add_argument("--min-width-days", type=float, default=3)
    parser.add_argument("--min-normalized-prominence", type=float, default=.20)
    parser.add_argument("--min-domain-relative-elevation", type=float, default=.75)
    parser.add_argument("--max-spaghetti", type=int, default=40)
    parser.add_argument("--max-examples", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fold != 0: raise SystemExit("ERROR: this diagnostic reproduces fold 0 only")
    if args.srvf_workers < 1: raise SystemExit("ERROR: --srvf-workers must be positive")
    if min(args.batch_size,args.phase_steps,args.prototype_memory_mb,args.cpu_threads) < 1 or args.phase_segments < 2 or args.max_examples < 0:
        raise SystemExit("ERROR: invalid batch/phase/memory/thread/example configuration")
    torch.set_num_threads(args.cpu_threads)
    if args.phase_solver == "srvf_dp":
        try:
            import fdasrsf  # Explicit reference backend: never download dependencies.
        except ImportError as error:
            raise SystemExit(f"ERROR: --phase-solver srvf_dp requires local fdasrsf: {error}")
    if (args.source_domain is None) != (args.target_domain is None): raise SystemExit("ERROR: source/target must be supplied together")
    for path, name in ((args.data_root, "data root"), (args.source_checkpoint_root, "checkpoint root"), (args.shift_visualization_root, "shift root")):
        if not path.is_dir(): raise SystemExit(f"ERROR: {name} not found: {path}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    overrides = _key_values(args.source_checkpoint)
    tasks = ((args.source_domain, args.target_domain),) if args.source_domain else TASKS
    for source, target in tasks: run_task(args, source, target, overrides)


if __name__ == "__main__":
    main()
