#!/usr/bin/env python3
"""Render offline Raw-PSE shift comparisons from local artifacts."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.shift_visualization import (
    SourceClassPC1,
    build_samplewise_time_mapped_curves,
    estimate_class_residual_shift,
    interpolate_latent,
    read_best_validation_shift,
    render_class_residual_figure,
    render_local_nonlinear_figure,
    render_task_class_figures,
    replay_train_indices,
    update_class_residual_outputs,
    update_local_nonlinear_outputs,
    update_multi_event_outputs,
    validate_existing_task_outputs,
    write_task_outputs,
)
from analysis.recon_event_diagnostic import (
    detect_circular_events,
    evaluate_source_event_stability,
    greedy_match_events,
)
from analysis.recon_anchor_diagnostic import (
    Extremum,
    anchor_domain_elevation,
    apply_whole_window_forward_map,
    build_joint_anchor_window,
    detect_salient_extrema,
    domain_projection_baseline,
    estimate_whole_window_local_phase,
    local_multivariate_correlation,
    match_sample_anchor,
    sample_curve_with_query,
    select_gated_anchor,
    summarize_matches,
)
from models.fourier_reconstruction import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
)
from models.stclassifier import PseLTae


DOMAINS = {
    "AT1": "austria/33UVP/2017",
    "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
}
TASKS = (("AT1", "DK1"), ("DK1", "FR1"), ("FR1", "FR2"), ("FR2", "AT1"))


def _key_values(values):
    result = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"expected KEY=PATH, got {value!r}")
        key, path = value.split("=", 1)
        result[key.upper()] = Path(path)
    return result


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_source_checkpoint(root, alias, source_path, seed, fold, overrides):
    if alias in overrides:
        candidate = overrides[alias]
        checkpoint = candidate if candidate.name == "model.pt" else candidate / f"fold_{fold}" / "model.pt"
        configs = [checkpoint.parent.parent / "train_config.json"]
    else:
        configs = sorted(Path(root).rglob("train_config.json"))
        matches = []
        for config_path in configs:
            try:
                config = _read_json(config_path)
            except (OSError, ValueError):
                continue
            if (
                config.get("model") == "pseltae"
                and config.get("source") == source_path
                and config.get("target") == source_path
                and int(config.get("seed", -1)) == seed
            ):
                checkpoint = config_path.parent / f"fold_{fold}" / "model.pt"
                if checkpoint.is_file():
                    score = 10 if config_path.parent.name == f"pseltae_{alias}_source_seed{seed}" else 0
                    matches.append((score, config_path, checkpoint))
        if not matches:
            raise FileNotFoundError(f"no Raw PseLTae source checkpoint for {alias} below {root}")
        matches.sort(key=lambda item: (-item[0], len(str(item[2])), str(item[2])))
        if len(matches) > 1 and matches[0][0] == matches[1][0]:
            paths = "\n".join(str(item[2]) for item in matches)
            raise ValueError(f"ambiguous source checkpoint for {alias}; use --source-checkpoint {alias}=PATH:\n{paths}")
        _, config_path, checkpoint = matches[0]
        configs = [config_path]
    checkpoint = Path(checkpoint)
    config_path = Path(configs[0])
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"checkpoint/config missing: {checkpoint}, {config_path}")
    config = _read_json(config_path)
    expected = {"model": "pseltae", "source": source_path, "target": source_path, "seed": seed}
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if checkpoint.parent.name != f"fold_{fold}":
        mismatch["fold"] = (checkpoint.parent.name, f"fold_{fold}")
    if mismatch:
        raise ValueError(f"source checkpoint metadata mismatch: {mismatch}")
    return checkpoint, config


def resolve_task_log(root, source_alias, target_alias, kind, overrides):
    key = f"{source_alias}_{target_alias}"
    if key in overrides:
        path = overrides[key]
        if not path.is_file():
            raise FileNotFoundError(f"configured {kind} log missing: {path}")
        return path
    candidates = []
    pair = key.lower()
    for path in Path(root).rglob("*.log"):
        name = str(path).lower().replace("to_", "").replace("->", "_")
        score = 0
        if pair in name:
            score += 30
        if source_alias.lower() in name and target_alias.lower() in name:
            score += 10
        if kind == "timematch" and "original" in name:
            score += 10
        if kind == "reconshift13" and "reconshift13" in name:
            score += 10
        if score:
            candidates.append((score, len(str(path)), str(path), path))
    if not candidates:
        raise FileNotFoundError(f"no {kind} log for {key} below {root}")
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
        paths = "\n".join(str(item[3]) for item in candidates)
        raise ValueError(f"ambiguous {kind} log for {key}; use --{kind}-log {key}=PATH:\n{paths}")
    return candidates[0][3]


def resolve_task_output(root, source_path, target_path, seed, kind):
    matches = []
    for config_path in Path(root).rglob("train_config.json"):
        try:
            config = _read_json(config_path)
        except (OSError, ValueError):
            continue
        if (
            config.get("model") != "pseltae"
            or config.get("source") != source_path
            or config.get("target") != target_path
            or int(config.get("seed", -1)) != seed
            or config.get("method") != "timematch"
        ):
            continue
        view = config.get("shift_estimation_view", "raw")
        is_recon = view == "fourier_recon"
        if (kind == "reconshift13") != is_recon:
            continue
        matches.append(config_path.parent)
    return matches[0] if len(matches) == 1 else None


def load_classes(checkpoint, source_path, data_root):
    protocol_path = checkpoint.parent.parent / "closed_set_protocol.json"
    if protocol_path.is_file():
        classes = _read_json(protocol_path).get("classes")
        if classes:
            return list(classes)
    from dataset import PixelSetData
    from utils import label_utils

    candidates = [name for name in label_utils.get_classes(source_path.split("/")[0]) if name != "unknown"]
    dataset = PixelSetData(data_root, source_path, candidates, closed_set=True)
    labels, counts = np.unique(dataset.get_labels(), return_counts=True)
    return [candidates[int(label)] for label, count in zip(labels, counts) if count >= 200]


def load_raw_spatial_encoder(checkpoint, config, classes, device):
    model = PseLTae(
        input_dim=int(config.get("input_dim", 10)),
        with_extra=bool(config.get("with_extra", False)),
        num_classes=len(classes),
    )
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    state = payload.get("state_dict", payload)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model.spatial_encoder.to(device).eval().requires_grad_(False)


def build_split_datasets(data_root, source_path, target_path, classes, seed, config):
    from dataset import PixelSetData
    from torchvision.transforms import transforms
    from transforms import Normalize, ToTensor

    base_source = PixelSetData(data_root, source_path, classes, closed_set=True)
    base_target = PixelSetData(data_root, target_path, classes, closed_set=True)
    eligible = {
        source_path: base_source.get_parcel_indices().tolist(),
        target_path: base_target.get_parcel_indices().tolist(),
    }
    splits = replay_train_indices(
        (source_path, target_path),
        eligible,
        seed,
        float(config.get("val_ratio", 0.1)),
        float(config.get("test_ratio", 0.2)),
    )
    transform = transforms.Compose([Normalize(), ToTensor()])
    common = dict(
        data_root=data_root,
        classes=classes,
        transform=transform,
        with_extra=bool(config.get("with_extra", False)),
        closed_set=True,
        combine_spring_and_winter=False,
    )
    source = PixelSetData(dataset_name=source_path, indices=splits[source_path]["train"], **common)
    target = PixelSetData(dataset_name=target_path, indices=splits[target_path]["train"], **common)
    return source, target


def _loader(dataset, batch_size):
    from dataset import GroupByShapesBatchSampler

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_sampler=GroupByShapesBatchSampler(dataset, batch_size),
    )


@torch.inference_mode()
def fit_class_projections(spatial_encoder, dataset, class_count, grid, batch_size, device, with_extra):
    statistics = {
        class_id: {"count": 0, "sum": None, "cross": None}
        for class_id in range(class_count)
    }
    latent_dim = None
    for sample in _loader(dataset, batch_size):
        pixels = sample["pixels"].to(device)
        mask = sample["valid_pixels"].to(device)
        extra = sample["extra"].to(device) if with_extra else None
        features = spatial_encoder(pixels, mask, extra).cpu().numpy()
        positions = sample["positions"].numpy()
        labels = sample["label"].numpy()
        latent_dim = features.shape[-1]
        for feature, position, label in zip(features, positions, labels):
            dense = interpolate_latent(position, feature, grid)
            item = statistics[int(label)]
            item["count"] += dense.shape[0]
            value_sum = dense.sum(axis=0)
            cross = dense.T @ dense
            item["sum"] = value_sum if item["sum"] is None else item["sum"] + value_sum
            item["cross"] = cross if item["cross"] is None else item["cross"] + cross
    projections = {}
    for class_id, item in statistics.items():
        if item["count"] == 0:
            continue
        center = item["sum"] / item["count"]
        covariance = item["cross"] - item["count"] * np.outer(center, center)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0:
            axis = -axis
        projections[class_id] = SourceClassPC1(center, axis)
    return projections, latent_dim


@torch.inference_mode()
def project_dataset(
    spatial_encoder,
    dataset,
    projections,
    grid,
    batch_size,
    device,
    with_extra,
    fourier_analyzer=None,
    collect_samplewise=False,
):
    curves = defaultdict(list)
    coefficients = defaultdict(list)
    sample_records = defaultdict(list)
    domain_values = defaultdict(list)
    sequential_id = 0
    for sample in _loader(dataset, batch_size):
        pixels = sample["pixels"].to(device)
        mask = sample["valid_pixels"].to(device)
        extra = sample["extra"].to(device) if with_extra else None
        feature_tensor = spatial_encoder(pixels, mask, extra)
        position_tensor = sample["positions"].to(
            device=device, dtype=feature_tensor.dtype
        )
        coefficient_tensor = None
        if fourier_analyzer is not None:
            coefficient_tensor, _ = fourier_analyzer(
                feature_tensor, position_tensor, collect_diagnostics=False
            )
        features = feature_tensor.cpu().numpy()
        positions = position_tensor.cpu().numpy()
        labels = sample["label"].numpy()
        coefficient_values = (
            None if coefficient_tensor is None else coefficient_tensor.cpu().numpy()
        )
        if collect_samplewise:
            for projected_class, projection in projections.items():
                values = np.einsum(
                    "bld,d->bl", features, projection.axis, optimize=True
                ) - float(projection.center @ projection.axis)
                domain_values[int(projected_class)].append(values.reshape(-1))
        parcel_ids = sample.get("parcel_index")
        if parcel_ids is None:
            parcel_ids = np.arange(sequential_id, sequential_id + len(features))
        else:
            parcel_ids = np.asarray(parcel_ids).reshape(-1)
        for offset, (feature, position, label) in enumerate(
            zip(features, positions, labels)
        ):
            class_id = int(label)
            if class_id not in projections:
                continue
            dense = interpolate_latent(position, feature, grid)
            curves[class_id].append(projections[class_id].transform(dense[None])[0])
            if collect_samplewise:
                projection = projections[class_id]
                raw_values = (
                    np.asarray(feature, dtype=np.float64) @ projection.axis
                    - float(projection.center @ projection.axis)
                )
                sample_records[class_id].append(
                    {
                        "sample_id": int(parcel_ids[offset]),
                        "positions": np.asarray(position, dtype=np.float64).copy(),
                        "raw_pc1": raw_values.copy(),
                        "coefficients": coefficient_values[offset].copy(),
                    }
                )
        if coefficient_tensor is not None:
            for class_id in np.unique(labels):
                class_id = int(class_id)
                if class_id in projections:
                    coefficients[class_id].append(
                        coefficient_values[labels == class_id]
                    )
        sequential_id += len(features)
    projected = {class_id: np.stack(values) for class_id, values in curves.items()}
    if fourier_analyzer is None:
        return projected
    coefficient_groups = {
        class_id: np.concatenate(values, axis=0)
        for class_id, values in coefficients.items()
    }
    if not collect_samplewise:
        return projected, coefficient_groups
    baselines = {
        class_id: domain_projection_baseline(
            np.concatenate(values)[:, None, None], np.ones(1), np.zeros(1)
        )
        for class_id, values in domain_values.items()
    }
    return projected, coefficient_groups, dict(sample_records), baselines


@torch.inference_mode()
def reconstruct_class_prototype(
    coefficients,
    synthesizer,
    device,
    sample_batch_size=256,
    day_chunk_size=16,
):
    """Pointwise median Recon13 prototype on the non-wrapped day 0..364 grid."""
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 3 or coefficients.shape[0] == 0:
        raise ValueError("coefficients must be non-empty [N,F,D]")
    count, _, latent_dim = coefficients.shape
    prototype = np.empty((365, latent_dim), dtype=np.float32)
    for day_start in range(0, 365, day_chunk_size):
        days = np.arange(
            day_start, min(day_start + day_chunk_size, 365), dtype=np.float32
        )
        reconstructed = np.empty((count, len(days), latent_dim), dtype=np.float32)
        for sample_start in range(0, count, sample_batch_size):
            sample_stop = min(sample_start + sample_batch_size, count)
            coefficient_batch = torch.as_tensor(
                coefficients[sample_start:sample_stop], device=device
            )
            positions = torch.as_tensor(days, device=device).unsqueeze(0).expand(
                sample_stop - sample_start, -1
            )
            reconstructed[sample_start:sample_stop] = (
                synthesizer(coefficient_batch, positions).cpu().numpy()
            )
        prototype[day_start : day_start + len(days)] = np.median(
            reconstructed, axis=0
        )
    return prototype


def reconstruct_projected_coefficients(coefficients, projection, query_days):
    """Evaluate Mode13 coefficients in one source-class PC1 without materializing D."""
    coefficients = np.asarray(coefficients)
    query = np.asarray(query_days, dtype=np.float64)
    modes = np.arange(-6, 7, dtype=np.float64)
    points = (2.0 * np.pi * query / 365.0 + np.pi) % (2.0 * np.pi) - np.pi
    basis = np.exp(1j * points[:, None] * modes)
    projected_coefficients = np.einsum(
        "nfd,d->nf", coefficients, projection.axis, optimize=True
    )
    result = np.empty((len(coefficients), len(query)), dtype=np.float64)
    for start in range(0, len(coefficients), 2048):
        stop = min(start + 2048, len(coefficients))
        result[start:stop] = (
            projected_coefficients[start:stop] @ basis.T
        ).real - float(projection.center @ projection.axis)
    return result


@torch.inference_mode()
def reconstruct_one_window(coefficients, raw_query_days, synthesizer, device):
    coefficient = torch.as_tensor(coefficients, device=device).unsqueeze(0)
    positions = torch.as_tensor(
        raw_query_days, device=device, dtype=coefficient.real.dtype
    ).unsqueeze(0)
    return synthesizer(coefficient, positions)[0].cpu().numpy().astype(np.float64)


@torch.inference_mode()
def reconstruct_windows_batched(
    coefficients,
    raw_query_days,
    synthesizer,
    device,
    batch_size=256,
):
    """Reconstruct sample-specific grids without one GPU launch per sample."""
    coefficients = np.asarray(coefficients)
    queries = np.asarray(raw_query_days, dtype=np.float64)
    if coefficients.ndim != 3 or queries.ndim != 2:
        raise ValueError("batched reconstruction requires [N,F,D] and [N,T]")
    if len(coefficients) != len(queries):
        raise ValueError("coefficient/query batch sizes must match")
    if not len(coefficients):
        return np.empty((0, queries.shape[1], coefficients.shape[2]), dtype=np.float64)
    reconstructed = np.empty(
        (len(coefficients), queries.shape[1], coefficients.shape[2]),
        dtype=np.float64,
    )
    for start in range(0, len(coefficients), int(batch_size)):
        stop = min(start + int(batch_size), len(coefficients))
        coefficient_batch = torch.as_tensor(coefficients[start:stop], device=device)
        position_batch = torch.as_tensor(
            queries[start:stop],
            device=device,
            dtype=coefficient_batch.real.dtype,
        )
        reconstructed[start:stop] = (
            synthesizer(coefficient_batch, position_batch).cpu().numpy()
        )
    return reconstructed


def _class_observed_support(records):
    starts = [float(np.min(item["positions"])) for item in records]
    ends = [float(np.max(item["positions"])) for item in records]
    if not starts:
        return (float("nan"), float("nan"))
    return max(starts), min(ends)


def _finite_median(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _plot_local_nonlinear_diagnostic(
    output_path,
    class_name,
    grid,
    source_prototype_pc1,
    source_anchor,
    before_segments,
    after_segments,
    target_anchor_days,
    mapped_anchor_days,
    windows,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    before_color, after_color = "#F28E2B", "#59A14F"
    for days, values in before_segments[:40]:
        axes[0].plot(days, values, color=before_color, alpha=0.15, lw=0.7)
    for days, values in after_segments[:40]:
        axes[1].plot(days, values, color=after_color, alpha=0.15, lw=0.7)
    for axis in axes:
        axis.plot(
            grid,
            source_prototype_pc1,
            color="#1F4E79",
            lw=2.2,
            label="source prototype",
        )
        axis.axvline(source_anchor.day, color="#1F4E79", ls="--", lw=1.2)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Day in source temporal frame")
    if before_segments:
        all_before = np.stack(
            [
                np.interp(grid, days, values, left=np.nan, right=np.nan)
                for days, values in before_segments
            ]
        )
        axes[0].plot(
            grid,
            np.nanmedian(all_before, axis=0),
            color="#B85C00",
            lw=2.0,
            label="target median",
        )
    if after_segments:
        all_after = np.stack(
            [
                np.interp(grid, days, values, left=np.nan, right=np.nan)
                for days, values in after_segments
            ]
        )
        axes[1].plot(
            grid,
            np.nanmedian(all_after, axis=0),
            color="#2F7D32",
            lw=2.0,
            label="aligned median",
        )
    if target_anchor_days:
        axes[0].scatter(
            target_anchor_days,
            np.interp(target_anchor_days, grid, source_prototype_pc1),
            s=14,
            alpha=0.5,
            color=before_color,
            label="target anchors",
        )
    if mapped_anchor_days:
        axes[1].scatter(
            mapped_anchor_days,
            np.interp(mapped_anchor_days, grid, source_prototype_pc1),
            s=14,
            alpha=0.5,
            color=after_color,
            label="mapped anchors",
        )
    if windows:
        low = min(item[0] for item in windows)
        high = max(item[1] for item in windows)
        for axis in axes:
            axis.axvspan(low, high, color="grey", alpha=0.07, label="window envelope")
            axis.set_xlim(max(0, low - 15), min(365, high + 15))
    axes[0].set_title("Global Recon13")
    axes[1].set_title("After whole-window nonlinear")
    axes[0].set_ylabel("Source-class PC1 score")
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"{class_name} | salient-anchor local nonlinear diagnostic")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _event_detector_config(args):
    return {
        "min_distance_days": args.event_min_distance_days,
        "min_width_days": args.event_min_width_days,
        "min_relative_prominence": args.event_min_relative_prominence,
        "min_domain_prominence": args.event_min_domain_prominence,
        "min_domain_elevation": args.event_min_domain_elevation,
        "source_occurrence_radius_days": args.event_source_occurrence_radius_days,
        "min_source_occurrence": args.event_min_source_occurrence,
        "max_source_timing_mad_days": args.event_max_source_timing_mad_days,
        "match_radius_days": args.event_match_radius_days,
    }


def _event_detection_kwargs(config):
    return {
        "min_distance_days": config["min_distance_days"],
        "min_width_days": config["min_width_days"],
        "min_relative_prominence": config["min_relative_prominence"],
        "min_domain_prominence": config["min_domain_prominence"],
        "min_domain_elevation": config["min_domain_elevation"],
    }


def _event_row(task, class_id, class_name, event, source=True, sample_id=""):
    row = {
        "task": task,
        "class_id" if source else "true_class": class_id,
        "class_name": class_name,
        "event_id": event.event_id,
        "kind": event.kind,
        "day" if source else "day_global": event.day,
        "value": event.value,
        "prominence": event.prominence,
        "relative_prominence": event.relative_prominence,
        "domain_prominence": event.domain_relative_prominence,
        "domain_elevation": event.domain_relative_elevation,
        "width_days": event.width_days,
        "left_base_day": event.left_base_day,
        "right_base_day": event.right_base_day,
        "accepted": event.accepted,
        "rejection_reason": event.rejection_reason,
        "crosses_year_boundary": event.boundary_crossing,
    }
    if source:
        row.update(
            source_occurrence_rate=event.source_occurrence_rate,
            source_timing_mad_days=event.source_timing_mad_days,
        )
    else:
        row["sample_id"] = sample_id
    return row


def _match_row(task, class_id, class_name, sample_id, match):
    return {
        "task": task,
        "class_id": class_id,
        "class_name": class_name,
        "sample_id": sample_id,
        "source_event_id": match.source_event_id,
        "source_kind": match.source_kind,
        "source_day": match.source_day,
        "target_event_id": match.target_event_id,
        "target_kind": match.target_kind,
        "target_day": match.target_day,
        "linear_day_distance": match.linear_day_distance,
        "circular_day_distance": match.circular_day_distance,
        "width_ratio": match.width_ratio,
        "relative_prominence_difference": match.relative_prominence_difference,
        "domain_elevation_difference": match.domain_elevation_difference,
        "match_status": match.match_status,
        "boundary_crossing_candidate": match.boundary_crossing_candidate,
    }


def _boundary_view(curve):
    values = np.asarray(curve)
    return np.arange(300.0, 425.0), np.concatenate((values[300:365], values[:60]))


def _plot_multi_event_class(
    output_path,
    detail_path,
    class_name,
    source_curve,
    target_curves,
    source_events,
    target_events_by_sample,
    matches_by_sample,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = np.arange(365.0)
    target_median = np.median(np.asarray(target_curves), axis=0)
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharey=True)
    axes[0].plot(days, source_curve, color="#1F4E79", lw=2.3, label="source prototype")
    axes[0].plot(days, target_median, color="#B85C00", lw=2.0, label="target median")
    for event in source_events:
        marker = "^" if event.kind == "peak" else "v"
        if event.accepted:
            axes[0].scatter(event.day, event.value, marker=marker, color="#1F4E79", s=55)
            axes[0].annotate(event.event_id, (event.day, event.value), xytext=(3, 5), textcoords="offset points")
        else:
            axes[0].scatter(event.day, event.value, marker="x", color="#777777", s=36)
    for kind, marker in (("peak", "^"), ("valley", "v")):
        event_days = np.asarray(
            [
                event.day
                for sample_events in target_events_by_sample
                for event in sample_events
                if event.accepted and event.kind == kind
            ],
            dtype=np.float64,
        )
        if event_days.size:
            axes[0].scatter(
                event_days,
                np.interp(event_days, days, target_median),
                marker=marker,
                color="#F28E2B",
                alpha=0.18,
                s=18,
            )
    displayed = 0
    for sample_matches in matches_by_sample:
        for match in sample_matches:
            if match.match_status.startswith("MATCHED") and displayed < 30:
                axes[0].plot(
                    [match.source_day, match.target_day],
                    [np.interp(match.source_day, days, source_curve), np.interp(match.target_day, days, target_median)],
                    color="#777777",
                    ls="--",
                    lw=0.6,
                    alpha=0.25,
                )
                axes[0].annotate(
                    match.target_event_id,
                    (match.target_day, np.interp(match.target_day, days, target_median)),
                    xytext=(3, -10),
                    textcoords="offset points",
                    fontsize=7,
                    color="#B85C00",
                )
                displayed += 1
    axes[0].set_xlim(0, 364)
    axes[0].set_title("Full-year global-aligned Recon13")
    axes[0].set_xlabel("Day in global source calendar")
    axes[0].set_ylabel("Source-class PC1 score")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    boundary_days, source_boundary = _boundary_view(source_curve)
    _, target_boundary = _boundary_view(target_median)
    axes[1].plot(boundary_days, source_boundary, color="#1F4E79", lw=2.3)
    axes[1].plot(boundary_days, target_boundary, color="#B85C00", lw=2.0)
    axes[1].axvline(365, color="#555555", ls=":", lw=1.1, label="year boundary")
    for event in source_events:
        shown_day = event.day + 365 if event.day < 60 else event.day
        if 300 <= shown_day <= 425:
            axes[1].scatter(
                shown_day,
                np.interp(event.day, days, source_curve),
                marker=("^" if event.kind == "peak" else "v") if event.accepted else "x",
                color="#1F4E79" if event.accepted else "#777777",
                s=48,
            )
            axes[1].annotate(event.event_id, (shown_day, np.interp(event.day, days, source_curve)), xytext=(3, 5), textcoords="offset points")
    for kind, marker in (("peak", "^"), ("valley", "v")):
        shown_target_days = np.asarray(
            [
                event.day + 365 if event.day < 60 else event.day
                for events in target_events_by_sample
                for event in events
                if event.accepted
                and event.kind == kind
                and (event.day >= 300 or event.day < 60)
            ],
            dtype=np.float64,
        )
        if shown_target_days.size:
            raw_days = np.mod(shown_target_days, 365.0)
            axes[1].scatter(
                shown_target_days,
                np.interp(raw_days, days, target_median),
                marker=marker,
                color="#F28E2B",
                alpha=0.22,
                s=20,
            )
    displayed_boundary = 0
    for matches in matches_by_sample:
        for match in matches:
            if not match.boundary_crossing_candidate or displayed_boundary >= 30:
                continue
            source_day = match.source_day + 365 if match.source_day < 60 else match.source_day
            target_day = match.target_day + 365 if match.target_day < 60 else match.target_day
            axes[1].plot(
                [source_day, target_day],
                [
                    np.interp(match.source_day, days, source_curve),
                    np.interp(match.target_day, days, target_median),
                ],
                color="#777777",
                ls="--",
                lw=0.7,
                alpha=0.35,
            )
            displayed_boundary += 1
    axes[1].set_xlim(300, 425)
    axes[1].set_xticks([300, 330, 360, 365, 380, 395, 425], ["300", "330", "360", "365", "Jan+15", "Jan+30", "Jan+60"])
    axes[1].set_title("Circular year-boundary view")
    axes[1].set_xlabel("Unwrapped calendar day")
    axes[1].set_ylabel("Source-class PC1 score")
    axes[1].grid(alpha=0.2)
    figure.suptitle(f"{class_name} | Mode13 multi-event structural capture")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    detail, axis = plt.subplots(figsize=(13, 5.5))
    axis.plot(days, source_curve, color="#1F4E79", lw=2)
    for event in source_events:
        axis.scatter(event.day, event.value, marker="o" if event.accepted else "x", color="#2F7D32" if event.accepted else "#B22222")
        label = (
            f"{event.event_id} {event.kind} d={event.day:.0f}\n"
            f"P_rel={event.relative_prominence:.2f} P_dom={event.domain_relative_prominence:.2f} "
            f"E_dom={event.domain_relative_elevation:.2f} w={event.width_days:.1f}\n"
            f"occ={event.source_occurrence_rate:.2f} MAD={event.source_timing_mad_days:.1f} "
            f"{'accepted' if event.accepted else event.rejection_reason}"
        )
        axis.annotate(label, (event.day, event.value), xytext=(5, 8), textcoords="offset points", fontsize=7)
    axis.set_xlim(0, 364)
    axis.set_title(f"{class_name} | source event acceptance audit")
    axis.set_xlabel("Day")
    axis.set_ylabel("Source-class PC1 score")
    axis.grid(alpha=0.2)
    detail.tight_layout()
    detail_path = Path(detail_path)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.savefig(detail_path, dpi=180, bbox_inches="tight")
    plt.close(detail)


def run_multi_event_extension(
    args,
    output_dir,
    task_name,
    classes,
    projections,
    source_coefficients,
    target_coefficients,
    source_records,
    target_records,
    source_baselines,
    target_baselines,
    source_prototypes,
    recon_shift_days,
    existing_manifest,
    expected_filenames,
):
    config = _event_detector_config(args)
    detection_kwargs = _event_detection_kwargs(config)
    daily = np.arange(365.0)
    source_rows, target_rows, match_rows, class_rows, metadata = [], [], [], [], {}
    folder = Path(output_dir) / "04_reconshift13_multi_event"
    for class_id, class_name in enumerate(classes):
        projection = projections[class_id]
        source_pc1 = projection.transform(source_prototypes[class_id][None])[0]
        source_sample_curves = reconstruct_projected_coefficients(
            source_coefficients[class_id], projection, daily
        )
        source_candidates = detect_circular_events(
            source_pc1,
            daily,
            source_baselines[class_id],
            event_prefix="S",
            **detection_kwargs,
        )
        source_events = tuple(
            evaluate_source_event_stability(
                event,
                source_sample_curves,
                daily,
                source_baselines[class_id],
                occurrence_radius_days=config["source_occurrence_radius_days"],
                min_occurrence=config["min_source_occurrence"],
                max_timing_mad_days=config["max_source_timing_mad_days"],
                **detection_kwargs,
            )
            for event in source_candidates
        )
        source_rows.extend(
            _event_row(task_name, class_id, class_name, event, source=True)
            for event in source_events
        )
        target_curves = reconstruct_projected_coefficients(
            target_coefficients[class_id],
            projection,
            daily - float(recon_shift_days),
        )
        target_events_by_sample, matches_by_sample = [], []
        for record, curve in zip(target_records[class_id], target_curves):
            events = detect_circular_events(
                curve,
                daily,
                target_baselines[class_id],
                event_prefix="T",
                **detection_kwargs,
            )
            target_events_by_sample.append(events)
            target_rows.extend(
                _event_row(
                    task_name,
                    class_id,
                    class_name,
                    event,
                    source=False,
                    sample_id=record["sample_id"],
                )
                for event in events
            )
            matches = greedy_match_events(
                source_events, events, config["match_radius_days"]
            )
            matches_by_sample.append(matches)
            match_rows.extend(
                _match_row(
                    task_name,
                    class_id,
                    class_name,
                    record["sample_id"],
                    match,
                )
                for match in matches
            )
        safe_name = next(
            name for name in expected_filenames if name.startswith(f"{class_id:02d}_")
        )
        output_path = folder / safe_name
        detail_path = folder / "diagnostics" / safe_name.replace(".png", "_event_detail.png")
        _plot_multi_event_class(
            output_path,
            detail_path,
            class_name,
            source_pc1,
            target_curves,
            source_events,
            target_events_by_sample,
            matches_by_sample,
        )
        metadata[str(class_id)] = {
            "path": str(output_path),
            "diagnostic_path": str(detail_path),
            "source_event_count": sum(event.accepted for event in source_events),
        }
        accepted_target_counts = [sum(event.accepted for event in events) for events in target_events_by_sample]
        flat_matches = [match for matches in matches_by_sample for match in matches]
        matched = [item for item in flat_matches if item.match_status.startswith("MATCHED")]
        unmatched_source = [item for item in flat_matches if item.match_status == "UNMATCHED_SOURCE"]
        unmatched_target = [item for item in flat_matches if item.match_status == "UNMATCHED_TARGET"]
        stable = [event for event in source_events if event.accepted]
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "num_source_candidates": len(source_events),
                "num_source_stable_events": len(stable),
                "num_source_peaks": sum(event.kind == "peak" for event in stable),
                "num_source_valleys": sum(event.kind == "valley" for event in stable),
                "mean_target_events_per_sample": float(np.mean(accepted_target_counts)) if accepted_target_counts else np.nan,
                "matched_rate": len(matched) / max(1, len(matched) + len(unmatched_target)),
                "boundary_candidate_rate": sum(item.boundary_crossing_candidate for item in matched) / max(1, len(matched)),
                "unmatched_source_rate": len(unmatched_source) / max(1, len(unmatched_source) + len(matched)),
                "unmatched_target_rate": len(unmatched_target) / max(1, len(unmatched_target) + len(matched)),
                "median_match_circular_distance": _finite_median(item.circular_day_distance for item in matched),
                "source_event_days": ";".join(f"{event.day:.1f}" for event in stable),
                "source_event_types": ";".join(event.kind for event in stable),
            }
        )
    generated = {path.name for path in folder.glob("*.png")}
    if generated != expected_filenames:
        raise RuntimeError(
            "multi-event class files differ from existing views: "
            f"expected={sorted(expected_filenames)}, generated={sorted(generated)}"
        )
    update_multi_event_outputs(
        output_dir,
        existing_manifest,
        config,
        source_rows,
        target_rows,
        match_rows,
        class_rows,
        metadata,
    )


def run_local_nonlinear_extension(
    args,
    output_dir,
    task_name,
    classes,
    grid,
    projections,
    source_curves,
    target_curves,
    source_coefficients,
    target_coefficients,
    source_records,
    target_records,
    source_baselines,
    target_baselines,
    source_prototypes,
    recon_shift_days,
    synthesizer,
    device,
    existing_manifest,
    expected_filenames,
):
    sample_rows, class_rows, class_metadata = [], [], {}
    daily_grid = np.arange(365, dtype=np.float64)
    for class_id, class_name in enumerate(classes):
        source_group = source_curves.get(class_id)
        target_group = target_curves.get(class_id)
        source_items = source_records.get(class_id, [])
        target_items = target_records.get(class_id, [])
        source_prototype = source_prototypes.get(class_id)
        projection = projections.get(class_id)
        if any(value is None for value in (source_group, target_group, source_prototype, projection)):
            raise RuntimeError(f"missing class data for {task_name} class {class_id}")
        if len(target_group) != len(target_items):
            raise RuntimeError("target Raw curves and sample records lost ordering")

        source_pc1 = projection.transform(source_prototype[None])[0]
        detection = detect_salient_extrema(
            source_pc1,
            daily_grid,
            min_distance_days=7,
            min_width_days=3,
            min_normalized_prominence=0.20,
        )
        source_baseline = source_baselines[class_id]
        anchor, source_reason = select_gated_anchor(
            detection.extrema,
            source_pc1,
            daily_grid,
            source_baseline,
            min_normalized_prominence=0.20,
            min_domain_relative_elevation=0.75,
        )
        source_stats = {
            "occurrence_rate": float("nan"),
            "timing_mad": float("nan"),
        }
        if anchor is not None:
            source_recon_pc1 = reconstruct_projected_coefficients(
                source_coefficients[class_id], projection, daily_grid
            )
            source_matches = [
                match_sample_anchor(
                    curve,
                    daily_grid,
                    anchor,
                    search_radius_days=20,
                    min_width_days=3,
                    min_normalized_prominence=0.20,
                    calendar_shift_days=0,
                    min_distance_days=7,
                )
                for curve in source_recon_pc1
            ]
            source_stats = summarize_matches(source_matches)
            if source_stats["occurrence_rate"] < 0.70:
                anchor, source_reason = None, "low_source_occurrence"
            elif (
                not np.isfinite(source_stats["timing_mad"])
                or source_stats["timing_mad"] > 20
            ):
                anchor, source_reason = None, "high_source_timing_mad"

        source_elevation = (
            float("nan")
            if anchor is None
            else anchor_domain_elevation(
                source_pc1, daily_grid, anchor, source_baseline
            )
        )
        source_support = _class_observed_support(source_items)
        target_global_pc1 = None
        if anchor is not None:
            target_global_pc1 = reconstruct_projected_coefficients(
                target_coefficients[class_id],
                projection,
                daily_grid - float(recon_shift_days),
            )

        mapped_positions = []
        class_sample_rows = []
        pending = []
        before_segments, after_segments, windows = [], [], []
        target_anchor_days, mapped_anchor_days = [], []
        for sample_index, record in enumerate(target_items):
            raw_positions = np.asarray(record["positions"], dtype=np.float64)
            global_positions = raw_positions + float(recon_shift_days)
            mapped = global_positions.copy()
            row = {
                "sample_id": record["sample_id"],
                "class_id": class_id,
                "class_name": class_name,
                "global_shift_days": float(recon_shift_days),
                "source_anchor_type": "" if anchor is None else anchor.kind,
                "source_anchor_day": np.nan if anchor is None else anchor.day,
                "source_anchor_relative_prominence": np.nan if anchor is None else anchor.normalized_prominence,
                "source_anchor_domain_elevation": source_elevation,
                "target_anchor_found": False,
                "target_anchor_day_global": np.nan,
                "target_anchor_relative_prominence": np.nan,
                "target_anchor_domain_elevation": np.nan,
                "anchor_error_before": np.nan,
                "window_start_day": np.nan,
                "window_end_day": np.nan,
                "window_width_days": np.nan,
                "common_support_valid": False,
                "nonlinear_attempted": False,
                "nonlinear_valid": False,
                "fallback_reason": source_reason if anchor is None else "",
                "mapped_target_anchor_day": np.nan,
                "anchor_error_after": np.nan,
                "anchor_error_improvement": np.nan,
                "local_corr_before": np.nan,
                "local_corr_after": np.nan,
                "local_corr_gain": np.nan,
                "max_warp_displacement_days": 0.0,
                "min_warp_derivative": 1.0,
                "median_warp_derivative": 1.0,
                "max_warp_derivative": 1.0,
                "nonlinear_extreme": False,
                "raw_timestamp_monotone": bool(np.all(np.diff(global_positions) > 0)),
            }
            if anchor is not None:
                target_curve = target_global_pc1[sample_index]
                match = match_sample_anchor(
                    target_curve,
                    daily_grid,
                    anchor,
                    search_radius_days=20,
                    min_width_days=3,
                    min_normalized_prominence=0.20,
                    calendar_shift_days=0,
                    min_distance_days=7,
                )
                if not match.matched:
                    row["fallback_reason"] = "LOCAL_INELIGIBLE_NO_MATCHING_ANCHOR"
                else:
                    target_anchor = Extremum(
                        anchor.kind,
                        match.day,
                        match.prominence,
                        match.normalized_prominence,
                        0.0,
                    )
                    target_elevation = anchor_domain_elevation(
                        target_curve,
                        daily_grid,
                        target_anchor,
                        target_baselines[class_id],
                    )
                    row.update(
                        target_anchor_found=True,
                        target_anchor_day_global=match.day,
                        target_anchor_relative_prominence=match.normalized_prominence,
                        target_anchor_domain_elevation=target_elevation,
                        anchor_error_before=abs(match.day - anchor.day),
                    )
                    if target_elevation < 0.75:
                        row["fallback_reason"] = "LOCAL_INELIGIBLE_WEAK_TARGET_ANCHOR"
                    else:
                        window = build_joint_anchor_window(
                            anchor.day,
                            match.day,
                            source_support,
                            (float(global_positions.min()), float(global_positions.max())),
                            margin_days=20,
                            max_anchor_distance_days=20,
                            min_anchor_side_support_days=15,
                        )
                        row.update(
                            window_start_day=window.start_day,
                            window_end_day=window.end_day,
                            window_width_days=window.width_days,
                            common_support_valid=window.valid,
                        )
                        if not window.valid:
                            row["fallback_reason"] = window.failure_reason
                        else:
                            row["nonlinear_attempted"] = True
                            window_days = np.linspace(
                                window.start_day, window.end_day, 128, dtype=np.float64
                            )
                            source_window = np.column_stack(
                                [
                                    np.interp(window_days, daily_grid, source_prototype[:, channel])
                                    for channel in range(source_prototype.shape[1])
                                ]
                            )
                            pending.append(
                                {
                                    "sample_index": sample_index,
                                    "row": row,
                                    "record": record,
                                    "global_positions": global_positions,
                                    "source_window": source_window,
                                    "window_days": window_days,
                                    "window": window,
                                    "target_anchor_day": match.day,
                                }
                            )
            mapped_positions.append(mapped)
            class_sample_rows.append(row)
            sample_rows.append(row)

        if pending:
            target_windows = reconstruct_windows_batched(
                np.stack([item["record"]["coefficients"] for item in pending]),
                np.stack(
                    [
                        item["window_days"] - float(recon_shift_days)
                        for item in pending
                    ]
                ),
                synthesizer,
                device,
                args.batch_size,
            )
            for item, target_window in zip(pending, target_windows):
                row = item["row"]
                source_window = item["source_window"]
                window_days = item["window_days"]
                target_anchor_day = item["target_anchor_day"]
                corr_before, _ = local_multivariate_correlation(
                    source_window, target_window
                )
                phase = estimate_whole_window_local_phase(
                    source_window, target_window, window_days
                )
                aligned = target_window
                mapped_anchor = target_anchor_day
                if phase.valid:
                    candidate_positions = apply_whole_window_forward_map(
                        item["global_positions"], phase
                    )
                    monotone = bool(np.all(np.diff(candidate_positions) > 0))
                    if monotone:
                        mapped_positions[item["sample_index"]] = candidate_positions
                        aligned = sample_curve_with_query(
                            target_window, window_days, phase.query_days
                        )
                        mapped_anchor = float(
                            apply_whole_window_forward_map(
                                np.asarray([target_anchor_day]), phase
                            )[0]
                        )
                        row["nonlinear_valid"] = True
                        row["fallback_reason"] = ""
                    else:
                        row["fallback_reason"] = (
                            "NONLINEAR_RAW_TIMESTAMPS_NOT_STRICT"
                        )
                    row["raw_timestamp_monotone"] = monotone
                else:
                    row["fallback_reason"] = phase.failure_reason
                corr_after, _ = local_multivariate_correlation(source_window, aligned)
                row.update(
                    mapped_target_anchor_day=mapped_anchor,
                    anchor_error_after=abs(mapped_anchor - anchor.day),
                    anchor_error_improvement=(
                        abs(target_anchor_day - anchor.day)
                        - abs(mapped_anchor - anchor.day)
                    ),
                    local_corr_before=corr_before,
                    local_corr_after=corr_after,
                    local_corr_gain=corr_after - corr_before,
                    max_warp_displacement_days=(
                        phase.max_displacement_days if row["nonlinear_valid"] else 0.0
                    ),
                    min_warp_derivative=(
                        phase.forward_min_derivative if row["nonlinear_valid"] else 1.0
                    ),
                    median_warp_derivative=(
                        phase.forward_median_derivative if row["nonlinear_valid"] else 1.0
                    ),
                    max_warp_derivative=(
                        phase.forward_max_derivative if row["nonlinear_valid"] else 1.0
                    ),
                    nonlinear_extreme=(
                        phase.extreme if row["nonlinear_valid"] else False
                    ),
                )
                before_segments.append(
                    (window_days, projection.transform(target_window[None])[0])
                )
                after_segments.append(
                    (window_days, projection.transform(aligned[None])[0])
                )
                windows.append((item["window"].start_day, item["window"].end_day))
                target_anchor_days.append(target_anchor_day)
                mapped_anchor_days.append(mapped_anchor)

        raw_positions = tuple(
            np.asarray(item["positions"], dtype=np.float64) + float(recon_shift_days)
            for item in target_items
        )
        raw_values = tuple(np.asarray(item["raw_pc1"]).copy() for item in target_items)
        display = build_samplewise_time_mapped_curves(
            grid,
            source_group,
            raw_positions,
            raw_values,
            tuple(mapped_positions),
        )
        raw_metadata = (
            existing_manifest.get("class_outputs", {})
            .get(str(class_id), {})
            .get("raw", {})
        )
        if "ylim" not in raw_metadata:
            raise ValueError(f"missing Raw ylim for {task_name} class {class_id}")
        eligible = [row for row in class_sample_rows if row["nonlinear_attempted"]]
        valid = [row for row in class_sample_rows if row["nonlinear_valid"]]
        before_anchor = _finite_median(row["anchor_error_before"] for row in valid)
        after_anchor = _finite_median(row["anchor_error_after"] for row in valid)
        before_corr = _finite_median(row["local_corr_before"] for row in valid)
        after_corr = _finite_median(row["local_corr_after"] for row in valid)
        metadata = render_local_nonlinear_figure(
            output_dir,
            task_name.replace("_", "→"),
            class_id,
            class_name,
            display,
            recon_shift_days,
            tuple(raw_metadata["ylim"]),
            len(eligible) / len(target_items) if target_items else 0.0,
            len(valid) / len(target_items) if target_items else 0.0,
            before_anchor,
            after_anchor,
            before_corr,
            after_corr,
            args.max_spaghetti,
            args.seed,
            global_only_reason=(source_reason if anchor is None else ""),
        )
        class_metadata[str(class_id)] = metadata
        if anchor is not None and before_segments:
            filename = Path(metadata["path"]).name
            _plot_local_nonlinear_diagnostic(
                Path(output_dir)
                / "04_reconshift13_local_nonlinear"
                / "diagnostics"
                / filename,
                class_name,
                daily_grid,
                source_pc1,
                anchor,
                before_segments,
                after_segments,
                target_anchor_days,
                mapped_anchor_days,
                windows,
            )
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "source_anchor_exists": anchor is not None,
                "source_anchor_type": "" if anchor is None else anchor.kind,
                "source_anchor_day": np.nan if anchor is None else anchor.day,
                "source_anchor_relative_prominence": np.nan if anchor is None else anchor.normalized_prominence,
                "source_anchor_domain_elevation": source_elevation,
                "source_occurrence_rate": source_stats["occurrence_rate"],
                "source_timing_mad": source_stats["timing_mad"],
                "target_count": len(target_items),
                "target_anchor_eligible_count": len(eligible),
                "target_anchor_eligible_rate": len(eligible) / len(target_items) if target_items else np.nan,
                "nonlinear_valid_count": len(valid),
                "nonlinear_valid_rate": len(valid) / len(target_items) if target_items else np.nan,
                "anchor_error_before_median": before_anchor,
                "anchor_error_after_median": after_anchor,
                "anchor_improvement_median": _finite_median(row["anchor_error_improvement"] for row in valid),
                "anchor_improved_rate": np.mean([row["anchor_error_improvement"] > 0 for row in valid]) if valid else np.nan,
                "local_corr_before_median": before_corr,
                "local_corr_after_median": after_corr,
                "local_corr_gain_median": _finite_median(row["local_corr_gain"] for row in valid),
                "max_warp_displacement_median": _finite_median(row["max_warp_displacement_days"] for row in valid),
                "nonlinear_extreme_rate": np.mean([row["nonlinear_extreme"] for row in valid]) if valid else np.nan,
                "global_only_count": len(target_items) - len(valid),
            }
        )

    generated = {
        path.name
        for path in (Path(output_dir) / "04_reconshift13_local_nonlinear").glob("*.png")
    }
    if generated != expected_filenames:
        raise RuntimeError(
            "local nonlinear class files differ from existing views: "
            f"expected={sorted(expected_filenames)}, generated={sorted(generated)}"
        )
    update_local_nonlinear_outputs(
        output_dir,
        existing_manifest,
        sample_rows,
        class_rows,
        class_metadata,
    )


def run_task(args, source_alias, target_alias, checkpoint_overrides, tm_log_overrides, recon_log_overrides):
    source_path, target_path = DOMAINS[source_alias], DOMAINS[target_alias]
    checkpoint, config = resolve_source_checkpoint(
        args.source_checkpoint_root, source_alias, source_path, args.seed, args.fold, checkpoint_overrides
    )
    classes = load_classes(checkpoint, source_path, args.data_root)
    tm_log = resolve_task_log(args.timematch_log_root, source_alias, target_alias, "timematch", tm_log_overrides)
    recon_log = resolve_task_log(args.reconshift_log_root, source_alias, target_alias, "reconshift13", recon_log_overrides)
    tm_output = resolve_task_output(args.timematch_output_root, source_path, target_path, args.seed, "timematch")
    recon_output = resolve_task_output(args.reconshift_output_root, source_path, target_path, args.seed, "reconshift13")
    tm_shift = read_best_validation_shift(tm_log, tm_output)
    recon_shift = read_best_validation_shift(recon_log, recon_output)

    source_dataset, target_dataset = build_split_datasets(
        args.data_root, source_path, target_path, classes, args.seed, config
    )
    device = torch.device(args.device)
    spatial_encoder = load_raw_spatial_encoder(checkpoint, config, classes, device)
    grid = np.linspace(0.0, 365.0, args.grid_size, dtype=np.float64)
    projections, latent_dim = fit_class_projections(
        spatial_encoder, source_dataset, len(classes), grid, args.batch_size, device, bool(config.get("with_extra", False))
    )
    task_name = f"{source_alias}_{target_alias}"
    output_dir = Path(args.output_root) / task_name
    existing_manifest = None
    expected_filenames = None
    extension_count = (
        int(args.add_class_residual_shift)
        + int(args.add_recon13_local_nonlinear)
        + int(args.add_recon13_multi_event)
    )
    if extension_count > 1:
        raise ValueError("select only one visualization extension per invocation")
    if extension_count:
        existing_manifest, expected_filenames = validate_existing_task_outputs(output_dir)

    fourier_analyzer = None
    fourier_synthesizer = None
    if extension_count:
        fourier_analyzer = BatchedDirectFourierAnalyzer(
            num_modes=13, period_days=365.0, reg=0.001
        ).to(device)
        fourier_synthesizer = BatchedDirectFourierSynthesizer(
            num_modes=13, period_days=365.0
        ).to(device)
    source_projected = project_dataset(
        spatial_encoder,
        source_dataset,
        projections,
        grid,
        args.batch_size,
        device,
        bool(config.get("with_extra", False)),
        fourier_analyzer,
        collect_samplewise=(
            args.add_recon13_local_nonlinear or args.add_recon13_multi_event
        ),
    )
    source_prototypes = None
    if args.add_class_residual_shift:
        source_curves, source_coefficients = source_projected
        source_prototypes = {
            class_id: reconstruct_class_prototype(
                values, fourier_synthesizer, device, args.batch_size
            )
            for class_id, values in source_coefficients.items()
        }
        del source_coefficients
    elif args.add_recon13_local_nonlinear or args.add_recon13_multi_event:
        (
            source_curves,
            source_coefficients,
            source_records,
            source_baselines,
        ) = source_projected
        source_prototypes = {
            class_id: reconstruct_class_prototype(
                values, fourier_synthesizer, device, args.batch_size
            )
            for class_id, values in source_coefficients.items()
        }
    target_projected = project_dataset(
        spatial_encoder,
        target_dataset,
        projections,
        grid,
        args.batch_size,
        device,
        bool(config.get("with_extra", False)),
        fourier_analyzer,
        collect_samplewise=(
            args.add_recon13_local_nonlinear or args.add_recon13_multi_event
        ),
    )
    if args.add_class_residual_shift:
        target_curves, target_coefficients = target_projected
        target_prototypes = {
            class_id: reconstruct_class_prototype(
                values, fourier_synthesizer, device, args.batch_size
            )
            for class_id, values in target_coefficients.items()
        }
        del target_coefficients
    elif args.add_recon13_local_nonlinear or args.add_recon13_multi_event:
        (
            target_curves,
            target_coefficients,
            target_records,
            target_baselines,
        ) = target_projected
    else:
        source_curves, target_curves = source_projected, target_projected

    if args.add_class_residual_shift:
        summary_rows = []
        score_curves = {}
        class_metadata = {}
        for class_id, class_name in enumerate(classes):
            matching = [
                name for name in expected_filenames if name.startswith(f"{class_id:02d}_")
            ]
            if not matching:
                continue
            source_group = source_curves.get(class_id)
            target_group = target_curves.get(class_id)
            source_prototype = source_prototypes.get(class_id)
            target_prototype = target_prototypes.get(class_id)
            if any(
                group is None
                for group in (
                    source_group,
                    target_group,
                    source_prototype,
                    target_prototype,
                )
            ):
                raise RuntimeError(
                    f"missing extracted class data for {task_name} class {class_id}"
                )
            result = estimate_class_residual_shift(
                source_prototype,
                target_prototype,
                recon_shift.shift_days,
                args.class_residual_max_days,
            )
            raw_metadata = (
                existing_manifest.get("class_outputs", {})
                .get(str(class_id), {})
                .get("raw")
            )
            if not raw_metadata or "ylim" not in raw_metadata:
                raise ValueError(
                    f"missing baseline Raw ylim for {task_name} class {class_id}"
                )
            metadata = render_class_residual_figure(
                output_dir,
                task_name.replace("_", "→"),
                class_id,
                class_name,
                grid,
                source_group,
                target_group,
                recon_shift.shift_days,
                result.class_residual_shift_days,
                result.score_at_residual_0,
                result.best_score,
                result.score_gain,
                tuple(raw_metadata["ylim"]),
                args.max_spaghetti,
                args.seed,
            )
            class_metadata[str(class_id)] = metadata
            score_curves[class_id] = result.candidates
            summary_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "source_count": len(source_group),
                    "target_count": len(target_group),
                    "global_reconshift_days": result.global_shift_days,
                    "class_residual_shift_days": result.class_residual_shift_days,
                    "final_shift_days": result.final_shift_days,
                    "score_at_residual_0": result.score_at_residual_0,
                    "best_score": result.best_score,
                    "score_gain": result.score_gain,
                    "boundary_hit": result.boundary_hit,
                    "num_valid_channels": result.num_valid_channels,
                    "common_support_days": result.common_support_days,
                }
            )
        generated = {
            path.name
            for path in (output_dir / "04_reconshift13_class_shift20").glob("*.png")
        }
        if generated != expected_filenames:
            raise RuntimeError(
                "configuration 04 class files differ from 01/02/03: "
                f"expected={sorted(expected_filenames)}, generated={sorted(generated)}"
            )
        update_class_residual_outputs(
            output_dir,
            existing_manifest,
            summary_rows,
            score_curves,
            class_metadata,
            args.class_residual_max_days,
        )
        print(f"[FINISHED] {task_name} class residual extension: {output_dir}")
        return

    if args.add_recon13_multi_event:
        run_multi_event_extension(
            args,
            output_dir,
            task_name,
            classes,
            projections,
            source_coefficients,
            target_coefficients,
            source_records,
            target_records,
            source_baselines,
            target_baselines,
            source_prototypes,
            recon_shift.shift_days,
            existing_manifest,
            expected_filenames,
        )
        print(f"[FINISHED] {task_name} multi-event extension: {output_dir}")
        return

    if args.add_recon13_local_nonlinear:
        run_local_nonlinear_extension(
            args,
            output_dir,
            task_name,
            classes,
            grid,
            projections,
            source_curves,
            target_curves,
            source_coefficients,
            target_coefficients,
            source_records,
            target_records,
            source_baselines,
            target_baselines,
            source_prototypes,
            recon_shift.shift_days,
            fourier_synthesizer,
            device,
            existing_manifest,
            expected_filenames,
        )
        print(f"[FINISHED] {task_name} local nonlinear extension: {output_dir}")
        return

    class_records, summary_rows, skipped = {}, [], []
    for class_id, class_name in enumerate(classes):
        source_group = source_curves.get(class_id)
        target_group = target_curves.get(class_id)
        source_count = 0 if source_group is None else len(source_group)
        target_count = 0 if target_group is None else len(target_group)
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "source_count": source_count,
            "target_count": target_count,
            "timematch_shift_days": tm_shift.shift_days,
            "reconshift_shift_days": recon_shift.shift_days,
        }
        summary_rows.append(row)
        if source_count < args.min_class_samples or target_count < args.min_class_samples:
            skipped.append({**row, "reason": "insufficient_source_or_target_samples"})
            continue
        class_records[str(class_id)] = render_task_class_figures(
            output_dir,
            task_name.replace("_", "→"),
            class_id,
            class_name,
            grid,
            source_group,
            target_group,
            tm_shift.shift_days,
            recon_shift.shift_days,
            args.max_spaghetti,
            args.seed,
        )
    manifest = {
        "audit_type": "offline_raw_pse_shift_visualization",
        "training_performed": False,
        "feature_semantics": "Raw PSE latent trajectory from one source-only PseLTae checkpoint",
        "feature_fixed_across_configs": True,
        "shift_effect": "target time coordinates only; source coordinates and all y values unchanged",
        "task": task_name,
        "source": source_path,
        "target": target_path,
        "seed": args.seed,
        "fold": args.fold,
        "source_checkpoint": str(checkpoint),
        "artifact_inputs": {
            "timematch_output": None if tm_output is None else str(tm_output),
            "timematch_log": str(tm_log),
            "reconshift13_output": None if recon_output is None else str(recon_output),
            "reconshift13_log": str(recon_log),
        },
        "timematch_shift": tm_shift.as_dict(),
        "reconshift13_shift": recon_shift.as_dict(),
        "source_train_count": len(source_dataset),
        "target_train_count": len(target_dataset),
        "classes": classes,
        "skipped_classes": skipped,
        "class_outputs": class_records,
        "pca": "one source-only PC1 per class; target is transform-only; deterministic sign fixing",
        "interpolation": {"method": "linear", "grid": "linspace(0,365,K)", "grid_size": args.grid_size},
        "latent_dim": latent_dim,
        "loader": {"shuffle": False, "num_workers": 0, "full_temporal_sequence": True, "full_pixel_set": True},
        "target_label_use": "offline class grouping and visualization only; never shift selection",
    }
    write_task_outputs(output_dir, manifest, summary_rows)
    print(f"[FINISHED] {task_name}: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--source-checkpoint-root", required=True, type=Path)
    parser.add_argument("--timematch-output-root", required=True, type=Path)
    parser.add_argument("--timematch-log-root", required=True, type=Path)
    parser.add_argument("--reconshift-output-root", required=True, type=Path)
    parser.add_argument("--reconshift-log-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-domain", choices=DOMAINS)
    parser.add_argument("--target-domain", choices=DOMAINS)
    parser.add_argument("--source-checkpoint", action="append", default=[])
    parser.add_argument("--timematch-log", action="append", default=[])
    parser.add_argument("--reconshift13-log", action="append", default=[])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-spaghetti", type=int, default=40)
    parser.add_argument("--min-class-samples", type=int, default=2)
    parser.add_argument("--add-class-residual-shift", action="store_true")
    parser.add_argument("--add-recon13-local-nonlinear", action="store_true")
    parser.add_argument("--add-recon13-multi-event", action="store_true")
    parser.add_argument("--class-residual-max-days", type=int, default=20)
    parser.add_argument("--event-min-distance-days", type=float, default=15)
    parser.add_argument("--event-min-width-days", type=float, default=5)
    parser.add_argument("--event-min-relative-prominence", type=float, default=0.15)
    parser.add_argument("--event-min-domain-prominence", type=float, default=0.20)
    parser.add_argument("--event-min-domain-elevation", type=float, default=0.50)
    parser.add_argument("--event-source-occurrence-radius-days", type=float, default=20)
    parser.add_argument("--event-min-source-occurrence", type=float, default=0.60)
    parser.add_argument("--event-max-source-timing-mad-days", type=float, default=20)
    parser.add_argument("--event-match-radius-days", type=float, default=25)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fold != 0:
        raise SystemExit("ERROR: this audit reproduces fold 0 only")
    if (args.source_domain is None) != (args.target_domain is None):
        raise SystemExit("ERROR: --source-domain and --target-domain must be provided together")
    for path, label in (
        (args.data_root, "data root"),
        (args.source_checkpoint_root, "source checkpoint root"),
        (args.timematch_output_root, "Original TimeMatch output root"),
        (args.timematch_log_root, "Original TimeMatch log root"),
        (args.reconshift_output_root, "ReconShift13 output root"),
        (args.reconshift_log_root, "ReconShift13 log root"),
    ):
        if not path.is_dir():
            raise SystemExit(f"ERROR: {label} not found: {path}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tasks = ((args.source_domain, args.target_domain),) if args.source_domain else TASKS
    checkpoint_overrides = _key_values(args.source_checkpoint)
    tm_log_overrides = _key_values(args.timematch_log)
    recon_log_overrides = _key_values(args.reconshift13_log)
    for source_alias, target_alias in tasks:
        run_task(args, source_alias, target_alias, checkpoint_overrides, tm_log_overrides, recon_log_overrides)


if __name__ == "__main__":
    main()
