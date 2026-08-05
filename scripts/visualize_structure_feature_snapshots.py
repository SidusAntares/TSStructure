#!/usr/bin/env python3
"""Visualize legacy or compact PSE/aligned-Shape feature snapshots."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods.structure_da.feature_snapshots import (
    fit_deterministic_pca,
    load_projection_basis,
    load_selected_samples,
    project_features,
)


COMPACT_FIELDS = {
    "epoch", "projection_fitted_epoch", "schema_version",
    "pse_pc_values", "pse_times", "pse_offsets",
    "aligned_shape_pc", "shape_grid", "shape_valid",
}
LEGACY_FIELDS = {
    "epoch", "pse_values", "pse_times", "pse_offsets",
    "aligned_shape", "shape_grid", "shape_valid",
}
FEATURES = ("pse", "shape")
DOMAINS = ((0, "source"), (1, "target"))
OUTLIER_FIELDS = (
    "epoch", "domain", "class_id", "class_name", "parcel_index",
    "component", "max_abs_pc", "outside_robust_fraction",
)


def _class_colors(classes: Iterable[int]) -> dict[int, tuple[float, float, float, float]]:
    """Map a class id to the same tab20 color regardless of classes present."""

    palette = plt.get_cmap("tab20")
    return {int(label): palette(int(label) % palette.N) for label in classes}


@dataclass(frozen=True)
class ProjectedEpoch:
    epoch: int
    pse_pc_values: np.ndarray
    pse_times: np.ndarray
    pse_offsets: np.ndarray
    aligned_shape_pc: np.ndarray
    shape_grid: np.ndarray
    shape_valid: np.ndarray


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def _epoch_paths(snapshot_dir: Path) -> list[Path]:
    paths = sorted(snapshot_dir.glob("epoch_*.npz"))
    if not paths:
        raise ValueError("no epoch snapshots found")
    return paths


def _validate_common(snapshot: dict[str, np.ndarray], count: int, *, compact: bool) -> None:
    offsets = snapshot["pse_offsets"]
    pse_key = "pse_pc_values" if compact else "pse_values"
    shape_key = "aligned_shape_pc" if compact else "aligned_shape"
    pse = snapshot[pse_key]
    shape = snapshot[shape_key]
    valid = snapshot["shape_valid"]
    if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(pse):
        raise ValueError("snapshot PSE ragged offsets are invalid")
    if len(snapshot["pse_times"]) != len(pse):
        raise ValueError("snapshot PSE values and times disagree")
    if shape.ndim != 3 or shape.shape[0] != count or valid.shape != (count,):
        raise ValueError("snapshot Shape sample dimensions are invalid")
    if snapshot["shape_grid"].shape != (shape.shape[1],):
        raise ValueError("snapshot Shape grid is invalid")
    if compact and (pse.shape[-1] != 2 or shape.shape[-1] != 2):
        raise ValueError("compact snapshots must contain exactly components 1 and 2")


def _load_projected_epochs(
    snapshot_dir: Path, count: int
) -> tuple[str, str, list[ProjectedEpoch]]:
    raw = [_read_npz(path) for path in _epoch_paths(snapshot_dir)]
    first_fields = set(raw[0])
    if first_fields == COMPACT_FIELDS:
        if any(set(snapshot) != COMPACT_FIELDS for snapshot in raw):
            raise ValueError("snapshot epochs mix incompatible schemas")
        basis = load_projection_basis(snapshot_dir / "projection_basis.npz")
        if basis.fitted_epoch != 25:
            raise ValueError("compact snapshots require a projection basis fitted at epoch 25")
        projected: list[ProjectedEpoch] = []
        for snapshot in raw:
            _validate_common(snapshot, count, compact=True)
            if int(snapshot["schema_version"].item()) != 2:
                raise ValueError("compact snapshot schema version is invalid")
            if int(snapshot["projection_fitted_epoch"].item()) != basis.fitted_epoch:
                raise ValueError("compact snapshot projection epoch is inconsistent")
            projected.append(ProjectedEpoch(
                epoch=int(snapshot["epoch"].item()),
                pse_pc_values=snapshot["pse_pc_values"].astype(np.float32),
                pse_times=snapshot["pse_times"].astype(np.float32),
                pse_offsets=snapshot["pse_offsets"].astype(np.int64),
                aligned_shape_pc=snapshot["aligned_shape_pc"].astype(np.float32),
                shape_grid=snapshot["shape_grid"].astype(np.float32),
                shape_valid=snapshot["shape_valid"].astype(bool),
            ))
        return "compact_fixed_pca", "stored_fixed_epoch25_basis", projected

    if first_fields != LEGACY_FIELDS or any(set(snapshot) != LEGACY_FIELDS for snapshot in raw):
        raise ValueError("snapshot schema is unsupported")
    for snapshot in raw:
        _validate_common(snapshot, count, compact=False)
    pse_fit = fit_deterministic_pca(np.concatenate([
        snapshot["pse_values"].astype(np.float32) for snapshot in raw
    ], axis=0))
    shape_rows = [
        snapshot["aligned_shape"][snapshot["shape_valid"].astype(bool)].reshape(
            -1, snapshot["aligned_shape"].shape[-1]
        ).astype(np.float32)
        for snapshot in raw
    ]
    nonempty_shape_rows = [values for values in shape_rows if len(values)]
    if not nonempty_shape_rows:
        raise ValueError("legacy snapshots contain no valid aligned Shape")
    shape_fit = fit_deterministic_pca(np.concatenate(nonempty_shape_rows, axis=0))
    projected = []
    for snapshot in raw:
        shape = snapshot["aligned_shape"].astype(np.float32)
        shape_pc = project_features(
            shape.reshape(-1, shape.shape[-1]), shape_fit.mean, shape_fit.components
        ).reshape(*shape.shape[:-1], 2)
        shape_pc[~snapshot["shape_valid"].astype(bool)] = 0
        projected.append(ProjectedEpoch(
            epoch=int(snapshot["epoch"].item()),
            pse_pc_values=project_features(
                snapshot["pse_values"], pse_fit.mean, pse_fit.components
            ).astype(np.float32),
            pse_times=snapshot["pse_times"].astype(np.float32),
            pse_offsets=snapshot["pse_offsets"].astype(np.int64),
            aligned_shape_pc=shape_pc.astype(np.float32),
            shape_grid=snapshot["shape_grid"].astype(np.float32),
            shape_valid=snapshot["shape_valid"].astype(bool),
        ))
    return "legacy_full_features", "joint_all_epochs_in_memory", projected


def _load_class_names(snapshot_dir: Path, labels: np.ndarray) -> dict[int, str]:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = value.get("class_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        return {}
    return {
        int(label): names[int(label)]
        for label in np.unique(labels)
        if 0 <= int(label) < len(names)
    }


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return slug.lower()


def _class_stem(label: int, names: dict[int, str]) -> str:
    suffix = _slug(names[label]) if label in names else ""
    return f"class_{label:02d}" + (f"_{suffix}" if suffix else "")


def _class_display(label: int, names: dict[int, str]) -> str:
    return names.get(label, f"class {label}")


def _uniform_sample_indices(
    sample_index: np.ndarray,
    domain: np.ndarray,
    labels: np.ndarray,
    domain_id: int,
    label: int,
    maximum: int,
) -> np.ndarray:
    indices = np.flatnonzero((domain == domain_id) & (labels == label))
    indices = indices[np.argsort(sample_index[indices], kind="stable")]
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=maximum).round().astype(np.int64)
    return indices[positions]


def _curve(
    snapshot: ProjectedEpoch, feature: str, sample: int, component: int
) -> tuple[np.ndarray, np.ndarray] | None:
    index = component - 1
    if feature == "pse":
        start, end = snapshot.pse_offsets[sample:sample + 2]
        return snapshot.pse_times[start:end], snapshot.pse_pc_values[start:end, index]
    if not snapshot.shape_valid[sample]:
        return None
    return snapshot.shape_grid, snapshot.aligned_shape_pc[sample, :, index]


def _all_values(
    snapshots: list[ProjectedEpoch],
    feature: str,
    component: int,
    sample_indices: Iterable[int],
) -> np.ndarray:
    values: list[np.ndarray] = []
    indices = tuple(int(index) for index in sample_indices)
    for snapshot in snapshots:
        for sample in indices:
            curve = _curve(snapshot, feature, sample, component)
            if curve is not None:
                finite = curve[1][np.isfinite(curve[1])]
                if len(finite):
                    values.append(finite)
    if not values:
        raise ValueError("comparison group has no finite projected values")
    return np.concatenate(values)


def _limits(values: np.ndarray, lower: float, upper: float) -> dict[str, tuple[float, float]]:
    full = (float(np.min(values)), float(np.max(values)))
    robust = tuple(float(value) for value in np.percentile(values, (lower, upper)))
    result: dict[str, tuple[float, float]] = {}
    for name, pair in (("full", full), ("robust", robust)):
        low, high = pair
        if low == high:
            padding = max(abs(low) * 0.05, 1e-6)
            low, high = low - padding, high + padding
        result[name] = (low, high)
    return result


def _save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_class_evolution(
    output_dir: Path,
    snapshots: list[ProjectedEpoch],
    selected,
    names: dict[int, str],
    colors: dict[int, object],
    feature: str,
    label: int,
    component: int,
    display_samples: int,
    lower: float,
    upper: float,
    dpi: int,
) -> int:
    class_indices = np.flatnonzero(selected.label == label)
    ranges = _limits(
        _all_values(snapshots, feature, component, class_indices), lower, upper
    )
    plotted_maximum = 0
    for range_name, y_limits in ranges.items():
        figure, axes = plt.subplots(
            2, len(snapshots),
            figsize=(3.1 * len(snapshots), 5.5),
            sharey=True,
            squeeze=False,
            constrained_layout=True,
        )
        for row, (domain_id, domain_name) in enumerate(DOMAINS):
            fixed_indices = _uniform_sample_indices(
                selected.sample_index, selected.domain, selected.label,
                domain_id, label, display_samples,
            )
            for column, snapshot in enumerate(snapshots):
                axis = axes[row, column]
                count = 0
                for sample in fixed_indices:
                    curve = _curve(snapshot, feature, int(sample), component)
                    if curve is None:
                        continue
                    axis.plot(curve[0], curve[1], color=colors[label], linewidth=0.8, alpha=0.65)
                    count += 1
                plotted_maximum = max(plotted_maximum, count)
                axis.set_title(f"{domain_name}, epoch {snapshot.epoch}")
                axis.set_ylim(*y_limits)
                if row == 1:
                    axis.set_xlabel("physical time" if feature == "pse" else "canonical grid")
                if column == 0:
                    axis.set_ylabel(f"PC{component}")
        path = (
            output_dir / "class_evolution" /
            f"{feature}_{_class_stem(label, names)}_pc{component}_{range_name}.png"
        )
        _save_figure(figure, path, dpi)
    return plotted_maximum


def _plot_class_separation(
    output_dir: Path,
    snapshots: list[ProjectedEpoch],
    selected,
    names: dict[int, str],
    colors: dict[int, object],
    feature: str,
    domain_id: int,
    domain_name: str,
    component: int,
    samples_per_class: int,
    lower: float,
    upper: float,
    dpi: int,
) -> tuple[int, list[str]]:
    domain_indices = np.flatnonzero(selected.domain == domain_id)
    ranges = _limits(
        _all_values(snapshots, feature, component, domain_indices), lower, upper
    )
    maximum = 0
    legend_labels: list[str] = []
    classes = [int(value) for value in np.unique(selected.label[domain_indices])]
    for snapshot in snapshots:
        for range_name, y_limits in ranges.items():
            figure, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
            for label in classes:
                indices = _uniform_sample_indices(
                    selected.sample_index, selected.domain, selected.label,
                    domain_id, label, samples_per_class,
                )
                class_count = 0
                for sample in indices:
                    curve = _curve(snapshot, feature, int(sample), component)
                    if curve is None:
                        continue
                    display = _class_display(label, names)
                    axis.plot(
                        curve[0], curve[1], color=colors[label], linewidth=0.8, alpha=0.65,
                        label=display if class_count == 0 else None,
                    )
                    class_count += 1
                maximum = max(maximum, class_count)
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
                legend_labels = labels
            axis.set_xlabel("physical time" if feature == "pse" else "canonical grid")
            axis.set_ylabel(f"PC{component}")
            axis.set_title(f"{domain_name}, epoch {snapshot.epoch}")
            axis.set_ylim(*y_limits)
            path = (
                output_dir / "class_separation" /
                f"{feature}_{domain_name}_epoch_{snapshot.epoch:04d}_pc{component}_{range_name}.png"
            )
            _save_figure(figure, path, dpi)
    return maximum, legend_labels


def _outlier_rows(
    snapshots: list[ProjectedEpoch],
    selected,
    names: dict[int, str],
    feature: str,
    components: tuple[int, ...],
    lower: float,
    upper: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_id, domain_name in DOMAINS:
        indices = np.flatnonzero(selected.domain == domain_id)
        for component in components:
            values = _all_values(snapshots, feature, component, indices)
            low, high = np.percentile(values, (lower, upper))
            for snapshot in snapshots:
                for sample in indices:
                    curve = _curve(snapshot, feature, int(sample), component)
                    if curve is None:
                        continue
                    projected = curve[1][np.isfinite(curve[1])]
                    if not len(projected):
                        continue
                    outside = (projected < low) | (projected > high)
                    if not outside.any():
                        continue
                    label = int(selected.label[sample])
                    rows.append({
                        "epoch": snapshot.epoch,
                        "domain": domain_name,
                        "class_id": label,
                        "class_name": names.get(label, ""),
                        "parcel_index": int(selected.sample_index[sample]),
                        "component": component,
                        "max_abs_pc": f"{float(np.max(np.abs(projected))):.8g}",
                        "outside_robust_fraction": f"{float(np.mean(outside)):.8g}",
                    })
    return rows


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTLIER_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def visualize_snapshots(
    snapshot_dir: Path,
    output_dir: Path,
    *,
    display_samples_per_class: int = 8,
    separation_samples_per_class: int = 3,
    components: tuple[int, ...] = (1, 2),
    robust_lower_percentile: float = 1.0,
    robust_upper_percentile: float = 99.0,
    dpi: int = 180,
) -> dict[str, object]:
    snapshot_dir, output_dir = Path(snapshot_dir), Path(output_dir)
    if display_samples_per_class <= 0 or separation_samples_per_class <= 0:
        raise ValueError("sample display limits must be positive")
    components = tuple(int(value) for value in components)
    if not components or any(value not in (1, 2) for value in components):
        raise ValueError("only projection components 1 and 2 are available")
    if len(set(components)) != len(components):
        raise ValueError("projection components must be unique")
    if not 0 <= robust_lower_percentile < robust_upper_percentile <= 100:
        raise ValueError("robust percentiles must satisfy 0 <= lower < upper <= 100")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    selected = load_selected_samples(snapshot_dir / "selected_samples.npz")
    schema, pca_mode, snapshots = _load_projected_epochs(
        snapshot_dir, len(selected.sample_index)
    )
    epochs = [snapshot.epoch for snapshot in snapshots]
    if epochs != [25, 50, 75, 100]:
        raise ValueError("visualization requires epoch snapshots 25, 50, 75, and 100")
    names = _load_class_names(snapshot_dir, selected.label)
    classes = [int(value) for value in np.unique(selected.label)]
    colors = _class_colors(classes)

    print(f"VISUALIZATION_SCHEMA|type={schema}", flush=True)
    print(f"VISUALIZATION_PCA|mode={pca_mode}", flush=True)
    figure_count = 0
    max_evolution = 0
    max_separation = 0
    legend_labels: list[str] = []
    for feature in FEATURES:
        for component in components:
            for label in classes:
                max_evolution = max(
                    max_evolution,
                    _plot_class_evolution(
                        output_dir, snapshots, selected, names, colors, feature, label,
                        component, display_samples_per_class,
                        robust_lower_percentile, robust_upper_percentile, dpi,
                    ),
                )
                figure_count += 2
            for domain_id, domain_name in DOMAINS:
                maximum, labels = _plot_class_separation(
                    output_dir, snapshots, selected, names, colors, feature,
                    domain_id, domain_name, component, separation_samples_per_class,
                    robust_lower_percentile, robust_upper_percentile, dpi,
                )
                max_separation = max(max_separation, maximum)
                if labels:
                    legend_labels = labels
                figure_count += len(snapshots) * 2

    pse_outliers = _outlier_rows(
        snapshots, selected, names, "pse", components,
        robust_lower_percentile, robust_upper_percentile,
    )
    shape_outliers = _outlier_rows(
        snapshots, selected, names, "shape", components,
        robust_lower_percentile, robust_upper_percentile,
    )
    _write_tsv(output_dir / "pse_pc_outliers.tsv", pse_outliers)
    _write_tsv(output_dir / "shape_pc_outliers.tsv", shape_outliers)
    invalid_shape_skipped = sum(int((~snapshot.shape_valid).sum()) for snapshot in snapshots)

    summary = {
        "schema": schema,
        "pca_mode": pca_mode,
        "epochs": epochs,
        "classes": classes,
        "components": list(components),
        "figures": figure_count,
        "pse_outliers": len(pse_outliers),
        "shape_outliers": len(shape_outliers),
        "max_evolution_curves_per_axis": max_evolution,
        "max_separation_curves_per_class": max_separation,
        "evolution_grid": (2, len(snapshots)),
        "legend_labels": legend_labels,
        "invalid_shape_skipped": invalid_shape_skipped,
    }
    print(f"VISUALIZATION_SUMMARY|schema={schema}", flush=True)
    print(f"VISUALIZATION_SUMMARY|epochs={','.join(map(str, epochs))}", flush=True)
    print(f"VISUALIZATION_SUMMARY|classes={','.join(map(str, classes))}", flush=True)
    print(f"VISUALIZATION_SUMMARY|components={','.join(map(str, components))}", flush=True)
    print(f"VISUALIZATION_SUMMARY|figures={figure_count}", flush=True)
    print(f"VISUALIZATION_SUMMARY|pse_outliers={len(pse_outliers)}", flush=True)
    print(f"VISUALIZATION_SUMMARY|shape_outliers={len(shape_outliers)}", flush=True)
    print(f"VISUALIZATION_SUMMARY|output_dir={output_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize fixed-PC PSE and aligned-Shape snapshot curves"
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--display-samples-per-class", type=int, default=8)
    parser.add_argument("--separation-samples-per-class", type=int, default=3)
    parser.add_argument("--components", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--robust-lower-percentile", type=float, default=1.0)
    parser.add_argument("--robust-upper-percentile", type=float, default=99.0)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    visualize_snapshots(
        args.snapshot_dir,
        args.output_dir,
        display_samples_per_class=args.display_samples_per_class,
        separation_samples_per_class=args.separation_samples_per_class,
        components=tuple(args.components),
        robust_lower_percentile=args.robust_lower_percentile,
        robust_upper_percentile=args.robust_upper_percentile,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
