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


SCHEMA2_FIELDS = {
    "epoch", "projection_fitted_epoch", "schema_version",
    "pse_pc_values", "pse_times", "pse_offsets",
    "aligned_shape_pc", "shape_grid", "shape_valid",
}
SCHEMA3_FIELDS = {
    "epoch", "projection_fitted_epoch", "schema_version",
    "pse_pc_values", "pse_times", "pse_offsets",
    "unaligned_shape_pc", "aligned_shape_pc", "shape_grid", "shape_valid",
    "phase_status", "accepted_warp",
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
    unaligned_shape_pc: np.ndarray | None
    shape_grid: np.ndarray
    shape_valid: np.ndarray
    phase_status: np.ndarray | None
    accepted_warp: np.ndarray | None


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def _epoch_paths(snapshot_dir: Path) -> list[Path]:
    discovered: list[tuple[int, Path]] = []
    for path in snapshot_dir.glob("epoch_*.npz"):
        match = re.fullmatch(r"epoch_(\d+)\.npz", path.name)
        if match is None:
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                epoch = int(data["epoch"].item())
        except Exception as error:
            raise ValueError(f"corrupt epoch snapshot {path.name}: {error}") from error
        if epoch != int(match.group(1)):
            raise ValueError(f"epoch field disagrees with filename for {path.name}")
        discovered.append((epoch, path))
    discovered.sort(key=lambda item: item[0])
    paths = [path for _, path in discovered]
    if not paths:
        raise ValueError("no epoch snapshots found")
    if len({epoch for epoch, _ in discovered}) != len(discovered):
        raise ValueError("snapshot epochs must be unique")
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
    if compact and (pse.ndim != 2 or pse.shape[-1] != shape.shape[-1]):
        raise ValueError("compact snapshot projection dimensions disagree")


def _load_projected_epochs(
    snapshot_dir: Path, count: int
) -> tuple[str, str, list[ProjectedEpoch]]:
    raw = [_read_npz(path) for path in _epoch_paths(snapshot_dir)]
    first_fields = set(raw[0])
    if first_fields in (SCHEMA2_FIELDS, SCHEMA3_FIELDS):
        expected_fields = first_fields
        if any(set(snapshot) != expected_fields for snapshot in raw):
            raise ValueError("snapshot epochs mix incompatible schemas")
        basis = load_projection_basis(snapshot_dir / "projection_basis.npz")
        schema_version = 2 if first_fields == SCHEMA2_FIELDS else 3
        projected: list[ProjectedEpoch] = []
        for snapshot in raw:
            _validate_common(snapshot, count, compact=True)
            if int(snapshot["schema_version"].item()) != schema_version:
                raise ValueError("compact snapshot schema version is invalid")
            if int(snapshot["projection_fitted_epoch"].item()) != basis.fitted_epoch:
                raise ValueError("compact snapshot projection epoch is inconsistent")
            if snapshot["pse_pc_values"].shape[-1] != basis.num_components:
                raise ValueError("snapshot component count disagrees with projection basis")
            unaligned = (
                snapshot["unaligned_shape_pc"].astype(np.float32)
                if schema_version == 3 else None
            )
            phase_status = (
                snapshot["phase_status"].astype(np.uint8)
                if schema_version == 3 else None
            )
            accepted_warp = (
                snapshot["accepted_warp"].astype(np.float32)
                if schema_version == 3 else None
            )
            if schema_version == 3 and (
                unaligned.shape != snapshot["aligned_shape_pc"].shape
                or phase_status.shape != (count,)
                or accepted_warp.shape != (count, len(snapshot["shape_grid"]))
                or np.any(phase_status > 2)
            ):
                raise ValueError("schema 3 phase diagnostic dimensions are invalid")
            projected.append(ProjectedEpoch(
                epoch=int(snapshot["epoch"].item()),
                pse_pc_values=snapshot["pse_pc_values"].astype(np.float32),
                pse_times=snapshot["pse_times"].astype(np.float32),
                pse_offsets=snapshot["pse_offsets"].astype(np.int64),
                aligned_shape_pc=snapshot["aligned_shape_pc"].astype(np.float32),
                unaligned_shape_pc=unaligned,
                shape_grid=snapshot["shape_grid"].astype(np.float32),
                shape_valid=snapshot["shape_valid"].astype(bool),
                phase_status=phase_status,
                accepted_warp=accepted_warp,
            ))
        return (
            "compact_fixed_pca" if schema_version == 2 else "schema_3_fixed_pca",
            f"stored_fixed_epoch{basis.fitted_epoch}_basis",
            projected,
        )

    if first_fields != LEGACY_FIELDS or any(set(snapshot) != LEGACY_FIELDS for snapshot in raw):
        raise ValueError("snapshot schema is unsupported")
    for snapshot in raw:
        _validate_common(snapshot, count, compact=False)
    pse_fit = fit_deterministic_pca(
        np.concatenate([
            snapshot["pse_values"].astype(np.float32) for snapshot in raw
        ], axis=0),
        num_components=2,
    )
    shape_rows = [
        snapshot["aligned_shape"][snapshot["shape_valid"].astype(bool)].reshape(
            -1, snapshot["aligned_shape"].shape[-1]
        ).astype(np.float32)
        for snapshot in raw
    ]
    nonempty_shape_rows = [values for values in shape_rows if len(values)]
    if not nonempty_shape_rows:
        raise ValueError("legacy snapshots contain no valid aligned Shape")
    shape_fit = fit_deterministic_pca(
        np.concatenate(nonempty_shape_rows, axis=0), num_components=2
    )
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
            unaligned_shape_pc=None,
            shape_grid=snapshot["shape_grid"].astype(np.float32),
            shape_valid=snapshot["shape_valid"].astype(bool),
            phase_status=None,
            accepted_warp=None,
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
                selected.parcel_index, selected.domain, selected.label,
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
                    selected.parcel_index, selected.domain, selected.label,
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
                        "parcel_index": int(selected.parcel_index[sample]),
                        "component": component,
                        "max_abs_pc": f"{float(np.max(np.abs(projected))):.8g}",
                        "outside_robust_fraction": f"{float(np.mean(outside)):.8g}",
                    })
    return rows


def _write_tsv(
    path: Path,
    rows: list[dict[str, object]],
    fields: tuple[str, ...] = OUTLIER_FIELDS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _plot_shape_alignment(
    output_dir: Path,
    snapshots: list[ProjectedEpoch],
    selected,
    names: dict[int, str],
    colors: dict[int, object],
    domain_id: int,
    domain_name: str,
    label: int,
    component: int,
    display_samples: int,
    lower: float,
    upper: float,
    dpi: int,
) -> int:
    fixed = _uniform_sample_indices(
        selected.parcel_index, selected.domain, selected.label,
        domain_id, label, display_samples,
    )
    all_values: list[np.ndarray] = []
    for snapshot in snapshots:
        if snapshot.unaligned_shape_pc is None:
            continue
        for values in (
            snapshot.unaligned_shape_pc[fixed, :, component - 1],
            snapshot.aligned_shape_pc[fixed, :, component - 1],
        ):
            finite = values[np.isfinite(values)]
            if len(finite):
                all_values.append(finite)
    if not all_values:
        return 0
    ranges = _limits(np.concatenate(all_values), lower, upper)
    count = 0
    for range_name, y_limits in ranges.items():
        figure, axes = plt.subplots(
            2, len(snapshots),
            figsize=(3.1 * len(snapshots), 5.5),
            sharex=True, sharey=True, squeeze=False, constrained_layout=True,
        )
        for column, snapshot in enumerate(snapshots):
            for row, (title, array) in enumerate((
                ("unaligned Shape", snapshot.unaligned_shape_pc),
                ("aligned Shape", snapshot.aligned_shape_pc),
            )):
                axis = axes[row, column]
                for sample in fixed:
                    if not snapshot.shape_valid[sample]:
                        continue
                    axis.plot(
                        snapshot.shape_grid,
                        array[sample, :, component - 1],
                        color=colors[label], linewidth=0.8, alpha=0.65,
                    )
                    count = max(count, 1)
                axis.set_title(f"{title}, epoch {snapshot.epoch}")
                axis.set_ylim(*y_limits)
                if row == 1:
                    axis.set_xlabel("canonical grid")
                if column == 0:
                    axis.set_ylabel(f"PC{component}")
        path = output_dir / "shape_alignment" / (
            f"{domain_name}_{_class_stem(label, names)}_pc{component}_{range_name}.png"
        )
        _save_figure(figure, path, dpi)
    return count


PHASE_STATUS_FIELDS = (
    "epoch", "domain", "class_id", "class_name", "num_samples",
    "failure_count", "identity_count", "nonidentity_count",
    "failure_rate", "identity_rate", "nonidentity_rate",
)


def _phase_status_rows(
    snapshots: list[ProjectedEpoch], selected, names: dict[int, str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for snapshot in snapshots:
        if snapshot.phase_status is None:
            continue
        for domain_id, domain_name in DOMAINS:
            for label in np.unique(selected.label[selected.domain == domain_id]):
                indices = np.flatnonzero(
                    (selected.domain == domain_id) & (selected.label == label)
                )
                status = snapshot.phase_status[indices]
                count = len(status)
                failure = int(np.count_nonzero(status == 0))
                identity = int(np.count_nonzero(status == 1))
                nonidentity = int(np.count_nonzero(status == 2))
                rows.append({
                    "epoch": snapshot.epoch,
                    "domain": domain_name,
                    "class_id": int(label),
                    "class_name": names.get(int(label), ""),
                    "num_samples": count,
                    "failure_count": failure,
                    "identity_count": identity,
                    "nonidentity_count": nonidentity,
                    "failure_rate": f"{failure / count:.8g}",
                    "identity_rate": f"{identity / count:.8g}",
                    "nonidentity_rate": f"{nonidentity / count:.8g}",
                })
    return rows


def _plot_warp_evolution(
    output_dir: Path,
    snapshots: list[ProjectedEpoch],
    selected,
    names: dict[int, str],
    domain_id: int,
    domain_name: str,
    label: int,
    dpi: int,
) -> None:
    fixed = _uniform_sample_indices(
        selected.parcel_index, selected.domain, selected.label,
        domain_id, label, 3,
    )
    figure, axes = plt.subplots(
        1, len(snapshots), figsize=(3.1 * len(snapshots), 3.2),
        squeeze=False, sharex=True, sharey=True, constrained_layout=True,
    )
    for column, snapshot in enumerate(snapshots):
        axis = axes[0, column]
        valid_count = 0
        for sample in fixed:
            if snapshot.phase_status[sample] == 0:
                continue
            axis.plot(
                snapshot.shape_grid, snapshot.accepted_warp[sample],
                linewidth=0.9, alpha=0.75,
            )
            valid_count += 1
        axis.plot(
            snapshot.shape_grid, snapshot.shape_grid,
            color="black", linestyle="--", linewidth=1.0,
        )
        axis.set_title(f"epoch {snapshot.epoch}, valid={valid_count}")
        axis.set_xlabel("identity grid")
        if column == 0:
            axis.set_ylabel("accepted gamma")
    _save_figure(
        figure,
        output_dir / "warp_evolution" /
        f"{domain_name}_{_class_stem(label, names)}.png",
        dpi,
    )


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
    if not components or any(value <= 0 for value in components):
        raise ValueError("projection components must be positive")
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
    available_components = int(snapshots[0].pse_pc_values.shape[-1])
    if any(value > available_components for value in components):
        raise ValueError(
            f"requested component exceeds available PC1-PC{available_components}"
        )
    basis_path = snapshot_dir / "projection_basis.npz"
    fitted_epoch = (
        load_projection_basis(basis_path).fitted_epoch
        if basis_path.exists() else None
    )
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

    shape_alignment_status = "SKIPPED"
    if all(snapshot.unaligned_shape_pc is not None for snapshot in snapshots):
        shape_alignment_status = "GENERATED"
        for component in components:
            for domain_id, domain_name in DOMAINS:
                for label in classes:
                    _plot_shape_alignment(
                        output_dir, snapshots, selected, names, colors,
                        domain_id, domain_name, label, component,
                        display_samples_per_class,
                        robust_lower_percentile, robust_upper_percentile, dpi,
                    )
                    figure_count += 2
        phase_rows = _phase_status_rows(snapshots, selected, names)
        _write_tsv(
            output_dir / "phase_status_summary.tsv",
            phase_rows,
            PHASE_STATUS_FIELDS,
        )
        for domain_id, domain_name in DOMAINS:
            for label in classes:
                _plot_warp_evolution(
                    output_dir, snapshots, selected, names,
                    domain_id, domain_name, label, dpi,
                )
                figure_count += 1
    else:
        reason = (
            "schema_2_has_no_unaligned_shape"
            if schema == "compact_fixed_pca"
            else "schema_1_has_no_unaligned_shape"
        )
        print(f"SKIPPED|reason={reason}", flush=True)

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
        "projection_fitted_epoch": fitted_epoch,
        "shape_alignment_status": shape_alignment_status,
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
