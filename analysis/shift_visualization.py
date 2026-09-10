"""Offline Raw-PSE visualization helpers for comparing temporal shifts.

This module does not import training entry points and never estimates a shift.
It only reads recorded shifts and changes the plotted target time coordinates.
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
CONFIG_FOLDERS = {
    "raw": "01_raw_pse",
    "timematch": "02_timematch_shift",
    "reconshift13": "03_reconshift13_shift",
}


@dataclass(frozen=True)
class ShiftedCurves:
    source_x: np.ndarray
    source_y: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray


@dataclass(frozen=True)
class ShiftSelection:
    shift_days: float
    best_epoch: Optional[int]
    best_validation_f1: Optional[float]
    selection: str
    source_format: str
    source_path: str
    fallback: bool = False

    def as_dict(self) -> dict:
        return {
            "shift_days": self.shift_days,
            "best_epoch": self.best_epoch,
            "best_validation_f1": self.best_validation_f1,
            "selection": self.selection,
            "source_format": self.source_format,
            "source_path": self.source_path,
            "fallback": self.fallback,
        }


@dataclass(frozen=True)
class SourceClassPC1:
    center: np.ndarray
    axis: np.ndarray

    def transform(self, curves: np.ndarray) -> np.ndarray:
        curves = np.asarray(curves, dtype=np.float64)
        if curves.ndim != 3 or curves.shape[-1] != self.axis.size:
            raise ValueError("curves must have shape [N, K, D]")
        return np.einsum("nkd,d->nk", curves - self.center, self.axis)


def fit_source_class_pc1(source_curves: np.ndarray) -> SourceClassPC1:
    """Fit one deterministic PC1 using source-class curves only."""
    source = np.asarray(source_curves, dtype=np.float64)
    if source.ndim != 3 or source.shape[0] == 0:
        raise ValueError("source_curves must be non-empty [N, K, D]")
    flat = source.reshape(-1, source.shape[-1])
    center = flat.mean(axis=0)
    covariance = (flat - center).T @ (flat - center)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0:
        axis = -axis
    return SourceClassPC1(center=center, axis=axis)


def interpolate_latent(
    positions: np.ndarray, features: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Linearly interpolate one irregular [L,D] Raw-PSE trajectory."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    features = np.asarray(features, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != positions.size:
        raise ValueError("features/positions must have shapes [L,D] and [L]")
    order = np.argsort(positions, kind="stable")
    x = positions[order]
    y = features[order]
    unique_x, first = np.unique(x, return_index=True)
    if unique_x.size < 2:
        raise ValueError("at least two distinct observation positions are required")
    if unique_x.size != x.size:
        sums = np.zeros((unique_x.size, y.shape[1]), dtype=np.float64)
        counts = np.zeros(unique_x.size, dtype=np.float64)
        inverse = np.searchsorted(unique_x, x)
        np.add.at(sums, inverse, y)
        np.add.at(counts, inverse, 1.0)
        y = sums / counts[:, None]
    else:
        y = y[first]
    return np.column_stack(
        [np.interp(grid, unique_x, y[:, channel]) for channel in range(y.shape[1])]
    )


def build_shift_views(
    grid: np.ndarray,
    source_curves: np.ndarray,
    target_curves: np.ndarray,
    timematch_shift: float,
    reconshift_shift: float,
) -> Dict[str, ShiftedCurves]:
    """Create three coordinate views while preserving all curve values."""
    grid = np.asarray(grid, dtype=np.float64)
    source = np.asarray(source_curves)
    target = np.asarray(target_curves)
    return {
        "raw": ShiftedCurves(grid, source, grid, target),
        "timematch": ShiftedCurves(
            grid, source, grid + float(timematch_shift), target
        ),
        "reconshift13": ShiftedCurves(
            grid, source, grid + float(reconshift_shift), target
        ),
    }


def _read_validation_scores(text: str) -> Sequence[float]:
    return [
        float(value)
        for value in re.findall(
            rf"^Validation result: .*?f1=({NUMBER})\s*$", text, re.MULTILINE
        )
    ]


def _trajectory_from_text(text: str) -> Tuple[Dict[int, float], str]:
    trajectory: Dict[int, float] = {}
    for line in text.splitlines():
        if not line.startswith("SHIFT_TRAJECTORY|"):
            continue
        fields = dict(
            item.split("=", 1)
            for item in line.split("|")[1:]
            if "=" in item
        )
        if "epoch" not in fields:
            continue
        shift = fields.get("actual_training_shift")
        if shift is None:
            shift = fields.get("target_to_source_shift")
        if shift is not None:
            trajectory[int(fields["epoch"])] = float(shift)
    if trajectory:
        return trajectory, "shift_trajectory_log"
    legacy = [
        float(value)
        for value in re.findall(
            rf"^Best AM Score shift ({NUMBER})\s+", text, re.MULTILINE
        )
    ]
    return {epoch: value for epoch, value in enumerate(legacy)}, "legacy_am_log"


def _structured_trajectory(output_dir: Optional[Path]):
    if output_dir is None or not Path(output_dir).is_dir():
        return None
    candidates = sorted(
        path
        for pattern in ("*shift*trajectory*.csv", "*temporal*shift*.csv")
        for path in Path(output_dir).rglob(pattern)
    )
    for path in candidates:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        trajectory = {}
        validation = {}
        for index, row in enumerate(rows):
            epoch = int(float(row.get("epoch", index)))
            shift = next(
                (
                    row[key]
                    for key in (
                        "actual_training_shift",
                        "target_to_source_shift",
                        "estimated_shift",
                        "shift_days",
                        "shift",
                    )
                    if row.get(key) not in (None, "")
                ),
                None,
            )
            if shift is not None:
                trajectory[epoch] = float(shift)
            score = next(
                (
                    row[key]
                    for key in ("validation_macro_f1", "val_macro_f1", "val_f1")
                    if row.get(key) not in (None, "")
                ),
                None,
            )
            if score is not None:
                validation[epoch] = float(score)
        if trajectory and validation:
            return path, trajectory, validation
    return None


def read_best_validation_shift(
    log_path: Path, output_dir: Optional[Path] = None
) -> ShiftSelection:
    """Read the recorded shift belonging to the highest validation Macro-F1."""
    structured = _structured_trajectory(output_dir)
    if structured is not None:
        path, trajectory, validation = structured
        best_epoch = max(validation, key=validation.get)
        if best_epoch not in trajectory:
            raise ValueError(f"best epoch {best_epoch} has no shift in {path}")
        return ShiftSelection(
            trajectory[best_epoch],
            best_epoch,
            validation[best_epoch],
            "best_validation_epoch",
            "structured_csv",
            str(path),
        )

    log_path = Path(log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"shift log not found: {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    scores = _read_validation_scores(text)
    trajectory, source_format = _trajectory_from_text(text)
    if not trajectory:
        raise ValueError(f"no temporal-shift records in {log_path}")
    if not scores:
        fallback_epoch = max(trajectory)
        return ShiftSelection(
            shift_days=trajectory[fallback_epoch],
            best_epoch=fallback_epoch,
            best_validation_f1=None,
            selection="last_recorded_epoch_fallback",
            source_format=source_format,
            source_path=str(log_path),
            fallback=True,
        )
    best_epoch = int(np.argmax(np.asarray(scores)))
    if best_epoch not in trajectory:
        raise ValueError(
            f"best validation epoch {best_epoch} has no corresponding shift in {log_path}"
        )
    return ShiftSelection(
        shift_days=trajectory[best_epoch],
        best_epoch=best_epoch,
        best_validation_f1=scores[best_epoch],
        selection="best_validation_epoch",
        source_format=source_format,
        source_path=str(log_path),
    )


def shared_ylimits(*curve_groups: np.ndarray) -> Tuple[float, float]:
    values = np.concatenate([np.asarray(group).reshape(-1) for group in curve_groups])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    low, high = float(values.min()), float(values.max())
    span = high - low
    padding = 0.05 * span if span > 0 else max(abs(low) * 0.05, 0.1)
    return low - padding, high + padding


def _stable_sample(curves: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    curves = np.asarray(curves)
    if len(curves) <= maximum:
        return curves
    rng = np.random.default_rng(seed)
    return curves[np.sort(rng.choice(len(curves), size=maximum, replace=False))]


def _plot_view(
    output_path: Path,
    task_name: str,
    class_name: str,
    config_key: str,
    view: ShiftedCurves,
    shift_days: float,
    ylim: Tuple[float, float],
    max_spaghetti: int,
    seed: int,
    delta_days: Optional[float] = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_sample = _stable_sample(view.source_y, max_spaghetti, seed)
    target_sample = _stable_sample(view.target_y, max_spaghetti, seed + 7919)
    source_median = np.median(view.source_y, axis=0)
    target_median = np.median(view.target_y, axis=0)
    source_q25, source_q75 = np.quantile(view.source_y, (0.25, 0.75), axis=0)
    target_q25, target_q75 = np.quantile(view.target_y, (0.25, 0.75), axis=0)
    labels = {
        "raw": "Raw PSE (no shift)",
        "timematch": "Original TimeMatch shift",
        "reconshift13": "Recon-guided TimeMatch shift",
    }
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for curve in source_sample:
        axes[0].plot(view.source_x, curve, color="#4C78A8", alpha=0.13, lw=0.7)
    for curve in target_sample:
        axes[0].plot(view.target_x, curve, color="#F28E2B", alpha=0.13, lw=0.7)
    axes[0].plot(view.source_x, source_median, color="#1F4E79", lw=2.4, label="source median")
    axes[0].plot(view.target_x, target_median, color="#B85C00", lw=2.4, label="target median")
    axes[0].legend(loc="best", frameon=False, ncol=2)
    axes[0].set_ylabel("Source-class PC1 score")
    axes[0].grid(alpha=0.2)

    axes[1].fill_between(view.source_x, source_q25, source_q75, color="#4C78A8", alpha=0.22)
    axes[1].fill_between(view.target_x, target_q25, target_q75, color="#F28E2B", alpha=0.22)
    axes[1].plot(view.source_x, source_median, color="#1F4E79", lw=2.4, label="source median + IQR")
    axes[1].plot(view.target_x, target_median, color="#B85C00", lw=2.4, label="target median + IQR")
    axes[1].set_xlabel("Day in source temporal frame")
    axes[1].set_ylabel("Source-class PC1 score")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="best", frameon=False, ncol=2)
    for axis in axes:
        axis.set_xlim(0.0, 365.0)
        axis.set_ylim(*ylim)
    subtitle = (
        f"source n={len(view.source_y)} | target n={len(view.target_y)} | "
        f"target shift={shift_days:+g} days"
    )
    if delta_days is not None:
        subtitle += f" | difference from Original TimeMatch={delta_days:+g} days"
    fig.suptitle(f"{task_name} | {class_name} | {labels[config_key]}\n{subtitle}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_task_class_figures(
    output_dir: Path,
    task_name: str,
    class_id: int,
    class_name: str,
    grid: np.ndarray,
    source_curves: np.ndarray,
    target_curves: np.ndarray,
    timematch_shift: float,
    reconshift_shift: float,
    max_spaghetti: int = 40,
    seed: int = 1,
) -> Dict[str, dict]:
    """Render the three comparable figures for one task/class."""
    source_curves = np.asarray(source_curves, dtype=np.float64)
    target_curves = np.asarray(target_curves, dtype=np.float64)
    if len(source_curves) == 0 or len(target_curves) == 0:
        raise ValueError("both source and target need at least one class sample")
    views = build_shift_views(
        grid, source_curves, target_curves, timematch_shift, reconshift_shift
    )
    ylim = shared_ylimits(source_curves, target_curves)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", class_name).strip("_")
    filename = f"{int(class_id):02d}_{safe_name}.png"
    shifts = {"raw": 0.0, "timematch": timematch_shift, "reconshift13": reconshift_shift}
    metadata = {}
    for key, view in views.items():
        output_path = Path(output_dir) / CONFIG_FOLDERS[key] / filename
        _plot_view(
            output_path,
            task_name,
            class_name,
            key,
            view,
            shifts[key],
            ylim,
            max_spaghetti,
            seed + int(class_id),
            (reconshift_shift - timematch_shift) if key == "reconshift13" else None,
        )
        metadata[key] = {"path": str(output_path), "ylim": [ylim[0], ylim[1]]}
    return metadata


def replay_train_indices(
    dataset_names: Sequence[str],
    eligible_indices: Mapping[str, Iterable[int]],
    seed: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Dict[str, Dict[str, set]]:
    """Replay fold-0 splitting without importing the training entry module."""
    rng = random.Random(seed)
    splits = {}
    for name in dataset_names:
        indices = list(eligible_indices[name])
        n_test, n_val = int(test_ratio * len(indices)), int(val_ratio * len(indices))
        n_train = len(indices) - n_test - n_val
        rng.shuffle(indices)
        splits[name] = {
            "train": set(indices[:n_train]),
            "val": set(indices[n_train : n_train + n_val]),
            "test": set(indices[n_train + n_val :]),
        }
    return splits


def write_task_outputs(
    output_dir: Path, manifest: Mapping, summary_rows: Sequence[Mapping]
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(dict(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fieldnames = [
        "class_id",
        "class_name",
        "source_count",
        "target_count",
        "timematch_shift_days",
        "reconshift_shift_days",
    ]
    with (output_dir / "shifts_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
