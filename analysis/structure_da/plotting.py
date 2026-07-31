"""Headless matplotlib rendering for Structure DA diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .log_parser import ParsedRun


DOMAIN_ORDER = ("DK1", "FR2", "FR1", "AT1")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _heatmap(frame: pd.DataFrame, value: str, path: Path, title: str) -> None:
    matrix = np.full((4, 4), np.nan)
    counts = np.zeros((4, 4), dtype=int)
    for row in frame.itertuples(index=False):
        i, j = DOMAIN_ORDER.index(row.source), DOMAIN_ORDER.index(row.target)
        matrix[i, j] = getattr(row, value) * 100
        counts[i, j] = row.n_completed_seeds
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap="viridis")
    for i in range(4):
        for j in range(4):
            if np.isfinite(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:.1f}\n(n={counts[i, j]})", ha="center", va="center", color="white" if matrix[i, j] < np.nanmean(matrix) else "black")
    axis.set_xticks(range(4), DOMAIN_ORDER)
    axis.set_yticks(range(4), DOMAIN_ORDER)
    axis.set_xlabel("Target")
    axis.set_ylabel("Source")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="percentage points")
    _save(fig, path)


def _asymmetry(frame: pd.DataFrame, path: Path) -> None:
    lookup = {(row.source, row.target): row.target_macro_f1_mean for row in frame.itertuples(index=False)}
    matrix = np.full((4, 4), np.nan)
    for i, source in enumerate(DOMAIN_ORDER):
        for j, target in enumerate(DOMAIN_ORDER):
            if (source, target) in lookup and (target, source) in lookup:
                matrix[i, j] = 100 * (lookup[source, target] - lookup[target, source])
    limit = max(1.0, float(np.nanmax(np.abs(matrix)))) if np.isfinite(matrix).any() else 1.0
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap="coolwarm", vmin=-limit, vmax=limit)
    for i in range(4):
        for j in range(4):
            if np.isfinite(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center")
    axis.set_xticks(range(4), DOMAIN_ORDER)
    axis.set_yticks(range(4), DOMAIN_ORDER)
    axis.set_xlabel("Reverse target")
    axis.set_ylabel("Direction source")
    axis.set_title("Directional asymmetry: F1(A→B) − F1(B→A)")
    fig.colorbar(image, ax=axis, label="percentage points")
    _save(fig, path)


def _source_target_scatter(frame: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 5.5))
    if not frame.empty:
        axis.scatter(100 * frame["best_source_val_f1"], 100 * frame["target_macro_f1"], alpha=0.65)
        for row in frame.itertuples(index=False):
            axis.annotate(f"{row.source}→{row.target} s{row.seed}", (100 * row.best_source_val_f1, 100 * row.target_macro_f1), fontsize=6, alpha=0.8)
    axis.set_xlabel("Best source validation Macro-F1 (%)")
    axis.set_ylabel("Target test Macro-F1 (%)")
    axis.set_title("Source validation versus target performance (descriptive)")
    axis.grid(alpha=0.25)
    _save(fig, path)


def _training_figures(run: ParsedRun, root: Path) -> None:
    if not run.epochs:
        return
    epochs = np.asarray([item.epoch for item in run.epochs])
    val = np.asarray([item.val_macro_f1 if item.val_macro_f1 is not None else np.nan for item in run.epochs])
    run_dir = root / run.run_name

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(epochs, [item.task for item in run.epochs])
    axes[0].set_ylabel("Task loss")
    axes[1].plot(epochs, 100 * val)
    axes[1].set_ylabel("Source-val Macro-F1 (%)")
    axes[1].set_xlabel("Epoch")
    fig.suptitle(run.run_name)
    _save(fig, run_dir / "classification_and_validation.png")

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, [item.quality for item in run.epochs], label="quality")
    axis.plot(epochs, [item.geometry for item in run.epochs], label="geometry")
    axis.plot(epochs, [item.alignment for item in run.epochs], label="alignment")
    axis.set_xlabel("Epoch"); axis.set_ylabel("Loss"); axis.set_title(f"{run.run_name}: auxiliary losses")
    axis.legend(fontsize=8)
    _save(fig, run_dir / "quality_losses.png")

    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.plot(epochs, [item.domain_accuracy for item in run.epochs])
    axis.axhline(0.5, linestyle="--", color="black", alpha=0.6)
    axis.set_xlabel("Epoch"); axis.set_ylabel("Accuracy")
    axis.set_title(f"{run.run_name}: domain accuracy")
    _save(fig, run_dir / "domain_accuracy.png")

    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.plot(epochs, 100 * val)
    finite = np.flatnonzero(np.isfinite(val))
    if finite.size:
        best_index = finite[np.argmax(val[finite])]
        final_index = finite[-1]
        axis.scatter(epochs[best_index], 100 * val[best_index], marker="*", s=120, label=f"best e{epochs[best_index]}={100 * val[best_index]:.1f}")
        axis.scatter(epochs[final_index], 100 * val[final_index], marker="x", s=70, label=f"final={100 * val[final_index]:.1f}")
        axis.legend()
    axis.set_xlabel("Epoch"); axis.set_ylabel("Source-val Macro-F1 (%)"); axis.set_title(run.run_name)
    _save(fig, run_dir / "validation_f1.png")


def class_metric_plot_data(
    task: pd.DataFrame, class_name: str, metric: str,
) -> tuple[list[tuple[int, float, int]], float, float, tuple[int, ...]]:
    """Keep every seed point while excluding support-zero seeds from summaries."""

    rows = task.loc[
        task["class_name"] == class_name, ["seed", metric, "support"]
    ].sort_values("seed")
    points = [
        (int(row.seed), float(getattr(row, metric)), int(row.support))
        for row in rows.itertuples(index=False)
    ]
    valid = np.asarray([value for _, value, support in points if support > 0], dtype=float)
    mean = float(valid.mean()) if len(valid) else float("nan")
    std = float(valid.std(ddof=0)) if len(valid) else float("nan")
    absent = tuple(seed for seed, _, support in points if support == 0)
    return points, mean, std, absent


def _class_figures(frame: pd.DataFrame, root: Path) -> None:
    if frame.empty:
        return
    for (source, target), task in frame.groupby(["source", "target"], sort=False):
        classes = list(dict.fromkeys(task["class_name"]))
        y = np.arange(len(classes))
        for metric in ("f1", "recall", "precision"):
            fig, axis = plt.subplots(figsize=(8, max(4.5, 0.38 * len(classes))))
            means, stds = [], []
            absent_by_class: dict[str, tuple[int, ...]] = {}
            for class_name in classes:
                points, mean, std, absent_seeds = class_metric_plot_data(task, class_name, metric)
                means.append(100 * mean)
                stds.append(100 * std)
                absent_by_class[class_name] = absent_seeds
                for seed, value, support in points:
                    axis.scatter(
                        100 * value, classes.index(class_name), s=28, alpha=0.7,
                        marker="x" if support == 0 else "o",
                        color="tab:red" if support == 0 else None,
                        label=f"seed {seed}" if classes.index(class_name) == 0 else None,
                    )
            axis.errorbar(means, y, xerr=stds, fmt="o", color="black", capsize=3, label="mean ± std")
            labels = [
                f"{name} (absent: {','.join(f's{seed}' for seed in absent_by_class[name])})"
                if absent_by_class[name] else name
                for name in classes
            ]
            axis.set_yticks(y, labels)
            axis.set_xlabel(f"Target {metric} (%)")
            axis.set_title(f"{source}→{target}: class {metric}")
            axis.legend(fontsize=8)
            filename = f"{source}_to_{target}_class_{metric}.png" if metric == "f1" else f"{source}_to_{target}_class_{metric}.png"
            _save(fig, root / filename)


def _confusion_image(matrix: np.ndarray, classes: tuple[str, ...], path: Path, title: str, normalize: bool) -> None:
    values = matrix.astype(float)
    if normalize:
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.full_like(values, np.nan), where=totals > 0)
    fig, axis = plt.subplots(figsize=(max(6, 0.55 * len(classes)), max(5, 0.5 * len(classes))))
    image = axis.imshow(np.ma.masked_invalid(values), cmap="Blues")
    axis.set_xticks(range(len(classes)), classes, rotation=90)
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted"); axis.set_ylabel("True"); axis.set_title(title)
    fig.colorbar(image, ax=axis)
    _save(fig, path)


def _confusion_figures(runs: Iterable[ParsedRun], root: Path) -> None:
    grouped: dict[tuple[str, str, tuple[str, ...]], list[np.ndarray]] = {}
    for run in runs:
        if run.status != "completed" or run.confusion is None:
            continue
        _confusion_image(run.confusion, run.classes, root / f"{run.run_name}_counts.png", f"{run.run_name}: counts", False)
        _confusion_image(run.confusion, run.classes, root / f"{run.run_name}_row_normalized.png", f"{run.run_name}: row-normalized", True)
        grouped.setdefault((run.source, run.target, run.classes), []).append(run.confusion)
    for (source, target, classes), matrices in grouped.items():
        aggregate = np.sum(matrices, axis=0)
        _confusion_image(aggregate, classes, root / f"{source}_to_{target}_aggregate.png", f"{source}→{target}: aggregate row-normalized", True)


def plot_log_diagnostics(runs: list[ParsedRun], tables: dict[str, pd.DataFrame], figure_root: Path) -> None:
    """Render all figures from already parsed records/tables."""

    overview = figure_root / "overview"
    task = tables["task_summary"]
    if not task.empty:
        for value, filename, title in (
            ("target_macro_f1_mean", "target_macro_f1_mean.png", "Target Macro-F1 mean"),
            ("target_macro_f1_std", "target_macro_f1_std.png", "Target Macro-F1 standard deviation"),
            ("best_source_val_f1_mean", "best_source_val_f1_mean.png", "Best source-val Macro-F1 mean"),
            ("source_target_f1_gap_mean", "source_target_f1_gap.png", "Source-val minus target Macro-F1"),
        ):
            _heatmap(task, value, overview / filename, title)
        _asymmetry(task, overview / "directional_asymmetry.png")
    _source_target_scatter(tables["run_summary"], overview / "source_val_vs_target_f1.png")
    for run in runs:
        _training_figures(run, figure_root / "training")
    _class_figures(tables["per_class_metrics"], figure_root / "classes")
    _confusion_figures(runs, figure_root / "confusion")
