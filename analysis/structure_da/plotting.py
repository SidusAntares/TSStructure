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


def _plot_lines(
    x: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for label, values in series:
        if np.isfinite(values).any():
            axis.plot(x, values, label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if axis.lines:
        axis.legend(fontsize=8, ncol=2)
    _save(fig, path)


def _record_series(records: list[dict], name: str) -> np.ndarray:
    return np.asarray([record.get(name, np.nan) for record in records], dtype=float)


def _training_figures(run: ParsedRun, root: Path) -> None:
    if not run.epochs:
        return
    epochs = np.asarray([item.epoch for item in run.epochs])
    run_dir = root / run.run_name
    val = np.asarray([
        item.val_macro_f1 if item.val_macro_f1 is not None else np.nan
        for item in run.epochs
    ])
    _plot_lines(
        epochs,
        [
            ("total", np.asarray([item.total for item in run.epochs])),
            ("task", np.asarray([item.task for item in run.epochs])),
            ("quality", np.asarray([item.quality for item in run.epochs])),
            ("quality structural cls", np.asarray([item.quality_structural_cls for item in run.epochs])),
            ("quality structural domain", np.asarray([item.quality_structural_domain for item in run.epochs])),
            ("quality component cls", np.asarray([item.quality_component_cls for item in run.epochs])),
            ("quality component domain", np.asarray([item.quality_component_domain for item in run.epochs])),
            ("geometry", np.asarray([item.geometry for item in run.epochs])),
            ("alignment", np.asarray([item.alignment for item in run.epochs])),
        ],
        run_dir / "losses.png",
        f"{run.run_name}: training losses",
        "Loss",
    )
    _plot_lines(
        epochs,
        [
            ("source train accuracy", 100 * np.asarray([item.train_accuracy for item in run.epochs])),
            ("source val Macro-F1", 100 * val),
        ],
        run_dir / "source_train_accuracy_and_validation.png",
        f"{run.run_name}: source training and validation",
        "Percent",
    )
    _plot_lines(
        epochs,
        [
            ("domain accuracy", np.asarray([item.domain_accuracy for item in run.epochs])),
            ("GRL coefficient", np.asarray([item.grl for item in run.epochs])),
        ],
        run_dir / "domain_accuracy_and_grl.png",
        f"{run.run_name}: domain accuracy and GRL",
        "Value",
    )

    structure = run.structure_diagnostics
    if structure:
        structure_epochs = _record_series(structure, "epoch")
        _plot_lines(
            structure_epochs,
            [
                (f"{domain} {component}", _record_series(structure, f"energy_{component}_{domain}"))
                for domain in ("s", "t")
                for component in ("T", "D", "R")
            ],
            run_dir / "energy_fractions.png",
            f"{run.run_name}: decomposition energy fractions",
            "Fraction",
        )
        _plot_lines(
            structure_epochs,
            [
                (f"{domain} {branch}", _record_series(structure, f"{branch}_fusion_norm_{domain}"))
                for domain in ("s", "t")
                for branch in ("raw", "temporal", "channel")
            ],
            run_dir / "fusion_norms.png",
            f"{run.run_name}: fusion feature norms",
            "Mean L2 norm",
        )
        _plot_lines(
            structure_epochs,
            [
                (f"{domain} {operator} {component}", _record_series(structure, f"{operator}_{component}_valid_{domain}"))
                for domain in ("s", "t")
                for operator in ("temporal", "channel")
                for component in ("T", "D")
            ],
            run_dir / "structure_valid_rates.png",
            f"{run.run_name}: structure valid rates",
            "Valid rate",
        )

    quality = run.quality_diagnostics
    if quality:
        quality_epochs = _record_series(quality, "epoch")
        _plot_lines(
            quality_epochs,
            [
                (f"{domain} alpha {component}", _record_series(quality, f"alpha_{component}_{domain}"))
                for domain in ("s", "t")
                for component in ("T", "D", "R")
            ],
            run_dir / "quality_alpha.png",
            f"{run.run_name}: component quality alpha",
            "Coefficient",
        )
        _plot_lines(
            quality_epochs,
            [
                (f"{domain} beta {component} {operator}", _record_series(quality, f"beta_{component}_{operator}_{domain}"))
                for domain in ("s", "t")
                for component in ("T", "D")
                for operator in ("temporal", "channel")
            ],
            run_dir / "quality_beta.png",
            f"{run.run_name}: structure quality beta",
            "Coefficient",
        )

    geometry = run.geometry_diagnostics
    if geometry:
        geometry_epochs = _record_series(geometry, "epoch")
        _plot_lines(
            geometry_epochs,
            [
                (f"{component} {loss}", _record_series(geometry, f"{component}_{loss}"))
                for component in ("T", "D")
                for loss in ("align", "rough", "unsupported", "center")
            ],
            run_dir / "geometry_losses.png",
            f"{run.run_name}: temporal geometry losses",
            "Loss",
        )


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
