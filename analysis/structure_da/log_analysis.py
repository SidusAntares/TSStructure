"""Normalize parsed task logs into reproducible tables and figures."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .log_parser import ParsedRun, parse_task_log


DOMAIN_ORDER = ("DK1", "FR2", "FR1", "AT1")
DEFAULT_SEEDS = (1, 2, 3)


def _population_std(values: pd.Series) -> float:
    return float(values.std(ddof=0)) if len(values) else float("nan")


def _diagnostic_rows(
    runs: Iterable[ParsedRun], attribute: str
) -> list[dict[str, object]]:
    rows = []
    for run in runs:
        for values in getattr(run, attribute):
            rows.append(
                {
                    "run_name": run.run_name,
                    "source": run.source,
                    "target": run.target,
                    "seed": run.seed,
                    **values,
                }
            )
    return rows


def build_analysis_tables(runs: Iterable[ParsedRun]) -> dict[str, pd.DataFrame]:
    """Build all normalized log tables without reading or plotting files."""

    runs = list(runs)
    status_rows, summary_rows, epoch_rows = [], [], []
    class_rows, confusion_rows, loss_rows = [], [], []
    for run in runs:
        status_rows.append({
            "run_name": run.run_name, "source": run.source, "target": run.target,
            "seed": run.seed, "status": run.status, "num_classes": run.num_classes,
            "epochs_seen": len(run.epochs), "has_test_result": run.target_macro_f1 is not None,
        })
        for epoch in run.epochs:
            epoch_rows.append({
                "run_name": run.run_name, "source": run.source, "target": run.target,
                "seed": run.seed, "epoch": epoch.epoch, "total": epoch.total,
                "task": epoch.task, "quality": epoch.quality,
                "quality_structural_cls": epoch.quality_structural_cls,
                "quality_structural_domain": epoch.quality_structural_domain,
                "quality_component_cls": epoch.quality_component_cls,
                "quality_component_domain": epoch.quality_component_domain,
                "geometry": epoch.geometry, "alignment": epoch.alignment,
                "train_accuracy": epoch.train_accuracy,
                "domain_accuracy": epoch.domain_accuracy,
                "alpha_T": epoch.alpha_T, "alpha_D": epoch.alpha_D,
                "alpha_R": epoch.alpha_R,
                "beta_T_temp": epoch.beta_T_temp,
                "beta_D_temp": epoch.beta_D_temp,
                "grl": epoch.grl, "lr": epoch.lr, "val_loss": epoch.val_loss,
                "val_accuracy": epoch.val_accuracy, "val_macro_f1": epoch.val_macro_f1,
            })
        if run.status == "completed" and run.target_macro_f1 is not None:
            best = run.best_epoch_record
            final = next((epoch for epoch in reversed(run.epochs) if epoch.val_macro_f1 is not None), None)
            summary_rows.append({
                "run_name": run.run_name, "source": run.source, "target": run.target,
                "seed": run.seed,
                "best_source_val_f1": best.val_macro_f1 if best else np.nan,
                "best_source_val_epoch": best.epoch if best else np.nan,
                "best_source_val_accuracy": best.val_accuracy if best else np.nan,
                "final_source_val_f1": final.val_macro_f1 if final else np.nan,
                "target_accuracy": run.target_accuracy,
                "target_macro_f1": run.target_macro_f1,
                "source_target_f1_gap": (best.val_macro_f1 - run.target_macro_f1) if best else np.nan,
                "num_classes": run.num_classes,
                "steps_per_epoch": run.epochs[0].steps if run.epochs else run.config.get("steps_per_epoch"),
            })
            if run.epochs:
                values = {
                    name: np.asarray([
                        getattr(epoch, name) for epoch in run.epochs
                    ])
                    for name in (
                        "task", "quality", "geometry", "alignment",
                    )
                }
                loss_row = {"run_name": run.run_name}
                for name, array in values.items():
                    loss_row.update({f"{name}_epoch1": array[0], f"{name}_final": array[-1], f"{name}_min": array.min()})
                loss_row.update({
                    "best_val_epoch": best.epoch if best else np.nan,
                    "best_val_f1": best.val_macro_f1 if best else np.nan,
                    "final_val_f1": final.val_macro_f1 if final else np.nan,
                    "val_drop_from_best": (best.val_macro_f1 - final.val_macro_f1) if best and final else np.nan,
                })
                loss_rows.append(loss_row)
            for metric in run.class_metrics:
                class_rows.append({
                    "run_name": run.run_name, "source": run.source, "target": run.target,
                    "seed": run.seed, "class_name": metric.class_name,
                    "precision": metric.precision, "recall": metric.recall,
                    "f1": metric.f1, "support": metric.support,
                })
            if run.confusion is not None:
                for true_index, true_class in enumerate(run.classes):
                    for pred_index, pred_class in enumerate(run.classes):
                        confusion_rows.append({
                            "run_name": run.run_name, "source": run.source,
                            "target": run.target, "seed": run.seed,
                            "true_class": true_class, "pred_class": pred_class,
                            "count": int(run.confusion[true_index, pred_index]),
                        })

    run_summary = pd.DataFrame(summary_rows)
    task_rows = []
    if not run_summary.empty:
        for (source, target), group in run_summary.groupby(["source", "target"], sort=False):
            row = {"source": source, "target": target, "n_completed_seeds": len(group)}
            for column in ("target_accuracy", "target_macro_f1", "best_source_val_f1", "source_target_f1_gap"):
                row[f"{column}_mean"] = float(group[column].mean())
                row[f"{column}_std"] = _population_std(group[column])
            task_rows.append(row)
    correlation_rows = []
    if len(run_summary) >= 2:
        x = run_summary["best_source_val_f1"].to_numpy(dtype=float)
        y = run_summary["target_macro_f1"].to_numpy(dtype=float)
        if np.ptp(x) > 0 and np.ptp(y) > 0:
            correlation_rows = [
                {"method": "pearson", "correlation": float(pearsonr(x, y).statistic), "n_runs": len(x)},
                {"method": "spearman", "correlation": float(spearmanr(x, y).statistic), "n_runs": len(x)},
            ]
    step_history = pd.DataFrame(_diagnostic_rows(runs, "steps"))
    structure_rows = _diagnostic_rows(runs, "structure_diagnostics")
    decomposition_rows = []
    relation_rows = []
    metadata = {"run_name", "source", "target", "seed", "epoch"}
    for row in structure_rows:
        decomposition_rows.append(
            {
                key: value
                for key, value in row.items()
                if key in metadata
                or key.startswith("tau_")
                or "energy_" in key
                or "reconstruction" in key
            }
        )
        relation_rows.append(
            {
                key: value
                for key, value in row.items()
                if key in metadata
                or (
                    not key.startswith("tau_")
                    and "energy_" not in key
                    and "reconstruction" not in key
                )
            }
        )
    return {
        "run_status": pd.DataFrame(status_rows),
        "run_summary": run_summary,
        "epoch_history": pd.DataFrame(epoch_rows),
        "per_class_metrics": pd.DataFrame(class_rows),
        "confusion_long": pd.DataFrame(confusion_rows),
        "task_summary": pd.DataFrame(task_rows),
        "correlation_summary": pd.DataFrame(correlation_rows),
        "loss_diagnostics": pd.DataFrame(loss_rows),
        "step_history": step_history,
        "decomposition_diagnostics": pd.DataFrame(decomposition_rows),
        "structure_diagnostics": pd.DataFrame(relation_rows),
        "quality_diagnostics": pd.DataFrame(
            _diagnostic_rows(runs, "quality_diagnostics")
        ),
        "geometry_diagnostics": pd.DataFrame(
            _diagnostic_rows(runs, "geometry_diagnostics")
        ),
    }


def _add_missing_statuses(status: pd.DataFrame) -> pd.DataFrame:
    existing = set(status["run_name"]) if not status.empty else set()
    missing = []
    for source in DOMAIN_ORDER:
        for target in DOMAIN_ORDER:
            if source == target:
                continue
            for seed in DEFAULT_SEEDS:
                run_name = f"{source}_to_{target}_seed{seed}"
                if run_name not in existing:
                    missing.append({
                        "run_name": run_name, "source": source, "target": target,
                        "seed": seed, "status": "missing", "num_classes": np.nan,
                        "epochs_seen": 0, "has_test_result": False,
                    })
    return pd.concat([status, pd.DataFrame(missing)], ignore_index=True)


def analyze_logs(experiment_dir: Path | str, output_dir: Path | str) -> dict[str, object]:
    """Parse a log directory, save normalized CSV/PNG outputs, and return counts."""

    experiment_dir, output_dir = Path(experiment_dir), Path(output_dir)
    paths = sorted(experiment_dir.glob("*_to_*_seed*.log"))
    runs = [parse_task_log(path) for path in paths]
    tables = build_analysis_tables(runs)
    tables["run_status"] = _add_missing_statuses(tables["run_status"])
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)

    from .plotting import plot_log_diagnostics

    plot_log_diagnostics(runs, tables, output_dir / "figures")
    status_counts = tables["run_status"]["status"].value_counts().to_dict()
    manifest = {
        "command": "logs", "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_experiment_dir": str(experiment_dir), "data_root": None,
        "task_logs_used": [str(path) for path in paths],
        "completed": int(status_counts.get("completed", 0)),
        "incomplete": int(status_counts.get("incomplete", 0)),
        "failed": int(status_counts.get("failed", 0)),
        "missing": int(status_counts.get("missing", 0)),
        "diagnostic_sampling_seed": None, "samples_per_class": None,
        "checkpoint_path": None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"runs": runs, "tables": tables, "manifest": manifest}
