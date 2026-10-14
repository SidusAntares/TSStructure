#!/usr/bin/env python3
"""Two-GPU dependency scheduler for G class-organization A/B, Seed1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_STATUS_LOCK = threading.Lock()


@dataclass(frozen=True)
class Pipeline:
    task: str
    source_alias: str
    source: str
    target_alias: str
    target: str


def pipelines():
    return (
        Pipeline("FR1_FR2", "FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
        Pipeline("FR2_DK1", "FR2", "france/31TCJ/2017", "DK1", "denmark/32VNH/2017"),
    )


def source_assignments(gpus):
    """Train the two independent Source models on the first two GPUs."""
    if len(gpus) != 4:
        raise ValueError("the class-organization experiment requires exactly four GPUs")
    by_source = {pipeline.source_alias: pipeline for pipeline in pipelines()}
    return (
        (str(gpus[0]), by_source["FR1"]),
        (str(gpus[1]), by_source["FR2"]),
    )


def uda_assignments(gpus):
    """Cross the Source-training cards, then occupy all four GPUs for UDA."""
    if len(gpus) != 4:
        raise ValueError("the class-organization experiment requires exactly four GPUs")
    by_task = {pipeline.task: pipeline for pipeline in pipelines()}
    return (
        # The FR1 Source card deliberately receives an FR2 UDA task.
        (str(gpus[0]), "A", by_task["FR2_DK1"]),
        # The FR2 Source card deliberately receives an FR1 UDA task.
        (str(gpus[1]), "A", by_task["FR1_FR2"]),
        (str(gpus[2]), "B", by_task["FR1_FR2"]),
        (str(gpus[3]), "B", by_task["FR2_DK1"]),
    )


def _model_args():
    return [
        "--model", "psestructureprotoltae", "--structure-branch", "true",
        "--structure-exposer", "fourier", "--fourier_num_modes", "13",
        "--shape-dim", "128", "--shape-window-scales", "24",
        "--shape-window-stride", "8", "--shapelet-count", "16",
        "--shapelet-beta", "5", "--shape-resample-length", "16",
        "--shape-representation", "phase_moment", "--shape-injection", "current_query",
        "--phase-query-view", "full", "--shape-phase-k2-enabled", "true",
        "--structure-shift-mode", "none", "--shapelet-diversity-margin", "0.5",
        "--shapelet-diversity-weight", "0.01", "--shape-class-weight", "0.1",
        "--source-minority-mode", "base",
    ]


def source_directory(args, pipeline):
    return args.root / "outputs/source" / f"SRC_ORG_{pipeline.source_alias}_seed1"


def reference_path(args, pipeline):
    return source_directory(args, pipeline) / "fold_0/class_organization_reference.pt"


def summary_paths(root):
    root = Path(root)
    return tuple(root / name for name in (
        "Source_Org_FR1_metrics.csv", "Source_Org_FR2_metrics.csv",
        "A_B_2tasks_seed1.csv", "per_class_comparison.csv",
        "organization_diagnostics.csv", "FINAL_report.md",
    ))


def source_command(args, pipeline):
    experiment = f"SRC_ORG_{pipeline.source_alias}_seed1"
    return [
        args.python, "-u", "train.py", "-e", experiment,
        "--data_root", args.data_root, "--source", pipeline.source,
        "--target", pipeline.source, *_model_args(),
        "--class-org-source-weight", "0.02", "--class-org-target-weight", "0.0",
        "--class-org-warmup-epochs", "10", "--class-org-min-support", "8",
        "--with_shift_aug", "false", "--seed", "1", "--num_folds", "1",
        "--epochs", "100", "--batch_size", "128", "--lr", "0.001",
        "--weight_decay", "0.0001", "--focal_loss_gamma", "1.0",
        "--seq_length", "30", "--num_pixels", "64", "--closed_set", "true",
        "--progress_bar", "off", "--output_dir", str(args.root / "outputs/source"),
        "--tensorboard_log_dir", str(args.root / f"runs/{experiment}"),
    ]


def uda_command(args, pipeline, variant):
    if variant not in ("A", "B"):
        raise ValueError("variant must be A or B")
    experiment = f"{variant}_{pipeline.task}_seed1"
    target_weight = "0.0" if variant == "A" else "0.02"
    weights = source_directory(args, pipeline)
    return [
        args.python, "-u", "train.py", "-e", experiment,
        "--data_root", args.data_root, "--source", pipeline.source,
        "--target", pipeline.target, *_model_args(),
        "--shape-align-weight", "0", "--class-org-source-weight", "0.0",
        "--class-org-target-weight", target_weight,
        "--class-org-target-start-epoch", "5",
        "--class-org-reference", str(reference_path(args, pipeline)),
        "--with_shift_aug", "false", "--seed", "1", "--num_folds", "1",
        "--batch_size", "128", "--seq_length", "30", "--num_pixels", "64",
        "--closed_set", "true", "--progress_bar", "off",
        "--output_dir", str(args.root / "outputs/uda"),
        "--tensorboard_log_dir", str(args.root / f"runs/{experiment}"),
        "timematch", "--weights", str(weights), "--epochs", "20",
        "--steps_per_epoch", "500", "--lr", "0.0001",
        "--pseudo_threshold", "0.9", "--ema_decay", "0.9999",
        "--trade_off", "2.0", "--estimate_shift", "true",
        "--balance_source", "true", "--use_focal_loss", "true",
        "--shift_source", "true", "--sample_size", "100",
        "--max_temporal_shift", "60", "--domain_specific_bn", "true",
        "--shift_estimator", "AM", "--run_validation", "--output_student", "true",
        "--shape-da-mode", "batch_align", "--shape-alignment-view", "none",
        "--oracle-pseudo-labels", "false", "--adaptive-pseudo-selection", "false",
        "--shape-equivariance-weight", "0.05", "--shape-equivariance-max-shift", "60",
        "--detach-target-structure", "false", "--target-structure-detach-epochs", "5",
    ]


def diagnostics_commands(args, pipeline, variant):
    """Build the two frozen D1-D4 stages for one completed A/B checkpoint."""
    scope = f"{variant}_{pipeline.task}"
    fold = args.root / f"outputs/uda/{variant}_{pipeline.task}_seed1/fold_0"
    checkpoint = fold / "checkpoint_last.pt"
    source_checkpoint = source_directory(args, pipeline) / "fold_0/model.pt"
    cache = args.root / f"diagnostics/cache/{scope}"
    output = args.root / "diagnostics"
    common = [
        args.python, "-u", "-m", "analysis.structure_g_response_organization_audit",
        "--stage", "uda_final", "--variant", variant, "--scope", scope,
        "--task", pipeline.task, "--checkpoint", str(checkpoint),
        "--cache-dir", str(cache), "--output-root", str(output),
    ]
    extract = common + [
        "--mode", "extract", "--source-checkpoint", str(source_checkpoint),
        "--data-root", args.data_root, "--device", "cuda",
        "--batch-size", str(args.diagnostic_batch_size),
        "--pixel-budget", str(args.diagnostic_pixel_budget),
    ]
    audit = common + ["--mode", "audit", "--device", "cpu"]
    return (
        ("extract", extract, cache / "checkpoint_manifest.json"),
        ("audit", audit, output / "scopes" / scope / "scope_manifest.json"),
    )


def _config(packet, name, default=None):
    config = packet.get("config", {})
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _artifact_status_unlocked(fold, kind, variant=None):
    fold = Path(fold)
    if not fold.exists():
        return "missing"
    last = fold / "checkpoint_last.pt"
    if not last.is_file() or not (fold / "model.pt").is_file():
        return "partial"
    try:
        import torch

        packet = torch.load(last, map_location="cpu", weights_only=False)
        expected_epoch = 99 if kind == "source" else 19
        if int(packet.get("epoch", -1)) != expected_epoch:
            return "partial"
        if _config(packet, "shape_representation") != "phase_moment":
            return "incompatible"
        if kind == "source":
            reference = fold / "class_organization_reference.pt"
            training_reference = fold / "class_organization_reference_training.pt"
            if not reference.is_file() or not training_reference.is_file():
                return "partial"
            from methods.structure_da.class_organization_loss import load_organization_reference

            restored = load_organization_reference(
                reference, source_checkpoint=fold / "model.pt",
                expected_class_ids=list(range(len(_config(packet, "classes", [])))),
                expected_anchor_count=16, expected_response_dim=96,
            )
            if not bool(restored.metadata.get("source_org_effective", False)):
                return "ineffective"
        else:
            expected_weight = 0. if variant == "A" else .02
            if float(_config(packet, "class_org_target_weight", -1.)) != expected_weight:
                return "incompatible"
            if int(_config(packet, "target_structure_detach_epochs", -1)) != 5:
                return "incompatible"
    except Exception as error:
        print(
            f"ARTIFACT_CHECK_ERROR|path={fold}|kind={kind}|variant={variant}|"
            f"error_type={type(error).__name__}|error={error}",
            flush=True,
        )
        return "partial"
    return "complete"


def artifact_status(fold, kind, variant=None):
    """Validate serialized artifacts without concurrent torch deserialization."""
    with _ARTIFACT_STATUS_LOCK:
        return _artifact_status_unlocked(fold, kind, variant)


def _run(command, log, gpu, dry_run=False):
    rendered = " ".join(map(str, command))
    print(f"COMMAND|gpu={gpu}|{rendered}", flush=True)
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"COMMAND|gpu={gpu}|{rendered}\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"command failed with exit={result.returncode}; log={log}")


def _require_or_run(
    args, command, fold, kind, label, gpu, variant=None,
    prechecked_status=None, verify_after=True,
):
    status = (
        prechecked_status
        if prechecked_status is not None
        else artifact_status(fold, kind, variant)
    )
    if status == "complete":
        if args.mode == "run":
            raise RuntimeError(f"completed artifact exists; use resume: {fold}")
        print(f"REUSE|gpu={gpu}|kind={kind}|label={label}|path={fold}", flush=True)
        return
    if status != "missing" and args.mode != "dry-run":
        raise RuntimeError(
            f"{kind} artifact is {status}; refusing overwrite and stopping downstream: {fold}"
        )
    _run(command, args.root / f"logs/{label}.log", gpu, args.mode == "dry-run")
    produced_status = (
        artifact_status(fold, kind, variant)
        if verify_after and args.mode != "dry-run"
        else None
    )
    if verify_after and args.mode != "dry-run" and produced_status != "complete":
        raise RuntimeError(
            f"{kind} did not produce a complete compatible artifact: {fold} "
            f"(status={produced_status})"
        )


def _run_diagnostics(args, pipeline, variant, gpu):
    for stage, command, artifact in diagnostics_commands(args, pipeline, variant):
        if artifact.is_file():
            print(
                f"REUSE|gpu={gpu}|kind=diagnostic_{stage}|"
                f"label={variant}_{pipeline.task}|path={artifact}", flush=True,
            )
            continue
        log = args.root / f"logs/{variant}_{pipeline.task}_diagnostic_{stage}.log"
        _run(command, log, gpu, args.mode == "dry-run")
        if args.mode != "dry-run" and not artifact.is_file():
            raise RuntimeError(
                f"diagnostic {stage} did not produce expected artifact: {artifact}"
            )


def _write_csv(path, rows):
    rows = list(rows)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _config_dict(packet):
    raw = packet.get("config", {})
    return dict(raw if isinstance(raw, dict) else vars(raw))


def _metric_packet(fold):
    """Read the official saved evaluation without rerunning or selecting on Test."""
    fold = Path(fold)
    candidates = sorted(fold.glob("test_metrics_final_*.json"))
    if not candidates:
        candidates = sorted(fold.glob("test_metrics_*.json"))
    if not candidates:
        return None
    metric_path = candidates[0]
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    suffix = metric_path.name[len("test_metrics_"):-len(".json")]
    confusion_path = fold / f"conf_mat_{suffix}.pkl"
    confusion = None
    if confusion_path.is_file():
        with confusion_path.open("rb") as stream:
            confusion = pickle.load(stream)
    per_class = list(metrics.get("per_class_f1", []))
    support = list(metrics.get("support", []))
    precision = list(metrics.get("per_class_precision", []))
    recall = list(metrics.get("per_class_recall", []))
    if confusion is not None and not per_class:
        import numpy as np

        matrix = np.asarray(confusion, dtype=np.float64)
        support_array = matrix.sum(1)
        predicted = matrix.sum(0)
        true_positive = np.diag(matrix)
        precision_array = np.divide(
            true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0,
        )
        recall_array = np.divide(
            true_positive, support_array, out=np.zeros_like(true_positive), where=support_array > 0,
        )
        per_class_array = np.divide(
            2 * precision_array * recall_array, precision_array + recall_array,
            out=np.zeros_like(true_positive), where=(precision_array + recall_array) > 0,
        )
        precision, recall = precision_array.tolist(), recall_array.tolist()
        per_class, support = per_class_array.tolist(), support_array.astype(int).tolist()
    return {
        "path": str(metric_path), "macro_f1": metrics.get("macro_f1"),
        "accuracy": metrics.get("accuracy"), "per_class_f1": per_class,
        "per_class_precision": precision, "per_class_recall": recall,
        "support": support,
        "fixed_class_macro_f1": (
            sum(per_class) / len(per_class) if per_class else None
        ),
    }


def _log_history(path):
    history, pseudo = [], []
    if not Path(path).is_file():
        return history, pseudo
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"Validation result:.*f1=([0-9.eE+-]+)", line)
        if match:
            history.append(float(match.group(1)))
        if line.startswith("SHAPE_V2CLEAN_EPOCH|"):
            values = {}
            for token in line.split("|")[1:]:
                if "=" in token:
                    key, value = token.split("=", 1)
                    try:
                        values[key] = float(value)
                    except ValueError:
                        pass
            pseudo.append(values)
    return history, pseudo


def _result_row(method, task, fold, log, source_sha=""):
    import torch

    fold = Path(fold)
    checkpoint = fold / "checkpoint_last.pt"
    if not checkpoint.is_file():
        return {"record_type": "summary", "method": method, "task": task,
                "status": "MISSING_FINAL"}, [], []
    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _config_dict(packet)
    metrics = _metric_packet(fold)
    history, epoch_rows = _log_history(log)
    row = {
        "record_type": "summary", "method": method, "task": task,
        "status": "complete" if metrics else "MISSING_TEST_METRICS",
        "checkpoint": str(checkpoint), "checkpoint_epoch": packet.get("epoch", -1),
        "source_checkpoint_sha256": source_sha,
        "target_structure_detach_epochs": config.get("target_structure_detach_epochs"),
        "class_org_target_weight": config.get("class_org_target_weight", 0.0),
        "best_val_macro_f1": max(history) if history else None,
        "final_val_macro_f1": history[-1] if history else None,
        "test_macro_f1": metrics.get("macro_f1") if metrics else None,
        "fixed_class_macro_f1": metrics.get("fixed_class_macro_f1") if metrics else None,
        "test_accuracy": metrics.get("accuracy") if metrics else None,
        "pseudo_coverage_final": (
            epoch_rows[-1].get("pseudo_coverage") if epoch_rows else None
        ),
        "accepted_pseudo_accuracy_final": (
            epoch_rows[-1].get("accepted_pseudo_accuracy") if epoch_rows else None
        ),
        "loss_class_organization_target_final": (
            epoch_rows[-1].get("loss_class_organization_target") if epoch_rows else None
        ),
        "test_labels_used_for_model_selection": False,
    }
    class_rows = []
    if metrics:
        classes = list(config.get("classes", []))
        for class_id, class_name in enumerate(classes):
            if class_id >= len(metrics["per_class_f1"]):
                break
            class_rows.append({
                "method": method, "task": task, "class_id": class_id,
                "class_name": class_name,
                "precision": metrics["per_class_precision"][class_id],
                "recall": metrics["per_class_recall"][class_id],
                "f1": metrics["per_class_f1"][class_id],
                "support": metrics["support"][class_id],
            })
    epoch_history = []
    for index, values in enumerate(epoch_rows):
        epoch_history.append({
            "record_type": "epoch", "method": method, "task": task,
            "epoch": int(values.get("epoch", index)),
            "validation_macro_f1": history[index] if index < len(history) else None,
            **values,
        })
    return row, class_rows, epoch_history


def _source_summary(args, pipeline):
    import torch
    from methods.structure_da.class_organization_loss import sha256_file

    fold = source_directory(args, pipeline) / "fold_0"
    checkpoint = fold / "checkpoint_last.pt"
    metrics = _metric_packet(fold)
    history, _ = _log_history(args.root / f"logs/SRC_ORG_{pipeline.source_alias}_seed1.log")
    reference_path_value = reference_path(args, pipeline)
    reference = torch.load(reference_path_value, map_location="cpu", weights_only=False)
    training_reference_path = fold / "class_organization_reference_training.pt"
    training_reference = torch.load(
        training_reference_path, map_location="cpu", weights_only=False,
    )
    metadata = dict(reference.get("metadata", {}))
    rows = [{
        "record_type": "summary",
        "source": pipeline.source_alias, "checkpoint": str(checkpoint),
        "best_checkpoint": str(fold / "model.pt"),
        "best_checkpoint_sha256": sha256_file(fold / "model.pt"),
        "source_org_effective": metadata.get("source_org_effective"),
        "source_org_warmup_epochs": metadata.get("warmup_epochs", 10),
        "reference": str(reference_path_value),
        "reference_valid_statistics": int(
            (reference["state_dict"]["weight"] > 0).sum().item()
        ),
        "best_val_macro_f1": max(history) if history else None,
        "final_val_macro_f1": history[-1] if history else None,
        "test_macro_f1": metrics.get("macro_f1") if metrics else None,
    }]
    source_log = args.root / f"logs/SRC_ORG_{pipeline.source_alias}_seed1.log"
    text = (
        source_log.read_text(encoding="utf-8", errors="replace")
        if source_log.is_file() else ""
    )
    validation_epoch = -1
    for line in text.splitlines():
        if line.startswith("CLASS_ORG_SOURCE_EPOCH|") or line.startswith("SHAPE_AUX_EPOCH|"):
            values = {"record_type": "epoch", "source": pipeline.source_alias}
            for token in line.split("|")[1:]:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                try: values[key] = float(value)
                except ValueError: values[key] = value
            values["metric_family"] = line.split("|", 1)[0]
            rows.append(values)
        match = re.search(r"Validation result:.*f1=([0-9.eE+-]+)", line)
        if match:
            validation_epoch += 1
            rows.append({
                "record_type": "epoch", "source": pipeline.source_alias,
                "metric_family": "validation", "epoch": validation_epoch,
                "validation_macro_f1": float(match.group(1)),
            })
    final_state = reference["state_dict"]
    initial_state = training_reference["state_dict"]
    components = ("strength", "concentration", "phase_k1_magnitude", "phase_k2_magnitude")
    for class_id in range(final_state["mu"].shape[0]):
        for anchor in range(final_state["mu"].shape[1]):
            for component, name in enumerate(components):
                rows.append({
                    "record_type": "reference", "source": pipeline.source_alias,
                    "class_id": class_id, "anchor": anchor, "component": name,
                    "support": int(final_state["support"][class_id]),
                    "training_mu": float(initial_state["mu"][class_id, anchor, component]),
                    "final_mu": float(final_state["mu"][class_id, anchor, component]),
                    "training_tau": float(initial_state["tau"][class_id, anchor, component]),
                    "final_tau": float(final_state["tau"][class_id, anchor, component]),
                    "training_weight": float(initial_state["weight"][class_id, anchor, component]),
                    "final_weight": float(final_state["weight"][class_id, anchor, component]),
                })
    return rows


def _organization_diagnostic_rows(root):
    rows = []
    scopes = Path(root) / "diagnostics/scopes"
    for scope in sorted(scopes.glob("[AB]_*")):
        increment = scope / "D3_relation_increment.csv"
        if increment.is_file():
            values = list(csv.DictReader(increment.open(newline="", encoding="utf-8")))
            grouped = {}
            for value in values:
                grouped.setdefault(value.get("split"), {})[value.get("probe")] = value
            for split, probes in grouped.items():
                left, right = probes.get("independent"), probes.get("independent_plus_relations")
                if left and right:
                    rows.append({
                        "scope": scope.name, "diagnostic": "cross_anchor_increment",
                        "split": split,
                        "value": float(right["macro_f1"]) - float(left["macro_f1"]),
                    })
        failure = scope / "D4_class_failure_modes.csv"
        if failure.is_file():
            counts = {}
            for value in csv.DictReader(failure.open(newline="", encoding="utf-8")):
                mode = value.get("failure_mode", "unknown")
                counts[mode] = counts.get(mode, 0) + 1
            rows.extend({
                "scope": scope.name, "diagnostic": "class_failure_mode",
                "split": mode, "value": count,
            } for mode, count in sorted(counts.items()))
    return rows


def generate_summary(args):
    import torch
    from methods.structure_da.class_organization_loss import sha256_file

    source_rows = {}
    comparison, per_class, history_rows = [], [], []
    for pipeline in pipelines():
        source_rows[pipeline.source_alias] = _source_summary(args, pipeline)
        source_sha = sha256_file(source_directory(args, pipeline) / "fold_0/model.pt")
        for variant in ("A", "B"):
            fold = args.root / f"outputs/uda/{variant}_{pipeline.task}_seed1/fold_0"
            row, classes, epochs = _result_row(
                variant, pipeline.task, fold,
                args.root / f"logs/{variant}_{pipeline.task}_seed1.log", source_sha,
            )
            comparison.append(row); per_class.extend(classes); history_rows.extend(epochs)
        baseline = args.g_baseline_root / f"G_{pipeline.task}_seed1/fold_0"
        if baseline.exists():
            row, classes, epochs = _result_row(
                "G", pipeline.task, baseline,
                args.g_baseline_log_root / f"G_{pipeline.task}_seed1.log", "",
            )
            comparison.append(row); per_class.extend(classes); history_rows.extend(epochs)
        else:
            comparison.append({
                "method": "G", "task": pipeline.task,
                "status": "MISSING_BASELINE", "checkpoint": str(baseline),
            })
    for alias, row in source_rows.items():
        _write_csv(args.root / f"Source_Org_{alias}_metrics.csv", row)
    _write_csv(args.root / "A_B_2tasks_seed1.csv", comparison + history_rows)
    _write_csv(args.root / "per_class_comparison.csv", per_class)
    diagnostics = _organization_diagnostic_rows(args.root)
    _write_csv(args.root / "organization_diagnostics.csv", diagnostics)

    index = {(row["method"], row["task"]): row for row in comparison}
    lines = [
        "# G class-conditional organization A/B", "",
        "All Source references were fit from Source Train only. Target labels were used only by frozen D1-D4 evaluation; Target Test was used only by the official final evaluation.", "",
        "| Task | G final | A final | B final | G→A | A→B |", "|---|---:|---:|---:|---:|---:|",
    ]
    for pipeline in pipelines():
        values = [index.get((method, pipeline.task), {}).get("test_macro_f1") for method in ("G", "A", "B")]
        def cell(value): return "MISSING" if value is None else f"{float(value):.4f}"
        ga = None if values[0] is None or values[1] is None else float(values[1]) - float(values[0])
        ab = None if values[1] is None or values[2] is None else float(values[2]) - float(values[1])
        lines.append(
            f"| {pipeline.task} | {cell(values[0])} | {cell(values[1])} | {cell(values[2])} | {cell(ga)} | {cell(ab)} |"
        )
    lines.extend([
        "", "D1-D4 frozen diagnostics are aggregated in organization_diagnostics.csv and retained per scope under diagnostics/scopes.",
        "Observed differences are associations under a controlled A/B protocol, not proof of a causal mechanism.",
    ])
    (args.root / "FINAL_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"SUMMARY_COMPLETE|root={args.root}", flush=True)


def run_source_job(args, pipeline, gpu, prechecked_status):
    source_fold = source_directory(args, pipeline) / "fold_0"
    _require_or_run(
        args, source_command(args, pipeline), source_fold, "source",
        f"SRC_ORG_{pipeline.source_alias}_seed1", gpu,
        prechecked_status=prechecked_status, verify_after=False,
    )


def run_uda_job(args, pipeline, variant, gpu, prechecked_status):
    fold = args.root / f"outputs/uda/{variant}_{pipeline.task}_seed1/fold_0"
    _require_or_run(
        args, uda_command(args, pipeline, variant), fold, "uda",
        f"{variant}_{pipeline.task}_seed1", gpu, variant=variant,
        prechecked_status=prechecked_status, verify_after=False,
    )
    _run_diagnostics(args, pipeline, variant, gpu)


def _parallel_round(name, jobs, target):
    failures = {}
    lock = threading.Lock()

    def worker(key, arguments):
        started = time.time()
        try:
            print(f"JOB_START|round={name}|key={key}", flush=True)
            target(*arguments)
            print(
                f"JOB_DONE|round={name}|key={key}|seconds={time.time() - started:.1f}",
                flush=True,
            )
        except Exception as error:
            with lock:
                failures[key] = str(error)
            print(f"JOB_FAILED|round={name}|key={key}|error={error}", flush=True)

    threads = [
        threading.Thread(target=worker, args=(key, arguments))
        for key, arguments in jobs
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def run_scheduler(args):
    args.root = Path(args.root)
    for relative in ("outputs/source", "outputs/uda", "logs", "runs"):
        (args.root / relative).mkdir(parents=True, exist_ok=True)
    source_plan = source_assignments(args.gpus)
    source_status = {}
    for gpu, pipeline in source_plan:
        key = f"SRC_ORG_{pipeline.source_alias}_seed1"
        source_status[key] = (
            "missing" if args.mode == "dry-run" else artifact_status(
                source_directory(args, pipeline) / "fold_0", "source",
            )
        )
    source_jobs = [
        (
            f"SRC_ORG_{pipeline.source_alias}_seed1",
            (
                args, pipeline, gpu,
                source_status[f"SRC_ORG_{pipeline.source_alias}_seed1"],
            ),
        )
        for gpu, pipeline in source_plan
    ]
    source_failures = _parallel_round("source", source_jobs, run_source_job)
    if args.mode != "dry-run":
        for gpu, pipeline in source_plan:
            key = f"SRC_ORG_{pipeline.source_alias}_seed1"
            if key in source_failures:
                continue
            status = artifact_status(
                source_directory(args, pipeline) / "fold_0", "source",
            )
            if status != "complete":
                source_failures[key] = (
                    f"Source post-check failed in scheduler main thread: status={status}"
                )

    uda_jobs = []
    uda_plan = []
    for gpu, variant, pipeline in uda_assignments(args.gpus):
        source_key = f"SRC_ORG_{pipeline.source_alias}_seed1"
        if source_key in source_failures:
            print(
                f"JOB_BLOCKED|round=uda|key={variant}_{pipeline.task}_seed1|"
                f"dependency={source_key}", flush=True,
            )
            continue
        fold = args.root / f"outputs/uda/{variant}_{pipeline.task}_seed1/fold_0"
        status = (
            "missing" if args.mode == "dry-run"
            else artifact_status(fold, "uda", variant)
        )
        uda_plan.append((gpu, variant, pipeline))
        uda_jobs.append((
            f"{variant}_{pipeline.task}_seed1",
            (args, pipeline, variant, gpu, status),
        ))
    uda_failures = _parallel_round("uda", uda_jobs, run_uda_job)
    if args.mode != "dry-run":
        for gpu, variant, pipeline in uda_plan:
            key = f"{variant}_{pipeline.task}_seed1"
            if key in uda_failures:
                continue
            fold = args.root / f"outputs/uda/{variant}_{pipeline.task}_seed1/fold_0"
            status = artifact_status(fold, "uda", variant)
            if status != "complete":
                uda_failures[key] = (
                    f"UDA post-check failed in scheduler main thread: status={status}"
                )
    failures = {**source_failures, **uda_failures}
    if failures:
        raise RuntimeError("class-organization failures: " + json.dumps(failures))
    diagnostic_summary = [
        args.python, "-u", "-m", "analysis.structure_g_response_organization_audit",
        "--mode", "summary", "--output-root", str(args.root / "diagnostics"),
    ]
    if args.mode == "dry-run":
        _run(diagnostic_summary, args.root / "logs/diagnostic_summary.log", "cpu", True)
        print(
            "SUMMARY_PLAN|outputs=" + ",".join(str(path) for path in summary_paths(args.root)),
            flush=True,
        )
    else:
        _run(diagnostic_summary, args.root / "logs/diagnostic_summary.log", "cpu")
        generate_summary(args)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--gpus", nargs=4, default=["0", "1", "2", "3"])
    parser.add_argument("--root", type=Path, default=Path("experiments/structure_g_class_org_seed1"))
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--diagnostic-batch-size", type=int, default=64)
    parser.add_argument("--diagnostic-pixel-budget", type=int, default=8192)
    parser.add_argument(
        "--g-baseline-root", type=Path,
        default=Path("experiments/structure_e_cross_seed1/outputs/uda"),
    )
    parser.add_argument(
        "--g-baseline-log-root", type=Path,
        default=Path("experiments/structure_e_cross_seed1/logs"),
    )
    return parser


if __name__ == "__main__":
    run_scheduler(build_parser().parse_args())
