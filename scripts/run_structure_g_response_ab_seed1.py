#!/usr/bin/env python3
"""Four-GPU Source→UDA scheduler for phase-response support/independent A/B."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "AT1_DK1": ("AT1", "austria/33UVP/2017", "DK1", "denmark/32VNH/2017"),
    "FR1_FR2": ("FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
    "FR2_DK1": ("FR2", "france/31TCJ/2017", "DK1", "denmark/32VNH/2017"),
    "DK1_AT1": ("DK1", "denmark/32VNH/2017", "AT1", "austria/33UVP/2017"),
}
MODES = {"A": "support", "B": "independent"}


@dataclass(frozen=True)
class Job:
    variant: str
    mode: str
    task: str
    source_alias: str
    source: str
    target_alias: str
    target: str


def planned_jobs():
    return [
        Job(variant, mode, task, *domains)
        for variant, mode in MODES.items()
        for task, domains in TASKS.items()
    ]


def _common_model_args(mode):
    return [
        "--model", "psestructureprotoltae", "--structure-branch", "true",
        "--structure-exposer", "fourier", "--fourier_num_modes", "13",
        "--shape-dim", "128", "--shape-window-scales", "24",
        "--shape-window-stride", "8", "--shapelet-count", "16",
        "--shapelet-beta", "5", "--shape-resample-length", "16",
        "--shape-representation", "phase_moment", "--shape-injection", "current_query",
        "--phase-query-view", "full", "--phase-response-mode", mode,
        "--shape-phase-k2-enabled", "true", "--structure-shift-mode", "none",
        "--shapelet-diversity-margin", "0.5", "--shapelet-diversity-weight", "0.01",
        "--shape-class-weight", "0.1", "--source-minority-mode", "base",
    ]


def _source_experiment(job):
    return f"SRC_{job.variant}_{job.source_alias}_seed1"


def _uda_experiment(job):
    return f"{job.variant}_{job.task}_seed1"


def source_command(args, job):
    experiment = _source_experiment(job)
    return [
        args.python, "-u", "train.py", "-e", experiment,
        "--data_root", str(args.data_root), "--source", job.source, "--target", job.source,
        *_common_model_args(job.mode), "--with_shift_aug", "false", "--seed", "1",
        "--num_folds", "1", "--epochs", "100", "--batch_size", "128",
        "--lr", "0.001", "--weight_decay", "0.0001", "--focal_loss_gamma", "1.0",
        "--seq_length", "30", "--num_pixels", "64", "--closed_set", "true",
        "--progress_bar", "off", "--output_dir", str(args.output_root / "outputs/source"),
        "--tensorboard_log_dir", str(args.output_root / f"runs/{experiment}"),
    ]


def uda_command(args, job):
    experiment = _uda_experiment(job)
    weights = args.output_root / "outputs/source" / _source_experiment(job)
    return [
        args.python, "-u", "train.py", "-e", experiment,
        "--data_root", str(args.data_root), "--source", job.source, "--target", job.target,
        *_common_model_args(job.mode), "--shape-align-weight", "0",
        "--with_shift_aug", "false", "--seed", "1", "--num_folds", "1",
        "--batch_size", "128", "--seq_length", "30", "--num_pixels", "64",
        "--closed_set", "true", "--progress_bar", "off",
        "--output_dir", str(args.output_root / "outputs/uda"),
        "--tensorboard_log_dir", str(args.output_root / f"runs/{experiment}"),
        "timematch", "--weights", str(weights), "--epochs", "20",
        "--steps_per_epoch", "500", "--lr", "0.0001", "--pseudo_threshold", "0.9",
        "--ema_decay", "0.9999", "--trade_off", "2.0", "--estimate_shift", "true",
        "--balance_source", "true", "--use_focal_loss", "true", "--shift_source", "true",
        "--sample_size", "100", "--max_temporal_shift", "60",
        "--domain_specific_bn", "true", "--shift_estimator", "AM", "--run_validation",
        "--output_student", "true", "--shape-da-mode", "batch_align",
        "--shape-alignment-view", "none", "--oracle-pseudo-labels", "false",
        "--adaptive-pseudo-selection", "false", "--shape-equivariance-weight", "0.05",
        "--shape-equivariance-max-shift", "60", "--detach-target-structure", "false",
        "--target-structure-detach-epochs", "5",
    ]


def _config_value(packet, key, default=None):
    config = packet.get("config", {})
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def artifact_status(fold, kind, mode, source_alias, target_alias=None):
    fold = Path(fold)
    if not fold.exists():
        return "missing"
    last = fold / "checkpoint_last.pt"
    if not last.is_file() or (kind == "source" and not (fold / "model.pt").is_file()):
        return "partial"
    try:
        import torch
        packet = torch.load(last, map_location="cpu", weights_only=False)
        expected_epoch = 99 if kind == "source" else 19
        checks = (
            int(packet.get("epoch", -1)) == expected_epoch,
            int(_config_value(packet, "seed", -1)) == 1,
            _config_value(packet, "shape_representation") == "phase_moment",
            _config_value(packet, "phase_response_mode", "softmax") == mode,
            str(_config_value(packet, "source", "")) == TASKS[
                next(task for task, values in TASKS.items() if values[0] == source_alias)
            ][1],
        )
        if kind == "uda":
            checks += (
                int(_config_value(packet, "target_structure_detach_epochs", -1)) == 5,
                int(_config_value(packet, "steps_per_epoch", -1)) == 500,
                str(_config_value(packet, "target", "")) == TASKS[
                    next(task for task, values in TASKS.items()
                         if values[0] == source_alias and values[2] == target_alias)
                ][3],
            )
        return "complete" if all(checks) else "incompatible"
    except Exception:
        return "partial"


def _run(command, log, gpu, dry_run):
    rendered = " ".join(map(str, command))
    print(f"COMMAND|gpu={gpu}|{rendered}", flush=True)
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"COMMAND|gpu={gpu}|{rendered}\n"); stream.flush()
        result = subprocess.run(
            command, cwd=ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"exit={result.returncode}; log={log}")


def _require_safe_status(status, fold, mode):
    if status in ("partial", "incompatible"):
        raise RuntimeError(
            f"{status} {mode} artifact; refusing overwrite or implicit resume: {fold}"
        )


def run_job(args, job, gpu):
    source_fold = args.output_root / "outputs/source" / _source_experiment(job) / "fold_0"
    uda_fold = args.output_root / "outputs/uda" / _uda_experiment(job) / "fold_0"
    source_status = artifact_status(source_fold, "source", job.mode, job.source_alias)
    _require_safe_status(source_status, source_fold, job.mode)
    if source_status == "complete":
        print(f"REUSE|kind=source|mode={job.mode}|source={job.source_alias}", flush=True)
    else:
        _run(
            source_command(args, job),
            args.output_root / f"logs/{_source_experiment(job)}.log",
            gpu, args.mode == "dry-run",
        )
        if args.mode != "dry-run" and artifact_status(
            source_fold, "source", job.mode, job.source_alias,
        ) != "complete":
            raise RuntimeError(f"Source did not produce a complete artifact: {source_fold}")

    uda_status = artifact_status(
        uda_fold, "uda", job.mode, job.source_alias, job.target_alias,
    )
    _require_safe_status(uda_status, uda_fold, job.mode)
    if uda_status == "complete":
        print(f"REUSE|kind=uda|mode={job.mode}|task={job.task}", flush=True)
        return
    _run(
        uda_command(args, job), args.output_root / f"logs/{_uda_experiment(job)}.log",
        gpu, args.mode == "dry-run",
    )
    if args.mode != "dry-run" and artifact_status(
        uda_fold, "uda", job.mode, job.source_alias, job.target_alias,
    ) != "complete":
        raise RuntimeError(f"UDA did not produce a complete artifact: {uda_fold}")


def _write_csv(path, rows):
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def _read_final_metrics(fold):
    metrics = next(iter(sorted(Path(fold).glob("test_metrics_final_*.json"))), None)
    confusion = next(iter(sorted(Path(fold).glob("conf_mat_final_*.pkl"))), None)
    if metrics is None or confusion is None:
        return None
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    with confusion.open("rb") as stream:
        matrix = np.asarray(pickle.load(stream), dtype=np.int64)
    support = matrix.sum(1); predicted = matrix.sum(0); tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    classes = []
    checkpoint = Path(fold) / "checkpoint_last.pt"
    if checkpoint.is_file():
        import torch
        packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
        classes = list(_config_value(packet, "classes", ()))
    if len(classes) != len(f1):
        classes = [f"class_{index}" for index in range(len(f1))]
    return payload, support, precision, recall, f1, classes


def _checkpoint_summary(fold):
    last = Path(fold) / "checkpoint_last.pt"
    best = Path(fold) / "checkpoint_best.pt"
    if not last.is_file():
        return {"final_epoch": "", "best_val_macro_f1": "", "best_epoch": ""}
    import torch
    final_packet = torch.load(last, map_location="cpu", weights_only=False)
    best_packet = torch.load(best, map_location="cpu", weights_only=False) if best.is_file() else {}
    return {
        "final_epoch": final_packet.get("epoch", ""),
        "best_val_macro_f1": best_packet.get("best_f1", ""),
        "best_epoch": best_packet.get("epoch", ""),
    }


def _log_diagnostics(path, variant, task):
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("STRUCTURE_USAGE_DIAG|"):
            continue
        values = dict(item.split("=", 1) for item in line.split("|")[1:] if "=" in item)
        rows.append({"variant": variant, "task": task, **values})
    return rows


def _runtime_seconds(output_root, variant, task):
    path = Path(output_root) / "runtime" / f"{variant}_{task}_seed1.json"
    if not path.is_file():
        return ""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("seconds", "")
    except (OSError, ValueError, TypeError):
        return ""


def summarize_results(args):
    task_rows, class_rows, diagnostic_rows, source_rows = [], [], [], []
    for job in planned_jobs():
        source_fold = args.output_root / "outputs/source" / _source_experiment(job) / "fold_0"
        source_packet = None
        if (source_fold / "model.pt").is_file():
            import torch
            source_packet = torch.load(source_fold / "model.pt", map_location="cpu", weights_only=False)
        source_rows.append({
            "variant": job.variant, "mode": job.mode, "source": job.source_alias,
            "source_best_epoch": source_packet.get("epoch", "") if source_packet else "",
            "source_best_val_macro_f1": source_packet.get("best_f1", "") if source_packet else "",
        })
        fold = args.output_root / "outputs/uda" / _uda_experiment(job) / "fold_0"
        result = _read_final_metrics(fold)
        row = {
            "task": job.task, "variant": job.variant, "mode": job.mode,
            "runtime_seconds": _runtime_seconds(
                args.output_root, job.variant, job.task,
            ),
            **_checkpoint_summary(fold),
        }
        if result is None:
            row.update(status="missing", final_test_macro_f1="")
        else:
            payload, support, precision, recall, f1, classes = result
            row.update(status="complete", final_test_macro_f1=payload.get("macro_f1", ""))
            for index in range(len(f1)):
                class_rows.append({
                    "task": job.task, "variant": job.variant, "class_index": index,
                    "class_name": classes[index],
                    "support": int(support[index]), "precision": precision[index],
                    "recall": recall[index], "f1": f1[index],
                })
        task_rows.append(row)
        diagnostic_rows.extend(_log_diagnostics(
            args.output_root / f"logs/{_uda_experiment(job)}.log", job.variant, job.task,
        ))
        diagnostic_rows.extend(_log_diagnostics(
            args.output_root / f"logs/{_source_experiment(job)}.log",
            job.variant, f"SOURCE_{job.source_alias}",
        ))
    for task, domains in TASKS.items():
        candidates = (
            args.g_root / f"G_{task}_seed1/fold_0",
            args.g_root / f"{task}_seed1/fold_0",
        )
        fold = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        result = _read_final_metrics(fold)
        row = {
            "task": task, "variant": "G", "mode": "softmax",
            "runtime_seconds": "", **_checkpoint_summary(fold),
        }
        if result is None:
            row.update(status="missing", final_test_macro_f1="")
        else:
            payload, support, precision, recall, f1, classes = result
            row.update(status="complete", final_test_macro_f1=payload.get("macro_f1", ""))
            for index in range(len(f1)):
                class_rows.append({
                    "task": task, "variant": "G", "class_index": index,
                    "class_name": classes[index],
                    "support": int(support[index]), "precision": precision[index],
                    "recall": recall[index], "f1": f1[index],
                })
        task_rows.append(row)
    lookup = {(row["task"], row["variant"]): row for row in task_rows}
    for task in TASKS:
        scores = {
            variant: lookup[(task, variant)].get("final_test_macro_f1", "")
            for variant in ("G", "A", "B")
        }
        if all(value not in (None, "") for value in scores.values()):
            g, a, b = (float(scores[variant]) for variant in ("G", "A", "B"))
            lookup[(task, "A")]["delta_vs_g"] = a - g
            lookup[(task, "B")]["delta_vs_g"] = b - g
            lookup[(task, "B")]["delta_vs_a"] = b - a
    unique_source = {(row["variant"], row["source"]): row for row in source_rows}.values()
    _write_csv(args.output_root / "Source_AB_metrics.csv", unique_source)
    _write_csv(args.output_root / "UDA_AB_4tasks_seed1.csv", task_rows)
    _write_csv(args.output_root / "per_class_comparison.csv", class_rows)
    _write_csv(args.output_root / "response_diagnostics.csv", diagnostic_rows)
    lines = [
        "# G Response A/B Seed1", "",
        "Primary metric: final-epoch Student Test Macro-F1. Test is not used for selection.", "",
        "| Task | G | A | B | A-G | B-G | B-A |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        values = {
            variant: lookup[(task, variant)]["final_test_macro_f1"]
            for variant in ("G", "A", "B")
        }
        if all(value not in (None, "") for value in values.values()):
            g, a, b = (float(values[key]) for key in ("G", "A", "B"))
            cells = (f"{g:.4f}", f"{a:.4f}", f"{b:.4f}",
                     f"{a-g:+.4f}", f"{b-g:+.4f}", f"{b-a:+.4f}")
        else:
            cells = tuple(
                "MISSING" if values.get(key) in (None, "") else f"{float(values[key]):.4f}"
                for key in ("G", "A", "B")
            ) + ("MISSING", "MISSING", "MISSING")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    means = {}
    for variant in ("G", "A", "B"):
        values = [lookup[(task, variant)]["final_test_macro_f1"] for task in TASKS]
        means[variant] = (
            float(np.mean([float(value) for value in values]))
            if all(value not in (None, "") for value in values) else None
        )
    if all(value is not None for value in means.values()):
        g, a, b = means["G"], means["A"], means["B"]
        lines.append(
            f"| 4-task mean | {g:.4f} | {a:.4f} | {b:.4f} | "
            f"{a-g:+.4f} | {b-g:+.4f} | {b-a:+.4f} |"
        )
    else:
        lines.append("| 4-task mean | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
    lines.extend(["", f"G baseline root (read-only): `{args.g_root}`.",
                  "No G baseline training was launched by this scheduler."])
    (args.output_root / "FINAL_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scheduler(args):
    args.output_root = Path(args.output_root)
    jobs = planned_jobs()
    if args.mode == "dry-run":
        for index, job in enumerate(jobs):
            gpu = args.gpus[index % len(args.gpus)]
            print(f"SOURCE_PLAN|gpu={gpu}|variant={job.variant}|mode={job.mode}|source={job.source_alias}")
            print("COMMAND|gpu=" + str(gpu) + "|" + " ".join(source_command(args, job)))
            print(f"UDA_PLAN|gpu={gpu}|variant={job.variant}|mode={job.mode}|task={job.task}")
            print("COMMAND|gpu=" + str(gpu) + "|" + " ".join(uda_command(args, job)))
        return
    if not Path(args.data_root).is_dir():
        raise FileNotFoundError(f"missing data root: {args.data_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures = {}; lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                started = time.time(); run_job(args, job, gpu)
                elapsed = time.time() - started
                runtime_path = (
                    args.output_root / "runtime" /
                    f"{job.variant}_{job.task}_seed1.json"
                )
                runtime_path.parent.mkdir(parents=True, exist_ok=True)
                runtime_path.write_text(json.dumps({
                    "variant": job.variant, "mode": job.mode,
                    "task": job.task, "gpu": str(gpu), "seconds": elapsed,
                }, indent=2, sort_keys=True), encoding="utf-8")
                print(f"JOB_DONE|gpu={gpu}|variant={job.variant}|task={job.task}|seconds={elapsed:.1f}")
            except Exception as error:
                with lock:
                    failures[f"{job.variant}_{job.task}"] = str(error)
                print(f"JOB_FAILED|gpu={gpu}|variant={job.variant}|task={job.task}|error={error}")
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    summarize_results(args)
    (args.output_root / "scheduler_status.json").write_text(
        json.dumps({"failures": failures}, indent=2, sort_keys=True), encoding="utf-8",
    )
    if failures:
        raise RuntimeError("G response A/B failures: " + json.dumps(failures))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--output-root", type=Path,
                        default=Path("experiments/structure_g_response_ab_seed1"))
    parser.add_argument("--g-root", type=Path,
                        default=Path("experiments/structure_e_cross_seed1/outputs/uda"))
    parser.add_argument("--gpus", nargs="+", default=("0", "1", "2", "3"))
    parser.add_argument("--python", default=sys.executable)
    return parser


if __name__ == "__main__":
    run_scheduler(build_parser().parse_args())
