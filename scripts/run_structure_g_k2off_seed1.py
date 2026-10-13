#!/usr/bin/env python3
"""Two-GPU resumable runner for the four-task G-K2off Seed1 ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "AT1_DK1": ("AT1", "austria/33UVP/2017", "DK1", "denmark/32VNH/2017"),
    "FR2_DK1": ("FR2", "france/31TCJ/2017", "DK1", "denmark/32VNH/2017"),
    "FR1_FR2": ("FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
    "DK1_AT1": ("DK1", "denmark/32VNH/2017", "AT1", "austria/33UVP/2017"),
}


@dataclass(frozen=True)
class Job:
    task: str
    source_alias: str
    source: str
    target_alias: str
    target: str


def jobs():
    return [Job(name, *values) for name, values in TASKS.items()]


def common_model_args():
    return [
        "--model", "psestructureprotoltae", "--structure-branch", "true",
        "--structure-exposer", "fourier", "--fourier_num_modes", "13",
        "--shape-dim", "128", "--shape-window-scales", "24",
        "--shape-window-stride", "8", "--shapelet-count", "16",
        "--shapelet-beta", "5", "--shape-resample-length", "16",
        "--shape-representation", "phase_moment", "--shape-injection", "current_query",
        "--structure-shift-mode", "none", "--shape-phase-k2-enabled", "false",
        "--shapelet-diversity-margin", "0.5", "--shapelet-diversity-weight", "0.01",
        "--shape-class-weight", "0.1", "--source-minority-mode", "base",
    ]


def source_command(args, job):
    experiment = f"source_{job.source_alias}_seed1"
    return [args.python, "-u", "train.py", "-e", experiment,
            "--data_root", args.data_root, "--source", job.source, "--target", job.source,
            *common_model_args(), "--with_shift_aug", "false", "--seed", "1",
            "--num_folds", "1", "--epochs", "100", "--batch_size", "128",
            "--lr", "0.001", "--weight_decay", "0.0001", "--focal_loss_gamma", "1.0",
            "--seq_length", "30", "--num_pixels", "64", "--closed_set", "true",
            "--progress_bar", "off", "--output_dir", str(args.root / "outputs/source"),
            "--tensorboard_log_dir", str(args.root / f"runs/source_{job.source_alias}_seed1")]


def uda_command(args, job):
    weights = args.root / "outputs/source" / f"source_{job.source_alias}_seed1"
    return [args.python, "-u", "train.py", "-e", f"G_K2off_{job.task}_seed1",
            "--data_root", args.data_root, "--source", job.source, "--target", job.target,
            *common_model_args(), "--shape-align-weight", "0", "--with_shift_aug", "false",
            "--seed", "1", "--num_folds", "1", "--batch_size", "128",
            "--seq_length", "30", "--num_pixels", "64", "--closed_set", "true",
            "--progress_bar", "off", "--output_dir", str(args.root / "outputs/uda"),
            "--tensorboard_log_dir", str(args.root / f"runs/G_K2off_{job.task}_seed1"),
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
            "--target-structure-detach-epochs", "5"]


def _config_value(packet, name, default=None):
    config = packet.get("config")
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def artifact_status(fold, kind):
    """Return complete/missing/partial/incompatible without modifying artifacts."""
    fold = Path(fold)
    if not fold.exists():
        return "missing"
    path = fold / "checkpoint_last.pt"
    if not path.is_file() or (kind == "source" and not (fold / "model.pt").is_file()):
        return "partial"
    try:
        import torch
        packet = torch.load(path, map_location="cpu", weights_only=False)
        if bool(_config_value(packet, "shape_phase_k2_enabled", True)):
            return "incompatible"
        epochs = int(_config_value(packet, "epochs", 100 if kind == "source" else 20))
        if int(packet.get("epoch", -1)) != epochs - 1:
            return "partial"
        if kind == "uda":
            if int(_config_value(packet, "target_structure_detach_epochs", -1)) != 5:
                return "incompatible"
    except Exception:
        return "partial"
    return "complete"


def _run(command, log, gpu, dry_run):
    rendered = " ".join(map(str, command))
    print(f"COMMAND|gpu={gpu}|{rendered}", flush=True)
    if dry_run:
        return 0.
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"COMMAND|gpu={gpu}|{rendered}\n"); stream.flush()
        started = time.time()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed with exit={result.returncode}; log={log}")
    return time.time() - started


def run_frozen_audits(args):
    """Use existing frozen caches on CPU while the two GPUs remain dedicated to training."""
    cache_root = Path(args.cache_root)
    report_rows = []
    if not cache_root.is_dir():
        message = f"MISSING_CACHE_ROOT|path={cache_root}"
        print(message, flush=True)
        report_rows.append(message)
    else:
        for cache in sorted(cache_root.iterdir()):
            if not (cache / "features").is_dir():
                continue
            identity = cache.name
            output = args.root / "frozen" / identity
            for kind, script in (
                ("X1", "analysis/structure_evidence_increment_audit.py"),
                ("X2", "analysis/structure_anchor_conflict_audit.py"),
            ):
                expected = output / f"{kind}_report.md"
                if args.mode == "resume" and expected.is_file() and expected.stat().st_size:
                    print(f"REUSE|kind={kind}|cache={cache}", flush=True); continue
                command = [args.python, "-u", script, "--cache-dir", str(cache),
                           "--output-dir", str(output), "--task", identity.split("_", 1)[-1],
                           "--variant", identity.split("_", 1)[0]]
                rendered = " ".join(command); print(f"AUDIT_COMMAND|{rendered}", flush=True)
                if args.mode == "dry-run":
                    continue
                output.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
                log = args.root / f"logs/{kind}_{identity}.log"
                with log.open("a", encoding="utf-8") as stream:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    report_rows.append(f"FAILED|kind={kind}|cache={cache}|log={log}")
                elif not expected.is_file() or expected.stat().st_size == 0:
                    report_rows.append(f"MISSING_ARTIFACT|kind={kind}|cache={cache}|path={expected}")
    if args.mode != "dry-run":
        for kind in ("X1", "X2"):
            reports = sorted((args.root / "frozen").glob(f"*/{kind}_report.md"))
            body = [f"# {kind} frozen audit index", ""]
            body.extend(f"- {path.parent.name}: `{path}`" for path in reports)
            body.extend(["", *report_rows])
            (args.root / f"{kind}_report.md").write_text("\n".join(body) + "\n", encoding="utf-8")
        if report_rows:
            raise RuntimeError("frozen audits incomplete: " + json.dumps(report_rows))


def quarantine_partial(args, experiment_dir, label):
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.exists():
        return
    failed = args.root / "incomplete"
    failed.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = failed / f"{experiment_dir.name}.{label}.{timestamp}"
    experiment_dir.rename(destination)
    print(f"QUARANTINE|from={experiment_dir}|to={destination}", flush=True)


def run_job(args, job, gpu):
    source_fold = args.root / "outputs/source" / f"source_{job.source_alias}_seed1/fold_0"
    uda_fold = args.root / "outputs/uda" / f"G_K2off_{job.task}_seed1/fold_0"
    source_status = artifact_status(source_fold, "source")
    if source_status == "complete" and args.mode == "run":
        raise RuntimeError(f"completed Source exists; use resume: {source_fold}")
    if source_status == "incompatible":
        raise RuntimeError(f"refusing to overwrite {source_status} Source: {source_fold}")
    if source_status == "partial":
        if args.mode != "resume":
            raise RuntimeError(f"refusing to overwrite partial Source: {source_fold}")
        quarantine_partial(args, source_fold.parent, "source")
        source_status = "missing"
    source_seconds = uda_seconds = 0.
    if source_status != "complete":
        source_seconds = _run(source_command(args, job), args.root / f"logs/SRC_K2off_{job.source_alias}_seed1.log", gpu, args.mode == "dry-run")
        if args.mode != "dry-run" and artifact_status(source_fold, "source") != "complete":
            raise RuntimeError(f"Source did not complete: {source_fold}")
    uda_status = artifact_status(uda_fold, "uda")
    if uda_status == "complete":
        if args.mode == "run":
            raise RuntimeError(f"completed UDA exists; use resume: {uda_fold}")
        print(f"REUSE|kind=uda|task={job.task}|path={uda_fold}", flush=True); return
    if uda_status == "incompatible":
        raise RuntimeError(f"refusing to overwrite {uda_status} UDA: {uda_fold}")
    if uda_status == "partial":
        if args.mode != "resume":
            raise RuntimeError(f"refusing to overwrite partial UDA: {uda_fold}")
        quarantine_partial(args, uda_fold.parent, "uda")
        uda_status = "missing"
    uda_seconds = _run(uda_command(args, job), args.root / f"logs/G_K2off_{job.task}_seed1.log", gpu, args.mode == "dry-run")
    if args.mode != "dry-run" and artifact_status(uda_fold, "uda") != "complete":
        raise RuntimeError(f"UDA did not complete: {uda_fold}")
    if args.mode != "dry-run":
        runtime = args.root / "runtime"; runtime.mkdir(parents=True, exist_ok=True)
        (runtime / f"{job.task}.json").write_text(json.dumps({
            "task": job.task, "gpu": str(gpu), "source_seconds": source_seconds,
            "uda_seconds": uda_seconds,
        }, indent=2), encoding="utf-8")


def _parse_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace") if Path(path).is_file() else ""
    validation = [float(value) for value in re.findall(r"Validation result:.*?f1=([0-9.]+)", text)]
    macro = [float(value) for value in re.findall(r"macro avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)", text)]
    return {"best_val_macro_f1": max(validation) if validation else "MISSING",
            "official_test_macro_f1": macro[-1] if macro else "MISSING"}


def _metrics(fold):
    paths = sorted(Path(fold).glob("test_metrics_final_*.json"))
    if not paths:
        return None
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def write_summary(args):
    rows, per_class = [], []
    for job in jobs():
        source = _parse_log(args.root / f"logs/SRC_K2off_{job.source_alias}_seed1.log")
        uda = _parse_log(args.root / f"logs/G_K2off_{job.task}_seed1.log")
        fold = args.root / "outputs/uda" / f"G_K2off_{job.task}_seed1/fold_0"
        metrics = _metrics(fold)
        if metrics is not None:
            uda["official_test_macro_f1"] = metrics.get("macro_f1", "MISSING")
            for index, support in enumerate(metrics.get("support", [])):
                per_class.append({
                    "task": job.task, "class_id": index,
                    "precision": metrics["per_class_precision"][index],
                    "recall": metrics["per_class_recall"][index],
                    "f1": metrics["per_class_f1"][index], "support": support,
                })
        baseline_fold = Path(args.baseline_g_root) / f"G_{job.task}_seed1/fold_0"
        baseline = _metrics(baseline_fold)
        rows.append({"task": job.task, "seed": 1, "source_val_macro_f1": source["best_val_macro_f1"],
                     **uda, "baseline_g_test_macro_f1": (
                         baseline.get("macro_f1", "MISSING") if baseline else "MISSING"
                     ), "k2_enabled": False})
    path = args.root / "G_K2off_4tasks_seed1.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    per_class_path = args.root / "G_K2off_per_class_seed1.csv"
    with per_class_path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(per_class[0]) if per_class else ["status"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(per_class)
    (args.root / "FINAL_summary.md").write_text(
        "# X1 / X2 / G-K2off summary\n\n"
        "Frozen-audit indexes: `X1_report.md`, `X2_report.md`. Training table: "
        "`G_K2off_4tasks_seed1.csv`. The table is generated only from official training logs. "
        "Missing values remain explicit; Target Test is not used for model selection. "
        "Observed associations do not establish a causal mechanism.\n",
        encoding="utf-8",
    )


def run_scheduler(args):
    args.root = Path(args.root)
    for name in ("outputs/source", "outputs/uda", "logs", "runs"):
        (args.root / name).mkdir(parents=True, exist_ok=True)
    audit_thread, audit_failures = None, []
    if not args.skip_frozen_audits:
        def audit_target():
            try: run_frozen_audits(args)
            except Exception as error: audit_failures.append(str(error))
        audit_thread = threading.Thread(target=audit_target, daemon=False)
        audit_thread.start()
    pending = queue.Queue()
    for job in jobs(): pending.put(job)
    failures, lock = {}, threading.Lock()

    def worker(gpu):
        while True:
            try: job = pending.get_nowait()
            except queue.Empty: return
            started = time.time()
            try:
                print(f"JOB_START|gpu={gpu}|task={job.task}", flush=True)
                run_job(args, job, gpu)
                print(f"JOB_DONE|gpu={gpu}|task={job.task}|seconds={time.time()-started:.1f}", flush=True)
            except Exception as error:
                with lock: failures[job.task] = str(error)
                print(f"JOB_FAILED|gpu={gpu}|task={job.task}|error={error}", flush=True)
            finally: pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in args.gpus]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if audit_thread is not None: audit_thread.join()
    if audit_failures:
        failures["frozen_audits"] = "; ".join(audit_failures)
    if failures:
        raise RuntimeError("G-K2off failures: " + json.dumps(failures, ensure_ascii=False))
    if args.mode != "dry-run": write_summary(args)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--gpus", nargs=2, default=("0", "1"))
    parser.add_argument("--root", default="experiments/structure_g_k2off_seed1")
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--cache-root", default="experiments/structure_pair_semantics_seed1/cache")
    parser.add_argument("--baseline-g-root", default="experiments/structure_e_cross_seed1/outputs/uda")
    parser.add_argument("--skip-frozen-audits", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


if __name__ == "__main__":
    run_scheduler(build_parser().parse_args())
