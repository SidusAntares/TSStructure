#!/usr/bin/env python3
"""Event-driven four-GPU scheduler for paired E/G recheck and diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.diagnose_eg_pair import (
    TASKS, require_complete_checkpoint, sha256_file, validate_source,
    validate_uda_checkpoint,
)


EXPERIMENT_ROOT = Path("experiments/eg_recheck_pair_seed1")
SOURCE_ROOT = Path("outputs/recheck_e_final_seed1/source")
INITIAL_GPU_ASSIGNMENT = {
    "uda:E:FR1_FR2": "0", "uda:E:FR2_DK1": "1",
    "uda:G:FR1_FR2": "2", "uda:G:FR2_DK1": "3",
}


@dataclass(frozen=True)
class Job:
    key: str
    kind: str
    task: str
    variant: str | None = None
    dependencies: frozenset[str] = field(default_factory=frozenset)
    weight: str = "medium"


def paired_uda_config(variant):
    if variant not in ("E", "G"):
        raise ValueError("variant must be E or G")
    return {
        "shape_representation": "phase_moment", "shape_injection": "current_query",
        "structure_shift_mode": "none", "fourier_num_modes": 13,
        "shape_dim": 128, "shape_window_scales": [24], "shape_window_stride": 8,
        "shapelet_count": 16, "shapelet_beta": 5, "shape_resample_length": 16,
        "shapelet_diversity_margin": .5, "shapelet_diversity_weight": .01,
        "shape_class_weight": .1, "shape_align_weight": 0.,
        "source_minority_mode": "base", "with_shift_aug": False,
        "seed": 1, "num_folds": 1, "seq_length": 30, "num_pixels": 64,
        "closed_set": True, "epochs": 20,
        "steps_per_epoch": 500, "batch_size": 128, "lr": .0001,
        "pseudo_threshold": .9, "ema_decay": .9999, "trade_off": 2.,
        "estimate_shift": True, "balance_source": True, "use_focal_loss": True,
        "shift_source": True, "sample_size": 100, "max_temporal_shift": 60,
        "domain_specific_bn": True, "shift_estimator": "AM", "run_validation": True,
        "output_student": True, "shape_da_mode": "batch_align",
        "shape_alignment_view": "none", "oracle_pseudo_labels": False,
        "adaptive_pseudo_selection": False, "shape_equivariance_weight": .05,
        "shape_equivariance_max_shift": 60, "detach_target_structure": False,
        "target_structure_detach_epochs": 0 if variant == "E" else 5,
    }


def source_checkpoint_path(source_root, task):
    source_alias = TASKS[task][0]
    return Path(source_root) / f"source_{source_alias}_seed1/fold_0/model.pt"


def build_job_graph():
    jobs = []
    for variant in ("E", "G"):
        for task in TASKS:
            uda = f"uda:{variant}:{task}"
            diagnostic = f"diagnostic:{variant}:{task}"
            # Measured Seed1 wall time showed FR2_DK1 longer than FR1_FR2;
            # actual GPU-seconds remain the primary dynamic balancing signal.
            load = "heavy" if task == "FR2_DK1" else "medium"
            jobs.append(Job(uda, "uda", task, variant, weight=load))
            jobs.append(Job(
                diagnostic, "diagnostic", task, variant,
                frozenset((uda,)), weight=load,
            ))
    for task in TASKS:
        jobs.append(Job(
            f"summary:{task}", "summary", task, None,
            frozenset((f"diagnostic:E:{task}", f"diagnostic:G:{task}")), "light",
        ))
    return jobs


def ready_jobs(jobs, completed, running):
    completed, running = set(completed), set(running)
    return [
        job for job in jobs
        if job.key not in completed and job.key not in running
        and job.dependencies <= completed
    ]


def select_gpu(free_gpus, gpu_seconds, job):
    free_gpus = tuple(str(gpu) for gpu in free_gpus)
    if not free_gpus:
        return None
    preferred = INITIAL_GPU_ASSIGNMENT.get(job.key)
    if job.kind == "uda" and preferred in free_gpus:
        return preferred
    return min(free_gpus, key=lambda gpu: (float(gpu_seconds.get(gpu, 0.)), gpu))


def _bool(value): return str(bool(value)).lower()


def git_commit_or_unavailable():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "unavailable"


def uda_command(job, args):
    _, source, _, target = TASKS[job.task]
    config = paired_uda_config(job.variant)
    experiment = f"{job.variant}_{job.task}_seed1"
    weights = source_checkpoint_path(args.source_root, job.task).parents[1]
    output = Path(args.experiment_root) / "outputs/uda"
    run = Path(args.experiment_root) / "runs" / experiment
    command = [
        args.python, "-u", "train.py", "-e", experiment,
        "--data_root", args.data_root, "--source", source, "--target", target,
        "--model", "psestructureprotoltae", "--structure-branch", "true",
        "--structure-exposer", "fourier", "--fourier_num_modes", "13",
        "--shape-dim", "128", "--shape-window-scales", "24",
        "--shape-window-stride", "8", "--shapelet-count", "16",
        "--shapelet-beta", "5", "--shape-resample-length", "16",
        "--shape-representation", "phase_moment", "--shape-injection", "current_query",
        "--structure-shift-mode", "none", "--shapelet-diversity-margin", "0.5",
        "--shapelet-diversity-weight", "0.01", "--shape-class-weight", "0.1",
        "--shape-align-weight", "0", "--source-minority-mode", "base",
        "--seed", "1", "--num_folds", "1", "--batch_size", "128",
        "--seq_length", "30", "--num_pixels", "64", "--closed_set", "true",
        "--with_shift_aug", "false", "--progress_bar", "off",
        "--output_dir", str(output), "--tensorboard_log_dir", str(run),
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
        "--detach-target-structure", "false", "--target-structure-detach-epochs",
        str(config["target_structure_detach_epochs"]),
    ]
    return command


def diagnostic_command(job, args):
    checkpoint = Path(args.experiment_root) / "outputs/uda" / f"{job.variant}_{job.task}_seed1/fold_0/checkpoint_last.pt"
    source = source_checkpoint_path(args.source_root, job.task)
    output = Path(args.experiment_root) / "diagnostics" / f"{job.variant}_{job.task}"
    return [
        args.python, "-u", "analysis/diagnose_eg_pair.py", "--mode", "diagnose",
        "--task", job.task, "--variant", job.variant, "--checkpoint", str(checkpoint),
        "--source-checkpoint", str(source), "--output-dir", str(output),
        "--data-root", args.data_root, "--device", "cuda",
        "--batch-size", str(args.audit_batch_size),
    ]


def summary_command(job, args):
    return [
        args.python, "-u", "analysis/diagnose_eg_pair.py", "--mode", "summarize-task",
        "--task", job.task, "--experiment-root", args.experiment_root,
        "--data-root", args.data_root, "--device", "cuda",
        "--batch-size", str(args.audit_batch_size),
    ]


def job_command(job, args):
    if job.kind == "uda": return uda_command(job, args)
    if job.kind == "diagnostic": return diagnostic_command(job, args)
    return summary_command(job, args)


def artifact_complete(job, args):
    root = Path(args.experiment_root)
    if job.kind == "uda":
        path = root / "outputs/uda" / f"{job.variant}_{job.task}_seed1/fold_0/checkpoint_last.pt"
        try:
            packet = require_complete_checkpoint(path)
            validate_uda_checkpoint(packet, job.task, job.variant)
            manifest_path = root / "pair_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected = next(
                    row["sha256"] for row in manifest["sources"]
                    if row["task"] == job.task
                )
                actual = sha256_file(source_checkpoint_path(args.source_root, job.task))
                if actual != expected:
                    raise ValueError(
                        f"source checkpoint changed during run: expected={expected} actual={actual}"
                    )
            return True
        except (FileNotFoundError, ValueError, KeyError, StopIteration): return False
    if job.kind == "diagnostic":
        path = root / "diagnostics" / f"{job.variant}_{job.task}/diagnostic_manifest.json"
    else:
        path = root / "reports" / f"{job.task}/summary_manifest.json"
    if not path.is_file(): return False
    try: return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, ValueError): return False


def _artifact_exists(job, args):
    root = Path(args.experiment_root)
    if job.kind == "uda": return (root / "outputs/uda" / f"{job.variant}_{job.task}_seed1/fold_0").exists()
    if job.kind == "diagnostic": return (root / "diagnostics" / f"{job.variant}_{job.task}").exists()
    return (root / "reports" / job.task).exists()


def _artifact_path(job, args):
    root = Path(args.experiment_root)
    if job.kind == "uda":
        return root / "outputs/uda" / f"{job.variant}_{job.task}_seed1"
    if job.kind == "diagnostic":
        return root / "diagnostics" / f"{job.variant}_{job.task}"
    return root / "reports" / job.task


def _quarantine_incomplete(job, args):
    path = _artifact_path(job, args)
    if path.exists() and not artifact_complete(job, args):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = path.with_name(f"{path.name}.failed_{stamp}")
        path.rename(destination)
    log = job_log_path(args.experiment_root, job)
    if log.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log.rename(log.with_name(f"{log.stem}.failed_{stamp}{log.suffix}"))


def preflight(args):
    root = Path(args.experiment_root); root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for task in TASKS:
        checkpoint = source_checkpoint_path(args.source_root, task)
        manifests.append(validate_source(checkpoint, task, "cpu"))
    config_e, config_g = paired_uda_config("E"), paired_uda_config("G")
    differences = {key: (config_e[key], config_g[key]) for key in config_e if config_e[key] != config_g[key]}
    if set(differences) != {"target_structure_detach_epochs"}:
        raise RuntimeError(f"E/G configuration differs beyond detach epochs: {differences}")
    git_commit = git_commit_or_unavailable()
    payload = {
        "git_commit": git_commit,
        "git_commit_available": git_commit != "unavailable",
        "sources": manifests, "E_config": config_e, "G_config": config_g,
        "differences": differences,
    }
    manifest_path = root / "pair_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_sources = {row["task"]: row["sha256"] for row in existing["sources"]}
        new_sources = {row["task"]: row["sha256"] for row in manifests}
        if old_sources != new_sources:
            raise RuntimeError(
                f"source checkpoint SHA256 changed since initial run: "
                f"recorded={old_sources} current={new_sources}"
            )
        if existing.get("E_config") != config_e or existing.get("G_config") != config_g:
            raise RuntimeError("paired training configuration changed since initial run")
        return existing
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _timestamp(): return datetime.now(timezone.utc).astimezone().isoformat()


def job_log_path(root, job):
    if job.kind == "uda":
        name = f"{job.variant}_{job.task}_seed1.log"
    elif job.kind == "diagnostic":
        name = f"DIAG_{job.variant}_{job.task}_seed1.log"
    else:
        name = f"SUMMARY_{job.task}.log"
    return Path(root) / "logs" / name


def _append_event(path, event):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def _save_state(path, state):
    Path(path).write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _load_state(path, gpus):
    if Path(path).is_file(): return json.loads(Path(path).read_text(encoding="utf-8"))
    return {"completed": [], "failed": {}, "gpu_seconds": {gpu: 0. for gpu in gpus}}


def run_scheduler(args):
    root = Path(args.experiment_root)
    for folder in ("outputs/uda", "logs", "runs", "diagnostics", "reports"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    preflight(args)
    jobs = build_job_graph()
    if args.only_task:
        jobs = [job for job in jobs if job.task == args.only_task]
    if args.only_job:
        selected = next((job for job in jobs if job.key == args.only_job), None)
        if selected is None: raise ValueError(f"unknown --only-job: {args.only_job}")
        required = set(selected.dependencies) | {selected.key}
        jobs = [job for job in jobs if job.key in required]
    state_path = root / "scheduler_state.json"; event_path = root / "scheduler_events.jsonl"
    if args.mode == "run" and state_path.exists():
        raise RuntimeError(f"scheduler state exists; use resume: {state_path}")
    gpus = tuple(str(value) for value in args.gpus)
    state = _load_state(state_path, gpus)
    completed = set(state.get("completed", []))
    for job in jobs:
        if artifact_complete(job, args): completed.add(job.key)
    if args.retry_failed:
        if args.mode != "resume":
            raise ValueError("--retry-failed requires resume mode")
        for job in jobs:
            if job.key in state.get("failed", {}):
                if job.key not in completed:
                    _quarantine_incomplete(job, args)
                state["failed"].pop(job.key, None)
    if args.mode == "dry-run":
        for job in jobs:
            gpu = INITIAL_GPU_ASSIGNMENT.get(job.key, "dynamic")
            print(f"DRY_RUN|job={job.key}|depends={','.join(sorted(job.dependencies))}|gpu={gpu}|command={shlex.join(job_command(job, args))}")
        return
    running = {}; free = set(gpus)
    while len(completed) + len(state.get("failed", {})) < len(jobs):
        progressed = False
        for job in ready_jobs(jobs, completed, running):
            if job.key in state.get("failed", {}): continue
            if _artifact_exists(job, args) and not artifact_complete(job, args):
                state.setdefault("failed", {})[job.key] = "incomplete artifact exists; refusing overwrite"
                _append_event(event_path, {"time": _timestamp(), "job": job.key, "status": "blocked_incomplete"})
                progressed = True; continue
            if job.kind == "summary":
                gpu = min(free, key=lambda value: state["gpu_seconds"].get(value, 0.)) if free else None
                if gpu is None: continue
            else:
                gpu = select_gpu(free, state["gpu_seconds"], job)
                if gpu is None: continue
            command = job_command(job, args)
            log = job_log_path(root, job)
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            stream = log.open("w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            running[job.key] = {"process": process, "job": job, "gpu": gpu, "start": time.monotonic(), "stream": stream}
            free.remove(gpu); progressed = True
            _append_event(event_path, {"time": _timestamp(), "job": job.key, "status": "started", "gpu": gpu, "command": command})
        for key, item in list(running.items()):
            code = item["process"].poll()
            if code is None: continue
            item["stream"].close(); elapsed = time.monotonic() - item["start"]
            gpu = item["gpu"]; free.add(gpu)
            state["gpu_seconds"][gpu] = float(state["gpu_seconds"].get(gpu, 0.)) + elapsed
            if code == 0 and artifact_complete(item["job"], args):
                completed.add(key); status = "complete"
            else:
                state.setdefault("failed", {})[key] = f"exit={code} artifact_complete={artifact_complete(item['job'], args)}"
                status = "failed"
            del running[key]; progressed = True
            _append_event(event_path, {"time": _timestamp(), "job": key, "status": status, "gpu": gpu, "elapsed_seconds": elapsed})
        state["completed"] = sorted(completed); _save_state(state_path, state)
        if not progressed:
            if not running: break
            time.sleep(float(args.poll_seconds))
    subprocess.run([
        args.python, "-u", "analysis/diagnose_eg_pair.py", "--mode", "summarize-all",
        "--experiment-root", args.experiment_root,
    ], check=False)
    if state.get("failed"):
        print("SCHEDULER_FINISHED_WITH_FAILURES|" + json.dumps(state["failed"], sort_keys=True))
        raise SystemExit(1)
    print(f"SCHEDULER_COMPLETE|root={root}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--experiment-root", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs=4, default=("0", "1", "2", "3"))
    parser.add_argument("--audit-batch-size", type=int, default=64)
    parser.add_argument("--poll-seconds", type=float, default=5.)
    parser.add_argument("--only-job")
    parser.add_argument("--only-task", choices=tuple(TASKS))
    parser.add_argument("--retry-failed", action="store_true")
    return parser


if __name__ == "__main__": run_scheduler(build_parser().parse_args())
