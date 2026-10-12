#!/usr/bin/env python3
"""Event-driven four-GPU launcher for the frozen structure semantic ABC audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.diagnose_eg_pair import (
    TASKS, require_complete_checkpoint, validate_source, validate_uda_checkpoint,
)
from analysis.structure_semantic_abc import (
    STAGE_ORDER, required_stage_artifacts,
)


EXPERIMENT_ROOT = Path("experiments/eg_recheck_pair_seed1")
SOURCE_ROOT = Path("outputs/recheck_e_final_seed1/source")
INITIAL_GPU_ASSIGNMENT = {
    "extract:E:FR1_FR2": "0", "extract:E:FR2_DK1": "1",
    "extract:G:FR1_FR2": "2", "extract:G:FR2_DK1": "3",
}


@dataclass(frozen=True)
class Job:
    key: str
    kind: str
    task: str | None = None
    variant: str | None = None
    stage: str | None = None
    dependencies: frozenset[str] = field(default_factory=frozenset)
    weight: str = "cpu"


def build_job_graph():
    jobs = []
    for variant in ("E", "G"):
        for task in TASKS:
            previous = None
            for stage in STAGE_ORDER:
                key = f"{stage}:{variant}:{task}"
                dependencies = frozenset() if previous is None else frozenset((previous,))
                weight = "gpu" if stage in ("extract", "C") else "cpu"
                jobs.append(Job(key, "checkpoint", task, variant, stage, dependencies, weight))
                previous = key
    for task in TASKS:
        jobs.append(Job(
            f"pair:{task}", "pair", task=task,
            dependencies=frozenset((f"report:E:{task}", f"report:G:{task}")),
        ))
    jobs.append(Job(
        "summary", "summary",
        dependencies=frozenset(f"pair:{task}" for task in TASKS),
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
    free_gpus = tuple(str(value) for value in free_gpus)
    if not free_gpus:
        return None
    preferred = INITIAL_GPU_ASSIGNMENT.get(job.key)
    if preferred in free_gpus:
        return preferred
    return min(free_gpus, key=lambda gpu: (float(gpu_seconds.get(gpu, 0.)), gpu))


def checkpoint_path(experiment_root, variant, task):
    return Path(experiment_root) / "outputs" / "uda" / f"{variant}_{task}_seed1" / "fold_0" / "checkpoint_last.pt"


def source_checkpoint_path(source_root, task):
    source_alias = TASKS[task][0]
    return Path(source_root) / f"source_{source_alias}_seed1" / "fold_0" / "model.pt"


def audit_root(args):
    return Path(args.experiment_root) / "abc_audit"


def job_command(job, args):
    root = audit_root(args)
    if job.kind == "checkpoint":
        return [
            args.python, "-u", "analysis/structure_semantic_abc.py",
            "--mode", "checkpoint", "--stage", job.stage,
            "--task", job.task, "--variant", job.variant,
            "--checkpoint", str(checkpoint_path(args.experiment_root, job.variant, job.task)),
            "--source-checkpoint", str(source_checkpoint_path(args.source_root, job.task)),
            "--audit-root", str(root), "--data-root", str(args.data_root),
            "--device", "cuda" if job.weight == "gpu" else "cpu",
            "--batch-size", str(args.batch_size),
            "--pixel-budget", str(args.pixel_budget),
            "--gradient-samples-per-class", str(args.gradient_samples_per_class),
        ]
    if job.kind == "pair":
        return [
            args.python, "-u", "analysis/structure_semantic_abc.py",
            "--mode", "pair-report", "--task", job.task,
            "--audit-root", str(root),
        ]
    return [
        args.python, "-u", "analysis/structure_semantic_abc.py",
        "--mode", "summary", "--audit-root", str(root),
    ]


def artifact_complete(job, args):
    root = audit_root(args)
    if job.kind == "pair":
        return (root / "pair_reports" / f"{job.task}.md").is_file()
    if job.kind == "summary":
        return (root / "ABC_summary.md").is_file()
    output = root / f"{job.variant}_{job.task}"
    state_path = output / "stage_status.json"
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        complete = state.get("stages", {}).get(job.stage, {}).get("status") == "complete"
    except (OSError, ValueError):
        return False
    return complete and all((output / name).is_file() for name in required_stage_artifacts(job.stage))


def preflight(args, cuda_available=None, cuda_count=None):
    blockers = []
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        blockers.append({"code": "missing_data_root", "path": str(data_root)})
    for variant in ("E", "G"):
        for task in TASKS:
            path = checkpoint_path(args.experiment_root, variant, task)
            if not path.is_file():
                blockers.append({
                    "code": "missing_checkpoint", "variant": variant,
                    "task": task, "path": str(path),
                })
            else:
                try:
                    validate_uda_checkpoint(require_complete_checkpoint(path), task, variant)
                except Exception as error:
                    blockers.append({
                        "code": "invalid_checkpoint", "variant": variant,
                        "task": task, "path": str(path), "error": str(error),
                    })
    for task in TASKS:
        path = source_checkpoint_path(args.source_root, task)
        if not path.is_file():
            blockers.append({"code": "missing_source_checkpoint", "task": task, "path": str(path)})
        else:
            try:
                validate_source(path, task, "cpu")
            except Exception as error:
                blockers.append({
                    "code": "invalid_source_checkpoint", "task": task,
                    "path": str(path), "error": str(error),
                })
    available = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    count = torch.cuda.device_count() if cuda_count is None else int(cuda_count)
    if not available:
        blockers.append({"code": "cuda_unavailable"})
    if count < 4:
        blockers.append({"code": "insufficient_cuda_devices", "required": 4, "available": count})
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        blockers.append({"code": "four_distinct_gpus_required", "requested": list(args.gpus)})
    commands = [job_command(job, args) for job in build_job_graph()]
    training = any("train.py" in token for command in commands for token in command)
    if training:
        blockers.append({"code": "training_command_forbidden"})
    return {
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers, "cuda_count": count,
        "training_commands_present": training,
    }


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def save_state(path, state):
    _atomic_json(path, state)


def load_state(path, gpus):
    path = Path(path)
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {"completed": [], "failed": {}, "gpu_seconds": {}}
    state.setdefault("completed", [])
    state.setdefault("failed", {})
    state.setdefault("gpu_seconds", {})
    for gpu in gpus:
        state["gpu_seconds"].setdefault(str(gpu), 0.)
    return state


def _timestamp():
    return datetime.now(timezone.utc).astimezone().isoformat()


def append_event(path, event):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": _timestamp(), **event}, sort_keys=True) + "\n")


def write_manifest(args, preflight_result):
    root = audit_root(args); root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_root": str(args.experiment_root),
        "audit_root": str(root), "seed": 1,
        "tasks": list(TASKS), "variants": ["E", "G"],
        "initial_gpu_assignment": INITIAL_GPU_ASSIGNMENT,
        "target_test_used": False, "training_allowed": False,
        "preflight": preflight_result,
    }
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def run_dry_run(args):
    check = preflight(args)
    write_manifest(args, check)
    for job in build_job_graph():
        print("DRY_RUN|" + json.dumps({
            "job": job.key, "dependencies": sorted(job.dependencies),
            "weight": job.weight, "command": job_command(job, args),
        }, sort_keys=True))
    print("PREFLIGHT|" + json.dumps(check, sort_keys=True))
    return check


def run_scheduler(args):
    check = preflight(args)
    write_manifest(args, check)
    if check["blockers"]:
        raise RuntimeError("ABC preflight blocked: " + json.dumps(check["blockers"], sort_keys=True))
    root = audit_root(args)
    state_path, event_path = root / "scheduler_state.json", root / "scheduler_events.jsonl"
    state = load_state(state_path, args.gpus)
    jobs = build_job_graph()
    completed = set(state["completed"])
    for job in jobs:
        if artifact_complete(job, args):
            completed.add(job.key)
    failed = dict(state["failed"])
    if args.mode == "resume" and args.retry_failed:
        failed = {}
    running = {}
    cpu_limit = 4
    while len(completed) < len(jobs):
        progressed = False
        used_gpus = {info["gpu"] for info in running.values() if info["gpu"] is not None}
        free_gpus = [gpu for gpu in args.gpus if gpu not in used_gpus]
        cpu_running = sum(info["job"].weight == "cpu" for info in running.values())
        for job in ready_jobs(jobs, completed, running):
            if job.key in failed:
                continue
            gpu = None
            if job.weight == "gpu":
                gpu = select_gpu(free_gpus, state["gpu_seconds"], job)
                if gpu is None:
                    continue
                free_gpus.remove(gpu)
            elif cpu_running >= cpu_limit:
                continue
            command = job_command(job, args)
            log = root / "logs" / (job.key.replace(":", "_") + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("a", encoding="utf-8")
            env = os.environ.copy()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, text=True,
            )
            running[job.key] = {
                "job": job, "process": process, "gpu": gpu,
                "started": time.monotonic(), "stream": stream,
            }
            if job.weight == "cpu":
                cpu_running += 1
            append_event(event_path, {"event": "started", "job": job.key, "gpu": gpu, "command": command})
            progressed = True
        finished = []
        for key, info in running.items():
            code = info["process"].poll()
            if code is None:
                continue
            elapsed = time.monotonic() - info["started"]
            info["stream"].close()
            if code == 0 and artifact_complete(info["job"], args):
                completed.add(key)
                if info["gpu"] is not None:
                    state["gpu_seconds"][info["gpu"]] += elapsed
                append_event(event_path, {
                    "event": "completed", "job": key, "gpu": info["gpu"],
                    "elapsed_seconds": elapsed, "peak_memory_mb": "recorded_by_stage_if_available",
                })
            else:
                failed[key] = {"returncode": code, "elapsed_seconds": elapsed}
                append_event(event_path, {
                    "event": "failed", "job": key, "gpu": info["gpu"],
                    "returncode": code, "elapsed_seconds": elapsed,
                })
            finished.append(key); progressed = True
        for key in finished:
            del running[key]
        state.update({"completed": sorted(completed), "failed": failed, "updated_at": _timestamp()})
        save_state(state_path, state)
        if not running and not progressed:
            break
        if running and not finished:
            time.sleep(args.poll_seconds)
    if failed:
        raise RuntimeError("ABC scheduler finished with failures: " + json.dumps(failed, sort_keys=True))
    if len(completed) != len(jobs):
        blocked = sorted(job.key for job in jobs if job.key not in completed)
        raise RuntimeError("ABC scheduler dependency-blocked jobs: " + json.dumps(blocked))
    print(f"SCHEDULER_COMPLETE|root={root}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--experiment-root", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs=4, default=("0", "1", "2", "3"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--gradient-samples-per-class", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=3.)
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "dry-run":
        run_dry_run(args)
    else:
        run_scheduler(args)


if __name__ == "__main__":
    main()
