#!/usr/bin/env python3
"""Event-driven four-GPU launcher for the frozen E1/E2/E3 audit."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import torch
from analysis.diagnose_eg_pair import TASKS, require_complete_checkpoint, validate_source, validate_uda_checkpoint
from analysis.structure_e123_audit import STAGE_ORDER, required_stage_artifacts

EXPERIMENT_ROOT = Path("experiments/eg_recheck_pair_seed1")
SOURCE_ROOT = Path("outputs/recheck_e_final_seed1/source")
INITIAL_GPU_ASSIGNMENT = {
    "extract:E:FR1_FR2": "0", "extract:E:FR2_DK1": "1",
    "extract:G:FR1_FR2": "2", "extract:G:FR2_DK1": "3",
}


@dataclass(frozen=True)
class Job:
    key: str; kind: str; task: str | None = None; variant: str | None = None
    stage: str | None = None; dependencies: frozenset[str] = field(default_factory=frozenset)
    weight: str = "cpu"


def build_job_graph():
    jobs = []
    for variant in ("E", "G"):
        for task in TASKS:
            previous = None
            for stage in STAGE_ORDER:
                key = f"{stage}:{variant}:{task}"
                jobs.append(Job(
                    key, "checkpoint", task, variant, stage,
                    frozenset() if previous is None else frozenset((previous,)),
                    "gpu" if stage in ("extract", "E1", "E2") else "cpu",
                ))
                previous = key
    for task in TASKS:
        jobs.append(Job(f"pair:{task}", "pair", task=task, dependencies=frozenset((f"report:E:{task}", f"report:G:{task}"))))
    jobs.append(Job("summary", "summary", dependencies=frozenset(f"pair:{task}" for task in TASKS)))
    return jobs


def checkpoint_path(root, variant, task):
    return Path(root) / "outputs" / "uda" / f"{variant}_{task}_seed1" / "fold_0" / "checkpoint_last.pt"


def source_path(root, task):
    return Path(root) / f"source_{TASKS[task][0]}_seed1" / "fold_0" / "model.pt"


def audit_root(args): return Path(args.experiment_root) / "e123_audit"


def job_command(job, args):
    if job.kind == "checkpoint":
        return [
            args.python, "-u", "analysis/structure_e123_audit.py", "--mode", "checkpoint",
            "--stage", job.stage, "--task", job.task, "--variant", job.variant,
            "--checkpoint", str(checkpoint_path(args.experiment_root, job.variant, job.task)),
            "--source-checkpoint", str(source_path(args.source_root, job.task)),
            "--audit-root", str(audit_root(args)), "--abc-root", str(Path(args.experiment_root) / "abc_audit"),
            "--data-root", str(args.data_root), "--device", "cuda" if job.weight == "gpu" else "cpu",
            "--batch-size", str(args.batch_size), "--pixel-budget", str(args.pixel_budget),
            "--representative-count", str(args.representative_count),
        ]
    if job.kind == "pair":
        return [args.python, "-u", "analysis/structure_e123_audit.py", "--mode", "pair-report", "--task", job.task, "--audit-root", str(audit_root(args))]
    return [args.python, "-u", "analysis/structure_e123_audit.py", "--mode", "summary", "--audit-root", str(audit_root(args))]


def ready_jobs(jobs, completed, running):
    return [job for job in jobs if job.key not in completed and job.key not in running and job.dependencies <= set(completed)]


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp"); temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"); os.replace(temporary, path)


def _timestamp(): return datetime.now(timezone.utc).astimezone().isoformat()


def artifact_complete(job, args):
    root = audit_root(args)
    if job.kind == "pair": return (root / "pair_reports" / f"{job.task}.md").is_file()
    if job.kind == "summary":
        return all((root / name).is_file() for name in ("E123_summary.md", "E2_cross_task_comparison.csv"))
    output = root / f"{job.variant}_{job.task}"
    return all((output / name).is_file() for name in required_stage_artifacts(job.stage))


def preflight(args, cuda_available=None, cuda_count=None):
    blockers = []
    if not Path(args.data_root).is_dir(): blockers.append({"code": "missing_data_root", "path": str(args.data_root)})
    for variant in ("E", "G"):
        for task in TASKS:
            path = checkpoint_path(args.experiment_root, variant, task)
            if not path.is_file(): blockers.append({"code": "missing_checkpoint", "path": str(path), "variant": variant, "task": task})
            else:
                try: validate_uda_checkpoint(require_complete_checkpoint(path), task, variant)
                except Exception as error: blockers.append({"code": "invalid_checkpoint", "path": str(path), "error": str(error)})
    for task in TASKS:
        path = source_path(args.source_root, task)
        if not path.is_file(): blockers.append({"code": "missing_source_checkpoint", "path": str(path), "task": task})
        else:
            try: validate_source(path, task, "cpu")
            except Exception as error: blockers.append({"code": "invalid_source_checkpoint", "path": str(path), "error": str(error)})
    available = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    count = torch.cuda.device_count() if cuda_count is None else int(cuda_count)
    if not available: blockers.append({"code": "cuda_unavailable"})
    if count < 4: blockers.append({"code": "insufficient_cuda_devices", "required": 4, "available": count})
    training = any("train.py" in token for job in build_job_graph() for token in job_command(job, args))
    if training: blockers.append({"code": "training_command_forbidden"})
    return {"status": "ready" if not blockers else "blocked", "blockers": blockers, "cuda_count": count, "training_commands_present": training}


def write_manifest(args, check):
    root = audit_root(args); root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "manifest.json", {"seed": 1, "target_test_used": False, "E4_implemented": False, "training_allowed": False, "preflight": check, "initial_gpu_assignment": INITIAL_GPU_ASSIGNMENT})


def run_dry(args):
    check = preflight(args); write_manifest(args, check)
    for job in build_job_graph(): print("DRY_RUN|" + json.dumps({"job": job.key, "dependencies": sorted(job.dependencies), "weight": job.weight, "command": job_command(job, args)}, sort_keys=True))
    print("PREFLIGHT|" + json.dumps(check, sort_keys=True))


def run_scheduler(args):
    check = preflight(args); write_manifest(args, check)
    if check["blockers"]: raise RuntimeError("E123 preflight blocked: " + json.dumps(check["blockers"], sort_keys=True))
    root = audit_root(args); state_path = root / "scheduler_state.json"; events = root / "scheduler_events.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"completed": [], "failed": {}, "gpu_seconds": {}}
    for gpu in args.gpus: state["gpu_seconds"].setdefault(gpu, 0.)
    completed = set(state.get("completed", [])); failed = {} if args.retry_failed else dict(state.get("failed", {}))
    jobs = build_job_graph()
    completed.update(job.key for job in jobs if artifact_complete(job, args))
    running = {}
    while len(completed) < len(jobs):
        progressed = False
        used = {value["gpu"] for value in running.values() if value["gpu"] is not None}
        free = [gpu for gpu in args.gpus if gpu not in used]
        for job in ready_jobs(jobs, completed, running):
            if job.key in failed: continue
            gpu = None
            if job.weight == "gpu":
                if not free: continue
                preferred = INITIAL_GPU_ASSIGNMENT.get(job.key)
                gpu = preferred if preferred in free else min(free, key=lambda item: (state["gpu_seconds"][item], item)); free.remove(gpu)
            command = job_command(job, args); log = root / "logs" / f"{job.key.replace(':', '_')}.log"; log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("a", encoding="utf-8"); env = os.environ.copy()
            if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
            running[job.key] = {"job": job, "process": process, "gpu": gpu, "start": time.monotonic(), "stream": stream}
            with events.open("a", encoding="utf-8") as out: out.write(json.dumps({"time": _timestamp(), "event": "started", "job": job.key, "gpu": gpu}) + "\n")
            progressed = True
        finished = []
        for key, info in running.items():
            code = info["process"].poll()
            if code is None: continue
            elapsed = time.monotonic() - info["start"]; info["stream"].close(); finished.append(key); progressed = True
            if code == 0 and artifact_complete(info["job"], args):
                completed.add(key)
                if info["gpu"] is not None: state["gpu_seconds"][info["gpu"]] += elapsed
                event = {"time": _timestamp(), "event": "completed", "job": key, "gpu": info["gpu"], "elapsed_seconds": elapsed}
            else:
                failed[key] = {"returncode": code, "elapsed_seconds": elapsed}
                event = {"time": _timestamp(), "event": "failed", "job": key, "gpu": info["gpu"], "elapsed_seconds": elapsed, "returncode": code}
            with events.open("a", encoding="utf-8") as out: out.write(json.dumps(event) + "\n")
        for key in finished: del running[key]
        state.update({"completed": sorted(completed), "failed": failed, "updated_at": _timestamp()}); _atomic_json(state_path, state)
        if not running and not progressed: break
        if running and not finished: time.sleep(args.poll_seconds)
    if failed: raise RuntimeError("E123 scheduler failures: " + json.dumps(failed, sort_keys=True))
    if len(completed) != len(jobs): raise RuntimeError("E123 scheduler dependency blocked")
    print(f"SCHEDULER_COMPLETE|root={root}")


def build_parser():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--experiment-root", default=str(EXPERIMENT_ROOT)); parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data"); parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs=4, default=("0", "1", "2", "3")); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192); parser.add_argument("--representative-count", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=3.); parser.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "dry-run": run_dry(args)
    else: run_scheduler(args)


if __name__ == "__main__": main()
