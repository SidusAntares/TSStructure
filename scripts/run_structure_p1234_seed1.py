#!/usr/bin/env python3
"""Four-GPU resumable scheduler for the frozen P1-P4 audit."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_p1234_audit import STAGE_ARTIFACTS, TASKS

INITIAL_GPU_ASSIGNMENT = {
    "P1:E:FR1_FR2": "0", "P1:E:FR2_DK1": "1",
    "P1:G:FR1_FR2": "2", "P1:G:FR2_DK1": "3",
}


@dataclass(frozen=True)
class Job:
    key: str
    kind: str
    stage: str | None = None
    variant: str | None = None
    task: str | None = None
    dependencies: frozenset[str] = field(default_factory=frozenset)
    weight: str = "cpu"


def build_job_graph():
    jobs = []
    for variant in ("E", "G"):
        for task in TASKS:
            p1 = f"P1:{variant}:{task}"
            jobs.extend([
                Job(p1, "checkpoint", "P1", variant, task, weight="cpu"),
                Job(f"P2:{variant}:{task}", "checkpoint", "P2", variant, task, weight="gpu"),
                Job(f"P3:{variant}:{task}", "checkpoint", "P3", variant, task, frozenset((p1,)), "cpu"),
                Job(f"P4:{variant}:{task}", "checkpoint", "P4", variant, task, weight="gpu"),
                Job(f"report:{variant}:{task}", "checkpoint", "report", variant, task,
                    frozenset((p1, f"P2:{variant}:{task}", f"P3:{variant}:{task}", f"P4:{variant}:{task}"))),
            ])
    jobs.append(Job("summary", "summary", dependencies=frozenset(
        f"report:{variant}:{task}" for variant in ("E", "G") for task in TASKS
    )))
    return jobs


def _output(args, job):
    return Path(args.audit_root) / f"{job.variant}_{job.task}"


def _checkpoint_root(args, variant, task):
    return Path(args.experiment_root) / "outputs" / "uda" / f"{variant}_{task}_seed1" / "fold_0"


def _source(args, task):
    source = TASKS[task][0]
    return Path(args.source_root) / f"source_{source}_seed1" / "fold_0" / "model.pt"


def _cache(args, variant, task):
    return Path(args.experiment_root) / "abc_audit" / f"{variant}_{task}"


def job_command(job, args):
    if job.kind == "summary":
        return [args.python, "-u", "analysis/structure_p1234_audit.py", "--mode", "summary", "--audit-root", str(args.audit_root)]
    fold = _checkpoint_root(args, job.variant, job.task)
    return [
        args.python, "-u", "analysis/structure_p1234_audit.py",
        "--mode", "checkpoint", "--stage", job.stage,
        "--variant", job.variant, "--task", job.task,
        "--cache-root", str(_cache(args, job.variant, job.task)),
        "--audit-root", str(args.audit_root),
        "--source-checkpoint", str(_source(args, job.task)),
        "--best-checkpoint", str(fold / "checkpoint_best.pt"),
        "--final-checkpoint", str(fold / "checkpoint_last.pt"),
        "--data-root", str(args.data_root),
        "--device", "cuda" if job.weight == "gpu" else "cpu",
        "--batch-size", str(args.batch_size), "--pixel-budget", str(args.pixel_budget),
        "--samples-per-class", str(args.samples_per_class),
        "--e123-root", str(Path(args.experiment_root) / "e123_audit"),
    ]


def artifact_complete(job, args):
    if job.kind == "summary":
        return (Path(args.audit_root) / "P1234_summary.md").is_file()
    return all((_output(args, job) / name).is_file() for name in STAGE_ARTIFACTS[job.stage])


def job_log_path(args, job):
    return Path(args.log_root) / f"{job.key.replace(':', '_')}.log"


def preflight(args, require_gpu=True):
    blockers = []
    for variant in ("E", "G"):
        for task in TASKS:
            for kind, path in (
                ("cache", _cache(args, variant, task) / "checkpoint_manifest.json"),
                ("final", _checkpoint_root(args, variant, task) / "checkpoint_last.pt"),
                ("source", _source(args, task)),
            ):
                if not path.is_file(): blockers.append({"code": f"missing_{kind}", "path": str(path), "variant": variant, "task": task})
    if any("train.py" in token for job in build_job_graph() for token in job_command(job, args)):
        blockers.append({"code": "training_command_forbidden"})
    if require_gpu:
        try:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
                blockers.append({"code": "insufficient_cuda", "available": torch.cuda.device_count()})
        except Exception as error:
            blockers.append({"code": "cuda_check_failed", "error": str(error)})
    return {"status": "ready" if not blockers else "blocked", "blockers": blockers, "target_test_used": False, "training_allowed": False}


def _write_json(path, packet):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _timestamp():
    return datetime.now(timezone.utc).astimezone().isoformat()


def run_dry(args):
    check = preflight(args, require_gpu=False)
    _write_json(Path(args.audit_root) / "manifest.json", {"preflight": check, "initial_gpu_assignment": INITIAL_GPU_ASSIGNMENT, "E4_implemented": False})
    for job in build_job_graph():
        print("DRY_RUN|" + json.dumps({"job": job.key, "dependencies": sorted(job.dependencies), "weight": job.weight, "command": job_command(job, args)}))
    print("PREFLIGHT|" + json.dumps(check, sort_keys=True))


def run_scheduler(args):
    check = preflight(args, require_gpu=True)
    _write_json(Path(args.audit_root) / "manifest.json", {"preflight": check, "initial_gpu_assignment": INITIAL_GPU_ASSIGNMENT, "E4_implemented": False})
    if check["blockers"]:
        raise RuntimeError("P1234 preflight blocked: " + json.dumps(check["blockers"], sort_keys=True))
    root = Path(args.audit_root); state_path = root / "scheduler_state.json"; events = root / "scheduler_events.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"completed": [], "failed": {}, "gpu_seconds": {}}
    for gpu in args.gpus: state["gpu_seconds"].setdefault(gpu, 0.)
    completed = set(state.get("completed", [])); failed = {} if args.retry_failed else dict(state.get("failed", {}))
    jobs = build_job_graph(); completed.update(job.key for job in jobs if artifact_complete(job, args))
    running = {}
    while len(completed) < len(jobs):
        progressed = False; used = {v["gpu"] for v in running.values() if v["gpu"] is not None}; free = [g for g in args.gpus if g not in used]
        for job in jobs:
            if job.key in completed or job.key in running or job.key in failed or not job.dependencies <= completed: continue
            gpu = None
            if job.weight == "gpu":
                if not free: continue
                preferred = INITIAL_GPU_ASSIGNMENT.get(job.key)
                gpu = preferred if preferred in free else min(free, key=lambda x: (state["gpu_seconds"][x], x)); free.remove(gpu)
            log = job_log_path(args, job); log.parent.mkdir(parents=True, exist_ok=True); stream = log.open("a", encoding="utf-8")
            env = os.environ.copy()
            if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(job_command(job, args), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
            running[job.key] = {"job": job, "process": process, "gpu": gpu, "stream": stream, "start": time.monotonic()}
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
            else: failed[key] = {"returncode": code, "elapsed_seconds": elapsed}
        for key in finished: del running[key]
        state.update({"completed": sorted(completed), "failed": failed, "updated_at": _timestamp()}); _write_json(state_path, state)
        if not running and not progressed: break
        if running and not finished: time.sleep(args.poll_seconds)
    if failed: raise RuntimeError("P1234 stages failed: " + json.dumps(failed, sort_keys=True))
    if len(completed) != len(jobs): raise RuntimeError("P1234 dependency graph blocked")
    print(f"SCHEDULER_COMPLETE|root={root}")


def build_parser():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--experiment-root", default="experiments/eg_recheck_pair_seed1")
    parser.add_argument("--source-root", default="outputs/recheck_e_final_seed1/source")
    parser.add_argument("--audit-root", default="experiments/eg_recheck_pair_seed1/p1234_audit")
    parser.add_argument("--log-root", default="logs/structure_p1234_seed1")
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--python", default=sys.executable); parser.add_argument("--gpus", nargs=4, default=("0", "1", "2", "3"))
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=3.); parser.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "dry-run": run_dry(args)
    else: run_scheduler(args)


if __name__ == "__main__": main()
