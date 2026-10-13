#!/usr/bin/env python3
"""Event-driven frozen all-pair semantics scheduler."""

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

import torch

from analysis.structure_pair_semantics_audit import REQUIRED_OUTPUTS, TASKS, sha256_file


DEFAULT_CHECKPOINT_ROOTS = (
    "experiments/eg_recheck_pair_seed1/outputs/uda",
    "outputs/recheck_e_final_seed1/uda",
    "outputs/structure_phase_equivariance_4tasks_seed1/uda",
    "experiments/structure_e_cross_seed1/outputs/uda",
)
DEFAULT_SOURCE_ROOTS = (
    "outputs/recheck_e_final_seed1/source",
    "outputs/structure_phase_moment_4tasks_seed1/source",
)
DEFAULT_CACHE_ROOTS = (
    "experiments/eg_recheck_pair_seed1/abc_audit",
    "experiments/structure_pair_semantics_seed1/cache",
)


@dataclass(frozen=True)
class InventoryItem:
    variant: str
    task: str
    checkpoint: str
    source_checkpoint: str
    cache_dir: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class Job:
    key: str
    kind: str
    variant: str | None = None
    task: str | None = None
    item: InventoryItem | None = None
    dependencies: frozenset[str] = field(default_factory=frozenset)
    weight: str = "cpu"


def _config(packet):
    raw = packet.get("config", {})
    return dict(raw if isinstance(raw, dict) else vars(raw))


def _checkpoint_identity(path):
    packet = torch.load(path, map_location="cpu", weights_only=False)
    config = _config(packet)
    detach_epochs = int(config.get("target_structure_detach_epochs", 0))
    if detach_epochs not in (0, 5): return None
    variant = "G" if detach_epochs == 5 else "E"
    if config.get("shape_representation") != "phase_moment" or config.get("shape_injection") != "current_query":
        return None
    if abs(float(config.get("shape_equivariance_weight", 0)) - .05) > 1e-9:
        return None
    task = next((name for name, (_, source, _, target) in TASKS.items()
                 if config.get("source") == source and config.get("target") == target), None)
    if task is None: return None
    if int(packet.get("epoch", -1)) != 19: return None
    return variant, task, config


def _source_for(source_roots, source_domain, classes):
    for root in map(Path, source_roots):
        if not root.is_dir(): continue
        for path in sorted(root.rglob("model.pt")):
            try:
                config = _config(torch.load(path, map_location="cpu", weights_only=False))
            except Exception:
                continue
            if config.get("source") == source_domain and config.get("target") == source_domain and list(config.get("classes", [])) == list(classes):
                return path
    return None


def _cache_for(cache_roots, variant, task, digest):
    for root in map(Path, cache_roots):
        candidate = root / f"{variant}_{task}"
        manifest = candidate / "checkpoint_manifest.json"
        if not manifest.is_file(): continue
        try: packet = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError): continue
        if packet.get("checkpoint_sha256") == digest:
            return candidate
    return Path(cache_roots[-1]) / f"{variant}_{task}"


def resolve_inventory(checkpoint_roots, source_roots, tasks=TASKS, cache_roots=DEFAULT_CACHE_ROOTS):
    found, alternatives = {}, {}
    for root in map(Path, checkpoint_roots):
        if not root.is_dir(): continue
        for path in sorted(root.rglob("checkpoint_last.pt")):
            try: identity = _checkpoint_identity(path)
            except Exception: continue
            if identity is None: continue
            variant, task, config = identity
            if task not in tasks: continue
            key = (variant, task)
            alternatives.setdefault(key, []).append(str(path))
            if key in found: continue
            digest = sha256_file(path)
            source = _source_for(source_roots, TASKS[task][1], config.get("classes", []))
            cache = _cache_for(cache_roots, variant, task, digest)
            found[key] = InventoryItem(variant, task, str(path), str(source or ""), str(cache), digest)
    missing = []
    for variant in ("E", "G"):
        for task in tasks:
            if (variant, task) not in found:
                missing.append({"variant": variant, "task": task,
                                "searched_roots": [str(value) for value in checkpoint_roots],
                                "reason": "no compatible epoch-19 phase-moment checkpoint"})
    return {"items": list(found.values()), "missing": missing,
            "alternatives": {f"{k[0]}:{k[1]}": v for k, v in alternatives.items()}}


def _cache_complete(item):
    root = Path(item.cache_dir)
    return (root / "checkpoint_manifest.json").is_file() and all(
        (root / "features" / f"teacher_{split}.npz").is_file()
        for split in ("source_train", "source_val", "target_train", "target_val")
    )


def build_job_graph(items):
    jobs = []
    for item in items:
        dependency = frozenset()
        if not _cache_complete(item):
            key = f"extract:{item.variant}:{item.task}"
            jobs.append(Job(key, "extract", item.variant, item.task, item, weight="gpu"))
            dependency = frozenset((key,))
        jobs.append(Job(f"audit:{item.variant}:{item.task}", "audit", item.variant, item.task,
                        item, dependency, "cpu"))
    audits = frozenset(job.key for job in jobs if job.kind == "audit")
    if audits:
        jobs.append(Job("summary", "summary", dependencies=audits))
    return jobs


def job_command(job, args):
    if job.kind == "summary":
        return [args.python, "-u", "analysis/structure_pair_semantics_audit.py", "--mode", "summary",
                "--output-root", str(args.output_root)]
    command = [args.python, "-u", "analysis/structure_pair_semantics_audit.py", "--mode", job.kind,
               "--task", job.task, "--variant", job.variant, "--checkpoint", job.item.checkpoint,
               "--cache-dir", job.item.cache_dir, "--output-root", str(args.output_root),
               "--data-root", str(args.data_root), "--device", "cuda" if job.weight == "gpu" else "cpu",
               "--batch-size", str(args.batch_size), "--pixel-budget", str(args.pixel_budget)]
    if job.kind == "extract": command.extend(("--source-checkpoint", job.item.source_checkpoint))
    return command


def artifact_complete(job, args):
    if job.kind == "summary":
        summary = Path(args.output_root) / "FINAL_summary.md"
        if not summary.is_file(): return False
        inputs = [
            path for folder in Path(args.output_root).glob("[EG]_*")
            for name in REQUIRED_OUTPUTS for path in (folder / name,) if path.is_file()
        ]
        return bool(inputs) and summary.stat().st_mtime >= max(path.stat().st_mtime for path in inputs)
    if job.kind == "extract": return _cache_complete(job.item)
    root = Path(args.output_root) / f"{job.variant}_{job.task}"
    return all((root / name).is_file() and (root / name).stat().st_size > 0 for name in REQUIRED_OUTPUTS)


def build_parser():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--checkpoint-root", action="append", default=[])
    parser.add_argument("--source-root", action="append", default=[])
    parser.add_argument("--cache-root", action="append", default=[])
    parser.add_argument("--output-root", default="experiments/structure_pair_semantics_seed1")
    parser.add_argument("--log-root", default="logs/structure_pair_semantics_seed1")
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--python", default=sys.executable); parser.add_argument("--gpus", nargs=4, default=("0", "1", "2", "3"))
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--poll-seconds", type=float, default=3.); parser.add_argument("--retry-failed", action="store_true")
    return parser


def _roots(args):
    return (args.checkpoint_root or list(DEFAULT_CHECKPOINT_ROOTS),
            args.source_root or list(DEFAULT_SOURCE_ROOTS), args.cache_root or list(DEFAULT_CACHE_ROOTS))


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"); os.replace(temporary, path)


def _timestamp(): return datetime.now(timezone.utc).astimezone().isoformat()


def _prepare(args):
    checkpoint_roots, source_roots, cache_roots = _roots(args)
    inventory = resolve_inventory(checkpoint_roots, source_roots, TASKS, cache_roots)
    runnable = []
    for item in inventory["items"]:
        if not _cache_complete(item) and not item.source_checkpoint:
            inventory["missing"].append({"variant": item.variant, "task": item.task,
                                         "reason": "cache absent and compatible Source checkpoint missing",
                                         "cache_path": item.cache_dir})
        else:
            runnable.append(item)
    inventory["items"] = runnable
    output = Path(args.output_root); output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "inventory.json", {"items": [item.__dict__ for item in inventory["items"]],
                                              "missing": inventory["missing"], "alternatives": inventory["alternatives"],
                                              "target_test_used": False, "training_allowed": False})
    return inventory


def run_dry(args):
    inventory = _prepare(args); jobs = build_job_graph(inventory["items"])
    for job in jobs:
        print("DRY_RUN|" + json.dumps({"job": job.key, "dependencies": sorted(job.dependencies),
                                       "command": job_command(job, args)}))
    print("MISSING|" + json.dumps(inventory["missing"]))


def run_scheduler(args):
    inventory = _prepare(args); jobs = build_job_graph(inventory["items"])
    if not inventory["items"]: raise RuntimeError("no compatible checkpoints discovered")
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4: raise ValueError("four distinct GPUs required")
    state_path = Path(args.output_root) / "scheduler_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"completed": [], "failed": {}, "gpu_seconds": {}}
    completed = set(state["completed"]); failed = {} if args.retry_failed else dict(state["failed"])
    completed.update(job.key for job in jobs if artifact_complete(job, args))
    for gpu in args.gpus: state["gpu_seconds"].setdefault(gpu, 0.)
    running = {}
    while len(completed) < len(jobs):
        progressed = False; used = {value["gpu"] for value in running.values() if value["gpu"] is not None}
        free = [gpu for gpu in args.gpus if gpu not in used]
        for job in jobs:
            if job.key in completed or job.key in failed or job.key in running or not job.dependencies <= completed: continue
            gpu = None
            if job.weight == "gpu":
                if not free: continue
                gpu = min(free, key=lambda value: (state["gpu_seconds"][value], value)); free.remove(gpu)
            log = Path(args.log_root) / f"{job.key.replace(':', '_')}.log"; log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("a", encoding="utf-8"); env = os.environ.copy()
            if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(job_command(job, args), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
            running[job.key] = {"process": process, "job": job, "gpu": gpu, "stream": stream, "start": time.monotonic()}; progressed = True
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
        state.update(completed=sorted(completed), failed=failed, updated_at=_timestamp()); _atomic_json(state_path, state)
        if not running and not progressed: break
        if running and not finished: time.sleep(args.poll_seconds)
    if failed: raise RuntimeError("pair semantics stages failed: " + json.dumps(failed))
    if len(completed) != len(jobs): raise RuntimeError("pair semantics dependency graph blocked")
    print(f"SCHEDULER_COMPLETE|root={args.output_root}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "dry-run": run_dry(args)
    else: run_scheduler(args)


if __name__ == "__main__": main()
