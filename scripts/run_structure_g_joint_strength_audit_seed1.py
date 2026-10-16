#!/usr/bin/env python3
"""Four-GPU scheduler for the frozen G joint-Strength audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "AT1_DK1": "AT1", "FR1_FR2": "FR1",
    "FR2_DK1": "FR2", "DK1_AT1": "DK1",
}


@dataclass(frozen=True)
class AuditTask:
    task: str
    source_alias: str
    checkpoint: Path
    source_checkpoint: Path


def planned_tasks(g_root="experiments/structure_e_cross_seed1/outputs/uda",
                  source_root="outputs/structure_phase_moment_4tasks_seed1/source"):
    return [AuditTask(
        task, source,
        Path(g_root) / f"G_{task}_seed1/fold_0/checkpoint_last.pt",
        Path(source_root) / f"source_{source}_seed1/fold_0/model.pt",
    ) for task, source in TASKS.items()]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_cache(path, checkpoint):
    path, checkpoint = Path(path), Path(checkpoint)
    manifest = path / "checkpoint_manifest.json"
    if not manifest.is_file() or not checkpoint.is_file(): return False
    try:
        packet = json.loads(manifest.read_text(encoding="utf-8"))
        if packet.get("checkpoint_sha256") != _sha256(checkpoint): return False
        if packet.get("target_test_used") or packet.get("target_train_labels_cached"): return False
        required = ("source_train", "source_val", "target_train", "target_val")
        for split in required:
            feature = path / f"features/student_{split}.npz"
            if not feature.is_file(): return False
        import numpy as np
        with np.load(path / "features/student_target_train.npz", allow_pickle=False) as values:
            if "label" in values.files: return False
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _complete(output, checkpoint):
    names = (
        "joint_discrimination.csv", "class_geometry.csv", "observation_stability.csv",
        "target_transfer.csv", "constraint_feasibility.csv", "checkpoint_manifest.json",
        "JOINT_READINESS.json", "TASK_report.md",
    )
    if not all((Path(output) / name).is_file() and (Path(output) / name).stat().st_size
               for name in names):
        return False
    if not Path(checkpoint).is_file(): return False
    try:
        packet = json.loads((Path(output) / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        return packet.get("checkpoint_sha256") == _sha256(checkpoint)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _valid_views(path, checkpoint):
    if not Path(path).is_file() or not Path(checkpoint).is_file(): return False
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as packet:
            return str(np.asarray(packet["checkpoint_sha256"]).item()) == _sha256(checkpoint)
    except (OSError, ValueError, KeyError):
        return False


def _base(args, mode, task, cache, views, output):
    return [
        args.python, "-u", "-m", "analysis.structure_g_joint_strength_audit",
        "--mode", mode, "--task", task.task,
        "--checkpoint", str(task.checkpoint), "--source-checkpoint", str(task.source_checkpoint),
        "--cache-dir", str(cache), "--views", str(views), "--output-root", str(output),
        "--data-root", args.data_root, "--device", "cuda",
        "--batch-size", str(args.batch_size), "--pixel-budget", str(args.pixel_budget),
        "--seed", "1",
    ]


def _run(command, gpu, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    with log.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND|" + " ".join(map(str, command)) + "\n"); stream.flush()
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"exit={result.returncode} log={log}")


def _choose_cache(args, task):
    candidates = (
        Path(args.reuse_cache_root) / f"G_{task.task}",
        Path(args.output_root) / "cache" / f"G_{task.task}",
    )
    return next((path for path in candidates if _valid_cache(path, task.checkpoint)), candidates[-1])


def run_scheduler(args):
    args.output_root = Path(args.output_root)
    tasks = planned_tasks(args.g_root, args.source_root)
    if args.mode == "dry-run":
        for index, task in enumerate(tasks):
            gpu = args.gpus[index % len(args.gpus)]
            cache = _choose_cache(args, task)
            views = args.output_root / "views" / f"G_{task.task}.npz"
            output = args.output_root / "tasks" / task.task
            print(f"PLAN|gpu={gpu}|task={task.task}|checkpoint={task.checkpoint}|cache={cache}")
            for mode in ("extract-cache", "views", "audit"):
                print(mode.upper().replace("-", "_") + "_COMMAND|" + " ".join(
                    _base(args, mode, task, cache, views, output)
                ))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue(); [pending.put(task) for task in tasks]
    failures, lock = {}, threading.Lock()

    def worker(gpu):
        while True:
            try: task = pending.get_nowait()
            except queue.Empty: return
            try:
                output = args.output_root / "tasks" / task.task
                if _complete(output, task.checkpoint):
                    print(f"REUSE_TASK|gpu={gpu}|task={task.task}", flush=True); continue
                missing = [str(path) for path in (task.checkpoint, task.source_checkpoint)
                           if not path.is_file()]
                if missing: raise FileNotFoundError(f"missing checkpoints: {missing}")
                cache = _choose_cache(args, task)
                views = args.output_root / "views" / f"G_{task.task}.npz"
                log = args.output_root / "logs" / f"G_{task.task}.log"
                if not _valid_cache(cache, task.checkpoint):
                    _run(_base(args, "extract-cache", task, cache, views, output), gpu, log)
                    if not _valid_cache(cache, task.checkpoint):
                        raise RuntimeError("feature cache incomplete after extraction")
                if not _valid_views(views, task.checkpoint):
                    _run(_base(args, "views", task, cache, views, output), gpu, log)
                    if not _valid_views(views, task.checkpoint):
                        raise RuntimeError("observation-view cache incomplete after extraction")
                _run(_base(args, "audit", task, cache, views, output), gpu, log)
                if not _complete(output, task.checkpoint):
                    raise RuntimeError("task outputs incomplete or checkpoint identity changed")
                print(f"TASK_DONE|gpu={gpu}|task={task.task}", flush=True)
            except Exception as error:
                with lock: failures[task.task] = str(error)
                print(f"TASK_FAILED|gpu={gpu}|task={task.task}|error={error}", flush=True)
            finally: pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    subprocess.run([
        args.python, "-u", "-m", "analysis.structure_g_joint_strength_audit",
        "--mode", "summary", "--output-root", str(args.output_root),
    ], cwd=ROOT, check=True)
    (args.output_root / "scheduler_status.json").write_text(
        json.dumps({"failures": failures}, indent=2, sort_keys=True), encoding="utf-8",
    )
    if failures: raise RuntimeError("joint Strength audit failures: " + json.dumps(failures))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--gpus", nargs="+", default=("0", "1", "2", "3"))
    parser.add_argument("--g-root", default="experiments/structure_e_cross_seed1/outputs/uda")
    parser.add_argument("--source-root", default="outputs/structure_phase_moment_4tasks_seed1/source")
    parser.add_argument("--reuse-cache-root", default="experiments/structure_pair_semantics_seed1/cache")
    parser.add_argument("--output-root", default="experiments/structure_g_joint_strength_audit_seed1")
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    return parser


if __name__ == "__main__":
    run_scheduler(build_parser().parse_args())
