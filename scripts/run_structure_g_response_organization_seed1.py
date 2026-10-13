#!/usr/bin/env python3
"""Four-GPU, resumable frozen Response96 organization audit scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "AT1_DK1": ("AT1", "DK1"),
    "FR1_FR2": ("FR1", "FR2"),
    "FR2_DK1": ("FR2", "DK1"),
    "DK1_AT1": ("DK1", "AT1"),
}


@dataclass(frozen=True)
class Scope:
    identity: str
    stage: str
    task: str
    source_alias: str
    checkpoint: str
    source_checkpoint: str
    command_hint: tuple[str, ...]


def planned_scopes(source_root="outputs/structure_phase_moment_4tasks_seed1/source",
                   g_root="experiments/structure_e_cross_seed1/outputs/uda"):
    scopes = []
    for task, (source, _) in TASKS.items():
        source_checkpoint = str(Path(source_root) / f"source_{source}_seed1/fold_0/model.pt")
        if not any(row.identity == f"SRC_{source}" for row in scopes):
            scopes.append(Scope(
                identity=f"SRC_{source}", stage="source", task=task, source_alias=source,
                checkpoint=source_checkpoint, source_checkpoint=source_checkpoint,
                command_hint=("analysis/structure_g_response_organization_audit.py", "--mode", "extract"),
            ))
        scopes.append(Scope(
            identity=f"G_{task}", stage="uda_final", task=task, source_alias=source,
            checkpoint=str(Path(g_root) / f"G_{task}_seed1/fold_0/checkpoint_last.pt"),
            source_checkpoint=source_checkpoint,
            command_hint=("analysis/structure_g_response_organization_audit.py", "--mode", "extract"),
        ))
    return scopes


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_cache(path, checkpoint):
    manifest = Path(path) / "checkpoint_manifest.json"
    if not manifest.is_file() or not Path(checkpoint).is_file():
        return False
    try:
        packet = json.loads(manifest.read_text(encoding="utf-8"))
        if packet.get("checkpoint_sha256") != _sha256(checkpoint):
            return False
        if packet.get("target_test_used") or packet.get("target_train_labels_cached"):
            return False
        required = ("source_train", "source_val")
        if packet.get("stage") != "source":
            required += ("target_val",)
        return all((Path(path) / f"features/student_{split}.npz").is_file() for split in required)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _complete_scope(output_root, scope):
    folder = Path(output_root) / "scopes" / scope.identity
    required = (
        "D1_class_anchor_distribution.csv", "D2_class_discriminability.csv",
        "D3_cross_anchor_relations.csv", "D3_relation_increment.csv",
        "D4_source_target_transfer.csv", "D4_class_failure_modes.csv",
        "scope_manifest.json",
    )
    return all((folder / name).is_file() and (folder / name).stat().st_size for name in required)


def _command(args, scope, mode, cache):
    command = [
        args.python, "-u", "analysis/structure_g_response_organization_audit.py",
        "--mode", mode, "--stage", scope.stage, "--scope", scope.identity,
        "--task", scope.task, "--checkpoint", scope.checkpoint,
        "--cache-dir", str(cache), "--output-root", str(args.output_root),
        "--data-root", args.data_root, "--device", "cuda",
        "--batch-size", str(args.batch_size), "--pixel-budget", str(args.pixel_budget),
    ]
    if scope.stage == "uda_final":
        command.extend(("--source-checkpoint", scope.source_checkpoint))
    return command


def _run(command, gpu, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    with log.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND|" + " ".join(map(str, command)) + "\n"); stream.flush()
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"exit={result.returncode} log={log}")


def _reuse_cache(args, scope):
    candidates = []
    if scope.stage == "uda_final":
        candidates.append(Path(args.reuse_cache_root) / scope.identity)
    candidates.append(Path(args.output_root) / "cache" / scope.identity)
    for candidate in candidates:
        if _valid_cache(candidate, scope.checkpoint):
            return candidate
    return candidates[-1]


def run_scheduler(args):
    args.output_root = Path(args.output_root)
    scopes = planned_scopes(args.source_root, args.g_root)
    missing = []
    if args.mode == "dry-run":
        for index, scope in enumerate(scopes):
            cache = _reuse_cache(args, scope)
            print(f"PLAN|gpu={args.gpus[index % len(args.gpus)]}|scope={scope.identity}|stage={scope.stage}|cache={cache}")
            print("EXTRACT_COMMAND|" + " ".join(_command(args, scope, "extract", cache)))
            print("AUDIT_COMMAND|" + " ".join(_command(args, scope, "audit", cache)))
            if not Path(scope.checkpoint).is_file():
                print(f"MISSING_CHECKPOINT|scope={scope.identity}|path={Path(scope.checkpoint).resolve()}")
        return
    args.output_root.mkdir(parents=True, exist_ok=True)

    pending = queue.Queue()
    for scope in scopes:
        pending.put(scope)
    failures, lock = {}, threading.Lock()

    def worker(gpu):
        while True:
            try:
                scope = pending.get_nowait()
            except queue.Empty:
                return
            try:
                if args.mode == "resume" and _complete_scope(args.output_root, scope):
                    print(f"REUSE_SCOPE|gpu={gpu}|scope={scope.identity}", flush=True)
                    continue
                absent = [path for path in (scope.checkpoint, scope.source_checkpoint)
                          if not Path(path).is_file()]
                if absent:
                    message = f"MISSING|scope={scope.identity}|paths={absent}"
                    with lock: missing.append(message)
                    print(message, flush=True); continue
                cache = _reuse_cache(args, scope)
                log = args.output_root / "logs" / f"{scope.identity}.log"
                if not _valid_cache(cache, scope.checkpoint):
                    _run(_command(args, scope, "extract", cache), gpu, log)
                _run(_command(args, scope, "audit", cache), gpu, log)
                print(f"SCOPE_DONE|gpu={gpu}|scope={scope.identity}", flush=True)
            except Exception as error:
                with lock: failures[scope.identity] = str(error)
                print(f"SCOPE_FAILED|gpu={gpu}|scope={scope.identity}|error={error}", flush=True)
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in args.gpus]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    (args.output_root / "missing_dependencies.json").write_text(
        json.dumps({"missing": missing, "failures": failures}, indent=2), encoding="utf-8",
    )
    summary = [args.python, "-u", "analysis/structure_g_response_organization_audit.py",
               "--mode", "summary", "--output-root", str(args.output_root)]
    result = subprocess.run(summary, cwd=ROOT)
    if result.returncode:
        raise RuntimeError("summary generation failed")
    if failures:
        raise RuntimeError("audit failures: " + json.dumps(failures, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--output-root", default="experiments/structure_g_response_organization_seed1")
    parser.add_argument("--source-root", default="outputs/structure_phase_moment_4tasks_seed1/source")
    parser.add_argument("--g-root", default="experiments/structure_e_cross_seed1/outputs/uda")
    parser.add_argument("--reuse-cache-root", default="experiments/structure_pair_semantics_seed1/cache")
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    return parser


if __name__ == "__main__":
    run_scheduler(build_parser().parse_args())
