#!/usr/bin/env python3
"""Four-Source scheduler for the frozen P0 Anchor calibration audit."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.anchor_matching_calibration_audit import (  # noqa: E402
    P0R_PROTOCOL_VERSION,
    REQUIRED_OUTPUTS,
    SOURCE_DOMAINS,
    settings_fingerprint,
    sha256_file,
)


@dataclass(frozen=True)
class SourcePlan:
    alias: str
    domain: str
    checkpoint: Path


def _checkpoint_override(args, alias):
    value = getattr(args, f"checkpoint_{alias.lower()}")
    if value:
        return Path(value)
    return Path(args.source_root) / f"source_{alias}_seed1/fold_0/model.pt"


def planned_sources(args):
    return [
        SourcePlan(alias, domain, _checkpoint_override(args, alias))
        for alias, domain in SOURCE_DOMAINS.items()
    ]


def _settings_fingerprint(args, plan):
    if plan.checkpoint.is_file():
        checkpoint_sha = sha256_file(plan.checkpoint)
    else:
        checkpoint_sha = "MISSING:" + str(plan.checkpoint.resolve())
    return settings_fingerprint(
        checkpoint_sha, args.data_root, args.seed, args.batch_size, args.pixel_budget,
    )


def artifact_complete(folder, checkpoint, fingerprint):
    folder = Path(folder); checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        return False
    if any(not (folder / name).is_file() or not (folder / name).stat().st_size for name in REQUIRED_OUTPUTS):
        return False
    try:
        manifest = json.loads((folder / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("status") == "complete"
        and manifest.get("checkpoint_sha256") == sha256_file(checkpoint)
        and manifest.get("settings_fingerprint") == fingerprint
        and manifest.get("target_data_used") is False
        and manifest.get("training_performed") is False
        and manifest.get("protocol_version") == P0R_PROTOCOL_VERSION
    )


def source_command(args, plan, fingerprint):
    return [
        args.python, "-u", "-m", "analysis.anchor_matching_calibration_audit",
        "--mode", "source", "--source-alias", plan.alias,
        "--checkpoint", str(plan.checkpoint), "--data-root", str(args.data_root),
        "--output-root", str(args.output_root), "--device", "cuda",
        "--seed", str(args.seed), "--batch-size", str(args.batch_size),
        "--pixel-budget", str(args.pixel_budget),
        "--settings-fingerprint", fingerprint,
    ]


def _run(command, gpu, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    with log.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND|" + " ".join(command) + "\n"); stream.flush()
        result = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"exit={result.returncode}; log={log}")


def _mark_in_progress(folder, checkpoint, fingerprint):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    path = folder / "checkpoint_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "status": "in_progress", "protocol_version": P0R_PROTOCOL_VERSION,
        "checkpoint_sha256": sha256_file(checkpoint),
        "settings_fingerprint": fingerprint,
        "target_data_used": False, "training_performed": False,
    }, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def run_scheduler(args):
    args.output_root = Path(args.output_root)
    plans = planned_sources(args)
    if args.mode == "dry-run":
        for index, plan in enumerate(plans):
            gpu = args.gpus[index % len(args.gpus)]
            fingerprint = _settings_fingerprint(args, plan)
            print(
                f"P0_PLAN|gpu={gpu}|source={plan.alias}|domain={plan.domain}|"
                f"checkpoint={plan.checkpoint}|output={args.output_root / plan.alias}|"
                f"protocol={P0R_PROTOCOL_VERSION}",
            )
            print("P0_COMMAND|" + " ".join(source_command(args, plan, fingerprint)))
            if not plan.checkpoint.is_file():
                print(f"P0_MISSING_CHECKPOINT_DEFERRED|source={plan.alias}|path={plan.checkpoint}")
        return

    if not Path(args.data_root).is_dir():
        raise FileNotFoundError(f"missing data root: {args.data_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    for plan in plans:
        pending.put(plan)
    failures = {}; lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                plan = pending.get_nowait()
            except queue.Empty:
                return
            try:
                if not plan.checkpoint.is_file():
                    raise FileNotFoundError(f"missing Source checkpoint: {plan.checkpoint}")
                fingerprint = _settings_fingerprint(args, plan)
                folder = args.output_root / plan.alias
                if args.mode == "resume" and artifact_complete(folder, plan.checkpoint, fingerprint):
                    print(f"P0_REUSE|gpu={gpu}|source={plan.alias}", flush=True)
                    continue
                if args.mode != "resume" and folder.exists() and any(folder.iterdir()):
                    raise RuntimeError(
                        f"partial or incompatible output; refusing overwrite: {folder}"
                    )
                _mark_in_progress(folder, plan.checkpoint, fingerprint)
                _run(
                    source_command(args, plan, fingerprint), gpu,
                    args.output_root / "logs" / f"{plan.alias}.log",
                )
                if not artifact_complete(folder, plan.checkpoint, fingerprint):
                    raise RuntimeError(f"Source audit did not produce a verified artifact: {folder}")
                print(f"P0_DONE|gpu={gpu}|source={plan.alias}", flush=True)
            except Exception as error:
                with lock:
                    failures[plan.alias] = str(error)
                print(f"P0_FAILED|gpu={gpu}|source={plan.alias}|error={error}", flush=True)
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = [
        args.python, "-u", "-m", "analysis.anchor_matching_calibration_audit",
        "--mode", "summary", "--output-root", str(args.output_root),
    ]
    if args.previous_root:
        summary.extend(["--previous-root", str(args.previous_root)])
    subprocess.run(summary, cwd=ROOT, check=True)
    (args.output_root / "scheduler_status.json").write_text(
        json.dumps({"failures": failures}, indent=2, sort_keys=True), encoding="utf-8",
    )
    if failures:
        raise RuntimeError("P0 Source failures: " + json.dumps(failures, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--source-root", default="outputs/structure_phase_moment_4tasks_seed1/source")
    for alias in SOURCE_DOMAINS:
        parser.add_argument(f"--checkpoint-{alias.lower()}", default="")
    parser.add_argument("--output-root", default="experiments/anchor_matching_calibration_p0r_seed1")
    parser.add_argument("--previous-root", default="experiments/anchor_matching_calibration_seed1")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    return parser


if __name__ == "__main__":
    run_scheduler(build_parser().parse_args())
