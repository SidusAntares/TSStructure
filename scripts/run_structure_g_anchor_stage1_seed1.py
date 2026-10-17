#!/usr/bin/env python3
"""Four-GPU scheduler for the frozen G Anchor Stage-1 audit."""

from __future__ import annotations

import argparse
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

from analysis.anchor_matching_calibration_audit import SOURCE_DOMAINS, sha256_file  # noqa: E402
from analysis.structure_g_anchor_stage1_audit import (  # noqa: E402
    REQUIRED_OUTPUTS,
    STAGE1_PROTOCOL_VERSION,
    settings_fingerprint,
)


TASK_BY_SOURCE = {
    "AT1": "AT1_DK1", "FR1": "FR1_FR2",
    "FR2": "FR2_DK1", "DK1": "DK1_AT1",
}


@dataclass(frozen=True)
class Stage1Job:
    source_alias: str
    source_domain: str
    task: str
    source_checkpoint: Path
    uda_checkpoint: Path


def _override(args, prefix, key):
    value = getattr(args, f"{prefix}_{key.lower()}")
    return Path(value) if value else None


def planned_jobs(args):
    jobs = []
    for alias, domain in SOURCE_DOMAINS.items():
        task = TASK_BY_SOURCE[alias]
        source = _override(args, "checkpoint", alias) or (
            Path(args.source_root) / f"source_{alias}_seed1/fold_0/model.pt"
        )
        uda = _override(args, "uda", task) or (
            Path(args.uda_root) / f"G_{task}_seed1/fold_0/checkpoint_last.pt"
        )
        jobs.append(Stage1Job(alias, domain, task, source, uda))
    return jobs


def _sha_or_missing(path):
    return sha256_file(path) if Path(path).is_file() else "MISSING:" + str(Path(path).resolve())


def _fingerprint(args, job):
    return settings_fingerprint(
        _sha_or_missing(job.source_checkpoint), _sha_or_missing(job.uda_checkpoint),
        args.data_root, args.seed, args.batch_size, args.pixel_budget,
    )


def artifact_complete(folder, source_sha, uda_sha, fingerprint):
    folder = Path(folder)
    if any(not (folder / name).is_file() or not (folder / name).stat().st_size
           for name in REQUIRED_OUTPUTS):
        return False
    try:
        manifest = json.loads((folder / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("status") == "complete"
        and manifest.get("protocol_version") == STAGE1_PROTOCOL_VERSION
        and manifest.get("source_checkpoint_sha256") == source_sha
        and manifest.get("uda_checkpoint_sha256") == uda_sha
        and manifest.get("settings_fingerprint") == fingerprint
        and manifest.get("target_test_used") is False
        and manifest.get("training_performed") is False
    )


def _command(args, job, fingerprint):
    return [
        args.python, "-u", "-m", "analysis.structure_g_anchor_stage1_audit",
        "--mode", "source", "--source-alias", job.source_alias, "--task", job.task,
        "--source-checkpoint", str(job.source_checkpoint),
        "--uda-checkpoint", str(job.uda_checkpoint),
        "--data-root", str(args.data_root), "--p0r-root", str(args.p0r_root),
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
            command, cwd=ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"exit={result.returncode}; log={log}")


def run_scheduler(args):
    args.output_root = Path(args.output_root)
    jobs = planned_jobs(args)
    if args.mode == "dry-run":
        for index, job in enumerate(jobs):
            gpu = args.gpus[index % len(args.gpus)]
            fingerprint = _fingerprint(args, job)
            print(
                f"ANCHOR_STAGE1_PLAN|gpu={gpu}|source={job.source_alias}|task={job.task}|"
                f"source_checkpoint={job.source_checkpoint}|uda_checkpoint={job.uda_checkpoint}|"
                f"output={args.output_root / job.source_alias}|protocol={STAGE1_PROTOCOL_VERSION}"
            )
            print("ANCHOR_STAGE1_COMMAND|" + " ".join(_command(args, job, fingerprint)))
        return
    if not Path(args.data_root).is_dir():
        raise FileNotFoundError(f"missing data root: {args.data_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    for job in jobs: pending.put(job)
    failures = {}; lock = threading.Lock()

    def worker(gpu):
        while True:
            try: job = pending.get_nowait()
            except queue.Empty: return
            try:
                for role, checkpoint in (("Source Best", job.source_checkpoint), ("G UDA Final", job.uda_checkpoint)):
                    if not checkpoint.is_file(): raise FileNotFoundError(f"missing {role} checkpoint: {checkpoint}")
                source_sha = sha256_file(job.source_checkpoint)
                uda_sha = sha256_file(job.uda_checkpoint)
                fingerprint = _fingerprint(args, job)
                folder = args.output_root / job.source_alias
                if args.mode == "resume" and artifact_complete(folder, source_sha, uda_sha, fingerprint):
                    print(f"ANCHOR_STAGE1_REUSE|gpu={gpu}|source={job.source_alias}|task={job.task}", flush=True)
                    continue
                if folder.exists() and any(folder.iterdir()):
                    if args.mode != "resume":
                        raise RuntimeError(f"existing output; refusing overwrite: {folder}")
                    # Resume may reuse compatible compact caches, but never a completed incompatible result.
                    manifest = folder / "checkpoint_manifest.json"
                    if manifest.is_file():
                        try:
                            old = json.loads(manifest.read_text(encoding="utf-8"))
                        except (OSError, ValueError, json.JSONDecodeError): old = {}
                        if old.get("status") == "complete":
                            raise RuntimeError(f"complete but incompatible result; refusing overwrite: {folder}")
                _run(
                    _command(args, job, fingerprint), gpu,
                    args.output_root / "logs" / f"{job.source_alias}_{job.task}.log",
                )
                if not artifact_complete(folder, source_sha, uda_sha, fingerprint):
                    raise RuntimeError(f"audit did not produce a verified artifact: {folder}")
                print(f"ANCHOR_STAGE1_DONE|gpu={gpu}|source={job.source_alias}|task={job.task}", flush=True)
            except Exception as error:
                with lock: failures[f"{job.source_alias}:{job.task}"] = str(error)
                print(f"ANCHOR_STAGE1_FAILED|gpu={gpu}|source={job.source_alias}|task={job.task}|error={error}", flush=True)
            finally: pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in args.gpus]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    subprocess.run([
        args.python, "-u", "-m", "analysis.structure_g_anchor_stage1_audit",
        "--mode", "summary", "--output-root", str(args.output_root),
    ], cwd=ROOT, check=True)
    (args.output_root / "scheduler_status.json").write_text(
        json.dumps({"failures": failures}, indent=2, sort_keys=True), encoding="utf-8",
    )
    if failures:
        raise RuntimeError("Anchor Stage-1 failures: " + json.dumps(failures, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--source-root", default="outputs/structure_phase_moment_4tasks_seed1/source")
    parser.add_argument("--uda-root", default="experiments/structure_e_cross_seed1/outputs/uda")
    parser.add_argument("--p0r-root", default="experiments/anchor_matching_calibration_p0r_seed1")
    parser.add_argument("--output-root", default="experiments/structure_g_anchor_stage1_seed1")
    for alias in SOURCE_DOMAINS:
        parser.add_argument(f"--checkpoint-{alias.lower()}", default="")
    for task in TASK_BY_SOURCE.values():
        parser.add_argument(f"--uda-{task.lower()}", default="")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    return parser


if __name__ == "__main__":
    run_scheduler(build_parser().parse_args())
