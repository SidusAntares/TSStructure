#!/usr/bin/env python3
"""Four-GPU Source→UDA scheduler for Direct Shape Token Query."""

from __future__ import annotations

import argparse
import csv
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
    "AT1_DK1": ("AT1", "austria/33UVP/2017", "DK1", "denmark/32VNH/2017"),
    "FR1_FR2": ("FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
    "FR2_DK1": ("FR2", "france/31TCJ/2017", "DK1", "denmark/32VNH/2017"),
    "DK1_AT1": ("DK1", "denmark/32VNH/2017", "AT1", "austria/33UVP/2017"),
}


@dataclass(frozen=True)
class Job:
    task: str
    source_alias: str
    source: str
    target_alias: str
    target: str
    representation: str = "direct_shape_token"

    @property
    def source_experiment(self):
        return f"SRC_DSTQ_{self.source_alias}_seed1"

    @property
    def uda_experiment(self):
        return f"DSTQ_{self.task}_seed1"


def planned_jobs():
    return [Job(task, *domains) for task, domains in TASKS.items()]


def _model_args():
    return [
        "--model", "psestructureprotoltae", "--structure-branch", "true",
        "--structure-exposer", "fourier", "--fourier_num_modes", "13",
        "--shape-dim", "128", "--shape-window-scales", "24",
        "--shape-window-stride", "8", "--shapelet-count", "16",
        "--shapelet-beta", "5", "--shape-resample-length", "16",
        "--shape-representation", "direct_shape_token",
        "--shape-injection", "current_query", "--structure-shift-mode", "none",
        "--shapelet-diversity-weight", "0", "--shapelet-shaping-weight", "0",
        "--shape-class-weight", "0.1", "--source-minority-mode", "base",
        "--class-org-source-weight", "0", "--class-org-target-weight", "0",
    ]


def source_command(args, job):
    return [
        args.python, "-u", "train.py", "-e", job.source_experiment,
        "--data_root", str(args.data_root), "--source", job.source,
        "--target", job.source, *_model_args(), "--shape-align-weight", "0",
        "--with_shift_aug", "false", "--seed", "1", "--num_folds", "1",
        "--epochs", "100", "--batch_size", "128", "--lr", "0.001",
        "--weight_decay", "0.0001", "--focal_loss_gamma", "1.0",
        "--seq_length", "30", "--num_pixels", "64", "--closed_set", "true",
        "--progress_bar", "off", "--output_dir", str(args.root / "outputs/source"),
        "--tensorboard_log_dir", str(args.root / "runs" / job.source_experiment),
    ]


def uda_command(args, job):
    weights = args.root / "outputs/source" / job.source_experiment
    return [
        args.python, "-u", "train.py", "-e", job.uda_experiment,
        "--data_root", str(args.data_root), "--source", job.source,
        "--target", job.target, *_model_args(), "--shape-align-weight", "0",
        "--with_shift_aug", "false", "--seed", "1", "--num_folds", "1",
        "--batch_size", "128", "--seq_length", "30", "--num_pixels", "64",
        "--closed_set", "true", "--progress_bar", "off",
        "--output_dir", str(args.root / "outputs/uda"),
        "--tensorboard_log_dir", str(args.root / "runs" / job.uda_experiment),
        "timematch", "--weights", str(weights), "--epochs", "20",
        "--steps_per_epoch", "500", "--lr", "0.0001",
        "--pseudo_threshold", "0.9", "--ema_decay", "0.9999",
        "--trade_off", "2.0", "--estimate_shift", "true",
        "--balance_source", "true", "--use_focal_loss", "true",
        "--shift_source", "true", "--sample_size", "100",
        "--max_temporal_shift", "60", "--domain_specific_bn", "true",
        "--shift_estimator", "AM", "--run_validation", "--output_student", "true",
        "--shape-da-mode", "batch_align", "--shape-alignment-view", "none",
        "--shape-equivariance-weight", "0", "--oracle-pseudo-labels", "false",
        "--adaptive-pseudo-selection", "false", "--detach-target-structure", "false",
        "--target-structure-detach-epochs", "5",
        "--structure-basis-mode", "adaptive", "--uda-anchor-update", "none",
        "--anchor-geometric-mode", "none",
    ]


def _cfg(packet, name, default=None):
    config = packet.get("config", {})
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def artifact_status(fold, kind, job):
    fold = Path(fold)
    if not fold.exists():
        return "missing"
    last = fold / "checkpoint_last.pt"
    best = fold / "model.pt"
    if not last.is_file() or (kind == "source" and not best.is_file()):
        return "partial"
    try:
        import torch
        packet = torch.load(last, map_location="cpu", weights_only=False)
        state = packet.get("state_dict", {})
        checks = [
            int(packet.get("epoch", -1)) == (99 if kind == "source" else 19),
            int(_cfg(packet, "seed", -1)) == 1,
            _cfg(packet, "shape_representation") == "direct_shape_token",
            _cfg(packet, "shape_injection") == "current_query",
            str(_cfg(packet, "source", "")) == job.source,
            not any("shapelet_dictionary.anchors" in key for key in state),
            any("shape_token_pool" in key for key in state),
        ]
        if kind == "uda":
            checks.extend((
                str(_cfg(packet, "target", "")) == job.target,
                int(_cfg(packet, "target_structure_detach_epochs", -1)) == 5,
                int(_cfg(packet, "steps_per_epoch", -1)) == 500,
            ))
        else:
            best_packet = torch.load(best, map_location="cpu", weights_only=False)
            best_state = best_packet.get("state_dict", {})
            checks.extend((
                _cfg(best_packet, "shape_representation") == "direct_shape_token",
                _cfg(best_packet, "shape_injection") == "current_query",
                not any("shapelet_dictionary.anchors" in key for key in best_state),
                any("shape_token_pool" in key for key in best_state),
            ))
        return "complete" if all(checks) else "incompatible"
    except Exception:
        return "partial"


def _run(command, log, gpu, dry_run=False):
    rendered = " ".join(map(str, command))
    print(f"COMMAND|gpu={gpu}|{rendered}", flush=True)
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"COMMAND|gpu={gpu}|{rendered}\n"); stream.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"exit={result.returncode}; log={log}")


def _worker(args, gpu, work, failures):
    while True:
        try:
            job = work.get_nowait()
        except queue.Empty:
            return
        try:
            source_fold = args.root / "outputs/source" / job.source_experiment / "fold_0"
            uda_fold = args.root / "outputs/uda" / job.uda_experiment / "fold_0"
            source_status = artifact_status(source_fold, "source", job)
            if source_status in ("partial", "incompatible"):
                raise RuntimeError(f"{source_status} source artifact; refusing overwrite: {source_fold}")
            if source_status != "complete":
                _run(source_command(args, job), args.root / "logs" / f"{job.source_experiment}.log", gpu, args.mode == "dry-run")
                if args.mode != "dry-run" and artifact_status(source_fold, "source", job) != "complete":
                    raise RuntimeError("source did not produce a compatible final artifact")
            uda_status = artifact_status(uda_fold, "uda", job)
            if uda_status in ("partial", "incompatible"):
                raise RuntimeError(f"{uda_status} UDA artifact; refusing overwrite: {uda_fold}")
            if uda_status != "complete":
                _run(uda_command(args, job), args.root / "logs" / f"{job.uda_experiment}.log", gpu, args.mode == "dry-run")
                if args.mode != "dry-run" and artifact_status(uda_fold, "uda", job) != "complete":
                    raise RuntimeError("UDA did not produce a compatible epoch-19 artifact")
            print(f"JOB_DONE|gpu={gpu}|task={job.task}", flush=True)
        except Exception as error:
            failures[job.task] = str(error)
            print(f"JOB_FAILED|gpu={gpu}|task={job.task}|error={error}", flush=True)
        finally:
            work.task_done()


def summarize_results(args):
    from scripts.run_structure_g_response_ab_seed1 import _read_final_metrics

    rows, per_class = [], []
    for task in TASKS:
        for variant, fold in (
            ("DSTQ", args.root / "outputs/uda" / f"DSTQ_{task}_seed1/fold_0"),
            ("G", args.g_root / f"G_{task}_seed1/fold_0"),
        ):
            result = _read_final_metrics(fold)
            row = {"task": task, "variant": variant, "checkpoint": str(fold / "checkpoint_last.pt")}
            if result is None:
                row.update(status="missing", fixed_class_macro_f1="")
            else:
                payload, support, precision, recall, f1, classes, metrics = result
                row.update(
                    status="complete", fixed_class_macro_f1=metrics["fixed_class_macro_f1"],
                    original_macro_f1=payload.get("macro_f1", ""),
                    class_count=len(classes), class_order="|".join(classes),
                )
                per_class.extend({
                    "task": task, "variant": variant, "class_index": index,
                    "class_name": name, "support": int(support[index]),
                    "precision": precision[index], "recall": recall[index], "f1": f1[index],
                } for index, name in enumerate(classes))
            rows.append(row)
    for task in TASKS:
        orders = {
            row["class_order"] for row in rows
            if row["task"] == task and row["status"] == "complete"
        }
        if len(orders) > 1:
            raise RuntimeError(f"class mapping mismatch for {task}: {sorted(orders)}")
    for path, data in (
        (args.root / "DSTQ_G_4tasks_seed1.csv", rows),
        (args.root / "per_class_comparison.csv", per_class),
    ):
        if data:
            keys = sorted({key for row in data for key in row})
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader(); writer.writerows(data)
    lookup = {(row["task"], row["variant"]): row for row in rows}
    lines = [
        "# Direct Shape Token Query Seed1", "",
        "Final-epoch Student fixed-class Test Macro-F1 from complete confusion matrices.", "",
        "| Task | G | DSTQ | DSTQ-G |", "|---|---:|---:|---:|",
    ]
    for task in TASKS:
        g = lookup[(task, "G")].get("fixed_class_macro_f1", "")
        dstq = lookup[(task, "DSTQ")].get("fixed_class_macro_f1", "")
        lines.append(
            f"| {task} | {float(g):.4f} | {float(dstq):.4f} | {float(dstq)-float(g):+.4f} |"
            if g != "" and dstq != "" else f"| {task} | MISSING | MISSING | MISSING |"
        )
    lines.extend([
        "", "DSTQ uses a 128D attention-pooled Shape Token and has no Anchor dictionary.",
        "Its only structure-specific supervision is Source Shape CE; diversity, shaping, alignment and phase equivariance are disabled.",
        f"G baseline root (read-only): `{args.g_root}`.",
    ])
    (args.root / "FINAL_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scheduler(args):
    args.root = Path(args.root)
    args.data_root = Path(args.data_root)
    args.g_root = Path(args.g_root)
    if args.mode != "dry-run" and not args.data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {args.data_root}")
    if args.mode == "dry-run":
        for index, job in enumerate(planned_jobs()):
            gpu = args.gpus[index % len(args.gpus)]
            _run(source_command(args, job), args.root / "logs" / f"{job.source_experiment}.log", gpu, True)
            _run(uda_command(args, job), args.root / "logs" / f"{job.uda_experiment}.log", gpu, True)
        return
    work = queue.Queue()
    for job in planned_jobs():
        work.put(job)
    failures = {}
    threads = [
        threading.Thread(target=_worker, args=(args, gpu, work, failures), daemon=False)
        for gpu in args.gpus
    ]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if failures:
        raise RuntimeError("DSTQ failures: " + json.dumps(failures, sort_keys=True))
    if args.mode != "dry-run":
        summarize_results(args)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dry-run", "run", "resume"))
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--root", default="experiments/structure_direct_shape_token_seed1")
    parser.add_argument("--g-root", default="experiments/structure_e_cross_seed1/outputs/uda")
    parser.add_argument("--gpus", nargs="+", default=("0", "1", "2", "3"))
    parser.add_argument("--python", default=sys.executable)
    return parser


if __name__ == "__main__":
    run_scheduler(build_parser().parse_args())
