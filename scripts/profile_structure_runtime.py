#!/usr/bin/env python3
"""Component-level runtime profiler for the frozen structure model."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_structure_efficiency import (
    DEFAULT_CHECKPOINT_ROOT, _build_train_datasets, _fixed_indices, _loader,
)
from scripts.diagnose_structure_final_audit import _strict_load, expected_checkpoints

DEFAULT_OUTPUT = Path("outputs/structure_final_audit")


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _measure(device, operation):
    _sync(device); start = time.perf_counter(); result = operation(); _sync(device)
    return result, (time.perf_counter() - start) * 1000.


def _sample(sample, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in sample.items()}


def _peak(device):
    return int(torch.cuda.max_memory_allocated(device)) if str(device).startswith("cuda") else 0


def _reset_peak(device):
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def _source_step(model, sample, device):
    timings = {}; branch = model.structure_branch
    spatial, timings["PSE"] = _measure(
        device, lambda: model.spatial_encoder(sample["pixels"], sample["valid_pixels"], sample["extra"]),
    )
    analyzed, timings["Fourier analyzer"] = _measure(
        device, lambda: branch.exposer.analyzer(spatial, sample["positions"]),
    )
    coefficients, _ = analyzed
    exposed, timings["Fourier synthesizer"] = _measure(
        device, lambda: branch.exposer.synthesize_canonical(coefficients),
    )
    extracted, timings["window extraction + resample"] = _measure(
        device, lambda: branch.window_extractor(exposed),
    )
    groups, _ = extracted; components = []; resample_ms = 0.
    for windows in groups:
        current, elapsed = _measure(device, lambda windows=windows: branch.token_generator.components(windows))
        components.append(current); resample_ms += elapsed
    timings["window extraction + resample"] += resample_ms
    raw_values, timings["raw_encoder"] = _measure(
        device, lambda: [branch.token_generator.raw_encoder(value["normalized"]) for value in components],
    )
    diff_values, timings["diff_encoder"] = _measure(
        device, lambda: [branch.token_generator.diff_encoder(value["difference"]) for value in components],
    )
    def fuse():
        tokens = []
        for raw, diff, value in zip(raw_values, diff_values, components):
            mean = branch.token_generator.mean_encoder(value["mean"])
            std = branch.token_generator.std_encoder(value["std"])
            tokens.append(branch.token_generator.fusion(torch.cat((raw, diff, mean, std), -1)))
        return torch.cat(tokens, dim=1)
    tokens, timings["mean/std + fusion"] = _measure(device, fuse)
    response, timings["shapelet dictionary"] = _measure(device, lambda: branch.compute_rich_response(tokens))
    query, timings["response_to_query"] = _measure(device, lambda: branch.response_to_query(response))
    instance, timings["LTAE"] = _measure(
        device, lambda: model.temporal_encoder(spatial, sample["positions"], external_query=query),
    )
    logits, timings["decoder"] = _measure(device, lambda: model.decoder(instance))
    loss = torch.nn.functional.cross_entropy(logits, sample["label"])
    _, timings["backward"] = _measure(device, loss.backward)
    return timings


@torch.no_grad()
def _shift_step(model, sample, device, shifts):
    timings = {}; calls = 0
    spatial, timings["PSE"] = _measure(
        device, lambda: model.spatial_encoder(sample["pixels"], sample["valid_pixels"], sample["extra"]),
    )
    def structure():
        nonlocal calls
        calls += 1
        return model.prepare_structure(spatial, sample["positions"])
    prepared, timings["StructureBranch"] = _measure(device, structure)
    def classify():
        return torch.stack([
            model.classify_prepared(
                spatial, sample["positions"], temporal_shift=shift,
                prepared_structure=prepared,
            ) for shift in shifts
        ], dim=1)
    _, timings["all-shift LTAE/classifier"] = _measure(device, classify)
    return timings, calls


def _rows(task, stage, measured, peak, extra=None):
    extra = extra or {}; components = sorted(measured[0]) if measured else []
    rows = [
        {"task": task, "stage": stage, "component": component,
         "mean_ms": sum(value[component] for value in measured) / len(measured),
         "peak_gpu_memory_bytes": peak, **extra}
        for component in components
    ]
    rows.append({
        "task": task, "stage": stage, "component": "total",
        "mean_ms": sum(sum(value.values()) for value in measured) / len(measured),
        "peak_gpu_memory_bytes": peak, **extra,
    })
    return rows


def _write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    retained = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as stream:
            retained = [row for row in csv.DictReader(stream) if row.get("task") != rows[0].get("task")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(retained); writer.writerows(rows)


def run(args):
    source, target = args.task.split("_", 1)
    checkpoint_stage = "source" if args.mode == "source" else "uda"
    checkpoint = expected_checkpoints(args.checkpoint_root, source, target)[checkpoint_stage]
    if not checkpoint.is_file():
        print(f"MISSING|task={args.task}|stage={checkpoint_stage}|checkpoint={checkpoint}"); return
    model, config, _, _, _ = _strict_load(checkpoint, args.device)
    model.train(args.mode == "source")
    datasets = _build_train_datasets(config, source, target, args.data_root, args.seed)
    domain = source if args.mode == "source" else target
    classes = sorted(set(datasets[domain].get_labels().tolist()))
    indices = _fixed_indices(datasets[domain], classes, args.samples_per_class, args.seed)
    loader = _loader(datasets[domain], indices, args.batch_size, args.seed)
    iterator = itertools.cycle(loader); shifts = tuple(range(args.min_shift, args.max_shift + 1))
    for _ in range(args.warmup):
        sample = _sample(next(iterator), args.device); model.zero_grad(set_to_none=True)
        if args.mode == "source": _source_step(model, sample, args.device)
        else: _shift_step(model, sample, args.device, shifts)
    _reset_peak(args.device); measured, call_counts = [], []
    for _ in range(args.steps):
        sample = _sample(next(iterator), args.device); model.zero_grad(set_to_none=True)
        if args.mode == "source": measured.append(_source_step(model, sample, args.device))
        else:
            timings, calls = _shift_step(model, sample, args.device, shifts)
            measured.append(timings); call_counts.append(calls)
    peak = _peak(args.device)
    extra = ({"num_shifts": len(shifts), "structure_branch_call_count": max(call_counts)} if call_counts else {})
    rows = _rows(args.task, args.mode, measured, peak, extra)
    output = Path(args.output_dir)
    if args.mode == "source":
        path = output / "profile_source.csv"
        fields = ["task", "stage", "component", "mean_ms", "peak_gpu_memory_bytes"]
    else:
        path = output / "profile_shift.csv"
        fields = ["task", "stage", "component", "mean_ms", "num_shifts", "structure_branch_call_count", "peak_gpu_memory_bytes"]
    _write(path, rows, fields)
    for row in rows: print("PROFILE|" + "|".join(f"{key}={value}" for key, value in row.items()))
    print(f"PROFILE_SAVED|path={path}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source", "shift"), required=True)
    parser.add_argument("--task", default="AT1_DK1")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--data-root", type=Path, default=Path("/data/user/dataset/timematch_data"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20); parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--samples-per-class", type=int, default=64)
    parser.add_argument("--min-shift", type=int, default=-60); parser.add_argument("--max-shift", type=int, default=60)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
