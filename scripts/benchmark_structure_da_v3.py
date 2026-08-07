"""Benchmark the current two-stage source-only training step without real data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.structure_da import SourceClassificationTrainer, TSStructureModel


def _model(device: torch.device) -> TSStructureModel:
    model = TSStructureModel(
        num_classes=3,
        input_dim=2,
        mlp1=(2, 4, 4),
        mlp2=(8, 4),
        time_reference=0.0,
        time_scale=365.0,
        trend_num_basis=4,
        structure_num_basis=4,
        canonical_grid_size=5,
        roughness_grid_size=64,
        trend_smoothing=1e-2,
        structure_smoothing=1e-3,
        n_head=1,
        d_k=2,
        d_model=8,
        ltae_mlp=(8, 4),
        dropout=0.0,
        classifier_hidden=(4,),
        max_initial_frequency=4.0,
    )
    return model.to(device=device).train()


def _sample(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    *,
    labels: bool,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        1701 + sequence_length
    )
    sample = {
        "pixels": torch.randn(
            batch_size,
            sequence_length,
            2,
            4,
            generator=generator,
        ).to(device),
        "valid_pixels": torch.ones(
            batch_size,
            sequence_length,
            4,
            dtype=torch.bool,
            device=device,
        ),
        "positions": torch.linspace(
            0, 300, sequence_length, device=device
        ).round().long().expand(batch_size, -1),
    }
    if labels:
        sample["label"] = torch.arange(batch_size, device=device) % 3
    return sample


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    index = round(0.95 * (len(ordered) - 1))
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[index],
    }


def _run(
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    torch.manual_seed(1701)
    model = _model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = SourceClassificationTrainer(
        model,
        optimizer,
        device=device,
        amp_enabled=False,
        amp_dtype="float16",
    )
    source = _sample(3, 5, device, labels=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    elapsed_ms: list[float] = []
    for iteration in range(warmup + iterations):
        _synchronize(device)
        started = time.perf_counter()
        trainer.train_step(source, warmup=True)
        _synchronize(device)
        if iteration >= warmup:
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        "benchmark_mode": "synthetic_source_only_ce_step",
        "device": str(device),
        "warmup": warmup,
        "iterations": iterations,
        "whole_train_iteration_ms": _summary(elapsed_ms),
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024.0**2
            if device.type == "cuda"
            else None
        ),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "state_dict_key_count": len(model.state_dict()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    print(
        json.dumps(
            _run(torch.device(args.device), args.warmup, args.iterations),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
