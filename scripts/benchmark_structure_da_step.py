"""Benchmark one synthetic Structure DA training step by optimization phase."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.structure_da.full_model import StructureAwareDomainAdaptationModel


def _bool_argument(value: str) -> bool:
    normalized = value.casefold()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model(device: torch.device) -> StructureAwareDomainAdaptationModel:
    return StructureAwareDomainAdaptationModel(
        num_classes=3,
        input_dim=3,
        mlp1=(3, 12, 8),
        mlp2=(16, 8),
        structure_dim=8,
        temporal_options={
            "num_basis": 6,
            "canonical_grid_size": 7,
            "roughness_grid_size": 64,
            "min_mean_support": 0.0,
            "min_dynamic_energy": 0.0,
            "min_template_mean_support": 0.0,
            "warp_hidden_dim": 12,
            "warp_kernel_size": 3,
            "num_shape_basis": 4,
            "num_phase_basis": 3,
            "attribute_projection_dim": 4,
            "coordinate_hidden_dim": 12,
            "dropout": 0.0,
        },
        representation_options={
            "n_head": 2,
            "d_k": 4,
            "d_model": 16,
            "ltae_mlp": (16, 8),
            "dropout": 0.0,
            "max_position": 366,
            "max_temporal_shift": 0,
            "classifier_hidden": (8,),
            "quality_domain_hidden_dim": 8,
        },
        alignment_hidden_dim=8,
        grl_max_iters=100,
    ).to(device=device).train()


def _inputs(device: torch.device, batch_size: int):
    generator = torch.Generator(device="cpu").manual_seed(1701)
    source = torch.randn(
        batch_size, 9, 3, 8, generator=generator
    ).to(device)
    target = torch.randn(
        batch_size, 11, 3, 8, generator=generator
    ).to(device)
    return (
        (
            source,
            torch.ones(batch_size, 9, 8, dtype=torch.bool, device=device),
            torch.linspace(0, 300, 9, device=device).round().long(),
        ),
        (
            target,
            torch.ones(batch_size, 11, 8, dtype=torch.bool, device=device),
            torch.linspace(15, 350, 11, device=device).round().long(),
        ),
    )


def _measure(
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
    amp_dtype: str,
    warmup: int,
    steps: int,
) -> dict[str, object]:
    model = _model(device)
    source, target = _inputs(device, batch_size)
    task_optimizer = torch.optim.Adam(model.task_parameters(), lr=1e-3)
    geometry_optimizer = torch.optim.Adam(model.geometry_parameters(), lr=1e-3)
    amp_enabled = amp and device.type == "cuda"
    dtype = getattr(torch, amp_dtype)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == "float16"
    )
    timings = {
        name: []
        for name in (
            "forward", "geometry_backward", "task_backward",
            "optimizer_step", "iteration",
        )
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for index in range(warmup + steps):
        iteration_started = time.perf_counter()
        task_optimizer.zero_grad(set_to_none=True)
        geometry_optimizer.zero_grad(set_to_none=True)
        _synchronize(device)
        started = time.perf_counter()
        with torch.autocast(
            device_type=device.type, dtype=dtype, enabled=amp_enabled
        ):
            source_backbone = model.forward_backbone(*source)
            target_backbone = model.forward_backbone(*target)
            model.update_source_state_from_backbone(
                model.detach_backbone_for_state(source_backbone), source[2]
            )
            source_output = model.forward_from_backbone(source_backbone, source[2])
            target_output = model.forward_from_backbone(target_backbone, target[2])
            task_loss = (
                source_output.representation.logits.square().mean()
                + target_output.representation.logits.square().mean()
                + model.align(source_output, target_output).loss
            )
        with torch.autocast(device_type=device.type, enabled=False):
            geometry_loss = model.forward_source_geometry(
                source_output, source[2]
            ).total_loss.float()
        _synchronize(device)
        forward_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        geometry_loss.backward()
        _synchronize(device)
        geometry_backward_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        if scaler.is_enabled():
            scaler.scale(task_loss).backward()
        else:
            task_loss.backward()
        _synchronize(device)
        task_backward_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        geometry_optimizer.step()
        if scaler.is_enabled():
            scaler.step(task_optimizer)
            scaler.update()
        else:
            task_optimizer.step()
        _synchronize(device)
        optimizer_elapsed = time.perf_counter() - started
        iteration_elapsed = time.perf_counter() - iteration_started

        if index >= warmup:
            for name, elapsed in (
                ("forward", forward_elapsed),
                ("geometry_backward", geometry_backward_elapsed),
                ("task_backward", task_backward_elapsed),
                ("optimizer_step", optimizer_elapsed),
                ("iteration", iteration_elapsed),
            ):
                timings[name].append(elapsed * 1000.0)

    return {
        "timings": timings,
        "peak_allocated": (
            torch.cuda.max_memory_allocated(device) / 1024.0**2
            if device.type == "cuda" else None
        ),
        "peak_reserved": (
            torch.cuda.max_memory_reserved(device) / 1024.0**2
            if device.type == "cuda" else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--amp", type=_bool_argument, default=False)
    parser.add_argument(
        "--amp_dtype", choices=("float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.steps < 1:
        parser.error("batch size/steps must be positive and warmup nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    torch.manual_seed(1701)
    device = torch.device(args.device)
    modes = (
        (False, True)
        if device.type == "cuda" and args.amp
        else ((args.amp,) if device.type == "cuda" else (False,))
    )
    for amp in modes:
        try:
            result = _measure(
                device=device,
                batch_size=args.batch_size,
                amp=amp,
                amp_dtype=args.amp_dtype,
                warmup=args.warmup,
                steps=args.steps,
            )
        except torch.OutOfMemoryError:
            print(
                f"BENCHMARK|device={device}|batch_size={args.batch_size}"
                f"|amp={str(amp).lower()}|amp_dtype={args.amp_dtype}|oom=true"
            )
            print(
                "MEMORY|peak_allocated_mib="
                f"{torch.cuda.max_memory_allocated(device) / 1024.0**2:.2f}"
                "|peak_reserved_mib="
                f"{torch.cuda.max_memory_reserved(device) / 1024.0**2:.2f}"
            )
            raise
        print(
            f"BENCHMARK|device={device}|batch_size={args.batch_size}"
            f"|amp={str(amp).lower()}|amp_dtype={args.amp_dtype}"
            f"|warmup={args.warmup}|steps={args.steps}|oom=false"
        )
        timings = result["timings"]
        for name, values in timings.items():
            print(
                f"TIMING|phase={name}|median_ms={statistics.median(values):.3f}"
                f"|p90_ms={_percentile(values, 0.9):.3f}"
            )
        allocated = result["peak_allocated"]
        reserved = result["peak_reserved"]
        print(
            "MEMORY|peak_allocated_mib="
            f"{'n/a' if allocated is None else f'{allocated:.2f}'}"
            "|peak_reserved_mib="
            f"{'n/a' if reserved is None else f'{reserved:.2f}'}"
        )


if __name__ == "__main__":
    main()
