"""Synthetic Structure DA step benchmark for cached and reference execution paths."""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.structure_da.full_model import StructureAwareDomainAdaptationModel


def _model(device: torch.device) -> StructureAwareDomainAdaptationModel:
    model = StructureAwareDomainAdaptationModel(
        num_classes=3,
        num_channels=3,
        channel_feature_dim=4,
        pixel_hidden_dim=6,
        structure_dim=8,
        time_scale=366.0,
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
        channel_options={
            "lag_centers": (-0.15, 0.0, 0.15),
            "lag_widths": (0.1, 0.1, 0.1),
            "velocity_bandwidth": 0.15,
            "edge_hidden_dim": 8,
            "min_effective_pairs": 1.0,
            "min_relation_mass": 0.0,
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
    )
    return model.to(device=device).train()


def _inputs(device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(1701)
    source = torch.randn(4, 9, 3, 8, generator=generator).to(device)
    target = torch.randn(4, 11, 3, 8, generator=generator).to(device)
    source_valid = torch.ones(4, 9, 8, dtype=torch.bool, device=device)
    target_valid = torch.ones(4, 11, 8, dtype=torch.bool, device=device)
    source_positions = torch.linspace(0, 300, 9, device=device).round().long()
    target_positions = torch.linspace(15, 350, 11, device=device).round().long()
    return (
        (source, source_valid, source_positions),
        (target, target_valid, target_positions),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loss_from_outputs(model, source_output, target_output, source_positions):
    geometry = model.forward_source_geometry(source_output, source_positions)
    alignment = model.align(source_output, target_output)
    return (
        source_output.representation.logits.square().mean()
        + target_output.representation.logits.square().mean()
        + geometry.total_loss
        + alignment.loss
    )


def _reference_forward(model, source, target):
    model.update_source_state(*source)
    source_output = model.forward_details(*source)
    target_output = model.forward_details(*target)
    source_trend = source_output.backbone.decomposition.trend
    geometry = model.temporal_operator.forward_geometry(
        source_trend,
        source_output.backbone.decomposition.dynamics,
        source[2],
        source_output.backbone.time_mask,
        torch.ones(source_trend.shape[0], dtype=torch.bool, device=source_trend.device),
    )
    alignment = model.align(source_output, target_output)
    return (
        source_output.representation.logits.square().mean()
        + target_output.representation.logits.square().mean()
        + geometry.total_loss
        + alignment.loss
    )


def _optimized_forward(model, source, target):
    source_backbone = model.forward_backbone(*source)
    target_backbone = model.forward_backbone(*target)
    model.update_source_state_from_backbone(
        model.detach_backbone_for_state(source_backbone), source[2]
    )
    source_output = model.forward_from_backbone(source_backbone, source[2])
    target_output = model.forward_from_backbone(target_backbone, target[2])
    return _loss_from_outputs(model, source_output, target_output, source[2])


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5))
    return ordered[index]


def _measure_steps(
    model,
    source,
    target,
    forward: Callable,
    device: torch.device,
    warmup: int,
    steps: int,
):
    forward_times: list[float] = []
    backward_times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index in range(warmup + steps):
        model.zero_grad(set_to_none=True)
        _synchronize(device)
        started = time.perf_counter()
        loss = forward(model, source, target)
        _synchronize(device)
        forward_elapsed = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        loss.backward()
        _synchronize(device)
        backward_elapsed = (time.perf_counter() - started) * 1000.0
        if index >= warmup:
            forward_times.append(forward_elapsed)
            backward_times.append(backward_elapsed)
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else None
    )
    return forward_times, backward_times, peak_memory


def _count_calls(model, source, target, forward: Callable) -> dict[str, int]:
    counts = {"backbone": 0, "registration": 0, "ltae_attention": 0, "channel_precompute": 0}
    hooks = [
        model.backbone.register_forward_hook(
            lambda *_: counts.__setitem__("backbone", counts["backbone"] + 1)
        ),
        model.temporal_operator.extractor.registration.register_forward_hook(
            lambda *_: counts.__setitem__(
                "registration", counts["registration"] + 1
            )
        ),
        model.representation.component_ltae.attention_heads.register_forward_hook(
            lambda *_: counts.__setitem__(
                "ltae_attention", counts["ltae_attention"] + 1
            )
        ),
    ]
    extractor = model.channel_operator.extractor
    original = extractor._precompute_temporal_geometry

    def counted(*args, **kwargs):
        counts["channel_precompute"] += 1
        return original(*args, **kwargs)

    extractor._precompute_temporal_geometry = counted
    try:
        model.zero_grad(set_to_none=True)
        forward(model, source, target)
    finally:
        extractor._precompute_temporal_geometry = original
        for hook in hooks:
            hook.remove()
    return counts


def _measure_callable(
    function: Callable[[], object], device: torch.device, warmup: int, steps: int
) -> tuple[float, float]:
    values = []
    with torch.no_grad():
        for index in range(warmup + steps):
            _synchronize(device)
            started = time.perf_counter()
            function()
            _synchronize(device)
            if index >= warmup:
                values.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(values), _percentile(values, 0.9)


def _module_timings(model, source, device, warmup, steps):
    backbone = model.forward_backbone(*source)
    decomposition = backbone.decomposition
    channel = model.channel_operator
    ltae = model.representation.component_ltae
    sequence_inputs = (
        decomposition.trend.flatten(start_dim=2),
        decomposition.dynamics.flatten(start_dim=2),
        decomposition.residual.flatten(start_dim=2),
    )
    positions = source[2]
    mask = backbone.time_mask
    channel_reference = lambda: (
        channel.extractor(
            decomposition.trend, positions, time_mask=mask
        ),
        channel.extractor(
            decomposition.dynamics, positions, time_mask=mask
        ),
    )
    channel_optimized = lambda: channel(
        decomposition.trend, decomposition.dynamics, positions, time_mask=mask
    )
    ltae_optimized = lambda: ltae(*sequence_inputs, positions, time_mask=mask)

    def ltae_reference():
        outputs = []
        resolved_positions = ltae._resolve_positions(
            positions, sequence_inputs[0].shape[0], sequence_inputs[0].shape[1], device
        )
        safe_positions = torch.where(mask, resolved_positions, torch.zeros_like(resolved_positions))
        for name, component in zip(ltae.component_names, sequence_inputs):
            safe = torch.where(mask.unsqueeze(-1), component, torch.zeros_like(component))
            encoded = ltae.stems[name](safe) + ltae.positional_enc(
                safe_positions + ltae.max_temporal_shift
            )
            encoded, _ = ltae.attention_heads(encoded, time_mask=mask)
            outputs.append(ltae.output_norms[name](ltae.dropout(ltae.shared_projection(encoded))))
        return outputs

    return {
        "channel_reference": _measure_callable(channel_reference, device, warmup, steps),
        "channel_optimized": _measure_callable(channel_optimized, device, warmup, steps),
        "ltae_reference": _measure_callable(ltae_reference, device, warmup, steps),
        "ltae_optimized": _measure_callable(ltae_optimized, device, warmup, steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1:
        parser.error("--warmup must be nonnegative and --steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    torch.manual_seed(1701)
    device = torch.device(args.device)
    source, target = _inputs(device)
    optimized_model = _model(device)
    reference_model = copy.deepcopy(optimized_model)
    reference = _measure_steps(
        reference_model, source, target, _reference_forward, device, args.warmup, args.steps
    )
    optimized = _measure_steps(
        optimized_model, source, target, _optimized_forward, device, args.warmup, args.steps
    )
    reference_counts = _count_calls(reference_model, source, target, _reference_forward)
    optimized_counts = _count_calls(optimized_model, source, target, _optimized_forward)
    module_times = _module_timings(
        optimized_model, source, device, args.warmup, args.steps
    )

    print(f"BENCHMARK|device={device}|warmup={args.warmup}|steps={args.steps}")
    for label, result in (("reference", reference), ("optimized", optimized)):
        forward_times, backward_times, peak_memory = result
        memory = "n/a" if peak_memory is None else f"{peak_memory:.2f}MiB"
        print(
            f"STEP|path={label}|forward_median_ms={statistics.median(forward_times):.3f}"
            f"|forward_p90_ms={_percentile(forward_times, 0.9):.3f}"
            f"|backward_median_ms={statistics.median(backward_times):.3f}"
            f"|backward_p90_ms={_percentile(backward_times, 0.9):.3f}"
            f"|peak_memory={memory}"
        )
    for label, counts in (("reference", reference_counts), ("optimized", optimized_counts)):
        print("CALLS|path=" + label + "|" + "|".join(f"{key}={value}" for key, value in counts.items()))
    for label, (median, p90) in module_times.items():
        print(f"MODULE|name={label}|median_ms={median:.3f}|p90_ms={p90:.3f}")


if __name__ == "__main__":
    main()
