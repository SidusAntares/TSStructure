"""Audit real PSE reconstruction with the configured FreDN NUFFT operators."""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fredn.disentangler import FrequencyDisentangler
from models.fredn.nufft import (
    IrregularFourierAnalyzer,
    IrregularFourierSynthesizer,
    PytorchFinufftBackend,
    centered_modes,
    positions_to_periodic_points,
)
from models.stclassifier import PseLTae

DOMAIN_PATHS = {
    "AT1": "austria/33UVP/2017",
    "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
}

REQUIRED_SAMPLE_FIELDS = (
    "domain",
    "pse_state",
    "num_modes",
    "regularization",
    "sample_index",
    "sequence_length",
    "position_span",
    "cg_iterations",
    "cg_converged",
    "reconstruction_error",
    "additivity_error",
)


def summarize_rows(rows, metric):
    values = np.asarray([row[metric] for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {key: float("nan") for key in ("mean", "std", "median", "p75", "p90", "p95", "max")}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "p75": float(np.quantile(finite, 0.75)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def build_sweep(num_modes, regularizations):
    modes = sorted({int(value) for value in num_modes})
    regs = sorted({float(value) for value in regularizations})
    if any(value <= 0 or value % 2 == 0 for value in modes):
        raise ValueError("num_modes values must be positive odd integers")
    if any(value < 0 for value in regs):
        raise ValueError("regularization values must be non-negative")
    return [(mode, reg) for mode in modes for reg in regs]


def describe_modes(num_modes):
    modes = centered_modes(num_modes).tolist()
    return {
        "mode_index_min": int(modes[0]),
        "mode_index_max": int(modes[-1]),
        "complex_coefficient_count": len(modes),
    }


def summarize_frequency_mask(frequency_means):
    values = np.asarray(frequency_means, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or values.size % 2 == 0:
        raise ValueError("frequency mask means must be a non-empty odd vector")
    pair_difference = np.abs(values - values[::-1])
    center = values.size // 2
    magnitude_means = [values[center]]
    for offset in range(1, center + 1):
        magnitude_means.append((values[center - offset] + values[center + offset]) / 2.0)
    monotonic = all(
        magnitude_means[index] >= magnitude_means[index + 1] - 1e-12
        for index in range(len(magnitude_means) - 1)
    )
    return {
        "mask_pair_max_abs_diff": float(pair_difference.max()),
        "mask_low_to_high_monotonic": bool(monotonic),
    }


def load_spatial_encoder_checkpoint(spatial_encoder, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("source-trained PSE checkpoint unavailable")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    spatial_state = {}
    for name, value in state_dict.items():
        normalized = name.removeprefix("module.")
        prefix = "spatial_encoder."
        if normalized.startswith(prefix):
            spatial_state[normalized[len(prefix):]] = value
    if not spatial_state:
        raise ValueError("checkpoint does not contain spatial_encoder parameters")
    spatial_encoder.load_state_dict(spatial_state, strict=True)
    spatial_encoder.to(device)
    spatial_encoder.eval()
    spatial_encoder.requires_grad_(False)
    return spatial_encoder


@torch.no_grad()
def encode_pse(spatial_encoder, pixels, mask, extra):
    spatial_encoder.eval()
    return spatial_encoder(pixels, mask, extra)


def _relative_errors(reference, candidate):
    numerator = torch.linalg.vector_norm(candidate - reference, dim=(1, 2))
    denominator = torch.linalg.vector_norm(reference, dim=(1, 2)).clamp_min(1e-12)
    return numerator / denominator


@torch.no_grad()
def audit_fourier_batch(features, positions, analyzer, synthesizer, disentangler):
    coeffs, analysis = analyzer(features, positions)
    trend_coeffs, seasonal_coeffs, mask = disentangler(coeffs)

    reconstructed_complex = synthesizer.synthesize_complex(coeffs, positions)
    full_synthesis = dict(synthesizer.last_diagnostics)
    trend_complex = synthesizer.synthesize_complex(trend_coeffs, positions)
    trend_synthesis = dict(synthesizer.last_diagnostics)
    seasonal_complex = synthesizer.synthesize_complex(seasonal_coeffs, positions)
    seasonal_synthesis = dict(synthesizer.last_diagnostics)

    reconstructed = reconstructed_complex.real
    recombined = trend_complex.real + seasonal_complex.real
    reconstruction_errors = _relative_errors(features, reconstructed)
    additivity_errors = _relative_errors(reconstructed, recombined)
    iterations = analysis["per_sample_solver_iterations"]
    converged = analysis["per_sample_solver_converged"]
    residuals = analysis["per_sample_solver_residual"]

    rows = []
    for index in range(features.shape[0]):
        rows.append(
            {
                "sample_index": index,
                "sequence_length": int(features.shape[1]),
                "position_span": float((positions[index].max() - positions[index].min()).cpu()),
                "cg_iterations": int(iterations[index]),
                "cg_converged": bool(converged[index]),
                "cg_residual": float(residuals[index]),
                "reconstruction_error": float(reconstruction_errors[index].cpu()),
                "additivity_error": float(additivity_errors[index].cpu()),
            }
        )

    diagnostics = {
        "analysis_time": float(analysis["analysis_time"]),
        "synthesis_time": float(
            full_synthesis["synthesis_time"]
            + trend_synthesis["synthesis_time"]
            + seasonal_synthesis["synthesis_time"]
        ),
        "imaginary_residual": float(full_synthesis["imaginary_residual"]),
        "frequency_mask_mean": mask.detach().mean(dim=1).cpu().tolist(),
    }
    return rows, diagnostics


def write_sample_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted(set().union(*(row.keys() for row in rows)) - set(REQUIRED_SAMPLE_FIELDS))
    fieldnames = list(REQUIRED_SAMPLE_FIELDS) + extras
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run_correctness_smoke(
    backend,
    device,
    num_modes=9,
    period_days=365.0,
    regularization=1e-8,
    tolerance=1e-8,
    max_iterations=100,
):
    real_dtype = torch.float64
    complex_dtype = torch.complex128
    positions = torch.tensor(
        [[0.0, 7.0, 19.0, 37.0, 61.0, 89.0, 121.0, 157.0, 197.0, 239.0, 283.0, 329.0]],
        device=device,
        dtype=real_dtype,
    )
    points = positions_to_periodic_points(positions, period_days)[0]
    generator = torch.Generator(device=device).manual_seed(23)
    coeffs = torch.zeros(1, num_modes, 2, device=device, dtype=complex_dtype)
    modes = centered_modes(num_modes).tolist()
    zero_index = modes.index(0)
    coeffs[:, zero_index] = torch.randn(1, 2, generator=generator, device=device, dtype=real_dtype)
    for mode in range(1, num_modes // 2 + 1):
        value = torch.complex(
            torch.randn(1, 2, generator=generator, device=device, dtype=real_dtype),
            torch.randn(1, 2, generator=generator, device=device, dtype=real_dtype),
        )
        coeffs[:, modes.index(mode)] = value
        coeffs[:, modes.index(-mode)] = value.conj()

    probe_coeffs = torch.complex(
        torch.randn(1, num_modes, 2, generator=generator, device=device, dtype=real_dtype),
        torch.randn(1, num_modes, 2, generator=generator, device=device, dtype=real_dtype),
    )
    probe_values = torch.complex(
        torch.randn(1, positions.shape[1], 2, generator=generator, device=device, dtype=real_dtype),
        torch.randn(1, positions.shape[1], 2, generator=generator, device=device, dtype=real_dtype),
    )
    evaluated = backend.type2(points, probe_coeffs, isign=1)
    adjointed = backend.type1(points, probe_values, num_modes=num_modes, isign=-1)
    left = torch.sum(evaluated.conj() * probe_values)
    right = torch.sum(probe_coeffs.conj() * adjointed)
    adjoint_error = torch.abs(left - right) / torch.abs(left).clamp_min(1e-12)

    synthesizer = IrregularFourierSynthesizer(num_modes, period_days, backend=backend)
    synthetic_complex = synthesizer.synthesize_complex(coeffs, positions)
    features = synthetic_complex.real
    analyzer = IrregularFourierAnalyzer(
        num_modes,
        period_days,
        regularization,
        tolerance,
        max_iterations,
        backend=backend,
    )
    recovered, _ = analyzer(features, positions)
    reconstruction_complex = synthesizer.synthesize_complex(recovered, positions)
    disentangler = FrequencyDisentangler(num_modes, features.shape[-1]).to(device)
    trend, seasonal, _ = disentangler(recovered)
    trend_signal = synthesizer.synthesize_complex(trend, positions)
    seasonal_signal = synthesizer.synthesize_complex(seasonal, positions)

    coefficient_error = torch.linalg.vector_norm(recovered - coeffs) / torch.linalg.vector_norm(coeffs)
    reconstruction_error = torch.linalg.vector_norm(reconstruction_complex.real - features) / torch.linalg.vector_norm(features)
    additivity_error = torch.linalg.vector_norm(
        trend_signal.real + seasonal_signal.real - reconstruction_complex.real
    ) / torch.linalg.vector_norm(reconstruction_complex.real).clamp_min(1e-12)
    imaginary_residual = torch.linalg.vector_norm(reconstruction_complex.imag) / torch.linalg.vector_norm(
        reconstruction_complex.real
    ).clamp_min(1e-12)
    return {
        "adjoint_relative_error": float(adjoint_error.cpu()),
        "synthetic_coefficient_error": float(coefficient_error.cpu()),
        "synthetic_reconstruction_error": float(reconstruction_error.cpu()),
        "additivity_error": float(additivity_error.cpu()),
        "imaginary_residual": float(imaginary_residual.cpu()),
    }


def _parse_list(value, converter):
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--domains", default="AT1,DK1,FR1,FR2")
    parser.add_argument("--output-dir", default="outputs/fredn_audit")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-modes", default="9,17,25,33,49,65")
    parser.add_argument("--regularizations", default="1e-6,1e-5,1e-4,1e-3")
    parser.add_argument("--period-days", type=float, default=365.0)
    parser.add_argument("--solver-tol", type=float, default=1e-5)
    parser.add_argument("--solver-max-iter", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-pixels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--with-extra", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--official-smoke", action="store_true")
    parser.add_argument("--official-smoke-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.num_workers != 0:
        raise SystemExit("audit sweep requires --num-workers 0 for deterministic sampling")

    backend = PytorchFinufftBackend()
    if args.official_smoke or args.official_smoke_only:
        result = run_correctness_smoke(
            backend,
            device,
            period_days=args.period_days,
            tolerance=args.solver_tol,
            max_iterations=args.solver_max_iter,
        )
        print("OFFICIAL_NUFFT_SMOKE=" + json.dumps(result, sort_keys=True))
    if args.official_smoke_only:
        return
    if not args.data_root:
        raise SystemExit("--data-root is required unless --official-smoke-only is used")

    from torch.utils.data import DataLoader
    from torchvision.transforms import transforms

    from dataset import GroupByShapesBatchSampler, PixelSetData
    from transforms import Normalize, RandomSamplePixels, ToTensor
    from utils import label_utils

    _seed_everything(args.seed)
    pse_model = PseLTae(
        input_dim=args.input_dim,
        with_extra=args.with_extra,
        num_classes=1,
    )
    spatial_encoder = pse_model.spatial_encoder.to(device).eval()
    pse_state = "random_or_untrained_pse"
    if args.checkpoint:
        try:
            load_spatial_encoder_checkpoint(spatial_encoder, args.checkpoint, device)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc))
        pse_state = "source_trained_pse"
    else:
        spatial_encoder.requires_grad_(False)

    domains = [value.strip() for value in args.domains.split(",") if value.strip()]
    sweep = build_sweep(
        _parse_list(args.num_modes, int),
        _parse_list(args.regularizations, float),
    )
    for domain in domains:
        if domain not in DOMAIN_PATHS:
            raise SystemExit(f"unknown domain alias: {domain}")
        country = DOMAIN_PATHS[domain].split("/")[0]
        classes = label_utils.get_classes(country)
        transform = transforms.Compose(
            [RandomSamplePixels(args.num_pixels), Normalize(), ToTensor()]
        )
        dataset = PixelSetData(
            args.data_root,
            DOMAIN_PATHS[domain],
            classes,
            transform=transform,
            with_extra=args.with_extra,
        )
        domain_rows = []
        for num_modes, regularization in sweep:
            _seed_everything(args.seed)
            sampler = GroupByShapesBatchSampler(
                dataset,
                args.batch_size,
                by_time=True,
                by_pixel_dim=False,
            )
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
            analyzer = IrregularFourierAnalyzer(
                num_modes,
                args.period_days,
                regularization,
                args.solver_tol,
                args.solver_max_iter,
                backend=backend,
            ).to(device)
            synthesizer = IrregularFourierSynthesizer(
                num_modes,
                args.period_days,
                backend=backend,
            ).to(device)
            config_rows = []
            analysis_time = synthesis_time = 0.0
            sample_offset = 0
            mask_means = None
            for sample in loader:
                pixels = sample["pixels"].to(device)
                mask = sample["valid_pixels"].to(device)
                positions = sample["positions"].to(device)
                extra = sample.get("extra")
                if extra is not None:
                    extra = extra.to(device)
                features = encode_pse(spatial_encoder, pixels, mask, extra)
                disentangler = FrequencyDisentangler(num_modes, features.shape[-1]).to(device)
                batch_rows, diagnostics = audit_fourier_batch(
                    features,
                    positions,
                    analyzer,
                    synthesizer,
                    disentangler,
                )
                for row in batch_rows:
                    row.update(
                        {
                            "domain": domain,
                            "pse_state": pse_state,
                            "num_modes": num_modes,
                            "regularization": regularization,
                            "sample_index": sample_offset + row["sample_index"],
                        }
                    )
                config_rows.extend(batch_rows)
                sample_offset += len(batch_rows)
                analysis_time += diagnostics["analysis_time"]
                synthesis_time += diagnostics["synthesis_time"]
                mask_means = diagnostics["frequency_mask_mean"]
                if args.max_samples and sample_offset >= args.max_samples:
                    config_rows = config_rows[: args.max_samples]
                    break

            domain_rows.extend(config_rows)
            lengths = [row["sequence_length"] for row in config_rows]
            convergence = [row["cg_converged"] for row in config_rows]
            iterations = np.asarray([row["cg_iterations"] for row in config_rows])
            summary = {
                "domain": domain,
                "pse_state": pse_state,
                "num_modes_cli": num_modes,
                "regularization": regularization,
                "median_sequence_length": float(np.median(lengths)),
                "num_modes_over_median_length": float(num_modes / np.median(lengths)),
                "reconstruction": summarize_rows(config_rows, "reconstruction_error"),
                "additivity": summarize_rows(config_rows, "additivity_error"),
                "cg_convergence_rate": float(np.mean(convergence)),
                "cg_iterations_mean": float(np.mean(iterations)),
                "cg_iterations_p95": float(np.quantile(iterations, 0.95)),
                "solver_failures": int(len(convergence) - sum(convergence)),
                "nan_inf_count": int(
                    sum(
                        not math.isfinite(row["reconstruction_error"])
                        or not math.isfinite(row["additivity_error"])
                        for row in config_rows
                    )
                ),
                "analysis_runtime": analysis_time,
                "synthesis_runtime": synthesis_time,
                "frequency_mask_mean": mask_means,
            }
            summary.update(describe_modes(num_modes))
            summary.update(summarize_frequency_mask(mask_means))
            if device.type == "cuda":
                summary["gpu_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
            print("FREDN_AUDIT=" + json.dumps(summary, sort_keys=True))

        output_path = Path(args.output_dir) / f"reconstruction_{domain}.csv"
        write_sample_csv(output_path, domain_rows)


if __name__ == "__main__":
    main()
