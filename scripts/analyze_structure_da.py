#!/usr/bin/env python3
"""Unified CLI for read-only Structure DA post-hoc diagnostics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("analysis_outputs/structure_da_full_3seeds_v1")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _nonnegative_finite_float(value: str) -> float:
    try:
        converted = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a finite number greater than or equal to zero"
        ) from error
    if not math.isfinite(converted) or converted < 0:
        raise argparse.ArgumentTypeError(
            "must be a finite number greater than or equal to zero"
        )
    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    logs = commands.add_parser("logs", help="parse archived task logs")
    logs.add_argument("--experiment-dir", type=Path, required=True)
    logs.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    raw = commands.add_parser("raw", help="analyze raw four-domain time series")
    raw.add_argument("--data-root", type=Path, required=True)
    raw.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    decomposition = commands.add_parser(
        "ndvi-decomposition", help="decompose existing class-domain mean NDVI curves"
    )
    decomposition.add_argument("--ndvi-csv", type=Path, required=True)
    decomposition.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    ts_diagnostic = commands.add_parser(
        "ndvi-ts-diagnostic",
        help="diagnose the T/S hierarchy on mean and sampled parcel NDVI curves",
    )
    ts_diagnostic.add_argument("--data-root", type=Path, required=True)
    ts_diagnostic.add_argument("--output-dir", type=Path, required=True)
    ts_diagnostic.add_argument("--samples-per-group", type=int, default=5)
    ts_diagnostic.add_argument("--sample-seed", type=int, default=1)
    ts_diagnostic.add_argument(
        "--crossing-eps", type=_nonnegative_finite_float, default=1e-4
    )
    ts_diagnostic.add_argument("--classes", nargs="+", default=None)

    domain_style = commands.add_parser(
        "ndvi-domain-style",
        help="run ORACLE target-label raw-NDVI domain-style diagnostics",
        description=(
            "ORACLE DIAGNOSTIC: uses true target class labels and is not "
            "deployable unsupervised domain adaptation."
        ),
    )
    domain_style.add_argument("--data-root", type=Path, required=True)
    domain_style.add_argument("--output-dir", type=Path, required=True)
    domain_style.add_argument("--source-domain", choices=("DK1", "FR2", "FR1", "AT1"), required=True)
    domain_style.add_argument("--target-domain", choices=("DK1", "FR2", "FR1", "AT1"), required=True)
    domain_style.add_argument("--samples-per-group", type=int, default=100)
    domain_style.add_argument("--sample-seed", type=int, default=1)
    domain_style.add_argument("--bootstrap-repeats", type=int, default=200)
    domain_style.add_argument("--canonical-grid-size", type=int, default=128)
    domain_style.add_argument("--min-class-samples", type=int, default=20)
    domain_style.add_argument("--min-common-support", type=float, default=0.65)
    domain_style.add_argument("--min-bootstrap-valid-rate", type=float, default=0.80)
    domain_style.add_argument("--peak-search-start", type=float, default=45.0)
    domain_style.add_argument("--peak-search-end", type=float, default=330.0)
    domain_style.add_argument("--min-peak-prominence-ratio", type=float, default=0.15)
    domain_style.add_argument("--max-shift-days", type=float, default=90.0)
    domain_style.add_argument("--shift-refine-radius-days", type=float, default=14.0)
    domain_style.add_argument("--max-interpolation-gap-days", type=float, default=60.0)
    domain_style.add_argument("--min-relative-phase-gain", type=float, default=0.02)
    domain_style.add_argument(
        "--style-lambdas", nargs="+", type=float,
        default=[0.0, 0.5, 1.0, 1.5],
    )
    domain_style.add_argument("--classes", nargs="+", default=None)

    return parser


def validate_domain_style_args(args):
    """Validate and normalize the ndvi-domain-style CLI namespace."""

    from analysis.structure_da.domain_style_diagnostics import DomainStyleConfig

    config = DomainStyleConfig(
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        samples_per_group=args.samples_per_group,
        sample_seed=args.sample_seed,
        bootstrap_repeats=args.bootstrap_repeats,
        canonical_grid_size=args.canonical_grid_size,
        min_class_samples=args.min_class_samples,
        min_common_support=args.min_common_support,
        min_bootstrap_valid_rate=args.min_bootstrap_valid_rate,
        peak_search_start=args.peak_search_start,
        peak_search_end=args.peak_search_end,
        min_peak_prominence_ratio=args.min_peak_prominence_ratio,
        max_shift_days=args.max_shift_days,
        shift_refine_radius_days=args.shift_refine_radius_days,
        max_interpolation_gap_days=args.max_interpolation_gap_days,
        min_relative_phase_gain=args.min_relative_phase_gain,
        style_lambdas=tuple(args.style_lambdas),
    )
    args.style_lambdas = list(config.style_lambdas)
    return config


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "logs":
        from analysis.structure_da.log_analysis import analyze_logs

        result = analyze_logs(args.experiment_dir, args.output_dir)
        manifest = result["manifest"]
        print(
            "LOG_ANALYSIS|"
            f"logs={len(manifest['task_logs_used'])}|completed={manifest['completed']}|"
            f"incomplete={manifest['incomplete']}|failed={manifest['failed']}|"
            f"missing={manifest['missing']}"
        )
    elif args.command == "raw":
        from analysis.structure_da.raw_timeseries import run_raw_analysis

        result = run_raw_analysis(args.data_root, args.output_dir)
        print(
            f"RAW_ANALYSIS|groups={len(result['aggregates'])}|"
            f"classes_union={len(result['classes_union'])}|"
            f"classes_intersection={len(result['classes_intersection'])}"
        )
    elif args.command == "ndvi-decomposition":
        from analysis.structure_da.decomposition_diagnostics import (
            run_ndvi_decomposition,
        )

        result = run_ndvi_decomposition(args.ndvi_csv, args.output_dir)
        print(
            "NDVI_DECOMPOSITION|"
            f"classes={len(result['classes'])}|"
            f"groups={len(result['reconstruction'])}|"
            f"max_reconstruction_error={result['max_reconstruction_error']:.3e}"
        )
    elif args.command == "ndvi-ts-diagnostic":
        from analysis.structure_da.decomposition_diagnostics import (
            run_ndvi_ts_diagnostic,
        )

        result = run_ndvi_ts_diagnostic(
            args.data_root,
            args.output_dir,
            samples_per_group=args.samples_per_group,
            sample_seed=args.sample_seed,
            classes=args.classes,
            crossing_eps=args.crossing_eps,
        )
        print(
            "NDVI_TS_DIAGNOSTIC|"
            f"classes={len(result['classes'])}|"
            f"samples={len(result['sampled_parcels'])}|"
            f"groups={len(result['reconstruction'])}"
        )
    elif args.command == "ndvi-domain-style":
        from analysis.structure_da.domain_style_diagnostics import (
            run_ndvi_domain_style_diagnostic,
        )

        config = validate_domain_style_args(args)
        result = run_ndvi_domain_style_diagnostic(
            args.data_root, args.output_dir, config, classes=args.classes,
        )
        print(
            "NDVI_DOMAIN_STYLE|"
            f"source={config.source_domain}|target={config.target_domain}|"
            f"classes={len(result['classes'])}|eligible={len(result['eligible_classes'])}|"
            f"samples={len(result['records'])}|bootstrap={config.bootstrap_repeats}"
        )


if __name__ == "__main__":
    main()
