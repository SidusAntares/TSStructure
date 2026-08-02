#!/usr/bin/env python3
"""Unified CLI for read-only Structure DA post-hoc diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("analysis_outputs/structure_da_full_3seeds_v1")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    ts_diagnostic.add_argument("--classes", nargs="+", default=None)

    return parser


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
        )
        print(
            "NDVI_TS_DIAGNOSTIC|"
            f"classes={len(result['classes'])}|"
            f"samples={len(result['sampled_parcels'])}|"
            f"groups={len(result['reconstruction'])}"
        )


if __name__ == "__main__":
    main()
