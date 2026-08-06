#!/usr/bin/env python3
"""Validate a completed V3 smoke run without judging model quality."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def check_smoke(
    run_directory: Path, log_directory: Path | None = None
) -> dict[str, object]:
    run_directory = Path(run_directory)
    log_directory = run_directory if log_directory is None else Path(log_directory)
    log_path = log_directory / "smoke.log"
    checkpoint = run_directory / "fold_0" / "model.pt"
    metrics = sorted((run_directory / "fold_0").glob("test_metrics_*.json"))
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    numeric_losses = [
        float(value)
        for value in re.findall(
            r"(?:loss|cls)=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
            text,
        )
    ]
    combined = text
    failures: list[str] = []
    checks = {
        "checkpoint": checkpoint.is_file(),
        "metrics": bool(metrics),
        "train_step": bool(re.search(r"TRAIN_STEP\|", text)),
        "train_epoch": bool(re.search(r"TRAIN_EPOCH\|", text)),
        "validation": "Validation result:" in text,
        "finite_losses": bool(numeric_losses) and all(math.isfinite(value) for value in numeric_losses),
        "no_oom": re.search(r"out of memory|cuda oom", combined, re.IGNORECASE) is None,
        "no_nonfinite_marker": re.search(
            r"(?:loss|cls)=([+-]?(?:nan|[+-]?inf))",
            combined,
            re.IGNORECASE,
        ) is None,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    report: dict[str, object] = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "checkpoint": str(checkpoint),
        "metrics_files": [str(path) for path in metrics],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--log-directory", type=Path)
    args = parser.parse_args()
    report = check_smoke(args.run_directory, args.log_directory)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
