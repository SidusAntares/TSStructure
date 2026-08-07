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
    legacy_checkpoint = run_directory / "fold_0" / "model.pt"
    stage1_checkpoint = run_directory / "fold_0" / "stage1_best.pt"
    stage2_checkpoint = run_directory / "fold_0" / "stage2_last_ema.pt"
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
        "stage1_checkpoint": stage1_checkpoint.is_file(),
        "stage2_checkpoint": stage2_checkpoint.is_file(),
        "stage1_train_step": bool(re.search(r"TRAIN_STEP\|", text)),
        "stage1_validation": "Validation result:" in text,
        "stage2_statistics": "STAGE2_INIT_COMPLETE|statistics_ready=true" in text,
        "stage2_train_step": bool(re.search(r"STAGE2_TRAIN\|", text)),
        "stage2_ema_step": bool(re.search(r"optimizer_step_success=(?:1(?:\.0+)?|0\.[1-9]\d*)", text)),
        "stage2_validation": bool(re.search(r"STAGE2_TARGET_VAL\|", text)),
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
        "legacy_stage1_model_checkpoint": str(legacy_checkpoint),
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_checkpoint": str(stage2_checkpoint),
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
