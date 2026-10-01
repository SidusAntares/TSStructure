#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs/structure_proto_v2clean_4tasks_seed1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/anchor_drift_v2clean_seed1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -u analysis/anchor_drift_audit.py \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --output-root "$OUTPUT_ROOT"
