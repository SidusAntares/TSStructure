#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
VIS_GPU="${VIS_GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_G="${SKIP_G:-0}"
SKIP_VIS="${SKIP_VIS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
ROOT="${ROOT:-experiments/structure_e_cross_seed1}"
LOG_ROOT="$ROOT/logs"

mkdir -p "$LOG_ROOT" "$ROOT/outputs/VIS"

if [[ "$SKIP_G" != "1" ]]; then
  echo "G_VALIDATION_ROUND|tasks=AT1_DK1,DK1_AT1|source_retraining=false"
  GPU0="$GPU0" GPU1="$GPU1" DRY_RUN="$DRY_RUN" RUN_ROUND=G_EXTEND \
    PYTHON_BIN="$PYTHON_BIN" DATA_ROOT="$DATA_ROOT" ROOT="$ROOT" \
    bash scripts/run_structure_e_cross_seed1.sh
fi

if [[ "$SKIP_VIS" != "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN|stage=VIS|command=CUDA_VISIBLE_DEVICES=%q %q -u analysis/structure_e_g_validation.py --data-root %q --cross-root %q\n' \
      "$VIS_GPU" "$PYTHON_BIN" "$DATA_ROOT" "$ROOT"
  else
    CUDA_VISIBLE_DEVICES="$VIS_GPU" "$PYTHON_BIN" -u \
      analysis/structure_e_g_validation.py \
      --data-root "$DATA_ROOT" --cross-root "$ROOT" \
      > "$LOG_ROOT/VIS_audit.log" 2>&1
  fi
fi
