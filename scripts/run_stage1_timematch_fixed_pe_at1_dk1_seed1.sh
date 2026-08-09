#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-stage1_timematch_fixed_pe_at1_dk1_seed1}"
LOG_ROOT="${LOG_ROOT:-logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

mkdir -p "${LOG_ROOT}/${RUN_NAME}"

exec "${PYTHON_BIN}" -u train.py \
  -e "${RUN_NAME}" \
  --output_dir "${OUTPUT_ROOT}" \
  --source austria/33UVP/2017 \
  --target denmark/32VNH/2017 \
  --seed 1 \
  --device "cuda:${GPU}" \
  --stage1_only \
  --time_encoder_type timematch_fixed_sinusoidal \
  --timematch_pe_period 1000 \
  --timematch_pe_max_shift 100 \
  --stage1_epochs 100 \
  --amp true \
  --amp_dtype float16 \
  --progress_bar off
