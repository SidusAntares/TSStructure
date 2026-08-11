#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
REGISTRATION_WORKERS="${REGISTRATION_WORKERS:-4}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${PROJECT_ROOT}/outputs/phase_only_at1_dk1_seed1/fold_0/stage1_best.pt}"
REFERENCE_PHASE_CHECKPOINT="${REFERENCE_PHASE_CHECKPOINT:-${PROJECT_ROOT}/outputs/phase_only_at1_dk1_seed1/fold_0/stage2_best_target_val_ema.pt}"
STAGE2_CONFIG="${STAGE2_CONFIG:-${PROJECT_ROOT}/configs/stage2_phase_only_v1.json}"
CALIBRATION_RUN_GROUP="${CALIBRATION_RUN_GROUP:-phase_class_center_calibration_at1_dk1_seed1}"
CALIBRATION_CHECKPOINT="${CALIBRATION_CHECKPOINT:-${PROJECT_ROOT}/outputs/${CALIBRATION_RUN_GROUP}/AT1_DK1_seed1/fold_0/stage2_calibration_state.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/visualization_phase_only_at1_dk1_seed1/05_class_center_vs_group_center_phase_diagnostic}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PSE_GRID_SIZE="${PSE_GRID_SIZE:-128}"

if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
  echo "ERROR: missing Stage1 checkpoint: ${STAGE1_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${STAGE2_CONFIG}" ]]; then
  echo "ERROR: missing Stage2 config: ${STAGE2_CONFIG}" >&2
  exit 2
fi

# Old Stage2 checkpoints do not contain class-center/progressive Phase state.
# Reuse a newly generated calibration checkpoint when available; otherwise run
# the exact same no-training 64->128->256->512 calibration once.
if [[ ! -f "${CALIBRATION_CHECKPOINT}" ]]; then
  echo "PHASE_CLASS_CENTER_CALIBRATION_REQUIRED|checkpoint=${CALIBRATION_CHECKPOINT}"
  env \
    CUDA_DEVICE="${CUDA_DEVICE}" \
    REGISTRATION_WORKERS="${REGISTRATION_WORKERS}" \
    RUN_GROUP="${CALIBRATION_RUN_GROUP}" \
    STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT}" \
    STAGE2_CONFIG="${STAGE2_CONFIG}" \
    bash scripts/run_stage2_calibration_at1_dk1.sh
fi

if [[ ! -f "${CALIBRATION_CHECKPOINT}" ]]; then
  echo "ERROR: calibration did not produce ${CALIBRATION_CHECKPOINT}" >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}"
REFERENCE_ARGS=()
if [[ -f "${REFERENCE_PHASE_CHECKPOINT}" ]]; then
  REFERENCE_ARGS=(--reference-phase-checkpoint "${REFERENCE_PHASE_CHECKPOINT}")
else
  echo "PHASE_REFERENCE_CHECKPOINT_MISSING|path=${REFERENCE_PHASE_CHECKPOINT}|comparison=skipped"
fi

echo "PHASE_CLASS_CENTER_DIAGNOSTIC_START|calibration=${CALIBRATION_CHECKPOINT}|stage1=${STAGE1_CHECKPOINT}|samples_per_class=${SAMPLES_PER_CLASS}|output=${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python -u scripts/diagnose_class_center_vs_group_phase.py \
  --checkpoint "${CALIBRATION_CHECKPOINT}" \
  --model-checkpoint "${STAGE1_CHECKPOINT}" \
  "${REFERENCE_ARGS[@]}" \
  --data-root "${DATA_ROOT}" \
  --device cuda:0 \
  --samples-per-class "${SAMPLES_PER_CLASS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --pse-grid-size "${PSE_GRID_SIZE}" \
  --output-dir "${OUTPUT_DIR}"

echo "PHASE_CLASS_CENTER_DIAGNOSTIC_COMPLETE|output=${OUTPUT_DIR}"
