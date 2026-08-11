#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${PROJECT_ROOT}/outputs/phase_only_at1_dk1_seed1/fold_0/stage1_best.pt}"
CALIBRATION_CHECKPOINT="${CALIBRATION_CHECKPOINT:-${PROJECT_ROOT}/outputs/phase_class_center_calibration_at1_dk1_seed1/AT1_DK1_seed1/fold_0/stage2_calibration_state.pt}"
STAGE2_CONFIG="${STAGE2_CONFIG:-${PROJECT_ROOT}/configs/stage2_phase_only_v1.json}"
CALIBRATION_RUN_GROUP="${CALIBRATION_RUN_GROUP:-phase_class_center_calibration_at1_dk1_seed1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/visualization_phase_only_at1_dk1_seed1/06_sample_level_phase_validity_diagnostic}"
REGISTRATION_WORKERS="${REGISTRATION_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PSE_GRID_SIZE="${PSE_GRID_SIZE:-128}"
STAGE_B_SAMPLES_PER_CLASS="${STAGE_B_SAMPLES_PER_CLASS:-128}"
MDS_SAMPLES_PER_CLASS="${MDS_SAMPLES_PER_CLASS:-256}"
SPAGHETTI_SAMPLES_PER_CLASS="${SPAGHETTI_SAMPLES_PER_CLASS:-128}"
DP_TARGET_CHUNK_SIZE="${DP_TARGET_CHUNK_SIZE:-512}"

if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
  echo "ERROR: missing Stage1 checkpoint: ${STAGE1_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${STAGE2_CONFIG}" ]]; then
  echo "ERROR: missing Stage2 config: ${STAGE2_CONFIG}" >&2
  exit 2
fi

# 06 needs class-center and final M=1 group-center state from the same Phase
# calibration used by 05.  Reuse it when available; otherwise run calibration
# only.  No Stage2 training is performed here.
if [[ ! -f "${CALIBRATION_CHECKPOINT}" ]]; then
  echo "SAMPLE_PHASE_CALIBRATION_REQUIRED|checkpoint=${CALIBRATION_CHECKPOINT}"
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
echo "SAMPLE_PHASE_VALIDITY_START|calibration=${CALIBRATION_CHECKPOINT}|stage1=${STAGE1_CHECKPOINT}|stage_a=full_target_test|stage_b_per_class=${STAGE_B_SAMPLES_PER_CLASS}|output=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python -u scripts/diagnose_sample_level_phase_validity.py \
  --calibration-checkpoint "${CALIBRATION_CHECKPOINT}" \
  --model-checkpoint "${STAGE1_CHECKPOINT}" \
  --data-root "${DATA_ROOT}" \
  --device cuda:0 \
  --registration-workers "${REGISTRATION_WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --pse-grid-size "${PSE_GRID_SIZE}" \
  --stage-b-samples-per-class "${STAGE_B_SAMPLES_PER_CLASS}" \
  --mds-samples-per-class "${MDS_SAMPLES_PER_CLASS}" \
  --spaghetti-samples-per-class "${SPAGHETTI_SAMPLES_PER_CLASS}" \
  --dp-target-chunk-size "${DP_TARGET_CHUNK_SIZE}" \
  --output-dir "${OUTPUT_DIR}"

echo "SAMPLE_PHASE_VALIDITY_COMPLETE|output=${OUTPUT_DIR}"
