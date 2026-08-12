#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${PROJECT_ROOT}/outputs/phase_only_at1_dk1_seed1/fold_0/stage1_best.pt}"
CALIBRATION_CHECKPOINT="${CALIBRATION_CHECKPOINT:-${PROJECT_ROOT}/outputs/phase_class_center_calibration_at1_dk1_seed1/AT1_DK1_seed1/fold_0/stage2_calibration_state.pt}"
STAGE_A_CACHE="${STAGE_A_CACHE:-${PROJECT_ROOT}/outputs/visualization_phase_only_at1_dk1_seed1/06_sample_level_phase_validity_diagnostic/cache/stage_a_t_only_registrations.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/visualization_phase_only_at1_dk1_seed1/08_oracle_true_class_gamma_harm_mechanism_diagnostic}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

for required in "${STAGE1_CHECKPOINT}" "${CALIBRATION_CHECKPOINT}" "${STAGE_A_CACHE}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ORACLE_GAMMA_08_MISSING_INPUT|path=${required}|exact_dp_fallback=false" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
echo "ORACLE_GAMMA_08_LAUNCH|stage_a_cache=${STAGE_A_CACHE}|stage1=${STAGE1_CHECKPOINT}|exact_dp_calls=0|output=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python -u scripts/diagnose_oracle_gamma_harm_mechanism.py \
  --stage-a-cache "${STAGE_A_CACHE}" \
  --calibration-checkpoint "${CALIBRATION_CHECKPOINT}" \
  --model-checkpoint "${STAGE1_CHECKPOINT}" \
  --data-root "${DATA_ROOT}" \
  --device cuda:0 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --output-dir "${OUTPUT_DIR}"

echo "ORACLE_GAMMA_08_COMPLETE|output=${OUTPUT_DIR}"
