#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=structure_da_v3_common.sh
source "${SCRIPT_DIR}/structure_da_v3_common.sh"

SOURCE_DOMAIN="austria/33UVP/2017"
TARGET_DOMAIN="denmark/32VNH/2017"
SEED="1"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
REGISTRATION_WORKERS="${REGISTRATION_WORKERS:-4}"
RUN_GROUP="${RUN_GROUP:-stage2_calibration_b_at1_dk1}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${PROJECT_ROOT}/outputs/v3_pilot_round1_seed1/AT1_DK1_seed1/fold_0/stage1_best.pt}"
STAGE2_CONFIG="${STAGE2_CONFIG:-${PROJECT_ROOT}/configs/stage2_calibration_b_at1_dk1.json}"

GROUP_OUTPUT_DIRECTORY="${OUTPUT_ROOT}/${RUN_GROUP}"
GROUP_LOG_ROOT="${LOG_ROOT}/${RUN_GROUP}"
GROUP_TRAIN_LOG_DIRECTORY="${GROUP_LOG_ROOT}/train_logs"
RUN_OUTPUT_DIRECTORY="${GROUP_OUTPUT_DIRECTORY}/AT1_DK1_seed1"
TASK_LOG="${GROUP_TRAIN_LOG_DIRECTORY}/AT1_DK1_seed1.log"

require_file "${STAGE1_CHECKPOINT}"
require_file "${STAGE2_CONFIG}"
prepare_run_group "${GROUP_OUTPUT_DIRECTORY}" "${GROUP_LOG_ROOT}"
make_run_output_directory "${RUN_OUTPUT_DIRECTORY}"

export STAGE2_CONFIG
export CODE_VERSION="structure_da_v3_roundc_calibration_b"

echo "STAGE2_CALIBRATION_B_START|checkpoint=${STAGE1_CHECKPOINT}|config=${STAGE2_CONFIG}|gpu=${CUDA_DEVICE}"
run_training \
    "${SOURCE_DOMAIN}" \
    "${TARGET_DOMAIN}" \
    "${SEED}" \
    "${CUDA_DEVICE}" \
    "${RUN_OUTPUT_DIRECTORY}" \
    "${TASK_LOG}" \
    --stage2_only \
    --stage1_checkpoint "${STAGE1_CHECKPOINT}" \
    --stage2_diagnostic_only \
    --stage2_epochs 1 \
    --stage2_block_epochs 1 \
    --stage2_phase_evidence_initial_samples 64 \
    --stage2_phase_evidence_max_samples 512 \
    --stage2_registration_workers "${REGISTRATION_WORKERS}" \
    --feature_snapshot_interval 0 \
    --progress_bar off

echo "STAGE2_CALIBRATION_B_COMPLETE|output=${RUN_OUTPUT_DIRECTORY}/fold_0|log=${TASK_LOG}"
