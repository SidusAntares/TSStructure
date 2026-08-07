#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=structure_da_v3_common.sh
source "${SCRIPT_DIR}/structure_da_v3_common.sh"

SOURCE_DOMAIN="${SOURCE_DOMAIN:-austria/33UVP/2017}"
TARGET_DOMAIN="${TARGET_DOMAIN:-denmark/32VNH/2017}"
SEED="${SEED:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_STEPS_PER_EPOCH="${SMOKE_STEPS_PER_EPOCH:-2}"
SMOKE_STAGE2_EPOCHS="${SMOKE_STAGE2_EPOCHS:-1}"
SMOKE_STAGE2_STEPS_PER_EPOCH="${SMOKE_STAGE2_STEPS_PER_EPOCH:-1}"
RUN_OUTPUT_DIRECTORY="${OUTPUT_ROOT}/smoke_structure_da_v3"
SMOKE_LOG_ROOT="${LOG_ROOT}/smoke_structure_da_v3"
SMOKE_TRAIN_LOG_DIRECTORY="${SMOKE_LOG_ROOT}/train_logs"
SMOKE_SNAPSHOT_DIRECTORY="${SMOKE_LOG_ROOT}/snapshots"
SMOKE_LOG_FILE="${SMOKE_TRAIN_LOG_DIRECTORY}/smoke.log"

if [[ -z "${STAGE2_CONFIG}" ]]; then
    echo "STAGE2_CONFIG=/path/to/stage2_config.json is required for the V3 smoke." >&2
    exit 1
fi
require_file "${STAGE2_CONFIG}"

prepare_run_group "${RUN_OUTPUT_DIRECTORY}" "${SMOKE_LOG_ROOT}"

status=0
if run_training \
    "${SOURCE_DOMAIN}" "${TARGET_DOMAIN}" "${SEED}" "${CUDA_DEVICE}" \
    "${RUN_OUTPUT_DIRECTORY}" "${SMOKE_LOG_FILE}" \
    --stage1_epochs "${SMOKE_EPOCHS}" \
    --steps_per_epoch "${SMOKE_STEPS_PER_EPOCH}" --log_step 1 \
    --stage2_epochs "${SMOKE_STAGE2_EPOCHS}" \
    --stage2_block_epochs "${SMOKE_STAGE2_EPOCHS}" \
    --stage2_steps_per_epoch "${SMOKE_STAGE2_STEPS_PER_EPOCH}" \
    --feature_snapshot_interval 0
then
    status=0
else
    status=$?
fi

if [[ "${status}" -eq 0 ]] && \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/check_structure_da_smoke.py" \
        "${RUN_OUTPUT_DIRECTORY}" --log-directory "${SMOKE_TRAIN_LOG_DIRECTORY}" \
        >> "${SMOKE_LOG_FILE}" 2>&1
then
    echo "SMOKE_RESULT|status=SUCCESS" >> "${SMOKE_LOG_FILE}"
    exit 0
fi

echo "SMOKE_RESULT|status=FAILED" >> "${SMOKE_LOG_FILE}"
exit 1
