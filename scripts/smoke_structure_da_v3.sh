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
RUN_OUTPUT_DIRECTORY="${OUTPUT_ROOT}/smoke_structure_da_v3"
RUN_LOG_DIRECTORY="${LOG_ROOT}/smoke_structure_da_v3"

activate_environment
prepare_run_group "${RUN_OUTPUT_DIRECTORY}" "${RUN_LOG_DIRECTORY}"

status=0
if run_training \
    "${SOURCE_DOMAIN}" "${TARGET_DOMAIN}" "${SEED}" "${CUDA_DEVICE}" \
    "${RUN_OUTPUT_DIRECTORY}" "${RUN_LOG_DIRECTORY}" \
    --epochs "${SMOKE_EPOCHS}" \
    --steps_per_epoch "${SMOKE_STEPS_PER_EPOCH}" --log_step 1
then
    status=0
else
    status=$?
fi

if [[ "${status}" -eq 0 ]] && \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/check_structure_da_smoke.py" \
        "${RUN_OUTPUT_DIRECTORY}" --log-directory "${RUN_LOG_DIRECTORY}"
then
    : > "${RUN_LOG_DIRECTORY}/SMOKE_SUCCESS"
    echo "SMOKE_SUCCESS|output=${RUN_OUTPUT_DIRECTORY}|logs=${RUN_LOG_DIRECTORY}"
    exit 0
fi

: > "${RUN_LOG_DIRECTORY}/SMOKE_FAILED"
echo "SMOKE_FAILED|output=${RUN_OUTPUT_DIRECTORY}|logs=${RUN_LOG_DIRECTORY}" >&2
exit 1
