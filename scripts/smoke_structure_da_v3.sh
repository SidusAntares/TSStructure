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
RUN_DIRECTORY="${OUTPUT_ROOT}/smoke/AT1_DK1/seed_${SEED}"

activate_environment
make_run_directory "${RUN_DIRECTORY}"

status=0
if ! run_training \
    "${SOURCE_DOMAIN}" "${TARGET_DOMAIN}" "${SEED}" "${CUDA_DEVICE}" \
    "${SMOKE_EPOCHS}" "${RUN_DIRECTORY}" \
    --steps_per_epoch "${SMOKE_STEPS_PER_EPOCH}" --log_step 1
then
    status=$?
    [[ "${status}" -eq 0 ]] && status=1
fi

if [[ "${status}" -eq 0 ]] && \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/check_structure_da_smoke.py" "${RUN_DIRECTORY}"
then
    : > "${RUN_DIRECTORY}/SMOKE_SUCCESS"
    echo "SMOKE_SUCCESS|run_directory=${RUN_DIRECTORY}"
    exit 0
fi

: > "${RUN_DIRECTORY}/SMOKE_FAILED"
echo "SMOKE_FAILED|run_directory=${RUN_DIRECTORY}" >&2
exit 1
