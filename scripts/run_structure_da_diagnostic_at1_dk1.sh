#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=structure_da_v3_common.sh
source "${SCRIPT_DIR}/structure_da_v3_common.sh"

# 这是诊断 pilot，不是最终报告结果。它使用正式 V3 参数与约四分之一正式周期。
SOURCE_DOMAIN="austria/33UVP/2017"
TARGET_DOMAIN="denmark/32VNH/2017"
SEED="1"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DIAGNOSTIC_EPOCHS="${DIAGNOSTIC_EPOCHS:-25}"
RUN_DIRECTORY="${OUTPUT_ROOT}/diagnostic_pilot/AT1_DK1/seed_1"

activate_environment
make_run_directory "${RUN_DIRECTORY}"

if ! run_training \
    "${SOURCE_DOMAIN}" "${TARGET_DOMAIN}" "${SEED}" "${CUDA_DEVICE}" \
    "${DIAGNOSTIC_EPOCHS}" "${RUN_DIRECTORY}"
then
    : > "${RUN_DIRECTORY}/PILOT_FAILED"
    exit 1
fi

analysis_exit=0
"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_structure_da_diagnostic.py" \
    "${RUN_DIRECTORY}" || analysis_exit=$?
status="FAILED"
if [[ -f "${RUN_DIRECTORY}/diagnostic_summary.json" ]]; then
    status="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${RUN_DIRECTORY}/diagnostic_summary.json")"
fi
printf '%s\n' "${status}" > "${RUN_DIRECTORY}/${status}"
[[ "${analysis_exit}" -eq 0 && "${status}" != "FAILED" ]]
