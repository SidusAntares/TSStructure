#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=structure_da_v3_common.sh
source "${SCRIPT_DIR}/structure_da_v3_common.sh"

# 这是诊断 pilot，不是最终报告结果。
PILOT_EPOCHS="${PILOT_EPOCHS:-25}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
PILOT_ROOT="${OUTPUT_ROOT}/pilot4"
STATUS_ROOT="${PILOT_ROOT}/.launcher_status"
PILOT_MANIFEST="${PILOT_ROOT}/pilot4_manifest.json"
PILOT_SUMMARY_JSON="${PILOT_ROOT}/pilot4_summary.json"
PILOT_SUMMARY_MD="${PILOT_ROOT}/pilot4_summary.md"

TASKS=(
    "AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|1|${GPU0}"
    "AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|2|${GPU1}"
    "AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|3|${GPU2}"
    "DK1|denmark/32VNH/2017|FR2|france/30TXT/2017|1|${GPU3}"
)

if [[ -e "${PILOT_ROOT}" ]]; then
    if [[ "${OVERWRITE}" != "1" ]]; then
        echo "Pilot root exists; set OVERWRITE=1 to replace it: ${PILOT_ROOT}" >&2
        exit 1
    fi
    rm -rf -- "${PILOT_ROOT}"
fi
mkdir -p "${STATUS_ROOT}"

run_one() {
    local specification="$1"
    local source_short source_domain target_short target_domain seed physical_gpu
    IFS='|' read -r source_short source_domain target_short target_domain seed physical_gpu \
        <<< "${specification}"
    local run_name="${source_short}_${target_short}_seed${seed}"
    local run_directory="${PILOT_ROOT}/${run_name}"
    local status_file="${STATUS_ROOT}/${run_name}.status"
    local start_time end_time exit_code diagnostic_status

    start_time="$(date --iso-8601=seconds)"
    make_run_directory "${run_directory}"
    printf '%s\n' "${BASHPID}" > "${run_directory}/pid.txt"
    echo "TASK_START|gpu=${physical_gpu}|source=${source_short}|target=${target_short}|seed=${seed}"
    if CUDA_VISIBLE_DEVICES="${physical_gpu}" run_training \
        "${source_domain}" "${target_domain}" "${seed}" "${physical_gpu}" \
        "${run_directory}" "${run_directory}/train.log" \
        --stage1_epochs "${PILOT_EPOCHS}" --feature_snapshot_interval 0
    then
        exit_code=0
        "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_structure_da_diagnostic.py" \
            "${run_directory}" || exit_code=$?
    else
        exit_code=$?
    fi
    end_time="$(date --iso-8601=seconds)"
    diagnostic_status="NOT_ANALYZED"
    if [[ -f "${run_directory}/diagnostic_summary.json" ]]; then
        diagnostic_status="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${run_directory}/diagnostic_summary.json")"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${run_name}" "${source_short}" "${target_short}" "${seed}" \
        "${physical_gpu}" "${start_time}" "${end_time}" "${exit_code}" \
        "${diagnostic_status}" "${run_directory}" > "${status_file}"
    printf '%s\n' "${exit_code}" > "${run_directory}/exit_code.txt"
    if [[ "${exit_code}" -eq 0 ]]; then
        : > "${run_directory}/TASK_DONE"
        echo "TASK_DONE|gpu=${physical_gpu}|run=${run_name}"
    else
        : > "${run_directory}/TASK_FAILED"
        echo "TASK_FAILED|gpu=${physical_gpu}|run=${run_name}|exit_code=${exit_code}" >&2
    fi
    return 0
}

PIDS=()
for task in "${TASKS[@]}"; do
    run_one "${task}" &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
    wait "${pid}" || true
done

analysis_exit=0
"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_structure_da_pilot4.py" "${PILOT_ROOT}" \
    || analysis_exit=$?
require_file "${PILOT_MANIFEST}"
require_file "${PILOT_SUMMARY_JSON}"
require_file "${PILOT_SUMMARY_MD}"
failed_count="$("${PYTHON_BIN}" -c 'import json,sys; print(sum(int(r["exit_code"] != 0) for r in json.load(open(sys.argv[1]))["runs"]))' "${PILOT_MANIFEST}")"
echo "EXPERIMENT_SUMMARY|total=4|failed=${failed_count}"
[[ "${analysis_exit}" -eq 0 && "${failed_count}" -eq 0 ]]
