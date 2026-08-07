#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${PROJECT_ROOT}/scripts/structure_da_v3_common.sh"

# Force train.py to use its parser default data_root.
# Do not inherit a DATA_ROOT override from the shell.
DATA_ROOT=""

STAGE2_CONFIG="${STAGE2_CONFIG:-${PROJECT_ROOT}/configs/stage2_pilot_v1.json}"
RUN_GROUP="${RUN_GROUP:-v3_pilot_round1_seed1}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"

SEED=1

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"

require_file "${STAGE2_CONFIG}"

GROUP_OUTPUT_DIRECTORY="${OUTPUT_ROOT}/${RUN_GROUP}"
GROUP_LOG_ROOT="${LOG_ROOT}/${RUN_GROUP}"
GROUP_TRAIN_LOG_DIRECTORY="${GROUP_LOG_ROOT}/train_logs"

prepare_run_group "${GROUP_OUTPUT_DIRECTORY}" "${GROUP_LOG_ROOT}"

declare -a PIDS=()
declare -a TASKS=()

launch_task() {
    local physical_gpu="$1"
    local task="$2"
    local source_domain="$3"
    local target_domain="$4"

    local run_output_directory="${GROUP_OUTPUT_DIRECTORY}/${task}"
    local task_log_file="${GROUP_TRAIN_LOG_DIRECTORY}/${task}.log"

    make_run_output_directory "${run_output_directory}"

    echo "PILOT_START|task=${task}|gpu=${physical_gpu}|seed=${SEED}|log=${task_log_file}"

    run_training \
        "${source_domain}" \
        "${target_domain}" \
        "${SEED}" \
        "${physical_gpu}" \
        "${run_output_directory}" \
        "${task_log_file}" \
        --stage1_epochs 100 \
        --stage2_epochs 60 \
        --stage2_block_epochs 20 \
        --progress_bar off \
        --feature_snapshot_interval 0 \
        &

    PIDS+=("$!")
    TASKS+=("${task}")
}

launch_task "${GPU0}" "AT1_DK1_seed1" "${AT1}" "${DK1}"
launch_task "${GPU1}" "DK1_FR2_seed1" "${DK1}" "${FR2}"
launch_task "${GPU2}" "FR1_AT1_seed1" "${FR1}" "${AT1}"
launch_task "${GPU3}" "FR2_FR1_seed1" "${FR2}" "${FR1}"

echo "PILOT_LAUNCHED|count=${#PIDS[@]}|run_group=${RUN_GROUP}"
echo "PILOT_LOG_DIR|${GROUP_TRAIN_LOG_DIRECTORY}"
echo "PILOT_OUTPUT_DIR|${GROUP_OUTPUT_DIRECTORY}"

FAILED=0

for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "PILOT_DONE|task=${TASKS[$i]}"
    else
        code=$?
        echo "PILOT_FAILED|task=${TASKS[$i]}|exit_code=${code}" >&2
        FAILED=1
    fi
done

if [[ "${FAILED}" -eq 0 ]]; then
    echo "PILOT_SUMMARY|status=SUCCESS|runs=4"
else
    echo "PILOT_SUMMARY|status=FAILED|runs=4" >&2
fi

exit "${FAILED}"
