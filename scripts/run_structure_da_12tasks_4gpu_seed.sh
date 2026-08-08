#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=structure_da_v3_common.sh
source "${SCRIPT_DIR}/structure_da_v3_common.sh"

SEED="${SEED:-1}"
RUN_GROUP="${RUN_GROUP:-structure_da_v3_seed${SEED}}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
EXPECTED_RUNS=12

if [[ -z "${STAGE2_CONFIG}" ]]; then
    echo "STAGE2_CONFIG=/path/to/stage2_config.json is required" >&2
    exit 1
fi
require_file "${STAGE2_CONFIG}"
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be an integer" >&2
    exit 1
fi
if [[ ! "${RUN_GROUP}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUN_GROUP must contain only letters, digits, underscores, and hyphens" >&2
    exit 1
fi

TASKS=(
    "AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017"
    "AT1|austria/33UVP/2017|FR1|france/30TXT/2017"
    "AT1|austria/33UVP/2017|FR2|france/31TCJ/2017"
    "DK1|denmark/32VNH/2017|AT1|austria/33UVP/2017"
    "DK1|denmark/32VNH/2017|FR1|france/30TXT/2017"
    "DK1|denmark/32VNH/2017|FR2|france/31TCJ/2017"
    "FR1|france/30TXT/2017|AT1|austria/33UVP/2017"
    "FR1|france/30TXT/2017|DK1|denmark/32VNH/2017"
    "FR1|france/30TXT/2017|FR2|france/31TCJ/2017"
    "FR2|france/31TCJ/2017|AT1|austria/33UVP/2017"
    "FR2|france/31TCJ/2017|DK1|denmark/32VNH/2017"
    "FR2|france/31TCJ/2017|FR1|france/30TXT/2017"
)
GPUS=("${GPU0}" "${GPU1}" "${GPU2}" "${GPU3}")

if [[ "${#TASKS[@]}" -ne "${EXPECTED_RUNS}" ]]; then
    echo "Internal task-count error: expected ${EXPECTED_RUNS}, got ${#TASKS[@]}" >&2
    exit 1
fi

GROUP_LOG_ROOT="${LOG_ROOT}/${RUN_GROUP}"
GROUP_TRAIN_LOG_DIRECTORY="${GROUP_LOG_ROOT}/train_logs"
GROUP_OUTPUT_DIRECTORY="${OUTPUT_ROOT}/${RUN_GROUP}"
MANIFEST_FILE="${GROUP_TRAIN_LOG_DIRECTORY}/manifest.tsv"
COMPLETED_FILE="${GROUP_TRAIN_LOG_DIRECTORY}/completed.tsv"
FAILED_FILE="${GROUP_TRAIN_LOG_DIRECTORY}/failed.tsv"
STATUS_FILE="${GROUP_TRAIN_LOG_DIRECTORY}/experiment_status.tsv"

prepare_run_group "${GROUP_OUTPUT_DIRECTORY}" "${GROUP_LOG_ROOT}"
printf '%s\n' "$$" > "${GROUP_TRAIN_LOG_DIRECTORY}/launcher.pid"
printf 'run_name\tsource\ttarget\tseed\tgpu\toutput_directory\tlog_file\n' > "${MANIFEST_FILE}"
printf 'run_name\texit_code\n' > "${COMPLETED_FILE}"
printf 'run_name\texit_code\n' > "${FAILED_FILE}"
printf 'time\trun_name\tgpu\tpid\tstatus\texit_code\n' > "${STATUS_FILE}"

for index in "${!TASKS[@]}"; do
    IFS='|' read -r src_short src dst_short dst <<< "${TASKS[$index]}"
    worker=$((index % 4))
    gpu="${GPUS[$worker]}"
    run_name="${src_short}_${dst_short}_seed${SEED}"
    out="${GROUP_OUTPUT_DIRECTORY}/${run_name}"
    log="${GROUP_TRAIN_LOG_DIRECTORY}/${run_name}.log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${run_name}" "${src}" "${dst}" "${SEED}" "${gpu}" "${out}" "${log}" \
        >> "${MANIFEST_FILE}"
done

run_worker() {
    local worker="$1"
    local gpu="${GPUS[$worker]}"
    local index="$worker"
    while [[ "$index" -lt "${#TASKS[@]}" ]]; do
        local src_short src dst_short dst
        IFS='|' read -r src_short src dst_short dst <<< "${TASKS[$index]}"
        local run_name="${src_short}_${dst_short}_seed${SEED}"
        local out="${GROUP_OUTPUT_DIRECTORY}/${run_name}"
        local log="${GROUP_TRAIN_LOG_DIRECTORY}/${run_name}.log"
        local code=0
        local status=""
        LAST_TRAINING_PID=""

        echo "TASK_START|run=${run_name}|gpu=${gpu}|worker=${worker}"
        if make_run_output_directory "${out}" && run_training \
            "${src}" "${dst}" "${SEED}" "${gpu}" "${out}" "${log}" \
            --stage1_epochs 100 \
            --stage2_epochs 60 \
            --stage2_block_epochs 20 \
            --stage2_phase_evidence_initial_samples 64 \
            --stage2_phase_evidence_max_samples 512 \
            --stage2_registration_workers 4 \
            --feature_snapshot_interval 0 \
            --progress_bar off
        then
            status="COMPLETED"
            printf '%s\t0\n' "${run_name}" >> "${COMPLETED_FILE}"
        else
            code=$?
            status="FAILED"
            printf '%s\t%s\n' "${run_name}" "${code}" >> "${FAILED_FILE}"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${run_name}" "${gpu}" \
            "${LAST_TRAINING_PID:-}" "${status}" "${code}" >> "${STATUS_FILE}"
        echo "TASK_${status}|run=${run_name}|gpu=${gpu}|exit_code=${code}"
        ((index += 4))
    done
}

PIDS=()
for worker in 0 1 2 3; do
    run_worker "${worker}" &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
    wait "${pid}" || true
done

completed_count="$(($(wc -l < "${COMPLETED_FILE}") - 1))"
failed_count="$(($(wc -l < "${FAILED_FILE}") - 1))"
echo "EXPERIMENT_SUMMARY|seed=${SEED}|total=${EXPECTED_RUNS}|completed=${completed_count}|failed=${failed_count}"
if [[ "${completed_count}" -eq "${EXPECTED_RUNS}" && "${failed_count}" -eq 0 ]]; then
    printf '%s\tEXPERIMENT\t-\t%s\tCOMPLETED\t0\n' \
        "$(date --iso-8601=seconds)" "$$" >> "${STATUS_FILE}"
    exit 0
fi
printf '%s\tEXPERIMENT\t-\t%s\tFAILED\t1\n' \
    "$(date --iso-8601=seconds)" "$$" >> "${STATUS_FILE}"
exit 1
