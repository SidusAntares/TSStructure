#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=structure_da_v3_common.sh
source "${SCRIPT_DIR}/structure_da_v3_common.sh"

RUN_GROUP="${RUN_GROUP:-}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
EXPECTED_RUNS=36

if [[ -z "${RUN_GROUP}" ]]; then
    echo "RUN_GROUP is required." >&2
    exit 1
fi
if [[ ! "${RUN_GROUP}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUN_GROUP must contain only letters, digits, underscores, and hyphens" >&2
    exit 1
fi

GROUP_LOG_DIRECTORY="${LOG_ROOT}/${RUN_GROUP}"
GROUP_OUTPUT_DIRECTORY="${OUTPUT_ROOT}/${RUN_GROUP}"
MANIFEST_FILE="${GROUP_LOG_DIRECTORY}/manifest.tsv"
COMPLETED_FILE="${GROUP_LOG_DIRECTORY}/completed.tsv"
FAILED_FILE="${GROUP_LOG_DIRECTORY}/failed.tsv"

TASKS=(
    "AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017"
    "AT1|austria/33UVP/2017|FR1|france/31TCJ/2017"
    "AT1|austria/33UVP/2017|FR2|france/30TXT/2017"
    "DK1|denmark/32VNH/2017|AT1|austria/33UVP/2017"
    "DK1|denmark/32VNH/2017|FR1|france/31TCJ/2017"
    "DK1|denmark/32VNH/2017|FR2|france/30TXT/2017"
    "FR1|france/31TCJ/2017|AT1|austria/33UVP/2017"
    "FR1|france/31TCJ/2017|DK1|denmark/32VNH/2017"
    "FR1|france/31TCJ/2017|FR2|france/30TXT/2017"
    "FR2|france/30TXT/2017|AT1|austria/33UVP/2017"
    "FR2|france/30TXT/2017|DK1|denmark/32VNH/2017"
    "FR2|france/30TXT/2017|FR1|france/31TCJ/2017"
)
SEEDS=(1 2 3)
PHYSICAL_GPUS=("${GPU0}" "${GPU1}" "${GPU2}" "${GPU3}")
JOBS=()
for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${task}|${seed}")
    done
done
if [[ "${#JOBS[@]}" -ne "${EXPECTED_RUNS}" ]]; then
    echo "Internal job-count error: expected ${EXPECTED_RUNS}, got ${#JOBS[@]}" >&2
    exit 1
fi

activate_environment
prepare_run_group "${GROUP_OUTPUT_DIRECTORY}" "${GROUP_LOG_DIRECTORY}"
printf '%s\n' "$$" > "${GROUP_LOG_DIRECTORY}/launcher.pid"
printf 'run_name\tsource\ttarget\tseed\tgpu\toutput_directory\tlog_directory\n' \
    > "${MANIFEST_FILE}"
printf 'run_name\texit_code\n' > "${COMPLETED_FILE}"
printf 'run_name\texit_code\n' > "${FAILED_FILE}"

for job_index in "${!JOBS[@]}"; do
    IFS='|' read -r source_short source_domain target_short target_domain seed \
        <<< "${JOBS[$job_index]}"
    run_name="${source_short}_${target_short}_seed${seed}"
    worker_id=$((job_index % 4))
    physical_gpu="${PHYSICAL_GPUS[$worker_id]}"
    run_output_directory="${GROUP_OUTPUT_DIRECTORY}/${run_name}"
    run_log_directory="${GROUP_LOG_DIRECTORY}/${run_name}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${run_name}" "${source_domain}" "${target_domain}" "${seed}" \
        "${physical_gpu}" "${run_output_directory}" "${run_log_directory}" \
        >> "${MANIFEST_FILE}"
done

run_worker() {
    local worker_id="$1"
    local physical_gpu="$2"
    local job_index="${worker_id}"
    while [[ "${job_index}" -lt "${#JOBS[@]}" ]]; do
        local source_short source_domain target_short target_domain seed
        IFS='|' read -r source_short source_domain target_short target_domain seed \
            <<< "${JOBS[$job_index]}"
        local run_name="${source_short}_${target_short}_seed${seed}"
        local RUN_OUTPUT_DIRECTORY="${GROUP_OUTPUT_DIRECTORY}/${run_name}"
        local RUN_LOG_DIRECTORY="${GROUP_LOG_DIRECTORY}/${run_name}"
        local exit_code=0

        echo "TASK_START|run=${run_name}|gpu=${physical_gpu}|worker=${worker_id}"
        if make_run_directories "${RUN_OUTPUT_DIRECTORY}" "${RUN_LOG_DIRECTORY}" && \
            run_training \
                "${source_domain}" "${target_domain}" "${seed}" "${physical_gpu}" \
                "${RUN_OUTPUT_DIRECTORY}" "${RUN_LOG_DIRECTORY}"
        then
            exit_code=0
            : > "${RUN_LOG_DIRECTORY}/TASK_DONE"
            printf '%s\t%s\n' "${run_name}" "${exit_code}" >> "${COMPLETED_FILE}"
            echo "TASK_DONE|run=${run_name}|gpu=${physical_gpu}|exit_code=0"
        else
            exit_code=$?
            mkdir -p "${RUN_LOG_DIRECTORY}"
            : > "${RUN_LOG_DIRECTORY}/TASK_FAILED"
            printf '%s\t%s\n' "${run_name}" "${exit_code}" >> "${FAILED_FILE}"
            echo "TASK_FAILED|run=${run_name}|gpu=${physical_gpu}|exit_code=${exit_code}" >&2
        fi
        printf '%s\n' "${exit_code}" > "${RUN_LOG_DIRECTORY}/exit_code.txt"
        echo "TASK_PROGRESS|completed_or_failed=$((job_index / 4 + 1))/9|worker=${worker_id}"
        ((job_index += 4))
    done
    return 0
}

WORKER_PIDS=()
run_worker 0 "${GPU0}" &
WORKER_PIDS+=("$!")
run_worker 1 "${GPU1}" &
WORKER_PIDS+=("$!")
run_worker 2 "${GPU2}" &
WORKER_PIDS+=("$!")
run_worker 3 "${GPU3}" &
WORKER_PIDS+=("$!")

for worker_pid in "${WORKER_PIDS[@]}"; do
    wait "${worker_pid}" || true
done

completed_count="$(($(wc -l < "${COMPLETED_FILE}") - 1))"
failed_count="$(($(wc -l < "${FAILED_FILE}") - 1))"
echo "EXPERIMENT_SUMMARY|total=${EXPECTED_RUNS}|completed=${completed_count}|failed=${failed_count}"
if [[ "${failed_count}" -eq 0 && "${completed_count}" -eq "${EXPECTED_RUNS}" ]]; then
    : > "${GROUP_LOG_DIRECTORY}/EXPERIMENT_DONE"
    exit 0
fi
: > "${GROUP_LOG_DIRECTORY}/EXPERIMENT_FAILED"
exit 1
