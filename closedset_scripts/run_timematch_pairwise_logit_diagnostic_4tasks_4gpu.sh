#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/timematch_historical_rerun_seed1/source}"
UDA_ROOT="${UDA_ROOT:-outputs/timematch_historical_rerun_seed1/uda}"
TASKS="${TASKS:-AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1}"
GPUS="${GPUS:-3 1 2 0}"
EXPERIMENT="${EXPERIMENT:-timematch_pairwise_logit_diagnostic_seed1}"
OUTPUT_ROOT="outputs/${EXPERIMENT}"
LOG_ROOT="logs/${EXPERIMENT}"
mkdir -p "$LOG_ROOT"
exec >> "$LOG_ROOT/diagnostic_control.log" 2>&1

[[ -d "$DATA_ROOT" ]] || { echo "ERROR: dataset root missing: $DATA_ROOT"; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || {
    echo "ERROR: existing output: $OUTPUT_ROOT; choose a new EXPERIMENT"; exit 1;
}
read -r -a tasks <<< "${TASKS//,/ }"
read -r -a gpus <<< "${GPUS//,/ }"
(( ${#tasks[@]} > 0 && ${#tasks[@]} <= ${#gpus[@]} )) || {
    echo 'ERROR: provide one GPU per task'; exit 1;
}
shift_args=()
if [[ -n "${STUDENT_SOURCE_SHIFT:-}" ]]; then
    (( ${#tasks[@]} == 1 )) || { echo 'ERROR: explicit shift requires exactly one task'; exit 1; }
    shift_args=(--student-source-shift "$STUDENT_SOURCE_SHIFT")
fi

resolve_task() {
    case "$1" in
        AT1_DK1) source_tile=33UVP; target_tile=32VNH ;;
        FR1_FR2) source_tile=30TXT; target_tile=31TCJ ;;
        FR2_DK1) source_tile=31TCJ; target_tile=32VNH ;;
        DK1_AT1) source_tile=32VNH; target_tile=33UVP ;;
        *) echo "ERROR: unknown task $1"; return 1 ;;
    esac
}

declare -A seen_tasks=() seen_gpus=()
for index in "${!tasks[@]}"; do
    task="${tasks[$index]}"; gpu="${gpus[$index]}"
    [[ -z "${seen_tasks[$task]:-}" && -z "${seen_gpus[$gpu]:-}" ]] || {
        echo 'ERROR: duplicate task or GPU'; exit 1;
    }
    seen_tasks[$task]=1; seen_gpus[$gpu]=1
    resolve_task "$task"
    for checkpoint in "$SOURCE_ROOT/closedset_source_${source_tile}_seed1/fold_0/model.pt" \
                      "$UDA_ROOT/closedset_timematch_${source_tile}_to_${target_tile}_seed1/fold_0/model.pt"; do
        [[ -f "$checkpoint" ]] || { echo "ERROR: missing checkpoint: $checkpoint"; exit 1; }
    done
done

run_task() {
    local gpu="$1" task="$2"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u -m analysis.timematch_pairwise_logits \
        --data-root "$DATA_ROOT" --source-root "$SOURCE_ROOT" --uda-root "$UDA_ROOT" \
        --task "$task" --output "$OUTPUT_ROOT/$task" --device cuda "${shift_args[@]}" \
        > "$LOG_ROOT/diagnostic_${task}.log" 2>&1
}

mkdir -p "$OUTPUT_ROOT"
pids=()
for index in "${!tasks[@]}"; do
    run_task "${gpus[$index]}" "${tasks[$index]}" &
    pids+=("$!")
done
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "TASK_COMPLETE|task=${tasks[$index]}|cases=4"
    else
        echo "FAILED|task=${tasks[$index]}|see=$LOG_ROOT/diagnostic_${tasks[$index]}.log"
        status=1
    fi
done
if (( status == 0 )); then
    echo "RUN_COMPLETE|tasks=${#tasks[@]}|cases=$((4*${#tasks[@]}))|complete_logits=true"
fi
exit "$status"
