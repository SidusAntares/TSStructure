#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
TASKS="${TASKS:-AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1}"
GPUS="${GPUS:-0 1 2 3}"
SEED="${SEED:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EXPERIMENT="${EXPERIMENT:-timematch_original_diagnostic_seed${SEED}}"
OUTPUT_ROOT="outputs/${EXPERIMENT}"
LOG_ROOT="logs/${EXPERIMENT}"
mkdir -p "$LOG_ROOT"
exec > "$LOG_ROOT/diagnostic_control.log" 2>&1

[[ -d "$DATA_ROOT" ]] || { echo "ERROR: dataset root missing: $DATA_ROOT"; exit 1; }
read -r -a tasks <<< "${TASKS//,/ }"
read -r -a gpus <<< "${GPUS//,/ }"
(( ${#tasks[@]} > 0 && ${#tasks[@]} <= ${#gpus[@]} )) || {
    echo 'ERROR: provide one GPU per task'; exit 1;
}

resolve_task() {
    case "$1" in
        AT1_DK1) source="austria/33UVP/2017"; target="denmark/32VNH/2017"; tile="33UVP" ;;
        FR1_FR2) source="france/30TXT/2017"; target="france/31TCJ/2017"; tile="30TXT" ;;
        FR2_DK1) source="france/31TCJ/2017"; target="denmark/32VNH/2017"; tile="31TCJ" ;;
        DK1_AT1) source="denmark/32VNH/2017"; target="austria/33UVP/2017"; tile="32VNH" ;;
        *) echo "ERROR: unknown task $1"; return 1 ;;
    esac
}

# Validate all checkpoints before any UDA starts. This script never trains Source.
declare -A seen_tasks=() seen_gpus=()
for index in "${!tasks[@]}"; do
    task="${tasks[$index]}"
    gpu="${gpus[$index]}"
    [[ -z "${seen_tasks[$task]:-}" && -z "${seen_gpus[$gpu]:-}" ]] || {
        echo 'ERROR: duplicate task or GPU'; exit 1;
    }
    seen_tasks[$task]=1
    seen_gpus[$gpu]=1
    resolve_task "$task"
    source_weights="outputs/closedset_source_${tile}_seed${SEED}"
    [[ -f "$source_weights/fold_0/model.pt" ]] || {
        echo "ERROR: source checkpoint missing: $source_weights/fold_0/model.pt"; exit 1;
    }
    [[ ! -e "$OUTPUT_ROOT/uda/$task" && ! -e "$OUTPUT_ROOT/diagnostics/$task" ]] || {
        echo "ERROR: existing experiment for $task; choose a new EXPERIMENT to preserve results"; exit 1;
    }
    "$PYTHON_BIN" -u -m analysis.timematch_diagnostic validate-source \
        --root "$source_weights" --source "$source" --seed "$SEED"
done

run_task() {
    local gpu="$1" task="$2" source target tile
    resolve_task "$task"
    local source_weights="outputs/closedset_source_${tile}_seed${SEED}"
    local diagnostic_root="$OUTPUT_ROOT/diagnostics/$task"
    echo "UDA_START|gpu=$gpu|task=$task|source=$source_weights|epochs=20"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
        --data_root "$DATA_ROOT" --source "$source" --target "$target" \
        --seed "$SEED" --num_workers "$NUM_WORKERS" --num_folds 1 \
        --model pseltae --device cuda --closed_set true \
        --combine_spring_and_winter false --with_shift_aug false \
        --progress_bar off --experiment_name "$task" \
        --output_dir "$OUTPUT_ROOT/uda" \
        --tensorboard_log_dir "$OUTPUT_ROOT/tensorboard" \
        timematch --weights "$source_weights" \
        --epochs 20 --steps_per_epoch 500 --evidence_mode off --output_student true \
        --diagnostic_dir "$diagnostic_root" \
        --diagnostic_log "$LOG_ROOT/diagnostic_${task}.log" \
        --diagnostic_task "$task" \
        > "$LOG_ROOT/${task}.log" 2>&1
    "$PYTHON_BIN" -u -m analysis.timematch_diagnostic verify-task \
        --root "$diagnostic_root" --task "$task" \
        --training-dir "$OUTPUT_ROOT/uda/$task/fold_0"
}

pids=()
for index in "${!tasks[@]}"; do
    run_task "${gpus[$index]}" "${tasks[$index]}" &
    pids+=("$!")
done
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "TASK_COMPLETE|task=${tasks[$index]}|tested_epoch=20|structures=18|contributions=54"
    else
        echo "ERROR: task ${tasks[$index]} failed; see ${tasks[$index]}.log and diagnostic_${tasks[$index]}.log"
        status=1
    fi
done
if (( status == 0 )); then
    echo "RUN_VERIFIED|tasks=${#tasks[@]}|structures=$((18*${#tasks[@]}))|contributions=$((54*${#tasks[@]}))"
fi
exit "$status"
