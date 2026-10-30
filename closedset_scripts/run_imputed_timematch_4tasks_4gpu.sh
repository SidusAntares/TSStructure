#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
TASKS="${TASKS:-AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1}"
GPUS="${GPUS:-3 1 2 0}"
EXPERIMENT="${EXPERIMENT:-timematch_imputation_seed1}"
OUTPUT_ROOT="outputs/$EXPERIMENT"
LOG_ROOT="logs/$EXPERIMENT"
SOURCE_ROOT="outputs/timematch_historical_rerun_seed1/source"
mkdir -p "$LOG_ROOT"
exec >> "$LOG_ROOT/launcher.log" 2>&1

[[ "${SEED:-1}" == 1 && "${NUM_WORKERS:-8}" == 8 ]] || {
    echo 'ERROR: experiment requires seed 1 and workers 8'; exit 1;
}
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: missing dataset: $DATA_ROOT"; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "ERROR: output exists: $OUTPUT_ROOT"; exit 1; }
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
declare -A seen_tasks=() seen_gpus=()
for index in "${!tasks[@]}"; do
    task="${tasks[$index]}"; gpu="${gpus[$index]}"
    [[ -z "${seen_tasks[$task]:-}" && -z "${seen_gpus[$gpu]:-}" ]] || {
        echo 'ERROR: duplicate task or GPU'; exit 1;
    }
    seen_tasks[$task]=1; seen_gpus[$gpu]=1
    resolve_task "$task"
    [[ -f "$SOURCE_ROOT/closedset_source_${tile}_seed1/fold_0/model.pt" ]] || {
        echo "ERROR: historical original Source missing for $task"; exit 1;
    }
done

run_task() {
    local gpu="$1" task="$2" source target tile variant
    resolve_task "$task"
    local source_weights="$SOURCE_ROOT/closedset_source_${tile}_seed1"
    local imputer_checkpoint="$OUTPUT_ROOT/imputer/$task/fold_0/imputer_epoch20.pt"
    local common=(--source "$source" --target "$target" --weights "$source_weights"
        --data_root "$DATA_ROOT" --seed 1 --num_workers 8 --epochs 20
        --experiment_name "$task" --progress_bar off)
    echo "TASK_START|gpu=$gpu|task=$task|source=$source_weights"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_imputed_timematch.py \
        --stage pretrain "${common[@]}" --output_dir "$OUTPUT_ROOT/imputer/$task" \
        --tensorboard_log_dir "$OUTPUT_ROOT/tensorboard/imputer/$task" \
        > "$LOG_ROOT/${task}_imputer.log" 2>&1 || return $?
    # Each CLI validates the completed epoch-20 packet; A and B both start here.
    for variant in A B; do
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_imputed_timematch.py \
            --stage uda --variant "$variant" "${common[@]}" --steps_per_epoch 500 \
            --imputer_checkpoint "$imputer_checkpoint" --output_dir "$OUTPUT_ROOT/uda_${variant}/$task" \
            --tensorboard_log_dir "$OUTPUT_ROOT/tensorboard/uda_${variant}/$task" \
            > "$LOG_ROOT/${task}_${variant}.log" 2>&1 || return $?
    done
}

mkdir "$OUTPUT_ROOT"
pids=()
for index in "${!tasks[@]}"; do
    run_task "${gpus[$index]}" "${tasks[$index]}" &
    pids+=("$!")
done
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "TASK_COMPLETE|task=${tasks[$index]}|imputer_epochs=20|uda_A_epochs=20|uda_B_epochs=20"
    else
        echo "ERROR: task ${tasks[$index]} failed; inspect $LOG_ROOT/${tasks[$index]}_{imputer,A,B}.log"
        status=1
    fi
done
exit "$status"
