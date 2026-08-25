#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

SEED="${SEED:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-reimts_mtan_openset_AT1_3tasks_seed1}"

LOG_ROOT="logs/$EXPERIMENT_NAME"
OUTPUT_ROOT="outputs/$EXPERIMENT_NAME"
RUN_ROOT="runs/$EXPERIMENT_NAME"
STATE_ROOT="$LOG_ROOT/.launcher_state"
mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$RUN_ROOT" "$STATE_ROOT"

SOURCE_ALIAS="AT1"
SOURCE="austria/33UVP/2017"

format_shell_duration() {
    local total="$1"
    printf '%02d:%02d:%02d' \
        "$((total / 3600))" \
        "$(((total % 3600) / 60))" \
        "$((total % 60))"
}

run_task() {
    local gpu_id="$1"
    local task="$2"
    shift 2
    local log_file="$LOG_ROOT/$task.log"
    local started ended exit_code runtime_seconds
    started="$(date +%s)"
    {
        echo "[RUN_START]"
        echo "task=$task"
        echo "gpu=$gpu_id"
        echo "time=$(date -Iseconds)"
        echo
        CUDA_VISIBLE_DEVICES="$gpu_id" "$@"
        exit_code="$?"
        ended="$(date +%s)"
        runtime_seconds="$((ended - started))"
        echo
        echo "[RUN_END]"
        echo "task=$task"
        echo "gpu=$gpu_id"
        echo "time=$(date -Iseconds)"
        echo "exit_code=$exit_code"
        echo "runtime_seconds=$runtime_seconds"
        echo "runtime=$(format_shell_duration "$runtime_seconds")"
    } > "$log_file" 2>&1
    return "$exit_code"
}

run_source() {
    local gpu_id="$1"
    local task="source_AT1"
    echo "[GPU$gpu_id] START source $SOURCE_ALIAS"
    if ! run_task "$gpu_id" "$task" \
        "$PYTHON_BIN" -u train.py \
        --source "$SOURCE" \
        --target "$SOURCE" \
        --seed "$SEED" \
        --device cuda \
        --closed_set false \
        --combine_spring_and_winter false \
        --with_shift_aug false \
        --model psereimtsmtanltae \
        --reimts_levels 3 \
        --reimts_scale_factor 2 \
        --reimts_period 365 \
        --mtan_num_ref_points 8 \
        --mtan_latent_dim 128 \
        --mtan_heads 1 \
        --reimts_loss_mode sample \
        --num_folds 1 \
        --progress_bar off \
        --epochs 100 \
        --output_dir "$OUTPUT_ROOT" \
        --tensorboard_log_dir "$RUN_ROOT" \
        --experiment_name "$task"; then
        echo "[GPU$gpu_id] FAILED source $SOURCE_ALIAS"
        return 1
    fi
    echo "[GPU$gpu_id] DONE source $SOURCE_ALIAS"
}

run_da() {
    local gpu_id="$1"
    local target_alias="$2"
    local target="$3"
    local task="AT1_to_${target_alias}"
    echo "[GPU$gpu_id] START $task"
    if ! run_task "$gpu_id" "$task" \
        "$PYTHON_BIN" -u train.py \
        --source "$SOURCE" \
        --target "$target" \
        --seed "$SEED" \
        --device cuda \
        --closed_set false \
        --combine_spring_and_winter false \
        --with_shift_aug false \
        --model psereimtsmtanltae \
        --reimts_levels 3 \
        --reimts_scale_factor 2 \
        --reimts_period 365 \
        --mtan_num_ref_points 8 \
        --mtan_latent_dim 128 \
        --mtan_heads 1 \
        --reimts_loss_mode sample \
        --num_folds 1 \
        --progress_bar off \
        --output_dir "$OUTPUT_ROOT" \
        --tensorboard_log_dir "$RUN_ROOT" \
        --experiment_name "$task" \
        timematch \
        --weights "$OUTPUT_ROOT/source_AT1" \
        --epochs 20 \
        --steps_per_epoch 500; then
        echo "$task" > "$STATE_ROOT/$task.failed"
        echo "[GPU$gpu_id] FAILED $task"
        return 1
    fi
    echo "[GPU$gpu_id] DONE $task"
}

if ! run_source 1; then
    echo "[LAUNCHER] source AT1 failed; DA tasks were not started"
    exit 1
fi

source_checkpoint="$OUTPUT_ROOT/source_AT1/fold_0/model.pt"
if [[ ! -f "$source_checkpoint" ]]; then
    echo "[LAUNCHER] [ERROR] source checkpoint missing: $source_checkpoint"
    exit 1
fi

run_da 1 DK1 "denmark/32VNH/2017" &
pid1="$!"
run_da 2 FR1 "france/30TXT/2017" &
pid2="$!"
run_da 3 FR2 "france/31TCJ/2017" &
pid3="$!"

launcher_status=0
if ! wait "$pid1"; then launcher_status=1; fi
if ! wait "$pid2"; then launcher_status=1; fi
if ! wait "$pid3"; then launcher_status=1; fi

failed_tasks=()
for failure_file in "$STATE_ROOT"/*.failed; do
    if [[ -f "$failure_file" ]]; then
        while IFS= read -r task; do
            if [[ -n "$task" ]]; then
                failed_tasks+=("$task")
            fi
        done < "$failure_file"
    fi
done

if [[ "${#failed_tasks[@]}" -gt 0 ]]; then
    launcher_status=1
    echo "[LAUNCHER] failed tasks:"
    printf '  %s\n' "${failed_tasks[@]}"
else
    echo "[LAUNCHER] source AT1 and all 3 open-set DA tasks completed"
fi

exit "$launcher_status"
