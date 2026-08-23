#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

SEED="${SEED:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-reimts_mtan_closedset_12tasks_seed1}"

LOG_ROOT="logs/$EXPERIMENT_NAME"
OUTPUT_ROOT="outputs/$EXPERIMENT_NAME"
RUN_ROOT="runs/$EXPERIMENT_NAME"
STATE_ROOT="$LOG_ROOT/.launcher_state"
mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$RUN_ROOT" "$STATE_ROOT"

DOMAIN_ALIASES=("DK1" "FR1" "FR2" "AT1")
DOMAIN_PATHS=(
    "denmark/32VNH/2017"
    "france/30TXT/2017"
    "france/31TCJ/2017"
    "austria/33UVP/2017"
)

domain_path() {
    local requested="$1"
    local index
    for index in "${!DOMAIN_ALIASES[@]}"; do
        if [[ "${DOMAIN_ALIASES[$index]}" == "$requested" ]]; then
            printf '%s\n' "${DOMAIN_PATHS[$index]}"
            return 0
        fi
    done
    return 1
}

format_shell_duration() {
    local total="$1"
    printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
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

run_source_worker() {
    local gpu_id="$1"
    local source_alias="$2"
    local source
    source="$(domain_path "$source_alias")"
    local failure_file="$STATE_ROOT/worker_${gpu_id}.failed"
    local worker_failed=0
    local failed_tasks=()
    : > "$failure_file"

    local source_task="source_$source_alias"
    echo "[GPU$gpu_id] START source $source_alias"
    if ! run_task "$gpu_id" "$source_task" \
        "$PYTHON_BIN" -u train.py \
        --source "$source" \
        --target "$source" \
        --seed "$SEED" \
        --device cuda \
        --closed_set true \
        --combine_spring_and_winter false \
        --with_shift_aug false \
        --model psereimtsmtanltae \
        --reimts_levels 3 \
        --reimts_scale_factor 2 \
        --reimts_period 365 \
        --mtan_num_ref_points 8 \
        --mtan_latent_dim 128 \
        --mtan_heads 1 \
        --reimts_loss_mode patch \
        --num_folds 1 \
        --progress_bar off \
        --epochs 100 \
        --output_dir "$OUTPUT_ROOT" \
        --tensorboard_log_dir "$RUN_ROOT" \
        --experiment_name "$source_task"; then
        echo "$source_task" >> "$failure_file"
        echo "[GPU$gpu_id] FAILED source $source_alias"
        return 1
    fi

    local source_checkpoint="$OUTPUT_ROOT/source_$source_alias/fold_0/model.pt"
    if [[ ! -f "$source_checkpoint" ]]; then
        echo "$source_task" >> "$failure_file"
        echo "[GPU$gpu_id] [ERROR] source checkpoint missing: $source_checkpoint"
        return 1
    fi
    echo "[GPU$gpu_id] DONE source $source_alias"

    local target_alias target task
    for target_alias in "${DOMAIN_ALIASES[@]}"; do
        if [[ "$target_alias" == "$source_alias" ]]; then
            continue
        fi
        target="$(domain_path "$target_alias")"
        task="${source_alias}_to_${target_alias}"
        echo "[GPU$gpu_id] START $task"
        if ! run_task "$gpu_id" "$task" \
            "$PYTHON_BIN" -u train.py \
            --source "$source" \
            --target "$target" \
            --seed "$SEED" \
            --device cuda \
            --closed_set true \
            --combine_spring_and_winter false \
            --with_shift_aug false \
            --model psereimtsmtanltae \
            --reimts_levels 3 \
            --reimts_scale_factor 2 \
            --reimts_period 365 \
            --mtan_num_ref_points 8 \
            --mtan_latent_dim 128 \
            --mtan_heads 1 \
            --reimts_loss_mode patch \
            --num_folds 1 \
            --progress_bar off \
            --output_dir "$OUTPUT_ROOT" \
            --tensorboard_log_dir "$RUN_ROOT" \
            --experiment_name "$task" \
            timematch \
            --weights "$OUTPUT_ROOT/source_$source_alias" \
            --epochs 20 \
            --steps_per_epoch 500; then
            failed_tasks+=("$task")
            echo "$task" >> "$failure_file"
            echo "[GPU$gpu_id] FAILED $task"
            worker_failed=1
            continue
        fi
        echo "[GPU$gpu_id] DONE $task"
    done
    return "$worker_failed"
}

run_source_worker 0 DK1 &
pid0="$!"
run_source_worker 1 FR1 &
pid1="$!"
run_source_worker 2 FR2 &
pid2="$!"
run_source_worker 3 AT1 &
pid3="$!"

launcher_status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
    if ! wait "$pid"; then
        launcher_status=1
    fi
done

failed_tasks=()
for failure_file in "$STATE_ROOT"/*.failed; do
    while IFS= read -r task; do
        if [[ -n "$task" ]]; then
            failed_tasks+=("$task")
        fi
    done < "$failure_file"
done

if [[ "${#failed_tasks[@]}" -gt 0 ]]; then
    launcher_status=1
    echo "[LAUNCHER] failed tasks:"
    printf '  %s\n' "${failed_tasks[@]}"
else
    echo "[LAUNCHER] all 4 source runs and 12 DA tasks completed"
fi

exit "$launcher_status"
