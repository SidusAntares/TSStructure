#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

SEED="${SEED:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-reimts_mtan_openset_timematch5_seed1}"

LOG_ROOT="logs/$EXPERIMENT_NAME"
OUTPUT_ROOT="outputs/$EXPERIMENT_NAME"
RUN_ROOT="runs/$EXPERIMENT_NAME"
STATE_ROOT="$LOG_ROOT/.launcher_state_$$"
mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$RUN_ROOT" "$STATE_ROOT"

SOURCE_ALIASES=("DK1" "FR1")
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

source_checkpoint() {
    local source_alias="$1"
    case "$source_alias" in
        DK1) printf '%s\n' "$OUTPUT_ROOT/source_DK1/fold_0/model.pt" ;;
        FR1) printf '%s\n' "$OUTPUT_ROOT/source_FR1/fold_0/model.pt" ;;
        *) return 1 ;;
    esac
}

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
    local source_alias="$2"
    local source
    source="$(domain_path "$source_alias")"
    local task="source_$source_alias"

    echo "[GPU$gpu_id] START source $source_alias"
    if ! run_task "$gpu_id" "$task" \
        "$PYTHON_BIN" -u train.py \
        --source "$source" \
        --target "$source" \
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
        echo "[GPU$gpu_id] FAILED source $source_alias"
        return 1
    fi
    echo "[GPU$gpu_id] DONE source $source_alias"
}

run_da() {
    local gpu_id="$1"
    local source_alias="$2"
    local target_alias="$3"
    local source target
    source="$(domain_path "$source_alias")"
    target="$(domain_path "$target_alias")"
    local task="${source_alias}_to_${target_alias}"

    echo "[GPU$gpu_id] START $task"
    if ! run_task "$gpu_id" "$task" \
        "$PYTHON_BIN" -u train.py \
        --source "$source" \
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
        --weights "$OUTPUT_ROOT/source_$source_alias" \
        --epochs 20 \
        --steps_per_epoch 500; then
        echo "$task" > "$STATE_ROOT/$task.failed"
        echo "[GPU$gpu_id] FAILED $task"
        return 1
    fi
    echo "[GPU$gpu_id] DONE $task"
}

run_da_worker() {
    local gpu_id="$1"
    shift
    local worker_failed=0
    local task_spec source_alias target_alias
    for task_spec in "$@"; do
        source_alias="${task_spec%%:*}"
        target_alias="${task_spec#*:}"
        if ! run_da "$gpu_id" "$source_alias" "$target_alias"; then
            worker_failed=1
        fi
    done
    return "$worker_failed"
}

run_source 1 DK1 &
source_pid1="$!"
run_source 2 FR1 &
source_pid2="$!"

source_status=0
if ! wait "$source_pid1"; then source_status=1; fi
if ! wait "$source_pid2"; then source_status=1; fi

for source_alias in "${SOURCE_ALIASES[@]}"; do
    checkpoint="$(source_checkpoint "$source_alias")"
    if [[ ! -f "$checkpoint" ]]; then
        echo "[LAUNCHER] [ERROR] source checkpoint missing: $checkpoint"
        source_status=1
    fi
done

if [[ "$source_status" -ne 0 ]]; then
    echo "[LAUNCHER] one or more source runs failed; DA tasks were not started"
    exit 1
fi

run_da_worker 1 "DK1:FR1" "FR1:DK1" &
da_pid1="$!"
run_da_worker 2 "DK1:FR2" "FR1:FR2" &
da_pid2="$!"
run_da_worker 3 "DK1:AT1" &
da_pid3="$!"

launcher_status=0
if ! wait "$da_pid1"; then launcher_status=1; fi
if ! wait "$da_pid2"; then launcher_status=1; fi
if ! wait "$da_pid3"; then launcher_status=1; fi

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
    echo "[LAUNCHER] both source runs and all 5 open-set DA tasks completed"
fi

exit "$launcher_status"
