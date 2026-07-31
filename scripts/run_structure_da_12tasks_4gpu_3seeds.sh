#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/user/TSStructure"
DATA_ROOT="/data/user/dataset/timematch_data"
CONDA_ENV="time"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-structure_eden_full_3seeds_v1}"

DATASETS=(
    "denmark/32VNH/2017"
    "france/30TXT/2017"
    "france/31TCJ/2017"
    "austria/33UVP/2017"
)
SHORT_NAMES=("DK1" "FR2" "FR1" "AT1")
SEEDS=(1 2 3)

cd "$REPO_ROOT"

LOG_DIR="logs/$EXPERIMENT_GROUP"
OUTPUT_ROOT="outputs/$EXPERIMENT_GROUP"
RUN_ROOT="runs/$EXPERIMENT_GROUP"
STATUS_DIR="$LOG_DIR/.launcher_status"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT" "$RUN_ROOT" "$STATUS_DIR"

echo "EXPERIMENT_GROUP|$EXPERIMENT_GROUP"
echo "LOG_DIR|$LOG_DIR"
echo "OUTPUT_ROOT|$OUTPUT_ROOT"

run_source_tasks() {
    local source_index="$1"
    local gpu_id="$2"
    local SOURCE="${DATASETS[$source_index]}"
    local SOURCE_SHORT="${SHORT_NAMES[$source_index]}"
    local status_file="$STATUS_DIR/gpu_${gpu_id}.status"
    local success_count=0
    local failed_count=0
    local skipped_count=0

    printf '0 0 0\n' > "$status_file"

    for seed in "${SEEDS[@]}"; do
        for target_index in "${!DATASETS[@]}"; do
            if [[ "$target_index" -eq "$source_index" ]]; then
                continue
            fi

            local TARGET="${DATASETS[$target_index]}"
            local TARGET_SHORT="${SHORT_NAMES[$target_index]}"
            local RUN_NAME="${SOURCE_SHORT}_to_${TARGET_SHORT}_seed${seed}"
            local RUN_OUTPUT="$OUTPUT_ROOT/$RUN_NAME"
            local TARGET_FILE_NAME="${TARGET//\//_}"
            local COMPLETION_FILE="$RUN_OUTPUT/fold_0/test_metrics_${TARGET_FILE_NAME}.json"
            local TASK_LOG="$LOG_DIR/${RUN_NAME}.log"

            if [[ -f "$COMPLETION_FILE" ]]; then
                echo "[SKIP][GPU $gpu_id] $RUN_NAME already completed"
                skipped_count=$((skipped_count + 1))
                continue
            fi

            if [[ -f "$TASK_LOG" ]]; then
                local archive_log="$LOG_DIR/${RUN_NAME}.previous_$(date +%Y%m%d_%H%M%S).log"
                if [[ -e "$archive_log" ]]; then
                    archive_log="${archive_log%.log}_${RANDOM}.log"
                fi
                mv -- "$TASK_LOG" "$archive_log"
            fi

            {
                echo "TASK_START|gpu=$gpu_id|source=$SOURCE_SHORT|target=$TARGET_SHORT|seed=$seed|run=$RUN_NAME"
                echo "DATE_START|$(date --iso-8601=seconds)"
            } > "$TASK_LOG"

            echo "[START][GPU $gpu_id] $RUN_NAME"
            if CUDA_VISIBLE_DEVICES="$gpu_id" \
                conda run -n "$CONDA_ENV" --no-capture-output \
                python -u train.py \
                    --data_root "$DATA_ROOT" \
                    --source "$SOURCE" \
                    --target "$TARGET" \
                    --seed "$seed" \
                    --device cuda \
                    --closed_set true \
                    --combine_spring_and_winter false \
                    --epochs 100 \
                    --batch_size 128 \
                    --num_pixels 64 \
                    --lr 1e-3 \
                    --weight_decay 1e-4 \
                    --channel_feature_dim 16 \
                    --pixel_hidden_dim 16 \
                    --structure_dim 128 \
                    --domain_hidden_dim 128 \
                    --grl_warmup_max_iters 250 \
                    --lambda_task 1 \
                    --lambda_geometry 1 \
                    --lambda_alignment 1 \
                    --lambda_structural_cls 1 \
                    --lambda_structural_domain 1 \
                    --lambda_component_cls 1 \
                    --lambda_component_domain 1 \
                    --time_scale 366 \
                    --tau_fast_init 0.05 \
                    --tau_slow_init 0.20 \
                    --tau_min 0.0001 \
                    --delta_tau_min 0.0001 \
                    --progress_bar off \
                    --output_dir "$OUTPUT_ROOT" \
                    --tensorboard_log_dir "$RUN_ROOT" \
                    --experiment_name "$RUN_NAME" \
                    >> "$TASK_LOG" 2>&1
            then
                {
                    echo "TASK_DONE|gpu=$gpu_id|source=$SOURCE_SHORT|target=$TARGET_SHORT|seed=$seed"
                    echo "DATE_END|$(date --iso-8601=seconds)"
                } >> "$TASK_LOG"
                echo "[DONE][GPU $gpu_id] $RUN_NAME"
                success_count=$((success_count + 1))
            else
                local exit_code=$?
                {
                    echo "TASK_FAILED|gpu=$gpu_id|source=$SOURCE_SHORT|target=$TARGET_SHORT|seed=$seed|exit_code=$exit_code"
                    echo "DATE_END|$(date --iso-8601=seconds)"
                } >> "$TASK_LOG"
                echo "[FAILED][GPU $gpu_id] $RUN_NAME (exit $exit_code)"
                failed_count=$((failed_count + 1))
            fi
        done
    done

    printf '%s %s %s\n' \
        "$success_count" "$failed_count" "$skipped_count" > "$status_file"
}

pids=()
for source_index in 0 1 2 3; do
    run_source_tasks "$source_index" "$source_index" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid" || true
done

success_total=0
failed_total=0
skipped_total=0
for gpu_id in 0 1 2 3; do
    worker_success=0
    worker_failed=0
    worker_skipped=0
    read -r worker_success worker_failed worker_skipped \
        < "$STATUS_DIR/gpu_${gpu_id}.status"
    success_total=$((success_total + worker_success))
    failed_total=$((failed_total + worker_failed))
    skipped_total=$((skipped_total + worker_skipped))
done

echo "EXPERIMENT_SUMMARY|success=$success_total|failed=$failed_total|skipped=$skipped_total"
