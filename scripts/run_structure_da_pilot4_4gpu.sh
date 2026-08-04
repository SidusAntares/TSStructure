#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/user/TSStructure"
CONDA_ENV="time"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-structure_eden_pilot4_100ep_v1}"

TASKS=(
    "0|AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|1"
    "1|AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|2"
    "2|AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|3"
    "3|DK1|denmark/32VNH/2017|FR2|france/30TXT/2017|1"
)

cd "$REPO_ROOT"

LOG_DIR="logs/$EXPERIMENT_GROUP"
OUTPUT_ROOT="outputs/$EXPERIMENT_GROUP"
RUN_ROOT="runs/$EXPERIMENT_GROUP"
STATUS_DIR="$LOG_DIR/.launcher_status"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT" "$RUN_ROOT" "$STATUS_DIR"

echo "EXPERIMENT_GROUP|$EXPERIMENT_GROUP"
echo "LOG_DIR|$LOG_DIR"
echo "OUTPUT_ROOT|$OUTPUT_ROOT"

run_task() {
    local specification="$1"
    local gpu_id source_short source target_short target seed
    IFS='|' read -r gpu_id source_short source target_short target seed \
        <<< "$specification"
    local run_name="${source_short}_to_${target_short}_seed${seed}"
    local run_output="$OUTPUT_ROOT/$run_name"
    local target_file_name="${target//\//_}"
    local completion_file="$run_output/fold_0/test_metrics_${target_file_name}.json"
    local task_log="$LOG_DIR/${run_name}.log"
    local status_file="$STATUS_DIR/gpu_${gpu_id}.status"

    if [[ -f "$completion_file" ]]; then
        echo "[SKIP][GPU $gpu_id] $run_name already completed"
        printf '0 0 1\n' > "$status_file"
        return
    fi

    if [[ -f "$task_log" ]]; then
        local archive_log="$LOG_DIR/${run_name}.previous_$(date +%Y%m%d_%H%M%S).log"
        if [[ -e "$archive_log" ]]; then
            archive_log="${archive_log%.log}_${RANDOM}.log"
        fi
        mv -- "$task_log" "$archive_log"
    fi

    {
        echo "TASK_START|gpu=$gpu_id|source=$source_short|target=$target_short|seed=$seed|run=$run_name"
        echo "DATE_START|$(date --iso-8601=seconds)"
    } > "$task_log"

    echo "[START][GPU $gpu_id] $run_name"
    if CUDA_VISIBLE_DEVICES="$gpu_id" \
        conda run -n "$CONDA_ENV" --no-capture-output \
        python -u train.py \
            --source "$source" \
            --target "$target" \
            --seed "$seed" \
            --device cuda \
            --epochs 100 \
            --batch_size 128 \
            --eval_batch_size 128 \
            --num_pixels 64 \
            --num_workers 8 \
            --lr 0.001 \
            --weight_decay 0.0001 \
            --structure_dim 128 \
            --domain_hidden_dim 128 \
            --grl_warmup_fraction 0.2 \
            --amp true \
            --amp_dtype float16 \
            --lambda_task 1 \
            --lambda_geometry 0.1 \
            --lambda_alignment 1 \
            --lambda_structural_cls 0.25 \
            --lambda_structural_domain 0.25 \
            --lambda_component_cls 0.25 \
            --lambda_component_domain 0.25 \
            --time_scale 365 \
            --tau_fast_init 0.05 \
            --tau_slow_init 0.20 \
            --tau_min 0.0001 \
            --delta_tau_min 0.0001 \
            --closed_set true \
            --balance-source \
            --combine_spring_and_winter false \
            --progress_bar off \
            --log_step 25 \
            --output_dir "$OUTPUT_ROOT" \
            --tensorboard_log_dir "$RUN_ROOT" \
            --experiment_name "$run_name" \
            >> "$task_log" 2>&1
    then
        {
            echo "TASK_DONE|gpu=$gpu_id|source=$source_short|target=$target_short|seed=$seed"
            echo "DATE_END|$(date --iso-8601=seconds)"
        } >> "$task_log"
        printf '1 0 0\n' > "$status_file"
        echo "[DONE][GPU $gpu_id] $run_name"
    else
        local exit_code=$?
        {
            echo "TASK_FAILED|gpu=$gpu_id|source=$source_short|target=$target_short|seed=$seed|exit_code=$exit_code"
            echo "DATE_END|$(date --iso-8601=seconds)"
        } >> "$task_log"
        printf '0 1 0\n' > "$status_file"
        echo "[FAILED][GPU $gpu_id] $run_name (exit $exit_code)"
    fi
}

pids=()
for task in "${TASKS[@]}"; do
    run_task "$task" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid" || true
done

success_total=0
failed_total=0
skipped_total=0
for gpu_id in 0 1 2 3; do
    read -r success failed skipped < "$STATUS_DIR/gpu_${gpu_id}.status"
    success_total=$((success_total + success))
    failed_total=$((failed_total + failed))
    skipped_total=$((skipped_total + skipped))
done

echo "EXPERIMENT_SUMMARY|success=$success_total|failed=$failed_total|skipped=$skipped_total"
