#!/bin/bash

set -euo pipefail

RUN_NAME=timematch_red_x_5tasks_50ep_4gpu
OUTPUT_ROOT=outputs/$RUN_NAME
TB_ROOT=runs/$RUN_NAME
LOG_ROOT=logs/$RUN_NAME

mkdir -p "$LOG_ROOT"

run_source_model() {
    local gpu_id=$1
    local source_model=$2
    local source_domain=$3

    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        -e "$source_model" \
        --source "$source_domain" \
        --target "$source_domain" \
        --output_dir "$OUTPUT_ROOT" \
        --tensorboard_log_dir "$TB_ROOT" \
        --with_shift_aug false \
        --epochs 50
}

run_task() {
    local gpu_id=$1
    local source_tile=$2
    local target_tile=$3
    local source_domain=$4
    local target_domain=$5

    local source_model="pseltae_${source_tile}"
    local task_name="timematch_${source_tile}_to_${target_tile}"

    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        -e "$source_model" \
        --source "$source_domain" \
        --target "$target_domain" \
        --output_dir "$OUTPUT_ROOT" \
        --tensorboard_log_dir "$TB_ROOT" \
        --with_shift_aug false \
        --eval

    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        -e "$task_name" \
        --source "$source_domain" \
        --target "$target_domain" \
        --output_dir "$OUTPUT_ROOT" \
        --tensorboard_log_dir "$TB_ROOT" \
        --with_shift_aug false \
        timematch \
        --weights "$OUTPUT_ROOT/$source_model" \
        --epochs 50
}

echo "[Phase 1] Train red-cross source models"
run_source_model 0 pseltae_32VNH denmark/32VNH/2017 > "$LOG_ROOT/source_32VNH_gpu0.log" 2>&1 &
run_source_model 1 pseltae_30TXT france/30TXT/2017 > "$LOG_ROOT/source_30TXT_gpu1.log" 2>&1 &
wait

echo "[Phase 2] Launch 5 TimeMatch tasks on 4 GPUs"
(
    run_task 0 32VNH 30TXT denmark/32VNH/2017 france/30TXT/2017
) > "$LOG_ROOT/timematch_32VNH_to_30TXT_gpu0.log" 2>&1 &

(
    run_task 1 32VNH 31TCJ denmark/32VNH/2017 france/31TCJ/2017
) > "$LOG_ROOT/timematch_32VNH_to_31TCJ_gpu1.log" 2>&1 &

(
    run_task 2 32VNH 33UVP denmark/32VNH/2017 austria/33UVP/2017
) > "$LOG_ROOT/timematch_32VNH_to_33UVP_gpu2.log" 2>&1 &

(
    run_task 3 30TXT 32VNH france/30TXT/2017 denmark/32VNH/2017
    run_task 3 30TXT 31TCJ france/30TXT/2017 france/31TCJ/2017
) > "$LOG_ROOT/timematch_30TXT_gpu3.log" 2>&1 &

wait

echo "All 5 tasks finished. Outputs: $OUTPUT_ROOT"
echo "Logs: $LOG_ROOT"
