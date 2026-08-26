#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SKIP_SOURCE="${SKIP_SOURCE:-0}"
export PYTHONUNBUFFERED=1

SOURCE="austria/33UVP/2017"
RUN_NAME="fredn_reim_at1_3tasks_seed1"
SOURCE_EXPERIMENT="fredn_reim_AT1_source_seed1"
OUTPUT_ROOT="outputs/${RUN_NAME}"
TENSORBOARD_ROOT="runs/${RUN_NAME}"
TASK_LOG_ROOT="logs/${RUN_NAME}"
SOURCE_WEIGHTS_DIR="${OUTPUT_ROOT}/source/${SOURCE_EXPERIMENT}"
SOURCE_CKPT="${SOURCE_WEIGHTS_DIR}/fold_0/model.pt"

mkdir -p logs outputs runs "$TASK_LOG_ROOT"

SOURCE_LOG="${TASK_LOG_ROOT}/source_AT1.log"
if [[ "$SKIP_SOURCE" == "1" ]]; then
    printf '%s\n' \
        '============================================================' \
        '[STAGE 0/3] SOURCE PRETRAIN SKIPPED' \
        'AT1 -> AT1' \
        "SOURCE CHECKPOINT: ${SOURCE_CKPT}" \
        '============================================================' \
        | tee -a "$SOURCE_LOG"
else
    printf '%s\n' \
        '============================================================' \
        '[STAGE 0/3] SOURCE PRETRAIN' \
        'AT1 -> AT1' \
        '============================================================' \
        | tee -a "$SOURCE_LOG"

    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u train.py \
        -e "$SOURCE_EXPERIMENT" \
        --data_root "$DATA_ROOT" \
        --source "$SOURCE" \
        --target "$SOURCE" \
        --model psefrednltae \
        --fredn_fourier_solver dense_direct \
        --seed 1 \
        --num_folds 1 \
        --seq_length 30 \
        --device cuda \
        --closed_set true \
        --combine_spring_and_winter false \
        --with_shift_aug false \
        --epochs 100 \
        --progress_bar off \
        --output_dir "${OUTPUT_ROOT}/source" \
        --tensorboard_log_dir "${TENSORBOARD_ROOT}/source" \
        2>&1 | tee -a "$SOURCE_LOG"

    printf '%s\n' '[FINISHED] AT1 -> AT1 SOURCE PRETRAIN' | tee -a "$SOURCE_LOG"
fi

if [[ ! -f "$SOURCE_CKPT" ]]; then
    echo "ERROR: source checkpoint not found: $SOURCE_CKPT" >&2
    exit 1
fi

run_timematch_task() {
    local stage="$1"
    local target_alias="$2"
    local target="$3"
    local experiment="fredn_reim_AT1_${target_alias}_timematch_seed1"
    local task_log="${TASK_LOG_ROOT}/AT1_${target_alias}_timematch.log"

    printf '%s\n' \
        '============================================================' \
        "[STAGE ${stage}/3] TIMEMATCH DOMAIN ADAPTATION" \
        "AT1 -> ${target_alias}" \
        "SOURCE CHECKPOINT: ${SOURCE_CKPT}" \
        '============================================================' \
        | tee -a "$task_log"

    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u train.py \
        -e "$experiment" \
        --data_root "$DATA_ROOT" \
        --source "$SOURCE" \
        --target "$target" \
        --model psefrednltae \
        --fredn_fourier_solver dense_direct \
        --seed 1 \
        --num_folds 1 \
        --seq_length 30 \
        --device cuda \
        --closed_set true \
        --combine_spring_and_winter false \
        --with_shift_aug false \
        --progress_bar off \
        --output_dir "${OUTPUT_ROOT}/AT1_${target_alias}" \
        --tensorboard_log_dir "${TENSORBOARD_ROOT}/AT1_${target_alias}" \
        timematch \
        --weights "$SOURCE_WEIGHTS_DIR" \
        --epochs 20 \
        --steps_per_epoch 500 \
        2>&1 | tee -a "$task_log"

    printf '%s\n' "[FINISHED] AT1 -> ${target_alias}" | tee -a "$task_log"
}

run_timematch_task 1 DK1 denmark/32VNH/2017
run_timematch_task 2 FR1 france/30TXT/2017
run_timematch_task 3 FR2 france/31TCJ/2017

printf '%s\n' \
    '============================================================' \
    'ALL FREDN AT1 EXPERIMENTS FINISHED' \
    '============================================================'
