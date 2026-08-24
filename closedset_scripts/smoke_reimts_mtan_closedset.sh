#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
DEVICE="${DEVICE:-cuda}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-outputs/reimts_mtan_smoke}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-1024}"

SOURCE="denmark/32VNH/2017"
TARGET="france/30TXT/2017"
SOURCE_EXPERIMENT="closedset_source_32VNH_reimts_mtan_smoke"
TIMEMATCH_EXPERIMENT="closedset_timematch_32VNH_to_30TXT_reimts_mtan_smoke"

"$PYTHON_BIN" -u train.py \
    --data_root "$DATA_ROOT" \
    --source "$SOURCE" \
    --target "$SOURCE" \
    --seed 1 \
    --device "$DEVICE" \
    --closed_set true \
    --combine_spring_and_winter false \
    --with_shift_aug false \
    --model psereimtsmtanltae \
    --reimts_levels 3 \
    --reimts_scale_factor 2 \
    --reimts_period 365 \
    --mtan_num_ref_points 8 \
    --reimts_loss_mode sample \
    --reimts_patch_diagnostics true \
    --num_folds 1 \
    --epochs 1 \
    --batch_size "$SMOKE_BATCH_SIZE" \
    --num_workers 0 \
    --output_dir "$SMOKE_OUTPUT_ROOT" \
    --experiment_name "$SOURCE_EXPERIMENT"

"$PYTHON_BIN" -u train.py \
    --data_root "$DATA_ROOT" \
    --source "$SOURCE" \
    --target "$TARGET" \
    --seed 1 \
    --device "$DEVICE" \
    --closed_set true \
    --combine_spring_and_winter false \
    --with_shift_aug false \
    --model psereimtsmtanltae \
    --reimts_levels 3 \
    --reimts_scale_factor 2 \
    --reimts_period 365 \
    --mtan_num_ref_points 8 \
    --reimts_loss_mode sample \
    --reimts_patch_diagnostics true \
    --num_folds 1 \
    --batch_size 4 \
    --num_workers 0 \
    --progress_bar off \
    --output_dir "$SMOKE_OUTPUT_ROOT" \
    --experiment_name "$TIMEMATCH_EXPERIMENT" \
    timematch \
    --weights "$SMOKE_OUTPUT_ROOT/$SOURCE_EXPERIMENT" \
    --epochs 1 \
    --steps_per_epoch 2 \
    --sample_size 1
