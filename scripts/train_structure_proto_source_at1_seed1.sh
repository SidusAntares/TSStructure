#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/structure_proto_v1}"
RUN_ROOT="${RUN_ROOT:-runs/structure_proto_v1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_proto_v1}"
EXPERIMENT="structure_proto_source_AT1_seed1"

mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u train.py \
  -e "$EXPERIMENT" \
  --data_root "$DATA_ROOT" \
  --source austria/33UVP/2017 \
  --target austria/33UVP/2017 \
  --model psestructureprotoltae \
  --structure-branch true \
  --structure-exposer fourier \
  --fourier_num_modes 13 \
  --shape-dim 128 \
  --shape-window-scales 8 16 24 \
  --shape-window-stride 4 \
  --shapelet-count 16 --shapelet-beta 5 \
  --shape-resample-length 16 \
  --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
  --shapelet-shaping-weight 0.01 --shapelet-shaping-temperature 0.1 \
  --proto-momentum 0.9 \
  --proto-temperature 0.1 \
  --proto-instance-weight 0.1 \
  --proto-init-epoch 1 \
  --proto-ramp-start 0.1 \
  --proto-ramp-epochs 5 \
  --with_shift_aug false \
  --seed 1 --num_folds 1 --epochs 100 --batch_size 128 \
  --lr 0.001 --weight_decay 0.0001 --focal_loss_gamma 1.0 \
  --seq_length 30 --num_pixels 64 --closed_set true \
  --progress_bar off \
  --output_dir "$OUTPUT_ROOT" \
  --tensorboard_log_dir "$RUN_ROOT" \
  2>&1 | tee "$LOG_ROOT/source_AT1_seed1.log"
