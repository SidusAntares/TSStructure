#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/structure_proto_v1}"
RUN_ROOT="${RUN_ROOT:-runs/structure_proto_v1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_proto_v1}"
SOURCE_EXPERIMENT="structure_proto_source_AT1_seed1"
EXPERIMENT="structure_proto_AT1_DK1_seed1"
SOURCE_WEIGHTS="$OUTPUT_ROOT/$SOURCE_EXPERIMENT"

mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1
if [[ ! -f "$SOURCE_WEIGHTS/fold_0/model.pt" ]]; then
  echo "ERROR: source checkpoint not found: $SOURCE_WEIGHTS/fold_0/model.pt"
  exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u train.py \
  -e "$EXPERIMENT" \
  --data_root "$DATA_ROOT" \
  --source austria/33UVP/2017 \
  --target denmark/32VNH/2017 \
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
  --seed 1 --num_folds 1 --batch_size 128 \
  --seq_length 30 --num_pixels 64 --closed_set true \
  --with_shift_aug false \
  --progress_bar off \
  --output_dir "$OUTPUT_ROOT" \
  --tensorboard_log_dir "$RUN_ROOT" \
  timematch \
  --weights "$SOURCE_WEIGHTS" \
  --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
  --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
  --estimate_shift true --balance_source true --use_focal_loss true \
  --shift_source true --sample_size 100 --max_temporal_shift 60 \
  --domain_specific_bn true --shift_estimator AM --run_validation \
  --output_student true \
  2>&1 | tee "$LOG_ROOT/AT1_DK1_seed1.log"
