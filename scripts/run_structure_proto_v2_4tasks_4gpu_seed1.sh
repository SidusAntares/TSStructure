#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
EXP_ROOT="${EXP_ROOT:-outputs/structure_proto_v2_4tasks_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_proto_v2_4tasks_seed1}"
RUN_ROOT="${RUN_ROOT:-runs/structure_proto_v2_4tasks_seed1}"

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"

mkdir -p "$LOG_ROOT" "$EXP_ROOT/source" "$EXP_ROOT/uda" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

run_task() {
  local gpu="$1" src_name="$2" src_data="$3" tgt_name="$4" tgt_data="$5"
  local task="${src_name}_${tgt_name}"
  local source_experiment="source_${src_name}_seed1"
  local uda_experiment="${task}_seed1"
  local source_weights="$EXP_ROOT/source/$source_experiment"
  local log_file="$LOG_ROOT/${task}.log"

  : > "$log_file"
  echo "[SOURCE START] $src_name GPU$gpu" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" python -u train.py \
    -e "$source_experiment" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$src_data" \
    --model psestructureprotoltae \
    --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shapelet-shaping-weight 0.01 --shapelet-shaping-temperature 0.1 \
    --shape-class-weight 0.1 --shape-target-weight 0.05 \
    --shape-align-weight 0.05 --stats-align-weight 0.02 \
    --proto-momentum 0.9 --proto-temperature 0.1 \
    --proto-instance-weight 0.1 \
    --proto-init-epoch 1 --proto-ramp-start 0.1 --proto-ramp-epochs 5 \
    --with_shift_aug false \
    --seed 1 --num_folds 1 --epochs 100 --batch_size 128 \
    --lr 0.001 --weight_decay 0.0001 --focal_loss_gamma 1.0 \
    --seq_length 30 --num_pixels 64 --closed_set true --progress_bar off \
    --output_dir "$EXP_ROOT/source" \
    --tensorboard_log_dir "$RUN_ROOT/source_${src_name}_seed1" \
    2>&1 | tee -a "$log_file"

  if [[ ! -f "$source_weights/fold_0/model.pt" ]]; then
    echo "ERROR: source checkpoint not found: $source_weights/fold_0/model.pt" | tee -a "$log_file"
    return 1
  fi

  echo "[UDA START] $src_name -> $tgt_name GPU$gpu" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" python -u train.py \
    -e "$uda_experiment" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae \
    --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shapelet-shaping-weight 0.01 --shapelet-shaping-temperature 0.1 \
    --shape-class-weight 0.1 --shape-target-weight 0.05 \
    --shape-align-weight 0.05 --stats-align-weight 0.02 \
    --proto-momentum 0.9 --proto-temperature 0.1 \
    --proto-instance-weight 0.1 \
    --proto-init-epoch 1 --proto-ramp-start 0.1 --proto-ramp-epochs 5 \
    --seed 1 --num_folds 1 --batch_size 128 \
    --seq_length 30 --num_pixels 64 --closed_set true \
    --with_shift_aug false --progress_bar off \
    --output_dir "$EXP_ROOT/uda" \
    --tensorboard_log_dir "$RUN_ROOT/${task}_seed1" \
    timematch --weights "$source_weights" \
    --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true \
    --shift_source true --sample_size 100 --max_temporal_shift 60 \
    --domain_specific_bn true --shift_estimator AM --run_validation \
    --output_student true \
    2>&1 | tee -a "$log_file"
  echo "[FINISHED] $src_name -> $tgt_name" | tee -a "$log_file"
}

run_task "$GPU0" AT1 "$AT1" DK1 "$DK1" & PID0=$!
run_task "$GPU1" FR1 "$FR1" FR2 "$FR2" & PID1=$!
run_task "$GPU2" FR2 "$FR2" DK1 "$DK1" & PID2=$!
run_task "$GPU3" DK1 "$DK1" AT1 "$AT1" & PID3=$!

status=0
wait "$PID0" || status=1
wait "$PID1" || status=1
wait "$PID2" || status=1
wait "$PID3" || status=1
exit "$status"
