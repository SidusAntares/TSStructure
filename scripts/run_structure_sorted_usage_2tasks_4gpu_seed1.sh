#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
EXP_ROOT="${EXP_ROOT:-outputs/structure_sorted_usage_2tasks_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_sorted_usage_2tasks_seed1}"
RUN_ROOT="${RUN_ROOT:-runs/structure_sorted_usage_2tasks_seed1}"
DRY_RUN="${DRY_RUN:-0}"

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR2="france/31TCJ/2017"

mkdir -p "$LOG_ROOT" "$EXP_ROOT" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

run_task() {
  local gpu="$1" injection="$2" src_name="$3" src_data="$4" tgt_name="$5" tgt_data="$6"
  local task="${src_name}_${tgt_name}"
  local worker_root="$EXP_ROOT/$injection/$task"
  local source_experiment="source_${src_name}_${injection}_seed1"
  local source_weights="$worker_root/source/$source_experiment"
  local uda_experiment="${task}_${injection}_seed1"
  local log_dir="$LOG_ROOT/$injection"
  local log_file="$log_dir/${task}.log"
  local run_dir="$RUN_ROOT/$injection/$task"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "STRUCTURE_USAGE_PLAN|gpu=$gpu|representation=sorted_profile|injection=$injection|task=$task|shape_align_weight=0"
    return 0
  fi

  mkdir -p "$log_dir" "$worker_root/source" "$worker_root/uda" "$run_dir"
  : > "$log_file"
  echo "STRUCTURE_USAGE_CONFIG|representation=sorted_profile|injection=$injection|evidence_dim=128|shape_align_weight=0" | tee -a "$log_file"
  echo "[SORTED SOURCE START] $src_name $injection GPU$gpu" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "$source_experiment" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$src_data" \
    --model psestructureprotoltae \
    --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shape-representation sorted_profile --shape-injection "$injection" \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0.0 \
    --with_shift_aug false --seed 1 --num_folds 1 --epochs 100 \
    --batch_size 128 --lr 0.001 --weight_decay 0.0001 \
    --focal_loss_gamma 1.0 --seq_length 30 --num_pixels 64 \
    --closed_set true --progress_bar off \
    --output_dir "$worker_root/source" \
    --tensorboard_log_dir "$run_dir/source" \
    2>&1 | tee -a "$log_file"

  if [[ ! -f "$source_weights/fold_0/model.pt" ]]; then
    echo "ERROR: sorted source checkpoint not found: $source_weights/fold_0/model.pt" | tee -a "$log_file"
    return 1
  fi

  echo "[SORTED UDA START] $src_name -> $tgt_name $injection GPU$gpu" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "$uda_experiment" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae \
    --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shape-representation sorted_profile --shape-injection "$injection" \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0.0 \
    --with_shift_aug false --seed 1 --num_folds 1 --batch_size 128 \
    --seq_length 30 --num_pixels 64 --closed_set true --progress_bar off \
    --output_dir "$worker_root/uda" \
    --tensorboard_log_dir "$run_dir/uda" \
    timematch --weights "$source_weights" \
    --shape-da-mode batch_align \
    --oracle-pseudo-labels false --adaptive-pseudo-selection false \
    --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true \
    --shift_source true --sample_size 100 --max_temporal_shift 60 \
    --domain_specific_bn true --shift_estimator AM --run_validation \
    --output_student true \
    2>&1 | tee -a "$log_file"
  echo "[FINISHED] $src_name -> $tgt_name $injection" | tee -a "$log_file"
}

run_task "$GPU0" direct_query FR2 "$FR2" DK1 "$DK1" & PID0=$!
run_task "$GPU1" direct_query AT1 "$AT1" DK1 "$DK1" & PID1=$!
run_task "$GPU2" late_fusion FR2 "$FR2" DK1 "$DK1" & PID2=$!
run_task "$GPU3" late_fusion AT1 "$AT1" DK1 "$DK1" & PID3=$!

status=0
wait "$PID0" || status=1
wait "$PID1" || status=1
wait "$PID2" || status=1
wait "$PID3" || status=1
exit "$status"
