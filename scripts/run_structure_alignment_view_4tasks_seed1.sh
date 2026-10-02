#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/structure_proto_v2clean_4tasks_seed1/source}"
EXP_ROOT="${EXP_ROOT:-outputs/structure_alignment_view_4tasks_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_alignment_view_4tasks_seed1}"
RUN_ROOT="${RUN_ROOT:-runs/structure_alignment_view_4tasks_seed1}"
EPOCHS="${EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"

mkdir -p "$LOG_ROOT" "$EXP_ROOT" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

echo "ALIGNMENT_VIEW_PLAN|views=strength,concentration,none|full_reused=true|source_retrained=false|target_gt_used=false"

run_task() {
  local gpu="$1" src_name="$2" src_data="$3" tgt_name="$4" tgt_data="$5" view="$6"
  local task="${src_name}_${tgt_name}"
  local experiment="${view}_${task}_seed1"
  local source_weights="$SOURCE_ROOT/source_${src_name}_seed1"
  local log_file="$LOG_ROOT/${view}_${task}.log"
  local alignment_dim=16
  if [[ "$view" == "none" ]]; then
    alignment_dim=0
  fi

  if [[ ! -f "$source_weights/fold_0/model.pt" ]]; then
    echo "MISSING|$source_weights/fold_0/model.pt" >&2
    return 1
  fi

  : > "$log_file"
  echo "ALIGNMENT_VIEW_START|gpu=$gpu|task=$task|view=$view|alignment_feature_dim=$alignment_dim|source_retrained=false|shape_representation=current|shape_injection=current_query|shape_da_mode=batch_align|shape_alignment_label_source=pseudo|target_gt_used=false" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "$experiment" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae \
    --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0.05 \
    --shape-representation current --shape-injection current_query \
    --source-minority-mode base --with_shift_aug false \
    --seed 1 --num_folds 1 --batch_size 128 --weight_decay 0.0001 \
    --seq_length 30 --num_pixels 64 --closed_set true --progress_bar off \
    --output_dir "$EXP_ROOT/$view" \
    --tensorboard_log_dir "$RUN_ROOT/$view/$task" \
    timematch --weights "$source_weights" \
    --shape-da-mode batch_align --shape-alignment-view "$view" \
    --shape-alignment-label-source pseudo \
    --oracle-pseudo-labels false --adaptive-pseudo-selection false \
    --epochs "$EPOCHS" --steps_per_epoch "$STEPS_PER_EPOCH" --lr 0.0001 \
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true \
    --shift_source true --sample_size "$SAMPLE_SIZE" --max_temporal_shift 60 \
    --domain_specific_bn true --shift_estimator AM --run_validation \
    --output_student true \
    2>&1 | tee -a "$log_file"
  echo "ALIGNMENT_VIEW_FINISHED|task=$task|view=$view" | tee -a "$log_file"
}

for view in strength concentration none; do
  echo "ALIGNMENT_VIEW_ROUND_START|view=$view"
  run_task "$GPU0" AT1 "$AT1" DK1 "$DK1" "$view" & PID0=$!
  run_task "$GPU1" FR1 "$FR1" FR2 "$FR2" "$view" & PID1=$!
  run_task "$GPU2" FR2 "$FR2" DK1 "$DK1" "$view" & PID2=$!
  run_task "$GPU3" DK1 "$DK1" AT1 "$AT1" "$view" & PID3=$!

  status=0
  wait "$PID0" || status=1
  wait "$PID1" || status=1
  wait "$PID2" || status=1
  wait "$PID3" || status=1
  if [[ "$status" -ne 0 ]]; then
    echo "ERROR: alignment-view round failed: $view" >&2
    exit "$status"
  fi
  echo "ALIGNMENT_VIEW_ROUND_FINISHED|view=$view"
done
