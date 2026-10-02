#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/structure_proto_v2clean_4tasks_seed1/source}"
EXP_ROOT="${EXP_ROOT:-outputs/structure_local_support_4tasks_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_local_support_4tasks_seed1}"
RUN_ROOT="${RUN_ROOT:-runs/structure_local_support_4tasks_seed1}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"

mkdir -p "$LOG_ROOT" "$EXP_ROOT" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

run_task() {
  local gpu="$1" src_name="$2" src_data="$3" tgt_name="$4" tgt_data="$5"
  local task="${src_name}_${tgt_name}"
  local source_weights="$SOURCE_ROOT/source_${src_name}_seed1"
  local checkpoint="$source_weights/fold_0/model.pt"
  local log_file="$LOG_ROOT/${task}.log"

  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: V2-Clean source checkpoint not found: $checkpoint" >&2
    return 1
  fi
  : > "$log_file"
  echo "[LOCAL SUPPORT START] $src_name -> $tgt_name GPU$gpu" | tee -a "$log_file"
  echo "SHAPE_DA_MODE=local_support|shape_representation=current|shape_injection=current_query|source_retrained=false|source_bank_refresh=every_epoch|source_bank_fixed_within_epoch=true|local_support_k=5|local_support_temperature=0.1|target_gt_used=false|batch_align=false|source_prototype=false|boundary_support=false" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" python -u train.py \
    -e "local_support_${task}_seed1" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae \
    --structure-branch true --structure-exposer fourier \
    --shape-representation current --shape-injection current_query \
    --source-minority-mode base \
    --fourier_num_modes 13 --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0.05 \
    --seed 1 --num_folds 1 --batch_size 128 \
    --weight_decay 0.0001 --focal_loss_gamma 1.0 \
    --seq_length 30 --num_pixels 64 --closed_set true \
    --with_shift_aug false --progress_bar off \
    --output_dir "$EXP_ROOT" \
    --tensorboard_log_dir "$RUN_ROOT/${task}_seed1" \
    timematch --weights "$source_weights" \
    --shape-da-mode local_support \
    --local-support-k 5 --local-support-temperature 0.1 \
    --adaptive-pseudo-selection false --oracle-pseudo-labels false \
    --epochs "$EPOCHS" --steps_per_epoch "$STEPS_PER_EPOCH" --lr 0.0001 \
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true \
    --shift_source true --sample_size "$SAMPLE_SIZE" --max_temporal_shift 60 \
    --domain_specific_bn true --shift_estimator AM --run_validation \
    --output_student true \
    2>&1 | tee -a "$log_file"
  echo "[FINISHED] $src_name -> $tgt_name" | tee -a "$log_file"
}

if [[ "$SMOKE" == "1" ]]; then
  run_task "$GPU0" AT1 "$AT1" DK1 "$DK1"
  exit $?
fi

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
