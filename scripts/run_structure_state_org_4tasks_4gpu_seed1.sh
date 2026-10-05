#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"; GPU3="${GPU3:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
EXP_ROOT="${EXP_ROOT:-outputs/structure_state_org_4tasks_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_state_org_4tasks_seed1}"
RUN_ROOT="${RUN_ROOT:-runs/structure_state_org_4tasks_seed1}"

AT1="austria/33UVP/2017"; DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"; FR2="france/31TCJ/2017"

mkdir -p "$LOG_ROOT" "$EXP_ROOT/source" "$EXP_ROOT/uda" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

run_task() {
  local gpu="$1" src="$2" src_data="$3" tgt="$4" tgt_data="$5"
  local task="${src}_${tgt}" source_experiment="source_${src}_seed1"
  local source_weights="$EXP_ROOT/source/$source_experiment"
  local log="$LOG_ROOT/${task}.log"
  : > "$log"
  echo "STATE_ORG_PLAN|gpu=$gpu|task=$task|fourier_modes=13|grid=64|window=24|stride=8|anchors=16|resample_record=24|representation=state_org|injection=direct_response_query|response_dim=48|source_minority_mode=base" | tee -a "$log"

  echo "STATE_ORG_SOURCE_START|gpu=$gpu|source=$src" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "$source_experiment" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$src_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 24 \
    --shape-representation state_org --shape-injection direct_response_query \
    --structure-shift-mode none --shapelet-diversity-margin 0.5 \
    --shapelet-diversity-weight 0.01 --shape-class-weight 0.1 \
    --source-minority-mode base --with_shift_aug false \
    --seed 1 --num_folds 1 --epochs 100 --batch_size 128 \
    --lr 0.001 --weight_decay 0.0001 --focal_loss_gamma 1.0 \
    --seq_length 30 --num_pixels 64 --closed_set true --progress_bar off \
    --output_dir "$EXP_ROOT/source" \
    --tensorboard_log_dir "$RUN_ROOT/source_${src}_seed1" >> "$log" 2>&1

  if [[ ! -f "$source_weights/fold_0/model.pt" ]]; then
    echo "ERROR: state_org source checkpoint missing: $source_weights/fold_0/model.pt" | tee -a "$log"
    return 1
  fi

  echo "STATE_ORG_UDA_START|gpu=$gpu|task=$task" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "${task}_seed1" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 24 \
    --shape-representation state_org --shape-injection direct_response_query \
    --structure-shift-mode none --shapelet-diversity-margin 0.5 \
    --shapelet-diversity-weight 0.01 --shape-class-weight 0.1 \
    --shape-align-weight 0 --source-minority-mode base --with_shift_aug false \
    --seed 1 --num_folds 1 --batch_size 128 --seq_length 30 --num_pixels 64 \
    --closed_set true --progress_bar off --output_dir "$EXP_ROOT/uda" \
    --tensorboard_log_dir "$RUN_ROOT/${task}_seed1" \
    timematch --weights "$source_weights" --epochs 20 --steps_per_epoch 500 \
    --lr 0.0001 --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true \
    --shift_source true --sample_size 100 --max_temporal_shift 60 \
    --domain_specific_bn true --shift_estimator AM --run_validation \
    --output_student true --shape-da-mode batch_align \
    --shape-alignment-view none --shape-equivariance-weight 0 \
    --adaptive-pseudo-selection false --oracle-pseudo-labels false \
    >> "$log" 2>&1

  if [[ ! -f "$EXP_ROOT/uda/${task}_seed1/fold_0/checkpoint_last.pt" ]]; then
    echo "ERROR: state_org final checkpoint missing: $EXP_ROOT/uda/${task}_seed1/fold_0/checkpoint_last.pt" | tee -a "$log"
    return 1
  fi
  echo "STATE_ORG_FINISHED|gpu=$gpu|task=$task|test_checkpoint=checkpoint_last.pt" | tee -a "$log"
}

run_task "$GPU0" AT1 "$AT1" DK1 "$DK1" & P0=$!
run_task "$GPU1" FR1 "$FR1" FR2 "$FR2" & P1=$!
run_task "$GPU2" FR2 "$FR2" DK1 "$DK1" & P2=$!
run_task "$GPU3" DK1 "$DK1" AT1 "$AT1" & P3=$!

status=0
wait "$P0" || status=1
wait "$P1" || status=1
wait "$P2" || status=1
wait "$P3" || status=1
if (( status != 0 )); then
  echo "ERROR: one or more state_org tasks failed; inspect $LOG_ROOT" >&2
  exit 1
fi
echo "STATE_ORG_ALL_FINISHED|output=$EXP_ROOT"
