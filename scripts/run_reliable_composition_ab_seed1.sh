#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"; GPU3="${GPU3:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUT_ROOT="${OUT_ROOT:-outputs/reliable_composition_ab}"
LOG_ROOT="${LOG_ROOT:-logs/reliable_composition_ab}"
RUN_ROOT="${RUN_ROOT:-runs/reliable_composition_ab}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SOURCE="${SKIP_SOURCE:-0}"

AT1="austria/33UVP/2017"; DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"; FR2="france/31TCJ/2017"
TASKS="AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1"
mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

execute_logged() {
  local gpu="$1" log="$2"; shift 2
  mkdir -p "$(dirname "$log")"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN|gpu=%s|' "$gpu" | tee -a "$log"
    printf '%q ' "$@" | tee -a "$log"
    printf '\n' | tee -a "$log"
  else
    env CUDA_VISIBLE_DEVICES="$gpu" "$@" >> "$log" 2>&1
  fi
}

checkpoint_status() {
  local kind="$1" fold="$2" log="$3"
  "$PYTHON_BIN" -u analysis/summarize_reliable_composition_ab.py \
    checkpoint-status --kind "$kind" --fold "$fold" >> "$log" 2>&1
}

train_source() {
  local gpu="$1" src="$2" src_data="$3" seed="$4" log="$5"
  local experiment="source_${src}_seed${seed}"
  local root="$OUT_ROOT/source/seed${seed}"
  echo "RELIABLE_SOURCE_START|gpu=$gpu|source=$src|seed=$seed|tau_init=0|scale_init=1|ema=0.9" | tee -a "$log"
  execute_logged "$gpu" "$log" "$PYTHON_BIN" -u train.py \
    -e "$experiment" --data_root "$DATA_ROOT" --source "$src_data" --target "$src_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 \
    --shape-window-stride 1 --shapelet-count 16 --shapelet-beta 5 \
    --shape-resample-length 24 --shape-representation state_org \
    --state-org-readout reliable_composition --shape-injection direct_response_query \
    --structure-shift-mode none --shapelet-diversity-margin 0.5 \
    --shapelet-diversity-weight 0.01 --shape-class-weight 0.1 \
    --source-minority-mode base --with_shift_aug false --seed "$seed" --num_folds 1 \
    --epochs 100 --batch_size 128 --lr 0.001 --weight_decay 0.0001 \
    --focal_loss_gamma 1.0 --seq_length 30 --num_pixels 64 --closed_set true \
    --progress_bar off --output_dir "$root" \
    --tensorboard_log_dir "$RUN_ROOT/source/seed${seed}/${src}"
}

ensure_source() {
  local gpu="$1" src="$2" src_data="$3" seed="$4" log="$5"
  local source_root="$OUT_ROOT/source/seed${seed}/source_${src}_seed${seed}"
  local fold="$source_root/fold_0" status
  if [[ "$DRY_RUN" == "1" && "$SKIP_SOURCE" == "1" ]]; then
    echo "RELIABLE_SOURCE_REUSE|gpu=$gpu|source=$src|seed=$seed|checkpoint=$fold/model.pt" | tee -a "$log"
    return
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    train_source "$gpu" "$src" "$src_data" "$seed" "$log"
    return
  fi
  if checkpoint_status source "$fold" "$log"; then
    echo "RELIABLE_SOURCE_REUSE|gpu=$gpu|source=$src|seed=$seed|checkpoint=$fold/model.pt" | tee -a "$log"
    return
  else
    status=$?
  fi
  if (( status != 3 )); then
    echo "ERROR: invalid or incomplete source output; refusing overwrite: $fold (status=$status)" | tee -a "$log" >&2
    return 1
  fi
  if [[ "$SKIP_SOURCE" == "1" ]]; then
    echo "ERROR: complete source checkpoint required by SKIP_SOURCE=1: $fold/model.pt" | tee -a "$log" >&2
    return 1
  fi
  train_source "$gpu" "$src" "$src_data" "$seed" "$log"
  checkpoint_status source "$fold" "$log" || {
    echo "ERROR: source training did not produce a complete final checkpoint: $fold" | tee -a "$log" >&2
    return 1
  }
}

evaluate_checkpoint() {
  local gpu="$1" checkpoint="$2" output="$3" log="$4"
  execute_logged "$gpu" "$log" "$PYTHON_BIN" -u \
    analysis/summarize_reliable_composition_ab.py evaluate \
    --checkpoint "$checkpoint" --data-root "$DATA_ROOT" \
    --output "$output" --device cuda --batch-size 128
}

run_uda() {
  local gpu="$1" variant="$2" detach_epochs="$3" src="$4" src_data="$5"
  local tgt="$6" tgt_data="$7" seed="$8" source_weights="$9" log="${10}"
  local task="${src}_${tgt}" experiment="${task}_seed${seed}"
  local root="$OUT_ROOT/$variant/seed${seed}/uda"
  local fold="$root/$experiment/fold_0" status
  if [[ "$DRY_RUN" != "1" ]]; then
    if checkpoint_status uda "$fold" "$log"; then
      echo "RELIABLE_UDA_REUSE|gpu=$gpu|variant=$variant|task=$task|seed=$seed|fold=$fold" | tee -a "$log"
      evaluate_checkpoint "$gpu" "$fold/checkpoint_best.pt" "$fold/test_metrics_best_audit.json" "$log"
      evaluate_checkpoint "$gpu" "$fold/checkpoint_last.pt" "$fold/test_metrics_final_audit.json" "$log"
      return
    else
      status=$?
    fi
    if (( status != 3 )); then
      echo "ERROR: invalid or incomplete UDA output; refusing overwrite or implicit resume: $fold (status=$status)" | tee -a "$log" >&2
      return 1
    fi
  fi
  echo "RELIABLE_UDA_START|gpu=$gpu|variant=$variant|task=$task|seed=$seed|detach_epochs=$detach_epochs|source_checkpoint=$source_weights/fold_0/model.pt" | tee -a "$log"
  execute_logged "$gpu" "$log" "$PYTHON_BIN" -u train.py \
    -e "$experiment" --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 \
    --shape-window-stride 1 --shapelet-count 16 --shapelet-beta 5 \
    --shape-resample-length 24 --shape-representation state_org \
    --state-org-readout reliable_composition --shape-injection direct_response_query \
    --structure-shift-mode none --shapelet-diversity-margin 0.5 \
    --shapelet-diversity-weight 0.01 --shape-class-weight 0.1 \
    --shape-align-weight 0 --source-minority-mode base --with_shift_aug false \
    --seed "$seed" --num_folds 1 --batch_size 128 --seq_length 30 --num_pixels 64 \
    --closed_set true --progress_bar off --output_dir "$root" \
    --tensorboard_log_dir "$RUN_ROOT/$variant/seed${seed}/${task}" \
    timematch --weights "$source_weights" --epochs 20 --steps_per_epoch 500 \
    --lr 0.0001 --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true \
    --shift_source true --sample_size 100 --max_temporal_shift 60 \
    --domain_specific_bn true --shift_estimator AM --run_validation \
    --output_student true --shape-da-mode batch_align --shape-alignment-view none \
    --uda-shape-class-weight 0 --shape-equivariance-weight 0 \
    --adaptive-pseudo-selection false --oracle-pseudo-labels false \
    --structure-basis-mode adaptive --freeze-structure-specific false \
    --freeze-state-org-query false --detach-target-structure false \
    --target-structure-detach-epochs "$detach_epochs"

  if [[ "$DRY_RUN" != "1" ]]; then
    checkpoint_status uda "$fold" "$log" || {
      echo "ERROR: UDA training did not produce complete Best and Final checkpoints: $fold" | tee -a "$log" >&2
      return 1
    }
  fi
  evaluate_checkpoint "$gpu" "$fold/checkpoint_best.pt" \
    "$fold/test_metrics_best_audit.json" "$log"
  evaluate_checkpoint "$gpu" "$fold/checkpoint_last.pt" \
    "$fold/test_metrics_final_audit.json" "$log"
}

run_task() {
  local gpu="$1" src="$2" src_data="$3" tgt="$4" tgt_data="$5"
  local task="${src}_${tgt}" seed=1
  local log_dir="$LOG_ROOT/$task/seed1"
  local source_root="$OUT_ROOT/source/seed1/source_${src}_seed1"
  mkdir -p "$log_dir"
  ensure_source "$gpu" "$src" "$src_data" "$seed" "$log_dir/source.log"
  run_uda "$gpu" A 0 "$src" "$src_data" "$tgt" "$tgt_data" "$seed" \
    "$source_root" "$log_dir/A.log"
  run_uda "$gpu" B 5 "$src" "$src_data" "$tgt" "$tgt_data" "$seed" \
    "$source_root" "$log_dir/B.log"
}

run_task "$GPU0" AT1 "$AT1" DK1 "$DK1" & P0=$!
run_task "$GPU1" FR1 "$FR1" FR2 "$FR2" & P1=$!
run_task "$GPU2" FR2 "$FR2" DK1 "$DK1" & P2=$!
run_task "$GPU3" DK1 "$DK1" AT1 "$AT1" & P3=$!
status=0
wait "$P0" || status=1; wait "$P1" || status=1
wait "$P2" || status=1; wait "$P3" || status=1
if (( status != 0 )); then
  echo "ERROR: reliable-composition Seed1 A/B worker failed; inspect $LOG_ROOT" >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN|summary=$OUT_ROOT/summary"
else
  "$PYTHON_BIN" -u analysis/summarize_reliable_composition_ab.py summarize \
    --root "$OUT_ROOT" --output "$OUT_ROOT/summary" \
    > "$LOG_ROOT/summary.log" 2>&1
fi
echo "RELIABLE_COMPOSITION_AB_SEED1_FINISHED|output=$OUT_ROOT"
