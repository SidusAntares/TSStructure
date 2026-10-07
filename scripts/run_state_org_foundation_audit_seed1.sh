#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_SOURCE="${SKIP_SOURCE:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUT_ROOT="${OUT_ROOT:-outputs/state_org_foundation}"
LOG_ROOT="${LOG_ROOT:-logs/state_org_foundation}"
RUN_ROOT="${RUN_ROOT:-runs/state_org_foundation}"
AT1="austria/33UVP/2017"; DK1="denmark/32VNH/2017"; FR2="france/31TCJ/2017"
mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

train_source() {
  local gpu="$1" src="$2" src_data="$3" readout="$4"
  local experiment="source_${src}_seed1" root="$OUT_ROOT/source/$readout"
  local log="$LOG_ROOT/${src}/source_${readout}.log"
  mkdir -p "$root" "$LOG_ROOT/${src}" "$RUN_ROOT/source/$readout"
  echo "FOUNDATION_SOURCE_START|gpu=$gpu|source=$src|readout=$readout" > "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "$experiment" --data_root "$DATA_ROOT" --source "$src_data" --target "$src_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 24 \
    --shape-representation state_org --state-org-readout "$readout" \
    --shape-injection direct_response_query --structure-shift-mode none \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --source-minority-mode base --with_shift_aug false \
    --seed 1 --num_folds 1 --epochs 100 --batch_size 128 --lr 0.001 \
    --weight_decay 0.0001 --focal_loss_gamma 1.0 --seq_length 30 --num_pixels 64 \
    --closed_set true --progress_bar off --output_dir "$root" \
    --tensorboard_log_dir "$RUN_ROOT/source/$readout/${src}_seed1" >> "$log" 2>&1
  test -f "$root/$experiment/fold_0/model.pt"
}

train_uda() {
  local gpu="$1" src="$2" src_data="$3" tgt="$4" tgt_data="$5"
  local family="$6" variant="$7" readout="$8" anchor_mode="$9"
  local task="${src}_${tgt}"
  local source_root="$OUT_ROOT/source/$readout/source_${src}_seed1"
  local root="$OUT_ROOT/$family/$variant/uda"
  local log="$LOG_ROOT/$task/${family}_${variant}.log"
  mkdir -p "$root" "$LOG_ROOT/$task" "$RUN_ROOT/$family/$variant"
  echo "FOUNDATION_UDA_START|gpu=$gpu|task=$task|family=$family|variant=$variant|readout=$readout|anchor_update=$anchor_mode" > "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "${task}_seed1" --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 24 \
    --shape-representation state_org --state-org-readout "$readout" \
    --shape-injection direct_response_query --structure-shift-mode none \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0 \
    --shape-class-weight 0.1 --shape-align-weight 0 --source-minority-mode base \
    --with_shift_aug false --seed 1 --num_folds 1 --batch_size 128 --seq_length 30 \
    --num_pixels 64 --closed_set true --progress_bar off --output_dir "$root" \
    --tensorboard_log_dir "$RUN_ROOT/$family/$variant/${task}_seed1" \
    timematch --weights "$source_root" --epochs 20 --steps_per_epoch 500 \
    --lr 0.0001 --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true --shift_source true \
    --sample_size 100 --max_temporal_shift 60 --domain_specific_bn true \
    --shift_estimator AM --run_validation --output_student true --shape-da-mode batch_align \
    --shape-alignment-view none --shape-equivariance-weight 0 --adaptive-pseudo-selection false \
    --oracle-pseudo-labels false --uda-shape-class-weight 0 \
    --structure-basis-mode frozen_source --freeze-state-org-query true \
    --uda-anchor-update "$anchor_mode" >> "$log" 2>&1
  test -f "$root/${task}_seed1/fold_0/checkpoint_last.pt"
}

run_task() {
  local gpu="$1" src="$2" src_data="$3" tgt="$4" tgt_data="$5"
  local task="${src}_${tgt}"
  local readout mode checkpoint
  for readout in full composition presence; do
    if [[ "$SKIP_SOURCE" == "1" ]]; then
      checkpoint="$OUT_ROOT/source/$readout/source_${src}_seed1/fold_0/model.pt"
      if ! test -f "$checkpoint"; then
        echo "ERROR: requested source reuse but checkpoint is missing: $checkpoint" >&2
        return 1
      fi
      echo "FOUNDATION_SOURCE_REUSE|gpu=$gpu|source=$src|readout=$readout|checkpoint=$checkpoint"
    else
      train_source "$gpu" "$src" "$src_data" "$readout"
    fi
  done

  mkdir -p "$OUT_ROOT/audit/$task" "$LOG_ROOT/$task"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u analysis/state_org_foundation_audit.py \
    --checkpoint "full=$OUT_ROOT/source/full/source_${src}_seed1/fold_0/model.pt" \
    --checkpoint "composition=$OUT_ROOT/source/composition/source_${src}_seed1/fold_0/model.pt" \
    --checkpoint "presence=$OUT_ROOT/source/presence/source_${src}_seed1/fold_0/model.pt" \
    --source "$src_data" --target "$tgt_data" --data-root "$DATA_ROOT" \
    --output-dir "$OUT_ROOT/audit/$task" --device cuda \
    > "$LOG_ROOT/$task/foundation_audit.log" 2>&1

  for readout in full composition presence; do
    train_uda "$gpu" "$src" "$src_data" "$tgt" "$tgt_data" \
      organization "$readout" "$readout" none
  done
  for mode in fixed source target shared; do
    train_uda "$gpu" "$src" "$src_data" "$tgt" "$tgt_data" \
      anchor "$mode" full "$mode"
  done

  local query_args=()
  local source_checkpoint fold stage
  for readout in full presence; do
    source_checkpoint="$OUT_ROOT/source/$readout/source_${src}_seed1/fold_0/model.pt"
    query_args+=(--checkpoint "${readout}_source::${source_checkpoint}::${source_checkpoint}::${readout}")
    fold="$OUT_ROOT/organization/$readout/uda/${task}_seed1/fold_0"
    for stage in best last; do
      query_args+=(--checkpoint "${readout}_${stage}::${fold}/checkpoint_${stage}.pt::${source_checkpoint}::${readout}")
    done
  done
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u analysis/state_org_query_role_audit.py \
    "${query_args[@]}" \
    --shift-checkpoint "$OUT_ROOT/organization/full/uda/${task}_seed1/fold_0/checkpoint_last.pt" \
    --source "$src_data" --target "$tgt_data" --data-root "$DATA_ROOT" \
    --output-dir "$OUT_ROOT/audit/$task" --device cuda \
    > "$LOG_ROOT/$task/query_role_audit.log" 2>&1
  echo "FOUNDATION_TASK_FINISHED|gpu=$gpu|task=$task"
}

run_task "$GPU0" AT1 "$AT1" DK1 "$DK1" & P0=$!
run_task "$GPU1" FR2 "$FR2" DK1 "$DK1" & P1=$!
run_task "$GPU2" DK1 "$DK1" AT1 "$AT1" & P2=$!
status=0; wait "$P0" || status=1; wait "$P1" || status=1; wait "$P2" || status=1
(( status == 0 )) || { echo "ERROR: state-org foundation worker failed" >&2; exit 1; }
"$PYTHON_BIN" -u analysis/summarize_state_org_foundation.py \
  --root "$OUT_ROOT" --log-root "$LOG_ROOT"
echo "STATE_ORG_FOUNDATION_FINISHED|output=$OUT_ROOT"
