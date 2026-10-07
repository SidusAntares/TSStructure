#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
FOUNDATION_ROOT="${FOUNDATION_ROOT:-outputs/state_org_foundation}"
OUT_ROOT="${OUT_ROOT:-outputs/state_org_next_audits}"
LOG_ROOT="${LOG_ROOT:-logs/state_org_next_audits}"
RUN_ROOT="${RUN_ROOT:-runs/state_org_next_audits}"
AT1="austria/33UVP/2017"; DK1="denmark/32VNH/2017"; FR2="france/31TCJ/2017"
mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

require_file() {
  test -f "$1" || { echo "ERROR: missing required artifact: $1" >&2; return 1; }
}

run_geometric_uda() {
  local gpu="$1" src="$2" src_data="$3" tgt="$4" tgt_data="$5" mode="$6"
  local task="${src}_${tgt}"
  local source_root="$FOUNDATION_ROOT/source/composition/source_${src}_seed1"
  local root="$OUT_ROOT/anchor_geometric/$mode/uda"
  local log="$LOG_ROOT/$task/anchor_geometric_${mode}.log"
  require_file "$source_root/fold_0/model.pt"
  mkdir -p "$root" "$LOG_ROOT/$task" "$RUN_ROOT/anchor_geometric/$mode"
  echo "ANCHOR_GEOMETRIC_START|gpu=$gpu|task=$task|mode=$mode|source_checkpoint=$source_root/fold_0/model.pt" > "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "${task}_seed1" --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 24 \
    --shape-representation state_org --state-org-readout composition \
    --shape-injection direct_response_query --structure-shift-mode none \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0 \
    --shape-class-weight 0.1 --shape-align-weight 0 --source-minority-mode base \
    --with_shift_aug false --seed 1 --num_folds 1 --batch_size 128 --seq_length 30 \
    --num_pixels 64 --closed_set true --progress_bar off --output_dir "$root" \
    --tensorboard_log_dir "$RUN_ROOT/anchor_geometric/$mode/${task}_seed1" \
    timematch --weights "$source_root" --epochs 20 --steps_per_epoch 500 \
    --lr 0.0001 --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true --shift_source true \
    --sample_size 100 --max_temporal_shift 60 --domain_specific_bn true \
    --shift_estimator AM --run_validation --output_student true --shape-da-mode batch_align \
    --shape-alignment-view none --shape-equivariance-weight 0 --adaptive-pseudo-selection false \
    --oracle-pseudo-labels false --uda-shape-class-weight 0 \
    --structure-basis-mode frozen_source --freeze-structure-specific true \
    --freeze-state-org-query true --uda-anchor-update none \
    --anchor-geometric-mode "$mode" --anchor-geometric-step 0.1 \
    --anchor-geometric-token-limit 50000 >> "$log" 2>&1
  require_file "$root/${task}_seed1/fold_0/checkpoint_last.pt"
  require_file "$root/${task}_seed1/fold_0/anchor_geometric_dynamics.csv"
}

run_task() {
  local gpu="$1" src="$2" src_data="$3" tgt="$4" tgt_data="$5"
  local task="${src}_${tgt}"
  local presence="$FOUNDATION_ROOT/source/presence/source_${src}_seed1/fold_0/model.pt"
  local shift_checkpoint="$FOUNDATION_ROOT/organization/presence/uda/${task}_seed1/fold_0/checkpoint_last.pt"
  local audit="$OUT_ROOT/audit/$task"
  local mode
  require_file "$presence"; require_file "$shift_checkpoint"
  mkdir -p "$audit" "$LOG_ROOT/$task"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u analysis/state_org_shift_sensitivity_audit.py \
    --source-checkpoint "$presence" --shift-checkpoint "$shift_checkpoint" \
    --source "$src_data" --target "$tgt_data" --data-root "$DATA_ROOT" \
    --output-dir "$audit" --device cuda \
    > "$LOG_ROOT/$task/shift_sensitivity_audit.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u analysis/state_org_assignment_audit.py \
    --source-checkpoint "$presence" --shift-checkpoint "$shift_checkpoint" \
    --source "$src_data" --target "$tgt_data" --data-root "$DATA_ROOT" \
    --output-dir "$audit" --device cuda \
    > "$LOG_ROOT/$task/assignment_audit.log" 2>&1
  for mode in fixed target_ema shared_ema; do
    run_geometric_uda "$gpu" "$src" "$src_data" "$tgt" "$tgt_data" "$mode"
  done
  echo "STATE_ORG_NEXT_TASK_FINISHED|gpu=$gpu|task=$task"
}

run_task "$GPU0" AT1 "$AT1" DK1 "$DK1" & P0=$!
run_task "$GPU1" FR2 "$FR2" DK1 "$DK1" & P1=$!
run_task "$GPU2" DK1 "$DK1" AT1 "$AT1" & P2=$!
status=0; wait "$P0" || status=1; wait "$P1" || status=1; wait "$P2" || status=1
(( status == 0 )) || { echo "ERROR: state-org next-audit worker failed" >&2; exit 1; }
"$PYTHON_BIN" -u analysis/summarize_state_org_next_audits.py \
  --root "$OUT_ROOT" --log-root "$LOG_ROOT" --foundation-root "$FOUNDATION_ROOT"
echo "STATE_ORG_NEXT_AUDITS_FINISHED|output=$OUT_ROOT"
