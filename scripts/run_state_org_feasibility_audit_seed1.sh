#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
BASE_ROOT="${BASE_ROOT:-outputs/structure_state_org_4tasks_seed1}"
OUT_ROOT="${OUT_ROOT:-outputs/state_org_feasibility}"
LOG_ROOT="${LOG_ROOT:-logs/state_org_feasibility}"
RUN_ROOT="${RUN_ROOT:-runs/state_org_feasibility}"
AT1="austria/33UVP/2017"; DK1="denmark/32VNH/2017"
FR2="france/31TCJ/2017"

mkdir -p "$OUT_ROOT/audit" "$OUT_ROOT/variants" "$LOG_ROOT" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

run_variant() {
  local gpu="$1" src="$2" src_data="$3" tgt="$4" tgt_data="$5" variant="$6"
  local task="${src}_${tgt}" source_weights="$BASE_ROOT/source/source_${src}_seed1"
  local output="$OUT_ROOT/variants/$variant/uda" log_dir="$LOG_ROOT/$task"
  local -a causal_args=()
  mkdir -p "$output" "$log_dir" "$RUN_ROOT/$variant"
  case "$variant" in
    no_shape_aux) causal_args+=(--uda-shape-class-weight 0) ;;
    detach_target_structure) causal_args+=(--detach-target-structure true) ;;
    freeze_structure_specific)
      causal_args+=(--freeze-structure-specific true --uda-shape-class-weight 0) ;;
    *) echo "ERROR: unknown variant $variant" >&2; return 2 ;;
  esac
  echo "STATE_ORG_VARIANT_START|gpu=$gpu|task=$task|variant=$variant|epochs=8|steps=500"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "${task}_seed1" --data_root "$DATA_ROOT" \
    --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 \
    --shape-window-stride 8 --shapelet-count 16 --shapelet-beta 5 \
    --shape-resample-length 24 --shape-representation state_org \
    --shape-injection direct_response_query --structure-shift-mode none \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0 --source-minority-mode base \
    --with_shift_aug false --seed 1 --num_folds 1 --batch_size 128 \
    --seq_length 30 --num_pixels 64 --closed_set true --progress_bar off \
    --output_dir "$output" --tensorboard_log_dir "$RUN_ROOT/$variant/${task}_seed1" \
    timematch --weights "$source_weights" --epochs 8 --steps_per_epoch 500 \
    --lr 0.0001 --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true \
    --shift_source true --sample_size 100 --max_temporal_shift 60 \
    --domain_specific_bn true --shift_estimator AM --run_validation \
    --output_student true --shape-da-mode batch_align --shape-alignment-view none \
    --shape-equivariance-weight 0 --adaptive-pseudo-selection false \
    --oracle-pseudo-labels false "${causal_args[@]}" \
    > "$log_dir/${variant}.log" 2>&1
  test -f "$output/${task}_seed1/fold_0/checkpoint_last.pt" || {
    echo "ERROR: missing final checkpoint for $task $variant" >&2; return 1;
  }
}

run_task() {
  local gpu="$1" src="$2" src_data="$3" tgt="$4" tgt_data="$5"
  local task="${src}_${tgt}" source_ckpt best_ckpt last_ckpt
  source_ckpt="$BASE_ROOT/source/source_${src}_seed1/fold_0/model.pt"
  best_ckpt="$BASE_ROOT/uda/${task}_seed1/fold_0/checkpoint_best.pt"
  last_ckpt="$BASE_ROOT/uda/${task}_seed1/fold_0/checkpoint_last.pt"
  for checkpoint in "$source_ckpt" "$best_ckpt" "$last_ckpt"; do
    test -f "$checkpoint" || { echo "ERROR: missing baseline checkpoint $checkpoint" >&2; return 1; }
  done
  mkdir -p "$LOG_ROOT/$task"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u analysis/state_org_feasibility_audit.py \
    --source-checkpoint "$source_ckpt" --uda-best-checkpoint "$best_ckpt" \
    --uda-last-checkpoint "$last_ckpt" --source "$src_data" --target "$tgt_data" \
    --data-root "$DATA_ROOT" --output-dir "$OUT_ROOT/audit/$task" --device cuda \
    > "$LOG_ROOT/$task/audit.log" 2>&1
  run_variant "$gpu" "$src" "$src_data" "$tgt" "$tgt_data" no_shape_aux
  run_variant "$gpu" "$src" "$src_data" "$tgt" "$tgt_data" detach_target_structure
  run_variant "$gpu" "$src" "$src_data" "$tgt" "$tgt_data" freeze_structure_specific
}

run_task "$GPU0" AT1 "$AT1" DK1 "$DK1" & P0=$!
run_task "$GPU1" FR2 "$FR2" DK1 "$DK1" & P1=$!
run_task "$GPU2" DK1 "$DK1" AT1 "$AT1" & P2=$!
status=0
wait "$P0" || status=1; wait "$P1" || status=1; wait "$P2" || status=1
(( status == 0 )) || { echo "ERROR: state-org feasibility worker failed" >&2; exit 1; }
"$PYTHON_BIN" -u analysis/summarize_state_org_feasibility.py \
  --root "$OUT_ROOT" --log-root "$LOG_ROOT" --baseline-root "$BASE_ROOT"
echo "STATE_ORG_FEASIBILITY_FINISHED|output=$OUT_ROOT"
