#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/structure_state_org_4tasks_seed1/source}"
OLD_ROOT="${OLD_ROOT:-outputs/state_org_feasibility/variants/no_shape_aux}"
OUT_ROOT="${OUT_ROOT:-outputs/state_org_hierarchical}"
LOG_ROOT="${LOG_ROOT:-logs/state_org_hierarchical}"
RUN_ROOT="${RUN_ROOT:-runs/state_org_hierarchical}"
AT1="austria/33UVP/2017"; DK1="denmark/32VNH/2017"; FR2="france/31TCJ/2017"
mkdir -p "$OUT_ROOT/audit" "$OUT_ROOT/variants" "$LOG_ROOT" "$RUN_ROOT"

train_variant() {
  local gpu="$1" src="$2" src_data="$3" tgt="$4" tgt_data="$5" variant="$6" basis="$7" view="$8"
  local task="${src}_${tgt}" output="$OUT_ROOT/variants/$variant/uda"
  mkdir -p "$LOG_ROOT/$task" "$RUN_ROOT/$variant"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "${task}_seed1" --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 24 \
    --shape-representation state_org --shape-injection direct_response_query --structure-shift-mode none \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 --shape-class-weight 0.1 \
    --shape-align-weight 0 --source-minority-mode base --with_shift_aug false --seed 1 --num_folds 1 \
    --batch_size 128 --seq_length 30 --num_pixels 64 --closed_set true --progress_bar off \
    --output_dir "$output" --tensorboard_log_dir "$RUN_ROOT/$variant/${task}_seed1" \
    timematch --weights "$SOURCE_ROOT/source_${src}_seed1" --epochs 8 --steps_per_epoch 500 \
    --lr 0.0001 --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true --shift_source true \
    --sample_size 100 --max_temporal_shift 60 --domain_specific_bn true --shift_estimator AM \
    --run_validation --output_student true --shape-da-mode batch_align --shape-alignment-view none \
    --shape-equivariance-weight 0 --adaptive-pseudo-selection false --oracle-pseudo-labels false \
    --uda-shape-class-weight 0 --structure-basis-mode "$basis" --state-org-query-view "$view" \
    --shape-query-scale 1 > "$LOG_ROOT/$task/${variant}.log" 2>&1
  test -f "$output/${task}_seed1/fold_0/checkpoint_last.pt"
}

run_task() {
  local gpu="$1" src="$2" src_data="$3" tgt="$4" tgt_data="$5" task="${src}_${tgt}"
  local source_ckpt="$SOURCE_ROOT/source_${src}_seed1/fold_0/model.pt"
  local old_fold="$OLD_ROOT/uda/${task}_seed1/fold_0"
  test -f "$source_ckpt"; test -f "$old_fold/checkpoint_best.pt"; test -f "$old_fold/checkpoint_last.pt"
  mkdir -p "$OUT_ROOT/audit/$task" "$LOG_ROOT/$task"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u analysis/state_org_anchor_basis_audit.py \
    --source-checkpoint "$source_ckpt" --shift-checkpoint "$old_fold/checkpoint_last.pt" \
    --source "$src_data" --target "$tgt_data" \
    --data-root "$DATA_ROOT" --output-dir "$OUT_ROOT/audit/$task" --device cuda \
    > "$LOG_ROOT/$task/anchor_organization_audit.log" 2>&1
  train_variant "$gpu" "$src" "$src_data" "$tgt" "$tgt_data" adaptive_presence adaptive presence
  train_variant "$gpu" "$src" "$src_data" "$tgt" "$tgt_data" frozen_full frozen_source full
  train_variant "$gpu" "$src" "$src_data" "$tgt" "$tgt_data" frozen_presence frozen_source presence
  local query_args=(
    --checkpoint "source::${source_ckpt}::full::adaptive"
    --checkpoint "adaptive_full_best::${old_fold}/checkpoint_best.pt::full::adaptive"
    --checkpoint "adaptive_full_last::${old_fold}/checkpoint_last.pt::full::adaptive"
  )
  local variant view basis fold
  for variant in adaptive_presence frozen_full frozen_presence; do
    case "$variant" in adaptive_presence) view=presence; basis=adaptive;; frozen_full) view=full; basis=frozen_source;; *) view=presence; basis=frozen_source;; esac
    fold="$OUT_ROOT/variants/$variant/uda/${task}_seed1/fold_0"
    query_args+=(--checkpoint "${variant}_best::${fold}/checkpoint_best.pt::${view}::${basis}")
    query_args+=(--checkpoint "${variant}_last::${fold}/checkpoint_last.pt::${view}::${basis}")
  done
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u analysis/state_org_query_scale_audit.py \
    --source-checkpoint "$source_ckpt" --shift-config-checkpoint "$old_fold/checkpoint_last.pt" \
    "${query_args[@]}" --source "$src_data" --target "$tgt_data" \
    --data-root "$DATA_ROOT" --output-dir "$OUT_ROOT/audit/$task" --device cuda \
    > "$LOG_ROOT/$task/query_scale_audit.log" 2>&1
}

run_task "$GPU0" AT1 "$AT1" DK1 "$DK1" & P0=$!
run_task "$GPU1" FR2 "$FR2" DK1 "$DK1" & P1=$!
run_task "$GPU2" DK1 "$DK1" AT1 "$AT1" & P2=$!
status=0; wait "$P0" || status=1; wait "$P1" || status=1; wait "$P2" || status=1
(( status == 0 )) || { echo "ERROR: hierarchical state-org worker failed" >&2; exit 1; }
"$PYTHON_BIN" -u analysis/summarize_state_org_hierarchical.py --root "$OUT_ROOT" --log-root "$LOG_ROOT" --old-root "$OLD_ROOT"
echo "STATE_ORG_HIERARCHICAL_FINISHED|output=$OUT_ROOT"
