#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"; GPU3="${GPU3:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
BASE_ROOT="${BASE_ROOT:-outputs/structure_proto_v2clean_4tasks_seed1}"
EXP_ROOT="${EXP_ROOT:-outputs/v2clean_minority_ab_seed1}"
SHIFT_ROOT="${SHIFT_ROOT:-outputs/v2clean_shift_sensitivity_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/v2clean_minority_ab_seed1}"
RUN_ROOT="${RUN_ROOT:-runs/v2clean_minority_ab_seed1}"
DRY_RUN="${DRY_RUN:-0}"

DK1="denmark/32VNH/2017"; AT1="austria/33UVP/2017"
FR1="france/30TXT/2017"; FR2="france/31TCJ/2017"
mkdir -p "$LOG_ROOT" "$EXP_ROOT" "$SHIFT_ROOT" "$RUN_ROOT"

source_checkpoint() { printf '%s/source/source_%s_seed1/fold_0/model.pt' "$BASE_ROOT" "$1"; }
base_uda_checkpoint() { printf '%s/uda/%s_seed1/fold_0/model.pt' "$BASE_ROOT" "$1"; }

preflight() {
  local failed=0 path
  for path in \
    "$(source_checkpoint DK1)" "$(source_checkpoint FR1)" \
    "$(base_uda_checkpoint DK1_AT1)" "$(base_uda_checkpoint FR1_FR2)"
  do
    [[ -f "$path" ]] || { echo "BLOCKER|missing=$path" >&2; failed=1; }
  done
  (( failed == 0 ))
}

run_shift_sensitivity() {
  local gpu="$1" task="$2" source="$3"
  echo "SHIFT_S_PLAN|gpu=$gpu|task=$task"
  [[ "$DRY_RUN" == "1" ]] && return 0
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u analysis/v2clean_shift_ab_audit.py shift \
    --task "$task" --checkpoint "$(source_checkpoint "$source")" \
    --data-root "$DATA_ROOT" --output-root "$SHIFT_ROOT" --device cuda \
    > "$LOG_ROOT/S_${task}.log" 2>&1
}

run_failure_audit() {
  local gpu="$1" variant="$2" task="$3" stage="$4" checkpoint="$5" output="$6" log="$7"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u analysis/v2clean_shift_ab_audit.py failure \
    --task "$task" --variant "$variant" --stage "$stage" \
    --checkpoint "$checkpoint" --data-root "$DATA_ROOT" --output "$output" \
    --device cuda >> "$log" 2>&1
}

run_base_audits() {
  local gpu="$1" source="$2" task="$3"
  local log="$LOG_ROOT/Base_${task}.log"
  : > "$log"
  run_failure_audit "$gpu" Base "$task" source "$(source_checkpoint "$source")" \
    "$EXP_ROOT/Base/audit/$task/source" "$log"
  run_failure_audit "$gpu" Base "$task" uda_best "$(base_uda_checkpoint "$task")" \
    "$EXP_ROOT/Base/audit/$task/uda_best" "$log"
}

run_pipeline() {
  local gpu="$1" variant="$2" source="$3" source_data="$4" target="$5" target_data="$6"
  local task="${source}_${target}" mode log source_dir uda_dir
  [[ "$variant" == "A" ]] && mode="weighted" || mode="weighted_proto"
  log="$LOG_ROOT/${variant}_${task}.log"
  source_dir="$EXP_ROOT/$variant/source/source_${source}_seed1"
  uda_dir="$EXP_ROOT/$variant/uda/${task}_seed1"
  echo "MINORITY_PIPELINE_PLAN|gpu=$gpu|variant=$variant|task=$task|mode=$mode"
  [[ "$DRY_RUN" == "1" ]] && return 0
  : > "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "source_${source}_seed1" --data_root "$DATA_ROOT" \
    --source "$source_data" --target "$source_data" --model psestructureprotoltae \
    --structure-branch true --structure-exposer fourier --fourier_num_modes 13 \
    --shape-dim 128 --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --source-minority-mode "$mode" --proto-momentum 0.9 \
    --proto-ramp-epochs 5 --proto-ramp-start 0.1 --with_shift_aug false \
    --seed 1 --num_folds 1 --epochs 100 --batch_size 128 --lr 0.001 \
    --weight_decay 0.0001 --focal_loss_gamma 1.0 --seq_length 30 --num_pixels 64 \
    --closed_set true --progress_bar off --output_dir "$EXP_ROOT/$variant/source" \
    --tensorboard_log_dir "$RUN_ROOT/$variant/source_${source}_seed1" >> "$log" 2>&1
  [[ -f "$source_dir/fold_0/model.pt" ]] || { echo "ERROR: missing source checkpoint $source_dir/fold_0/model.pt" >> "$log"; return 1; }
  run_failure_audit "$gpu" "$variant" "$task" source "$source_dir/fold_0/model.pt" \
    "$EXP_ROOT/$variant/audit/$task/source" "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "${task}_seed1" --data_root "$DATA_ROOT" --source "$source_data" --target "$target_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0.05 --source-minority-mode base \
    --seed 1 --num_folds 1 --batch_size 128 --seq_length 30 --num_pixels 64 \
    --closed_set true --with_shift_aug false --progress_bar off \
    --output_dir "$EXP_ROOT/$variant/uda" --tensorboard_log_dir "$RUN_ROOT/$variant/${task}_seed1" \
    timematch --weights "$source_dir" --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 --estimate_shift true \
    --balance_source true --use_focal_loss true --shift_source true --sample_size 100 \
    --max_temporal_shift 60 --domain_specific_bn true --shift_estimator AM \
    --run_validation --output_student true >> "$log" 2>&1
  [[ -f "$uda_dir/fold_0/model.pt" ]] || { echo "ERROR: missing UDA checkpoint $uda_dir/fold_0/model.pt" >> "$log"; return 1; }
  run_failure_audit "$gpu" "$variant" "$task" uda_best "$uda_dir/fold_0/model.pt" \
    "$EXP_ROOT/$variant/audit/$task/uda_best" "$log"
}

if [[ "$DRY_RUN" != "1" ]]; then preflight; fi

run_shift_sensitivity "$GPU0" DK1_AT1 DK1 & s0=$!
run_shift_sensitivity "$GPU1" FR1_FR2 FR1 & s1=$!
wait "$s0"; wait "$s1"
[[ "$DRY_RUN" == "1" ]] || "$PYTHON_BIN" -u analysis/v2clean_shift_ab_audit.py shift-summary --output-root "$SHIFT_ROOT"

if [[ "$DRY_RUN" != "1" ]]; then
  run_base_audits "$GPU0" DK1 DK1_AT1 & b0=$!
  run_base_audits "$GPU1" FR1 FR1_FR2 & b1=$!
  wait "$b0"; wait "$b1"
fi

run_pipeline "$GPU0" A DK1 "$DK1" AT1 "$AT1" & p0=$!
run_pipeline "$GPU1" A FR1 "$FR1" FR2 "$FR2" & p1=$!
run_pipeline "$GPU2" B DK1 "$DK1" AT1 "$AT1" & p2=$!
run_pipeline "$GPU3" B FR1 "$FR1" FR2 "$FR2" & p3=$!
status=0
wait "$p0" || status=1; wait "$p1" || status=1
wait "$p2" || status=1; wait "$p3" || status=1
(( status == 0 )) || exit "$status"
[[ "$DRY_RUN" == "1" ]] || "$PYTHON_BIN" -u analysis/v2clean_shift_ab_audit.py aggregate --root "$EXP_ROOT"
