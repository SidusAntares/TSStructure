#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"; GPU3="${GPU3:-3}"
RUN_ROUND="${RUN_ROUND:-ALL}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
ROOT="${ROOT:-experiments/structure_e_cross_seed1}"
OUTPUT_ROOT="$ROOT/outputs"
LOG_ROOT="$ROOT/logs"
RUN_ROOT="$ROOT/runs"
SOURCE_ROOT="$OUTPUT_ROOT/source"
UDA_ROOT="$OUTPUT_ROOT/uda"
E_SOURCE_ROOT="${E_SOURCE_ROOT:-outputs/structure_phase_moment_4tasks_seed1/source}"

AT1="austria/33UVP/2017"; DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"; FR2="france/31TCJ/2017"

mkdir -p "$SOURCE_ROOT" "$UDA_ROOT" "$LOG_ROOT" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

run_command() {
  local log="$1"
  shift
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN|log=%s|command=' "$log"
    printf '%q ' "$@"
    printf '\n'
  else
    "$@" > "$log" 2>&1
  fi
}

checkpoint_status() {
  local fold="$1" kind="$2" status=0
  if [[ "$DRY_RUN" == "1" ]]; then return 3; fi
  "$PYTHON_BIN" -u analysis/summarize_reliable_composition_ab.py \
    checkpoint-status --fold "$fold" --kind "$kind" >/dev/null || status=$?
  return "$status"
}

prepare_output() {
  local fold="$1" kind="$2" label="$3" status=0
  checkpoint_status "$fold" "$kind" || status=$?
  case "$status" in
    0) echo "CHECKPOINT_REUSE|kind=$kind|experiment=$label|fold=$fold"; return 1 ;;
    3) return 0 ;;
    *) echo "ERROR: incomplete output; refusing overwrite: $fold" >&2; exit 1 ;;
  esac
}

require_source() {
  local path="$1"
  if [[ "$DRY_RUN" != "1" && ! -f "$path/fold_0/model.pt" ]]; then
    echo "MISSING_SOURCE_CHECKPOINT|path=$path/fold_0/model.pt" >&2
    exit 1
  fi
}

variant_options() {
  local variant="$1"
  case "$variant" in
    G) printf '%s\n' "--shape-window-stride 8 --shape-injection current_query --phase-query-view full" ;;
    Q) printf '%s\n' "--shape-window-stride 8 --shape-injection direct_response_query --phase-query-view full" ;;
    S) printf '%s\n' "--shape-window-stride 1 --shape-injection current_query --phase-query-view full" ;;
    R) printf '%s\n' "--shape-window-stride 8 --shape-injection current_query --phase-query-view rich32" ;;
    *) echo "ERROR: unknown variant $variant" >&2; exit 2 ;;
  esac
}

source_checkpoint() {
  local variant="$1" source="$2"
  if [[ "$variant" == "G" ]]; then
    printf '%s/source_%s_seed1' "$E_SOURCE_ROOT" "$source"
  else
    printf '%s/SRC_%s_%s_seed1' "$SOURCE_ROOT" "$variant" "$source"
  fi
}

run_source() {
  local gpu="$1" variant="$2" source="$3" source_data="$4"
  local experiment="SRC_${variant}_${source}_seed1"
  local fold="$SOURCE_ROOT/$experiment/fold_0"
  local options
  options="$(variant_options "$variant")"
  if ! prepare_output "$fold" source "$experiment"; then return 0; fi
  # shellcheck disable=SC2086
  run_command "$LOG_ROOT/${experiment}.log" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" -u train.py \
    -e "$experiment" --data_root "$DATA_ROOT" --source "$source_data" --target "$source_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 $options \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shape-representation phase_moment --structure-shift-mode none \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --source-minority-mode base --with_shift_aug false \
    --seed 1 --num_folds 1 --epochs 100 --batch_size 128 --lr 0.001 \
    --weight_decay 0.0001 --focal_loss_gamma 1.0 --seq_length 30 --num_pixels 64 \
    --closed_set true --progress_bar off --output_dir "$SOURCE_ROOT" \
    --tensorboard_log_dir "$RUN_ROOT/$experiment"
  if [[ "$DRY_RUN" != "1" ]]; then
    checkpoint_status "$fold" source || {
      echo "ERROR: source did not complete: $experiment" >&2; exit 1;
    }
  fi
}

run_uda() {
  local gpu="$1" variant="$2" source="$3" source_data="$4" target="$5" target_data="$6"
  local experiment="${variant}_${source}_${target}_seed1"
  local fold="$UDA_ROOT/$experiment/fold_0"
  local weights options detach_epochs=0
  weights="$(source_checkpoint "$variant" "$source")"
  options="$(variant_options "$variant")"
  [[ "$variant" == "G" ]] && detach_epochs=5
  require_source "$weights"
  if ! prepare_output "$fold" uda "$experiment"; then return 0; fi
  # shellcheck disable=SC2086
  run_command "$LOG_ROOT/${experiment}.log" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" -u train.py \
    -e "$experiment" --data_root "$DATA_ROOT" --source "$source_data" --target "$target_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 $options \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shape-representation phase_moment --structure-shift-mode none \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0 --source-minority-mode base \
    --seed 1 --num_folds 1 --batch_size 128 --seq_length 30 --num_pixels 64 \
    --closed_set true --with_shift_aug false --progress_bar off \
    --output_dir "$UDA_ROOT" --tensorboard_log_dir "$RUN_ROOT/$experiment" \
    timematch --weights "$weights" --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 --estimate_shift true \
    --balance_source true --use_focal_loss true --shift_source true --sample_size 100 \
    --max_temporal_shift 60 --domain_specific_bn true --shift_estimator AM \
    --run_validation --output_student true --shape-da-mode batch_align \
    --shape-alignment-view none --oracle-pseudo-labels false \
    --adaptive-pseudo-selection false --shape-equivariance-weight 0.05 \
    --shape-equivariance-max-shift 60 --detach-target-structure false \
    --target-structure-detach-epochs "$detach_epochs"
  if [[ "$DRY_RUN" != "1" ]]; then
    checkpoint_status "$fold" uda || {
      echo "ERROR: UDA did not complete: $experiment" >&2; exit 1;
    }
  fi
}

wait_round() {
  local name="$1" status=0 pid
  shift
  for pid in "$@"; do wait "$pid" || status=1; done
  if (( status != 0 )); then
    echo "ERROR: round failed: $name" >&2
    exit 1
  fi
  echo "ROUND_FINISHED|round=$name"
}

run_source_round() {
  echo "ROUND_START|round=SOURCE_QS"
  run_source "$GPU0" Q FR1 "$FR1" & P0=$!
  run_source "$GPU1" Q FR2 "$FR2" & P1=$!
  run_source "$GPU2" S FR1 "$FR1" & P2=$!
  run_source "$GPU3" S FR2 "$FR2" & P3=$!
  wait_round SOURCE_QS "$P0" "$P1" "$P2" "$P3"
  echo "ROUND_START|round=SOURCE_R"
  run_source "$GPU0" R FR1 "$FR1" & P0=$!
  run_source "$GPU1" R FR2 "$FR2" & P1=$!
  wait_round SOURCE_R "$P0" "$P1"
}

run_uda_round() {
  require_source "$E_SOURCE_ROOT/source_FR1_seed1"
  require_source "$E_SOURCE_ROOT/source_FR2_seed1"
  echo "ROUND_START|round=UDA_GQ"
  run_uda "$GPU0" G FR1 "$FR1" FR2 "$FR2" & P0=$!
  run_uda "$GPU1" G FR2 "$FR2" DK1 "$DK1" & P1=$!
  run_uda "$GPU2" Q FR1 "$FR1" FR2 "$FR2" & P2=$!
  run_uda "$GPU3" Q FR2 "$FR2" DK1 "$DK1" & P3=$!
  wait_round UDA_GQ "$P0" "$P1" "$P2" "$P3"
  echo "ROUND_START|round=UDA_SR"
  run_uda "$GPU0" S FR1 "$FR1" FR2 "$FR2" & P0=$!
  run_uda "$GPU1" S FR2 "$FR2" DK1 "$DK1" & P1=$!
  run_uda "$GPU2" R FR1 "$FR1" FR2 "$FR2" & P2=$!
  run_uda "$GPU3" R FR2 "$FR2" DK1 "$DK1" & P3=$!
  wait_round UDA_SR "$P0" "$P1" "$P2" "$P3"
}

run_g_extend_round() {
  require_source "$E_SOURCE_ROOT/source_AT1_seed1"
  require_source "$E_SOURCE_ROOT/source_DK1_seed1"
  echo "ROUND_START|round=G_EXTEND"
  run_uda "${GPU0}" G AT1 "$AT1" DK1 "$DK1" & P0=$!
  run_uda "${GPU1}" G DK1 "$DK1" AT1 "$AT1" & P1=$!
  wait_round G_EXTEND "$P0" "$P1"
}

case "$RUN_ROUND" in
  SOURCE) run_source_round ;;
  UDA) run_uda_round ;;
  G_EXTEND) run_g_extend_round ;;
  ALL) run_source_round; run_uda_round ;;
  *) echo "ERROR: RUN_ROUND must be SOURCE, UDA, G_EXTEND, or ALL; got $RUN_ROUND" >&2; exit 2 ;;
esac
