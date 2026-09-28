#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
EXP_ROOT="${EXP_ROOT:-outputs/structure_query_only_2tasks_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_query_only_2tasks_seed1}"
RUN_ROOT="${RUN_ROOT:-runs/structure_query_only_2tasks_seed1}"
UQ_ROOT="${UQ_ROOT:-outputs/structure_local_query_shift_2tasks_seed1/uda}"
DRY_RUN="${DRY_RUN:-0}"
AT1_UQ_CHECKPOINT="$UQ_ROOT/uq_AT1_DK1_seed1/fold_0/model.pt"
FR2_UQ_CHECKPOINT="$UQ_ROOT/uq_FR2_DK1_seed1/fold_0/model.pt"

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR2="france/31TCJ/2017"

mkdir -p "$EXP_ROOT/source" "$EXP_ROOT/uda" "$EXP_ROOT/audit" \
  "$LOG_ROOT/source" "$LOG_ROOT/uda" "$LOG_ROOT/audit" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

source_weights_path() {
  local source="$1"
  echo "$EXP_ROOT/source/$source/source_${source}_seed1"
}

manifest_matches() {
  local manifest="$1"
  "$PYTHON_BIN" - "$manifest" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
expected = {
    "method": "structure_query_only",
    "shape_injection": "local_query_only",
    "structure_shift_mode": "none",
    "fourier_num_modes": 13,
    "shape_window_scales": [24],
    "shape_window_stride": 8,
    "shapelet_count": 16,
    "shapelet_beta": 5.0,
    "shape_resample_length": 16,
    "shape_align_weight": 0.0,
    "local_structure_token_dim": 16,
    "local_structure_windows": 8,
    "local_query": "structure_only",
    "shared_temporal_memory": True,
    "shared_ltae_mlp": True,
}
raise SystemExit(0 if all(value.get(key) == item for key, item in expected.items()) else 1)
PY
}

run_source() {
  local gpu="$1" source="$2" source_data="$3"
  local weights checkpoint manifest log_file
  weights="$(source_weights_path "$source")"
  checkpoint="$weights/fold_0/model.pt"
  manifest="$weights/fold_0/manifest.json"
  log_file="$LOG_ROOT/source/${source}.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "QUERY_ONLY_SOURCE_PLAN|gpu=$gpu|source=$source|weights=$weights"
    return
  fi
  if [[ -f "$checkpoint" && -f "$manifest" ]]; then
    manifest_matches "$manifest" || {
      echo "ERROR: source manifest mismatch: $manifest" >&2
      return 1
    }
    echo "[SOURCE REUSE] $source $checkpoint" | tee -a "$log_file"
    return
  fi
  [[ ! -e "$weights" ]] || {
    echo "ERROR: incomplete source output exists: $weights" >&2
    return 1
  }
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "source_${source}_seed1" \
    --data_root "$DATA_ROOT" --source "$source_data" --target "$source_data" \
    --model psestructureprotoltae --structure-branch true \
    --structure-exposer fourier --fourier_num_modes 13 --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shape-representation current --shape-injection local_query_only \
    --structure-shift-mode none --shape-align-weight 0.0 \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --with_shift_aug false \
    --seed 1 --num_folds 1 --epochs 100 --batch_size 128 --lr 0.001 \
    --weight_decay 0.0001 --focal_loss_gamma 1.0 \
    --seq_length 30 --num_pixels 64 --closed_set true --progress_bar off \
    --output_dir "$EXP_ROOT/source/$source" \
    --tensorboard_log_dir "$RUN_ROOT/source/$source" \
    > "$log_file" 2>&1
  [[ -f "$checkpoint" ]] || {
    echo "ERROR: source checkpoint not found after training: $checkpoint" >&2
    return 1
  }
}

run_audit() {
  local gpu="$1" task="$2"
  local checkpoint
  case "$task" in
    AT1_DK1) checkpoint="$AT1_UQ_CHECKPOINT" ;;
    FR2_DK1) checkpoint="$FR2_UQ_CHECKPOINT" ;;
    *) echo "ERROR: unsupported audit task: $task" >&2; return 1 ;;
  esac
  local log_file="$LOG_ROOT/audit/${task}.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "LOCAL_QUERY_AUDIT_PLAN|gpu=$gpu|task=$task|checkpoint=$checkpoint"
    return
  fi
  [[ -f "$checkpoint" ]] || {
    echo "ERROR: UQ best checkpoint not found: $checkpoint" >&2
    return 1
  }
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u \
    analysis/local_structure_query_audit.py \
    --task "$task" --checkpoint-root "$UQ_ROOT" \
    --output-root "$EXP_ROOT/audit" --data-root "$DATA_ROOT" \
    --device cuda --seed 1 > "$log_file" 2>&1
}

run_uda() {
  local gpu="$1" source="$2" source_data="$3" target="$4" target_data="$5"
  local task="${source}_${target}" weights log_file
  weights="$(source_weights_path "$source")"
  log_file="$LOG_ROOT/uda/${task}.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "QUERY_ONLY_UDA_PLAN|gpu=$gpu|task=$task|weights=$weights"
    return
  fi
  [[ -f "$weights/fold_0/model.pt" ]] || {
    echo "ERROR: query-only source checkpoint not found: $weights/fold_0/model.pt" >&2
    return 1
  }
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "${task}_seed1" \
    --data_root "$DATA_ROOT" --source "$source_data" --target "$target_data" \
    --model psestructureprotoltae --structure-branch true \
    --structure-exposer fourier --fourier_num_modes 13 --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shape-representation current --shape-injection local_query_only \
    --structure-shift-mode none --shape-align-weight 0.0 \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --with_shift_aug false \
    --seed 1 --num_folds 1 --batch_size 128 --seq_length 30 --num_pixels 64 \
    --closed_set true --progress_bar off --output_dir "$EXP_ROOT/uda" \
    --tensorboard_log_dir "$RUN_ROOT/uda/${task}" \
    timematch --weights "$weights" --shape-da-mode batch_align \
    --oracle-pseudo-labels false --adaptive-pseudo-selection false \
    --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true \
    --shift_source true --sample_size 100 --max_temporal_shift 60 \
    --domain_specific_bn true --shift_estimator AM --run_validation \
    --output_student true > "$log_file" 2>&1
}

run_source "$GPU0" AT1 "$AT1" & PID0=$!
run_source "$GPU1" FR2 "$FR2" & PID1=$!
run_audit "$GPU2" AT1_DK1 & PID2=$!
run_audit "$GPU3" FR2_DK1 & PID3=$!
status=0
wait "$PID0" || status=1
wait "$PID1" || status=1
wait "$PID2" || status=1
wait "$PID3" || status=1
[[ "$status" == "0" ]] || exit "$status"

run_uda "$GPU0" AT1 "$AT1" DK1 "$DK1" & PID4=$!
run_uda "$GPU1" FR2 "$FR2" DK1 "$DK1" & PID5=$!
status=0
wait "$PID4" || status=1
wait "$PID5" || status=1
exit "$status"
