#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
EXP_ROOT="${EXP_ROOT:-outputs/structure_local_query_shift_2tasks_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_local_query_shift_2tasks_seed1}"
RUN_ROOT="${RUN_ROOT:-runs/structure_local_query_shift_2tasks_seed1}"
DRY_RUN="${DRY_RUN:-0}"

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR2="france/31TCJ/2017"

mkdir -p "$EXP_ROOT/source" "$EXP_ROOT/uda" "$LOG_ROOT/source" "$LOG_ROOT/uda" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

source_weights_path() {
  local src_name="$1"
  echo "$EXP_ROOT/source/${src_name}/source_${src_name}_seed1"
}

source_manifest_matches() {
  local manifest="$1"
  "$PYTHON_BIN" - "$manifest" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    value = json.load(stream)
expected = {
    "method": "local_structure_query_shift",
    "structure_exposer": "fourier",
    "shape_representation": "current",
    "shape_injection": "local_query",
    "structure_shift_mode": "none",
    "fourier_num_modes": 13,
    "shape_dim": 128,
    "shape_window_scales": [24],
    "shape_window_stride": 8,
    "shapelet_count": 16,
    "shapelet_beta": 5.0,
    "shape_resample_length": 16,
    "shapelet_diversity_margin": 0.5,
    "shapelet_diversity_weight": 0.01,
    "shape_class_weight": 0.1,
    "shape_align_weight": 0.0,
    "local_structure_token_dim": 16,
    "local_structure_windows": 8,
    "local_query": "base_plus_local_delta",
    "shared_temporal_memory": True,
    "shared_ltae_mlp": True,
    "structure_gamma_max": 0.5,
}
raise SystemExit(0 if all(value.get(key) == item for key, item in expected.items()) else 1)
PY
}

run_source() {
  local gpu="$1" src_name="$2" src_data="$3"
  local source_root="$EXP_ROOT/source/${src_name}"
  local source_experiment="source_${src_name}_seed1"
  local source_weights
  source_weights="$(source_weights_path "$src_name")"
  local checkpoint="$source_weights/fold_0/model.pt"
  local manifest="$source_weights/fold_0/manifest.json"
  local log_file="$LOG_ROOT/source/${src_name}.log"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "LOCAL_QUERY_SOURCE_PLAN|gpu=$gpu|source=$src_name|shared_by=uq,sq"
    return 0
  fi
  if [[ -f "$checkpoint" && -f "$manifest" ]]; then
    if source_manifest_matches "$manifest"; then
      echo "[SOURCE REUSE] $src_name $checkpoint" | tee -a "$log_file"
      return 0
    fi
    echo "ERROR: source checkpoint manifest mismatch: $manifest" | tee -a "$log_file"
    return 1
  fi
  if [[ -e "$source_weights" ]]; then
    echo "ERROR: incomplete source output exists: $source_weights" | tee -a "$log_file"
    return 1
  fi

  mkdir -p "$source_root" "$RUN_ROOT/source/${src_name}"
  : > "$log_file"
  echo "[LOCAL QUERY SOURCE START] $src_name GPU$gpu" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "$source_experiment" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$src_data" \
    --model psestructureprotoltae \
    --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shape-representation current --shape-injection local_query \
    --structure-shift-mode none \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0.0 \
    --with_shift_aug false --seed 1 --num_folds 1 --epochs 100 \
    --batch_size 128 --lr 0.001 --weight_decay 0.0001 \
    --focal_loss_gamma 1.0 --seq_length 30 --num_pixels 64 \
    --closed_set true --progress_bar off \
    --output_dir "$source_root" \
    --tensorboard_log_dir "$RUN_ROOT/source/${src_name}" \
    2>&1 | tee -a "$log_file"
  [[ -f "$checkpoint" ]] || {
    echo "ERROR: source checkpoint not found: $checkpoint" | tee -a "$log_file"
    return 1
  }
}

run_uda() {
  local gpu="$1" variant="$2" shift_mode="$3" src_name="$4" src_data="$5" tgt_name="$6" tgt_data="$7"
  local task="${src_name}_${tgt_name}"
  local experiment="${variant}_${task}_seed1"
  local source_weights="$EXP_ROOT/source/${src_name}/source_${src_name}_seed1"
  local log_file="$LOG_ROOT/uda/${experiment}.log"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "LOCAL_QUERY_UDA_PLAN|gpu=$gpu|variant=$variant|structure_shift_mode=$shift_mode|task=$task|source_weights=$source_weights"
    return 0
  fi
  [[ -f "$source_weights/fold_0/model.pt" ]] || {
    echo "ERROR: shared source checkpoint not found: $source_weights/fold_0/model.pt" | tee -a "$log_file"
    return 1
  }

  : > "$log_file"
  echo "[LOCAL QUERY UDA START] $variant $src_name -> $tgt_name GPU$gpu" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "$experiment" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae \
    --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shape-representation current --shape-injection local_query \
    --structure-shift-mode "$shift_mode" \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0.0 \
    --with_shift_aug false --seed 1 --num_folds 1 --batch_size 128 \
    --seq_length 30 --num_pixels 64 --closed_set true --progress_bar off \
    --output_dir "$EXP_ROOT/uda" \
    --tensorboard_log_dir "$RUN_ROOT/uda/${experiment}" \
    timematch --weights "$source_weights" \
    --shape-da-mode batch_align \
    --oracle-pseudo-labels false --adaptive-pseudo-selection false \
    --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true \
    --shift_source true --sample_size 100 --max_temporal_shift 60 \
    --domain_specific_bn true --shift_estimator AM --run_validation \
    --output_student true \
    2>&1 | tee -a "$log_file"
}

run_source "$GPU0" FR2 "$FR2" & SOURCE0=$!
run_source "$GPU1" AT1 "$AT1" & SOURCE1=$!
source_status=0
wait "$SOURCE0" || source_status=1
wait "$SOURCE1" || source_status=1
[[ "$source_status" == "0" ]] || exit "$source_status"

run_uda "$GPU0" uq none AT1 "$AT1" DK1 "$DK1" & PID0=$!
run_uda "$GPU1" uq none FR2 "$FR2" DK1 "$DK1" & PID1=$!
run_uda "$GPU2" sq timematch AT1 "$AT1" DK1 "$DK1" & PID2=$!
run_uda "$GPU3" sq timematch FR2 "$FR2" DK1 "$DK1" & PID3=$!

status=0
wait "$PID0" || status=1
wait "$PID1" || status=1
wait "$PID2" || status=1
wait "$PID3" || status=1
exit "$status"
