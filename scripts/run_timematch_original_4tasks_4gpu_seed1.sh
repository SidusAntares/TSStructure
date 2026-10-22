#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
SEED=1
EXP_ROOT="${EXP_ROOT:-outputs/timematch_original_4tasks_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/timematch_original_4tasks_seed1}"
RUN_ROOT="${RUN_ROOT:-runs/timematch_original_4tasks_seed1}"
ATTENTION_ROOT="${ATTENTION_ROOT:-outputs/timematch_ltae_attention_vis}"
AUTO_VISUALIZE="${AUTO_VISUALIZE:-1}"
VIS_DEVICE="${VIS_DEVICE:-cuda:0}"

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"

[[ "$AUTO_VISUALIZE" == "0" || "$AUTO_VISUALIZE" == "1" ]] || {
  echo "ERROR: AUTO_VISUALIZE must be 0 or 1" >&2
  exit 2
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
}
[[ -d "$DATA_ROOT" ]] || {
  echo "ERROR: DATA_ROOT not found: $DATA_ROOT" >&2
  exit 1
}

mkdir -p "$EXP_ROOT/source" "$EXP_ROOT/uda" "$LOG_ROOT" "$RUN_ROOT" "$ATTENTION_ROOT"
export PYTHONUNBUFFERED=1

validate_source_checkpoint() {
  local checkpoint="$1" config_path="$2" expected_source="$3"
  [[ -f "$checkpoint" ]] || {
    echo "ERROR: missing Source checkpoint: $checkpoint" >&2
    return 1
  }
  [[ -f "$config_path" ]] || {
    echo "ERROR: missing Source config: $config_path" >&2
    return 1
  }
  "$PYTHON_BIN" - "$config_path" "$expected_source" <<'PY'
import json
import sys

path, expected_source = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    config = json.load(stream)
assert config.get("model") == "pseltae", "Source model must be original PseLTAE"
assert config.get("source") == expected_source
assert config.get("target") == expected_source
assert int(config.get("seed", -1)) == 1
assert int(config.get("epochs", -1)) == 100
print(f"SOURCE_CHECKPOINT_VALID|path={path}|model=pseltae|seed=1")
PY
}

validate_final_checkpoint() {
  local checkpoint="$1" expected_source="$2" expected_target="$3"
  [[ -f "$checkpoint" ]] || {
    echo "ERROR: missing Final Student checkpoint: $checkpoint" >&2
    return 1
  }
  "$PYTHON_BIN" - "$checkpoint" "$expected_source" "$expected_target" <<'PY'
import pathlib
import sys
import torch

path = pathlib.Path(sys.argv[1])
expected_source, expected_target = sys.argv[2:]
assert path.name == "checkpoint_last.pt"
try:
    packet = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    packet = torch.load(path, map_location="cpu")
config = packet.get("config", {})
if not isinstance(config, dict):
    config = vars(config)
assert config.get("model") == "pseltae", "Final model must be original PseLTAE"
assert config.get("source") == expected_source
assert config.get("target") == expected_target
assert int(config.get("seed", -1)) == 1
assert bool(config.get("output_student")) is True
assert int(packet.get("epoch", -1)) == int(config.get("epochs", -1)) - 1
assert "state_dict" in packet and "teacher_state_dict" in packet
assert "global_temporal_shift" in packet
print(
    f"FINAL_CHECKPOINT_VALID|path={path}|epoch={packet['epoch']}|"
    "model=pseltae|output_student=true"
)
PY
}

run_task() {
  local gpu="$1" src_name="$2" src_data="$3" tgt_name="$4" tgt_data="$5"
  local task="${src_name}_${tgt_name}"
  local source_experiment="source_${src_name}_seed${SEED}"
  local source_weights="$EXP_ROOT/source/$source_experiment"
  local source_checkpoint="$source_weights/fold_0/model.pt"
  local source_config="$source_weights/train_config.json"
  local final_checkpoint="$EXP_ROOT/uda/${task}_seed${SEED}/fold_0/checkpoint_last.pt"
  local log_file="$LOG_ROOT/${task}.log"

  : > "$log_file"
  echo "TIMEMATCH_ORIGINAL_PLAN|gpu=$gpu|task=$task|seed=$SEED" | tee -a "$log_file"

  if [[ -f "$source_checkpoint" && -f "$source_config" ]]; then
    validate_source_checkpoint "$source_checkpoint" "$source_config" "$src_data" \
      2>&1 | tee -a "$log_file"
    echo "SOURCE_REUSED|task=$task|path=$source_checkpoint" | tee -a "$log_file"
  else
    if [[ -e "$source_checkpoint" || -e "$source_config" ]]; then
      echo "ERROR: incomplete Source artifact; refusing overwrite: $source_weights" \
        | tee -a "$log_file"
      return 1
    fi
    echo "SOURCE_START|gpu=$gpu|source=$src_name|model=pseltae" | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
      -e "$source_experiment" \
      --data_root "$DATA_ROOT" --source "$src_data" --target "$src_data" \
      --model pseltae --seed "$SEED" --num_folds 1 \
      --epochs 100 --batch_size 128 --lr 0.001 --weight_decay 0.0001 \
      --focal_loss_gamma 1.0 --seq_length 30 --num_pixels 64 \
      --closed_set true --combine_spring_and_winter false \
      --with_shift_aug false --progress_bar off --device cuda \
      --output_dir "$EXP_ROOT/source" \
      --tensorboard_log_dir "$RUN_ROOT/source_${src_name}_seed${SEED}" \
      2>&1 | tee -a "$log_file"
    validate_source_checkpoint "$source_checkpoint" "$source_config" "$src_data" \
      2>&1 | tee -a "$log_file"
  fi

  if [[ -f "$final_checkpoint" ]]; then
    validate_final_checkpoint "$final_checkpoint" "$src_data" "$tgt_data" \
      2>&1 | tee -a "$log_file"
    echo "UDA_REUSED|task=$task|path=$final_checkpoint" | tee -a "$log_file"
    return 0
  fi
  if [[ -d "$EXP_ROOT/uda/${task}_seed${SEED}/fold_0" ]]; then
    echo "ERROR: incomplete UDA artifact; refusing overwrite: $EXP_ROOT/uda/${task}_seed${SEED}" \
      | tee -a "$log_file"
    return 1
  fi

  echo "UDA_START|gpu=$gpu|task=$task|model=pseltae" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "${task}_seed${SEED}" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model pseltae --seed "$SEED" --num_folds 1 --batch_size 128 \
    --weight_decay 0.0001 --focal_loss_gamma 1.0 \
    --seq_length 30 --num_pixels 64 --closed_set true \
    --combine_spring_and_winter false --with_shift_aug false \
    --progress_bar off --device cuda \
    --output_dir "$EXP_ROOT/uda" \
    --tensorboard_log_dir "$RUN_ROOT/${task}_seed${SEED}" \
    timematch --weights "$source_weights" \
    --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true \
    --shift_source true --sample_size 100 --max_temporal_shift 60 \
    --domain_specific_bn true --shift_estimator AM --run_validation \
    --output_student true \
    2>&1 | tee -a "$log_file"
  validate_final_checkpoint "$final_checkpoint" "$src_data" "$tgt_data" \
    2>&1 | tee -a "$log_file"
  echo "TIMEMATCH_ORIGINAL_COMPLETE|gpu=$gpu|task=$task|checkpoint=$final_checkpoint" \
    | tee -a "$log_file"
}

echo "LAUNCH|gpu=$GPU0|task=AT1_DK1|log=$LOG_ROOT/AT1_DK1.log"
run_task "$GPU0" AT1 "$AT1" DK1 "$DK1" >/dev/null 2>&1 & PID0=$!
echo "LAUNCH|gpu=$GPU1|task=FR1_FR2|log=$LOG_ROOT/FR1_FR2.log"
run_task "$GPU1" FR1 "$FR1" FR2 "$FR2" >/dev/null 2>&1 & PID1=$!
echo "LAUNCH|gpu=$GPU2|task=FR2_DK1|log=$LOG_ROOT/FR2_DK1.log"
run_task "$GPU2" FR2 "$FR2" DK1 "$DK1" >/dev/null 2>&1 & PID2=$!
echo "LAUNCH|gpu=$GPU3|task=DK1_AT1|log=$LOG_ROOT/DK1_AT1.log"
run_task "$GPU3" DK1 "$DK1" AT1 "$AT1" >/dev/null 2>&1 & PID3=$!

status=0
wait "$PID0" || status=1
wait "$PID1" || status=1
wait "$PID2" || status=1
wait "$PID3" || status=1
if [[ "$status" != "0" ]]; then
  echo "ERROR: one or more original TimeMatch workers failed; inspect $LOG_ROOT" >&2
  exit 1
fi

if [[ "$AUTO_VISUALIZE" == "1" ]]; then
  echo "ATTENTION_VIS_START|checkpoint_root=$EXP_ROOT/uda|device=$VIS_DEVICE"
  CHECKPOINT_ROOT="$EXP_ROOT/uda" OUTPUT_ROOT="$ATTENTION_ROOT" \
    DEVICE="$VIS_DEVICE" SEED="$SEED" PYTHON_BIN="$PYTHON_BIN" \
    bash scripts/run_timematch_ltae_attention_vis.sh \
    2>&1 | tee "$ATTENTION_ROOT/run.log"
fi

echo "TIMEMATCH_ORIGINAL_4TASKS_FINISHED|output=$EXP_ROOT|attention=$ATTENTION_ROOT"
