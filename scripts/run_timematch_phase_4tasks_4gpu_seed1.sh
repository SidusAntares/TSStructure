#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
ALPHA_CANDIDATES="${ALPHA_CANDIDATES:-0}"
GEOMETRY_CONFIG="${GEOMETRY_CONFIG:-$PROJECT_DIR/configs/timematch_phase_geometry.json}"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-timematch_phase_alpha_${ALPHA_CANDIDATES//,/p}_seed1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/$EXPERIMENT_GROUP}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/$EXPERIMENT_GROUP}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REGISTRATION_WORKERS="${REGISTRATION_WORKERS:-4}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"; GPU3="${GPU3:-3}"

if [[ "$ALPHA_CANDIDATES" != "0" && ! -f "$GEOMETRY_CONFIG" ]]; then
  echo "TIMEMATCH_PHASE_CONFIG_ERROR|reason=nonzero_alpha_requires_geometry_config|path=$GEOMETRY_CONFIG" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

TASKS=(
  "AT1_DK1|austria/33UVP/2017|denmark/32VNH/2017|$GPU0"
  "DK1_FR2|denmark/32VNH/2017|france/31TCJ/2017|$GPU1"
  "FR1_AT1|france/30TXT/2017|austria/33UVP/2017|$GPU2"
  "FR2_FR1|france/31TCJ/2017|france/30TXT/2017|$GPU3"
)

run_task() {
  local name="$1" source="$2" target="$3" gpu="$4"
  local task_root="$OUTPUT_ROOT/$name"
  local source_root="$task_root/stage1_source"
  local source_checkpoint="$source_root/source/fold_0/model.pt"
  local stage2_root="$task_root/stage2"
  local log="$LOG_ROOT/${name,,}.log"
  local geometry_args=()
  if [[ -n "$GEOMETRY_CONFIG" ]]; then geometry_args=(--geometry-config "$GEOMETRY_CONFIG"); fi
  if [[ -f "$stage2_root/test_metrics.json" ]]; then
    echo "TIMEMATCH_PHASE_TASK_SKIP|task=$name|reason=complete_metrics|metrics=$stage2_root/test_metrics.json"
    return 0
  fi

  {
    echo "TIMEMATCH_PHASE_TASK_START|task=$name|gpu=$gpu|alpha=$ALPHA_CANDIDATES|seed=1"
    if [[ ! -f "$source_checkpoint" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/train_original_timematch_source.py \
        --data_root "$DATA_ROOT" --source "$source" --target "$target" \
        --output_dir "$source_root" --tensorboard_log_dir "$source_root/runs" \
        --experiment_name source --seed 1 --num_folds 1 --epochs 100 \
        --batch_size 128 --lr 0.001 --weight_decay 0.0001 \
        --num_pixels 64 --seq_length 30 --model pseltae --with_extra false \
        --closed_set true --combine_spring_and_winter false \
        --with_shift_aug false --progress_bar off
    else
      echo "TIMEMATCH_PHASE_STAGE1_SKIP|task=$name|checkpoint=$source_checkpoint"
    fi
    python -u scripts/train_timematch_phase.py \
      --device "cuda:$gpu" --data-root "$DATA_ROOT" \
      --source "$source" --target "$target" \
      --model-checkpoint "$source_checkpoint" --output-dir "$stage2_root" \
      --alpha-candidates "$ALPHA_CANDIDATES" "${geometry_args[@]}" \
      --epochs 20 --steps-per-epoch 500 --batch-size 128 \
      --num-workers "$NUM_WORKERS" --registration-workers "$REGISTRATION_WORKERS"
    echo "TIMEMATCH_PHASE_TASK_DONE|task=$name|metrics=$stage2_root/test_metrics.json"
  } > "$log" 2>&1
}

pids=()
for spec in "${TASKS[@]}"; do
  IFS='|' read -r name source target gpu <<<"$spec"
  run_task "$name" "$source" "$target" "$gpu" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
if [[ "$failed" -ne 0 ]]; then
  echo "TIMEMATCH_PHASE_EXPERIMENT_DONE|status=FAILED|failed=$failed|tasks=4" >&2
  exit 3
fi
echo "TIMEMATCH_PHASE_EXPERIMENT_DONE|status=SUCCESS|tasks=4|alpha=$ALPHA_CANDIDATES|output=$OUTPUT_ROOT"
