#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/timematch_ltae_attention_vis}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"
NUM_CLASSES="${NUM_CLASSES:-3}"
PAIRS_PER_CLASS="${PAIRS_PER_CLASS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PCA_MAX_PARCELS="${PCA_MAX_PARCELS:-512}"
PCA_MAX_OBSERVATIONS="${PCA_MAX_OBSERVATIONS:-50000}"
TASKS_TEXT="${TASKS:-AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1}"
CLASSES_TEXT="${CLASSES:-}"

read -r -a TASK_LIST <<< "$TASKS_TEXT"
CLASS_ARGS=()
if [[ -n "$CLASSES_TEXT" ]]; then
  read -r -a CLASS_LIST <<< "$CLASSES_TEXT"
  CLASS_ARGS=(--classes "${CLASS_LIST[@]}")
fi

mkdir -p "$OUTPUT_ROOT"
for task in "${TASK_LIST[@]}"; do
  echo "ATTENTION_VIS_PLAN|task=$task|device=$DEVICE|seed=$SEED"
  checkpoint_variable="CHECKPOINT_${task}"
  checkpoint="${!checkpoint_variable:-}"
  CHECKPOINT_ARGS=()
  if [[ -n "$checkpoint" ]]; then
    CHECKPOINT_ARGS=(--checkpoint "$checkpoint")
  fi
  "$PYTHON_BIN" -u scripts/visualize_timematch_ltae_attention.py run \
    --task "$task" \
    --seed "$SEED" \
    --data-root "$DATA_ROOT" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    "${CHECKPOINT_ARGS[@]}" \
    --output-root "$OUTPUT_ROOT" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --num-classes "$NUM_CLASSES" \
    --pairs-per-class "$PAIRS_PER_CLASS" \
    --pca-max-parcels "$PCA_MAX_PARCELS" \
    --pca-max-observations "$PCA_MAX_OBSERVATIONS" \
    "${CLASS_ARGS[@]}"
done

"$PYTHON_BIN" -u scripts/visualize_timematch_ltae_attention.py finalize \
  --output-root "$OUTPUT_ROOT" \
  --tasks "${TASK_LIST[@]}"
