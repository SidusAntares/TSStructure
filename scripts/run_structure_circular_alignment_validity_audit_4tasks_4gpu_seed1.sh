#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs/structure_proto_v2clean_4tasks_seed1/source}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/structure_circular_alignment_validity_audit_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_circular_alignment_validity_audit_seed1}"
PIXEL_BUDGET="${PIXEL_BUDGET:-8192}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1

run_task() {
  local gpu="$1" task="$2"
  local source="${task%%_*}"
  local checkpoint="$CHECKPOINT_ROOT/source_${source}_seed1/fold_0/model.pt"
  local log_file="$LOG_ROOT/${task}.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "CIRCULAR_ALIGNMENT_PLAN|gpu=$gpu|task=$task|checkpoint=$checkpoint|output=$OUTPUT_ROOT"
    return 0
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "MISSING|$checkpoint" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u \
    analysis/structure_circular_alignment_validity_audit.py \
    --task "$task" --data-root "$DATA_ROOT" \
    --checkpoint-root "$CHECKPOINT_ROOT" --output-root "$OUTPUT_ROOT" \
    --pixel-budget "$PIXEL_BUDGET" --sample-size "$SAMPLE_SIZE" \
    --device cuda --seed 1 > "$log_file" 2>&1
}

if [[ "$DRY_RUN" == "1" ]]; then
  run_task "$GPU0" AT1_DK1
  run_task "$GPU1" FR1_FR2
  run_task "$GPU2" FR2_DK1
  run_task "$GPU3" DK1_AT1
  exit 0
fi

run_task "$GPU0" AT1_DK1 & PID0=$!
run_task "$GPU1" FR1_FR2 & PID1=$!
run_task "$GPU2" FR2_DK1 & PID2=$!
run_task "$GPU3" DK1_AT1 & PID3=$!

status=0
wait "$PID0" || status=1
wait "$PID1" || status=1
wait "$PID2" || status=1
wait "$PID3" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "ERROR: circular-alignment audit failed; inspect $LOG_ROOT/*.log" >&2
  exit "$status"
fi

"$PYTHON_BIN" -u analysis/structure_circular_alignment_validity_audit.py \
  --merge-only --output-root "$OUTPUT_ROOT" --seed 1 \
  | tee "$LOG_ROOT/summary.log"
