#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs/structure_proto_v2clean_4tasks_seed1/source}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/structure_representation_chain_audit_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_representation_chain_audit_seed1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIXEL_BUDGET="${PIXEL_BUDGET:-8192}"
DRY_RUN="${DRY_RUN:-0}"

FR2_CHECKPOINT="$CHECKPOINT_ROOT/source_FR2_seed1/fold_0/model.pt"
AT1_CHECKPOINT="$CHECKPOINT_ROOT/source_AT1_seed1/fold_0/model.pt"

if [[ "$DRY_RUN" != "1" ]]; then
  for checkpoint in "$FR2_CHECKPOINT" "$AT1_CHECKPOINT"; do
    if [[ ! -f "$checkpoint" ]]; then
      echo "ERROR: source V2-Clean checkpoint not found: $checkpoint" >&2
      exit 1
    fi
  done
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1

run_task() {
  local gpu="$1" task="$2"
  local log_file="$LOG_ROOT/${task}.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "REP_CHAIN_PLAN|gpu=${gpu}|task=${task}|pixel_budget=${PIXEL_BUDGET}|checkpoint_root=${CHECKPOINT_ROOT}|output_root=${OUTPUT_ROOT}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u \
    analysis/structure_representation_chain_audit.py \
    --task "$task" \
    --data-root "$DATA_ROOT" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --pixel-budget "$PIXEL_BUDGET" \
    --device cuda --seed 1 \
    > "$log_file" 2>&1
}

if [[ "$DRY_RUN" == "1" ]]; then
  run_task "$GPU0" FR2_DK1
  run_task "$GPU1" AT1_DK1
  exit 0
fi

run_task "$GPU0" FR2_DK1 & PID0=$!
run_task "$GPU1" AT1_DK1 & PID1=$!

status=0
wait "$PID0" || status=1
wait "$PID1" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "ERROR: representation-chain audit failed; inspect $LOG_ROOT/*.log" >&2
  exit "$status"
fi

"$PYTHON_BIN" -u analysis/structure_representation_chain_audit.py \
  --merge --output-root "$OUTPUT_ROOT" --seed 1 \
  | tee "$LOG_ROOT/summary.log"
