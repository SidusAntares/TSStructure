#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
CURRENT_ROOT="${CURRENT_ROOT:-outputs/structure_proto_v2clean_4tasks_seed1/source}"
SET_ROOT="${SET_ROOT:-outputs/structure_set_response_4tasks_seed1/source}"
RESIDUAL_ROOT="${RESIDUAL_ROOT:-outputs/structure_residual_response_4tasks_seed1/source}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/structure_response_query_chain_audit_3variants_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_response_query_chain_audit_3variants_seed1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIXEL_BUDGET="${PIXEL_BUDGET:-8192}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" != "1" ]]; then
  for variant_root in "$CURRENT_ROOT" "$SET_ROOT" "$RESIDUAL_ROOT"; do
    for source in AT1 FR1 FR2 DK1; do
      checkpoint="$variant_root/source_${source}_seed1/fold_0/model.pt"
      if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: source checkpoint not found: $checkpoint" >&2
        exit 1
      fi
    done
  done
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1

run_variant_task() {
  local gpu="$1" variant="$2" task="$3" checkpoint_root="$4"
  local log_file="$LOG_ROOT/${variant}_${task}.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "REP_CHAIN_PLAN|gpu=$gpu|variant=$variant|task=$task|pixel_budget=$PIXEL_BUDGET|checkpoint_root=$checkpoint_root|output_root=$OUTPUT_ROOT"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u \
    analysis/structure_representation_chain_audit.py \
    --variant "$variant" --task "$task" \
    --data-root "$DATA_ROOT" \
    --checkpoint-root "$checkpoint_root" \
    --output-root "$OUTPUT_ROOT" \
    --pixel-budget "$PIXEL_BUDGET" \
    --device cuda --seed 1 \
    > "$log_file" 2>&1
}

run_task() {
  local gpu="$1" task="$2"
  run_variant_task "$gpu" current "$task" "$CURRENT_ROOT"
  run_variant_task "$gpu" set_response "$task" "$SET_ROOT"
  run_variant_task "$gpu" residual_response "$task" "$RESIDUAL_ROOT"
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
  echo "ERROR: representation-chain audit failed; inspect $LOG_ROOT/*.log" >&2
  exit "$status"
fi

"$PYTHON_BIN" -u analysis/structure_representation_chain_audit.py \
  --merge --output-root "$OUTPUT_ROOT" --seed 1 \
  | tee "$LOG_ROOT/summary.log"
