#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/structure_prediction_drift_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_prediction_drift_seed1}"
BATCH_SIZE="${BATCH_SIZE:-128}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
PART_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$PART_ROOT"' EXIT

run_audit() {
  local gpu="$1" method="$2" task="$3"
  local name="${method}_${task}"
  CUDA_VISIBLE_DEVICES="$gpu" python -u analysis/structure_prediction_drift_audit.py audit \
    --method "$method" --task "$task" \
    --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" \
    --output-part "$PART_ROOT/$name" --device cuda --batch-size "$BATCH_SIZE" \
    > "$LOG_ROOT/$name.log" 2>&1 &
  LAST_PID="$!"
}

pids=()
run_audit 0 v2 DK1_AT1; pids+=("$LAST_PID")
run_audit 1 v2clean DK1_AT1; pids+=("$LAST_PID")
run_audit 2 v2 FR1_FR2; pids+=("$LAST_PID")
run_audit 3 v2clean FR1_FR2; pids+=("$LAST_PID")
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then
  echo "ERROR: one or more structure-drift audits failed; inspect $LOG_ROOT" >&2
  exit 1
fi

run_audit 0 state_org_foundation_presence AT1_DK1
wait "$LAST_PID"

python -u analysis/structure_prediction_drift_audit.py merge \
  --parts-root "$PART_ROOT" --output-root "$OUTPUT_ROOT" \
  > "$LOG_ROOT/merge.log" 2>&1

echo "STRUCTURE_DRIFT_AUDIT_COMPLETE|output=$OUTPUT_ROOT"
