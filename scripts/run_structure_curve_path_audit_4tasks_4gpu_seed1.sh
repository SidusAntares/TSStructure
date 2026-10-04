#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs/structure_phase_moment_4tasks_seed1/source}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/structure_curve_path_audit_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_curve_path_audit_seed1}"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

run_task() {
  local gpu="$1" task="$2"
  echo "CURVE_PATH_AUDIT_START|gpu=$gpu|task=$task"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u \
    analysis/structure_curve_path_audit.py \
    --task "$task" --data-root "$DATA_ROOT" \
    --checkpoint-root "$CHECKPOINT_ROOT" --output-root "$OUTPUT_ROOT" \
    --device cuda --seed 1 > "$LOG_ROOT/${task}.log" 2>&1
}

run_task 0 AT1_DK1 & P0=$!
run_task 1 FR1_FR2 & P1=$!
run_task 2 FR2_DK1 & P2=$!
run_task 3 DK1_AT1 & P3=$!

STATUS=0
for pid in "$P0" "$P1" "$P2" "$P3"; do
  wait "$pid" || STATUS=1
done
if (( STATUS != 0 )); then
  echo "ERROR: curve path audit failed; inspect $LOG_ROOT" >&2
  exit 1
fi

"$PYTHON_BIN" -u analysis/structure_curve_path_audit.py \
  --merge --output-root "$OUTPUT_ROOT" --seed 1
