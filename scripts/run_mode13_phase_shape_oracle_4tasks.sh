#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"; DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"; GPU3="${GPU3:-3}"
AT1_WEIGHTS="${AT1_WEIGHTS:-outputs/pseltae_AT1_source_seed1}"
DK1_WEIGHTS="${DK1_WEIGHTS:-outputs/pseltae_DK1_source_seed1}"
FR1_WEIGHTS="${FR1_WEIGHTS:-outputs/pseltae_FR1_source_seed1}"
FR2_WEIGHTS="${FR2_WEIGHTS:-outputs/pseltae_FR2_source_seed1}"
OUT="outputs/phase_shape_diagnostic_mode13_seed1"; LOG="logs/phase_shape_diagnostic_mode13_seed1"
command -v "$PYTHON_BIN" >/dev/null || { echo "ERROR: Python missing" >&2; exit 1; }
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: DATA_ROOT missing: $DATA_ROOT" >&2; exit 1; }
"$PYTHON_BIN" -c 'import torch, numpy, scipy, matplotlib, fdasrsf' || { echo "ERROR: required local dependency missing" >&2; exit 1; }
for weights in "$AT1_WEIGHTS" "$DK1_WEIGHTS" "$FR1_WEIGHTS" "$FR2_WEIGHTS"; do
  checkpoint="${weights}/fold_0/model.pt"
  [[ -f "$checkpoint" && -f "${weights}/train_config.json" ]] || { echo "ERROR: checkpoint/config missing: $checkpoint" >&2; exit 1; }
done
mkdir -p "$OUT" "$LOG"
run(){
  local gpu="$1"
  local source="$2"
  local target="$3"
  local checkpoint="$4"
  local task="${source}_${target}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/diagnose_mode13_phase_shape_oracle.py \
    --data-root "$DATA_ROOT" --source "$source" --target "$target" --checkpoint "$checkpoint" \
    --device cuda --seed 1 --batch-size 128 --num-pixels 64 --output-dir "$OUT/$task" > "$LOG/$task.log" 2>&1
}
run "$GPU0" AT1 DK1 "$AT1_WEIGHTS/fold_0/model.pt" & p0=$!
run "$GPU1" DK1 FR1 "$DK1_WEIGHTS/fold_0/model.pt" & p1=$!
run "$GPU2" FR1 FR2 "$FR1_WEIGHTS/fold_0/model.pt" & p2=$!
run "$GPU3" FR2 AT1 "$FR2_WEIGHTS/fold_0/model.pt" & p3=$!
failed=0
for item in "AT1_DK1:$p0" "DK1_FR1:$p1" "FR1_FR2:$p2" "FR2_AT1:$p3"; do
  task="${item%%:*}"; pid="${item##*:}"; if wait "$pid"; then echo "SUCCESS $task"; else echo "FAILED $task" >&2; failed=1; fi
done
exit "$failed"
