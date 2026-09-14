#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/structure_reference_validity}"
VIEW_ROOT="${VIEW_ROOT:-outputs/shift_visualizations_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_reference_validity}"
RUN_STRUCTURE_REFERENCE_VALIDITY="${RUN_STRUCTURE_REFERENCE_VALIDITY:-1}"

AT1_WEIGHTS="${AT1_WEIGHTS:-outputs/pseltae_AT1_source_seed1}"
DK1_WEIGHTS="${DK1_WEIGHTS:-outputs/pseltae_DK1_source_seed1}"
FR1_WEIGHTS="${FR1_WEIGHTS:-outputs/pseltae_FR1_source_seed1}"
FR2_WEIGHTS="${FR2_WEIGHTS:-outputs/pseltae_FR2_source_seed1}"

BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-100}"
BOOTSTRAP_FRACTION="${BOOTSTRAP_FRACTION:-0.70}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-1}"

[[ "$RUN_STRUCTURE_REFERENCE_VALIDITY" == "1" ]] || {
    echo "RUN_STRUCTURE_REFERENCE_VALIDITY is disabled"
    exit 0
}
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: data root not found: $DATA_ROOT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1

for checkpoint_root in "$AT1_WEIGHTS" "$DK1_WEIGHTS" "$FR1_WEIGHTS" "$FR2_WEIGHTS"; do
    [[ -f "$checkpoint_root/fold_0/model.pt" ]] || {
        echo "ERROR: source checkpoint not found: $checkpoint_root/fold_0/model.pt" >&2
        exit 1
    }
    [[ -f "$checkpoint_root/train_config.json" ]] || {
        echo "ERROR: source config not found: $checkpoint_root/train_config.json" >&2
        exit 1
    }
done

run_one() {
    local gpu="$1"
    local source="$2"
    local reference_task="$3"
    local weights="$4"
    echo "[START] GPU${gpu} source=${source} reference=${reference_task}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/diagnose_structure_reference_validity.py \
        --data-root "$DATA_ROOT" \
        --source-domain "$source" \
        --source-checkpoint "$weights" \
        --reference-task "$reference_task" \
        --structure-view-root "$VIEW_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --prototype-bootstrap-repeats "$BOOTSTRAP_REPEATS" \
        --prototype-bootstrap-fraction "$BOOTSTRAP_FRACTION" \
        --prototype-bootstrap-seed "$BOOTSTRAP_SEED" \
        --prototype-match-center-radius-days 30 \
        --prototype-match-max-duration-ratio 2.0 \
        --prototype-high-stability-threshold 0.80 \
        --prototype-low-stability-threshold 0.50 \
        --individual-support-threshold 0.50 \
        --medoid-max-samples 128 \
        --device cuda
    echo "[FINISHED] source=${source}"
}

run_one 0 AT1 AT1_DK1 "$AT1_WEIGHTS" > "$LOG_ROOT/AT1.log" 2>&1 & pid0=$!
run_one 1 DK1 DK1_FR1 "$DK1_WEIGHTS" > "$LOG_ROOT/DK1.log" 2>&1 & pid1=$!
run_one 2 FR1 FR1_FR2 "$FR1_WEIGHTS" > "$LOG_ROOT/FR1.log" 2>&1 & pid2=$!
run_one 3 FR2 FR2_AT1 "$FR2_WEIGHTS" > "$LOG_ROOT/FR2.log" 2>&1 & pid3=$!

status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
    if ! wait "$pid"; then status=1; fi
done
if [[ "$status" -ne 0 ]]; then
    echo "ERROR: one or more 06A source diagnostics failed; inspect $LOG_ROOT" >&2
    exit 1
fi

"$PYTHON_BIN" -u scripts/diagnose_structure_reference_validity.py \
    --summarize-root "$OUTPUT_ROOT"

echo "[ALL FINISHED] 06A structure reference validity: $OUTPUT_ROOT"
