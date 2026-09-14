#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
VIEW_ROOT="${VIEW_ROOT:-outputs/shift_visualizations_seed1}"
VALIDITY_ROOT="${VALIDITY_ROOT:-outputs/06A_structure_reference_validity}"
OBSERVATION_ROOT="${OBSERVATION_ROOT:-$VIEW_ROOT/06B_structure_observation_support}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$VIEW_ROOT/06C_structure_identity_phase}"
LOG_ROOT="${LOG_ROOT:-logs/structure_identity_phase_06C}"
AT1_WEIGHTS="${AT1_WEIGHTS:-outputs/pseltae_AT1_source_seed1}"
DK1_WEIGHTS="${DK1_WEIGHTS:-outputs/pseltae_DK1_source_seed1}"
FR1_WEIGHTS="${FR1_WEIGHTS:-outputs/pseltae_FR1_source_seed1}"
FR2_WEIGHTS="${FR2_WEIGHTS:-outputs/pseltae_FR2_source_seed1}"
IDENTITY_COMPONENT_SCALE_QUANTILE="${IDENTITY_COMPONENT_SCALE_QUANTILE:-0.90}"
IDENTITY_COST_QUANTILE="${IDENTITY_COST_QUANTILE:-0.95}"
IDENTITY_CALIBRATION_MIN_PAIRS="${IDENTITY_CALIBRATION_MIN_PAIRS:-50}"
export PYTHONUNBUFFERED=1

[[ -d "$DATA_ROOT" ]] || { echo "ERROR: missing source data: $DATA_ROOT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 1; }
for checkpoint_root in "$AT1_WEIGHTS" "$DK1_WEIGHTS" "$FR1_WEIGHTS" "$FR2_WEIGHTS"; do
    for suffix in train_config.json fold_0/model.pt; do
        [[ -f "$checkpoint_root/$suffix" ]] || { echo "ERROR: missing source checkpoint/config: $checkpoint_root/$suffix" >&2; exit 1; }
    done
done
for task in AT1_DK1 DK1_FR1 FR1_FR2 FR2_AT1; do
    for suffix in manifest.json sample_structure_support.csv; do
        [[ -f "$OBSERVATION_ROOT/$task/$suffix" ]] || { echo "ERROR: missing 06B input: $OBSERVATION_ROOT/$task/$suffix" >&2; exit 1; }
    done
done
for source_alias in AT1 DK1 FR1 FR2; do
    [[ -f "$VALIDITY_ROOT/$source_alias/structure_stability.csv" ]] || {
        echo "ERROR: missing 06A input: $VALIDITY_ROOT/$source_alias/structure_stability.csv" >&2; exit 1;
    }
done
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
WORK_ROOT="$(mktemp -d "$OUTPUT_ROOT/.work_06C_XXXXXXXX")"
common=(--work-root "$WORK_ROOT" --output-root "$OUTPUT_ROOT")

prepare_one() {
    local gpu="$1"
    local task="$2"
    local weights="$3"
    echo "[PREPARE] GPU${gpu} ${task}: source only"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/diagnose_structure_identity_phase.py \
        --stage prepare --task "$task" "${common[@]}" \
        --data-root "$DATA_ROOT" --source-checkpoint "$weights" \
        --structure-view-root "$VIEW_ROOT" --validity-root "$VALIDITY_ROOT" \
        --observation-root "$OBSERVATION_ROOT" --seed 1 --device cuda
}

wait_wave() {
    local failed=0
    local pid
    for pid in "$@"; do
        if ! wait "$pid"; then failed=1; fi
    done
    if [[ "$failed" != 0 ]]; then
        echo "ERROR: stage failed; previous published outputs preserved. Logs: $LOG_ROOT; work: $WORK_ROOT" >&2
        return 1
    fi
}

echo "[STAGE 1] Prepare all four sources; work=$WORK_ROOT"
prepare_one 0 AT1_DK1 "$AT1_WEIGHTS" > "$LOG_ROOT/AT1_DK1.log" 2>&1 & p0=$!
prepare_one 1 DK1_FR1 "$DK1_WEIGHTS" > "$LOG_ROOT/DK1_FR1.log" 2>&1 & p1=$!
prepare_one 2 FR1_FR2 "$FR1_WEIGHTS" > "$LOG_ROOT/FR1_FR2.log" 2>&1 & p2=$!
prepare_one 3 FR2_AT1 "$FR2_WEIGHTS" > "$LOG_ROOT/FR2_AT1.log" 2>&1 & p3=$!
wait_wave "$p0" "$p1" "$p2" "$p3"

echo "[STAGE 2] Shared calibration from all four sources"
"$PYTHON_BIN" -u scripts/diagnose_structure_identity_phase.py --stage calibrate "${common[@]}" \
    --identity-component-scale-quantile "$IDENTITY_COMPONENT_SCALE_QUANTILE" \
    --identity-cost-quantile "$IDENTITY_COST_QUANTILE" \
    --identity-calibration-min-pairs "$IDENTITY_CALIBRATION_MIN_PAIRS" \
    2>&1 | tee "$LOG_ROOT/calibration.log"

echo "[STAGE 3] Time-free identity audit; preparation reused without feature extraction"
pids=()
for task in AT1_DK1 DK1_FR1 FR1_FR2 FR2_AT1; do
    "$PYTHON_BIN" -u scripts/diagnose_structure_identity_phase.py --stage audit --task "$task" \
        "${common[@]}" >> "$LOG_ROOT/$task.log" 2>&1 &
    pids+=("$!")
done
wait_wave "${pids[@]}"
# Only remove this invocation's mktemp directory, after every output is published.
if [[ "$WORK_ROOT" == "$OUTPUT_ROOT"/.work_06C_* && -d "$WORK_ROOT" ]]; then
    rm -r -- "$WORK_ROOT"
fi
echo "[ALL FINISHED] 06C: $OUTPUT_ROOT"
