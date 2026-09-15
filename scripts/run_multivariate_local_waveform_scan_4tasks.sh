#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
VIEW_ROOT="${VIEW_ROOT:-outputs/shift_visualizations_seed1}"
VALIDITY_ROOT="${VALIDITY_ROOT:-outputs/06A_structure_reference_validity}"
OBSERVATION_ROOT="${OBSERVATION_ROOT:-$VIEW_ROOT/06B_structure_observation_support}"
IDENTITY_ROOT="${IDENTITY_ROOT:-$VIEW_ROOT/06C_structure_identity_phase}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$VIEW_ROOT/07A_multivariate_local_waveform_scan}"
LOG_ROOT="${LOG_ROOT:-logs/multivariate_local_waveform_scan_07A}"
AT1_WEIGHTS="${AT1_WEIGHTS:-outputs/pseltae_AT1_source_seed1}"
DK1_WEIGHTS="${DK1_WEIGHTS:-outputs/pseltae_DK1_source_seed1}"
FR1_WEIGHTS="${FR1_WEIGHTS:-outputs/pseltae_FR1_source_seed1}"
FR2_WEIGHTS="${FR2_WEIGHTS:-outputs/pseltae_FR2_source_seed1}"
export PYTHONUNBUFFERED=1

[[ -d "$DATA_ROOT" ]] || { echo "ERROR: missing source data: $DATA_ROOT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 1; }
for root in "$AT1_WEIGHTS" "$DK1_WEIGHTS" "$FR1_WEIGHTS" "$FR2_WEIGHTS"; do
    [[ -f "$root/train_config.json" ]] || { echo "ERROR: missing source config: $root/train_config.json" >&2; exit 1; }
    [[ -f "$root/fold_0/model.pt" ]] || { echo "ERROR: missing source checkpoint: $root/fold_0/model.pt" >&2; exit 1; }
done
[[ -s "$IDENTITY_ROOT/calibration.json" ]] || { echo "ERROR: missing revised 06C calibration: $IDENTITY_ROOT/calibration.json" >&2; exit 1; }
for task in AT1_DK1 DK1_FR1 FR1_FR2 FR2_AT1; do
    [[ -s "$OBSERVATION_ROOT/$task/sample_structure_support.csv" ]] || { echo "ERROR: missing 06B input: $task" >&2; exit 1; }
    [[ -s "$IDENTITY_ROOT/$task/sample_structure_identity.csv" ]] || { echo "ERROR: missing revised 06C input: $task" >&2; exit 1; }
done

OUTPUT_PARENT="$(dirname "$OUTPUT_ROOT")"
mkdir -p "$OUTPUT_PARENT" "$LOG_ROOT"
REVISION_ROOT="$(mktemp -d "$OUTPUT_PARENT/.tmp_07A_multivariate_local_waveform_scan_XXXXXXXX")"

run_task() {
    local gpu="$1" task="$2" weights="$3"
    echo "[START] GPU${gpu} 07A ${task} source-only"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/diagnose_multivariate_local_waveform_scan.py \
        --stage task --task "$task" --data-root "$DATA_ROOT" --source-checkpoint "$weights" \
        --structure-view-root "$VIEW_ROOT" --validity-root "$VALIDITY_ROOT" \
        --observation-root "$OBSERVATION_ROOT" --identity-root "$IDENTITY_ROOT" \
        --output-root "$REVISION_ROOT" --local-waveform-points 32 --seed 1 --device cuda
}

wait_all() {
    local failed=0 pid
    for pid in "$@"; do if ! wait "$pid"; then failed=1; fi; done
    if [[ "$failed" != 0 ]]; then
        echo "ERROR: 07A task failed; old output preserved; inspect $LOG_ROOT; staging=$REVISION_ROOT" >&2
        return 1
    fi
}

run_task 0 AT1_DK1 "$AT1_WEIGHTS" > "$LOG_ROOT/AT1_DK1.log" 2>&1 & p0=$!
run_task 1 DK1_FR1 "$DK1_WEIGHTS" > "$LOG_ROOT/DK1_FR1.log" 2>&1 & p1=$!
run_task 2 FR1_FR2 "$FR1_WEIGHTS" > "$LOG_ROOT/FR1_FR2.log" 2>&1 & p2=$!
run_task 3 FR2_AT1 "$FR2_WEIGHTS" > "$LOG_ROOT/FR2_AT1.log" 2>&1 & p3=$!
wait_all "$p0" "$p1" "$p2" "$p3"

"$PYTHON_BIN" -u scripts/diagnose_multivariate_local_waveform_scan.py \
    --stage aggregate --output-root "$REVISION_ROOT" --bootstrap-repeats 1000 --seed 1 \
    2>&1 | tee "$LOG_ROOT/aggregate.log"
"$PYTHON_BIN" -u scripts/diagnose_multivariate_local_waveform_scan.py \
    --stage publish --revision-root "$REVISION_ROOT" --output-root "$OUTPUT_ROOT"
echo "[ALL FINISHED] 07A: $OUTPUT_ROOT"
