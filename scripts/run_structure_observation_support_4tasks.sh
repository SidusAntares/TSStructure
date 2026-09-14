#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
VIEW_ROOT="${VIEW_ROOT:-outputs/shift_visualizations_seed1}"
VALIDITY_ROOT="${VALIDITY_ROOT:-outputs/structure_reference_validity}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/shift_visualizations_seed1/06B_structure_observation_support}"
LOG_ROOT="${LOG_ROOT:-logs/structure_observation_support_06B}"
RUN_STRUCTURE_OBSERVATION_SUPPORT="${RUN_STRUCTURE_OBSERVATION_SUPPORT:-1}"
RUN_STRUCTURE_OBSERVATION_MASKING="${RUN_STRUCTURE_OBSERVATION_MASKING:-1}"

AT1_WEIGHTS="${AT1_WEIGHTS:-outputs/pseltae_AT1_source_seed1}"
DK1_WEIGHTS="${DK1_WEIGHTS:-outputs/pseltae_DK1_source_seed1}"
FR1_WEIGHTS="${FR1_WEIGHTS:-outputs/pseltae_FR1_source_seed1}"
FR2_WEIGHTS="${FR2_WEIGHTS:-outputs/pseltae_FR2_source_seed1}"

[[ "$RUN_STRUCTURE_OBSERVATION_SUPPORT" == "1" ]] || {
    echo "RUN_STRUCTURE_OBSERVATION_SUPPORT is disabled"
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
    local task="$2"
    local weights="$3"
    local masking=()
    if [[ "$RUN_STRUCTURE_OBSERVATION_MASKING" == "1" ]]; then
        masking=(--run-observation-masking-ablation)
    fi
    echo "[START] GPU${gpu} 06B ${task} masking=${RUN_STRUCTURE_OBSERVATION_MASKING}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/diagnose_structure_observation_support.py \
        --diagnose-structure-observation-support \
        "${masking[@]}" \
        --task "$task" \
        --data-root "$DATA_ROOT" \
        --source-checkpoint "$weights" \
        --structure-view-root "$VIEW_ROOT" \
        --validity-root "$VALIDITY_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --min-bootstrap-occurrence 0.8 \
        --observation-support-radius-days 15 \
        --mask-random-fractions 0.25,0.50,0.75 \
        --mask-random-repeats 5 \
        --mask-gap-days 30,60,90 \
        --mask-gap-repeats 5 \
        --mask-max-samples-per-structure 128 \
        --mask-seed 1 \
        --seed 1 \
        --device cuda
    echo "[FINISHED] 06B ${task}"
}

run_one 0 AT1_DK1 "$AT1_WEIGHTS" > "$LOG_ROOT/AT1_DK1.log" 2>&1 & pid0=$!
run_one 1 DK1_FR1 "$DK1_WEIGHTS" > "$LOG_ROOT/DK1_FR1.log" 2>&1 & pid1=$!
run_one 2 FR1_FR2 "$FR1_WEIGHTS" > "$LOG_ROOT/FR1_FR2.log" 2>&1 & pid2=$!
run_one 3 FR2_AT1 "$FR2_WEIGHTS" > "$LOG_ROOT/FR2_AT1.log" 2>&1 & pid3=$!

status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
    if ! wait "$pid"; then status=1; fi
done
if [[ "$status" -ne 0 ]]; then
    echo "ERROR: one or more 06B diagnostics failed; inspect $LOG_ROOT" >&2
    exit 1
fi
echo "[ALL FINISHED] 06B structure observation support: $OUTPUT_ROOT"
