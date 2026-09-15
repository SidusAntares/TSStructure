#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
VIEW_ROOT="${VIEW_ROOT:-outputs/shift_visualizations_seed1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$VIEW_ROOT/07B_structure_identity_encoder}"
LOG_ROOT="${LOG_ROOT:-logs/structure_identity_encoder_07B}"
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
for task in AT1_DK1 DK1_FR1 FR1_FR2 FR2_AT1; do
    [[ -s "$VIEW_ROOT/06B_structure_observation_support/$task/sample_structure_support.csv" ]] || {
        echo "ERROR: missing 06B G+ input: $task" >&2; exit 1;
    }
    [[ -s "$VIEW_ROOT/07A_multivariate_local_waveform_scan/$task/gplus_local_ranking.csv" ]] || {
        echo "ERROR: missing 07A candidate input: $task" >&2; exit 1;
    }
done

mkdir -p "$(dirname "$OUTPUT_ROOT")" "$LOG_ROOT"

run_source() {
    local gpu="$1" source="$2" weights="$3"
    echo "[START] GPU${gpu} 07B ${source}; seeds=1,2,3; variants=Waveform,Event,Fusion"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/train_structure_identity_encoder.py \
        --source "$source" --source-checkpoint "$weights" --data-root "$DATA_ROOT" \
        --structure-view-root "$VIEW_ROOT" --output-root "$OUTPUT_ROOT" \
        --model-seeds 1 2 3 --variants Waveform Event Fusion \
        --max-epochs 100 --patience 15 --batch-size 128 --device cuda
    echo "[FINISHED] GPU${gpu} 07B ${source}"
}

run_source 0 AT1 "$AT1_WEIGHTS" > "$LOG_ROOT/AT1_GPU0.log" 2>&1 & p0=$!
run_source 1 DK1 "$DK1_WEIGHTS" > "$LOG_ROOT/DK1_GPU1.log" 2>&1 & p1=$!
run_source 2 FR1 "$FR1_WEIGHTS" > "$LOG_ROOT/FR1_GPU2.log" 2>&1 & p2=$!
run_source 3 FR2 "$FR2_WEIGHTS" > "$LOG_ROOT/FR2_GPU3.log" 2>&1 & p3=$!

failed=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
    if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" != 0 ]]; then
    echo "ERROR: one or more 07B source workers failed; old per-source outputs were preserved" >&2
    exit 1
fi
echo "[ALL FINISHED] 07B source-only structure identity encoders"
