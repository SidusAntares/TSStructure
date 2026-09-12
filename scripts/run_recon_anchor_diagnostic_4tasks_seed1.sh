#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SOURCE_CHECKPOINT_ROOT="${SOURCE_CHECKPOINT_ROOT:-outputs}"
SHIFT_VISUALIZATION_ROOT="${SHIFT_VISUALIZATION_ROOT:-outputs/shift_visualizations_seed1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/recon_anchor_diagnostic_4tasks_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/recon_anchor_diagnostic_4tasks_seed1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

AT1_WEIGHTS="${AT1_WEIGHTS:-outputs/pseltae_AT1_source_seed1}"
DK1_WEIGHTS="${DK1_WEIGHTS:-outputs/pseltae_DK1_source_seed1}"
FR1_WEIGHTS="${FR1_WEIGHTS:-outputs/pseltae_FR1_source_seed1}"
FR2_WEIGHTS="${FR2_WEIGHTS:-outputs/pseltae_FR2_source_seed1}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 1; }
for directory in "$DATA_ROOT" "$SOURCE_CHECKPOINT_ROOT" "$SHIFT_VISUALIZATION_ROOT"; do
    [[ -d "$directory" ]] || { echo "ERROR: required local directory not found: $directory" >&2; exit 1; }
done
for checkpoint_root in "$AT1_WEIGHTS" "$DK1_WEIGHTS" "$FR1_WEIGHTS" "$FR2_WEIGHTS"; do
    [[ -f "$checkpoint_root/fold_0/model.pt" ]] || { echo "ERROR: source checkpoint not found: $checkpoint_root/fold_0/model.pt" >&2; exit 1; }
    [[ -f "$checkpoint_root/train_config.json" ]] || { echo "ERROR: source config not found: $checkpoint_root/train_config.json" >&2; exit 1; }
done
for task in AT1_DK1 DK1_FR1 FR1_FR2 FR2_AT1; do
    [[ -f "$SHIFT_VISUALIZATION_ROOT/$task/manifest.json" ]] || { echo "ERROR: shift manifest not found: $SHIFT_VISUALIZATION_ROOT/$task/manifest.json" >&2; exit 1; }
    [[ -f "$SHIFT_VISUALIZATION_ROOT/$task/shifts_summary.csv" ]] || { echo "ERROR: shift summary not found: $SHIFT_VISUALIZATION_ROOT/$task/shifts_summary.csv" >&2; exit 1; }
    [[ -d "$SHIFT_VISUALIZATION_ROOT/$task/03_reconshift13_shift" ]] || { echo "ERROR: ReconShift13 view not found for $task" >&2; exit 1; }
done

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1

run_task() {
    local gpu="$1"
    local source_alias="$2"
    local target_alias="$3"
    local task="${source_alias}_${target_alias}"
    local source_weights
    case "$source_alias" in
        AT1) source_weights="$AT1_WEIGHTS" ;;
        DK1) source_weights="$DK1_WEIGHTS" ;;
        FR1) source_weights="$FR1_WEIGHTS" ;;
        FR2) source_weights="$FR2_WEIGHTS" ;;
        *) echo "ERROR: unknown source alias: $source_alias" >&2; return 2 ;;
    esac
    echo "[START] GPU${gpu} ${source_alias} -> ${target_alias}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/diagnose_recon_anchor_modes_4tasks_seed1.py \
        --data-root "$DATA_ROOT" \
        --source-checkpoint-root "$SOURCE_CHECKPOINT_ROOT" \
        --source-checkpoint "${source_alias}=${source_weights}" \
        --shift-visualization-root "$SHIFT_VISUALIZATION_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --source-domain "$source_alias" \
        --target-domain "$target_alias" \
        --seed 1 \
        --fold 0 \
        --device cuda \
        > "$LOG_ROOT/${task}.log" 2>&1
}

run_task 0 AT1 DK1 & pid0=$!
run_task 1 DK1 FR1 & pid1=$!
run_task 2 FR1 FR2 & pid2=$!
run_task 3 FR2 AT1 & pid3=$!

status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if [[ "$status" -ne 0 ]]; then
    echo "ERROR: one or more recon anchor diagnostics failed; inspect $LOG_ROOT"
    exit 1
fi
echo "[ALL FINISHED] outputs: $OUTPUT_ROOT"
