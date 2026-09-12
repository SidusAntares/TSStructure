#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SOURCE_CHECKPOINT_ROOT="${SOURCE_CHECKPOINT_ROOT:-outputs}"
TIMEMATCH_OUTPUT_ROOT="${TIMEMATCH_OUTPUT_ROOT:-outputs/timematch_ablation_seed1/original}"
TIMEMATCH_LOG_ROOT="${TIMEMATCH_LOG_ROOT:-logs/timematch_ablation_seed1/original}"
RECONSHIFT_OUTPUT_ROOT="${RECONSHIFT_OUTPUT_ROOT:-outputs/reconshift13_raw_4tasks_seed1}"
RECONSHIFT_LOG_ROOT="${RECONSHIFT_LOG_ROOT:-logs/reconshift13_raw_4tasks_seed1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/shift_visualizations_seed1}"
SEED="${SEED:-1}"
FOLD="${FOLD:-0}"
GRID_SIZE="${GRID_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"
ADD_RECON13_LOCAL_NONLINEAR="${ADD_RECON13_LOCAL_NONLINEAR:-0}"
LOCAL_LOG_ROOT="${LOCAL_LOG_ROOT:-logs/reconshift13_local_nonlinear_visualization_seed1}"

run_local_task() {
    local gpu="$1"
    local source="$2"
    local target="$3"
    local task="${source}_${target}"
    local source_weights
    case "$source" in
        AT1) source_weights="$AT1_WEIGHTS" ;;
        DK1) source_weights="$DK1_WEIGHTS" ;;
        FR1) source_weights="$FR1_WEIGHTS" ;;
        FR2) source_weights="$FR2_WEIGHTS" ;;
        *) echo "ERROR: unknown source alias: $source" >&2; return 2 ;;
    esac
    echo "[START] GPU${gpu} ${source} -> ${target}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/visualize_shift_configs_4tasks.py \
        --data-root "$DATA_ROOT" \
        --source-checkpoint-root "$SOURCE_CHECKPOINT_ROOT" \
        --source-checkpoint "${source}=${source_weights}" \
        --timematch-output-root "$TIMEMATCH_OUTPUT_ROOT" \
        --timematch-log-root "$TIMEMATCH_LOG_ROOT" \
        --reconshift-output-root "$RECONSHIFT_OUTPUT_ROOT" \
        --reconshift-log-root "$RECONSHIFT_LOG_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --source-domain "$source" \
        --target-domain "$target" \
        --seed "$SEED" \
        --fold "$FOLD" \
        --grid-size "$GRID_SIZE" \
        --add-recon13-local-nonlinear \
        --device "$DEVICE"
}

AT1_WEIGHTS="${AT1_WEIGHTS:-outputs/pseltae_AT1_source_seed1}"
DK1_WEIGHTS="${DK1_WEIGHTS:-outputs/pseltae_DK1_source_seed1}"
FR1_WEIGHTS="${FR1_WEIGHTS:-outputs/pseltae_FR1_source_seed1}"
FR2_WEIGHTS="${FR2_WEIGHTS:-outputs/pseltae_FR2_source_seed1}"

[[ "$FOLD" == "0" ]] || { echo "ERROR: this audit supports fold 0 only" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 1; }
for directory in \
    "$DATA_ROOT" "$SOURCE_CHECKPOINT_ROOT" \
    "$TIMEMATCH_OUTPUT_ROOT" "$TIMEMATCH_LOG_ROOT" \
    "$RECONSHIFT_OUTPUT_ROOT" "$RECONSHIFT_LOG_ROOT"; do
    [[ -d "$directory" ]] || { echo "ERROR: required local directory not found: $directory" >&2; exit 1; }
done
for checkpoint_root in "$AT1_WEIGHTS" "$DK1_WEIGHTS" "$FR1_WEIGHTS" "$FR2_WEIGHTS"; do
    [[ -f "$checkpoint_root/fold_0/model.pt" ]] || { echo "ERROR: source checkpoint not found: $checkpoint_root/fold_0/model.pt" >&2; exit 1; }
    [[ -f "$checkpoint_root/train_config.json" ]] || { echo "ERROR: source config not found: $checkpoint_root/train_config.json" >&2; exit 1; }
done

mkdir -p "$OUTPUT_ROOT"
export PYTHONUNBUFFERED=1

if [[ "$ADD_RECON13_LOCAL_NONLINEAR" == "1" ]]; then
    mkdir -p "$LOCAL_LOG_ROOT"
    run_local_task 0 AT1 DK1 > "$LOCAL_LOG_ROOT/AT1_DK1.log" 2>&1 & pid0=$!
    run_local_task 1 DK1 FR1 > "$LOCAL_LOG_ROOT/DK1_FR1.log" 2>&1 & pid1=$!
    run_local_task 2 FR1 FR2 > "$LOCAL_LOG_ROOT/FR1_FR2.log" 2>&1 & pid2=$!
    run_local_task 3 FR2 AT1 > "$LOCAL_LOG_ROOT/FR2_AT1.log" 2>&1 & pid3=$!
    status=0
    for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
        if ! wait "$pid"; then
            status=1
        fi
    done
    if [[ "$status" -ne 0 ]]; then
        echo "ERROR: one or more local nonlinear visualizations failed; inspect $LOCAL_LOG_ROOT" >&2
        exit 1
    fi
    echo "[ALL FINISHED] Recon13 local nonlinear visualizations written to $OUTPUT_ROOT"
    exit 0
fi

printf '%s\n' \
    "============================================================" \
    "OFFLINE RAW-PSE SHIFT VISUALIZATION" \
    "tasks: AT1->DK1, DK1->FR1, FR1->FR2, FR2->AT1" \
    "training: disabled" \
    "device: ${DEVICE} (visible GPU ${GPU_ID})" \
    "output: ${OUTPUT_ROOT}" \
    "============================================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/visualize_shift_configs_4tasks.py \
    --data-root "$DATA_ROOT" \
    --source-checkpoint-root "$SOURCE_CHECKPOINT_ROOT" \
    --source-checkpoint "AT1=$AT1_WEIGHTS" \
    --source-checkpoint "DK1=$DK1_WEIGHTS" \
    --source-checkpoint "FR1=$FR1_WEIGHTS" \
    --source-checkpoint "FR2=$FR2_WEIGHTS" \
    --timematch-output-root "$TIMEMATCH_OUTPUT_ROOT" \
    --timematch-log-root "$TIMEMATCH_LOG_ROOT" \
    --reconshift-output-root "$RECONSHIFT_OUTPUT_ROOT" \
    --reconshift-log-root "$RECONSHIFT_LOG_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --seed "$SEED" \
    --fold "$FOLD" \
    --grid-size "$GRID_SIZE" \
    --add-class-residual-shift \
    --class-residual-max-days 20 \
    --device "$DEVICE"

echo "[ALL FINISHED] Offline shift visualizations written to $OUTPUT_ROOT"
