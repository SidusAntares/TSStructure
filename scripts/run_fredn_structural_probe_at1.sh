#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
FREDN9_CKPT="${FREDN9_CKPT:-}"
FREDN17_CKPT="${FREDN17_CKPT:-}"
RAW_CKPT="${RAW_CKPT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/fredn_structural_probe}"

if [[ -z "$FREDN9_CKPT" || ! -f "$FREDN9_CKPT" ]]; then
    echo "ERROR: FREDN9_CKPT must point to an existing source-only model.pt" >&2
    exit 1
fi
if [[ -z "$FREDN17_CKPT" || ! -f "$FREDN17_CKPT" ]]; then
    echo "ERROR: FREDN17_CKPT must point to an existing source-only model.pt" >&2
    exit 1
fi
if [[ -n "$RAW_CKPT" && ! -f "$RAW_CKPT" ]]; then
    echo "ERROR: RAW_CKPT was provided but does not exist: $RAW_CKPT" >&2
    exit 1
fi

run_probe() {
    local target="$1"
    local output_dir="${OUTPUT_ROOT}/AT1_to_${target}"
    local command=(
        "$PYTHON_BIN" -u scripts/probe_fredn_structure.py
        --data-root "$DATA_ROOT"
        --source AT1
        --target "$target"
        --fredn9-checkpoint "$FREDN9_CKPT"
        --fredn17-checkpoint "$FREDN17_CKPT"
        --device cuda
        --seed 1
        --grid-step-days 1
        --prominence-rel 0.15
        --min-distance-days 14
        --output-dir "$output_dir"
    )
    if [[ -n "$RAW_CKPT" ]]; then
        command+=(--raw-checkpoint "$RAW_CKPT")
    fi
    printf '%s\n' \
        '============================================================' \
        "STRUCTURAL DECOMPOSITION PROBE: AT1 -> ${target}" \
        'ORACLE_ANALYSIS_ONLY=true' \
        'TARGET_LABEL_USED_FOR_TRAINING=false' \
        "OUTPUT: ${output_dir}" \
        '============================================================'
    CUDA_VISIBLE_DEVICES="$GPU_ID" "${command[@]}"
}

run_probe DK1
run_probe FR1
run_probe FR2

echo "ALL AT1 STRUCTURAL PROBES FINISHED"
