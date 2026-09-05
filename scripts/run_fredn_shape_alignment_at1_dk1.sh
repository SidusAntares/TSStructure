#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
MODE9_CKPT="${MODE9_CKPT:-}"
MODE13_CKPT="${MODE13_CKPT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fredn_shape_alignment_probe/AT1_to_DK1}"
GIT_COMMIT="${GIT_COMMIT:-unavailable}"
GIT_BRANCH="${GIT_BRANCH:-unavailable}"
GIT_DIRTY="${GIT_DIRTY:-unavailable}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: data root not found: $DATA_ROOT" >&2
    exit 1
fi
if [[ -z "$MODE9_CKPT" || ! -f "$MODE9_CKPT" ]]; then
    echo "ERROR: MODE9_CKPT must point to an existing mode-9 source model.pt" >&2
    exit 1
fi
if [[ -z "$MODE13_CKPT" || ! -f "$MODE13_CKPT" ]]; then
    echo "ERROR: MODE13_CKPT must point to an existing mode-13 source model.pt" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/probe_fredn_shape_alignment.py \
    --data-root "$DATA_ROOT" \
    --source AT1 \
    --target DK1 \
    --mode9-checkpoint "$MODE9_CKPT" \
    --mode13-checkpoint "$MODE13_CKPT" \
    --device cuda \
    --seed 1 \
    --grid-step-days 1 \
    --prominence-rel 0.15 \
    --min-distance-days 14 \
    --bootstrap-repeats 500 \
    --output-dir "$OUTPUT_DIR" \
    --git-commit "$GIT_COMMIT" \
    --git-branch "$GIT_BRANCH" \
    --git-dirty "$GIT_DIRTY"
