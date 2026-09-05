#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fredn_structural_probe_modes/AT1_to_DK1}"
GIT_COMMIT="${GIT_COMMIT:-unavailable}"
GIT_BRANCH="${GIT_BRANCH:-unavailable}"
GIT_DIRTY="${GIT_DIRTY:-unavailable}"

CKPT_9="${CKPT_9:-}"
CKPT_11="${CKPT_11:-}"
CKPT_13="${CKPT_13:-}"
CKPT_15="${CKPT_15:-}"
CKPT_17="${CKPT_17:-}"
CKPT_19="${CKPT_19:-}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: dataset root not found: $DATA_ROOT" >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ "$GIT_DIRTY" != "clean" && "$GIT_DIRTY" != "dirty" && "$GIT_DIRTY" != "unavailable" ]]; then
    echo "ERROR: GIT_DIRTY must be clean, dirty, or unavailable" >&2
    exit 1
fi

for mode in 9 11 13 15 17 19; do
    variable="CKPT_${mode}"
    checkpoint="${!variable}"
    if [[ -z "$checkpoint" ]]; then
        echo "ERROR: ${variable} is required" >&2
        exit 1
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: checkpoint for mode ${mode} not found: $checkpoint" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/probe_fredn_structure.py \
    --data-root "$DATA_ROOT" \
    --source AT1 \
    --target DK1 \
    --fredn-checkpoint "9=$CKPT_9" \
    --fredn-checkpoint "11=$CKPT_11" \
    --fredn-checkpoint "13=$CKPT_13" \
    --fredn-checkpoint "15=$CKPT_15" \
    --fredn-checkpoint "17=$CKPT_17" \
    --fredn-checkpoint "19=$CKPT_19" \
    --git-commit "$GIT_COMMIT" \
    --git-branch "$GIT_BRANCH" \
    --git-dirty "$GIT_DIRTY" \
    --device cuda \
    --seed 1 \
    --grid-step-days 1 \
    --prominence-rel 0.15 \
    --min-distance-days 14 \
    --output-dir "$OUTPUT_DIR"
