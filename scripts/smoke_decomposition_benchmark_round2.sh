#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
DATA_ROOT=${DATA_ROOT:-/data/user/dataset/timematch_data}
NUM_WORKERS=${NUM_WORKERS:-0}
DEVICE=${DEVICE:-cuda}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/smoke_decomposition_benchmark_round2}
TENSORBOARD_ROOT=${TENSORBOARD_ROOT:-runs/smoke_decomposition_benchmark_round2}
SOURCE=${SOURCE:-austria/33UVP/2017}

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if ! "$PYTHON_BIN" -c "import statsmodels"; then
    echo "Missing dependency: statsmodels (required by stl). Install with: pip install statsmodels" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c "import pywt"; then
    echo "Missing dependency: PyWavelets (required by dwt). Install with: pip install PyWavelets" >&2
    exit 1
fi

for MODEL in dlinear_decomp micn_decomp xpatch_ema stl dwt; do
    "$PYTHON_BIN" train.py \
        --data_root "$DATA_ROOT" \
        --source "$SOURCE" \
        --target "$SOURCE" \
        --model "$MODEL" \
        --epochs 1 \
        --num_workers "$NUM_WORKERS" \
        --device "$DEVICE" \
        --progress_bar off \
        --experiment_name "round2_smoke_${MODEL}" \
        --output_dir "$OUTPUT_ROOT" \
        --tensorboard_log_dir "$TENSORBOARD_ROOT"
done
