#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/raw_decomposition_visualization}"
SIGNALS="${SIGNALS:-ALL_BANDS}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-1}"
SEED="${SEED:-1}"
SPATIAL_REDUCTION="${SPATIAL_REDUCTION:-mean}"

if [[ -z "${DATA_ROOT}" ]]; then
    echo "DATA_ROOT=/path/to/timematch_data is required" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
for domain in AT1 DK1 FR1 FR2; do
    echo "DECOMP_VIS_START|domain=${domain}|signals=${SIGNALS}|samples_per_class=${SAMPLES_PER_CLASS}"
    "${PYTHON_BIN}" -u scripts/visualize_raw_decompositions.py \
        --data-root "${DATA_ROOT}" \
        --dataset "${domain}" \
        --output-dir "${OUTPUT_DIR}" \
        --signals "${SIGNALS}" \
        --samples-per-class "${SAMPLES_PER_CLASS}" \
        --seed "${SEED}" \
        --spatial-reduction "${SPATIAL_REDUCTION}"
    echo "DECOMP_VIS_END|domain=${domain}"
done
