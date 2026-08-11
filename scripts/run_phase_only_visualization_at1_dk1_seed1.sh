#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-64}"
DISPLAY_SAMPLES="${DISPLAY_SAMPLES:-12}"
PSE_GRID_SIZE="${PSE_GRID_SIZE:-128}"

PHASE_CHECKPOINT="${PHASE_CHECKPOINT:-${PROJECT_ROOT}/outputs/phase_only_at1_dk1_seed1/fold_0/stage2_best_target_val_ema.pt}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${PROJECT_ROOT}/outputs/phase_only_at1_dk1_seed1/fold_0/stage1_best.pt}"
VIS_ROOT="${VIS_ROOT:-${PROJECT_ROOT}/outputs/visualization_phase_only_at1_dk1_seed1}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/visualization_phase_only_at1_dk1_seed1}"

for path in "${PHASE_CHECKPOINT}" "${MODEL_CHECKPOINT}"; do
    [[ -f "${path}" ]] || { echo "Required checkpoint not found: ${path}" >&2; exit 1; }
done
[[ -d "${DATA_ROOT}" ]] || { echo "Data root not found: ${DATA_ROOT}" >&2; exit 1; }
if [[ -e "${VIS_ROOT}" ]]; then
    echo "Visualization output already exists: ${VIS_ROOT}" >&2
    echo "Remove it or set VIS_ROOT to a new directory." >&2
    exit 1
fi
mkdir -p "${VIS_ROOT}" "${LOG_ROOT}"

cd "${PROJECT_ROOT}"

COMMON=(
    --checkpoint "${PHASE_CHECKPOINT}"
    --model-checkpoint "${MODEL_CHECKPOINT}"
    --data-root "${DATA_ROOT}"
    --device cuda:0
    --samples-per-class "${SAMPLES_PER_CLASS}"
    --batch-size 64
    --num-workers "${NUM_WORKERS}"
    --pse-grid-size "${PSE_GRID_SIZE}"
)

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -u scripts/visualize_stage2_phase_alignment.py \
    "${COMMON[@]}" \
    --display-samples "${DISPLAY_SAMPLES}" \
    --output-dir "${VIS_ROOT}" \
    > "${LOG_ROOT}/01_03_phase_validation.log" 2>&1

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -u scripts/compare_stage2_phase_vs_timematch_shift.py \
    "${COMMON[@]}" \
    --output-dir "${VIS_ROOT}/04_phase_vs_scalar_pse_effect" \
    > "${LOG_ROOT}/04_phase_vs_scalar_pse_effect.log" 2>&1

echo "PHASE_VISUALIZATION_SUITE_COMPLETE|output=${VIS_ROOT}|logs=${LOG_ROOT}"
