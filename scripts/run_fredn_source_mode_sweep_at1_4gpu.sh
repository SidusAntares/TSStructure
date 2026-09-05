#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-100}"

SOURCE="austria/33UVP/2017"
OUTPUT_ROOT="outputs/fredn_source_mode_sweep_at1_seed${SEED}"
TENSORBOARD_ROOT="runs/fredn_source_mode_sweep_at1_seed${SEED}"
LOG_ROOT="logs/fredn_source_mode_sweep_at1_seed${SEED}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: dataset root not found: $DATA_ROOT" >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$TENSORBOARD_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1

launch_source() {
    local gpu="$1"
    local mode="$2"
    local experiment="fredn_reim_AT1_source_modes${mode}_seed${SEED}"
    local log_path="${LOG_ROOT}/${experiment}.log"
    local output_path="${OUTPUT_ROOT}/${experiment}"

    nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
        -e "$experiment" \
        --data_root "$DATA_ROOT" \
        --source "$SOURCE" \
        --target "$SOURCE" \
        --model psefrednltae \
        --fredn_num_modes "$mode" \
        --fredn_fourier_solver dense_direct \
        --fredn_nufft_reg 0.001 \
        --fredn_period_days 365 \
        --seed "$SEED" \
        --num_folds 1 \
        --seq_length 30 \
        --num_pixels 64 \
        --batch_size 128 \
        --closed_set true \
        --combine_spring_and_winter false \
        --with_shift_aug false \
        --epochs "$EPOCHS" \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --focal_loss_gamma 1.0 \
        --progress_bar off \
        --output_dir "$OUTPUT_ROOT" \
        --tensorboard_log_dir "$TENSORBOARD_ROOT" \
        >"$log_path" 2>&1 &

    local pid=$!
    printf 'PID=%s GPU=%s MODE=%s LOG=%s OUTPUT=%s\n' \
        "$pid" "$gpu" "$mode" "$log_path" "$output_path"
}

launch_source "$GPU0" 11
launch_source "$GPU1" 13
launch_source "$GPU2" 15
launch_source "$GPU3" 19

echo "Four AT1 source-only jobs launched; this launcher will now exit."
