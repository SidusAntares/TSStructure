#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SOURCE_WEIGHTS="${SOURCE_WEIGHTS:?must provide original TimeMatch AT1 source weights}"
SEED="${SEED:-1}"
export PYTHONUNBUFFERED=1
export PYTHON_BIN GPU0 GPU1 GPU2 GPU3 DATA_ROOT SOURCE_WEIGHTS SEED

SOURCE="austria/33UVP/2017"
TARGET="denmark/32VNH/2017"
RUN_ROOT="timematch_shape_at1_dk1_seed${SEED}"
OUTPUT_ROOT="outputs/${RUN_ROOT}"
TENSORBOARD_ROOT="runs/${RUN_ROOT}"
LOG_ROOT="logs/${RUN_ROOT}"

mkdir -p logs outputs runs "$OUTPUT_ROOT" "$TENSORBOARD_ROOT" "$LOG_ROOT"

if [[ ! -f "${SOURCE_WEIGHTS}/fold_0/model.pt" ]]; then
    echo "ERROR: original PseLTae source checkpoint not found: ${SOURCE_WEIGHTS}/fold_0/model.pt" >&2
    exit 1
fi
if [[ ! -f "${SOURCE_WEIGHTS}/train_config.json" ]]; then
    echo "ERROR: source train_config.json not found: ${SOURCE_WEIGHTS}/train_config.json" >&2
    exit 1
fi
SOURCE_MODEL="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("model", ""))' "${SOURCE_WEIGHTS}/train_config.json")"
if [[ "$SOURCE_MODEL" != "pseltae" ]]; then
    echo "ERROR: SOURCE_WEIGHTS must be an original PseLTae checkpoint; found model=${SOURCE_MODEL}" >&2
    exit 1
fi

run_worker() {
    local gpu="$1"
    local experiment="$2"
    local event_weight="$3"
    shift 3
    local modes=("$@")

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
        -e "$experiment" \
        --data_root "$DATA_ROOT" \
        --source "$SOURCE" \
        --target "$TARGET" \
        --model pseltae \
        --seed "$SEED" \
        --num_folds 1 \
        --seq_length 30 \
        --num_pixels 64 \
        --batch_size 128 \
        --weight_decay 0.0001 \
        --focal_loss_gamma 1.0 \
        --device cuda \
        --closed_set true \
        --combine_spring_and_winter false \
        --with_shift_aug false \
        --progress_bar off \
        --output_dir "$OUTPUT_ROOT" \
        --tensorboard_log_dir "$TENSORBOARD_ROOT" \
        timematch \
        --weights "$SOURCE_WEIGHTS" \
        --epochs 20 \
        --steps_per_epoch 500 \
        --lr 0.0001 \
        --pseudo_threshold 0.9 \
        --ema_decay 0.9999 \
        --trade_off 2.0 \
        --estimate_shift true \
        --balance_source true \
        --use_focal_loss true \
        --shift_source true \
        --sample_size 100 \
        --max_temporal_shift 60 \
        --domain_specific_bn true \
        --shift_estimator AM \
        --run_validation \
        --output_student true \
        --shape_align true \
        --shape_modes "${modes[@]}" \
        --shape_lambda 0.1 \
        --shape_morph_weight 1.0 \
        --shape_event_weight "$event_weight" \
        --shape_grid_points 64 \
        --shape_reference_per_class 128 \
        --shape_reference_seed 1 \
        --shape_prominence_rel 0.15 \
        --shape_min_distance_days 14 \
        --shape_fourier_period_days 365 \
        --shape_fourier_reg 0.001 \
        --shape_diag_batches 10
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    run_worker "$@"
    exit $?
fi

launch() {
    local gpu="$1"
    local experiment="$2"
    local event_weight="$3"
    shift 3
    local modes=("$@")
    local log_path="${LOG_ROOT}/${experiment}.log"
    nohup "$0" --worker "$gpu" "$experiment" "$event_weight" "${modes[@]}" \
        > "$log_path" 2>&1 &
    local pid=$!
    printf 'GPU=%s PID=%s experiment=%s log=%s output=%s\n' \
        "$gpu" "$pid" "$experiment" "$log_path" "${OUTPUT_ROOT}/${experiment}"
}

launch "$GPU0" "timematch_shape9_AT1_DK1_seed${SEED}" 0.5 9
launch "$GPU1" "timematch_shape13_AT1_DK1_seed${SEED}" 0.5 13
launch "$GPU2" "timematch_shape9_13_AT1_DK1_seed${SEED}" 0.5 9 13
launch "$GPU3" "timematch_shape13_morph_AT1_DK1_seed${SEED}" 0.0 13

echo "Four shape-only TimeMatch pilots launched; Original TimeMatch baseline was not started."
