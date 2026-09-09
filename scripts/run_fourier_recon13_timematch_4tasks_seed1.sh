#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
SKIP_SOURCE="${SKIP_SOURCE:-0}"
SEED=1
RUN_NAME="fourier_recon13_4tasks_seed1"
OUTPUT_ROOT="outputs/${RUN_NAME}"
TENSORBOARD_ROOT="runs/${RUN_NAME}"
LOG_ROOT="logs/${RUN_NAME}"
export PYTHONUNBUFFERED=1

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"

if [[ "$SKIP_SOURCE" != "0" && "$SKIP_SOURCE" != "1" ]]; then
    echo "ERROR: SKIP_SOURCE must be 0 or 1, got: ${SKIP_SOURCE}" >&2
    exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}" >&2
    exit 1
fi
mkdir -p logs outputs runs "$OUTPUT_ROOT" "$TENSORBOARD_ROOT" "$LOG_ROOT"

run_worker() {
    local gpu="$1"
    local source_alias="$2"
    local source_path="$3"
    local target_alias="$4"
    local target_path="$5"
    local source_experiment="fourier_recon13_${source_alias}_source_seed${SEED}"
    local source_output_root="${OUTPUT_ROOT}/source/${source_alias}"
    local source_tensorboard_root="${TENSORBOARD_ROOT}/source/${source_alias}"
    local source_weights="${source_output_root}/${source_experiment}"
    local source_checkpoint="${source_weights}/fold_0/model.pt"
    local source_log="${LOG_ROOT}/source_${source_alias}.log"
    local da_experiment="fourier_recon13_${source_alias}_${target_alias}_timematch_seed${SEED}"
    local da_output_root="${OUTPUT_ROOT}/${source_alias}_${target_alias}"
    local da_tensorboard_root="${TENSORBOARD_ROOT}/${source_alias}_${target_alias}"
    local da_log="${LOG_ROOT}/${source_alias}_${target_alias}.log"

    if [[ "$SKIP_SOURCE" == "0" ]]; then
        printf '%s\n' \
            '============================================================' \
            "[GPU${gpu}] SOURCE PRETRAIN ${source_alias} -> ${source_alias}" \
            '============================================================' | tee -a "$source_log"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
            -e "$source_experiment" \
            --data_root "$DATA_ROOT" \
            --source "$source_path" \
            --target "$source_path" \
            --model psefourierreconltae \
            --fourier_num_modes 13 \
            --fourier_solver dense_direct \
            --fourier_reg 0.001 \
            --fourier_period_days 365 \
            --seed "$SEED" \
            --num_folds 1 \
            --seq_length 30 \
            --num_pixels 64 \
            --batch_size 128 \
            --device cuda \
            --closed_set true \
            --combine_spring_and_winter false \
            --with_shift_aug false \
            --epochs 100 \
            --lr 0.001 \
            --weight_decay 0.0001 \
            --focal_loss_gamma 1.0 \
            --progress_bar off \
            --output_dir "$source_output_root" \
            --tensorboard_log_dir "$source_tensorboard_root" \
            2>&1 | tee -a "$source_log"
        echo "[FINISHED] ${source_alias} SOURCE PRETRAIN" | tee -a "$source_log"
    else
        echo "[SKIPPED] ${source_alias} SOURCE PRETRAIN" | tee -a "$source_log"
    fi

    if [[ ! -f "$source_checkpoint" ]]; then
        echo "ERROR: source checkpoint not found: ${source_checkpoint}" >&2
        return 1
    fi

    printf '%s\n' \
        '============================================================' \
        "[GPU${gpu}] TIMEMATCH ${source_alias} -> ${target_alias}" \
        "SOURCE CHECKPOINT: ${source_checkpoint}" \
        '============================================================' | tee -a "$da_log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
        -e "$da_experiment" \
        --data_root "$DATA_ROOT" \
        --source "$source_path" \
        --target "$target_path" \
        --model psefourierreconltae \
        --fourier_num_modes 13 \
        --fourier_solver dense_direct \
        --fourier_reg 0.001 \
        --fourier_period_days 365 \
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
        --output_dir "$da_output_root" \
        --tensorboard_log_dir "$da_tensorboard_root" \
        timematch \
        --weights "$source_weights" \
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
        2>&1 | tee -a "$da_log"
    echo "[FINISHED] ${source_alias} -> ${target_alias}" | tee -a "$da_log"
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    run_worker "$@"
    exit $?
fi

launch_worker() {
    local gpu="$1"
    local source_alias="$2"
    local source_path="$3"
    local target_alias="$4"
    local target_path="$5"
    local worker_log="${LOG_ROOT}/${source_alias}_${target_alias}_worker.log"
    nohup bash "$SCRIPT_PATH" --worker \
        "$gpu" "$source_alias" "$source_path" "$target_alias" "$target_path" \
        > "$worker_log" 2>&1 &
    printf 'LAUNCHED GPU=%s PID=%s TASK=%s->%s LOG=%s\n' \
        "$gpu" "$!" "$source_alias" "$target_alias" "$worker_log"
}

launch_worker "$GPU0" AT1 "$AT1" DK1 "$DK1"
launch_worker "$GPU1" DK1 "$DK1" FR1 "$FR1"
launch_worker "$GPU2" FR1 "$FR1" FR2 "$FR2"
launch_worker "$GPU3" FR2 "$FR2" AT1 "$AT1"

echo "Four FourierRecon-13 source-to-TimeMatch workers launched."
