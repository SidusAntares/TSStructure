#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SKIP_SOURCE="${SKIP_SOURCE:-0}"
SEED=1
RUN_NAME="fourier_recon_mode_sweep_at1_dk1_seed1"
OUTPUT_ROOT="outputs/${RUN_NAME}"
TENSORBOARD_ROOT="runs/${RUN_NAME}"
LOG_ROOT="logs/${RUN_NAME}"
SUMMARY_SCRIPT="scripts/summarize_fourier_recon_modes_at1_dk1.py"
AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
MODES=(9 11 13 15 17 19)
MODE_QUEUE_GPU0=(9 17)
MODE_QUEUE_GPU1=(11 19)
MODE_QUEUE_GPU2=(13)
MODE_QUEUE_GPU3=(15)
export PYTHONUNBUFFERED=1

[[ "$SKIP_SOURCE" == 0 || "$SKIP_SOURCE" == 1 ]] || { echo "ERROR: SKIP_SOURCE must be 0 or 1"; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: DATA_ROOT not found: $DATA_ROOT"; exit 1; }
mkdir -p "$OUTPUT_ROOT" "$TENSORBOARD_ROOT" "$LOG_ROOT"

run_mode() {
    local gpu="$1" mode="$2" tag
    printf -v tag '%02d' "$mode"
    local mode_output="${OUTPUT_ROOT}/mode${tag}" mode_tb="${TENSORBOARD_ROOT}/mode${tag}" mode_log="${LOG_ROOT}/mode${tag}"
    local source_experiment="fourier_recon_m${tag}_AT1_source_seed${SEED}"
    local source_root="${mode_output}/source"
    local source_weights="${source_root}/${source_experiment}"
    local checkpoint="${source_weights}/fold_0/model.pt" source_log="${mode_log}/source.log"
    local eval_experiment="fourier_recon_m${tag}_AT1_on_DK1_seed${SEED}"
    local eval_root="${mode_output}/source_on_target" eval_log="${mode_log}/source_on_target.log"
    local da_experiment="fourier_recon_m${tag}_AT1_DK1_timematch_seed${SEED}"
    local da_root="${mode_output}/AT1_DK1" da_log="${mode_log}/da.log"
    mkdir -p "$mode_log" "$source_root" "$mode_tb"

    if [[ "$SKIP_SOURCE" == 0 ]]; then
        echo "[GPU${gpu}] MODE ${mode}: SOURCE AT1 -> AT1" | tee "$source_log"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
            -e "$source_experiment" --data_root "$DATA_ROOT" --source "$AT1" --target "$AT1" \
            --model psefourierreconltae --fourier_num_modes "$mode" \
            --fourier_solver dense_direct --fourier_reg 0.001 --fourier_period_days 365 \
            --seed "$SEED" --num_folds 1 --seq_length 30 --num_pixels 64 --batch_size 128 \
            --device cuda --closed_set true --combine_spring_and_winter false --with_shift_aug false \
            --epochs 100 --lr 0.001 --weight_decay 0.0001 --focal_loss_gamma 1.0 \
            --progress_bar off --output_dir "$source_root" --tensorboard_log_dir "${mode_tb}/source" \
            2>&1 | tee -a "$source_log"
    fi
    [[ -f "$checkpoint" ]] || { echo "ERROR: checkpoint not found: $checkpoint"; return 1; }

    echo "[GPU${gpu}] MODE ${mode}: SOURCE-ON-TARGET AT1 -> DK1" | tee "$eval_log"
    mkdir -p "${eval_root}/${eval_experiment}/fold_0"
    cp "$checkpoint" "${eval_root}/${eval_experiment}/fold_0/model.pt"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
        -e "$eval_experiment" --data_root "$DATA_ROOT" --source "$AT1" --target "$DK1" \
        --model psefourierreconltae --fourier_num_modes "$mode" \
        --fourier_solver dense_direct --fourier_reg 0.001 --fourier_period_days 365 \
        --seed "$SEED" --num_folds 1 --seq_length 30 --num_pixels 64 --batch_size 128 \
        --device cuda --closed_set true --combine_spring_and_winter false --with_shift_aug false \
        --progress_bar off --output_dir "$eval_root" --tensorboard_log_dir "${mode_tb}/source_on_target" --eval \
        2>&1 | tee -a "$eval_log"
    "$PYTHON_BIN" "$SUMMARY_SCRIPT" --emit-source-on-target "$eval_log" --mode "$mode" --seed "$SEED" | tee -a "$eval_log"

    echo "[GPU${gpu}] MODE ${mode}: TIMEMATCH AT1 -> DK1" | tee "$da_log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
        -e "$da_experiment" --data_root "$DATA_ROOT" --source "$AT1" --target "$DK1" \
        --model psefourierreconltae --fourier_num_modes "$mode" \
        --fourier_solver dense_direct --fourier_reg 0.001 --fourier_period_days 365 \
        --seed "$SEED" --num_folds 1 --seq_length 30 --num_pixels 64 --batch_size 128 \
        --weight_decay 0.0001 --focal_loss_gamma 1.0 --device cuda --closed_set true \
        --combine_spring_and_winter false --with_shift_aug false \
        --progress_bar off --output_dir "$da_root" --tensorboard_log_dir "${mode_tb}/AT1_DK1" \
        timematch --weights "$source_weights" --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
        --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 --estimate_shift true \
        --balance_source true --use_focal_loss true --shift_source true --sample_size 100 \
        --max_temporal_shift 60 --domain_specific_bn true --shift_estimator AM \
        --run_validation --output_student true 2>&1 | tee -a "$da_log"
    echo "[FINISHED] MODE ${mode}" | tee -a "$da_log"
}

run_queue() { local gpu="$1"; shift; for mode in "$@"; do run_mode "$gpu" "$mode"; done; }
run_queue 0 "${MODE_QUEUE_GPU0[@]}" >"${LOG_ROOT}/worker_gpu0.log" 2>&1 & PID0=$!
run_queue 1 "${MODE_QUEUE_GPU1[@]}" >"${LOG_ROOT}/worker_gpu1.log" 2>&1 & PID1=$!
run_queue 2 "${MODE_QUEUE_GPU2[@]}" >"${LOG_ROOT}/worker_gpu2.log" 2>&1 & PID2=$!
run_queue 3 "${MODE_QUEUE_GPU3[@]}" >"${LOG_ROOT}/worker_gpu3.log" 2>&1 & PID3=$!

status=0
for pid in "$PID0" "$PID1" "$PID2" "$PID3"; do if ! wait "$pid"; then status=1; fi; done
if ! "$PYTHON_BIN" "$SUMMARY_SCRIPT" --log-root "$LOG_ROOT" --output-dir "$OUTPUT_ROOT" --seed "$SEED"; then status=1; fi
[[ "$status" == 0 ]] && echo "[ALL FINISHED] Six FourierRecon modes completed."
exit "$status"
