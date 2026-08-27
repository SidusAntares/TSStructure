#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SKIP_SOURCE="${SKIP_SOURCE:-0}"
export PYTHONUNBUFFERED=1

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"

RUN_NAME="fredn_reim_f17_other3sources_seed1"
OUTPUT_ROOT="outputs/${RUN_NAME}"
TENSORBOARD_ROOT="runs/${RUN_NAME}"
TASK_LOG_ROOT="logs/${RUN_NAME}"

mkdir -p logs outputs runs "$OUTPUT_ROOT" "$TENSORBOARD_ROOT" "$TASK_LOG_ROOT"

if [[ "$SKIP_SOURCE" != "0" && "$SKIP_SOURCE" != "1" ]]; then
    echo "ERROR: SKIP_SOURCE must be 0 or 1, got: $SKIP_SOURCE" >&2
    exit 2
fi

run_source_worker() {
    local gpu="$1"
    local source_alias="$2"
    local source_path="$3"
    local target_aliases=("$4" "$6" "$8")
    local target_paths=("$5" "$7" "$9")
    local source_experiment="fredn_reim_f17_${source_alias}_source_seed1"
    local source_output_dir="${OUTPUT_ROOT}/${source_alias}_source"
    local source_tensorboard_dir="${TENSORBOARD_ROOT}/${source_alias}_source"
    local source_weights_dir="${source_output_dir}/${source_experiment}"
    local source_checkpoint="${source_weights_dir}/fold_0/model.pt"
    local source_log="${TASK_LOG_ROOT}/${source_alias}_source.log"

    if [[ "$SKIP_SOURCE" == "1" ]]; then
        printf '%s\n' \
            '============================================================' \
            "[${source_alias} / GPU${gpu}] SOURCE PRETRAIN SKIPPED" \
            "${source_alias} -> ${source_alias}" \
            "SOURCE CHECKPOINT: ${source_checkpoint}" \
            '============================================================' \
            | tee -a "$source_log"
    else
        printf '%s\n' \
            '============================================================' \
            "[${source_alias} / GPU${gpu}] SOURCE PRETRAIN" \
            "${source_alias} -> ${source_alias}" \
            '============================================================' \
            | tee -a "$source_log"

        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
            -e "$source_experiment" \
            --data_root "$DATA_ROOT" \
            --source "$source_path" \
            --target "$source_path" \
            --model psefrednltae \
            --fredn_num_modes 17 \
            --fredn_fourier_solver dense_direct \
            --fredn_nufft_reg 0.001 \
            --fredn_period_days 365 \
            --seed 1 \
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
            --output_dir "$source_output_dir" \
            --tensorboard_log_dir "$source_tensorboard_dir" \
            2>&1 | tee -a "$source_log"

        printf '%s\n' "[FINISHED] ${source_alias} -> ${source_alias} SOURCE PRETRAIN" \
            | tee -a "$source_log"
    fi

    if [[ ! -f "$source_checkpoint" ]]; then
        echo "ERROR: source checkpoint not found: $source_checkpoint" \
            | tee -a "$source_log" >&2
        return 1
    fi

    local index
    for index in 0 1 2; do
        local stage=$((index + 1))
        local target_alias="${target_aliases[$index]}"
        local target_path="${target_paths[$index]}"
        local experiment="fredn_reim_f17_${source_alias}_${target_alias}_timematch_seed1"
        local task_log="${TASK_LOG_ROOT}/${source_alias}_${target_alias}.log"
        local task_output_dir="${OUTPUT_ROOT}/${source_alias}_${target_alias}"
        local task_tensorboard_dir="${TENSORBOARD_ROOT}/${source_alias}_${target_alias}"

        printf '%s\n' \
            '============================================================' \
            "[${source_alias} / GPU${gpu} / STAGE ${stage}/3] TIMEMATCH" \
            "${source_alias} -> ${target_alias}" \
            "SOURCE CHECKPOINT: ${source_checkpoint}" \
            '============================================================' \
            | tee -a "$task_log"

        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
            -e "$experiment" \
            --data_root "$DATA_ROOT" \
            --source "$source_path" \
            --target "$target_path" \
            --model psefrednltae \
            --fredn_num_modes 17 \
            --fredn_fourier_solver dense_direct \
            --fredn_nufft_reg 0.001 \
            --fredn_period_days 365 \
            --seed 1 \
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
            --output_dir "$task_output_dir" \
            --tensorboard_log_dir "$task_tensorboard_dir" \
            timematch \
            --weights "$source_weights_dir" \
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
            2>&1 | tee -a "$task_log"

        printf '%s\n' "[FINISHED] ${source_alias} -> ${target_alias}" \
            | tee -a "$task_log"
    done

    printf '%s\n' "[WORKER FINISHED] ${source_alias} / GPU${gpu}"
}

run_source_worker \
    0 DK1 "$DK1" \
    AT1 "$AT1" FR1 "$FR1" FR2 "$FR2" \
    > "${TASK_LOG_ROOT}/DK1_worker.log" 2>&1 &
PID_DK1=$!

run_source_worker \
    1 FR1 "$FR1" \
    AT1 "$AT1" DK1 "$DK1" FR2 "$FR2" \
    > "${TASK_LOG_ROOT}/FR1_worker.log" 2>&1 &
PID_FR1=$!

run_source_worker \
    2 FR2 "$FR2" \
    AT1 "$AT1" DK1 "$DK1" FR1 "$FR1" \
    > "${TASK_LOG_ROOT}/FR2_worker.log" 2>&1 &
PID_FR2=$!

printf '%s\n' \
    '============================================================' \
    'FREDN F17 WORKERS STARTED' \
    "DK1 / GPU0 PID: ${PID_DK1}" \
    "FR1 / GPU1 PID: ${PID_FR1}" \
    "FR2 / GPU2 PID: ${PID_FR2}" \
    '============================================================'

if wait "$PID_DK1"; then
    STATUS_DK1=0
else
    STATUS_DK1=$?
fi
if wait "$PID_FR1"; then
    STATUS_FR1=0
else
    STATUS_FR1=$?
fi
if wait "$PID_FR2"; then
    STATUS_FR2=0
else
    STATUS_FR2=$?
fi

print_worker_status() {
    local source_alias="$1"
    local gpu="$2"
    local status="$3"
    if [[ "$status" -eq 0 ]]; then
        printf '%s\n' "${source_alias} / GPU${gpu} : SUCCESS"
    else
        printf '%s\n' "${source_alias} / GPU${gpu} : FAILED (${status})"
    fi
}

printf '%s\n' \
    '============================================================' \
    'FINAL WORKER STATUS'
print_worker_status DK1 0 "$STATUS_DK1"
print_worker_status FR1 1 "$STATUS_FR1"
print_worker_status FR2 2 "$STATUS_FR2"
printf '%s\n' '============================================================'

printf '%s\n' 'TEST RESULT SUMMARY'
grep -h "Test result for" \
    "${TASK_LOG_ROOT}/DK1_source.log" \
    "${TASK_LOG_ROOT}/DK1_AT1.log" \
    "${TASK_LOG_ROOT}/DK1_FR1.log" \
    "${TASK_LOG_ROOT}/DK1_FR2.log" \
    "${TASK_LOG_ROOT}/FR1_source.log" \
    "${TASK_LOG_ROOT}/FR1_AT1.log" \
    "${TASK_LOG_ROOT}/FR1_DK1.log" \
    "${TASK_LOG_ROOT}/FR1_FR2.log" \
    "${TASK_LOG_ROOT}/FR2_source.log" \
    "${TASK_LOG_ROOT}/FR2_AT1.log" \
    "${TASK_LOG_ROOT}/FR2_DK1.log" \
    "${TASK_LOG_ROOT}/FR2_FR1.log" \
    || true

if [[ "$STATUS_DK1" -ne 0 || "$STATUS_FR1" -ne 0 || "$STATUS_FR2" -ne 0 ]]; then
    exit 1
fi

printf '%s\n' \
    '[ALL FINISHED]' \
    '9 TimeMatch tasks completed.'
exit 0
