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
AT1_WEIGHTS="${AT1_WEIGHTS:-outputs/pseltae_33UVP}"
DK1_WEIGHTS="${DK1_WEIGHTS:-outputs/pseltae_32VNH}"
FR1_WEIGHTS="${FR1_WEIGHTS:-outputs/pseltae_30TXT}"
FR2_WEIGHTS="${FR2_WEIGHTS:-outputs/pseltae_31TCJ}"
SEED="${SEED:-1}"
RUN_ROOT="timematch_shape13_classphase_4tasks_seed${SEED}"
OUTPUT_ROOT="outputs/${RUN_ROOT}"
TENSORBOARD_ROOT="runs/${RUN_ROOT}"
LOG_ROOT="logs/${RUN_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHON_BIN DATA_ROOT SEED OUTPUT_ROOT TENSORBOARD_ROOT LOG_ROOT

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"

mkdir -p logs outputs runs "$OUTPUT_ROOT" "$TENSORBOARD_ROOT" "$LOG_ROOT"

check_weights() {
    local alias="$1"
    local weights="$2"
    if [[ ! -f "${weights}/fold_0/model.pt" ]]; then
        echo "ERROR: ${alias} PseLTae checkpoint not found: ${weights}/fold_0/model.pt" >&2
        exit 1
    fi
    if [[ ! -f "${weights}/train_config.json" ]]; then
        echo "ERROR: ${alias} train_config.json not found: ${weights}/train_config.json" >&2
        exit 1
    fi
    local model
    model="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("model", ""))' "${weights}/train_config.json")"
    if [[ "$model" != "pseltae" ]]; then
        echo "ERROR: ${alias} weights must use model=pseltae; found model=${model}" >&2
        exit 1
    fi
}

run_worker() {
    local gpu="$1"
    local source_alias="$2"
    local source="$3"
    local target_alias="$4"
    local target="$5"
    local weights="$6"
    local experiment="timematch_shape13_classphase_${source_alias}_${target_alias}_seed${SEED}"

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
        -e "$experiment" \
        --data_root "$DATA_ROOT" \
        --source "$source" \
        --target "$target" \
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
        --weights "$weights" \
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
        --shape_modes 13 \
        --shape_lambda 0.1 \
        --shape_morph_weight 1.0 \
        --shape_event_weight 0.0 \
        --shape_grid_points 64 \
        --shape_reference_per_class 128 \
        --shape_reference_seed 1 \
        --shape_prominence_rel 0.15 \
        --shape_min_distance_days 14 \
        --shape_fourier_period_days 365 \
        --shape_fourier_reg 0.001 \
        --shape_diag_batches 10 \
        --class_residual_phase true \
        --class_phase_mode 9 \
        --class_phase_radius_days 7 \
        --class_phase_step_days 1 \
        --class_phase_start_epoch 1 \
        --class_phase_min_samples 32 \
        --class_phase_max_samples_per_class 128 \
        --class_phase_min_corr_gain 0.005 \
        --class_phase_seed 1
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    run_worker "$@"
    exit $?
fi

check_weights AT1 "$AT1_WEIGHTS"
check_weights DK1 "$DK1_WEIGHTS"
check_weights FR1 "$FR1_WEIGHTS"
check_weights FR2 "$FR2_WEIGHTS"

launch() {
    local gpu="$1"
    local source_alias="$2"
    local source="$3"
    local target_alias="$4"
    local target="$5"
    local weights="$6"
    local experiment="timematch_shape13_classphase_${source_alias}_${target_alias}_seed${SEED}"
    local log_path="${LOG_ROOT}/${experiment}.log"
    nohup bash "$0" --worker \
        "$gpu" "$source_alias" "$source" "$target_alias" "$target" "$weights" \
        > "$log_path" 2>&1 &
    local pid=$!
    printf 'GPU=%s PID=%s source=%s target=%s output=%s log=%s\n' \
        "$gpu" "$pid" "$source_alias" "$target_alias" \
        "${OUTPUT_ROOT}/${experiment}" "$log_path"
}

launch "$GPU0" AT1 "$AT1" DK1 "$DK1" "$AT1_WEIGHTS"
launch "$GPU1" DK1 "$DK1" FR1 "$FR1" "$DK1_WEIGHTS"
launch "$GPU2" FR1 "$FR1" FR2 "$FR2" "$FR1_WEIGHTS"
launch "$GPU3" FR2 "$FR2" AT1 "$AT1" "$FR2_WEIGHTS"

echo "Four TimeMatch + Shape13 + class residual phase pilots launched."
