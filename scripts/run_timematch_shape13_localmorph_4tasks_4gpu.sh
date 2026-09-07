#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
AT1_WEIGHTS="${AT1_WEIGHTS:-outputs/pseltae_AT1_source_seed1}"
DK1_WEIGHTS="${DK1_WEIGHTS:-outputs/pseltae_DK1_source_seed1}"
FR1_WEIGHTS="${FR1_WEIGHTS:-outputs/pseltae_FR1_source_seed1}"
FR2_WEIGHTS="${FR2_WEIGHTS:-outputs/pseltae_FR2_source_seed1}"
SEED=1
RUN_ROOT="timematch_shape13_localmorph_4tasks_seed${SEED}"
OUTPUT_ROOT="outputs/${RUN_ROOT}"
TENSORBOARD_ROOT="runs/${RUN_ROOT}"
LOG_ROOT="logs/${RUN_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHON_BIN DATA_ROOT SEED OUTPUT_ROOT TENSORBOARD_ROOT LOG_ROOT

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}" >&2
    exit 1
fi

mkdir -p logs outputs runs "$OUTPUT_ROOT" "$TENSORBOARD_ROOT" "$LOG_ROOT"
for directory in logs outputs runs "$OUTPUT_ROOT" "$TENSORBOARD_ROOT" "$LOG_ROOT"; do
    if [[ ! -w "$directory" ]]; then
        echo "ERROR: output directory is not writable: ${directory}" >&2
        exit 1
    fi
done

check_weights() {
    local alias="$1"
    local weights="$2"
    local expected_source="$3"
    if [[ ! -f "${weights}/fold_0/model.pt" ]]; then
        echo "ERROR: ${alias} PseLTae checkpoint not found: ${weights}/fold_0/model.pt" >&2
        exit 1
    fi
    if [[ ! -f "${weights}/train_config.json" ]]; then
        echo "ERROR: ${alias} train_config.json not found: ${weights}/train_config.json" >&2
        exit 1
    fi
    local model source target seed num_folds
    IFS=$'\t' read -r model source target seed num_folds < <(
        "$PYTHON_BIN" -c \
            'import json,sys; c=json.load(open(sys.argv[1], encoding="utf-8")); print(c.get("model", ""), c.get("source", ""), c.get("target", ""), c.get("seed", ""), c.get("num_folds", ""), sep="\t")' \
            "${weights}/train_config.json"
    )
    if [[ "$model" != "pseltae" ]]; then
        echo "ERROR: ${alias} weights must use model=pseltae; found model=${model}" >&2
        exit 1
    fi
    if [[ "$source" != "$expected_source" || "$target" != "$expected_source" ]]; then
        echo "ERROR: ${alias} weights must be source-trained with source=target=${expected_source}; found source=${source} target=${target}" >&2
        exit 1
    fi
    if [[ "$seed" != "1" || "$num_folds" != "1" ]]; then
        echo "ERROR: ${alias} weights must use seed=1 and num_folds=1; found seed=${seed} num_folds=${num_folds}" >&2
        exit 1
    fi
}

print_task() {
    local gpu="$1"
    local source_alias="$2"
    local target_alias="$3"
    local weights="$4"
    local experiment="timematch_shape13_localmorph_${source_alias}_${target_alias}_seed${SEED}"
    local log_path="${LOG_ROOT}/${experiment}.log"
    printf '%s\n' \
        "============================================================" \
        "experiment=${experiment}" \
        "GPU=${gpu}" \
        "source=${source_alias}" \
        "target=${target_alias}" \
        "seed=${SEED}" \
        "source_weights=${weights}" \
        "shape_align=true" \
        "shape_modes=13" \
        "shape_loss_type=local_morph" \
        "shape_local_window_points=16" \
        "shape_local_stride_points=8" \
        "shape_local_slope_weight=0.5" \
        "shape_class_balanced=true" \
        "shape_lambda=0.1" \
        "shape_event_weight=0.0" \
        "class_residual_phase=false" \
        "log=${log_path}" \
        "output=${OUTPUT_ROOT}/${experiment}" \
        "============================================================"
}

run_worker() {
    local gpu="$1"
    local source_alias="$2"
    local source="$3"
    local target_alias="$4"
    local target="$5"
    local weights="$6"
    local experiment="timematch_shape13_localmorph_${source_alias}_${target_alias}_seed${SEED}"

    print_task "$gpu" "$source_alias" "$target_alias" "$weights"
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
        --shape_loss_type local_morph \
        --shape_local_window_points 16 \
        --shape_local_stride_points 8 \
        --shape_local_slope_weight 0.5 \
        --shape_class_balanced true \
        --class_residual_phase false
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    run_worker "$@"
    exit $?
fi

check_weights AT1 "$AT1_WEIGHTS" "$AT1"
check_weights DK1 "$DK1_WEIGHTS" "$DK1"
check_weights FR1 "$FR1_WEIGHTS" "$FR1"
check_weights FR2 "$FR2_WEIGHTS" "$FR2"

launch() {
    local gpu="$1"
    local source_alias="$2"
    local source="$3"
    local target_alias="$4"
    local target="$5"
    local weights="$6"
    local experiment="timematch_shape13_localmorph_${source_alias}_${target_alias}_seed${SEED}"
    local log_path="${LOG_ROOT}/${experiment}.log"

    print_task "$gpu" "$source_alias" "$target_alias" "$weights"
    nohup bash "$SCRIPT_PATH" --worker \
        "$gpu" "$source_alias" "$source" "$target_alias" "$target" "$weights" \
        > "$log_path" 2>&1 &
    local pid=$!
    printf 'LAUNCHED GPU=%s PID=%s source=%s target=%s log=%s output=%s\n' \
        "$gpu" "$pid" "$source_alias" "$target_alias" \
        "$log_path" "${OUTPUT_ROOT}/${experiment}"
}

launch "$GPU0" AT1 "$AT1" DK1 "$DK1" "$AT1_WEIGHTS"
launch "$GPU1" DK1 "$DK1" FR1 "$FR1" "$DK1_WEIGHTS"
launch "$GPU2" FR1 "$FR1" FR2 "$FR2" "$FR1_WEIGHTS"
launch "$GPU3" FR2 "$FR2" AT1 "$AT1" "$FR2_WEIGHTS"

echo "Four TimeMatch + Shape13 Local Morphology pilots launched."
