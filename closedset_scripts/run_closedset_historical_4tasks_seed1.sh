#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
TASKS="${TASKS:-AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1}"
# Match the archived source-tile GPU allocation, including AT1 on GPU 3.
GPUS="${GPUS:-3 1 2 0}"
EXPERIMENT="${EXPERIMENT:-timematch_historical_rerun_seed1}"
OUTPUT_ROOT="outputs/${EXPERIMENT}"
LOG_ROOT="logs/${EXPERIMENT}"
mkdir -p "$LOG_ROOT"
exec >> "$LOG_ROOT/launcher.log" 2>&1

[[ "${SEED:-1}" == 1 && "${NUM_WORKERS:-8}" == 8 ]] || {
    echo 'ERROR: historical rerun requires SEED=1 and NUM_WORKERS=8'; exit 1;
}
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: dataset root missing: $DATA_ROOT"; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || {
    echo "ERROR: output already exists: $OUTPUT_ROOT; choose a new EXPERIMENT"; exit 1;
}
read -r -a tasks <<< "${TASKS//,/ }"
read -r -a gpus <<< "${GPUS//,/ }"
(( ${#tasks[@]} > 0 && ${#tasks[@]} <= ${#gpus[@]} )) || {
    echo 'ERROR: provide one GPU per selected task'; exit 1;
}

resolve_task() {
    case "$1" in
        AT1_DK1) source="austria/33UVP/2017"; target="denmark/32VNH/2017"; tile="33UVP"; target_tile="32VNH" ;;
        FR1_FR2) source="france/30TXT/2017"; target="france/31TCJ/2017"; tile="30TXT"; target_tile="31TCJ" ;;
        FR2_DK1) source="france/31TCJ/2017"; target="denmark/32VNH/2017"; tile="31TCJ"; target_tile="32VNH" ;;
        DK1_AT1) source="denmark/32VNH/2017"; target="austria/33UVP/2017"; tile="32VNH"; target_tile="33UVP" ;;
        *) echo "ERROR: unknown task $1"; return 1 ;;
    esac
}

declare -A seen_tasks=() seen_gpus=()
for index in "${!tasks[@]}"; do
    task="${tasks[$index]}"
    gpu="${gpus[$index]}"
    [[ -z "${seen_tasks[$task]:-}" && -z "${seen_gpus[$gpu]:-}" ]] || {
        echo 'ERROR: duplicate task or GPU'; exit 1;
    }
    seen_tasks[$task]=1
    seen_gpus[$gpu]=1
    resolve_task "$task"
done

# Explicit values from closedset_12tasks_3seeds_performance.log.gz, seed 1.
# Source/UDA output and TensorBoard paths are isolated from historical runs.
common=(
    --data_root "$DATA_ROOT" --num_blocks 100 --seed 1 --num_workers 8
    --device cuda --num_folds 1 --val_ratio 0.1 --test_ratio 0.2
    --sample_pixels_val true --log_step 10 --closed_set true
    --combine_spring_and_winter false --batch_size 128 --weight_decay 0.0001
    --focal_loss_gamma 1.0 --num_pixels 64 --seq_length 30
    --model pseltae --input_dim 10 --with_extra false --with_shift_aug false
    --shift_aug_p 1.0 --max_shift_aug 60
    --tensorboard_log_dir "$OUTPUT_ROOT/tensorboard"
)

run_task() {
    local gpu="$1" task="$2" source target tile target_tile
    resolve_task "$task"
    local source_experiment="closedset_source_${tile}_seed1"
    local source_weights="$OUTPUT_ROOT/source/$source_experiment"
    local uda_experiment="closedset_timematch_${tile}_to_${target_tile}_seed1"

    # Always train Source afresh, even if other source checkpoints exist.
    echo "SOURCE_START|gpu=$gpu|task=$task|seed=1|workers=8"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py "${common[@]}" \
        --source "$source" --target "$source" \
        --epochs 100 --lr 0.001 --experiment_name "$source_experiment" \
        --output_dir "$OUTPUT_ROOT/source" \
        > "$LOG_ROOT/${task}_source.log" 2>&1

    echo "UDA_START|gpu=$gpu|task=$task|weights=$source_weights"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py "${common[@]}" \
        --source "$source" --target "$target" --experiment_name "$uda_experiment" \
        --output_dir "$OUTPUT_ROOT/uda" \
        timematch --weights "$source_weights" \
        --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
        --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
        --estimate_shift true --balance_source true --use_focal_loss true \
        --shift_source true --sample_size 100 --max_temporal_shift 60 \
        --domain_specific_bn true --shift_estimator AM --run_validation \
        --output_student true --evidence_mode off \
        > "$LOG_ROOT/${task}.log" 2>&1
}

mkdir -p "$OUTPUT_ROOT"
pids=()
for index in "${!tasks[@]}"; do
    run_task "${gpus[$index]}" "${tasks[$index]}" &
    pids+=("$!")
done
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "COMPLETED|task=${tasks[$index]}|source_epochs=100|uda_epochs=20"
    else
        echo "FAILED|task=${tasks[$index]}|see=$LOG_ROOT/${tasks[$index]}_source.log,$LOG_ROOT/${tasks[$index]}.log"
        status=1
    fi
done
exit "$status"
