#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/timematch_historical_rerun_seed1/source}"
EXPERIMENT="${EXPERIMENT:-timematch_ltae_finetune_original_uda_seed1}"
OUTPUT_ROOT="outputs/$EXPERIMENT"
LOG_ROOT="logs/$EXPERIMENT"
read -r -a gpus <<< "${GPUS:-3 1 2 0}"
tasks=(AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1)
sources=(austria/33UVP/2017 france/30TXT/2017 france/31TCJ/2017 denmark/32VNH/2017)
targets=(denmark/32VNH/2017 france/31TCJ/2017 denmark/32VNH/2017 austria/33UVP/2017)
tiles=(33UVP 30TXT 31TCJ 32VNH)

mkdir -p "$LOG_ROOT"
exec >> "$LOG_ROOT/launcher.log" 2>&1
[[ ${#gpus[@]} == 4 && "${SEED:-1}" == 1 && "${NUM_WORKERS:-8}" == 8 ]] || {
    echo 'ERROR: provide four GPUs and use seed 1 with 8 workers'; exit 1;
}
declare -A seen=()
for gpu in "${gpus[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ && -z "${seen[$gpu]:-}" ]] || { echo "ERROR: invalid or duplicate GPU $gpu"; exit 1; }
    seen[$gpu]=1
done
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: dataset root missing: $DATA_ROOT"; exit 1; }
for tile in "${tiles[@]}"; do
    original="$SOURCE_ROOT/closedset_source_${tile}_seed1/fold_0/model.pt"
    [[ -s "$original" ]] || { echo "ERROR: missing original Source checkpoint $original"; exit 1; }
done
mkdir -p outputs
mkdir "$OUTPUT_ROOT" || { echo "ERROR: existing output preserved: $OUTPUT_ROOT"; exit 1; }

run_task() {
    local index="$1" task="${tasks[$1]}" gpu="${gpus[$1]}"
    local original="$SOURCE_ROOT/closedset_source_${tiles[$index]}_seed1"
    local fine="$OUTPUT_ROOT/finetune/$task"
    local domains=(--source "${sources[$index]}" --target "${targets[$index]}" --data_root "$DATA_ROOT")
    echo "FINETUNE_START|task=$task|gpu=$gpu|weights=$original"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_segment_timematch.py \
        --stage finetune --variant inshape --finetune_ltae_only "${domains[@]}" \
        --weights "$original" --output_dir "$fine" --experiment_name "$task" \
        --seed 1 --num_workers 8 --progress_bar off > "$LOG_ROOT/${task}_finetune.log" 2>&1

    echo "ORIGINAL_UDA_START|task=$task|gpu=$gpu|weights=$fine"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py "${domains[@]}" \
        --num_blocks 100 --seed 1 --num_workers 8 --device cuda --num_folds 1 \
        --val_ratio 0.1 --test_ratio 0.2 --sample_pixels_val true --log_step 10 \
        --closed_set true --combine_spring_and_winter false --batch_size 128 --weight_decay 0.0001 \
        --focal_loss_gamma 1.0 --num_pixels 64 --seq_length 30 --model pseltae --input_dim 10 \
        --with_extra false --with_shift_aug false --shift_aug_p 1.0 --max_shift_aug 60 \
        --experiment_name "$task" --output_dir "$OUTPUT_ROOT/uda" \
        --tensorboard_log_dir "$OUTPUT_ROOT/tensorboard/$task" \
        timematch --weights "$fine" --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
        --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
        --estimate_shift true --balance_source true --use_focal_loss true --shift_source true \
        --sample_size 100 --max_temporal_shift 60 --domain_specific_bn true --shift_estimator AM \
        --run_validation --output_student true --strict_final_student true --evidence_mode off \
        > "$LOG_ROOT/${task}.log" 2>&1
}

pids=()
for index in "${!tasks[@]}"; do
    run_task "$index" &
    pids+=("$!")
done
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "COMPLETED|task=${tasks[$index]}|finetune_epochs=20|uda_epochs=20"
    else
        echo "FAILED|task=${tasks[$index]}|see=$LOG_ROOT/${tasks[$index]}_finetune.log,$LOG_ROOT/${tasks[$index]}.log"
        status=1
    fi
done
exit "$status"
