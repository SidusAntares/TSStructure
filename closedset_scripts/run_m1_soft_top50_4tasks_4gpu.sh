#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
MODE="${MODE:-M1}"
[[ "$MODE" == M0 || "$MODE" == M1 ]] || { echo 'ERROR: MODE must be M0 or M1'; exit 1; }
EXPERIMENT="${EXPERIMENT:-${MODE,,}_soft_top50_seed1}"
OUTPUT_ROOT="outputs/${EXPERIMENT}"
LOG_ROOT="logs/${EXPERIMENT}"
mkdir -p "$LOG_ROOT"
exec >> "$LOG_ROOT/launcher.log" 2>&1
[[ "${SEED:-1}" == 1 && "${NUM_WORKERS:-8}" == 8 ]] || {
    echo 'ERROR: matched Source experiments require SEED=1 and NUM_WORKERS=8'; exit 1;
}
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: dataset root missing: $DATA_ROOT"; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "ERROR: output already exists: $OUTPUT_ROOT"; exit 1; }
read -r -a tasks <<< "${TASKS:-AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1}"
read -r -a gpus <<< "${GPUS:-3 1 2 0}"
(( ${#tasks[@]} > 0 && ${#tasks[@]} <= ${#gpus[@]} )) || {
    echo 'ERROR: provide one GPU per task'; exit 1;
}
resolve_task() {
    case "$1" in
        AT1_DK1) source='austria/33UVP/2017'; tile=33UVP ;;
        FR1_FR2) source='france/30TXT/2017'; tile=30TXT ;;
        FR2_DK1) source='france/31TCJ/2017'; tile=31TCJ ;;
        DK1_AT1) source='denmark/32VNH/2017'; tile=32VNH ;;
        *) echo "ERROR: unknown task $1"; return 1 ;;
    esac
}
declare -A seen_tasks=() seen_gpus=()
for index in "${!tasks[@]}"; do
    task="${tasks[$index]}"; gpu="${gpus[$index]}"
    [[ -z "${seen_tasks[$task]:-}" && -z "${seen_gpus[$gpu]:-}" ]] || {
        echo 'ERROR: duplicate task or GPU'; exit 1;
    }
    seen_tasks[$task]=1; seen_gpus[$gpu]=1
    resolve_task "$task"
done
run_task() {
    local gpu="$1" task="$2" source tile
    resolve_task "$task"
    echo "SOURCE_START|gpu=$gpu|task=$task|mode=$MODE|seed=1|workers=8"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_m1_source.py \
        --mode "$MODE" --data_root "$DATA_ROOT" --source "$source" --target "$source" \
        --output_dir "$OUTPUT_ROOT/source" --experiment_name "${MODE,,}_source_${tile}_seed1" \
        --tensorboard_log_dir "$OUTPUT_ROOT/tensorboard" --seed 1 --num_workers 8 \
        --device cuda --epochs 100 --batch_size 128 --lr 0.001 --weight_decay 0.0001 \
        --num_pixels 64 --seq_length 30 --focal_loss_gamma 1.0 --model pseltae \
        --num_folds 1 --num_blocks 100 --val_ratio 0.1 --test_ratio 0.2 \
        --input_dim 10 --sample_pixels_val true --closed_set true \
        --combine_spring_and_winter false --with_extra false --with_shift_aug false \
        --shift_aug_p 1.0 --max_shift_aug 60 --log_step 10 \
        --m1_rho "${M1_RHO:-0.5}" --m1_lambda_keep "${M1_LAMBDA_KEEP:-1.0}" \
        --m1_lambda_budget "${M1_LAMBDA_BUDGET:-0.1}" --m1_sharpness "${M1_SHARPNESS:-1.0}" \
        --m1_tau_start "${M1_TAU_START:-0.5}" --m1_tau_end "${M1_TAU_END:-0.05}" \
        --m1_warmup_epochs "${M1_WARMUP_EPOCHS:-5}" \
        --m1_degenerate_tolerance "${M1_DEGENERATE_TOLERANCE:-0.001}" \
        --m1_degenerate_patience "${M1_DEGENERATE_PATIENCE:-5}" \
        > "$LOG_ROOT/${task}_source.log" 2>&1
}
mkdir -p "$(dirname "$OUTPUT_ROOT")"
mkdir "$OUTPUT_ROOT"
pids=()
for index in "${!tasks[@]}"; do
    run_task "${gpus[$index]}" "${tasks[$index]}" &
    pids+=("$!")
done
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "COMPLETED|task=${tasks[$index]}|source_epochs=100|mode=$MODE"
    else
        echo "FAILED|task=${tasks[$index]}|see=$LOG_ROOT/${tasks[$index]}_source.log"
        status=1
    fi
done
exit "$status"
