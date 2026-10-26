#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# All datasets and pretrained checkpoints live on the offline server.
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
TASKS="${TASKS:-AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1}"
GPUS="${GPUS:-0 1 2 3}"
SEED="${SEED:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
UDA_EPOCHS="${UDA_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
REUSE_SOURCE="${REUSE_SOURCE:-1}"
TOP_FRACTION="${TOP_FRACTION:-0.30}"
COVERAGE="${COVERAGE:-0.75}"
MIN_OBSERVATIONS="${MIN_OBSERVATIONS:-3}"
POS_WEIGHT="${POS_WEIGHT:-0.10}"
RANK_WEIGHT="${RANK_WEIGHT:-0.10}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "[ERROR] Dataset root not found: $DATA_ROOT" >&2
    exit 1
fi

read -r -a selected_tasks <<< "${TASKS//,/ }"
read -r -a selected_gpus <<< "${GPUS//,/ }"
if (( ${#selected_tasks[@]} == 0 || ${#selected_tasks[@]} > ${#selected_gpus[@]} )); then
    echo "[ERROR] Provide one GPU for each selected task (TASKS / GPUS)." >&2
    exit 1
fi

run_task() {
    local gpu="$1"
    local task="$2"
    local source target source_tile target_tile
    case "$task" in
        AT1_DK1)
            source="austria/33UVP/2017"; target="denmark/32VNH/2017"
            source_tile="33UVP"; target_tile="32VNH" ;;
        FR1_FR2)
            source="france/30TXT/2017"; target="france/31TCJ/2017"
            source_tile="30TXT"; target_tile="31TCJ" ;;
        FR2_DK1)
            source="france/31TCJ/2017"; target="denmark/32VNH/2017"
            source_tile="31TCJ"; target_tile="32VNH" ;;
        DK1_AT1)
            source="denmark/32VNH/2017"; target="austria/33UVP/2017"
            source_tile="32VNH"; target_tile="33UVP" ;;
        *)
            echo "[ERROR] Unknown task: $task" >&2; return 1 ;;
    esac

    local source_experiment="closedset_source_${source_tile}_seed${SEED}"
    local source_weights="outputs/${source_experiment}"

    # Each GPU runs its own source model, then immediately starts adaptation.
    # There is deliberately no cross-GPU source-stage barrier.
    if [[ "$REUSE_SOURCE" != "1" || ! -f "${source_weights}/fold_0/model.pt" ]]; then
        echo "[GPU $gpu] Training source $source_tile, seed $SEED"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
            --data_root "$DATA_ROOT" --source "$source" --target "$source" \
            --seed "$SEED" --num_workers "$NUM_WORKERS" \
            --device cuda --closed_set true \
            --combine_spring_and_winter false --with_shift_aug false \
            --progress_bar off --epochs "$SOURCE_EPOCHS" \
            --experiment_name "$source_experiment"
    else
        echo "[GPU $gpu] Reusing verified-path source checkpoint: $source_weights"
    fi

    local mode experiment
    for mode in positive positive_rank; do
        experiment="head_evidence_${mode}_${source_tile}_to_${target_tile}_seed${SEED}"
        echo "[GPU $gpu] $task: $mode from the same source checkpoint"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
            --data_root "$DATA_ROOT" --source "$source" --target "$target" \
            --seed "$SEED" --num_workers "$NUM_WORKERS" \
            --device cuda --closed_set true \
            --combine_spring_and_winter false --with_shift_aug false \
            --progress_bar off --experiment_name "$experiment" \
            timematch --weights "$source_weights" \
            --epochs "$UDA_EPOCHS" --steps_per_epoch "$STEPS_PER_EPOCH" \
            --evidence_mode "$mode" \
            --evidence_top_fraction "$TOP_FRACTION" \
            --evidence_coverage "$COVERAGE" \
            --evidence_min_observations "$MIN_OBSERVATIONS" \
            --evidence_positive_weight "$POS_WEIGHT" \
            --evidence_rank_weight "$RANK_WEIGHT" \
            --evidence_warmup_steps "$WARMUP_STEPS"
    done
    echo "[GPU $gpu] Completed $task"
}

pids=()
for index in "${!selected_tasks[@]}"; do
    run_task "${selected_gpus[$index]}" "${selected_tasks[$index]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
