#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"

DATASETS=(
    "denmark/32VNH/2017"
    "france/30TXT/2017"
    "france/31TCJ/2017"
    "austria/33UVP/2017"
)

TILE_NAMES=("32VNH" "30TXT" "31TCJ" "33UVP")
SEEDS=(1 2 3)

run_source_tasks() {
    local gpu_id="$1"
    local source_index="$2"
    local source="${DATASETS[$source_index]}"
    local source_tile="${TILE_NAMES[$source_index]}"

    for seed in "${SEEDS[@]}"; do
        local source_experiment="closedset_source_${source_tile}_seed${seed}"

        echo "[GPU ${gpu_id}] Training source ${source_tile}, seed ${seed}"
        CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u train.py \
            --data_root "$DATA_ROOT" \
            --source "$source" \
            --target "$source" \
            --seed "$seed" \
            --device cuda \
            --closed_set true \
            --combine_spring_and_winter false \
            --with_shift_aug false \
            --experiment_name "$source_experiment"

        for target_index in "${!DATASETS[@]}"; do
            if [[ "$target_index" -eq "$source_index" ]]; then
                continue
            fi

            local target="${DATASETS[$target_index]}"
            local target_tile="${TILE_NAMES[$target_index]}"
            local experiment="closedset_timematch_${source_tile}_to_${target_tile}_seed${seed}"

            echo "[GPU ${gpu_id}] Training ${source_tile} -> ${target_tile}, seed ${seed}"
            CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u train.py \
                --data_root "$DATA_ROOT" \
                --source "$source" \
                --target "$target" \
                --seed "$seed" \
                --device cuda \
                --closed_set true \
                --combine_spring_and_winter false \
                --with_shift_aug false \
                --experiment_name "$experiment" \
                timematch \
                --weights "outputs/${source_experiment}" \
                --epochs 20 \
                --steps_per_epoch 500
        done
    done
}

pids=()
for source_index in "${!DATASETS[@]}"; do
    run_source_tasks "$source_index" "$source_index" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if [[ "$status" -ne 0 ]]; then
    echo "[ERROR] At least one GPU worker failed." >&2
    exit "$status"
fi

echo "[SUCCESS] Completed 12 closed-set tasks for seeds 1, 2, and 3."
