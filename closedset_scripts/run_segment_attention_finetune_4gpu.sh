#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
EXPERIMENT="${EXPERIMENT:-timematch_segments_compare_seed1}"
OUTPUT_ROOT="outputs/$EXPERIMENT"
LOG_ROOT="logs/$EXPERIMENT"
VISUALIZATION_ROOT="outputs/${EXPERIMENT}_visualization"
VISUALIZATION_LOG="logs/${EXPERIMENT}_visualization"
SOURCE_ROOT="outputs/timematch_historical_rerun_seed1/source"
read -r -a gpu_ids <<< "${GPUS:-0 1 2 3}"
mkdir -p "$LOG_ROOT" "$VISUALIZATION_LOG"
exec >> "$LOG_ROOT/attention_launcher.log" 2>&1
[[ "${SEED:-1}" == 1 && "${NUM_WORKERS:-8}" == 8 && ${#gpu_ids[@]} == 4 ]] || {
    echo 'ERROR: requires seed 1, workers 8 and four GPU IDs'; exit 1;
}
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: missing dataset: $DATA_ROOT"; exit 1; }
[[ ! -e "$OUTPUT_ROOT/finetune_attention" && ! -e "$VISUALIZATION_ROOT/finetune_attention" ]] || {
    echo 'ERROR: attention experiment output already exists; preserved'; exit 1;
}
tasks=(FR1_FR2 FR2_DK1 DK1_AT1 AT1_DK1)
sources=(france/30TXT/2017 france/31TCJ/2017 denmark/32VNH/2017 austria/33UVP/2017)
targets=(france/31TCJ/2017 denmark/32VNH/2017 austria/33UVP/2017 denmark/32VNH/2017)
tiles=(30TXT 31TCJ 32VNH 33UVP)
for tile in "${tiles[@]}"; do
    [[ -f "$SOURCE_ROOT/closedset_source_${tile}_seed1/fold_0/model.pt" ]] || {
        echo "ERROR: missing historical Original Source $tile"; exit 1;
    }
done
# Fail on an old entry file before creating training output or starting any GPU.
"$PYTHON_BIN" -u train_segment_timematch.py --stage finetune --variant attention_shared \
    --source "${sources[0]}" --target "${targets[0]}" \
    --weights "$SOURCE_ROOT/closedset_source_30TXT_seed1" --output_dir "$OUTPUT_ROOT/finetune_attention/FR1_FR2/source" \
    --dry_run > "$LOG_ROOT/attention_preflight.log" 2>&1
run_task() {
    local index="$1" gpu="${gpu_ids[$1]}" task="${tasks[$1]}"
    local leaf="$OUTPUT_ROOT/finetune_attention/$task"
    local common=(--variant attention_shared --source "${sources[$index]}" --target "${targets[$index]}"
        --weights "$SOURCE_ROOT/closedset_source_${tiles[$index]}_seed1" --data_root "$DATA_ROOT"
        --seed 1 --num_workers 8 --experiment_name "$task" --progress_bar off)
    echo "TASK_START|gpu=$gpu|task=$task|route=finetune_attention"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_segment_timematch.py \
        --stage finetune "${common[@]}" --output_dir "$leaf/source" \
        > "$LOG_ROOT/${task}_finetune_attention_source.log" 2>&1 || return $?
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_segment_timematch.py \
        --stage uda "${common[@]}" --source_training_epochs 20 \
        --structure_checkpoint "$leaf/source/fold_0/model.pt" --output_dir "$leaf/uda" \
        > "$LOG_ROOT/${task}_finetune_attention_uda.log" 2>&1 || return $?
    echo "TASK_COMPLETE|gpu=$gpu|task=$task|route=finetune_attention|uda_epochs=20"
}
pids=()
for index in 0 1 2 3; do
    (if run_task "$index"; then exit 0; else
        status=$?
        echo "TASK_FAILED|task=${tasks[$index]}|status=$status"
        exit "$status"
    fi) &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if (( status != 0 )); then echo 'ERROR: task failed; visualization skipped'; exit "$status"; fi
echo 'ANALYSIS_START|route=finetune_attention'
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "$PYTHON_BIN" -u -m analysis.segment_structure_visualization \
    --layout attention_finetune --experiment_root "$OUTPUT_ROOT" --data_root "$DATA_ROOT" \
    --output_dir "$VISUALIZATION_ROOT" --device cuda --seed 1 --pairs_per_class 1 \
    > "$VISUALIZATION_LOG/attention_analysis.log" 2>&1
echo 'ANALYSIS_COMPLETE|route=finetune_attention'
