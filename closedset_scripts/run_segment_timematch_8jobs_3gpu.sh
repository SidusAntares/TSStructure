#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
EXPERIMENT="${EXPERIMENT:-timematch_segments_seed1}"
OUTPUT_ROOT="outputs/$EXPERIMENT"
LOG_ROOT="logs/$EXPERIMENT"
VISUALIZATION_ROOT="outputs/${EXPERIMENT}_visualization"
VISUALIZATION_LOG="logs/${EXPERIMENT}_visualization"
SOURCE_ROOT="outputs/timematch_historical_rerun_seed1/source"
mkdir -p "$LOG_ROOT"
exec >> "$LOG_ROOT/launcher.log" 2>&1

[[ "${SEED:-1}" == 1 && "${NUM_WORKERS:-8}" == 8 ]] || {
    echo 'ERROR: experiment requires seed 1 and workers 8'; exit 1;
}
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: missing dataset: $DATA_ROOT"; exit 1; }
[[ ! -e "$OUTPUT_ROOT" && ! -e "$VISUALIZATION_ROOT" ]] || {
    echo 'ERROR: experiment or visualization output exists'; exit 1;
}
[[ "${GPUS:-0 1 2}" == '0 1 2' ]] || {
    echo 'ERROR: this queue requires GPUs 0 1 2'; exit 1;
}
resolve_task() {
    case "$1" in
        AT1_DK1) source='austria/33UVP/2017'; target='denmark/32VNH/2017'; tile='33UVP' ;;
        FR1_FR2) source='france/30TXT/2017'; target='france/31TCJ/2017'; tile='30TXT' ;;
        FR2_DK1) source='france/31TCJ/2017'; target='denmark/32VNH/2017'; tile='31TCJ' ;;
        DK1_AT1) source='denmark/32VNH/2017'; target='austria/33UVP/2017'; tile='32VNH' ;;
        *) echo "ERROR: unknown task $1"; return 1 ;;
    esac
}
for task in FR1_FR2 FR2_DK1 DK1_AT1 AT1_DK1; do
    resolve_task "$task"
    [[ -f "$SOURCE_ROOT/closedset_source_${tile}_seed1/fold_0/model.pt" ]] || {
        echo "ERROR: historical Original Source missing for $task"; exit 1;
    }
done
# Slow FR1 jobs enter first; each free GPU takes the next pair immediately.
job_tasks=(FR1_FR2 FR1_FR2 FR2_DK1 FR2_DK1 DK1_AT1 DK1_AT1 AT1_DK1 AT1_DK1)
job_variants=(inshape attention inshape attention inshape attention inshape attention)
mkdir "$OUTPUT_ROOT"
queue_dir="$(mktemp -d "$LOG_ROOT/queue.XXXXXX")"
printf '%s\n' 0 > "$queue_dir/next"
cleanup_queue() {
    rm -f "$queue_dir/next"
    rmdir "$queue_dir/lock" 2>/dev/null || true
    rmdir "$queue_dir" 2>/dev/null || true
}
trap cleanup_queue EXIT

run_job() {
    local gpu="$1" task="$2" variant="$3" source target tile
    resolve_task "$task"
    local original="$SOURCE_ROOT/closedset_source_${tile}_seed1"
    local leaf="$OUTPUT_ROOT/$variant/$task"
    local common=(--source "$source" --target "$target" --weights "$original"
        --data_root "$DATA_ROOT" --seed 1 --num_workers 8 --epochs 20
        --experiment_name "$task" --progress_bar off)
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_segment_timematch.py \
        --stage finetune --variant "$variant" "${common[@]}" \
        --output_dir "$leaf/finetune" --tensorboard_log_dir "$leaf/finetune/tensorboard" \
        > "$LOG_ROOT/${task}_${variant}_finetune.log" 2>&1 || return $?
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_segment_timematch.py \
        --stage uda --variant "$variant" "${common[@]}" --steps_per_epoch 500 \
        --structure_checkpoint "$leaf/finetune/fold_0/model.pt" \
        --output_dir "$leaf/uda" --tensorboard_log_dir "$leaf/uda/tensorboard" \
        > "$LOG_ROOT/${task}_${variant}_uda.log" 2>&1 || return $?
}

worker() {
    local gpu="$1" index task variant status=0
    while true; do
        until mkdir "$queue_dir/lock" 2>/dev/null; do sleep 0.05; done
        read -r index < "$queue_dir/next"
        # Reserve one initial job per GPU, even when a mocked command finishes
        # before the other worker processes have reached the queue.
        if (( index < 3 && index != gpu )); then
            rmdir "$queue_dir/lock"
            sleep 0.05
            continue
        fi
        if (( index >= ${#job_tasks[@]} )); then
            rmdir "$queue_dir/lock"
            break
        fi
        printf '%s\n' "$((index + 1))" > "$queue_dir/next"
        task="${job_tasks[$index]}"; variant="${job_variants[$index]}"
        echo "TASK_START|gpu=$gpu|task=$task|variant=$variant|queue_index=$index"
        rmdir "$queue_dir/lock"
        if run_job "$gpu" "$task" "$variant"; then
            echo "TASK_COMPLETE|gpu=$gpu|task=$task|variant=$variant|finetune_epochs=20|uda_epochs=20"
        else
            echo "TASK_FAILED|gpu=$gpu|task=$task|variant=$variant|status=$?"
            status=1
        fi
    done
    return "$status"
}

pids=()
for gpu in 0 1 2; do
    worker "$gpu" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if (( status != 0 )); then
    echo 'ERROR: training job failed; visualization skipped'
    exit "$status"
fi
mkdir -p "$VISUALIZATION_LOG"
echo 'ANALYSIS_START'
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u -m analysis.segment_structure_visualization \
    --experiment_root "$OUTPUT_ROOT" --data_root "$DATA_ROOT" \
    --output_dir "$VISUALIZATION_ROOT" --device cuda --num_workers 8 --seed 1 \
    --tasks AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1 --variants inshape attention \
    > "$VISUALIZATION_LOG/analysis.log" 2>&1
echo 'ANALYSIS_COMPLETE'
