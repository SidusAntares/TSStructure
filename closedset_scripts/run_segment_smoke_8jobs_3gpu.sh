#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SPARSE_SWEEP="${SPARSE_SWEEP:-0}"
default_experiment=timematch_segments_smoke_seed1
if [[ "$SPARSE_SWEEP" == 1 ]]; then default_experiment=timematch_segments_sparse_sweep_seed1; fi
EXPERIMENT="${EXPERIMENT:-$default_experiment}"
OUTPUT_ROOT="outputs/$EXPERIMENT"
LOG_ROOT="logs/$EXPERIMENT"
SOURCE_ROOT="outputs/timematch_historical_rerun_seed1/source"
SMOKE_STEPS="${SMOKE_STEPS:-200}"

[[ ! -e "$OUTPUT_ROOT" && ! -e "$LOG_ROOT" ]] || {
    echo 'ERROR: smoke output or logs already exist'; exit 1;
}
mkdir -p "$LOG_ROOT"
exec >> "$LOG_ROOT/launcher.log" 2>&1
[[ "$SPARSE_SWEEP" == 0 || "$SPARSE_SWEEP" == 1 ]] || {
    echo 'ERROR: SPARSE_SWEEP must be 0 or 1'; exit 1;
}
[[ "${SEED:-1}" == 1 && "${NUM_WORKERS:-8}" == 8 ]] || {
    echo 'ERROR: smoke requires seed 1 and workers 8'; exit 1;
}
[[ "${GPUS:-0 1 2}" == '0 1 2' ]] || {
    echo 'ERROR: this queue requires GPUs 0 1 2'; exit 1;
}
[[ "$SMOKE_STEPS" =~ ^[0-9]{1,4}$ ]] && (( 10#$SMOKE_STEPS >= 1 && 10#$SMOKE_STEPS <= 1000 )) || {
    echo 'ERROR: SMOKE_STEPS must be between 1 and 1000'; exit 1;
}
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: missing dataset: $DATA_ROOT"; exit 1; }

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

base_tasks=(FR1_FR2 FR1_FR2 FR2_DK1 FR2_DK1 DK1_AT1 DK1_AT1 AT1_DK1 AT1_DK1)
base_variants=(inshape attention inshape attention inshape attention inshape attention)
weights=(0.01)
if [[ "$SPARSE_SWEEP" == 1 ]]; then
    read -r -a weights <<< "${SPARSE_WEIGHTS-0.01 0.03 0.1}"
    (( ${#weights[@]} > 0 )) || { echo 'ERROR: SPARSE_WEIGHTS is empty'; exit 1; }
    seen_weights=' '
    for weight in "${weights[@]}"; do
        case "$weight" in
            0.01|0.03|0.1) ;;
            *) echo "ERROR: unsupported sparse weight: $weight"; exit 1 ;;
        esac
        [[ "$seen_weights" != *" $weight "* ]] || {
            echo "ERROR: duplicate sparse weight: $weight"; exit 1;
        }
        seen_weights+="$weight "
    done
elif [[ -v SPARSE_WEIGHTS ]]; then
    echo 'ERROR: SPARSE_WEIGHTS requires SPARSE_SWEEP=1'; exit 1
fi
job_tasks=(); job_variants=(); job_weights=()
for weight in "${weights[@]}"; do
    for index in "${!base_tasks[@]}"; do
        job_tasks+=("${base_tasks[$index]}")
        job_variants+=("${base_variants[$index]}")
        job_weights+=("$weight")
    done
done
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
    local gpu="$1" task="$2" variant="$3" weight="$4" source target tile
    resolve_task "$task"
    local original="$SOURCE_ROOT/closedset_source_${tile}_seed1"
    local leaf="$OUTPUT_ROOT/$variant/$task"
    local log="$LOG_ROOT/${task}_${variant}.log"
    if [[ "$SPARSE_SWEEP" == 1 ]]; then
        local tag="sparse_${weight/./p}"
        leaf="$OUTPUT_ROOT/$tag/$variant/$task"
        log="$LOG_ROOT/${task}_${variant}_${tag}.log"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_segment_smoke.py \
        --source "$source" --target "$target" --variant "$variant" --weights "$original" \
        --data_root "$DATA_ROOT" --seed 1 --num_workers 8 --steps "$SMOKE_STEPS" \
        --sparse_weight "$weight" \
        --experiment_name "$task" --progress_bar off --output_dir "$leaf" \
        > "$log" 2>&1 || return $?
}

worker() {
    local gpu="$1" index task variant weight status=0
    while true; do
        until mkdir "$queue_dir/lock" 2>/dev/null; do sleep 0.05; done
        read -r index < "$queue_dir/next"
        # Reserve the first three jobs on their specified GPUs, including fast mocks.
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
        weight="${job_weights[$index]}"
        echo "TASK_START|gpu=$gpu|task=$task|variant=$variant|queue_index=$index|sparse_weight=$weight"
        rmdir "$queue_dir/lock"
        if run_job "$gpu" "$task" "$variant" "$weight"; then
            echo "TASK_COMPLETE|gpu=$gpu|task=$task|variant=$variant|steps=$SMOKE_STEPS|sparse_weight=$weight"
        else
            echo "TASK_FAILED|gpu=$gpu|task=$task|variant=$variant|status=$?|sparse_weight=$weight"
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
    echo 'ERROR: smoke job failed'
fi
exit "$status"
