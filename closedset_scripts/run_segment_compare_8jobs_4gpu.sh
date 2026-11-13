#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SEGMENT_CONTEXT="${SEGMENT_CONTEXT:-0}"
SEGMENT_REFINE="${SEGMENT_REFINE:-0}"
SEGMENT_REFINE_COMPARE="${SEGMENT_REFINE_COMPARE:-0}"
SHAPELET_GUIDED_MERGE="${SHAPELET_GUIDED_MERGE:-0}"
default_adapt=5
if [[ "$SHAPELET_GUIDED_MERGE" == 1 ]]; then default_adapt=3; fi
REFINEMENT_ADAPT_EPOCHS="${REFINEMENT_ADAPT_EPOCHS:-$default_adapt}"
default_experiment=timematch_segments_compare_seed1
if [[ "$SEGMENT_CONTEXT" == 1 ]]; then default_experiment=timematch_segments_context_seed1; fi
if [[ "$SEGMENT_REFINE" == 1 ]]; then default_experiment=timematch_segments_refine_seed1; fi
if [[ "$SEGMENT_REFINE_COMPARE" == 1 ]]; then default_experiment=timematch_segments_refine_compare_seed1; fi
if [[ "$SEGMENT_REFINE_COMPARE" == 1 && "$REFINEMENT_ADAPT_EPOCHS" == 3 ]]; then
    default_experiment=timematch_segments_refine_3epoch_seed1
fi
if [[ "$SHAPELET_GUIDED_MERGE" == 1 ]]; then
    default_experiment="timematch_segments_shapelet_guided_${REFINEMENT_ADAPT_EPOCHS}epoch_seed1"
fi
EXPERIMENT="${EXPERIMENT:-$default_experiment}"
RESUME="${RESUME:-0}"
REPAIR_SOURCE="${REPAIR_SOURCE:-0}"
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
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || {
    echo 'ERROR: RESUME must be 0 or 1'; exit 1;
}
[[ "$SEGMENT_CONTEXT" == 0 || "$SEGMENT_CONTEXT" == 1 ]] || {
    echo 'ERROR: SEGMENT_CONTEXT must be 0 or 1'; exit 1;
}
[[ "$SEGMENT_REFINE" == 0 || ( "$SEGMENT_REFINE" == 1 && "$SEGMENT_CONTEXT" == 0 ) ]] || {
    echo 'ERROR: SEGMENT_REFINE must be 0 or 1 and cannot be combined with SEGMENT_CONTEXT'; exit 1;
}
[[ "$SEGMENT_REFINE_COMPARE" == 0 ||
   ( "$SEGMENT_REFINE_COMPARE" == 1 && "$SEGMENT_REFINE" == 0 && "$SEGMENT_CONTEXT" == 0 ) ]] || {
    echo 'ERROR: SEGMENT_REFINE_COMPARE must be 0 or 1 and runs independently'; exit 1;
}
[[ "$SHAPELET_GUIDED_MERGE" == 0 ||
   ( "$SHAPELET_GUIDED_MERGE" == 1 && "$SEGMENT_REFINE_COMPARE" == 1 ) ]] || {
    echo 'ERROR: SHAPELET_GUIDED_MERGE=1 requires SEGMENT_REFINE_COMPARE=1'; exit 1;
}
[[ "$REFINEMENT_ADAPT_EPOCHS" == 5 ||
   ( "$REFINEMENT_ADAPT_EPOCHS" == 3 && "$SEGMENT_REFINE_COMPARE" == 1 ) ||
   ( "$SHAPELET_GUIDED_MERGE" == 1 && "$REFINEMENT_ADAPT_EPOCHS" =~ ^[1-9][0-9]*$ ) ]] || {
    echo 'ERROR: invalid REFINEMENT_ADAPT_EPOCHS'; exit 1;
}
[[ "$REPAIR_SOURCE" == 0 ||
   ( "$REPAIR_SOURCE" == 1 && "$RESUME" == 1 && "$SEGMENT_REFINE_COMPARE" == 1 ) ]] || {
    echo 'ERROR: REPAIR_SOURCE=1 requires RESUME=1 and SEGMENT_REFINE_COMPARE=1'; exit 1;
}
[[ ! -e "$OUTPUT_ROOT" || "$RESUME" == 1 ]] || {
    echo 'ERROR: experiment output exists; use RESUME=1 to reuse final Source checkpoints'; exit 1;
}
[[ "${GPUS:-0 1 2 3}" == '0 1 2 3' ]] || {
    echo 'ERROR: this queue requires GPUs 0 1 2 3'; exit 1;
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
if [[ "$SEGMENT_REFINE_COMPARE" == 1 ]]; then
    "$PYTHON_BIN" -c '
import sys
from utils.segment_training import validate_refinement_training

adapt = int(sys.argv[1])
guided = sys.argv[2] == "1"
if adapt == 3 or guided:
    from analysis.segment_refinement_compare import expected_phases
    expected_phases(4 if guided else 3, adapt)
    validate_refinement_training(adapt, guided)
else:
    validate_refinement_training()
' "$REFINEMENT_ADAPT_EPOCHS" "$SHAPELET_GUIDED_MERGE"
fi
# Slow FR1 jobs enter first; each free GPU takes the next pair immediately.
job_tasks=(FR1_FR2 FR1_FR2 FR2_DK1 DK1_AT1 FR2_DK1 DK1_AT1 AT1_DK1 AT1_DK1)
job_variants=(joint finetune joint joint finetune finetune joint finetune)
if [[ "$SEGMENT_CONTEXT" == 1 || "$SEGMENT_REFINE" == 1 ]]; then
    job_tasks=(FR1_FR2 FR2_DK1 DK1_AT1 AT1_DK1)
    job_variants=(finetune finetune finetune finetune)
fi
if [[ "$SEGMENT_REFINE_COMPARE" == 1 ]]; then
    job_tasks=(FR1_FR2 FR2_DK1 DK1_AT1 AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1 AT1_DK1)
    job_variants=(two_rounds two_rounds two_rounds two_rounds one_round one_round one_round one_round)
    if [[ "$REFINEMENT_ADAPT_EPOCHS" == 3 ]]; then
        job_tasks=(FR1_FR2 FR2_DK1 DK1_AT1 AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1 AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1 AT1_DK1)
        job_variants=(three_rounds three_rounds three_rounds three_rounds two_rounds two_rounds two_rounds two_rounds one_round one_round one_round one_round)
    fi
    if [[ "$SHAPELET_GUIDED_MERGE" == 1 ]]; then
        job_tasks=(FR1_FR2 FR2_DK1 DK1_AT1 AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1 AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1 AT1_DK1)
        job_variants=(four_rounds four_rounds four_rounds four_rounds three_rounds three_rounds three_rounds three_rounds two_rounds two_rounds two_rounds two_rounds)
    fi
fi
mkdir -p "$OUTPUT_ROOT"
queue_dir="$(mktemp -d "$LOG_ROOT/queue.XXXXXX")"
printf '%s\n' 0 > "$queue_dir/next"
cleanup_queue() {
    rm -f "$queue_dir/next"
    rmdir "$queue_dir/lock" 2>/dev/null || true
    rmdir "$queue_dir" 2>/dev/null || true
}
trap cleanup_queue EXIT

audit_refinement_checkpoint() {
    "$PYTHON_BIN" -c '
from pathlib import Path
import sys
from analysis.segment_refinement_compare import SourceScheduleError, verify_source_checkpoint

try:
    verify_source_checkpoint(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1")
except SourceScheduleError as error:
    print(error, file=sys.stderr)
    sys.exit(2)
' "$1" "$2" "$REFINEMENT_ADAPT_EPOCHS" "$SHAPELET_GUIDED_MERGE"
}

run_job() {
    local gpu="$1" task="$2" route="$3" source target tile
    resolve_task "$task"
    local original="$SOURCE_ROOT/closedset_source_${tile}_seed1"
    local leaf="$OUTPUT_ROOT/$route/$task"
    local stage=finetune source_epochs=20
    if [[ "$route" == joint ]]; then stage=joint_source; source_epochs=100; fi
    local common=(--source "$source" --target "$target" --weights "$original"
        --data_root "$DATA_ROOT" --seed 1 --num_workers 8 --variant inshape
        --experiment_name "$task" --progress_bar off)
    if [[ "$SEGMENT_CONTEXT" == 1 ]]; then common+=(--segment_context); fi
    if [[ "$SEGMENT_REFINE" == 1 ]]; then common+=(--segment_refine); fi
    if [[ "$SEGMENT_REFINE_COMPARE" == 1 ]]; then
        local rounds=1
        if [[ "$route" == two_rounds ]]; then rounds=2; fi
        if [[ "$route" == three_rounds ]]; then rounds=3; fi
        if [[ "$route" == four_rounds ]]; then rounds=4; fi
        local probe_ready=1
        if [[ ( "$REFINEMENT_ADAPT_EPOCHS" == 3 || "$SHAPELET_GUIDED_MERGE" == 1 ) &&
              ! -s "$leaf/source/refinement/validation_probe.json" ]]; then
            probe_ready=0
        fi
        local summary="$leaf/source/fold_0/test_${source//\//_}.json"
        local log="$LOG_ROOT/${task}_${route}_source.log"
        if [[ "$RESUME" == 1 && "$probe_ready" == 1 && -s "$summary" && -s "$leaf/source/fold_0/model.pt" &&
              -s "$leaf/source/refinement/parcels.jsonl" &&
              -s "$leaf/source/refinement/prototypes.json" ]]; then
            if audit_refinement_checkpoint "$leaf/source" "$rounds"; then
                echo "SOURCE_REUSE|gpu=$gpu|task=$task|route=$route|checkpoint=$leaf/source/fold_0/model.pt"
                return 0
            else
                local audit_status=$?
                # Leave the files in place if the checkpoint could not be read.
                [[ "$audit_status" == 2 ]] || return "$audit_status"
            fi
        fi
        if [[ -e "$leaf/source" ]]; then
            if [[ "$REPAIR_SOURCE" != 1 ]]; then
                echo "ERROR: Source output preserved: $leaf/source. To replace it use RESUME=1 REPAIR_SOURCE=1."
                return 1
            fi
            local backup
            backup="$(mktemp -d "$leaf/source_before_repair.XXXXXX")" || return $?
            if [[ -f "$log" ]]; then cp -- "$log" "$backup/source.log" || return $?; fi
            mv -- "$leaf/source" "$backup/source" || return $?
            echo "SOURCE_BACKUP|gpu=$gpu|task=$task|route=$route|path=$backup"
        fi
        local source_args=(--stage finetune "${common[@]}" --refinement_rounds "$rounds" --output_dir "$leaf/source")
        if [[ "$REFINEMENT_ADAPT_EPOCHS" != 5 || "$SHAPELET_GUIDED_MERGE" == 1 ]]; then
            source_args+=(--refinement_adapt_epochs "$REFINEMENT_ADAPT_EPOCHS")
        fi
        if [[ "$SHAPELET_GUIDED_MERGE" == 1 ]]; then source_args+=(--shapelet_guided_merge); fi
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_segment_timematch.py "${source_args[@]}" \
            > "$log" 2>&1
        return $?
    fi
    if [[ "$RESUME" == 1 && -s "$leaf/source/fold_0/model.pt" ]]; then
        # UDA validates the saved role, epoch count, route, protocol and source hash.
        echo "SOURCE_REUSE|gpu=$gpu|task=$task|route=$route|checkpoint=$leaf/source/fold_0/model.pt"
    elif [[ -e "$leaf/source" ]]; then
        echo "ERROR: incomplete Source output preserved: $leaf/source"
        return 1
    else
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_segment_timematch.py \
            --stage "$stage" "${common[@]}" --output_dir "$leaf/source" \
            > "$LOG_ROOT/${task}_${route}_source.log" 2>&1 || return $?
    fi
    if [[ -e "$leaf/uda" ]]; then
        echo "ERROR: existing UDA output preserved; inspect before restarting: $leaf/uda"
        return 1
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_segment_timematch.py \
        --stage uda "${common[@]}" --source_training_epochs "$source_epochs" \
        --structure_checkpoint "$leaf/source/fold_0/model.pt" --output_dir "$leaf/uda" \
        > "$LOG_ROOT/${task}_${route}_uda.log" 2>&1 || return $?
}

worker() {
    local gpu="$1" index task variant status=0
    while true; do
        until mkdir "$queue_dir/lock" 2>/dev/null; do sleep 0.05; done
        read -r index < "$queue_dir/next"
        # Reserve one initial job per GPU, even when a mocked command finishes
        # before the other worker processes have reached the queue.
        if (( index < 4 && index != gpu )); then
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
        echo "TASK_START|gpu=$gpu|task=$task|route=$variant|queue_index=$index"
        rmdir "$queue_dir/lock"
        if run_job "$gpu" "$task" "$variant"; then
            if [[ "$SEGMENT_REFINE_COMPARE" == 1 ]]; then
                echo "TASK_COMPLETE|gpu=$gpu|task=$task|route=$variant|stage=source"
            else
                echo "TASK_COMPLETE|gpu=$gpu|task=$task|route=$variant|uda_epochs=20"
            fi
        else
            echo "TASK_FAILED|gpu=$gpu|task=$task|route=$variant|status=$?"
            status=1
        fi
    done
    return "$status"
}

pids=()
for gpu in 0 1 2 3; do
    worker "$gpu" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if [[ "$status" == 0 && ( "$SEGMENT_REFINE" == 1 || "$SEGMENT_REFINE_COMPARE" == 1 ) ]]; then
    mkdir -p "$VISUALIZATION_LOG"
    analysis_args=(
        --experiment_root "$OUTPUT_ROOT"
        --data_root "$DATA_ROOT"
        --output_dir "$VISUALIZATION_ROOT"
    )
    analysis_module=analysis.segment_cut_transfer
    analysis_mode=refined_plots
    if [[ "$SEGMENT_REFINE_COMPARE" == 1 ]]; then
        analysis_module=analysis.segment_refinement_compare
        analysis_mode=source_refinement_compare
        analysis_args+=(--plot_task "${PLOT_TASK:-AT1_DK1}")
        if [[ "$REFINEMENT_ADAPT_EPOCHS" != 5 ]]; then analysis_args+=(--refinement_adapt_epochs "$REFINEMENT_ADAPT_EPOCHS"); fi
        if [[ "$SHAPELET_GUIDED_MERGE" == 1 ]]; then analysis_args+=(--shapelet_guided_merge); fi
    else
        analysis_args=(--mode refined_plots "${analysis_args[@]}")
    fi
    echo "ANALYSIS_START|mode=$analysis_mode"
    if CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u -m "$analysis_module" "${analysis_args[@]}" > "$VISUALIZATION_LOG/analysis.log" 2>&1; then
        echo "ANALYSIS_COMPLETE|mode=$analysis_mode"
    else
        echo "ANALYSIS_FAILED|mode=$analysis_mode"
        status=1
    fi
fi
exit "$status"
