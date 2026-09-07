#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
SEED="${SEED:-1}"
FOLD="${FOLD:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
AT1_WEIGHTS="${AT1_WEIGHTS:-outputs/pseltae_AT1_source_seed1}"
DK1_WEIGHTS="${DK1_WEIGHTS:-outputs/pseltae_DK1_source_seed1}"
FR1_WEIGHTS="${FR1_WEIGHTS:-outputs/pseltae_FR1_source_seed1}"
FR2_WEIGHTS="${FR2_WEIGHTS:-outputs/pseltae_FR2_source_seed1}"

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"

RUN_ROOT="timematch_ablation_seed${SEED}"
OUTPUT_ROOT="outputs/${RUN_ROOT}"
TENSORBOARD_ROOT="runs/${RUN_ROOT}"
LOG_ROOT="logs/${RUN_ROOT}"
STATUS_FILE="${OUTPUT_ROOT}/task_status.csv"
SUMMARY_FILE="${OUTPUT_ROOT}/summary.csv"
export PYTHONUNBUFFERED=1

if [[ "$FOLD" != "0" ]]; then
    echo "ERROR: only FOLD=0 is supported because train.py has no fold selector" >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}" >&2
    exit 1
fi

mkdir -p \
    logs outputs runs \
    "$OUTPUT_ROOT/original" "$OUTPUT_ROOT/local_sample" \
    "$TENSORBOARD_ROOT/original" "$TENSORBOARD_ROOT/local_sample" \
    "$LOG_ROOT/original" "$LOG_ROOT/local_sample"
for directory in \
    logs outputs runs \
    "$OUTPUT_ROOT" "$TENSORBOARD_ROOT" "$LOG_ROOT" \
    "$OUTPUT_ROOT/original" "$OUTPUT_ROOT/local_sample" \
    "$TENSORBOARD_ROOT/original" "$TENSORBOARD_ROOT/local_sample" \
    "$LOG_ROOT/original" "$LOG_ROOT/local_sample"; do
    if [[ ! -w "$directory" ]]; then
        echo "ERROR: directory is not writable: ${directory}" >&2
        exit 1
    fi
done

check_weights() {
    local alias="$1"
    local weights="$2"
    local expected_domain="$3"
    local checkpoint="${weights}/fold_${FOLD}/model.pt"
    local config_path="${weights}/train_config.json"
    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: ${alias} PseLTae checkpoint not found: ${checkpoint}" >&2
        exit 1
    fi
    if [[ ! -f "$config_path" ]]; then
        echo "ERROR: ${alias} train_config.json not found: ${config_path}" >&2
        exit 1
    fi
    local model source target seed num_folds
    IFS=$'\t' read -r model source target seed num_folds < <(
        "$PYTHON_BIN" -c \
            'import json,sys; c=json.load(open(sys.argv[1], encoding="utf-8")); print(c.get("model", ""), c.get("source", ""), c.get("target", ""), c.get("seed", ""), c.get("num_folds", ""), sep="\t")' \
            "$config_path"
    )
    if [[ "$model" != "pseltae" ]]; then
        echo "ERROR: ${alias} checkpoint must use model=pseltae; found ${model}" >&2
        exit 1
    fi
    if [[ "$source" != "$expected_domain" || "$target" != "$expected_domain" ]]; then
        echo "ERROR: ${alias} checkpoint must have source=target=${expected_domain}; found source=${source} target=${target}" >&2
        exit 1
    fi
    if [[ "$seed" != "$SEED" || "$num_folds" != "1" ]]; then
        echo "ERROR: ${alias} checkpoint must use seed=${SEED}, num_folds=1; found seed=${seed}, num_folds=${num_folds}" >&2
        exit 1
    fi
}

check_weights AT1 "$AT1_WEIGHTS" "$AT1"
check_weights DK1 "$DK1_WEIGHTS" "$DK1"
check_weights FR1 "$FR1_WEIGHTS" "$FR1"
check_weights FR2 "$FR2_WEIGHTS" "$FR2"

check_clean_destinations() {
    local group task experiment task_output task_log
    for group in original local_sample; do
        for task in AT1_DK1 DK1_FR1 FR1_FR2 FR2_AT1; do
            experiment="timematch_${group}_${task}_seed${SEED}"
            task_output="${OUTPUT_ROOT}/${group}/${experiment}"
            task_log="${LOG_ROOT}/${group}/${task}.log"
            if [[ -e "$task_output" ]]; then
                echo "ERROR: task output already exists; move or remove it before rerun: ${task_output}" >&2
                exit 1
            fi
            if [[ -e "$task_log" ]]; then
                echo "ERROR: task log already exists; move or remove it before rerun: ${task_log}" >&2
                exit 1
            fi
        done
    done
    if [[ -e "$STATUS_FILE" || -e "$SUMMARY_FILE" ]]; then
        echo "ERROR: prior status/summary exists under ${OUTPUT_ROOT}; move or remove it before rerun" >&2
        exit 1
    fi
}

check_clean_destinations
printf 'group,task,exit_code,status\n' > "$STATUS_FILE"

run_task() {
    local group="$1"
    local gpu="$2"
    local source_alias="$3"
    local source="$4"
    local target_alias="$5"
    local target="$6"
    local weights="$7"
    local task="${source_alias}_${target_alias}"
    local experiment="timematch_${group}_${task}_seed${SEED}"
    local shape_args

    printf '%s\n' \
        "============================================================" \
        "START group=${group} task=${source_alias}->${target_alias} GPU=${gpu}" \
        "experiment=${experiment}" \
        "weights=${weights}/fold_${FOLD}/model.pt" \
        "============================================================"

    if [[ "$group" == "original" ]]; then
        shape_args=(
            --shape_align false
            --class_residual_phase false
        )
    else
        shape_args=(
            --shape_align true
            --shape_modes 13
            --shape_lambda 0.1
            --shape_morph_weight 1.0
            --shape_event_weight 0.0
            --shape_grid_points 64
            --shape_reference_per_class 128
            --shape_reference_seed 1
            --shape_prominence_rel 0.15
            --shape_min_distance_days 14
            --shape_fourier_period_days 365
            --shape_fourier_reg 0.001
            --shape_diag_batches 10
            --shape_loss_type local_morph
            --shape_local_window_points 16
            --shape_local_stride_points 8
            --shape_local_slope_weight 0.5
            --shape_class_balanced false
            --class_residual_phase false
        )
    fi

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
        --output_dir "$OUTPUT_ROOT/$group" \
        --tensorboard_log_dir "$TENSORBOARD_ROOT/$group" \
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
        "${shape_args[@]}"

    echo "FINISHED group=${group} task=${source_alias}->${target_alias}"
}

declare -a WAVE_PIDS=()
declare -a WAVE_TASKS=()

launch_task() {
    local group="$1"
    local gpu="$2"
    local source_alias="$3"
    local source="$4"
    local target_alias="$5"
    local target="$6"
    local weights="$7"
    local task="${source_alias}_${target_alias}"
    local log_path="${LOG_ROOT}/${group}/${task}.log"
    run_task "$group" "$gpu" "$source_alias" "$source" \
        "$target_alias" "$target" "$weights" > "$log_path" 2>&1 &
    WAVE_PIDS+=("$!")
    WAVE_TASKS+=("$task")
    printf 'LAUNCHED group=%s task=%s GPU=%s PID=%s log=%s\n' \
        "$group" "$task" "$gpu" "$!" "$log_path"
}

wait_wave() {
    local group="$1"
    local failed=0
    local index exit_code task
    for index in "${!WAVE_PIDS[@]}"; do
        task="${WAVE_TASKS[$index]}"
        if wait "${WAVE_PIDS[$index]}"; then
            exit_code=0
            echo "SUCCESS group=${group} task=${task}"
        else
            exit_code=$?
            failed=1
            echo "FAILED group=${group} task=${task} exit_code=${exit_code}" >&2
        fi
        if [[ "$exit_code" == "0" ]]; then
            printf '%s,%s,%s,SUCCESS\n' "$group" "$task" "$exit_code" >> "$STATUS_FILE"
        else
            printf '%s,%s,%s,PROCESS_FAILED(%s)\n' \
                "$group" "$task" "$exit_code" "$exit_code" >> "$STATUS_FILE"
        fi
    done
    return "$failed"
}

run_wave() {
    local group="$1"
    WAVE_PIDS=()
    WAVE_TASKS=()
    echo "============================================================"
    echo "START WAVE: ${group}"
    echo "============================================================"
    launch_task "$group" "$GPU0" AT1 "$AT1" DK1 "$DK1" "$AT1_WEIGHTS"
    launch_task "$group" "$GPU1" DK1 "$DK1" FR1 "$FR1" "$DK1_WEIGHTS"
    launch_task "$group" "$GPU2" FR1 "$FR1" FR2 "$FR2" "$FR1_WEIGHTS"
    launch_task "$group" "$GPU3" FR2 "$FR2" AT1 "$AT1" "$FR2_WEIGHTS"
    wait_wave "$group"
}

run_summary() {
    "$PYTHON_BIN" -u scripts/summarize_timematch_ablation.py \
        --log-root "$LOG_ROOT" \
        --status-file "$STATUS_FILE" \
        --output "$SUMMARY_FILE" \
        --seed "$SEED" \
        --required-groups "$@"
}

audit_original_artifacts() {
    local task experiment fold_dir artifact
    local failed=0
    for task in AT1_DK1 DK1_FR1 FR1_FR2 FR2_AT1; do
        experiment="timematch_original_${task}_seed${SEED}"
        fold_dir="${OUTPUT_ROOT}/original/${experiment}/fold_${FOLD}"
        for artifact in \
            shape_training_metrics.csv \
            shape_class_metrics.csv \
            shape_reference_manifest.json \
            shape_reference.pt; do
            if [[ -e "${fold_dir}/${artifact}" ]]; then
                echo "ERROR: Original run produced Shape artifact: ${fold_dir}/${artifact}" >&2
                printf 'original,%s,0,UNEXPECTED_SHAPE_ARTIFACT\n' "$task" >> "$STATUS_FILE"
                failed=1
                break
            fi
        done
    done
    return "$failed"
}

audit_local_sample_artifacts() {
    local task experiment fold_dir artifact
    local missing=0
    for task in AT1_DK1 DK1_FR1 FR1_FR2 FR2_AT1; do
        experiment="timematch_local_sample_${task}_seed${SEED}"
        fold_dir="${OUTPUT_ROOT}/local_sample/${experiment}/fold_${FOLD}"
        local task_missing=0
        for artifact in \
            shape_training_metrics.csv \
            shape_class_metrics.csv \
            shape_reference_manifest.json \
            shape_reference.pt; do
            if [[ ! -f "${fold_dir}/${artifact}" ]]; then
                echo "ERROR: LocalSample artifact missing: ${fold_dir}/${artifact}" >&2
                missing=1
                task_missing=1
            fi
        done
        if [[ "$task_missing" == "1" ]]; then
            printf 'local_sample,%s,0,MISSING_SHAPE_ARTIFACT\n' "$task" >> "$STATUS_FILE"
        fi
    done
    return "$missing"
}

if ! run_wave "original"; then
    echo "ERROR: Original wave failed; LocalSample wave will not start" >&2
    run_summary original || true
    exit 1
fi
if ! audit_original_artifacts; then
    run_summary original || true
    exit 1
fi
if ! run_summary original; then
    echo "ERROR: Original wave is missing an exact Test F1; LocalSample wave will not start" >&2
    exit 1
fi

if ! run_wave "local_sample"; then
    echo "ERROR: LocalSample wave failed" >&2
    run_summary original local_sample || true
    exit 1
fi
if ! audit_local_sample_artifacts; then
    run_summary original local_sample || true
    exit 1
fi
if ! run_summary original local_sample; then
    echo "ERROR: completed logs are missing an exact Test F1" >&2
    exit 1
fi

echo "============================================================"
echo "ALL ORIGINAL AND LOCAL-SAMPLE TIMEMATCH TASKS FINISHED"
echo "summary=${SUMMARY_FILE}"
echo "============================================================"
