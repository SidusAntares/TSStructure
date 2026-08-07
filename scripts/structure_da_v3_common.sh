#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CODE_VERSION="${CODE_VERSION:-structure_da_v3_snapshot_schema3_no_fused_alignment_v4}"
OVERWRITE="${OVERWRITE:-0}"
STAGE2_CONFIG="${STAGE2_CONFIG:-}"

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Required command not found: $1" >&2
        return 1
    }
}

require_file() {
    [[ -f "$1" ]] || {
        echo "Required file not found: $1" >&2
        return 1
    }
}

require_directory() {
    [[ -d "$1" ]] || {
        echo "Required directory not found: $1" >&2
        return 1
    }
}

_validate_child_directory() {
    local child="$1"
    local parent="$2"
    case "${child}" in
        "${parent}"/*) ;;
        *)
            echo "Directory must be a child of ${parent}: ${child}" >&2
            return 1
            ;;
    esac
}

prepare_run_group() {
    local output_directory="$1"
    local group_log_root="$2"
    local train_log_directory="${group_log_root}/train_logs"
    local snapshot_directory="${group_log_root}/snapshots"
    _validate_child_directory "${output_directory}" "${OUTPUT_ROOT}"
    _validate_child_directory "${group_log_root}" "${LOG_ROOT}"

    if [[ -e "${output_directory}" ]]; then
        echo "Output directory already exists: ${output_directory}" >&2
        return 1
    fi
    if [[ -d "${train_log_directory}" ]]; then
        local entry base
        for entry in "${train_log_directory}"/* "${train_log_directory}"/.[!.]* "${train_log_directory}"/..?*; do
            [[ -e "${entry}" ]] || continue
            base="$(basename "${entry}")"
            if [[ "${base}" == "nohup.log" || "${base}" == "launcher.pid" ]]; then
                continue
            fi
            echo "Training log directory contains an earlier run: ${train_log_directory}" >&2
            return 1
        done
    elif [[ -e "${train_log_directory}" ]]; then
        echo "Training log path is not a directory: ${train_log_directory}" >&2
        return 1
    fi
    if [[ -d "${snapshot_directory}" ]] && \
        find "${snapshot_directory}" -mindepth 1 -print -quit | grep -q .
    then
        echo "Snapshot directory contains an earlier run: ${snapshot_directory}" >&2
        return 1
    elif [[ -e "${snapshot_directory}" && ! -d "${snapshot_directory}" ]]; then
        echo "Snapshot path is not a directory: ${snapshot_directory}" >&2
        return 1
    fi
    mkdir -p "${output_directory}" "${train_log_directory}" "${snapshot_directory}"
}

make_run_output_directory() {
    local run_output_directory="$1"
    _validate_child_directory "${run_output_directory}" "${OUTPUT_ROOT}"
    if [[ -e "${run_output_directory}" ]]; then
        echo "Run output directory already exists: ${run_output_directory}" >&2
        return 1
    fi
    mkdir -p "${run_output_directory}/fold_0" "${run_output_directory}/tensorboard"
}

make_run_directory() {
    local run_directory="$1"
    case "${run_directory}" in
        "${OUTPUT_ROOT}"/*) ;;
        *)
            echo "Run directory must be a child of OUTPUT_ROOT: ${run_directory}" >&2
            return 1
            ;;
    esac
    if [[ -e "${run_directory}" ]]; then
        if [[ "${OVERWRITE}" != "1" ]]; then
            echo "Run directory already exists; set OVERWRITE=1 to replace it: ${run_directory}" >&2
            return 1
        fi
        rm -rf -- "${run_directory}"
    fi
    mkdir -p "${run_directory}/fold_0" "${run_directory}/tensorboard"
}

V3_COMMON_ARGS=(
    --closed_set true
    --balance-source
    --combine_spring_and_winter false
    --num_folds 1
    --val_ratio 0.1
    --test_ratio 0.2
    --sample_pixels_val true
    --batch_size 128
    --eval_batch_size 128
    --num_pixels 64
    --num_workers "${NUM_WORKERS}"
    --lr 0.001
    --weight_decay 0.0001
    --input_dim 10
    --with_extra false
    --canonical_grid_size 64
    --roughness_grid_size 256
    --trend_num_basis 12
    --structure_num_basis 12
    --trend_smoothing 0.01
    --structure_smoothing 0.001
    --n_head 16
    --d_k 8
    --d_model 256
    --ltae_mlp 256,128
    --dropout 0.2
    --classifier_hidden 64,32
    --time2vec_max_frequency 16.0
    --amp true
    --amp_dtype float16
    --time_reference 0
    --time_scale 365
    --time_coordinate_mode canonical_day_of_year
    --tau_fast_init 0.05
    --tau_slow_init 0.20
    --tau_min 0.0001
    --delta_tau_min 0.0001
    --progress_bar auto
    --log_step 10
)

run_training() {
    local source_domain="$1"
    local target_domain="$2"
    local seed="$3"
    local physical_gpu="$4"
    local run_output_directory="$5"
    local task_log_file="$6"
    shift 6
    local extra_args=("$@")

    require_file "${PROJECT_ROOT}/train.py"
    if [[ -n "${DATA_ROOT}" ]]; then
        require_directory "${DATA_ROOT}"
    fi
    require_directory "${run_output_directory}"
    require_directory "$(dirname "${task_log_file}")"
    local stage2_config_path=""
    if [[ -n "${STAGE2_CONFIG}" ]]; then
        require_file "${STAGE2_CONFIG}"
        stage2_config_path="$(cd "$(dirname "${STAGE2_CONFIG}")" && pwd)/$(basename "${STAGE2_CONFIG}")"
    fi
    cd "${PROJECT_ROOT}"

    CMD=(
        "${PYTHON_BIN}" -u train.py
        --source "${source_domain}"
        --target "${target_domain}"
        --seed "${seed}"
        --device cuda:0
        --output_dir "${run_output_directory}"
        --tensorboard_log_dir "${run_output_directory}/tensorboard/events"
        "${V3_COMMON_ARGS[@]}"
        "${extra_args[@]}"
    )
    if [[ -n "${DATA_ROOT}" ]]; then
        CMD+=(--data_root "${DATA_ROOT}")
    fi
    if [[ -n "${stage2_config_path}" ]]; then
        CMD+=(--stage2_config "${stage2_config_path}")
    fi

    {
        echo "RUN_START|task=$(basename "${task_log_file}" .log)|seed=${seed}|gpu=${physical_gpu}|time=$(date --iso-8601=seconds)|code_version=${CODE_VERSION}|output=${run_output_directory}"
        printf 'RUN_COMMAND|'
        printf '%q ' "${CMD[@]}"
        printf '\n'
    } > "${task_log_file}"
    CUDA_VISIBLE_DEVICES="${physical_gpu}" "${CMD[@]}" \
        >> "${task_log_file}" 2>&1 &
    LAST_TRAINING_PID=$!
    echo "TASK_PROCESS|gpu=${physical_gpu}|pid=${LAST_TRAINING_PID}"
    if wait "${LAST_TRAINING_PID}"; then
        LAST_TRAINING_EXIT_CODE=0
    else
        LAST_TRAINING_EXIT_CODE=$?
    fi
    echo "RUN_END|task=$(basename "${task_log_file}" .log)|seed=${seed}|gpu=${physical_gpu}|exit_code=${LAST_TRAINING_EXIT_CODE}|time=$(date --iso-8601=seconds)" >> "${task_log_file}"
    return "${LAST_TRAINING_EXIT_CODE}"
}
