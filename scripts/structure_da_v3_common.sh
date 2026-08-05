#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CODE_VERSION="${CODE_VERSION:-structure_da_v3_fixed_pca_snapshots_no_fused_alignment_v2}"
OVERWRITE="${OVERWRITE:-0}"

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
    --shape_dim 128
    --canonical_grid_size 64
    --warp_num_candidates 3
    --candidate_init_warp_amplitude 0.015
    --num_shape_basis 8
    --num_phase_basis 8
    --shape_attribute_dim 8
    --time2vec_max_frequency 16.0
    --amp true
    --amp_dtype float16
    --lambda_geometry 1
    --lambda_cls 1
    --lambda_quality 1
    --lambda_source_shape 1
    --lambda_source_raw 1
    --lambda_target_semantic 1
    --lambda_quality_cls 1
    --lambda_quality_domain 1
    --lambda_q_compact 1
    --lambda_q_separate 1
    --lambda_z_proto 1
    --lambda_q_to_z_source 1
    --lambda_raw_proto 1
    --lambda_q_to_z_target 1
    --lambda_z_pull 1
    --lambda_q_to_raw_target 1
    --lambda_raw_pull 1
    --lambda_geometry_candidate 1
    --lambda_geometry_center 1
    --quality_domain_score_warmup_epochs 5
    --time_reference 0
    --time_scale 365
    --time_coordinate_mode canonical_day_of_year
    --tau_fast_init 0.05
    --tau_slow_init 0.20
    --tau_min 0.0001
    --delta_tau_min 0.0001
    --phase_gain_weight 1
    --phase_identity_weight 1
    --phase_roughness_weight 1
    --phase_unsupported_weight 1
    --phase_gain_temperature 0.05
    --phase_candidate_temperature 0.05
    --phase_min_common_support 0.05
    --phase_max_gain_ratio 1
    --phase_identity_tolerance 0.0001
    --phase_candidate_unique_tolerance 0.0001
    --phase_ambiguity_relative_tolerance 0.05
    --phase_ambiguity_absolute_tolerance 0.000001
    --structure_veto_ratio 1.05
    --structure_tie_tolerance 0.000001
    --prototype_momentum 0.99
    --radius_buffer_size 2048
    --min_radius_samples 32
    --q_inner_quantile 0.75
    --q_outer_quantile 0.95
    --feature_inner_quantile 0.75
    --prototype_min_common_support 0.05
    --q_temperature 0.10
    --z_temperature 0.10
    --trend_temperature 0.10
    --structure_temperature 0.10
    --q_separation_margin 1
    --target_q_margin 0.10
    --raw_pull_confidence 0.50
    --raw_huber_delta 0.10
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
