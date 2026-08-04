#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs}"
CONDA_ENV="${CONDA_ENV:-timematch}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CODE_VERSION="${CODE_VERSION:-a7751523794b48813ae9f294303889eed62ea2e7}"
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

activate_environment() {
    if [[ "${PYTHON_BIN}" != "python" ]]; then
        require_command "${PYTHON_BIN}"
        return
    fi
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
        require_command "${PYTHON_BIN}"
        return
    fi

    local conda_sh=""
    for candidate in \
        "${CONDA_EXE:-}" \
        "${HOME:-}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME:-}/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"
    do
        if [[ "${candidate}" == */bin/conda ]]; then
            candidate="${candidate%/bin/conda}/etc/profile.d/conda.sh"
        fi
        if [[ -n "${candidate}" && -f "${candidate}" ]]; then
            conda_sh="${candidate}"
            break
        fi
    done
    if [[ -z "${conda_sh}" ]]; then
        echo "Conda environment ${CONDA_ENV} is not active and conda.sh was not found." >&2
        echo "Activate it manually or set PYTHON_BIN to the environment Python." >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${conda_sh}"
    conda activate "${CONDA_ENV}"
    require_command "${PYTHON_BIN}"
}

print_run_header() {
    local run_output_directory="$1"
    local run_log_directory="$2"
    local source_domain="$3"
    local target_domain="$4"
    local seed="$5"
    local physical_gpu="$6"
    local batch_size="128"
    local environment_file="${run_log_directory}/environment.txt"

    {
        echo "CODE_VERSION=${CODE_VERSION}"
        echo "HOSTNAME=$(hostname)"
        echo "DATE=$(date --iso-8601=seconds)"
        echo "PYTHON_PATH=$(command -v "${PYTHON_BIN}")"
        echo "PHYSICAL_GPU=${physical_gpu}"
        echo "SOURCE=${source_domain}"
        echo "TARGET=${target_domain}"
        echo "SEED=${seed}"
        echo "BATCH_SIZE=${batch_size}"
        echo "OUTPUT_DIRECTORY=${run_output_directory}"
        echo "LOG_DIRECTORY=${run_log_directory}"
        "${PYTHON_BIN}" -c 'import sys, torch; print("PYTHON_VERSION=" + sys.version.replace("\n", " ")); print("PYTORCH_VERSION=" + torch.__version__); print("CUDA_AVAILABLE=" + str(torch.cuda.is_available())); print("CUDA_DEVICE_COUNT=" + str(torch.cuda.device_count()))'
    } > "${environment_file}"
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
    local log_directory="$2"
    _validate_child_directory "${output_directory}" "${OUTPUT_ROOT}"
    _validate_child_directory "${log_directory}" "${LOG_ROOT}"

    if [[ -e "${output_directory}" ]]; then
        if [[ "${OVERWRITE}" != "1" ]]; then
            echo "Output directory already exists: ${output_directory}" >&2
            return 1
        fi
        rm -rf -- "${output_directory}"
    fi
    if [[ -d "${log_directory}" ]]; then
        local entry base
        for entry in "${log_directory}"/* "${log_directory}"/.[!.]* "${log_directory}"/..?*; do
            [[ -e "${entry}" ]] || continue
            base="$(basename "${entry}")"
            if [[ "${base}" == "nohup.log" || "${base}" == "launcher.pid" ]]; then
                continue
            fi
            if [[ "${OVERWRITE}" != "1" ]]; then
                echo "Log directory contains an earlier run: ${log_directory}" >&2
                return 1
            fi
            rm -rf -- "${entry}"
        done
    elif [[ -e "${log_directory}" ]]; then
        echo "Log path is not a directory: ${log_directory}" >&2
        return 1
    fi
    mkdir -p "${output_directory}" "${log_directory}"
}

make_run_directories() {
    local run_output_directory="$1"
    local run_log_directory="$2"
    _validate_child_directory "${run_output_directory}" "${OUTPUT_ROOT}"
    _validate_child_directory "${run_log_directory}" "${LOG_ROOT}"
    for directory in "${run_output_directory}" "${run_log_directory}"; do
        if [[ -e "${directory}" ]]; then
            if [[ "${OVERWRITE}" != "1" ]]; then
                echo "Run directory already exists: ${directory}" >&2
                return 1
            fi
            rm -rf -- "${directory}"
        fi
    done
    mkdir -p "${run_output_directory}/fold_0" \
        "${run_output_directory}/tensorboard" "${run_log_directory}"
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
    --domain_hidden_dim 128
    --grl_warmup_fraction 0.2
    --amp true
    --amp_dtype float16
    --lambda_geometry 1
    --lambda_cls 1
    --lambda_quality 1
    --lambda_source_shape 1
    --lambda_source_raw 1
    --lambda_global_domain 1
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
    local fifth_argument="$5"
    local sixth_argument="$6"
    shift 6
    local run_output_directory run_log_directory
    local extra_args
    if [[ "${fifth_argument}" =~ ^[0-9]+$ ]]; then
        run_output_directory="${sixth_argument}"
        run_log_directory="${sixth_argument}"
        extra_args=(--epochs "${fifth_argument}" "$@")
    else
        run_output_directory="${fifth_argument}"
        run_log_directory="${sixth_argument}"
        extra_args=("$@")
    fi

    require_file "${PROJECT_ROOT}/train.py"
    if [[ -n "${DATA_ROOT}" ]]; then
        require_directory "${DATA_ROOT}"
    fi
    require_directory "${run_output_directory}"
    require_directory "${run_log_directory}"
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

    printf '%q ' "${CMD[@]}" > "${run_log_directory}/command.txt"
    printf '\n' >> "${run_log_directory}/command.txt"
    print_run_header "${run_output_directory}" "${run_log_directory}" \
        "${source_domain}" "${target_domain}" "${seed}" "${physical_gpu}"
    CUDA_VISIBLE_DEVICES="${physical_gpu}" "${CMD[@]}" \
        > "${run_log_directory}/train.log" \
        2> "${run_log_directory}/stderr.log" &
    local training_pid=$!
    printf '%s\n' "${training_pid}" > "${run_log_directory}/train.pid"
    echo "TASK_PROCESS|gpu=${physical_gpu}|pid=${training_pid}"
    wait "${training_pid}"
}
