#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/structure_da_v3}"
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
    local run_directory="$1"
    local source_domain="$2"
    local target_domain="$3"
    local seed="$4"
    local physical_gpu="$5"
    local epochs="$6"
    local batch_size="128"
    local environment_file="${run_directory}/environment.txt"

    {
        echo "CODE_VERSION=${CODE_VERSION}"
        echo "HOSTNAME=$(hostname)"
        echo "DATE=$(date --iso-8601=seconds)"
        echo "PYTHON_PATH=$(command -v "${PYTHON_BIN}")"
        echo "PHYSICAL_GPU=${physical_gpu}"
        echo "SOURCE=${source_domain}"
        echo "TARGET=${target_domain}"
        echo "SEED=${seed}"
        echo "EPOCHS=${epochs}"
        echo "BATCH_SIZE=${batch_size}"
        echo "OUTPUT_DIRECTORY=${run_directory}"
        "${PYTHON_BIN}" -c 'import sys, torch; print("PYTHON_VERSION=" + sys.version.replace("\n", " ")); print("PYTORCH_VERSION=" + torch.__version__); print("CUDA_AVAILABLE=" + str(torch.cuda.is_available())); print("CUDA_DEVICE_COUNT=" + str(torch.cuda.device_count()))'
    } | tee "${environment_file}"
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
    --progress_bar off
    --log_step 10
)

run_training() {
    local source_domain="$1"
    local target_domain="$2"
    local seed="$3"
    local physical_gpu="$4"
    local epochs="$5"
    local run_directory="$6"
    shift 6
    local extra_args=("$@")

    require_file "${PROJECT_ROOT}/train.py"
    require_directory "${DATA_ROOT}"
    cd "${PROJECT_ROOT}"

    CMD=(
        "${PYTHON_BIN}" -u train.py
        --data_root "${DATA_ROOT}"
        --source "${source_domain}"
        --target "${target_domain}"
        --seed "${seed}"
        --device cuda:0
        --epochs "${epochs}"
        --output_dir "${run_directory}"
        --tensorboard_log_dir "${run_directory}/tensorboard/events"
        "${V3_COMMON_ARGS[@]}"
        "${extra_args[@]}"
    )

    printf '%q ' "${CMD[@]}" | tee "${run_directory}/command.txt"
    printf '\n' | tee -a "${run_directory}/command.txt"
    print_run_header "${run_directory}" "${source_domain}" "${target_domain}" \
        "${seed}" "${physical_gpu}" "${epochs}"
    CUDA_VISIBLE_DEVICES="${physical_gpu}" "${CMD[@]}" \
        > >(tee "${run_directory}/train.log") \
        2> >(tee "${run_directory}/stderr.log" >&2)
}
