#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=structure_da_v3_common.sh
source "${SCRIPT_DIR}/structure_da_v3_common.sh"

# Diagnostic A/B is intentionally fixed to one Stage-1 checkpoint and never
# performs Stage-2 optimization.  It is a scientific check of (a) which gate
# blocks Phase evidence when roughness is disabled and (b) whether the V6
# high-recall proposal misses hypotheses that an exhaustive all-class scan
# would retain on the exact same first 64 target evidence samples.
DATA_ROOT=""
STAGE2_CONFIG="${STAGE2_CONFIG:-${PROJECT_ROOT}/configs/stage2_pilot_v1.json}"
SOURCE_DOMAIN="${SOURCE_DOMAIN:-austria/33UVP/2017}"
TARGET_DOMAIN="${TARGET_DOMAIN:-denmark/32VNH/2017}"
SEED="${SEED:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
REGISTRATION_WORKERS="${REGISTRATION_WORKERS:-4}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${PROJECT_ROOT}/outputs/v3_pilot_round1_seed1/AT1_DK1_seed1/fold_0/stage1_best.pt}"
RUN_GROUP="${RUN_GROUP:-stage2_phase_diagnostic_ab_at1_dk1_seed1}"
ROUGHNESS_DIAGNOSTIC_MAX="${ROUGHNESS_DIAGNOSTIC_MAX:-1000000000000}"

require_file "${STAGE2_CONFIG}"
require_file "${STAGE1_CHECKPOINT}"

GROUP_OUTPUT_DIRECTORY="${OUTPUT_ROOT}/${RUN_GROUP}"
GROUP_LOG_ROOT="${LOG_ROOT}/${RUN_GROUP}"
GROUP_TRAIN_LOG_DIRECTORY="${GROUP_LOG_ROOT}/train_logs"
prepare_run_group "${GROUP_OUTPUT_DIRECTORY}" "${GROUP_LOG_ROOT}"

run_case() {
    local case_name="$1"
    local initial_samples="$2"
    local max_samples="$3"
    local classifier_topk="$4"
    local identity_topk="$5"

    local out="${GROUP_OUTPUT_DIRECTORY}/${case_name}"
    local log="${GROUP_TRAIN_LOG_DIRECTORY}/${case_name}.log"
    make_run_output_directory "${out}"

    echo "PHASE_DIAGNOSTIC_START|case=${case_name}|gpu=${CUDA_DEVICE}|log=${log}"
    run_training \
        "${SOURCE_DOMAIN}" \
        "${TARGET_DOMAIN}" \
        "${SEED}" \
        "${CUDA_DEVICE}" \
        "${out}" \
        "${log}" \
        --stage2_only \
        --stage1_checkpoint "${STAGE1_CHECKPOINT}" \
        --stage2_diagnostic_only \
        --stage2_epochs 1 \
        --stage2_block_epochs 1 \
        --stage2_phase_evidence_initial_samples "${initial_samples}" \
        --stage2_phase_evidence_max_samples "${max_samples}" \
        --stage2_registration_workers "${REGISTRATION_WORKERS}" \
        --stage2_phase_proposal_classifier_topk "${classifier_topk}" \
        --stage2_phase_proposal_identity_topk "${identity_topk}" \
        --stage2_registration_max_roughness "${ROUGHNESS_DIAGNOSTIC_MAX}" \
        --feature_snapshot_interval 0 \
        --progress_bar off
}

# A: actual V6 proposal, progressive 64 -> 128.  Roughness alone is bypassed.
run_case "A_proposal_64_128_no_roughness" 64 128 2 2

# B: exhaustive all-class exact DP with the same 64 -> 128 cache/budgets.
# Matching the maximum cache size is essential: both runs then use the exact
# same first 64 deterministic evidence IDs. Very large top-k is capped
# internally by ready class count, so this remains generic if class count changes.
run_case "B_allclass_64_128_no_roughness" 64 128 999 999

"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_stage2_phase_diagnostics.py" \
    --proposal-json "${GROUP_OUTPUT_DIRECTORY}/A_proposal_64_128_no_roughness/fold_0/shape_diagnostics_000_initial.json" \
    --all-class-json "${GROUP_OUTPUT_DIRECTORY}/B_allclass_64_128_no_roughness/fold_0/shape_diagnostics_000_initial.json" \
    --proposal-log "${GROUP_TRAIN_LOG_DIRECTORY}/A_proposal_64_128_no_roughness.log" \
    --all-class-log "${GROUP_TRAIN_LOG_DIRECTORY}/B_allclass_64_128_no_roughness.log" \
    --compare-budget 64 \
    --output "${GROUP_LOG_ROOT}/comparison.json"

echo "PHASE_DIAGNOSTIC_COMPLETE|comparison=${GROUP_LOG_ROOT}/comparison.json"
