#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=structure_da_v3_common.sh
source "${SCRIPT_DIR}/structure_da_v3_common.sh"

failed=0
fail_check() {
    echo "CHECK_FAILED|$1" >&2
    failed=1
}

require_directory "${PROJECT_ROOT}" || fail_check "project_root"
require_file "${PROJECT_ROOT}/train.py" || fail_check "train_py"
activate_environment || fail_check "python_environment"

if [[ "${failed}" -eq 0 ]]; then
    (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" -c 'import torch, numpy, sklearn, zarr, tensorboard; import train, dataset, evaluation; print("DEPENDENCY_IMPORTS=PASS")') \
        || fail_check "python_dependencies"
    (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" train.py --help >/dev/null) \
        || fail_check "train_help"
    "${PYTHON_BIN}" -c 'import torch; print("CUDA_AVAILABLE=" + str(torch.cuda.is_available())); print("CUDA_DEVICE_COUNT=" + str(torch.cuda.device_count()))' \
        || fail_check "cuda_query"
fi

if [[ -n "${DATA_ROOT}" ]]; then
    TIME_MATCH_DOMAINS=(
        "austria/33UVP/2017"
        "denmark/32VNH/2017"
        "france/30TXT/2017"
        "france/31TCJ/2017"
    )
    for domain in "${TIME_MATCH_DOMAINS[@]}"; do
        domain_directory="${DATA_ROOT}/${domain}"
        if [[ ! -d "${domain_directory}" ]]; then
            fail_check "missing_domain:${domain}"
            continue
        fi
        if ! find "${domain_directory}" -type f -readable -print -quit | grep -q .; then
            fail_check "no_readable_data:${domain}"
        fi
    done
else
    echo "DATA_ROOT=TRAIN_PY_DEFAULT"
fi

if ! mkdir -p "${OUTPUT_ROOT}/.environment_check"; then
    fail_check "output_create"
elif ! printf 'offline write check\n' > "${OUTPUT_ROOT}/.environment_check/write_test.txt"; then
    fail_check "output_write"
else
    rm -f -- "${OUTPUT_ROOT}/.environment_check/write_test.txt"
    rmdir "${OUTPUT_ROOT}/.environment_check" 2>/dev/null || true
fi

df -P "${OUTPUT_ROOT}" || fail_check "disk_space"
echo "NETWORK_REQUIRED=false"
if [[ "${failed}" -ne 0 ]]; then
    echo "SERVER_ENV_CHECK=FAIL"
    exit 1
fi
echo "SERVER_ENV_CHECK=PASS"
