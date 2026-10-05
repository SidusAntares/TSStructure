#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"

V2_ROOT="${V2_ROOT:-outputs/recheck_v2clean_final_seed1}"
V2_LOG="${V2_LOG:-logs/recheck_v2clean_final_seed1}"
V2_RUN="${V2_RUN:-runs/recheck_v2clean_final_seed1}"

E_ROOT="${E_ROOT:-outputs/recheck_e_final_seed1}"
E_LOG="${E_LOG:-logs/recheck_e_final_seed1}"
E_RUN="${E_RUN:-runs/recheck_e_final_seed1}"

TASKS=(AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1)
SOURCES=(AT1 FR1 FR2 DK1)

require_phase_cli() {
  local model_help timematch_help option missing=0
  model_help="$("$PYTHON_BIN" train.py --help 2>&1)"
  timematch_help="$("$PYTHON_BIN" train.py timematch --help 2>&1)"
  for option in \
    --shape-representation \
    --shape-injection \
    --structure-shift-mode
  do
    if ! grep -Fq -- "$option" <<< "$model_help"; then
      echo "PREFLIGHT_FAILED|missing_cli=$option|action=sync_current_dec_test_FreDN" >&2
      missing=1
    fi
  done
  for option in \
    --shape-equivariance-weight \
    --shape-equivariance-max-shift
  do
    if ! grep -Fq -- "$option" <<< "$timematch_help"; then
      echo "PREFLIGHT_FAILED|missing_cli=$option|action=sync_current_dec_test_FreDN" >&2
      missing=1
    fi
  done
  if (( missing != 0 )); then
    return 1
  fi
  echo "PREFLIGHT_OK|phase_equivariance_cli=true"
}

require_last_checkpoints() {
  local label="$1" root="$2" task checkpoint missing=0
  for task in "${TASKS[@]}"; do
    checkpoint="$root/uda/${task}_seed1/fold_0/checkpoint_last.pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "MISSING_FINAL_CHECKPOINT|variant=$label|task=$task|path=$checkpoint" >&2
      missing=1
    fi
  done
  if (( missing != 0 )); then
    return 1
  fi
  echo "FINAL_CHECKPOINTS_VERIFIED|variant=$label|count=${#TASKS[@]}|policy=final_epoch"
}

require_source_checkpoints() {
  local root="$1" source checkpoint missing=0
  for source in "${SOURCES[@]}"; do
    checkpoint="$root/source/source_${source}_seed1/fold_0/model.pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "MISSING_SOURCE_CHECKPOINT|source=$source|path=$checkpoint" >&2
      missing=1
    fi
  done
  if (( missing != 0 )); then
    return 1
  fi
  echo "SOURCE_CHECKPOINTS_VERIFIED|variant=E|count=${#SOURCES[@]}"
}

mkdir -p "$V2_LOG" "$E_LOG"
require_phase_cli

echo "ROUND_START|round=V2CLEAN|root=$V2_ROOT"
if ! GPU0="$GPU0" GPU1="$GPU1" GPU2="$GPU2" GPU3="$GPU3" \
  DATA_ROOT="$DATA_ROOT" EXP_ROOT="$V2_ROOT" LOG_ROOT="$V2_LOG" RUN_ROOT="$V2_RUN" \
    bash scripts/run_structure_proto_v2clean_4tasks_4gpu_seed1.sh \
    > /dev/null 2>&1
then
  echo "ROUND_FAILED|round=V2CLEAN|logs=$V2_LOG" >&2
  exit 1
fi
require_last_checkpoints V2CLEAN "$V2_ROOT"
echo "ROUND_FINISHED|round=V2CLEAN"

echo "ROUND_START|round=E_SOURCE|root=$E_ROOT"
if ! GPU0="$GPU0" GPU1="$GPU1" GPU2="$GPU2" GPU3="$GPU3" \
  PYTHON_BIN="$PYTHON_BIN" DATA_ROOT="$DATA_ROOT" RUN_ROUND=SOURCE \
  P_ROOT="$E_ROOT" E_ROOT="$E_ROOT" \
  P_LOG="$E_LOG" E_LOG="$E_LOG" P_RUN="$E_RUN" E_RUN="$E_RUN" \
    bash scripts/run_structure_phase_equivariance_4tasks_4gpu_seed1.sh \
    > /dev/null 2>&1
then
  echo "ROUND_FAILED|round=E_SOURCE|logs=$E_LOG" >&2
  exit 1
fi
require_source_checkpoints "$E_ROOT"
echo "ROUND_FINISHED|round=E_SOURCE"

echo "ROUND_START|round=E_UDA|root=$E_ROOT"
# The existing E launcher maps its E tasks as GPU0=AT1_DK1, GPU1=FR2_DK1,
# GPU2=DK1_AT1, GPU3=FR1_FR2. Remap only device variables so the physical
# assignment remains GPU0/1/2/3 = AT1_DK1/FR1_FR2/FR2_DK1/DK1_AT1.
if ! GPU0="$GPU0" GPU1="$GPU2" GPU2="$GPU3" GPU3="$GPU1" \
  PYTHON_BIN="$PYTHON_BIN" DATA_ROOT="$DATA_ROOT" RUN_ROUND=E \
  P_ROOT="$E_ROOT" E_ROOT="$E_ROOT" \
  P_LOG="$E_LOG" E_LOG="$E_LOG" P_RUN="$E_RUN" E_RUN="$E_RUN" \
    bash scripts/run_structure_phase_equivariance_4tasks_4gpu_seed1.sh \
    > /dev/null 2>&1
then
  echo "ROUND_FAILED|round=E_UDA|logs=$E_LOG" >&2
  exit 1
fi
require_last_checkpoints E "$E_ROOT"
echo "ROUND_FINISHED|round=E_UDA"

echo "V2CLEAN_E_FINAL_FINISHED|v2_root=$V2_ROOT|e_root=$E_ROOT"
