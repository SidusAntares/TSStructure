#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs}"
LOGS_ROOT="${LOGS_ROOT:-logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/v2_v2clean_failure_audit_seed1}"
AUDIT_LOG_ROOT="${AUDIT_LOG_ROOT:-logs/v2_v2clean_failure_audit_seed1}"
DRY_RUN="${DRY_RUN:-0}"

declare -a pids=()

checkpoint_path() {
  local method="$1" task="$2" stage="$3"
  local experiment_root source
  if [[ "$method" == "v2" ]]; then
    experiment_root="structure_proto_v2_4tasks_seed1"
  else
    experiment_root="structure_proto_v2clean_4tasks_seed1"
  fi
  source="${task%%_*}"
  if [[ "$stage" == "source" ]]; then
    printf '%s/%s/source/source_%s_seed1/fold_0/model.pt' \
      "$CHECKPOINT_ROOT" "$experiment_root" "$source"
  else
    printf '%s/%s/uda/%s_seed1/fold_0/model.pt' \
      "$CHECKPOINT_ROOT" "$experiment_root" "$task"
  fi
}

log_path() {
  local method="$1" task="$2" experiment_root
  if [[ "$method" == "v2" ]]; then
    experiment_root="structure_proto_v2_4tasks_seed1"
  else
    experiment_root="structure_proto_v2clean_4tasks_seed1"
  fi
  printf '%s/%s/%s.log' "$LOGS_ROOT" "$experiment_root" "$task"
}

preflight() {
  local missing=0 method task path
  for method in v2 v2clean; do
    for task in FR1_FR2 DK1_AT1; do
      for path in \
        "$(checkpoint_path "$method" "$task" source)" \
        "$(checkpoint_path "$method" "$task" uda)" \
        "$(log_path "$method" "$task")"
      do
        if [[ ! -f "$path" ]]; then
          echo "BLOCKER|missing=$path" >&2
          missing=1
        fi
      done
    done
  done
  (( missing == 0 ))
}

run_job() {
  local gpu="$1" method="$2" task="$3"
  echo "FAILURE_AUDIT_PLAN|gpu=$gpu|method=$method|task=$task"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  mkdir -p "$AUDIT_LOG_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u \
    analysis/v2_v2clean_failure_audit.py \
    --method "$method" --task "$task" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --logs-root "$LOGS_ROOT" --output-root "$OUTPUT_ROOT" \
    --data-root "$DATA_ROOT" --device cuda \
    > "$AUDIT_LOG_ROOT/${method}_${task}.log" 2>&1 &
  pids+=("$!")
}

if [[ "$DRY_RUN" != "1" ]]; then
  preflight
fi

run_job "$GPU0" v2 FR1_FR2
run_job "$GPU1" v2 DK1_AT1
run_job "$GPU2" v2clean FR1_FR2
run_job "$GPU3" v2clean DK1_AT1

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if (( status != 0 )); then
  echo "ERROR: one or more failure-audit jobs failed; inspect $AUDIT_LOG_ROOT" >&2
  exit "$status"
fi

"$PYTHON_BIN" -u analysis/v2_v2clean_failure_audit.py \
  --combine --output-root "$OUTPUT_ROOT"
