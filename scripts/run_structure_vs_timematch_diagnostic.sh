#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/structure_vs_timematch_diagnostic}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"

declare -A DOMAIN=(
  [AT1]="austria/33UVP/2017"
  [DK1]="denmark/32VNH/2017"
  [FR1]="france/30TXT/2017"
  [FR2]="france/31TCJ/2017"
)
TASKS=(AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1)
SEEDS=(1 2)
METHODS=(original_timematch structure_shapelet)

rm -f "$OUTPUT_ROOT/summary.csv" "$OUTPUT_ROOT/per_class.csv" "$OUTPUT_ROOT/report.md"
mkdir -p "$OUTPUT_ROOT/figures"

resolve_paths() {
  local method="$1" source="$2" target="$3" seed="$4"
  local override_prefix
  override_prefix="${method^^}_${source}_${target}_SEED${seed}"
  override_prefix="${override_prefix//-/_}"
  local start_var="${override_prefix}_START"
  local end_var="${override_prefix}_END"
  local log_var="${override_prefix}_LOG"
  if [[ "$method" == "structure_shapelet" ]]; then
    local root="outputs/structure_proto_4tasks_seed${seed}"
    START_PATH="${!start_var:-$root/source/source_${source}_seed${seed}/fold_0/model.pt}"
    END_PATH="${!end_var:-$root/uda/${source}_${target}_seed${seed}/fold_0/checkpoint_last.pt}"
    LOG_PATH="${!log_var:-logs/fredn/structure_proto_4tasks_seed${seed}/${source}_${target}.log}"
  else
    START_PATH="${!start_var:-outputs/pseltae_${source}_source_seed${seed}/fold_0/model.pt}"
    END_PATH="${!end_var:-outputs/timematch_original_4tasks_seed${seed}/uda/${source}_${target}_seed${seed}/fold_0/checkpoint_last.pt}"
    LOG_PATH="${!log_var:-logs/fredn/timematch_original_4tasks_seed${seed}/${source}_${target}.log}"
  fi
}

found=0
missing=0
failed=0
for seed in "${SEEDS[@]}"; do
  for task in "${TASKS[@]}"; do
    source="${task%%_*}"
    target="${task##*_}"
    for method in "${METHODS[@]}"; do
      resolve_paths "$method" "$source" "$target" "$seed"
      absent=()
      [[ -f "$START_PATH" ]] || absent+=("START=$START_PATH")
      [[ -f "$END_PATH" ]] || absent+=("END=$END_PATH")
      [[ -f "$LOG_PATH" ]] || absent+=("LOG=$LOG_PATH")
      if (( ${#absent[@]} )); then
        printf 'MISSING|task=%s|seed=%s|method=%s|%s\n' "$task" "$seed" "$method" "$(IFS='|'; echo "${absent[*]}")"
        missing=$((missing + 1))
        continue
      fi
      found=$((found + 1))
      echo "DIAGNOSTIC_START|task=$task|seed=$seed|method=$method"
      if ! "$PYTHON_BIN" -u scripts/analyze_structure_vs_timematch.py run \
        --task "$task" --seed "$seed" --method "$method" \
        --source "${DOMAIN[$source]}" --target "${DOMAIN[$target]}" \
        --data-root "$DATA_ROOT" --start-checkpoint "$START_PATH" \
        --end-checkpoint "$END_PATH" --source-log "$LOG_PATH" \
        --output-root "$OUTPUT_ROOT" --device "$DEVICE"; then
        echo "UNAVAILABLE|task=$task|seed=$seed|method=$method|reason=checkpoint_role_or_runtime_validation_failed"
        failed=$((failed + 1))
      fi
    done
  done
done

"$PYTHON_BIN" -u scripts/analyze_structure_vs_timematch.py finalize --output-root "$OUTPUT_ROOT"
echo "DIAGNOSTIC_DISCOVERY|found=$found|missing=$missing|failed=$failed|expected=16"
(( failed == 0 ))
