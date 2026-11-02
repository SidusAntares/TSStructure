#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
EXPERIMENT="${EXPERIMENT:-timematch_segments_smoke_seed1}"
ROOT="outputs/$EXPERIMENT"
LOG_ROOT="logs/$EXPERIMENT"
tasks=(FR1_FR2 FR2_DK1 DK1_AT1 AT1_DK1)
read -r -a gpu_ids <<< "${GPUS:-0 1 2 3}"
(( ${#gpu_ids[@]} == 4 )) || { echo 'ERROR: requires four GPU IDs'; exit 1; }
seen=' '
for gpu in "${gpu_ids[@]}"; do
    [[ "$gpu" =~ ^(0|[1-9][0-9]*)$ && "$seen" != *" $gpu "* ]] || {
        echo "ERROR: invalid or duplicate GPU: $gpu"; exit 1;
    }
    seen+="$gpu "
done
[[ ! -e "$LOG_ROOT/length_control_launcher.log" ]] || {
    echo 'ERROR: length-control launcher log already exists'; exit 1;
}
for task in "${tasks[@]}"; do
    [[ -f "$ROOT/inshape/$task/fold_0/smoke_checkpoint.pt" ]] || {
        echo "ERROR: missing epoch5 smoke checkpoint for $task"; exit 1;
    }
    [[ ! -e "$ROOT/diagnostics/length_control/$task" && ! -e "$LOG_ROOT/${task}_length_control.log" ]] || {
        echo "ERROR: length-control diagnostic already exists for $task"; exit 1;
    }
done
mkdir -p "$LOG_ROOT"
exec >> "$LOG_ROOT/length_control_launcher.log" 2>&1

run_task() {
    local task="$1" gpu="$2"
    echo "DIAGNOSTIC_START|task=$task|gpu=$gpu"
    if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u -m analysis.segment_length_diagnostic \
        --experiment_root "$ROOT" --data_root "$DATA_ROOT" --task "$task" --device cuda \
        > "$LOG_ROOT/${task}_length_control.log" 2>&1; then
        echo "DIAGNOSTIC_COMPLETE|task=$task|gpu=$gpu"
    else
        local code="$?"
        echo "DIAGNOSTIC_FAILED|task=$task|gpu=$gpu|status=$code"
        return "$code"
    fi
}
pids=()
for slot in "${!tasks[@]}"; do
    run_task "${tasks[$slot]}" "${gpu_ids[$slot]}" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
exit "$status"
