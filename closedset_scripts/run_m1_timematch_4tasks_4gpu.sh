#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
MODE="${MODE:-M1}"
[[ "$MODE" == M0 || "$MODE" == M1 ]] || { echo 'ERROR: MODE must be M0 or M1'; exit 1; }
EXPERIMENT="${EXPERIMENT:-${MODE,,}_soft_top50_seed1}"
ROOT="outputs/${EXPERIMENT}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT/source}"
STUDENT_TARGET_AUG="${STUDENT_TARGET_AUG:-strong}"
[[ "$STUDENT_TARGET_AUG" == strong || "$STUDENT_TARGET_AUG" == weak ]] || { echo 'ERROR: STUDENT_TARGET_AUG must be strong or weak'; exit 1; }
LOG_ROOT="logs/${EXPERIMENT}"
mkdir -p "$LOG_ROOT"
exec >> "$LOG_ROOT/uda_launcher.log" 2>&1
[[ "${SEED:-1}" == 1 && "${NUM_WORKERS:-8}" == 8 ]] || {
    echo 'ERROR: matched UDA uses seed=1 and workers=8'; exit 1;
}
[[ -d "$DATA_ROOT" && ! -e "$ROOT/uda" ]] || {
    echo "ERROR: missing dataset or existing UDA output: $ROOT/uda"; exit 1;
}
read -r -a tasks <<< "${TASKS:-AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1}"
read -r -a gpus <<< "${GPUS:-3 1 2 0}"
(( ${#tasks[@]} > 0 && ${#tasks[@]} <= ${#gpus[@]} )) || { echo 'ERROR: one GPU per task required'; exit 1; }
resolve_task() {
    case "$1" in
        AT1_DK1) source=austria/33UVP/2017; target=denmark/32VNH/2017; tile=33UVP ;;
        FR1_FR2) source=france/30TXT/2017; target=france/31TCJ/2017; tile=30TXT ;;
        FR2_DK1) source=france/31TCJ/2017; target=denmark/32VNH/2017; tile=31TCJ ;;
        DK1_AT1) source=denmark/32VNH/2017; target=austria/33UVP/2017; tile=32VNH ;;
        *) echo "ERROR: unknown task $1"; return 1 ;;
    esac
}
declare -A seen_tasks=() seen_gpus=()
for index in "${!tasks[@]}"; do
    task="${tasks[$index]}"; gpu="${gpus[$index]}"
    [[ -z "${seen_tasks[$task]:-}" && -z "${seen_gpus[$gpu]:-}" ]] || { echo 'ERROR: duplicate task/GPU'; exit 1; }
    seen_tasks[$task]=1; seen_gpus[$gpu]=1
    resolve_task "$task"
    weights="$SOURCE_ROOT/${MODE,,}_source_${tile}_seed1"
    for file in train_config.json fold_0/model.pt "fold_0/test_metrics_${source//\//_}.json"; do
        [[ -f "$weights/$file" ]] || { echo "ERROR: incomplete source: $weights/$file"; exit 1; }
    done
    "$PYTHON_BIN" - "$weights" "$source" "$MODE" <<'PY'
import json
from pathlib import Path
import sys
root, source, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
config = json.loads((root / 'train_config.json').read_text())
for key, expected in dict(source=source, target=source, seed=1, num_workers=8, epochs=100,
                          method=None, model='pseltae', closed_set=True, mode=mode,
                          m1=(mode == 'M1')).items():
    if config.get(key) != expected:
        raise ValueError(f'Source config mismatch: {key}={config.get(key)!r}, expected {expected!r}')
print(f'SOURCE_VERIFIED|path={root}|mode={mode}', flush=True)
PY
done
run_task() {
    local gpu="$1" task="$2" source target tile
    resolve_task "$task"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
        --data_root "$DATA_ROOT" --source "$source" --target "$target" \
        --seed 1 --num_workers 8 --device cuda --num_folds 1 \
        --model pseltae --closed_set true --combine_spring_and_winter false --with_shift_aug false \
        --experiment_name "$task" --output_dir "$ROOT/uda" --tensorboard_log_dir "$ROOT/tensorboard" \
        timematch --weights "$SOURCE_ROOT/${MODE,,}_source_${tile}_seed1" \
        --epochs 20 --steps_per_epoch 500 --output_student true --evidence_mode off \
        --student_target_aug "$STUDENT_TARGET_AUG" \
        --run_validation false --log_target_labels false --strict_final_student true \
        > "$LOG_ROOT/${task}_uda.log" 2>&1
    "$PYTHON_BIN" - "$ROOT/uda/$task/fold_0" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch
root = Path(sys.argv[1])
packet = torch.load(root / 'model.pt', map_location='cpu', weights_only=False)
tested = json.loads((root / 'tested_checkpoint.json').read_text())
if packet.get('role') != 'student' or packet.get('completed_epochs') != 20:
    raise ValueError('Expected completed 20-epoch Student')
if tested.get('sha256') != hashlib.sha256((root / 'model.pt').read_bytes()).hexdigest():
    raise ValueError('Tested checkpoint differs from final Student')
print(f'FINAL_STUDENT_VERIFIED|path={root}|epochs=20', flush=True)
PY
}
mkdir -p "$ROOT"
mkdir "$ROOT/uda"
pids=()
for index in "${!tasks[@]}"; do
    run_task "${gpus[$index]}" "${tasks[$index]}" &
    pids+=("$!")
done
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "COMPLETED|task=${tasks[$index]}|mode=$MODE|epochs=20"
    else
        echo "FAILED|task=${tasks[$index]}|see=$LOG_ROOT/${tasks[$index]}_uda.log"
        status=1
    fi
done
exit "$status"
