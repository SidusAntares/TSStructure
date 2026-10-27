#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/timematch_historical_rerun_seed1/source}"
TASKS="${TASKS:-AT1_DK1 FR1_FR2 FR2_DK1 DK1_AT1}"
GPUS="${GPUS:-3 1 2 0}"
EXPERIMENT="${EXPERIMENT:-head_evidence_top30_gate05_seed1}"
OUTPUT_ROOT="outputs/${EXPERIMENT}"
LOG_ROOT="logs/${EXPERIMENT}"
mkdir -p "$LOG_ROOT"
exec >> "$LOG_ROOT/launcher.log" 2>&1

[[ "${SEED:-1}" == 1 && "${NUM_WORKERS:-8}" == 8 ]] || {
    echo 'ERROR: this comparison uses SEED=1 and NUM_WORKERS=8'; exit 1;
}
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: dataset root missing: $DATA_ROOT"; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || {
    echo "ERROR: output already exists: $OUTPUT_ROOT; choose a new EXPERIMENT"; exit 1;
}
read -r -a tasks <<< "${TASKS//,/ }"
read -r -a gpus <<< "${GPUS//,/ }"
(( ${#tasks[@]} > 0 && ${#tasks[@]} <= ${#gpus[@]} )) || {
    echo 'ERROR: provide one GPU per selected task'; exit 1;
}

resolve_task() {
    case "$1" in
        AT1_DK1) source="austria/33UVP/2017"; target="denmark/32VNH/2017"; tile="33UVP" ;;
        FR1_FR2) source="france/30TXT/2017"; target="france/31TCJ/2017"; tile="30TXT" ;;
        FR2_DK1) source="france/31TCJ/2017"; target="denmark/32VNH/2017"; tile="31TCJ" ;;
        DK1_AT1) source="denmark/32VNH/2017"; target="austria/33UVP/2017"; tile="32VNH" ;;
        *) echo "ERROR: unknown task $1"; return 1 ;;
    esac
}

# Validate every retrained Source before starting any A/B training.
declare -A seen_tasks=() seen_gpus=()
for index in "${!tasks[@]}"; do
    task="${tasks[$index]}"; gpu="${gpus[$index]}"
    [[ -z "${seen_tasks[$task]:-}" && -z "${seen_gpus[$gpu]:-}" ]] || {
        echo 'ERROR: duplicate task or GPU'; exit 1;
    }
    seen_tasks[$task]=1; seen_gpus[$gpu]=1
    resolve_task "$task"
    source_weights="$SOURCE_ROOT/closedset_source_${tile}_seed1"
    for file in fold_0/model.pt train_config.json closed_set_protocol.json \
                "fold_0/test_metrics_${source//\//_}.json"; do
        [[ -f "$source_weights/$file" ]] || {
            echo "ERROR: incomplete retrained source: $source_weights/$file"; exit 1;
        }
    done
    "$PYTHON_BIN" - "$source_weights" "$source" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch
from models.stclassifier import PseLTae
root, source = Path(sys.argv[1]), sys.argv[2]
config = json.loads((root / 'train_config.json').read_text())
for key, expected in dict(source=source, target=source, model='pseltae', seed=1,
                          num_workers=8, epochs=100, closed_set=True,
                          combine_spring_and_winter=False, with_extra=False,
                          input_dim=10, method=None).items():
    if config.get(key) != expected:
        raise ValueError(f'Source mismatch: {root}: {key}={config.get(key)!r}, expected {expected!r}')
protocol = json.loads((root / 'closed_set_protocol.json').read_text())
if protocol.get('source') != source or not protocol.get('closed_set'):
    raise ValueError(f'Source protocol mismatch: {root}')
path = root / 'fold_0/model.pt'
packet = torch.load(path, map_location='cpu', weights_only=False)
PseLTae(num_classes=len(protocol['classes']), with_extra=False).load_state_dict(packet['state_dict'])
print(f'SOURCE_VERIFIED|path={path}|sha256={hashlib.sha256(path.read_bytes()).hexdigest()}', flush=True)
PY
done

run_task() {
    local gpu="$1" task="$2" source target tile version mode
    resolve_task "$task"
    local source_weights="$SOURCE_ROOT/closedset_source_${tile}_seed1"
    for version in A B; do
        if [[ "$version" == A ]]; then mode=positive; else mode=positive_rank; fi
        echo "UDA_START|gpu=$gpu|task=$task|version=$version|structure=top30|gate=0.5"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
            --data_root "$DATA_ROOT" --source "$source" --target "$target" \
            --seed 1 --num_workers 8 --device cuda --num_folds 1 \
            --closed_set true --combine_spring_and_winter false --with_shift_aug false \
            --experiment_name "${task}_${version}" --output_dir "$OUTPUT_ROOT" \
            --tensorboard_log_dir "$OUTPUT_ROOT/tensorboard" \
            timematch --weights "$source_weights" --epochs 20 --steps_per_epoch 500 \
            --output_student true --evidence_mode "$mode" \
            --evidence_structure top30 --evidence_top_fraction 0.30 \
            --evidence_intervention gate --evidence_gate 0.5 \
            --evidence_positive_weight 0.10 --evidence_rank_weight 0.10 \
            --evidence_warmup_steps 100 \
            > "$LOG_ROOT/${task}_${version}.log" 2>&1
    done
}

mkdir -p "$OUTPUT_ROOT"
pids=()
for index in "${!tasks[@]}"; do
    run_task "${gpus[$index]}" "${tasks[$index]}" &
    pids+=("$!")
done
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "COMPLETED|task=${tasks[$index]}|versions=A,B"
    else
        echo "FAILED|task=${tasks[$index]}|see=$LOG_ROOT/${tasks[$index]}_A.log,$LOG_ROOT/${tasks[$index]}_B.log"
        status=1
    fi
done
exit "$status"
