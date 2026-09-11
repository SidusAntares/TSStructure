#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"; GPU3="${GPU3:-3}"
SEED=1
AT1="austria/33UVP/2017"; DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"; FR2="france/31TCJ/2017"
AT1_WEIGHTS="${AT1_WEIGHTS:-outputs/pseltae_AT1_source_seed1}"
DK1_WEIGHTS="${DK1_WEIGHTS:-outputs/pseltae_DK1_source_seed1}"
FR1_WEIGHTS="${FR1_WEIGHTS:-outputs/pseltae_FR1_source_seed1}"
FR2_WEIGHTS="${FR2_WEIGHTS:-outputs/pseltae_FR2_source_seed1}"
RUN_NAME="reconshift13_classres20_sourceonly_4tasks_seed1"
OUTPUT_ROOT="outputs/${RUN_NAME}"; LOG_ROOT="logs/${RUN_NAME}"; TB_ROOT="runs/${RUN_NAME}"
export PYTHONUNBUFFERED=1

check_weights() {
    local alias="$1" weights="$2" expected_source="$3"
    [[ -f "${weights}/fold_0/model.pt" ]] || { echo "ERROR: ${alias} checkpoint missing: ${weights}/fold_0/model.pt" >&2; return 1; }
    [[ -f "${weights}/train_config.json" ]] || { echo "ERROR: ${alias} config missing: ${weights}/train_config.json" >&2; return 1; }
    "$PYTHON_BIN" - "$weights/train_config.json" "$expected_source" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
expected = sys.argv[2]
if config.get("model") != "pseltae":
    raise SystemExit("ERROR: checkpoint model must be pseltae")
if config.get("source") != expected or config.get("target") != expected:
    raise SystemExit("ERROR: checkpoint must be source-only with source=target")
if config.get("seed") != 1 or config.get("num_folds") != 1:
    raise SystemExit("ERROR: checkpoint must use seed=1 and num_folds=1")
PY
}

run_worker() {
    local gpu="$1" source_alias="$2" source="$3" target_alias="$4" target="$5" weights="$6"
    check_weights "$source_alias" "$weights" "$source"
    local experiment="reconshift13_classres20_sourceonly_${source_alias}_${target_alias}_seed${SEED}"
    echo "[GPU${gpu}] ${source_alias} -> ${target_alias}; source-only frozen class residual shift"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
        -e "$experiment" --data_root "$DATA_ROOT" --source "$source" --target "$target" \
        --model pseltae --seed "$SEED" --num_folds 1 --seq_length 30 --num_pixels 64 \
        --batch_size 128 --weight_decay 0.0001 --focal_loss_gamma 1.0 --device cuda \
        --closed_set true --combine_spring_and_winter false --with_shift_aug false \
        --progress_bar off --output_dir "$OUTPUT_ROOT" --tensorboard_log_dir "$TB_ROOT" \
        timematch --weights "$weights" --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
        --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 --estimate_shift true \
        --balance_source true --use_focal_loss true --shift_source true --sample_size 100 \
        --max_temporal_shift 60 --domain_specific_bn true --shift_estimator AM \
        --run_validation --output_student true --shape_align false --class_residual_phase false \
        --shift-estimation-view fourier_recon --shift-fourier-num-modes 13 \
        --shift-fourier-reg 0.001 --shift-fourier-period-days 365.0 \
        --shift-fourier-solver dense_direct --source-class-residual-shift true \
        --source-class-residual-max-days 20 --source-class-residual-min-samples 32 \
        --source-class-residual-max-samples 128 --source-class-residual-min-gain 0.005
    echo "[FINISHED] ${source_alias} -> ${target_alias}"
}

if [[ "${1:-}" == "--worker" ]]; then shift; run_worker "$@"; exit $?; fi
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$TB_ROOT"
check_weights AT1 "$AT1_WEIGHTS" "$AT1"
check_weights DK1 "$DK1_WEIGHTS" "$DK1"
check_weights FR1 "$FR1_WEIGHTS" "$FR1"
check_weights FR2 "$FR2_WEIGHTS" "$FR2"
launch() {
    local gpu="$1" sa="$2" source="$3" ta="$4" target="$5" weights="$6"
    nohup bash "$SCRIPT_PATH" --worker "$gpu" "$sa" "$source" "$ta" "$target" "$weights" \
        >"${LOG_ROOT}/${sa}_${ta}.log" 2>&1 &
    echo "LAUNCHED GPU=${gpu} PID=$! TASK=${sa}->${ta}"
}
launch "$GPU0" AT1 "$AT1" DK1 "$DK1" "$AT1_WEIGHTS"
launch "$GPU1" DK1 "$DK1" FR1 "$FR1" "$DK1_WEIGHTS"
launch "$GPU2" FR1 "$FR1" FR2 "$FR2" "$FR1_WEIGHTS"
launch "$GPU3" FR2 "$FR2" AT1 "$AT1" "$FR2_WEIGHTS"
