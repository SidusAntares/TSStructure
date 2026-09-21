#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
SMOKE="${SMOKE:-0}"
TRAIN_MISSING_SOURCE="${TRAIN_MISSING_SOURCE:-1}"
SEED=1

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"
AT1_WEIGHTS="${AT1_WEIGHTS:-outputs/pseltae_AT1_source_seed1}"
DK1_WEIGHTS="${DK1_WEIGHTS:-outputs/pseltae_DK1_source_seed1}"
FR1_WEIGHTS="${FR1_WEIGHTS:-outputs/pseltae_FR1_source_seed1}"
FR2_WEIGHTS="${FR2_WEIGHTS:-outputs/pseltae_FR2_source_seed1}"

if [[ "$SMOKE" == "1" ]]; then
    RUN_NAME="reconshift13_raw_smoke_at1_dk1_seed1"
    DA_EPOCHS=1
    STEPS_PER_EPOCH=2
    SAMPLE_SIZE=2
else
    RUN_NAME="reconshift13_raw_4tasks_seed1"
    DA_EPOCHS=20
    STEPS_PER_EPOCH=500
    SAMPLE_SIZE=100
fi
OUTPUT_ROOT="outputs/${RUN_NAME}"
TENSORBOARD_ROOT="runs/${RUN_NAME}"
LOG_ROOT="logs/${RUN_NAME}"
export PYTHONUNBUFFERED=1

[[ "$SMOKE" == "0" || "$SMOKE" == "1" ]] || { echo "ERROR: SMOKE must be 0 or 1" >&2; exit 2; }
[[ "$TRAIN_MISSING_SOURCE" == "0" || "$TRAIN_MISSING_SOURCE" == "1" ]] || { echo "ERROR: TRAIN_MISSING_SOURCE must be 0 or 1" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: DATA_ROOT not found: $DATA_ROOT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT" "$TENSORBOARD_ROOT" "$LOG_ROOT"

check_weights() {
    local alias="$1" weights="$2" expected_source="$3"
    local checkpoint="${weights}/fold_0/model.pt"
    local config="${weights}/train_config.json"
    [[ -f "$checkpoint" ]] || { echo "ERROR: ${alias} Raw PseLTae checkpoint not found: $checkpoint" >&2; return 1; }
    [[ -f "$config" ]] || { echo "ERROR: ${alias} train_config.json not found: $config" >&2; return 1; }
    "$PYTHON_BIN" - "$config" "$expected_source" <<'PY'
import json
import sys

path, expected = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    cfg = json.load(handle)
if cfg.get("model") != "pseltae":
    raise SystemExit(f"ERROR: source checkpoint must use model=pseltae, found {cfg.get('model')}")
if cfg.get("source") != expected or cfg.get("target") != expected:
    raise SystemExit("ERROR: source checkpoint must be source-only with source=target")
if cfg.get("seed") != 1 or cfg.get("num_folds") != 1:
    raise SystemExit("ERROR: source checkpoint must use seed=1 and num_folds=1")
PY
}

ensure_weights() {
    local gpu="$1" alias="$2" source="$3" weights="$4"
    if [[ -f "${weights}/fold_0/model.pt" && -f "${weights}/train_config.json" ]]; then
        check_weights "$alias" "$weights" "$source"
        echo "[REUSE] ${alias} Original Raw PseLTae checkpoint: ${weights}/fold_0/model.pt"
        return
    fi
    if [[ "$TRAIN_MISSING_SOURCE" != "1" ]]; then
        echo "ERROR: ${alias} source checkpoint is missing and TRAIN_MISSING_SOURCE=0" >&2
        return 1
    fi
    if [[ -e "${weights}/fold_0/model.pt" || -e "${weights}/train_config.json" ]]; then
        echo "ERROR: ${alias} source checkpoint is incomplete; refusing to overwrite it" >&2
        return 1
    fi
    local source_parent source_experiment
    source_parent="$(dirname "$weights")"
    source_experiment="$(basename "$weights")"
    echo "[SOURCE PRETRAIN] ${alias} -> ${alias}, Raw PseLTae"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
        -e "$source_experiment" \
        --data_root "$DATA_ROOT" \
        --source "$source" \
        --target "$source" \
        --model pseltae \
        --seed "$SEED" \
        --num_folds 1 \
        --seq_length 30 \
        --num_pixels 64 \
        --batch_size 128 \
        --epochs 100 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --focal_loss_gamma 1.0 \
        --device cuda \
        --closed_set true \
        --combine_spring_and_winter false \
        --with_shift_aug false \
        --progress_bar off \
        --output_dir "$source_parent" \
        --tensorboard_log_dir "${TENSORBOARD_ROOT}/source_${alias}"
    check_weights "$alias" "$weights" "$source"
}

run_worker() {
    local gpu="$1" source_alias="$2" source="$3" target_alias="$4" target="$5" weights="$6"
    local experiment="reconshift13_raw_${source_alias}_${target_alias}_seed${SEED}"
    local checkpoint="${weights}/fold_0/model.pt"
    ensure_weights "$gpu" "$source_alias" "$source" "$weights"
    printf '%s\n' \
        "============================================================" \
        "[GPU${gpu}] RECONSHIFT13 -> RAW TIMEMATCH" \
        "${source_alias} -> ${target_alias}" \
        "MODEL: pseltae (Raw)" \
        "SOURCE CHECKPOINT: ${checkpoint}" \
        "SHIFT VIEW: FourierRecon13" \
        "============================================================"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
        -e "$experiment" \
        --data_root "$DATA_ROOT" \
        --source "$source" \
        --target "$target" \
        --model pseltae \
        --seed "$SEED" \
        --num_folds 1 \
        --seq_length 30 \
        --num_pixels 64 \
        --batch_size 128 \
        --weight_decay 0.0001 \
        --focal_loss_gamma 1.0 \
        --device cuda \
        --closed_set true \
        --combine_spring_and_winter false \
        --with_shift_aug false \
        --progress_bar off \
        --output_dir "$OUTPUT_ROOT" \
        --tensorboard_log_dir "$TENSORBOARD_ROOT" \
        timematch \
        --weights "$weights" \
        --epochs "$DA_EPOCHS" \
        --steps_per_epoch "$STEPS_PER_EPOCH" \
        --lr 0.0001 \
        --pseudo_threshold 0.9 \
        --ema_decay 0.9999 \
        --trade_off 2.0 \
        --estimate_shift true \
        --balance_source true \
        --use_focal_loss true \
        --shift_source true \
        --sample_size "$SAMPLE_SIZE" \
        --max_temporal_shift 60 \
        --domain_specific_bn true \
        --shift_estimator AM \
        --run_validation \
        --output_student true \
        --class_residual_phase false \
        --shift-estimation-view fourier_recon \
        --shift-fourier-num-modes 13 \
        --shift-fourier-reg 0.001 \
        --shift-fourier-period-days 365.0 \
        --shift-fourier-solver dense_direct
    echo "[FINISHED] ${source_alias} -> ${target_alias}"
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    run_worker "$@"
    exit $?
fi

launch() {
    local gpu="$1" source_alias="$2" source="$3" target_alias="$4" target="$5" weights="$6"
    local log_path="${LOG_ROOT}/${source_alias}_${target_alias}.log"
    nohup bash "$SCRIPT_PATH" --worker "$gpu" "$source_alias" "$source" "$target_alias" "$target" "$weights" \
        >"$log_path" 2>&1 &
    printf 'LAUNCHED GPU=%s PID=%s TASK=%s->%s LOG=%s\n' \
        "$gpu" "$!" "$source_alias" "$target_alias" "$log_path"
}

if [[ "$SMOKE" == "1" ]]; then
    launch "$GPU0" AT1 "$AT1" DK1 "$DK1" "$AT1_WEIGHTS"
    echo "ReconShift13 Raw AT1->DK1 smoke launched."
else
    launch "$GPU0" AT1 "$AT1" DK1 "$DK1" "$AT1_WEIGHTS"
    launch "$GPU1" DK1 "$DK1" FR1 "$FR1" "$DK1_WEIGHTS"
    launch "$GPU2" FR1 "$FR1" FR2 "$FR2" "$FR1_WEIGHTS"
    launch "$GPU3" FR2 "$FR2" AT1 "$AT1" "$FR2_WEIGHTS"
    echo "Four ReconShift13 -> Raw TimeMatch tasks launched."
fi
