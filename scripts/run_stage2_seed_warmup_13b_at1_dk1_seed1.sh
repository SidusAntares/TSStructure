#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
CALIBRATION_CHECKPOINT="${CALIBRATION_CHECKPOINT:-$PROJECT_DIR/outputs/phase_class_center_calibration_at1_dk1_seed1/AT1_DK1_seed1/fold_0/stage2_calibration_state.pt}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$PROJECT_DIR/outputs/phase_only_at1_dk1_seed1/fold_0/stage1_best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/visualization_phase_only_at1_dk1_seed1}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/13B_fixed_seed_short_semantic_adaptation_diagnostic}"

if [[ -z "${SEED_MANIFEST:-}" ]]; then
  echo "SEED13B_SEED_MANIFEST_REQUIRED|reason=T0_is_upstream_input_not_designed_inside_13B" >&2
  exit 2
fi
for required in "$CALIBRATION_CHECKPOINT" "$MODEL_CHECKPOINT" "$SEED_MANIFEST"; do
  if [[ ! -f "$required" ]]; then
    echo "SEED13B_MISSING_INPUT|path=$required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
AMP_ARGS=()
if [[ "${AMP:-0}" == "1" ]]; then AMP_ARGS=(--amp); fi
echo "SEED13B_LAUNCH|task=AT1_DK1|seed=1|fold=0|T0=fixed_external_manifest|student=PSE+LTAE+TimeEncoder+Classifier|teacher=EMA_observation_only|phase_update=false|seed_refresh=false|relabel=false|groups=source_only,main|checkpoints=0,1,3,5|output=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u scripts/diagnose_stage2_seed_warmup_13b.py \
  --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --seed-manifest "$SEED_MANIFEST" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --fold 0 \
  --batch-size "${BATCH_SIZE:-64}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --epochs 5 \
  --steps-per-epoch 500 \
  --checkpoint-epochs 0,1,3,5 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --lambda-target 1.0 \
  --ema-decay 0.99 \
  "${AMP_ARGS[@]}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u scripts/evaluate_stage2_seed_warmup_13b_oracle.py \
  --experiment-dir "$OUTPUT_DIR" \
  --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --data-root "$DATA_ROOT"
