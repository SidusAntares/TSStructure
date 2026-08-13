#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/visualization_phase_only_at1_dk1_seed1}"
CALIBRATION_CHECKPOINT="${CALIBRATION_CHECKPOINT:-$PROJECT_DIR/outputs/phase_class_center_calibration_at1_dk1_seed1/AT1_DK1_seed1/fold_0/stage2_calibration_state.pt}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$PROJECT_DIR/outputs/phase_only_at1_dk1_seed1/fold_0/stage1_best.pt}"
SOURCE_REG_BANK_CACHE="${SOURCE_REG_BANK_CACHE:-$OUTPUT_ROOT/06_sample_level_phase_validity_diagnostic/cache/source_registration_bank.pt}"
TIMEMATCH_MANIFEST="${TIMEMATCH_MANIFEST:-$OUTPUT_ROOT/04_phase_vs_scalar_pse_effect/manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/12_stage2_bootstrap_temporal_state_diagnostic}"
REGISTRATION_WORKERS="${REGISTRATION_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ORACLE_GEOMETRY_CHUNK_SIZE="${ORACLE_GEOMETRY_CHUNK_SIZE:-512}"
DP_TARGET_CHUNK_SIZE="${DP_TARGET_CHUNK_SIZE:-512}"

for required in "$CALIBRATION_CHECKPOINT" "$MODEL_CHECKPOINT"; do
  if [[ ! -f "$required" ]]; then
    echo "BOOTSTRAP12_MISSING_INPUT|path=$required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"

echo "BOOTSTRAP12_LAUNCH|output=$OUTPUT_DIR|stage2_training=false|pseudo_label_updates=false|oracle_phase_split=target-train|oracle_train_registration_cache=$OUTPUT_DIR/cache/target_train_oracle_true_class_registrations.pt"
if [[ ! -f "$SOURCE_REG_BANK_CACHE" ]]; then
  echo "BOOTSTRAP12_SOURCE_REG_BANK_MISSING|path=$SOURCE_REG_BANK_CACHE|action=deterministic_rebuild_from_source_train"
fi
if [[ ! -f "$TIMEMATCH_MANIFEST" ]]; then
  echo "BOOTSTRAP12_TIMEMATCH_MANIFEST_MISSING|path=$TIMEMATCH_MANIFEST|action=recompute_existing_unlabeled_TimeMatch_IS_scan"
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u scripts/diagnose_stage2_bootstrap_temporal_state.py \
  --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --source-registration-bank-cache "$SOURCE_REG_BANK_CACHE" \
  --timematch-manifest "$TIMEMATCH_MANIFEST" \
  --data-root "$DATA_ROOT" \
  --device cuda:0 \
  --fold 0 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --registration-workers "$REGISTRATION_WORKERS" \
  --oracle-geometry-chunk-size "$ORACLE_GEOMETRY_CHUNK_SIZE" \
  --dp-target-chunk-size "$DP_TARGET_CHUNK_SIZE" \
  --timematch-max-shift "${TIMEMATCH_MAX_SHIFT:-60}" \
  --timematch-estimation-batches "${TIMEMATCH_ESTIMATION_BATCHES:-100}" \
  --timematch-batch-size "${TIMEMATCH_BATCH_SIZE:-128}" \
  --timematch-num-pixels "${TIMEMATCH_NUM_PIXELS:-64}" \
  --output-dir "$OUTPUT_DIR"
