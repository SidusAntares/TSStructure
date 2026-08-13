#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/visualization_phase_only_at1_dk1_seed1}"
CALIBRATION_CHECKPOINT="${CALIBRATION_CHECKPOINT:-$PROJECT_DIR/outputs/phase_class_center_calibration_at1_dk1_seed1/AT1_DK1_seed1/fold_0/stage2_calibration_state.pt}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$PROJECT_DIR/outputs/phase_only_at1_dk1_seed1/fold_0/stage1_best.pt}"
EXPERIMENT12_SUMMARY="${EXPERIMENT12_SUMMARY:-$OUTPUT_ROOT/12_stage2_bootstrap_temporal_state_diagnostic/28_bootstrap_diagnostic_summary.json}"
SOURCE_REG_BANK_CACHE="${SOURCE_REG_BANK_CACHE:-$OUTPUT_ROOT/06_sample_level_phase_validity_diagnostic/cache/source_registration_bank.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/13A_trainable_seed_unlabeled_evidence_separability_diagnostic}"

# Experiment 13A deliberately does not infer delta_boot from target metrics.
# The theory/design window must explicitly pass the formal unlabeled bootstrap
# state chosen after experiment 12: identity or timematch_scalar.
if [[ -z "${BOOTSTRAP_STATE:-}" ]]; then
  echo "SEED13A_BOOTSTRAP_STATE_REQUIRED|allowed=identity,timematch_scalar|reason=experiment12_does_not_auto_select" >&2
  exit 2
fi
if [[ "$BOOTSTRAP_STATE" != "identity" && "$BOOTSTRAP_STATE" != "timematch_scalar" ]]; then
  echo "SEED13A_INVALID_BOOTSTRAP_STATE|value=$BOOTSTRAP_STATE|allowed=identity,timematch_scalar" >&2
  exit 2
fi

for required in "$CALIBRATION_CHECKPOINT" "$MODEL_CHECKPOINT" "$EXPERIMENT12_SUMMARY"; do
  if [[ ! -f "$required" ]]; then
    echo "SEED13A_MISSING_INPUT|path=$required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"

echo "SEED13A_LAUNCH|bootstrap_state=$BOOTSTRAP_STATE|split=target-train|raw_candidate=classifier_top1|labels_in_observable=false|geometry_relabels=false|training_updates=false|trainable_gate=false|output=$OUTPUT_DIR"
if [[ ! -f "$SOURCE_REG_BANK_CACHE" ]]; then
  echo "SEED13A_SOURCE_REG_BANK_MISSING|path=$SOURCE_REG_BANK_CACHE|action=deterministic_rebuild_from_source_train"
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u scripts/diagnose_stage2_trainable_seed_evidence_13a.py \
  --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --experiment12-summary "$EXPERIMENT12_SUMMARY" \
  --bootstrap-state "$BOOTSTRAP_STATE" \
  --source-registration-bank-cache "$SOURCE_REG_BANK_CACHE" \
  --data-root "$DATA_ROOT" \
  --device cuda:0 \
  --fold 0 \
  --batch-size "${BATCH_SIZE:-64}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --registration-workers "${REGISTRATION_WORKERS:-4}" \
  --geometry-chunk-size "${GEOMETRY_CHUNK_SIZE:-512}" \
  --dp-target-chunk-size "${DP_TARGET_CHUNK_SIZE:-512}" \
  --analysis-seed "${ANALYSIS_SEED:-20260813}" \
  --bootstrap-reps "${BOOTSTRAP_REPS:-500}" \
  --output-dir "$OUTPUT_DIR"
