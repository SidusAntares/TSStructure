#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/visualization_phase_only_at1_dk1_seed1}"
STAGE_A_CACHE="${STAGE_A_CACHE:-$OUTPUT_ROOT/06_sample_level_phase_validity_diagnostic/cache/stage_a_t_only_registrations.pt}"
AUDIT07_SAMPLE_CSV="${AUDIT07_SAMPLE_CSV:-$OUTPUT_ROOT/07_oracle_true_class_gamma_effectiveness_and_legality_audit/oracle_gamma_sample_level.csv}"
EXPERIMENT10_MANIFEST="${EXPERIMENT10_MANIFEST:-$OUTPUT_ROOT/10_class_balanced_shared_domain_phase_leave_one_class_out/00_manifest.json}"
EXPERIMENT10_PHASE_NPZ="${EXPERIMENT10_PHASE_NPZ:-$OUTPUT_ROOT/10_class_balanced_shared_domain_phase_leave_one_class_out/01_shared_phase_curves.npz}"
EXPERIMENT10_PER_CLASS_CSV="${EXPERIMENT10_PER_CLASS_CSV:-$OUTPUT_ROOT/10_class_balanced_shared_domain_phase_leave_one_class_out/06_per_class_classification_comparison.csv}"
CALIBRATION_CHECKPOINT="${CALIBRATION_CHECKPOINT:-$PROJECT_DIR/outputs/phase_class_center_calibration_at1_dk1_seed1/AT1_DK1_seed1/fold_0/stage2_calibration_state.pt}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$PROJECT_DIR/outputs/phase_only_at1_dk1_seed1/fold_0/stage1_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/11_shared_phase_residual_class_structure_and_generalization_diagnostic}"

for required in "$STAGE_A_CACHE" "$AUDIT07_SAMPLE_CSV" "$EXPERIMENT10_MANIFEST" "$EXPERIMENT10_PHASE_NPZ" "$EXPERIMENT10_PER_CLASS_CSV" "$CALIBRATION_CHECKPOINT" "$MODEL_CHECKPOINT"; do
  if [[ ! -f "$required" ]]; then
    echo "RESIDUAL_PHASE_11_MISSING_INPUT|path=$required|registration_fallback=false" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"

echo "RESIDUAL_PHASE_11_LAUNCH|stage_a_cache=$STAGE_A_CACHE|experiment10_phase=$EXPERIMENT10_PHASE_NPZ|output=$OUTPUT_DIR|registration_calls=0|training_updates=false|clustering=false|cv_folds=5"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u scripts/diagnose_shared_phase_residual_class_structure_and_generalization.py \
  --stage-a-cache "$STAGE_A_CACHE" \
  --audit07-sample-csv "$AUDIT07_SAMPLE_CSV" \
  --experiment10-manifest "$EXPERIMENT10_MANIFEST" \
  --experiment10-phase-npz "$EXPERIMENT10_PHASE_NPZ" \
  --experiment10-per-class-csv "$EXPERIMENT10_PER_CLASS_CSV" \
  --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --device cuda:0 \
  --fold 0 \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --expected-valid-count 10634 \
  --cv-folds 5 \
  --cv-seed 20260812 \
  --reconstruction-tolerance 1e-7 \
  --output-dir "$OUTPUT_DIR"
