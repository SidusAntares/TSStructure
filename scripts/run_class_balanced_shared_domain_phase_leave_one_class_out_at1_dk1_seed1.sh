#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/visualization_phase_only_at1_dk1_seed1}"
STAGE_A_CACHE="${STAGE_A_CACHE:-$OUTPUT_ROOT/06_sample_level_phase_validity_diagnostic/cache/stage_a_t_only_registrations.pt}"
AUDIT07_SAMPLE_CSV="${AUDIT07_SAMPLE_CSV:-$OUTPUT_ROOT/07_oracle_true_class_gamma_effectiveness_and_legality_audit/oracle_gamma_sample_level.csv}"
CALIBRATION_CHECKPOINT="${CALIBRATION_CHECKPOINT:-$PROJECT_DIR/outputs/phase_class_center_calibration_at1_dk1_seed1/AT1_DK1_seed1/fold_0/stage2_calibration_state.pt}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$PROJECT_DIR/outputs/phase_only_at1_dk1_seed1/fold_0/stage1_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/10_class_balanced_shared_domain_phase_leave_one_class_out}"

for required in "$STAGE_A_CACHE" "$AUDIT07_SAMPLE_CSV" "$CALIBRATION_CHECKPOINT" "$MODEL_CHECKPOINT"; do
  if [[ ! -f "$required" ]]; then
    echo "SHARED_PHASE_10_MISSING_INPUT|path=$required|registration_fallback=false" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"

echo "SHARED_PHASE_10_LAUNCH|stage_a_cache=$STAGE_A_CACHE|audit07=$AUDIT07_SAMPLE_CSV|output=$OUTPUT_DIR|registration_calls=0|clustering=false|training_updates=false"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u scripts/diagnose_class_balanced_shared_domain_phase_leave_one_class_out.py \
  --stage-a-cache "$STAGE_A_CACHE" \
  --audit07-sample-csv "$AUDIT07_SAMPLE_CSV" \
  --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --device cuda:0 \
  --fold 0 \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --expected-valid-count 10634 \
  --output-dir "$OUTPUT_DIR"
