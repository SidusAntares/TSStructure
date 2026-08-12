#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/visualization_phase_only_at1_dk1_seed1}"
STAGE_A_CACHE="${STAGE_A_CACHE:-$OUTPUT_ROOT/06_sample_level_phase_validity_diagnostic/cache/stage_a_t_only_registrations.pt}"
AUDIT07_SAMPLE_CSV="${AUDIT07_SAMPLE_CSV:-$OUTPUT_ROOT/07_oracle_true_class_gamma_effectiveness_and_legality_audit/oracle_gamma_sample_level.csv}"
AUDIT08_SAMPLE_CSV="${AUDIT08_SAMPLE_CSV:-$OUTPUT_ROOT/08_oracle_true_class_gamma_harm_mechanism_diagnostic/oracle_gamma_harm_sample_level.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/09_oracle_true_class_phase_population_structure_diagnostic}"
MDS_RANDOM_SIZE="${MDS_RANDOM_SIZE:-1000}"
MDS_EQUAL_PER_CLASS="${MDS_EQUAL_PER_CLASS:-100}"
PAIR_SAMPLES="${PAIR_SAMPLES:-100000}"
KNN_BLOCK_SIZE="${KNN_BLOCK_SIZE:-512}"
RANDOM_SEED="${RANDOM_SEED:-20260812}"

for required in "$STAGE_A_CACHE" "$AUDIT07_SAMPLE_CSV" "$AUDIT08_SAMPLE_CSV"; do
  if [[ ! -f "$required" ]]; then
    echo "ORACLE_PHASE_POPULATION_09_MISSING_INPUT|path=$required|registration_fallback=false" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"

echo "ORACLE_PHASE_POPULATION_09_LAUNCH|stage_a_cache=$STAGE_A_CACHE|audit07=$AUDIT07_SAMPLE_CSV|audit08=$AUDIT08_SAMPLE_CSV|output=$OUTPUT_DIR|registration_calls=0|clustering=false"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u scripts/diagnose_oracle_true_class_phase_population_structure.py \
  --stage-a-cache "$STAGE_A_CACHE" \
  --audit07-sample-csv "$AUDIT07_SAMPLE_CSV" \
  --audit08-sample-csv "$AUDIT08_SAMPLE_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --expected-valid-count 10634 \
  --random-seed "$RANDOM_SEED" \
  --mds-random-size "$MDS_RANDOM_SIZE" \
  --mds-equal-per-class "$MDS_EQUAL_PER_CLASS" \
  --pair-samples "$PAIR_SAMPLES" \
  --knn-k 5,10,20 \
  --knn-device cuda:0 \
  --knn-block-size "$KNN_BLOCK_SIZE"
