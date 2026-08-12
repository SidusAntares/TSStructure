#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
CALIBRATION_CHECKPOINT="${CALIBRATION_CHECKPOINT:-$PROJECT_DIR/outputs/phase_class_center_calibration_at1_dk1_seed1/AT1_DK1_seed1/fold_0/stage2_calibration_state.pt}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$PROJECT_DIR/outputs/phase_only_at1_dk1_seed1/fold_0/stage1_best.pt}"
STAGE_A_CACHE="${STAGE_A_CACHE:-$PROJECT_DIR/outputs/visualization_phase_only_at1_dk1_seed1/06_sample_level_phase_validity_diagnostic/cache/stage_a_t_only_registrations.pt}"
TIMEMATCH_MANIFEST="${TIMEMATCH_MANIFEST:-$PROJECT_DIR/outputs/visualization_phase_only_at1_dk1_seed1/04_phase_vs_scalar_pse_effect/manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/visualization_phase_only_at1_dk1_seed1/07_oracle_true_class_gamma_effectiveness_and_legality_audit}"

if [[ ! -f "$STAGE_A_CACHE" ]]; then
  echo "ORACLE_GAMMA_07_MISSING_STAGE_A_CACHE|path=$STAGE_A_CACHE|action=finish_06_stage_a_first" >&2
  exit 2
fi
if [[ ! -f "$TIMEMATCH_MANIFEST" ]]; then
  echo "ORACLE_GAMMA_07_MISSING_TIMEMATCH_MANIFEST|path=$TIMEMATCH_MANIFEST|action=reuse_or_run_04_unlabeled_shift_estimation" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
echo "ORACLE_GAMMA_07_START|stage_a_cache=$STAGE_A_CACHE|timematch_manifest=$TIMEMATCH_MANIFEST|output=$OUTPUT_DIR|exact_dp_calls=0"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u scripts/audit_oracle_true_class_gamma_effectiveness.py \
  --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --stage-a-cache "$STAGE_A_CACHE" \
  --timematch-manifest "$TIMEMATCH_MANIFEST" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --dpi "${DPI:-160}"
