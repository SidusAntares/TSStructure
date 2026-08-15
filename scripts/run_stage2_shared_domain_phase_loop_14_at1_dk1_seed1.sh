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
RAW_CANDIDATE_GEOMETRY_CACHE="${RAW_CANDIDATE_GEOMETRY_CACHE:-$OUTPUT_ROOT/13A_trainable_seed_unlabeled_evidence_separability_diagnostic/cache/raw_candidate_geometry.pt}"
ORACLE_REGISTRATION_CACHE="${ORACLE_REGISTRATION_CACHE:-$OUTPUT_ROOT/12_stage2_bootstrap_temporal_state_diagnostic/cache/target_train_oracle_true_class_registrations.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/14_shared_domain_phase_iterative_loop_diagnostic}"

for required in "$CALIBRATION_CHECKPOINT" "$MODEL_CHECKPOINT" "$RAW_CANDIDATE_GEOMETRY_CACHE"; do
  if [[ ! -f "$required" ]]; then echo "PHASE14_MISSING_INPUT|path=$required" >&2; exit 2; fi
done
mkdir -p "$OUTPUT_DIR"
AMP_ARGS=(); if [[ "${AMP:-0}" == "1" ]]; then AMP_ARGS=(--amp); fi

echo "PHASE14_LAUNCH|task=AT1_DK1|seed=1|fold=0|warmup=5x500|phase_loop=20x500|arms=NO_PHASE,STATIC_DOMAIN_PHASE,ITERATIVE_DOMAIN_PHASE|phase_refresh=epoch|target_truth_main=false|output=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u scripts/diagnose_stage2_shared_domain_phase_loop_14.py \
  --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --source-registration-bank-cache "$SOURCE_REG_BANK_CACHE" \
  --raw-candidate-geometry-cache "$RAW_CANDIDATE_GEOMETRY_CACHE" \
  --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR" --device cuda:0 --fold 0 \
  --batch-size "${BATCH_SIZE:-64}" --num-workers "${NUM_WORKERS:-4}" \
  --registration-workers "${REGISTRATION_WORKERS:-4}" --dp-target-chunk-size "${DP_TARGET_CHUNK_SIZE:-512}" \
  --warmup-epochs 5 --phase-epochs 20 --steps-per-epoch 500 \
  --pseudo-threshold "${PSEUDO_THRESHOLD:-0.9}" --lambda-target "${LAMBDA_TARGET:-1.0}" \
  --focal-gamma "${FOCAL_GAMMA:-1.0}" --ema-decay "${EMA_DECAY:-0.9999}" \
  --lr "${LR:-1e-4}" --weight-decay "${WEIGHT_DECAY:-1e-4}" "${AMP_ARGS[@]}"

EVAL_ARGS=()
if [[ -f "$ORACLE_REGISTRATION_CACHE" ]]; then EVAL_ARGS=(--oracle-registration-cache "$ORACLE_REGISTRATION_CACHE"); fi
python -u scripts/evaluate_stage2_shared_domain_phase_14_oracle.py \
  --experiment-dir "$OUTPUT_DIR" --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --data-root "$DATA_ROOT" "${EVAL_ARGS[@]}"
