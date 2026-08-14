#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
CALIBRATION_CHECKPOINT="${CALIBRATION_CHECKPOINT:-$PROJECT_DIR/outputs/phase_class_center_calibration_at1_dk1_seed1/AT1_DK1_seed1/fold_0/stage2_calibration_state.pt}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$PROJECT_DIR/outputs/phase_only_at1_dk1_seed1/fold_0/stage1_best.pt}"
STRUCTURE_DIAGNOSTIC_DIR="${STRUCTURE_DIAGNOSTIC_DIR:-$PROJECT_DIR/outputs/visualization_phase_only_at1_dk1_seed1/13A2_missing_semantic_structure_and_evidence_complementarity_diagnostic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/visualization_phase_only_at1_dk1_seed1}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/13B_bootstrap_strategy_short_adaptation_diagnostic}"

for required in "$CALIBRATION_CHECKPOINT" "$MODEL_CHECKPOINT" "$STRUCTURE_DIAGNOSTIC_DIR/01_unlabeled_sample_observables.csv" "$STRUCTURE_DIAGNOSTIC_DIR/02_unlabeled_dense_vectors.npz" "$STRUCTURE_DIAGNOSTIC_DIR/cache/source_frozen_ltae_features.pt"; do
  if [[ ! -e "$required" ]]; then
    echo "SEED13B_MISSING_INPUT|path=$required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
AMP_ARGS=()
if [[ "${AMP:-0}" == "1" ]]; then AMP_ARGS=(--amp); fi

echo "SEED13B_LAUNCH|task=AT1_DK1|seed=1|fold=0|groups=CTRL_SOURCE,PL_TIMEMATCH_CONF,PL_DAPL_BOOT,PL_IPL_BOOT,PL_TFDA_NN,PL_CONF_GEOM,CTRL_ORACLE|transpl=false|phase_update=false|pseudo_refresh=false|checkpoints=0,1,3,5|output=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u scripts/diagnose_stage2_seed_warmup_13b.py \
  --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --structure-diagnostic-dir "$STRUCTURE_DIAGNOSTIC_DIR" \
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
  --pseudo-threshold 0.9 \
  --conformity-percentile-threshold 0.95 \
  --conformity-crossfit-folds 5 \
  --bootstrap-knn-k 20 \
  --ipl-support-threshold 0.5 \
  --geometry-conflict-percentile 0.95 \
  "${AMP_ARGS[@]}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u scripts/evaluate_stage2_seed_warmup_13b_oracle.py \
  --experiment-dir "$OUTPUT_DIR" \
  --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --data-root "$DATA_ROOT"
