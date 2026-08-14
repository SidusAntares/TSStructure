#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/visualization_phase_only_at1_dk1_seed1}"
CALIBRATION_CHECKPOINT="${CALIBRATION_CHECKPOINT:-$PROJECT_DIR/outputs/phase_class_center_calibration_at1_dk1_seed1/AT1_DK1_seed1/fold_0/stage2_calibration_state.pt}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$PROJECT_DIR/outputs/phase_only_at1_dk1_seed1/fold_0/stage1_best.pt}"
EXPERIMENT13A1_DIR="${EXPERIMENT13A1_DIR:-$OUTPUT_ROOT/13A_trainable_seed_unlabeled_evidence_separability_diagnostic}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/13A2_missing_semantic_structure_and_evidence_complementarity_diagnostic}"

for required in "$CALIBRATION_CHECKPOINT" "$MODEL_CHECKPOINT" "$EXPERIMENT13A1_DIR/00_manifest.json" "$EXPERIMENT13A1_DIR/01_unlabeled_sample_observables.csv"; do
  if [[ ! -f "$required" ]]; then
    echo "SEED13A2_MISSING_INPUT|path=$required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"

echo "SEED13A2_LAUNCH|task=AT1_DK1|seed=1|fold=0|split=target-train|delta_boot=identity|candidate=13A1_raw_top1|knn_k=5,10,20,50|source_geometry_crossfit=5|target_labels_in_observable=false|training_updates=false|trainable_gate=false|output=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u scripts/diagnose_stage2_missing_semantic_structure_13a2.py \
  --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --experiment13a1-dir "$EXPERIMENT13A1_DIR" \
  --data-root "$DATA_ROOT" \
  --device cuda:0 \
  --knn-device cuda:0 \
  --fold 0 \
  --batch-size "${BATCH_SIZE:-64}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --knn-chunk-size "${KNN_CHUNK_SIZE:-512}" \
  --source-crossfit-folds 5 \
  --registration-workers "${REGISTRATION_WORKERS:-4}" \
  --geometry-chunk-size "${GEOMETRY_CHUNK_SIZE:-512}" \
  --dp-target-chunk-size "${DP_TARGET_CHUNK_SIZE:-512}" \
  --analysis-seed "${ANALYSIS_SEED:-20260814}" \
  --bootstrap-reps "${BOOTSTRAP_REPS:-500}" \
  --pca-max-points "${PCA_MAX_POINTS:-6000}" \
  --output-dir "$OUTPUT_DIR"
