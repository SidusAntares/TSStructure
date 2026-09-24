#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs/structure_proto_4tasks_seed2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pse_alignment_seed2}"

FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"
DK1="denmark/32VNH/2017"

FR1_START="$CHECKPOINT_ROOT/source/source_FR1_seed2/fold_0/model.pt"
FR2_START="$CHECKPOINT_ROOT/source/source_FR2_seed2/fold_0/model.pt"
FR1_FR2_END="$CHECKPOINT_ROOT/uda/FR1_FR2_seed2/fold_0/checkpoint_last.pt"
FR2_DK1_END="$CHECKPOINT_ROOT/uda/FR2_DK1_seed2/fold_0/checkpoint_last.pt"

for checkpoint in "$FR1_START" "$FR2_START" "$FR1_FR2_END" "$FR2_DK1_END"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: required checkpoint not found: $checkpoint"
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u scripts/visualize_shapelet_pse_alignment.py \
  --data-root "$DATA_ROOT" \
  --source "$FR1" --target "$FR2" \
  --start-checkpoint "$FR1_START" \
  --end-checkpoint "$FR1_FR2_END" \
  --seed 2 --samples-per-class 64 \
  --output-dir "$OUTPUT_ROOT/FR1_FR2" \
  --device cuda

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u scripts/visualize_shapelet_pse_alignment.py \
  --data-root "$DATA_ROOT" \
  --source "$FR2" --target "$DK1" \
  --start-checkpoint "$FR2_START" \
  --end-checkpoint "$FR2_DK1_END" \
  --seed 2 --samples-per-class 64 \
  --output-dir "$OUTPUT_ROOT/FR2_DK1" \
  --device cuda
