#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
P_ROOT="${P_ROOT:-outputs/structure_phase_moment_4tasks_seed1}"
E_ROOT="${E_ROOT:-outputs/structure_phase_equivariance_4tasks_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_phase_final_protocol_seed1}"
mkdir -p "$LOG_ROOT"

AT1="austria/33UVP/2017"; DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"; FR2="france/31TCJ/2017"

evaluate() {
  local gpu="$1" label="$2" root="$3" source="$4" source_data="$5" target="$6" target_data="$7" equiv="$8"
  local task="${source}_${target}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "${task}_seed1" --eval --data_root "$DATA_ROOT" \
    --source "$source_data" --target "$target_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 \
    --shape-window-stride 8 --shapelet-count 16 --shapelet-beta 5 \
    --shape-resample-length 16 --shape-representation phase_moment \
    --shape-injection current_query --structure-shift-mode none \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0 --source-minority-mode base \
    --seed 1 --num_folds 1 --batch_size 128 --seq_length 30 --num_pixels 64 \
    --closed_set true --with_shift_aug false --progress_bar off \
    --output_dir "$root/uda" --tensorboard_log_dir /tmp/phase_final_protocol \
    timematch --weights "$P_ROOT/source/source_${source}_seed1" \
    --shape-da-mode batch_align --shape-alignment-view none \
    --shape-equivariance-weight "$equiv" --shape-equivariance-max-shift 60 \
    > "$LOG_ROOT/${label}_${task}.log" 2>&1
}

run_task() {
  local gpu="$1" source="$2" source_data="$3" target="$4" target_data="$5"
  evaluate "$gpu" P "$P_ROOT" "$source" "$source_data" "$target" "$target_data" 0
  evaluate "$gpu" E "$E_ROOT" "$source" "$source_data" "$target" "$target_data" 0.05
}

run_task 0 AT1 "$AT1" DK1 "$DK1" & P0=$!
run_task 1 FR1 "$FR1" FR2 "$FR2" & P1=$!
run_task 2 FR2 "$FR2" DK1 "$DK1" & P2=$!
run_task 3 DK1 "$DK1" AT1 "$AT1" & P3=$!
STATUS=0
for pid in "$P0" "$P1" "$P2" "$P3"; do wait "$pid" || STATUS=1; done
if (( STATUS != 0 )); then
  echo "ERROR: final-protocol evaluation failed; inspect $LOG_ROOT" >&2
  exit 1
fi

"$PYTHON_BIN" -u scripts/summarize_structure_phase_final_protocol.py \
  --p-root "$P_ROOT/uda" --e-root "$E_ROOT/uda" \
  --output outputs/structure_phase_final_protocol_seed1.csv
