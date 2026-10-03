#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"; GPU3="${GPU3:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
P_ROOT="${P_ROOT:-outputs/structure_phase_moment_4tasks_seed1}"
P_LOG="${P_LOG:-logs/structure_phase_moment_4tasks_seed1}"
P_RUN="${P_RUN:-runs/structure_phase_moment_4tasks_seed1}"
E_ROOT="${E_ROOT:-outputs/structure_phase_equivariance_4tasks_seed1}"
E_LOG="${E_LOG:-logs/structure_phase_equivariance_4tasks_seed1}"
E_RUN="${E_RUN:-runs/structure_phase_equivariance_4tasks_seed1}"
P_SOURCE_ROOT="$P_ROOT/source"

AT1="austria/33UVP/2017"; DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"; FR2="france/31TCJ/2017"

mkdir -p "$P_SOURCE_ROOT" "$P_ROOT/uda" "$E_ROOT/uda" "$P_LOG" "$E_LOG" "$P_RUN" "$E_RUN"
export PYTHONUNBUFFERED=1

run_source() {
  local gpu="$1" alias="$2" dataset="$3"
  local experiment="source_${alias}_seed1" log="$P_LOG/source_${alias}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "$experiment" --data_root "$DATA_ROOT" --source "$dataset" --target "$dataset" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 \
    --shape-window-stride 8 --shapelet-count 16 --shapelet-beta 5 \
    --shape-resample-length 16 --shape-representation phase_moment \
    --shape-injection current_query --structure-shift-mode none \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --source-minority-mode base --with_shift_aug false \
    --seed 1 --num_folds 1 --epochs 100 --batch_size 128 --lr 0.001 \
    --weight_decay 0.0001 --focal_loss_gamma 1.0 --seq_length 30 --num_pixels 64 \
    --closed_set true --progress_bar off --output_dir "$P_SOURCE_ROOT" \
    --tensorboard_log_dir "$P_RUN/source_${alias}_seed1" > "$log" 2>&1
  test -f "$P_SOURCE_ROOT/$experiment/fold_0/model.pt"
}

run_p() {
  local gpu="$1" src="$2" src_data="$3" tgt="$4" tgt_data="$5"
  local task="${src}_${tgt}" weights="$P_SOURCE_ROOT/source_${src}_seed1"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "${task}_seed1" --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 \
    --shape-window-stride 8 --shapelet-count 16 --shapelet-beta 5 \
    --shape-resample-length 16 --shape-representation phase_moment \
    --shape-injection current_query --structure-shift-mode none \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0 --source-minority-mode base \
    --seed 1 --num_folds 1 --batch_size 128 --seq_length 30 --num_pixels 64 \
    --closed_set true --with_shift_aug false --progress_bar off \
    --output_dir "$P_ROOT/uda" --tensorboard_log_dir "$P_RUN/${task}_seed1" \
    timematch --weights "$weights" --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 --estimate_shift true \
    --balance_source true --use_focal_loss true --shift_source true --sample_size 100 \
    --max_temporal_shift 60 --domain_specific_bn true --shift_estimator AM \
    --run_validation --output_student true --shape-da-mode batch_align \
    --shape-alignment-view none --oracle-pseudo-labels false \
    --adaptive-pseudo-selection false --shape-equivariance-weight 0 \
    --shape-equivariance-max-shift 60 > "$P_LOG/P_${task}.log" 2>&1
}

run_e() {
  local gpu="$1" src="$2" src_data="$3" tgt="$4" tgt_data="$5"
  local task="${src}_${tgt}" weights="$P_SOURCE_ROOT/source_${src}_seed1"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "${task}_seed1" --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae --structure-branch true --structure-exposer fourier \
    --fourier_num_modes 13 --shape-dim 128 --shape-window-scales 24 \
    --shape-window-stride 8 --shapelet-count 16 --shapelet-beta 5 \
    --shape-resample-length 16 --shape-representation phase_moment \
    --shape-injection current_query --structure-shift-mode none \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0 --source-minority-mode base \
    --seed 1 --num_folds 1 --batch_size 128 --seq_length 30 --num_pixels 64 \
    --closed_set true --with_shift_aug false --progress_bar off \
    --output_dir "$E_ROOT/uda" --tensorboard_log_dir "$E_RUN/${task}_seed1" \
    timematch --weights "$weights" --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 --estimate_shift true \
    --balance_source true --use_focal_loss true --shift_source true --sample_size 100 \
    --max_temporal_shift 60 --domain_specific_bn true --shift_estimator AM \
    --run_validation --output_student true --shape-da-mode batch_align \
    --shape-alignment-view none --oracle-pseudo-labels false \
    --adaptive-pseudo-selection false --shape-equivariance-weight 0.05 \
    --shape-equivariance-max-shift 60 > "$E_LOG/E_${task}.log" 2>&1
}

# The explicit blocks make round barriers and GPU ownership auditable.
run_source "$GPU0" AT1 "$AT1" & P0=$!
run_source "$GPU1" FR1 "$FR1" & P1=$!
run_source "$GPU2" FR2 "$FR2" & P2=$!
run_source "$GPU3" DK1 "$DK1" & P3=$!
STATUS=0
for pid in "$P0" "$P1" "$P2" "$P3"; do wait "$pid" || STATUS=1; done
if (( STATUS != 0 )); then echo "ERROR: source round failed"; exit 1; fi

run_p "$GPU0" AT1 "$AT1" DK1 "$DK1" & P0=$!
run_p "$GPU1" FR1 "$FR1" FR2 "$FR2" & P1=$!
run_p "$GPU2" FR2 "$FR2" DK1 "$DK1" & P2=$!
run_p "$GPU3" DK1 "$DK1" AT1 "$AT1" & P3=$!
STATUS=0
for pid in "$P0" "$P1" "$P2" "$P3"; do wait "$pid" || STATUS=1; done
if (( STATUS != 0 )); then echo "ERROR: P round failed"; exit 1; fi

run_e "$GPU0" AT1 "$AT1" DK1 "$DK1" & P0=$!
run_e "$GPU1" FR1 "$FR1" FR2 "$FR2" & P1=$!
run_e "$GPU2" FR2 "$FR2" DK1 "$DK1" & P2=$!
run_e "$GPU3" DK1 "$DK1" AT1 "$AT1" & P3=$!
STATUS=0
for pid in "$P0" "$P1" "$P2" "$P3"; do wait "$pid" || STATUS=1; done
if (( STATUS != 0 )); then echo "ERROR: E round failed"; exit 1; fi
