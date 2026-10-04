#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"

COMMON_MODEL_ARGS=(
  --model psestructureprotoltae
  --structure-branch true --structure-exposer fourier
  --fourier_num_modes 13 --shape-dim 128
  --shape-window-scales 24 --shape-window-stride 8
  --shapelet-count 16 --shapelet-beta 5
  --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01
  --shapelet-shaping-weight 0.01 --shapelet-shaping-temperature 0.1
  --shape-class-weight 0.1 --shape-target-weight 0.05
  --shape-align-weight 0.05 --stats-align-weight 0.02
  --proto-momentum 0.9 --proto-temperature 0.1
  --proto-instance-weight 0.1
  --proto-init-epoch 1 --proto-ramp-start 0.1 --proto-ramp-epochs 5
  --with_shift_aug false
  --seed 1 --num_folds 1 --batch_size 128
  --seq_length 30 --num_pixels 64 --closed_set true --progress_bar off
)

run_variant() {
  local variant="$1" resample_length="$2"
  local exp_root log_root run_root
  case "$variant" in
    R16)
      exp_root="outputs/v2_r16_seed1"
      log_root="logs/v2_r16_seed1"
      run_root="runs/v2_r16_seed1"
      ;;
    R24)
      exp_root="outputs/v2_r24_seed1"
      log_root="logs/v2_r24_seed1"
      run_root="runs/v2_r24_seed1"
      ;;
    *)
      echo "ERROR: unsupported variant $variant" >&2
      return 2
      ;;
  esac
  mkdir -p "$exp_root/source" "$exp_root/uda" "$log_root" "$run_root"

  launch_task() {
    local gpu="$1" src_name="$2" src_data="$3" tgt_name="$4" tgt_data="$5"
    local task="${src_name}_${tgt_name}"
    local source_experiment="source_${src_name}_seed1"
    local uda_experiment="${task}_seed1"
    local source_weights="$exp_root/source/$source_experiment"
    local log_file="$log_root/${task}.log"

    : > "$log_file"
    echo "EXPERIMENT_CONFIG|variant=$variant|shape_window=24|shape_resample_length=$resample_length|seed=1|source=$src_name|target=$tgt_name|GPU=$gpu|test_checkpoint=checkpoint_last.pt" | tee -a "$log_file"
    echo "SOURCE_START|variant=$variant|task=$task|GPU=$gpu" | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
      -e "$source_experiment" \
      --data_root "$DATA_ROOT" --source "$src_data" --target "$src_data" \
      "${COMMON_MODEL_ARGS[@]}" \
      --shape-resample-length "$resample_length" \
      --epochs 100 --lr 0.001 --weight_decay 0.0001 --focal_loss_gamma 1.0 \
      --output_dir "$exp_root/source" \
      --tensorboard_log_dir "$run_root/source_${src_name}_seed1" \
      2>&1 | tee -a "$log_file"

    if [[ ! -f "$source_weights/fold_0/model.pt" ]]; then
      echo "ERROR|variant=$variant|task=$task|stage=source|missing=$source_weights/fold_0/model.pt" | tee -a "$log_file"
      return 1
    fi

    echo "UDA_START|variant=$variant|task=$task|GPU=$gpu|source_checkpoint=$source_weights/fold_0/model.pt|test_checkpoint=checkpoint_last.pt" | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
      -e "$uda_experiment" \
      --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
      "${COMMON_MODEL_ARGS[@]}" \
      --shape-resample-length "$resample_length" \
      --output_dir "$exp_root/uda" \
      --tensorboard_log_dir "$run_root/${task}_seed1" \
      timematch --weights "$source_weights" \
      --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
      --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
      --estimate_shift true --balance_source true --use_focal_loss true \
      --shift_source true --sample_size 100 --max_temporal_shift 60 \
      --domain_specific_bn true --shift_estimator AM --run_validation \
      --output_student true \
      2>&1 | tee -a "$log_file"
    echo "TASK_FINISHED|variant=$variant|task=$task" | tee -a "$log_file"
  }

  echo "VARIANT_START|variant=$variant|shape_resample_length=$resample_length"
  launch_task "$GPU0" AT1 "$AT1" DK1 "$DK1" &
  local pid0=$!
  launch_task "$GPU1" FR1 "$FR1" FR2 "$FR2" &
  local pid1=$!
  launch_task "$GPU2" FR2 "$FR2" DK1 "$DK1" &
  local pid2=$!
  launch_task "$GPU3" DK1 "$DK1" AT1 "$AT1" &
  local pid3=$!

  local status=0
  wait "$pid0" || { echo "FAILED|variant=$variant|task=AT1_DK1" >&2; status=1; }
  wait "$pid1" || { echo "FAILED|variant=$variant|task=FR1_FR2" >&2; status=1; }
  wait "$pid2" || { echo "FAILED|variant=$variant|task=FR2_DK1" >&2; status=1; }
  wait "$pid3" || { echo "FAILED|variant=$variant|task=DK1_AT1" >&2; status=1; }
  if (( status != 0 )); then
    echo "ERROR: $variant failed; R16/R24 pipeline stopped" >&2
    return 1
  fi
  echo "VARIANT_FINISHED|variant=$variant"
}

export PYTHONUNBUFFERED=1
run_variant "R16" 16
run_variant "R24" 24
echo "V2_RESAMPLE_ABLATION_FINISHED"
