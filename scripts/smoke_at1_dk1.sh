#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/user/TSStructure"
CONDA_ENV="time"
GPU_ID="${1:-0}"
EXP_NAME="smoke_at1_dk1_seed1"

case "$GPU_ID" in
    0|1|2|3) ;;
    *)
        echo "GPU_ID must be one of: 0, 1, 2, 3" >&2
        exit 1
        ;;
esac

cd "$REPO_ROOT"
mkdir -p outputs runs logs

if [[ -e "outputs/$EXP_NAME" ]]; then
    echo "Smoke output already exists: outputs/$EXP_NAME" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "PHYSICAL_GPU=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

conda run -n "$CONDA_ENV" --no-capture-output python -c '
import sys
import torch

print("torch=", torch.__version__)
print("cuda=", torch.cuda.is_available())
print("visible_gpus=", torch.cuda.device_count())
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if not torch.cuda.is_available():
    print("CUDA is not available; refusing CPU fallback", file=sys.stderr)
    sys.exit(1)
'

echo "SMOKE_START"
date
echo "GPU_ID=$GPU_ID"
nvidia-smi -i "$GPU_ID"

conda run -n "$CONDA_ENV" --no-capture-output \
    python -u train.py \
    --source "austria/33UVP/2017" \
    --target "denmark/32VNH/2017" \
    --closed_set True \
    --num_folds 1 \
    --seed 1 \
    --device cuda \
    --epochs 2 \
    --steps_per_epoch 5 \
    --batch_size 4 \
    --num_pixels 16 \
    --num_workers 0 \
    --structure_dim 128 \
    --domain_hidden_dim 128 \
    --grl_warmup_max_iters 250 \
    --amp true \
    --amp_dtype float16 \
    --lambda_task 1 \
    --lambda_geometry 1 \
    --lambda_alignment 1 \
    --lambda_structural_cls 1 \
    --lambda_structural_domain 1 \
    --lambda_component_cls 1 \
    --lambda_component_domain 1 \
    --progress_bar auto \
    --log_step 1 \
    --output_dir outputs \
    --tensorboard_log_dir runs \
    --experiment_name "$EXP_NAME"

for required_file in \
    "outputs/$EXP_NAME/train_config.json" \
    "outputs/$EXP_NAME/fold_0/model.pt"
do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing smoke output file: $required_file" >&2
        exit 1
    fi
done

echo "SMOKE_FINISHED"
date
find "outputs/$EXP_NAME" -maxdepth 2 -type f -print
nvidia-smi
