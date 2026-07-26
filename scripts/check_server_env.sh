#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/user/TSStructure"
DATA_ROOT="/data/user/dataset/timematch_data"
CONDA_ENV="time"

cd "$REPO_ROOT"

pwd
git branch --show-current
git rev-parse HEAD
git status --short

command -v conda

conda run -n "$CONDA_ENV" --no-capture-output python -c '
import torch
import sys

print("python=", sys.version)
print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
print("torch_cuda=", torch.version.cuda)
print("cuda_device_count=", torch.cuda.device_count())
'

nvidia-smi

for dataset_dir in \
    "$DATA_ROOT/austria/33UVP/2017" \
    "$DATA_ROOT/denmark/32VNH/2017" \
    "$DATA_ROOT/france/30TXT/2017" \
    "$DATA_ROOT/france/31TCJ/2017"
do
    if [[ ! -d "$dataset_dir" ]]; then
        echo "Missing TimeMatch dataset directory: $dataset_dir" >&2
        exit 1
    fi
done

conda run -n "$CONDA_ENV" --no-capture-output \
    python train.py --help >/dev/null

echo "SERVER_CHECK_OK"
