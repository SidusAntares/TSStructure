# TSStructure V3 offline server workflow

The server uses the already active Python environment, local code, and the
dataset path configured by `train.py`. The scripts do not activate Conda, use
Git, access a network, download files, or install packages. `DATA_ROOT`,
`PYTHON_BIN`, `OUTPUT_ROOT`, and `LOG_ROOT` remain optional overrides.

## Environment check

```bash
bash scripts/check_server_env.sh
```

## Smoke run

The smoke is fixed to AT1 -> DK1, seed 1, GPU 0, one epoch and two steps.
Feature snapshots are explicitly disabled. It verifies a complete training
step, validation, checkpoint restoration, final evaluation, and the disabled
snapshot path.

```bash
cd /data/user/TSStructure

mkdir -p \
    logs/smoke_structure_da_v3/train_logs \
    logs/smoke_structure_da_v3/snapshots

nohup bash scripts/smoke_structure_da_v3.sh \
    > logs/smoke_structure_da_v3/train_logs/nohup.log \
    2>&1 < /dev/null &

echo $! > logs/smoke_structure_da_v3/train_logs/launcher.pid
```

Monitor it with:

```bash
tail -f logs/smoke_structure_da_v3/train_logs/nohup.log
tail -f logs/smoke_structure_da_v3/train_logs/smoke.log
```

Training artifacts stay in `outputs/smoke_structure_da_v3/`. The smoke log is
the only per-task log and ends with `SMOKE_RESULT|status=SUCCESS` or
`SMOKE_RESULT|status=FAILED`.

## Formal 12-task x 3-seed experiment

Run this only after reviewing a successful smoke:

```bash
cd /data/user/TSStructure

RUN_GROUP="structure_da_v3_12tasks_3seeds_$(date +%Y%m%d_%H%M%S)"

mkdir -p \
    "logs/${RUN_GROUP}/train_logs" \
    "logs/${RUN_GROUP}/snapshots"

nohup env \
    RUN_GROUP="${RUN_GROUP}" \
    GPU0=0 GPU1=1 GPU2=2 GPU3=3 \
    bash scripts/run_structure_da_12tasks_4gpu_3seeds.sh \
    > "logs/${RUN_GROUP}/train_logs/nohup.log" \
    2>&1 < /dev/null &

echo $! > "logs/${RUN_GROUP}/train_logs/launcher.pid"
echo "${RUN_GROUP}"
```

Monitor scheduling, one task, and GPUs:

```bash
tail -f "logs/${RUN_GROUP}/train_logs/nohup.log"
tail -f "logs/${RUN_GROUP}/train_logs/AT1_DK1_seed1.log"
watch -n 5 nvidia-smi
```

The four GPU workers each run nine jobs sequentially. Each task has exactly one
merged stdout/stderr log in `logs/${RUN_GROUP}/train_logs/`. Experiment-level
PID, exit code, and completion state are recorded in `experiment_status.tsv`.
Snapshots at epochs 25, 50, 75, and 100 are stored under
`logs/${RUN_GROUP}/snapshots/<task_seed>/`. Checkpoints, metrics, and TensorBoard
events remain under `outputs/${RUN_GROUP}/<task_seed>/`. Each domain contributes
at most eight fixed parcels per class. Epoch 25 fits one source+target PCA basis
for PSE and a separate source+target basis for aligned Shape. Later epochs reuse
those bases and store only PC1/PC2 curves.

## Offline snapshot plots

Training never invokes visualization. After snapshots exist:

```bash
python scripts/visualize_structure_feature_snapshots.py \
    --snapshot-dir "logs/${RUN_GROUP}/snapshots/AT1_DK1_seed1" \
    --output-dir "analysis_output/${RUN_GROUP}/AT1_DK1_seed1" \
    --display-samples-per-class 8 \
    --separation-samples-per-class 3 \
    --components 1 2
```

The visualizer also reads the previous full-dimensional snapshot schema. For
legacy snapshots it fits one joint in-memory PCA per feature family across all
four epochs and never modifies the source NPZ files.

## Stopping safely

Killing only the launcher PID may leave already-started training children. Use
`experiment_status.tsv` to identify recorded task PIDs, inspect them with
`ps -fp <PID>`, and only then terminate the confirmed experiment processes.
