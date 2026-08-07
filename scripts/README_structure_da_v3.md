# TSStructure V3 offline server workflow

The server uses the already active Python environment, local code, and the
dataset path configured by `train.py`. The scripts do not activate Conda, use
Git, access a network, download files, or install packages. `DATA_ROOT`,
`PYTHON_BIN`, `OUTPUT_ROOT`, and `LOG_ROOT` remain optional overrides.  Round 7
requires an explicit `STAGE2_CONFIG` JSON because the statistical thresholds and
Stage-2 objective weights intentionally have no launcher defaults.

## Environment check

```bash
bash scripts/check_server_env.sh
```

## Smoke run

The smoke is fixed to AT1 -> DK1, seed 1, GPU 0.  By default it runs one
Stage-1 epoch/two source steps followed by one Stage-2 epoch/one optimizer step.
Feature snapshots are explicitly disabled.  It verifies the Stage-1 -> Stage-2
boundary, initial target-statistics settling, a Stage-2 optimizer/EMA update,
target validation, and `stage2_last_ema.pt`.  Formal target-test diagnostics
remain restricted to epochs 20/40/60 and are therefore not required by smoke.

```bash
cd /data/user/TSStructure

mkdir -p \
    logs/smoke_structure_da_v3/train_logs \
    logs/smoke_structure_da_v3/snapshots

STAGE2_CONFIG=/data/user/configs/tsstructure_v3_stage2.json

nohup env STAGE2_CONFIG="${STAGE2_CONFIG}" bash scripts/smoke_structure_da_v3.sh \
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

STAGE2_CONFIG=/data/user/configs/tsstructure_v3_stage2.json

nohup env \
    RUN_GROUP="${RUN_GROUP}" \
    STAGE2_CONFIG="${STAGE2_CONFIG}" \
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
at most eight fixed parcels per class. Stable per-parcel indices select exactly
`num_pixels` pixels with the same sampling/padding semantics as training, and all
epochs reuse those indices. The first successful scheduled snapshot fits one
parcel-weighted source+target PCA basis for PSE and a separate joint basis for
unaligned/aligned Shape. Later epochs reuse those bases and store only PC1-PC8
curves. Snapshot inference uses an independent batch size of 8 and retries CUDA
OOM failures with progressively smaller batches. Per-epoch outcomes are recorded
atomically in `snapshot_status.json`; training continues and the launcher reports
`COMPLETED_WITH_SNAPSHOT_FAILURE` only while the current status has failures.


The Stage-2 JSON is a flat object whose keys use the same names as the CLI
destinations, for example `stage2_registration_lambda`,
`stage2_phase_confirmation_patience`, `stage2_shape_confirmation_patience`,
`stage2_lambda_src_proto`, `stage2_ema_decay`, and `stage2_lambda_delta`.  The
launcher fails before starting any run if `STAGE2_CONFIG` is absent.  Values are
not documented here because they are experimental configuration, not frozen
method defaults.

Stage 2 always uses the source-val-selected `stage1_best.pt`, runs 60 epochs in
20-epoch fixed-statistics blocks, validates the EMA teacher every epoch, and
runs diagnostic target test only at epochs 20/40/60.  The fold directory keeps
`stage2_ema_020.pt`, `stage2_ema_040.pt`, `stage2_ema_060.pt`,
`stage2_best_target_val_ema.pt`, and `stage2_last_ema.pt`, plus Shape/oracle
diagnostics.

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

The visualizer discovers every available epoch, supports PC1-PC8 for schema 3,
and adds unaligned/aligned Shape, phase-status, and accepted-warp diagnostics.
It also reads schema 1 full-dimensional and schema 2 compact snapshots. Schema 2
cannot fabricate unaligned Shape and reports that diagnostic as skipped. Source
NPZ files are never modified.

## Stopping safely

Killing only the launcher PID may leave already-started training children. Use
`experiment_status.tsv` to identify recorded task PIDs, inspect them with
`ps -fp <PID>`, and only then terminate the confirmed experiment processes.
