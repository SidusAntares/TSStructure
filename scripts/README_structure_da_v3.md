# TSStructure V3 offline server workflow

The server scripts use the already-active Python environment and the dataset
path configured by `train.py`. They do not activate Conda, install packages,
or download data. `DATA_ROOT` is optional; leaving it unset preserves the
`train.py` parser default. Stage 2 requires an explicit `STAGE2_CONFIG` JSON.

## Stage-2 evidence runtime

Domain Phase no longer performs an exhaustive `target x all classes` DP scan.
The runtime protocol is:

1. cache a deterministic representative target-train subset once;
2. acquire nested Phase evidence (`64 -> 128 -> 256 -> 512` by default);
3. propose classes by the high-recall union of classifier and identity-T
   candidates;
4. run exact fdasrsf curve-DP only for proposed pairs, in CPU worker processes;
5. stop exact DP as soon as Domain Phase is confirmed;
6. estimate stable labels and Domain Shape directly under the confirmed
   domain-level Phase group, without requiring individual DP hypotheses.

Logs expose proposal counts, exact solver calls, every rejection stage, and
P10/P50/P90 diagnostic distributions. The runtime-only evidence budgets and
worker count can be overridden by CLI without changing scientific gates.

Feature snapshots are disabled in smoke and formal launchers. Stage-2 EMA
checkpoints now contain the full Phase state, full Domain Shape state, evidence
sample IDs, and stable-label IDs/classes. The small `shape_diagnostics_*.json`
files remain for quick inspection. Legacy feature-snapshot/visualization code is
kept only for old experiments and optional offline use.

## Environment check

```bash
bash scripts/check_server_env.sh
```

## Smoke run

The smoke is AT1 -> DK1, seed 1, GPU 0. It uses tiny Phase evidence budgets so
that it validates the Stage-1 -> Stage-2 integration rather than benchmarking
registration throughput.

```bash
cd /data/user/TSStructure
mkdir -p logs/smoke_structure_da_v3/train_logs
STAGE2_CONFIG=/data/user/TSStructure/configs/stage2_pilot_v1.json

nohup env STAGE2_CONFIG="${STAGE2_CONFIG}" \
    bash scripts/smoke_structure_da_v3.sh \
    > logs/smoke_structure_da_v3/train_logs/nohup.log \
    2>&1 < /dev/null &

echo $! > logs/smoke_structure_da_v3/train_logs/launcher.pid
```

Monitor:

```bash
tail -f logs/smoke_structure_da_v3/train_logs/nohup.log
tail -f logs/smoke_structure_da_v3/train_logs/smoke.log
```

## Recommended formal workflow: one seed at a time

Run all 12 directed domain pairs for one seed across four GPUs, then repeat for
seeds 2 and 3 with the same frozen config.

```bash
cd /data/user/TSStructure
SEED=1
RUN_GROUP="structure_da_v3_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "logs/${RUN_GROUP}/train_logs"
STAGE2_CONFIG=/data/user/TSStructure/configs/stage2_pilot_v1.json

nohup env \
    SEED="${SEED}" \
    RUN_GROUP="${RUN_GROUP}" \
    STAGE2_CONFIG="${STAGE2_CONFIG}" \
    GPU0=0 GPU1=1 GPU2=2 GPU3=3 \
    bash scripts/run_structure_da_12tasks_4gpu_seed.sh \
    > "logs/${RUN_GROUP}/train_logs/nohup.log" \
    2>&1 < /dev/null &

echo $! > "logs/${RUN_GROUP}/train_logs/launcher.pid"
echo "${RUN_GROUP}"
```

Each GPU worker runs three tasks sequentially. The launcher writes
`manifest.tsv`, `completed.tsv`, `failed.tsv`, and `experiment_status.tsv` under
`logs/${RUN_GROUP}/train_logs/`. Per-task checkpoints and TensorBoard events are
under `outputs/${RUN_GROUP}/<task_seed>/`.

The fixed domain naming is:

```text
AT1 = austria/33UVP/2017
DK1 = denmark/32VNH/2017
FR1 = france/30TXT/2017
FR2 = france/31TCJ/2017
```

After seed 1 finishes, keep the same code/config and launch `SEED=2`, then
`SEED=3`. Do not average seeds produced with different Stage-2 configs.

## All 36 runs in one launcher

The legacy all-at-once launcher remains available:

```bash
RUN_GROUP="structure_da_v3_12tasks_3seeds_$(date +%Y%m%d_%H%M%S)"
mkdir -p "logs/${RUN_GROUP}/train_logs"
STAGE2_CONFIG=/data/user/TSStructure/configs/stage2_pilot_v1.json

nohup env \
    RUN_GROUP="${RUN_GROUP}" \
    STAGE2_CONFIG="${STAGE2_CONFIG}" \
    GPU0=0 GPU1=1 GPU2=2 GPU3=3 \
    bash scripts/run_structure_da_12tasks_4gpu_3seeds.sh \
    > "logs/${RUN_GROUP}/train_logs/nohup.log" \
    2>&1 < /dev/null &
```

The seed-wise launcher is preferred when results need to be inspected between
seeds.

## Stage-2-only resume

A completed Stage-1 checkpoint can be reused without retraining Stage 1:

```bash
python -u train.py \
    --source austria/33UVP/2017 \
    --target denmark/32VNH/2017 \
    --seed 1 \
    --device cuda:0 \
    --output_dir outputs/<run>/AT1_DK1_seed1 \
    --tensorboard_log_dir outputs/<run>/AT1_DK1_seed1/tensorboard/events \
    --stage2_only \
    --stage1_checkpoint outputs/<run>/AT1_DK1_seed1/fold_0/stage1_best.pt \
    --stage2_config configs/stage2_pilot_v1.json \
    --feature_snapshot_interval 0 \
    --progress_bar off
```

The selected checkpoint restores the frozen Stage-1 source prototype bank.
Only the K_reg source registration prototypes, which are not stored in the
Stage-1 checkpoint, are rescanned before Stage 2.

## What to monitor

During Stage-2 initialization, useful lines are:

```text
STAGE2_SOURCE_BANK
STAGE2_SOURCE_REGISTRATION_SCAN
TARGET_GEOMETRY_CACHE_READY
TARGET_HYPOTHESIS_SCAN_STAGE_START
TARGET_DP_PROGRESS
TARGET_HYPOTHESIS_SCAN_STAGE_DONE
TARGET_HYPOTHESIS_SCAN_DISTRIBUTIONS
STAGE2_PHASE_EVIDENCE_STAGE
STAGE2_SHAPE_EVIDENCE_STAGE
STAGE2_STATISTICS
STAGE2_TRAIN
STAGE2_TARGET_VAL
```

`TARGET_HYPOTHESIS_SCAN_STAGE_DONE` reports pre-support, solver, each gamma
legality gate, gain, Shape-support, and Shape-outer rejection counts. This is
the primary input for tuning pilot thresholds.

## Stopping safely

Killing only the launcher PID may leave already-started training children. Use
`experiment_status.tsv` or `pgrep -af train.py` to identify task PIDs, inspect
them with `ps -fp <PID>`, and terminate only confirmed experiment processes.


## Stage-2 Phase A/B diagnostic

Use `run_stage2_phase_diagnostic_ab.sh` after a completed Stage-1 checkpoint when
Phase evidence is unexpectedly rejected.  This launcher performs **statistics
only**: it never executes a Stage-2 optimizer step.

It runs two deterministic cases on the same target split and seed:

- **A / proposal:** V6 classifier-top2 ∪ identity-T-top2 proposal, progressive
  64→128 evidence, with only the roughness gate bypassed.
- **B / all-class:** exhaustive exact DP for all ready source classes with the
  same 64→128 evidence cache. The comparison itself uses the exact same first
  64 samples; roughness alone is bypassed.

The comparison is written to
`logs/<RUN_GROUP>/comparison.json`.  In addition to gate counts/distributions it
reports whether both runs used identical sample IDs and how many hypotheses
retained by exhaustive DP were missed by the proposal.  A non-empty
`missed_by_proposal` list means the proposal requires revision before formal
experiments.

Example:

```bash
nohup env RUN_GROUP=stage2_phase_diag_at1_dk1 \
  bash scripts/run_stage2_phase_diagnostic_ab.sh \
  > logs/stage2_phase_diag_at1_dk1_nohup.log 2>&1 < /dev/null &
```
