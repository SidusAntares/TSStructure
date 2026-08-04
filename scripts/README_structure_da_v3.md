# TSStructure V3 offline server workflow

The server uses local code, its existing Python environment, and the dataset
path already configured as the `train.py` default. No data-path environment
variable is required. `DATA_ROOT` is only an optional explicit override.
These scripts perform no source-control operation, network access, download,
or package installation.

## 1. Environment check

From the repository root:

```bash
bash scripts/check_server_env.sh
```

## 2. Smoke run

The smoke is fixed to AT1 → DK1, seed 1, GPU 0, one epoch and two steps. It
checks execution and artifacts; its metrics are not a DA performance result.

```bash
cd /path/to/TSStructure

mkdir -p logs/smoke_structure_da_v3

nohup bash scripts/smoke_structure_da_v3.sh \
    > logs/smoke_structure_da_v3/nohup.log \
    2>&1 < /dev/null &

echo $! > logs/smoke_structure_da_v3/launcher.pid
```

View the launcher log:

```bash
tail -f logs/smoke_structure_da_v3/nohup.log
```

View the detailed training log:

```bash
tail -f logs/smoke_structure_da_v3/train.log
```

Smoke results are written to `outputs/smoke_structure_da_v3/`; text logs,
the command, environment record, PID and smoke status are written to
`logs/smoke_structure_da_v3/`.

## 3. Formal 12-task × 3-seed experiment

After the smoke succeeds, the standard next step is the complete 36-run
experiment. No diagnostic or four-task pilot is a prerequisite.

```bash
cd /path/to/TSStructure

RUN_GROUP="structure_da_v3_12tasks_3seeds_$(date +%Y%m%d_%H%M%S)"

mkdir -p "logs/${RUN_GROUP}"

nohup env \
    RUN_GROUP="${RUN_GROUP}" \
    GPU0=0 GPU1=1 GPU2=2 GPU3=3 \
    bash scripts/run_structure_da_12tasks_4gpu_3seeds.sh \
    > "logs/${RUN_GROUP}/nohup.log" \
    2>&1 < /dev/null &

echo $! > "logs/${RUN_GROUP}/launcher.pid"

echo "RUN_GROUP=${RUN_GROUP}"
echo "PID=$(cat "logs/${RUN_GROUP}/launcher.pid")"
```

View overall scheduling:

```bash
tail -f "logs/${RUN_GROUP}/nohup.log"
```

View one detailed task log:

```bash
tail -f "logs/${RUN_GROUP}/AT1_DK1_seed1/train.log"
```

View GPU use:

```bash
watch -n 5 nvidia-smi
```

Training artifacts are stored under `outputs/${RUN_GROUP}/<task_seed>/`.
All launcher and per-task text logs are stored under
`logs/${RUN_GROUP}/`. Each of the four GPU workers runs nine jobs
sequentially, so a physical GPU has at most one training process.

## Stopping safely

只 kill launcher PID 不一定会终止已经启动的训练子进程。

First inspect every recorded child PID and its command:

```bash
find "logs/${RUN_GROUP}" -name train.pid -type f -print | while read -r pid_file; do
    child_pid="$(cat "${pid_file}")"
    ps -fp "${child_pid}"
done
```

After confirming that each process belongs to this experiment group, stop the
recorded children explicitly, then stop the launcher:

```bash
find "logs/${RUN_GROUP}" -name train.pid -type f -print | while read -r pid_file; do
    child_pid="$(cat "${pid_file}")"
    kill "${child_pid}"
done
kill "$(cat "logs/${RUN_GROUP}/launcher.pid")"
```

Existing experiment output is not overwritten by default. `OVERWRITE=1`
must be intentional; it preserves the active `nohup.log` and `launcher.pid`
while replacing earlier experiment artifacts.
