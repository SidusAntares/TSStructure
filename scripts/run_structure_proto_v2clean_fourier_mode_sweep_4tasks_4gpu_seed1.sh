#!/usr/bin/env bash
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

EXP_ROOT="${EXP_ROOT:-outputs/structure_proto_v2clean_fourier_mode_sweep_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/structure_proto_v2clean_fourier_mode_sweep_seed1}"
RUN_ROOT="${RUN_ROOT:-runs/structure_proto_v2clean_fourier_mode_sweep_seed1}"
MODES=(9 13 17 21)

AT1="austria/33UVP/2017"
DK1="denmark/32VNH/2017"
FR1="france/30TXT/2017"
FR2="france/31TCJ/2017"

print_plan() {
  local gpu="$1" src_name="$2" tgt_name="$3"
  local task="${src_name}_${tgt_name}"
  local mode
  for mode in "${MODES[@]}"; do
    echo "SWEEP_PLAN|gpu=${gpu}|task=${task}|mode=${mode}"
  done
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "SWEEP_CONFIG|method=v2clean_fourier_mode_sweep|seed=1|modes=9,13,17,21"
  print_plan "$GPU0" AT1 DK1
  print_plan "$GPU1" FR1 FR2
  print_plan "$GPU2" FR2 DK1
  print_plan "$GPU3" DK1 AT1
  echo "SWEEP_TOTAL|source_commands=16|uda_commands=16|combinations=16"
  exit 0
fi

mkdir -p "$EXP_ROOT" "$LOG_ROOT" "$RUN_ROOT"
export PYTHONUNBUFFERED=1

annotate_manifest() {
  local manifest="$1" task="$2" mode="$3" stage="$4"
  "$PYTHON_BIN" - "$manifest" "$task" "$mode" "$stage" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"manifest missing: {path}")
packet = json.loads(path.read_text(encoding="utf-8"))
packet.update({
    "method": "v2clean_fourier_mode_sweep",
    "task": sys.argv[2],
    "fourier_num_modes": int(sys.argv[3]),
    "seed": 1,
    "stage": sys.argv[4],
})
path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
PY
}

run_combination() {
  local gpu="$1" src_name="$2" src_data="$3" tgt_name="$4" tgt_data="$5" mode="$6"
  local task="${src_name}_${tgt_name}"
  local task_root="$EXP_ROOT/mode_${mode}/${task}"
  local source_root="$task_root/source"
  local uda_root="$task_root/uda"
  local source_experiment="source_${src_name}_m${mode}_seed1"
  local uda_experiment="v2clean_sweep_m${mode}_${task}_seed1"
  local source_weights="$source_root/$source_experiment"
  local source_log="$LOG_ROOT/mode_${mode}/${task}/source.log"
  local uda_log="$LOG_ROOT/mode_${mode}/${task}/uda.log"
  local run_root="$RUN_ROOT/mode_${mode}/${task}"

  mkdir -p "$source_root" "$uda_root" "$(dirname "$source_log")" "$run_root"
  : > "$source_log"
  : > "$uda_log"

  echo "FOURIER_MODE_SWEEP|task=${task}|num_modes=${mode}|seed=1|stage=source" | tee -a "$source_log"
  if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "$source_experiment" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$src_data" \
    --model psestructureprotoltae \
    --structure-branch true --structure-exposer fourier \
    --fourier_num_modes "$mode" --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --with_shift_aug false \
    --seed 1 --num_folds 1 --epochs 100 --batch_size 128 \
    --lr 0.001 --weight_decay 0.0001 --focal_loss_gamma 1.0 \
    --seq_length 30 --num_pixels 64 --closed_set true --progress_bar off \
    --output_dir "$source_root" \
    --tensorboard_log_dir "$run_root/source" \
    2>&1 | tee -a "$source_log"; then
    echo "FOURIER_MODE_SWEEP_FAILED|task=${task}|num_modes=${mode}|stage=source" | tee -a "$source_log"
    return 1
  fi

  if [[ ! -f "$source_weights/fold_0/model.pt" ]]; then
    echo "FOURIER_MODE_SWEEP_FAILED|task=${task}|num_modes=${mode}|stage=source|missing=$source_weights/fold_0/model.pt" | tee -a "$source_log"
    return 1
  fi
  annotate_manifest "$source_weights/fold_0/manifest.json" "$task" "$mode" source

  echo "FOURIER_MODE_SWEEP|task=${task}|num_modes=${mode}|seed=1|stage=uda" | tee -a "$uda_log"
  if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train.py \
    -e "$uda_experiment" \
    --data_root "$DATA_ROOT" --source "$src_data" --target "$tgt_data" \
    --model psestructureprotoltae \
    --structure-branch true --structure-exposer fourier \
    --fourier_num_modes "$mode" --shape-dim 128 \
    --shape-window-scales 24 --shape-window-stride 8 \
    --shapelet-count 16 --shapelet-beta 5 --shape-resample-length 16 \
    --shapelet-diversity-margin 0.5 --shapelet-diversity-weight 0.01 \
    --shape-class-weight 0.1 --shape-align-weight 0.05 \
    --seed 1 --num_folds 1 --batch_size 128 \
    --seq_length 30 --num_pixels 64 --closed_set true \
    --with_shift_aug false --progress_bar off \
    --output_dir "$uda_root" \
    --tensorboard_log_dir "$run_root/uda" \
    timematch --weights "$source_weights" \
    --epochs 20 --steps_per_epoch 500 --lr 0.0001 \
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
    --estimate_shift true --balance_source true --use_focal_loss true \
    --shift_source true --sample_size 100 --max_temporal_shift 60 \
    --domain_specific_bn true --shift_estimator AM --run_validation \
    --output_student true \
    2>&1 | tee -a "$uda_log"; then
    echo "FOURIER_MODE_SWEEP_FAILED|task=${task}|num_modes=${mode}|stage=uda" | tee -a "$uda_log"
    return 1
  fi
  annotate_manifest "$uda_root/$uda_experiment/fold_0/manifest.json" "$task" "$mode" uda
  echo "FOURIER_MODE_SWEEP_FINISHED|task=${task}|num_modes=${mode}|seed=1" | tee -a "$uda_log"
}

worker_task() {
  local gpu="$1" src_name="$2" src_data="$3" tgt_name="$4" tgt_data="$5"
  local mode
  for mode in "${MODES[@]}"; do
    run_combination "$gpu" "$src_name" "$src_data" "$tgt_name" "$tgt_data" "$mode" || return 1
  done
}

summarize_results() {
  "$PYTHON_BIN" - "$EXP_ROOT" "$LOG_ROOT" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
log_root = Path(sys.argv[2])
modes = (9, 13, 17, 21)
tasks = (("AT1", "DK1"), ("FR1", "FR2"), ("FR2", "DK1"), ("DK1", "AT1"))
summary_rows = []
class_rows = []
validation_pattern = re.compile(r"Validation result:.*?f1=([0-9.]+)")

for mode in modes:
    for source, target in tasks:
        task = f"{source}_{target}"
        experiment = f"v2clean_sweep_m{mode}_{task}_seed1"
        fold = output_root / f"mode_{mode}" / task / "uda" / experiment / "fold_0"
        metrics_paths = list(fold.glob("test_metrics_*.json"))
        report_paths = list(fold.glob("class_report_*.txt"))
        if len(metrics_paths) != 1:
            raise SystemExit(f"expected one test metrics file for mode={mode} task={task}: {metrics_paths}")
        metrics = json.loads(metrics_paths[0].read_text(encoding="utf-8"))
        uda_log = log_root / f"mode_{mode}" / task / "uda.log"
        validation = [float(value) for value in validation_pattern.findall(uda_log.read_text(encoding="utf-8"))]
        if not validation:
            raise SystemExit(f"no validation metrics for mode={mode} task={task}: {uda_log}")
        best_value = max(validation)
        summary_rows.append({
            "mode": mode,
            "source": source,
            "target": target,
            "best_val_macro_f1": best_value,
            "test_macro_f1": metrics["macro_f1"],
            "test_accuracy": metrics["accuracy"],
            "best_epoch": validation.index(best_value) + 1,
        })
        if len(report_paths) == 1:
            for line in report_paths[0].read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) != 5 or fields[0] in {"accuracy", "macro", "weighted"}:
                    continue
                try:
                    f1, support = float(fields[3]), int(fields[4])
                except ValueError:
                    continue
                class_rows.append({
                    "mode": mode, "source": source, "target": target,
                    "class": fields[0], "support": support, "f1": f1,
                })

summary_path = output_root / "fourier_mode_sweep_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=(
        "mode", "source", "target", "best_val_macro_f1",
        "test_macro_f1", "test_accuracy", "best_epoch",
    ))
    writer.writeheader()
    writer.writerows(summary_rows)
if len(summary_rows) != 16:
    raise SystemExit(f"summary must contain 16 rows, found {len(summary_rows)}")

per_class_path = output_root / "fourier_mode_sweep_per_class.csv"
with per_class_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=(
        "mode", "source", "target", "class", "support", "f1",
    ))
    writer.writeheader()
    writer.writerows(class_rows)
print(f"FOURIER_MODE_SWEEP_SUMMARY|rows={len(summary_rows)}|path={summary_path}")
print(f"FOURIER_MODE_SWEEP_PER_CLASS|rows={len(class_rows)}|path={per_class_path}")
PY
}

worker_task "$GPU0" AT1 "$AT1" DK1 "$DK1" & PID0=$!
worker_task "$GPU1" FR1 "$FR1" FR2 "$FR2" & PID1=$!
worker_task "$GPU2" FR2 "$FR2" DK1 "$DK1" & PID2=$!
worker_task "$GPU3" DK1 "$DK1" AT1 "$AT1" & PID3=$!

status=0
wait "$PID0" || status=1
wait "$PID1" || status=1
wait "$PID2" || status=1
wait "$PID3" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "FOURIER_MODE_SWEEP_FAILED|one_or_more_workers_failed=true" >&2
  exit 1
fi

summarize_results
echo "FOURIER_MODE_SWEEP_ALL_FINISHED|combinations=16"
