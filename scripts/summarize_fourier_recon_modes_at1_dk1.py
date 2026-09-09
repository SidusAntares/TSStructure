#!/usr/bin/env python3
import argparse, csv, re
from pathlib import Path

MODES = (9, 11, 13, 15, 17, 19)
ORIGINAL = 0.8437
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
FIELDS = ("mode", "source_test_accuracy", "source_test_macro_f1",
          "source_on_target_accuracy", "source_on_target_macro_f1",
          "initial_shift", "best_val_macro_f1", "best_epoch",
          "test_macro_f1", "final_shift", "delta_original", "status")


def _read(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _test(text, experiment):
    pattern = rf"^Test result for {re.escape(experiment)}: accuracy=({NUM}), f1=({NUM})\s*$"
    found = re.findall(pattern, text, re.MULTILINE)
    return tuple(map(float, found[0])) if len(found) == 1 else None


def _parse_mode(log_root, mode, seed=1):
    folder = Path(log_root) / f"mode{mode:02d}"
    source = _test(_read(folder / "source.log"), f"fourier_recon_m{mode:02d}_AT1_source_seed{seed}")
    eval_text = _read(folder / "source_on_target.log")
    marker = re.findall(
        rf"^SOURCE_ON_TARGET\|source=AT1\|target=DK1\|mode={mode}\|accuracy=({NUM})\|macro_f1=({NUM})\s*$",
        eval_text, re.MULTILINE)
    on_target = tuple(map(float, marker[0])) if len(marker) == 1 else None
    da_text = _read(folder / "da.log")
    initial = re.findall(
        rf"^INITIAL_SHIFT\|source=AT1\|target=DK1\|mode={mode}\|shift_days=({NUM})\s*$",
        da_text, re.MULTILINE)
    vals = [float(x) for x in re.findall(rf"^Validation result: .*?f1=({NUM})\s*$", da_text, re.MULTILINE)]
    shifts = [float(x) for x in re.findall(rf"^Best AM Score shift ({NUM}) ", da_text, re.MULTILINE)]
    final = _test(da_text, f"fourier_recon_m{mode:02d}_AT1_DK1_timematch_seed{seed}")
    missing = []
    for value, name in ((source, "SOURCE_TEST"), (on_target, "SOURCE_ON_TARGET"),
                        (initial, "INITIAL_SHIFT"), (vals, "VALIDATION"), (final, "DA_TEST")):
        if not value:
            missing.append("MISSING_" + name)
    best = max(vals) if vals else None
    row = dict.fromkeys(FIELDS, "")
    row.update({"mode": mode,
                "source_test_accuracy": source[0] if source else "",
                "source_test_macro_f1": source[1] if source else "",
                "source_on_target_accuracy": on_target[0] if on_target else "",
                "source_on_target_macro_f1": on_target[1] if on_target else "",
                "initial_shift": float(initial[0]) if initial else "",
                "best_val_macro_f1": best if best is not None else "",
                "best_epoch": vals.index(best) + 1 if best is not None else "",
                "test_macro_f1": final[1] if final else "",
                "final_shift": shifts[-1] if shifts else "",
                "delta_original": final[1] - ORIGINAL if final else "",
                "status": ";".join(missing) if missing else "SUCCESS"})
    pseudo = [int(x) for x in re.findall(r"^Teacher pseudo label F1 .*?\(n=(\d+)\)\s*$", da_text, re.MULTILINE)]
    dynamics = [{"mode": mode, "epoch": i + 1,
                 "estimated_shift": shifts[i] if i < len(shifts) else "",
                 "accepted_pseudo_count": pseudo[i] if i < len(pseudo) else "",
                 "target_val_macro_f1": vals[i] if i < len(vals) else ""}
                for i in range(max(len(vals), len(shifts), len(pseudo)))]
    return row, dynamics


def parse_mode_result(log_root, mode, seed=1):
    return _parse_mode(log_root, mode, seed)[0]


def summarize(log_root, output_dir, modes=MODES, seed=1):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    parsed = [_parse_mode(log_root, mode, seed) for mode in modes]
    rows = [item[0] for item in parsed]
    with (output_dir / "mode_sweep_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    dyn_fields = ("mode", "epoch", "estimated_shift", "accepted_pseudo_count", "target_val_macro_f1")
    with (output_dir / "mode_sweep_dynamics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=dyn_fields); writer.writeheader()
        for _, dynamics in parsed: writer.writerows(dynamics)
    lines = ["# FourierRecon AT1→DK1 mode sweep", "",
             "Original TimeMatch: 0.8437; previous FourierRecon-13: 0.6972.", "",
             "| Mode | Source F1 | S→T F1 | Initial shift | Best val | Best epoch | Test F1 | Δ Original | Status |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|:---|"]
    for row in rows:
        lines.append("| {mode} | {source_test_macro_f1} | {source_on_target_macro_f1} | {initial_shift} | {best_val_macro_f1} | {best_epoch} | {test_macro_f1} | {delta_original} | {status} |".format(**row))
    complete = [row for row in rows if row["status"] == "SUCCESS"]
    by_val = sorted(complete, key=lambda row: row["best_val_macro_f1"], reverse=True)
    by_test = sorted(complete, key=lambda row: row["test_macro_f1"], reverse=True)
    lines += ["", "## Ranking by best validation Macro-F1", "",
              "| Rank | Mode | Best val Macro-F1 |", "|---:|---:|---:|"]
    lines += [f"| {rank} | {row['mode']} | {row['best_val_macro_f1']} |"
              for rank, row in enumerate(by_val, 1)]
    lines += ["", "## Ranking by test Macro-F1 (diagnostic only)", "",
              "| Rank | Mode | Test Macro-F1 |", "|---:|---:|---:|"]
    lines += [f"| {rank} | {row['mode']} | {row['test_macro_f1']} |"
              for rank, row in enumerate(by_test, 1)]
    lines += ["", "Select future settings by best validation Macro-F1, not test F1 alone."]
    (output_dir / "mode_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if all(row["status"] == "SUCCESS" for row in rows) else 1


def emit_source_on_target(log_path, mode, seed=1):
    result = _test(_read(Path(log_path)), f"fourier_recon_m{mode:02d}_AT1_on_DK1_seed{seed}")
    if result is None:
        raise SystemExit(f"ERROR: missing source-on-target result in {log_path}")
    print(f"SOURCE_ON_TARGET|source=AT1|target=DK1|mode={mode}|accuracy={result[0]:.4f}|macro_f1={result[1]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--emit-source-on-target", type=Path)
    parser.add_argument("--mode", type=int, choices=MODES)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.emit_source_on_target:
        if args.mode is None: parser.error("--mode is required")
        emit_source_on_target(args.emit_source_on_target, args.mode, args.seed); return
    if args.log_root is None or args.output_dir is None: parser.error("summary paths are required")
    raise SystemExit(summarize(args.log_root, args.output_dir, seed=args.seed))


if __name__ == "__main__":
    main()
