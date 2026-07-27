from pathlib import Path

import numpy as np

from analysis.structure_da.log_analysis import build_analysis_tables
from analysis.structure_da.log_parser import parse_task_log


def _mini_log(num_classes: int, status: str = "completed") -> str:
    classes = [f"class_{index}" for index in range(num_classes)]
    report = "\n".join(
        f"{name:>16}       0.50      0.40      0.44       {0 if index == 0 else 5}"
        for index, name in enumerate(classes)
    )
    confusion = "\n".join(
        f"{name} [{ ' '.join(str(int(row == col)) for col in range(num_classes)) }]"
        for row, name in enumerate(classes)
    )
    terminal = {
        "completed": "TASK_DONE|gpu=0|source=DK1|target=FR2|seed=1",
        "failed": "TASK_FAILED|gpu=0|source=DK1|target=FR2|seed=1|exit_code=7",
        "incomplete": "",
    }[status]
    test_section = "" if status != "completed" else f"""
Test result for DK1_to_FR2_seed1: accuracy=0.6000, f1=0.5000
                  precision    recall  f1-score   support

{report}

{confusion}
"""
    return f"""TASK_START|gpu=0|source=DK1|target=FR2|seed=1|run=DK1_to_FR2_seed1
DATE_START|2026-07-27T00:00:00+08:00
GIT_HEAD|abc123
Namespace(source='denmark/32VNH/2017', target='france/30TXT/2017', seed=1, epochs=2, batch_size=128, lr=0.001, weight_decay=0.0001, num_pixels=64, steps_per_epoch=None, quality_warmup_steps=None, grl_warmup_steps=None, lambda_qdom=1.0, lambda_qcls=1.0, lambda_div=1.0, lambda_sda=1.0, time_scale=365.0, tau_fast_init=0.05, tau_slow_init=0.2, tau_min=0.0001, delta_tau_min=0.0001, quality_hidden_cap=128, quality_eta=0.1, sda_hidden_dim=128, input_dim=10, with_extra=False)
CLOSED_SET_PROTOCOL|source=denmark/32VNH/2017|target=france/30TXT/2017|num_classes={num_classes}|classes={','.join(classes)}
CLOSED_SET_COUNTS|source_total=10|target_total=20
TRAIN_EPOCH|epoch=1/2|steps=3|total=5.0000|cls=2.0000|qdom=8.0000|qcls=10.0000|div=-8.0000|sda=1.3000|rho=1.000|grl=1.000|lr=1.00e-03
Validation result: loss=0.6000, acc=0.70, f1=0.4000
Validation F1 improved from -inf to 0.4000!
TRAIN_EPOCH|epoch=2/2|steps=3|total=4.0000|cls=1.0000|qdom=7.0000|qcls=9.0000|div=-9.0000|sda=1.2000|rho=1.000|grl=1.000|lr=0.00e+00
Validation result: loss=0.5000, acc=0.80, f1=0.6000
Validation F1 improved from 0.4000 to 0.6000!
{test_section}
{terminal}
"""


def test_parser_handles_completed_log_support_zero_and_confusion(tmp_path: Path):
    path = tmp_path / "DK1_to_FR2_seed1.log"
    path.write_text(_mini_log(10), encoding="utf-8")

    run = parse_task_log(path)

    assert run.status == "completed"
    assert run.num_classes == 10
    assert len(run.epochs) == 2
    assert run.epochs[1].val_macro_f1 == 0.6
    assert run.best_source_val_epoch == 2
    assert run.reported_improvement_epochs == (1, 2)
    assert run.class_metrics[0].support == 0
    assert run.confusion.shape == (10, 10)
    assert np.array_equal(run.confusion, np.eye(10, dtype=np.int64))


def test_parser_supports_twelve_classes_and_terminal_states(tmp_path: Path):
    runs = []
    for status in ("completed", "failed", "incomplete"):
        path = tmp_path / f"{status}.log"
        path.write_text(_mini_log(12, status), encoding="utf-8")
        runs.append(parse_task_log(path))

    assert [run.status for run in runs] == ["completed", "failed", "incomplete"]
    assert all(run.num_classes == 12 for run in runs)


def test_task_statistics_use_only_completed_runs(tmp_path: Path):
    paths = []
    for seed, status in ((1, "completed"), (2, "completed"), (3, "failed")):
        text = _mini_log(10, status).replace("seed=1", f"seed={seed}").replace(
            "seed1", f"seed{seed}"
        )
        if seed == 2:
            text = text.replace("accuracy=0.6000, f1=0.5000", "accuracy=0.8000, f1=0.7000")
        path = tmp_path / f"DK1_to_FR2_seed{seed}.log"
        path.write_text(text, encoding="utf-8")
        paths.append(path)

    tables = build_analysis_tables([parse_task_log(path) for path in paths])
    task = tables["task_summary"].iloc[0]

    assert task["n_completed_seeds"] == 2
    assert task["target_macro_f1_mean"] == 0.6
    assert np.isclose(task["target_macro_f1_std"], 0.1)
