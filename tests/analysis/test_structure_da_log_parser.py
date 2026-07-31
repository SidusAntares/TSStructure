from pathlib import Path

import numpy as np

from analysis.structure_da.log_analysis import analyze_logs, build_analysis_tables
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
Namespace(source='denmark/32VNH/2017', target='france/30TXT/2017', seed=1, epochs=2, batch_size=128, lr=0.001, weight_decay=0.0001, num_pixels=64, steps_per_epoch=None, grl_warmup_max_iters=250, lambda_task=1.0, lambda_geometry=1.0, lambda_alignment=1.0, lambda_structural_cls=1.0, lambda_structural_domain=1.0, lambda_component_cls=1.0, lambda_component_domain=1.0, time_scale=366.0, tau_fast_init=0.05, tau_slow_init=0.2, tau_min=0.0001, delta_tau_min=0.0001, channel_feature_dim=16, pixel_hidden_dim=16, structure_dim=128, domain_hidden_dim=128, input_dim=10, with_extra=False)
CLOSED_SET_PROTOCOL|source=denmark/32VNH/2017|target=france/30TXT/2017|num_classes={num_classes}|classes={','.join(classes)}
CLOSED_SET_COUNTS|source_total=10|target_total=20
TRAIN_STEP|epoch=1/2|step=1/3|total=5.1000|task=2.1000|q_total=1.0000|geometry=0.7000|alignment=1.3000|train_acc=0.6000|domain_acc=0.5000|grl=0.100|lr=1.00e-03
TRAIN_EPOCH|epoch=1/2|steps=3|total=5.0000|task=2.0000|q_total=1.0000|q_struct_cls=0.1000|q_struct_dom=0.2000|q_comp_cls=0.3000|q_comp_dom=0.4000|geometry=0.7000|alignment=1.3000|train_acc=0.6000|domain_acc=0.5000|grl=0.100|lr=1.00e-03
STRUCTURE_EPOCH|epoch=1|tau_fast=0.0500|tau_slow=0.2000|tau_gap=0.1500|energy_T_s=0.6000|energy_D_s=0.3000|energy_R_s=0.1000|energy_T_t=0.5000|energy_D_t=0.3500|energy_R_t=0.1500|reconstruction_s=0.0000|reconstruction_t=0.0000|temporal_T_valid_s=0.9000|temporal_T_valid_t=0.8000|channel_T_valid_s=0.7000|channel_T_valid_t=0.6000|raw_fusion_norm_s=1.1000|raw_fusion_norm_t=1.2000|temporal_fusion_norm_s=0.9000|temporal_fusion_norm_t=1.0000|channel_fusion_norm_s=0.8000|channel_fusion_norm_t=0.8500|channel_T_relation_mass_s=0.7000|channel_T_relation_mass_t=0.6500
QUALITY_EPOCH|epoch=1|alpha_T_s=0.6000|alpha_T_t=0.5000|alpha_D_s=0.3000|alpha_D_t=0.3500|alpha_R_s=0.1000|alpha_R_t=0.1500|beta_T_temporal_s=0.7000|beta_T_temporal_t=0.6500|beta_D_temporal_s=0.6000|beta_D_temporal_t=0.5500|beta_T_channel_s=0.3000|beta_T_channel_t=0.3500|beta_D_channel_s=0.4000|beta_D_channel_t=0.4500
GEOMETRY_EPOCH|epoch=1|T_align=0.1000|T_rough=0.2000|T_unsupported=0.3000|T_center=0.4000|D_align=0.5000|D_rough=0.6000|D_unsupported=0.7000|D_center=0.8000
Validation result: loss=0.6000, acc=0.70, f1=0.4000
Validation F1 improved from -inf to 0.4000!
TRAIN_STEP|epoch=2/2|step=1/3|total=4.1000|task=1.1000|q_total=1.1000|geometry=0.7000|alignment=1.2000|train_acc=0.7000|domain_acc=0.5500|grl=0.200|lr=0.00e+00
TRAIN_EPOCH|epoch=2/2|steps=3|total=4.0000|task=1.0000|q_total=1.1000|q_struct_cls=0.1100|q_struct_dom=0.2100|q_comp_cls=0.3100|q_comp_dom=0.4100|geometry=0.7000|alignment=1.2000|train_acc=0.7000|domain_acc=0.5500|grl=0.200|lr=0.00e+00
STRUCTURE_EPOCH|epoch=2|tau_fast=0.0510|tau_slow=0.2010|tau_gap=0.1500|energy_T_s=0.5500|energy_D_s=0.3500|energy_R_s=0.1000|energy_T_t=0.4800|energy_D_t=0.3600|energy_R_t=0.1600|reconstruction_s=0.0000|reconstruction_t=0.0000|temporal_T_valid_s=0.9100|temporal_T_valid_t=0.8100|channel_T_valid_s=0.7100|channel_T_valid_t=0.6100|raw_fusion_norm_s=1.1100|raw_fusion_norm_t=1.2100|temporal_fusion_norm_s=0.9100|temporal_fusion_norm_t=1.0100|channel_fusion_norm_s=0.8100|channel_fusion_norm_t=0.8600|channel_T_relation_mass_s=0.7100|channel_T_relation_mass_t=0.6600
QUALITY_EPOCH|epoch=2|alpha_T_s=0.5500|alpha_T_t=0.4800|alpha_D_s=0.3500|alpha_D_t=0.3600|alpha_R_s=0.1000|alpha_R_t=0.1600|beta_T_temporal_s=0.6500|beta_T_temporal_t=0.6200|beta_D_temporal_s=0.5500|beta_D_temporal_t=0.5200|beta_T_channel_s=0.3500|beta_T_channel_t=0.3800|beta_D_channel_s=0.4500|beta_D_channel_t=0.4800
GEOMETRY_EPOCH|epoch=2|T_align=0.1100|T_rough=0.2100|T_unsupported=0.3100|T_center=0.4100|D_align=0.5100|D_rough=0.6100|D_unsupported=0.7100|D_center=0.8100
Validation result: loss=0.5000, acc=0.80, f1=0.6000
Validation F1 improved from 0.4000 to 0.6000!
{test_section}
{terminal}
"""


def _legacy_log(num_classes: int) -> str:
    epoch = 0
    lines = []
    for line in _mini_log(num_classes).splitlines():
        if line.startswith(("TRAIN_STEP|", "STRUCTURE_EPOCH|", "QUALITY_EPOCH|", "GEOMETRY_EPOCH|")):
            continue
        if line.startswith("TRAIN_EPOCH|"):
            epoch += 1
            lines.append(
                f"TRAIN_EPOCH|epoch={epoch}/2|steps=3|total={6 - epoch:.4f}"
                f"|cls={3 - epoch:.4f}|qdom=8.0000|qcls=10.0000"
                f"|div=-8.0000|sda=1.3000|rho=1.000|grl=1.000"
                f"|lr={1e-3 if epoch == 1 else 0.0:.2e}"
            )
        else:
            lines.append(line)
    return "\n".join(lines)


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
    assert len(run.steps) == 2
    assert len(run.structure_diagnostics) == 2
    assert len(run.quality_diagnostics) == 2
    assert len(run.geometry_diagnostics) == 2


def test_parser_accepts_legacy_train_epoch_without_new_diagnostics(tmp_path: Path):
    path = tmp_path / "DK1_to_FR2_seed1.log"
    path.write_text(_legacy_log(10), encoding="utf-8")

    run = parse_task_log(path)

    assert len(run.epochs) == 2
    assert run.epochs[0].task == 2.0
    assert np.isnan(run.epochs[0].quality)
    assert run.steps == []
    assert run.structure_diagnostics == []


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


def test_new_diagnostics_create_normalized_tables_and_csv_files(
    tmp_path: Path, monkeypatch
):
    experiment_dir = tmp_path / "logs"
    experiment_dir.mkdir()
    (experiment_dir / "DK1_to_FR2_seed1.log").write_text(
        _mini_log(10), encoding="utf-8"
    )
    monkeypatch.setattr(
        "analysis.structure_da.plotting.plot_log_diagnostics",
        lambda *args, **kwargs: None,
    )

    result = analyze_logs(experiment_dir, tmp_path / "analysis")

    expected = {
        "step_history",
        "decomposition_diagnostics",
        "structure_diagnostics",
        "quality_diagnostics",
        "geometry_diagnostics",
    }
    assert expected <= result["tables"].keys()
    for name in expected:
        assert not result["tables"][name].empty
        assert (tmp_path / "analysis" / "tables" / f"{name}.csv").is_file()
