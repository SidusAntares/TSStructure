from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_at1_dk1.sh"
FORMAL_SCRIPT = REPO_ROOT / "scripts" / "run_structure_da_12tasks_4gpu_3seeds.sh"
PILOT_SCRIPT = REPO_ROOT / "scripts" / "run_structure_da_pilot4_4gpu.sh"


@pytest.mark.parametrize("script", [SMOKE_SCRIPT, FORMAL_SCRIPT, PILOT_SCRIPT])
def test_structure_da_shell_script_has_valid_bash_syntax(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_smoke_script_uses_default_data_root_and_no_legacy_arguments() -> None:
    source = SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert "--data_root" not in source
    assert "DATA_ROOT=" not in source
    for option in (
        "--quality_" + "warmup_steps", "--grl_warmup_steps", "--grl_gamma",
        "--lambda_qdom", "--lambda_qcls", "--lambda_" + "div",
        "--lambda_" + "sda",
    ):
        assert option not in source


def test_formal_script_has_new_group_and_explicit_current_arguments() -> None:
    source = FORMAL_SCRIPT.read_text(encoding="utf-8")
    assert 'EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-structure_eden_full_3seeds_v1}"' in source
    for fragment in (
        "--epochs 100", "--batch_size 128", "--num_pixels 64",
        "--lr 1e-3", "--weight_decay 1e-4",
        "--channel_feature_dim 16", "--pixel_hidden_dim 16",
        "--structure_dim 128", "--domain_hidden_dim 128",
        "--grl_warmup_max_iters 250", "--lambda_task 1",
        "--lambda_geometry 1", "--lambda_alignment 1",
        "--lambda_structural_cls 1", "--lambda_structural_domain 1",
        "--lambda_component_cls 1", "--lambda_component_domain 1",
        "--time_scale 366", "--tau_fast_init 0.05",
        "--tau_slow_init 0.20", "--tau_min 0.0001",
        "--delta_tau_min 0.0001",
    ):
        assert fragment in source
    for option in (
        "--quality_" + "warmup_steps", "--grl_warmup_steps", "--grl_gamma",
        "--lambda_qdom", "--lambda_qcls", "--lambda_" + "div",
        "--lambda_" + "sda",
    ):
        assert option not in source


def test_pilot_script_maps_exactly_four_tasks_to_four_gpus() -> None:
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    assert 'EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-structure_eden_pilot4_v1}"' in source
    for fragment in (
        '"0|AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|1"',
        '"1|AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|2"',
        '"2|AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|3"',
        '"3|DK1|denmark/32VNH/2017|FR2|france/30TXT/2017|1"',
    ):
        assert fragment in source
    assert "FR1" not in source
    assert "france/31TCJ/2017" not in source


def test_pilot_script_has_fixed_parameters_and_safety_guards() -> None:
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    for fragment in (
        "--epochs 20", "--steps_per_epoch 500", "--batch_size 8",
        "--eval_batch_size 128", "--num_pixels 64", "--num_workers 8",
        "--lr 0.001", "--weight_decay 0.0001",
        "--channel_feature_dim 16", "--pixel_hidden_dim 16",
        "--structure_dim 128", "--domain_hidden_dim 128",
        "--grl_warmup_max_iters 2500", "--lambda_task 1",
        "--lambda_geometry 0.1", "--lambda_alignment 1",
        "--lambda_structural_cls 0.25", "--lambda_structural_domain 0.25",
        "--lambda_component_cls 0.25", "--lambda_component_domain 0.25",
        "--time_scale 366", "--tau_fast_init 0.05",
        "--tau_slow_init 0.20", "--tau_min 0.0001",
        "--delta_tau_min 0.0001", "--closed_set true",
        "--combine_spring_and_winter false", "--progress_bar off",
        "--log_step 25",
    ):
        assert fragment in source
    for forbidden in ("--data_root", "DATA_ROOT=", "nohup"):
        assert forbidden not in source
    for guard in (
        "git status --porcelain", "git rev-parse HEAD",
        "git rev-parse origin/main", "TASK_START|", "TASK_DONE|",
        "TASK_FAILED|", "EXPERIMENT_SUMMARY|",
    ):
        assert guard in source
