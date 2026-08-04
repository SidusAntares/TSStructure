from pathlib import Path
import re
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_at1_dk1.sh"
FORMAL_SCRIPT = REPO_ROOT / "scripts" / "run_structure_da_12tasks_4gpu_3seeds.sh"
PILOT_SCRIPT = REPO_ROOT / "scripts" / "run_structure_da_pilot4_4gpu.sh"
ENV_CHECK_SCRIPT = REPO_ROOT / "scripts" / "check_server_env.sh"
SERVER_SCRIPTS = [SMOKE_SCRIPT, FORMAL_SCRIPT, PILOT_SCRIPT, ENV_CHECK_SCRIPT]


def _git_command_lines(source: str) -> list[str]:
    command_pattern = re.compile(r"(?<![A-Za-z0-9_])git(?=\s)")
    return [line for line in source.splitlines() if command_pattern.search(line)]


@pytest.mark.parametrize("script", SERVER_SCRIPTS)
def test_structure_da_shell_script_has_valid_bash_syntax(script: Path) -> None:
    relative_script = script.relative_to(REPO_ROOT).as_posix()
    subprocess.run(["bash", "-n", relative_script], cwd=REPO_ROOT, check=True)


@pytest.mark.parametrize("script", SERVER_SCRIPTS)
def test_server_script_does_not_execute_git_commands(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    assert _git_command_lines(source) == []
    for obsolete_marker in (
        "origin/main", "dirty_worktree", "head_mismatch", "GIT_HEAD",
    ):
        assert obsolete_marker not in source


def test_server_environment_check_does_not_access_github() -> None:
    source = ENV_CHECK_SCRIPT.read_text(encoding="utf-8")
    assert "github.com" not in source.casefold()
    assert "origin" not in source.casefold()


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
        "--epochs 100", "--batch_size 128", "--eval_batch_size 128",
        "--num_pixels 64",
        "--lr 1e-3", "--weight_decay 1e-4",
        "--structure_dim 128", "--domain_hidden_dim 128",
        "--grl_warmup_fraction 0.2", "--amp true", "--amp_dtype float16",
        "--lambda_task 1",
        "--lambda_geometry 1", "--lambda_alignment 1",
        "--lambda_structural_cls 1", "--lambda_structural_domain 1",
        "--lambda_component_cls 1", "--lambda_component_domain 1",
          "--time_scale 365", "--tau_fast_init 0.05",
        "--tau_slow_init 0.20", "--tau_min 0.0001",
        "--delta_tau_min 0.0001",
    ):
        assert fragment in source
    for option in (
        "--quality_" + "warmup_steps", "--grl_warmup_steps", "--grl_gamma",
        "--lambda_qdom", "--lambda_qcls", "--lambda_" + "div",
        "--lambda_" + "sda", "--steps_per_epoch", "--batch_size 8",
    ):
        assert option not in source


def test_pilot_script_maps_exactly_four_tasks_to_four_gpus() -> None:
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    assert (
        'EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-structure_eden_pilot4_100ep_v1}"'
        in source
    )
    assert "structure_eden_pilot4_v1" not in source
    for fragment in (
        '"0|AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|1"',
        '"1|AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|2"',
        '"2|AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|3"',
        '"3|DK1|denmark/32VNH/2017|FR2|france/30TXT/2017|1"',
    ):
        assert fragment in source
    assert "FR1" not in source
    assert "france/31TCJ/2017" not in source


def test_pilot_script_has_fixed_parameters_and_runtime_guards() -> None:
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    for fragment in (
        "--epochs 100", "--batch_size 128",
        "--eval_batch_size 128", "--num_pixels 64", "--num_workers 8",
        "--lr 0.001", "--weight_decay 0.0001",
        "--structure_dim 128", "--domain_hidden_dim 128",
        "--grl_warmup_fraction 0.2", "--amp true", "--amp_dtype float16",
        "--lambda_task 1",
        "--lambda_geometry 0.1", "--lambda_alignment 1",
        "--lambda_structural_cls 0.25", "--lambda_structural_domain 0.25",
        "--lambda_component_cls 0.25", "--lambda_component_domain 0.25",
          "--time_scale 365", "--tau_fast_init 0.05",
        "--tau_slow_init 0.20", "--tau_min 0.0001",
        "--delta_tau_min 0.0001", "--closed_set true",
        "--balance-source",
        "--combine_spring_and_winter false", "--progress_bar off",
        "--log_step 25",
    ):
        assert fragment in source
    for forbidden in ("--data_root", "DATA_ROOT=", "nohup"):
        assert forbidden not in source
    for obsolete_parameter in (
        "--epochs 20", "--grl_warmup_max_iters 2500",
        "--grl_warmup_max_iters 10000", "--steps_per_epoch 500",
        "--batch_size 8",
    ):
        assert obsolete_parameter not in source
    for guard in (
        "TASK_START|", "TASK_DONE|", "TASK_FAILED|",
        "EXPERIMENT_SUMMARY|", "completion_file", ".previous_",
        'pids+=("$!")', 'wait "$pid"',
    ):
        assert guard in source


def test_formal_script_keeps_twelve_domain_tasks_and_three_seeds() -> None:
    source = FORMAL_SCRIPT.read_text(encoding="utf-8")
    for dataset in (
        '"denmark/32VNH/2017"',
        '"france/30TXT/2017"',
        '"france/31TCJ/2017"',
        '"austria/33UVP/2017"',
    ):
        assert dataset in source
    assert "SEEDS=(1 2 3)" in source
    assert 'if [[ "$target_index" -eq "$source_index" ]]' in source
    assert "--progress_bar off" in source
    for runtime_behavior in (
        "COMPLETION_FILE", ".previous_", "TASK_FAILED|",
        "EXPERIMENT_SUMMARY|", 'wait "$pid"',
    ):
        assert runtime_behavior in source


def test_benchmark_cli_exposes_batch_and_amp_controls() -> None:
    result = subprocess.run(
        ["python", "scripts/benchmark_structure_da_step.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for option in ("--batch_size", "--amp", "--amp_dtype"):
        assert option in result.stdout
