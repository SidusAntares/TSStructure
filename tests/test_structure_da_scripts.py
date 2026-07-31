from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_at1_dk1.sh"
FORMAL_SCRIPT = REPO_ROOT / "scripts" / "run_structure_da_12tasks_4gpu_3seeds.sh"


@pytest.mark.parametrize("script", [SMOKE_SCRIPT, FORMAL_SCRIPT])
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
