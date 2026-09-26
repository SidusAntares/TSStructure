import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_structure_proto_v2clean_fourier_mode_sweep_4tasks_4gpu_seed1.sh"
MODES = (9, 13, 17, 21)
TASKS = ("AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1")


@pytest.mark.parametrize("num_modes", MODES)
def test_v2clean_structure_model_supports_sweep_mode(num_modes):
    from models.stclassifier import PseStructureProtoLTae

    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=10, shapelet_count=16,
        fourier_num_modes=num_modes, dropout=0,
    )

    assert model.structure_branch.exposer.analyzer.num_modes == num_modes
    assert model.shape_classifier.in_features == 32
    assert model.structure_branch.response_to_query[0].in_features == 32


def test_sweep_launcher_freezes_v2clean_and_isolates_each_mode():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'MODES=(9 13 17 21)' in source
    assert source.count('--fourier_num_modes "$mode"') == 2
    assert '--shapelet-count 16' in source
    assert '--shape-window-scales 24' in source
    assert '--shape-window-stride 8' in source
    assert '--pseudo_threshold 0.9' in source
    assert '--trade_off 2.0' in source
    assert '--shape-class-weight 0.1' in source
    assert '--shapelet-diversity-weight 0.01' in source
    assert '--shape-align-weight 0.05' in source
    assert '--epochs 100' in source
    assert '--epochs 20 --steps_per_epoch 500' in source
    assert 'mode_${mode}/${task}/source' in source
    assert 'mode_${mode}/${task}/uda' in source
    assert 'source_${src_name}_m${mode}_seed1' in source
    assert 'timematch --weights "$source_weights"' in source
    assert '--adaptive-pseudo-selection' not in source
    assert '--oracle-pseudo-labels' not in source
    assert 'method=v2clean_fourier_mode_sweep' in source
    assert 'fourier_mode_sweep_summary.csv' in source
    assert 'fourier_mode_sweep_per_class.csv' in source


@pytest.mark.skipif(os.name == "nt", reason="launcher execution requires a POSIX shell")
def test_sweep_launcher_dry_run_lists_16_unique_gpu_assignments():
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True, capture_output=True, check=True,
    )
    plans = [line for line in result.stdout.splitlines() if line.startswith("SWEEP_PLAN|")]
    assert len(plans) == 16
    assert len(set(plans)) == 16

    parsed = [dict(field.split("=", 1) for field in line.split("|")[1:]) for line in plans]
    assert {(row["task"], int(row["mode"])) for row in parsed} == {
        (task, mode) for task in TASKS for mode in MODES
    }
    assert Counter(row["gpu"] for row in parsed) == Counter({"0": 4, "1": 4, "2": 4, "3": 4})
    assert {(row["task"], row["gpu"]) for row in parsed} == {
        ("AT1_DK1", "0"), ("FR1_FR2", "1"),
        ("FR2_DK1", "2"), ("DK1_AT1", "3"),
    }
