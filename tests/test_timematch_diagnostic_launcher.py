"""Offline launcher checks: /bin/echo replaces all Python invocations."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'closedset_scripts/run_timematch_diagnostic_4tasks_4gpu.sh'
TASKS = ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1')
TILES = ('33UVP', '30TXT', '31TCJ', '32VNH')


def bash_bin():
    if os.name == 'nt':
        git = shutil.which('git')
        candidate = Path(git).parents[1] / 'bin/bash.exe' if git else None
        if candidate and candidate.is_file():
            return str(candidate)
    executable = shutil.which('bash')
    if not executable:
        pytest.skip('Bash not available')
    return executable


def setup_run(tmp_path):
    (tmp_path / 'closedset_scripts').mkdir()
    shutil.copyfile(SCRIPT, tmp_path / 'closedset_scripts' / SCRIPT.name)
    (tmp_path / 'data').mkdir()
    for tile in TILES:
        fold = tmp_path / f'outputs/closedset_source_{tile}_seed1/fold_0'
        fold.mkdir(parents=True)
        (fold / 'model.pt').touch()


def run_script(tmp_path, python='/bin/echo', **options):
    env = dict(os.environ, DATA_ROOT='data', PYTHON_BIN=python,
               EXPERIMENT='test_audit', TASKS=' '.join(TASKS), GPUS='0 1 2 3', SEED='1')
    env.update(options)
    return subprocess.run([bash_bin(), f'closedset_scripts/{SCRIPT.name}'],
                          cwd=tmp_path, env=env, capture_output=True, text=True)


def test_bash_syntax():
    subprocess.run([bash_bin(), '-n', str(SCRIPT)], check=True)


def test_four_task_echo_maps_source_final_epoch_and_isolated_logs(tmp_path):
    setup_run(tmp_path)
    result = run_script(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ''
    root = tmp_path / 'logs/test_audit'
    control = (root / 'diagnostic_control.log').read_text()
    assert 'structures=72|contributions=216' in control
    for task, tile in zip(TASKS, TILES):
        command = (root / f'{task}.log').read_text()
        assert f'--experiment_name {task}' in command
        assert f'--weights outputs/closedset_source_{tile}_seed1' in command
        assert '--epochs 20 --steps_per_epoch 500 --evidence_mode off --output_student true' in command
        assert f'--diagnostic_log logs/test_audit/diagnostic_{task}.log' in command
        assert f'--diagnostic_dir outputs/test_audit/diagnostics/{task}' in command
        assert '--output_dir outputs/test_audit/uda' in command
        assert all(f'--experiment_name {other}' not in command for other in TASKS if other != task)


def test_failed_task_returns_nonzero_and_other_tasks_complete(tmp_path):
    setup_run(tmp_path)
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "gpu=$CUDA_VISIBLE_DEVICES $*"\n'
                    'if [[ "$*" == *"--experiment_name FR1_FR2"* ]]; then exit 23; fi\n',
                    encoding='utf-8')
    fake.chmod(0o755)
    result = run_script(tmp_path, python='./fake-python')
    assert result.returncode == 1
    control = (tmp_path / 'logs/test_audit/diagnostic_control.log').read_text()
    assert 'ERROR: task FR1_FR2 failed' in control
    assert 'RUN_VERIFIED' not in control
    for i, task in enumerate(TASKS):
        assert f'gpu={i} ' in (tmp_path / f'logs/test_audit/{task}.log').read_text()
        if task != 'FR1_FR2':
            assert f'TASK_COMPLETE|task={task}' in control


def test_missing_source_or_existing_output_fails_before_training(tmp_path):
    setup_run(tmp_path)
    (tmp_path / 'outputs/closedset_source_32VNH_seed1/fold_0/model.pt').unlink()
    assert run_script(tmp_path).returncode == 1
    assert not (tmp_path / 'logs/test_audit/AT1_DK1.log').exists()


def test_subset_can_use_a_different_gpu(tmp_path):
    setup_run(tmp_path)
    assert run_script(tmp_path, TASKS='AT1_DK1', GPUS='3').returncode == 0
    assert (tmp_path / 'logs/test_audit/AT1_DK1.log').exists()
    assert not (tmp_path / 'logs/test_audit/FR1_FR2.log').exists()
