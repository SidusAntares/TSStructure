import os
from pathlib import Path
import shutil
import subprocess

from test_timematch_diagnostic_launcher import bash_bin


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'closedset_scripts/run_timematch_pairwise_logit_diagnostic_4tasks_4gpu.sh'
TASKS = ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1')
TILES = ('33UVP', '30TXT', '31TCJ', '32VNH')
TARGETS = ('32VNH', '31TCJ', '32VNH', '33UVP')


def setup_script(tmp_path):
    (tmp_path / 'closedset_scripts').mkdir()
    shutil.copyfile(SCRIPT, tmp_path / 'closedset_scripts' / SCRIPT.name)
    (tmp_path / 'data').mkdir()
    for tile, target in zip(TILES, TARGETS):
        for path in (f'outputs/timematch_historical_rerun_seed1/source/closedset_source_{tile}_seed1',
                     f'outputs/timematch_historical_rerun_seed1/uda/closedset_timematch_{tile}_to_{target}_seed1'):
            root = tmp_path / path / 'fold_0'
            root.mkdir(parents=True)
            (root / 'model.pt').touch()


def launch(tmp_path, python='/bin/echo', **overrides):
    env = dict(os.environ, DATA_ROOT='data', PYTHON_BIN=python, EXPERIMENT='test_pairwise',
               TASKS=' '.join(TASKS), GPUS='3 1 2 0', STUDENT_SOURCE_SHIFT='')
    env.update(overrides)
    return subprocess.run([bash_bin(), f'closedset_scripts/{SCRIPT.name}'], env=env,
                          cwd=tmp_path, capture_output=True, text=True)


def test_four_read_only_workers_and_isolated_logs(tmp_path):
    setup_script(tmp_path)
    result = launch(tmp_path)
    assert result.returncode == 0 and result.stdout == ''
    for task in TASKS:
        line = (tmp_path / f'logs/test_pairwise/diagnostic_{task}.log').read_text()
        assert '-m analysis.timematch_pairwise_logits' in line
        assert f'--task {task} --output outputs/test_pairwise/{task}' in line
        assert 'train.py' not in line
    control = (tmp_path / 'logs/test_pairwise/diagnostic_control.log').read_text()
    assert 'cases=16|complete_logits=true' in control


def test_failed_worker_is_reported_without_mixing_other_gpus(tmp_path):
    setup_script(tmp_path)
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "gpu=$CUDA_VISIBLE_DEVICES $*"\n'
                    'if [[ "$*" == *"--task FR1_FR2 "* ]]; then exit 23; fi\n', encoding='utf-8')
    fake.chmod(0o755)
    assert launch(tmp_path, python='./fake-python').returncode == 1
    for gpu, task in zip((3, 1, 2, 0), TASKS):
        assert f'gpu={gpu}' in (tmp_path / f'logs/test_pairwise/diagnostic_{task}.log').read_text()
    control = (tmp_path / 'logs/test_pairwise/diagnostic_control.log').read_text()
    assert 'FAILED|task=FR1_FR2' in control and 'RUN_COMPLETE' not in control


def test_missing_checkpoint_and_existing_output_are_errors(tmp_path):
    setup_script(tmp_path)
    checkpoint = tmp_path / 'outputs/timematch_historical_rerun_seed1/source/closedset_source_33UVP_seed1/fold_0/model.pt'
    checkpoint.unlink()
    assert launch(tmp_path).returncode == 1
    checkpoint.touch()
    (tmp_path / 'outputs/test_pairwise').mkdir()
    assert launch(tmp_path).returncode == 1
    assert not (tmp_path / 'logs/test_pairwise/diagnostic_AT1_DK1.log').exists()


def test_manual_shift_is_allowed_only_for_single_task_and_keeps_zero(tmp_path):
    setup_script(tmp_path)
    assert launch(tmp_path, STUDENT_SOURCE_SHIFT='0').returncode == 1
    assert launch(tmp_path, TASKS='AT1_DK1', GPUS='3', STUDENT_SOURCE_SHIFT='0').returncode == 0
    assert '--student-source-shift 0' in (tmp_path / 'logs/test_pairwise/diagnostic_AT1_DK1.log').read_text()


def test_bash_syntax():
    subprocess.run([bash_bin(), '-n', str(SCRIPT)], check=True)
