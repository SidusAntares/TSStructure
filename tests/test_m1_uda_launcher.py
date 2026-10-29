import os
from pathlib import Path
import shutil
import subprocess

from test_timematch_diagnostic_launcher import bash_bin
from test_closedset_historical_rerun import arguments


SCRIPT = Path(__file__).resolve().parents[1] / 'closedset_scripts/run_m1_timematch_4tasks_4gpu.sh'
TASKS = ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1')
SOURCES = ('austria_33UVP_2017', 'france_30TXT_2017', 'france_31TCJ_2017', 'denmark_32VNH_2017')
TILES = ('33UVP', '30TXT', '31TCJ', '32VNH')


def setup_run(tmp_path):
    (tmp_path / 'closedset_scripts').mkdir()
    shutil.copyfile(SCRIPT, tmp_path / 'closedset_scripts' / SCRIPT.name)
    (tmp_path / 'data').mkdir()
    for tile, source in zip(TILES, SOURCES):
        for mode in ('m0', 'm1'):
            root = tmp_path / f'outputs/{mode}_soft_top50_seed1/source/{mode}_source_{tile}_seed1'
            (root / 'fold_0').mkdir(parents=True)
            for path in ('train_config.json', 'fold_0/model.pt', f'fold_0/test_metrics_{source}.json'):
                (root / path).write_text('{}')


def launch(tmp_path, python='/bin/echo', **kwargs):
    env = dict(os.environ, PYTHON_BIN=python, DATA_ROOT='data', MODE='M1', EXPERIMENT='',
               TASKS=' '.join(TASKS), GPUS='3 1 2 0', SEED='1', NUM_WORKERS='8')
    env.update(kwargs)
    return subprocess.run([bash_bin(), f'closedset_scripts/{SCRIPT.name}'], cwd=tmp_path,
                          env=env, capture_output=True, text=True)


def test_m1_adaptation_is_original_20_epoch_without_true_target_monitoring(tmp_path):
    setup_run(tmp_path)
    assert launch(tmp_path).returncode == 0
    for task, tile in zip(TASKS, TILES):
        config = arguments((tmp_path / f'logs/m1_soft_top50_seed1/{task}_uda.log').read_text())
        assert config['weights'] == f'outputs/m1_soft_top50_seed1/source/m1_source_{tile}_seed1'
        assert config['evidence_mode'] == 'off'
        assert not config['run_validation'] and not config['log_target_labels']
        assert config['strict_final_student'] and config['output_student']
        assert config['epochs'] == 20 and config['steps_per_epoch'] == 500
        assert config['num_workers'] == 8 and config['seed'] == 1
        assert config['student_target_aug'] == 'strong'
    control = (tmp_path / 'logs/m1_soft_top50_seed1/uda_launcher.log').read_text()
    assert '- outputs/m1_soft_top50_seed1/uda/AT1_DK1/fold_0' in control
    assert all(f'COMPLETED|task={task}' in control for task in TASKS)


def test_m0_adaptation_uses_same_legal_protocol_and_isolated_source(tmp_path):
    setup_run(tmp_path)
    assert launch(tmp_path, MODE='M0').returncode == 0
    config = arguments((tmp_path / 'logs/m0_soft_top50_seed1/AT1_DK1_uda.log').read_text())
    assert config['weights'] == 'outputs/m0_soft_top50_seed1/source/m0_source_33UVP_seed1'
    assert not config['run_validation'] and config['strict_final_student']


def test_missing_source_and_worker_failure_return_nonzero(tmp_path):
    setup_run(tmp_path)
    missing = tmp_path / 'outputs/m1_soft_top50_seed1/source/m1_source_33UVP_seed1/fold_0/model.pt'
    missing.unlink()
    assert launch(tmp_path).returncode == 1
    missing.touch()
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "$*"\n'
                    'if [[ "$*" == *"--experiment_name FR1_FR2 "* ]]; then exit 23; fi\n', encoding='utf-8')
    fake.chmod(0o755)
    assert launch(tmp_path, python='./fake-python').returncode == 1
    assert 'FAILED|task=FR1_FR2' in (tmp_path / 'logs/m1_soft_top50_seed1/uda_launcher.log').read_text()


def test_bash_syntax():
    subprocess.run([bash_bin(), '-n', str(SCRIPT)], check=True)


def test_top25_weak_student_reuses_sources_in_separate_experiment(tmp_path):
    setup_run(tmp_path)
    original = tmp_path / 'outputs/m1_soft_top50_seed1'
    top25 = tmp_path / 'outputs/m1_soft_top25_seed1'
    original.rename(top25)
    result = launch(tmp_path, EXPERIMENT='m1_soft_top25_weak_student_seed1',
                    SOURCE_ROOT='outputs/m1_soft_top25_seed1/source', STUDENT_TARGET_AUG='weak')
    assert result.returncode == 0, result.stderr
    for task, tile in zip(TASKS, TILES):
        path = tmp_path / f'logs/m1_soft_top25_weak_student_seed1/{task}_uda.log'
        config = arguments(path.read_text())
        assert config['student_target_aug'] == 'weak'
        assert config['weights'] == f'outputs/m1_soft_top25_seed1/source/m1_source_{tile}_seed1'
        assert config['output_dir'] == 'outputs/m1_soft_top25_weak_student_seed1/uda'
        assert config['epochs'] == 20 and config['steps_per_epoch'] == 500
        assert config['strict_final_student'] and config['evidence_mode'] == 'off'
    assert not (top25 / 'uda').exists()
    assert launch(tmp_path, EXPERIMENT='invalid', STUDENT_TARGET_AUG='invalid').returncode == 1
