import os
from pathlib import Path
import shutil
import subprocess

from test_timematch_diagnostic_launcher import bash_bin
from test_closedset_historical_rerun import arguments


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'closedset_scripts/run_head_evidence_top30_gate05_4tasks_4gpu.sh'
TASKS = ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1')
TILES = ('33UVP', '30TXT', '31TCJ', '32VNH')


def setup_script(tmp_path):
    assert SCRIPT.is_file()
    (tmp_path / 'closedset_scripts').mkdir()
    shutil.copyfile(SCRIPT, tmp_path / 'closedset_scripts' / SCRIPT.name)
    (tmp_path / 'data').mkdir()
    sources = ('austria_33UVP_2017', 'france_30TXT_2017', 'france_31TCJ_2017', 'denmark_32VNH_2017')
    for tile, dataset in zip(TILES, sources):
        source = tmp_path / f'outputs/timematch_historical_rerun_seed1/source/closedset_source_{tile}_seed1'
        (source / 'fold_0').mkdir(parents=True)
        for path in ('fold_0/model.pt', 'train_config.json', 'closed_set_protocol.json',
                     f'fold_0/test_metrics_{dataset}.json'):
            (source / path).write_text('{}')


def run(tmp_path, python='/bin/echo', **overrides):
    env = dict(os.environ, DATA_ROOT='data', PYTHON_BIN=python, EXPERIMENT='test_top30',
               TASKS=' '.join(TASKS), GPUS='3 1 2 0', SEED='1', NUM_WORKERS='8',
               SOURCE_ROOT='outputs/timematch_historical_rerun_seed1/source')
    env.update(overrides)
    return subprocess.run([bash_bin(), f'closedset_scripts/{SCRIPT.name}'], cwd=tmp_path,
                          env=env, capture_output=True, text=True)


def test_eight_uda_commands_share_fresh_source_and_keep_original_losses(tmp_path):
    setup_script(tmp_path)
    result = run(tmp_path)
    assert result.returncode == 0, result.stderr
    for task, tile in zip(TASKS, TILES):
        for suffix, mode in (('A', 'positive'), ('B', 'positive_rank')):
            line = (tmp_path / f'logs/test_top30/{task}_{suffix}.log').read_text()
            config = arguments(line)
            assert config['method'] == 'timematch'
            assert config['weights'] == f'outputs/timematch_historical_rerun_seed1/source/closedset_source_{tile}_seed1'
            assert config['evidence_mode'] == mode
            assert config['evidence_structure'] == 'top30'
            assert config['evidence_top_fraction'] == .3
            assert config['evidence_intervention'] == 'gate' and config['evidence_gate'] == .5
            assert config['seed'] == 1 and config['num_workers'] == 8
            assert config['epochs'] == 20 and config['steps_per_epoch'] == 500
            assert config['evidence_positive_weight'] == config['evidence_rank_weight'] == .1
            assert config['evidence_warmup_steps'] == 100
            assert config['diagnostic_dir'] is None
    assert len(list((tmp_path / 'logs/test_top30').glob('*_[AB].log'))) == 8
    assert not list((tmp_path / 'logs/test_top30').glob('*source.log'))
    assert not (tmp_path / 'outputs/test_top30/source').exists()


def test_failure_in_a_stops_its_b_but_other_gpus_complete(tmp_path):
    setup_script(tmp_path)
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "gpu=$CUDA_VISIBLE_DEVICES $*"\n'
                    'if [[ "$*" == *"--experiment_name FR1_FR2_A"* ]]; then exit 23; fi\n', encoding='utf-8')
    fake.chmod(0o755)
    assert run(tmp_path, python='./fake-python').returncode == 1
    assert not (tmp_path / 'logs/test_top30/FR1_FR2_B.log').exists()
    for gpu, task in zip((3, 1, 2, 0), TASKS):
        if task != 'FR1_FR2':
            assert f'gpu={gpu}' in (tmp_path / f'logs/test_top30/{task}_B.log').read_text()


def test_missing_source_and_existing_output_fail_before_training(tmp_path):
    setup_script(tmp_path)
    checkpoint = tmp_path / 'outputs/timematch_historical_rerun_seed1/source/closedset_source_33UVP_seed1/fold_0/model.pt'
    checkpoint.unlink()
    assert run(tmp_path).returncode == 1
    assert not (tmp_path / 'logs/test_top30/AT1_DK1_A.log').exists()


def test_bash_syntax():
    assert SCRIPT.is_file()
    subprocess.run([bash_bin(), '-n', str(SCRIPT)], check=True)
