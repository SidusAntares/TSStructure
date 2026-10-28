"""Exercise the real Source CLI without importing optional training dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from test_timematch_diagnostic_launcher import bash_bin


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'closedset_scripts/run_m1_soft_top50_4tasks_4gpu.sh'
ENTRY = ROOT / 'train_m1_source.py'
TASKS = ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1')
TILES = ('austria/33UVP/2017', 'france/30TXT/2017',
         'france/31TCJ/2017', 'denmark/32VNH/2017')
spec = importlib.util.spec_from_file_location('m1_source_cli', ENTRY)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def setup_run(tmp_path):
    (tmp_path / 'closedset_scripts').mkdir()
    shutil.copyfile(SCRIPT, tmp_path / 'closedset_scripts' / SCRIPT.name)
    (tmp_path / 'data').mkdir()


def launch(tmp_path, python='/bin/echo', **overrides):
    env = dict(os.environ, PYTHON_BIN=python, DATA_ROOT='data',
               MODE='M1', EXPERIMENT='test_m1', TASKS=' '.join(TASKS),
               GPUS='3 1 2 0', SEED='1', NUM_WORKERS='8')
    env.update(overrides)
    return subprocess.run([bash_bin(), f'closedset_scripts/{SCRIPT.name}'],
                          cwd=tmp_path, env=env, capture_output=True, text=True)


def test_echo_commands_parse_real_cli_and_match_source_protocol(tmp_path):
    setup_run(tmp_path)
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ''
    for task, source in zip(TASKS, TILES):
        line = (tmp_path / f'logs/test_m1/{task}_source.log').read_text()
        args = shlex.split(line)
        assert args[:2] == ['-u', 'train_m1_source.py']
        cfg = cli.build_config(args[2:])
        assert cfg.source == cfg.target == source
        assert cfg.m1 and cfg.method is None and not cfg.train_on_target
        assert (cfg.seed, cfg.num_workers, cfg.epochs, cfg.batch_size) == (1, 8, 100, 128)
        assert (cfg.lr, cfg.weight_decay, cfg.num_pixels, cfg.seq_length) == (.001, .0001, 64, 30)
        assert cfg.focal_loss_gamma == 1 and cfg.closed_set and not cfg.with_shift_aug
        assert cfg.m1_rho == .5 and cfg.m1_lambda_keep == 1 and cfg.m1_lambda_budget == .1
        assert cfg.m1_tau_start == .5 and cfg.m1_tau_end == .05 and cfg.m1_warmup_epochs == 5
        assert Path(cfg.output_dir).as_posix() == f'outputs/test_m1/source/m1_source_{source.split("/")[1]}_seed1'
        assert Path(cfg.tensorboard_log_dir).as_posix().startswith('outputs/test_m1/tensorboard/')
        assert '--weights' not in line and 'timematch' not in line
    control = (tmp_path / 'logs/test_m1/launcher.log').read_text()
    for gpu, task in zip((3, 1, 2, 0), TASKS):
        assert f'gpu={gpu}|task={task}|mode=M1' in control
        assert f'COMPLETED|task={task}' in control


def test_m0_matched_control_has_separate_default_outputs(tmp_path):
    setup_run(tmp_path)
    assert launch(tmp_path, MODE='M0', EXPERIMENT='').returncode == 0
    line = (tmp_path / 'logs/m0_soft_top50_seed1/AT1_DK1_source.log').read_text()
    cfg = cli.build_config(shlex.split(line)[2:])
    assert not cfg.m1 and cfg.mode == 'M0'
    assert Path(cfg.output_dir).as_posix() == 'outputs/m0_soft_top50_seed1/source/m0_source_33UVP_seed1'


def test_source_failure_propagates_and_preserves_gpu_isolation(tmp_path):
    setup_run(tmp_path)
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "gpu=$CUDA_VISIBLE_DEVICES $*"\n'
                    'if [[ "$CUDA_VISIBLE_DEVICES" == 1 ]]; then exit 23; fi\n', encoding='utf-8')
    fake.chmod(0o755)
    assert launch(tmp_path, python='./fake-python').returncode == 1
    for gpu, task in zip((3, 1, 2, 0), TASKS):
        assert f'gpu={gpu}' in (tmp_path / f'logs/test_m1/{task}_source.log').read_text()
    assert 'FAILED|task=FR1_FR2' in (tmp_path / 'logs/test_m1/launcher.log').read_text()


@pytest.mark.parametrize('overrides', [dict(SEED='2'), dict(NUM_WORKERS='0'),
                                      dict(GPUS='3 3 2 0'), dict(TASKS='UNKNOWN'),
                                      dict(TASKS='AT1_DK1 AT1_DK1'), dict(MODE='bad')])
def test_invalid_launcher_settings_fail_before_workers(tmp_path, overrides):
    setup_run(tmp_path)
    assert launch(tmp_path, **overrides).returncode != 0
    assert not list((tmp_path / 'logs').glob('**/*_source.log'))


def test_existing_experiment_refused(tmp_path):
    setup_run(tmp_path)
    (tmp_path / 'outputs/test_m1').mkdir(parents=True)
    assert launch(tmp_path).returncode != 0
    assert not (tmp_path / 'logs/test_m1/AT1_DK1_source.log').exists()


@pytest.mark.parametrize('extra', [['--target', TILES[1]], ['--weights', 'old.pt'],
                                  ['--seed', '2'], ['--num_workers', '4'],
                                  ['--m1_rho', '0'], ['--m1_tau_end', 'nan'],
                                  ['--with_shift_aug', 'true']])
def test_invalid_source_configuration_rejected(extra):
    with pytest.raises(SystemExit):
        cli.build_config(['--source', TILES[0], *extra])


def test_help_and_dry_run_work_without_training_import(tmp_path):
    result = subprocess.run([sys.executable, str(ENTRY), '--help'], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0 and '--m1_rho' in result.stdout
    result = subprocess.run([sys.executable, str(ENTRY), '--source', TILES[0], '--dry_run'],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)
    assert cfg['m1'] and cfg['source'] == cfg['target']
    assert not (tmp_path / 'outputs').exists()


def test_entry_refuses_existing_output_before_import(tmp_path):
    parent = tmp_path / 'source'
    (parent / 'existing').mkdir(parents=True)
    with pytest.raises(FileExistsError):
        cli.main(['--source', TILES[0], '--output_dir', str(parent),
                  '--experiment_name', 'existing'])


def test_fresh_entry_writes_config_and_provenance_before_training(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []

    def fake_train(cfg):
        output = Path(cfg.output_dir)
        assert (output / 'fold_0').is_dir()
        saved = json.loads((output / 'train_config.json').read_text())
        manifest = json.loads((output / 'code_provenance.json').read_text())
        assert saved['m1'] and saved['method'] is None
        assert 'weights' not in saved
        assert saved['code_revision'] == manifest['code_revision'] == 'reviewed-revision'
        calls.append(cfg)

    monkeypatch.setenv('CODE_REVISION', 'reviewed-revision')
    monkeypatch.setitem(sys.modules, 'train', SimpleNamespace(main=fake_train))
    cli.main(['--source', TILES[0], '--output_dir', str(tmp_path / 'source')])
    assert len(calls) == 1


def test_manifest_hashes_actual_source_and_records_fixed_base_revision():
    import hashlib
    manifest = cli.source_provenance()
    assert manifest['base_revision'] == '8e4594de40581755db1d2aa6e4633273b17a9928'
    assert manifest['sha256']['train_m1_source.py'] == hashlib.sha256(ENTRY.read_bytes()).hexdigest()
    assert 'models/stclassifier.py' in manifest['sha256']


def test_bash_syntax():
    subprocess.run([bash_bin(), '-n', str(SCRIPT)], check=True)
