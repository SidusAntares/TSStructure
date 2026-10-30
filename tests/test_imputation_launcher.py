"""CPU-only checks for the independent imputation experiment."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'closedset_scripts/run_imputed_timematch_4tasks_4gpu.sh'
TASKS = ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1')
TILES = ('33UVP', '30TXT', '31TCJ', '32VNH')


def bash_bin():
    if os.name == 'nt':
        git = shutil.which('git')
        candidate = Path(git).parents[1] / 'bin/bash.exe' if git else None
        if candidate and candidate.is_file():
            return str(candidate)
    return shutil.which('bash') or pytest.skip('Bash unavailable')


def setup_run(tmp_path):
    (tmp_path / 'closedset_scripts').mkdir()
    shutil.copyfile(SCRIPT, tmp_path / 'closedset_scripts' / SCRIPT.name)
    (tmp_path / 'data').mkdir()
    for tile in TILES:
        fold = tmp_path / f'outputs/timematch_historical_rerun_seed1/source/closedset_source_{tile}_seed1/fold_0'
        fold.mkdir(parents=True)
        (fold / 'model.pt').touch()


def run_script(tmp_path, **options):
    env = dict(os.environ, DATA_ROOT='data', PYTHON_BIN='/bin/echo', EXPERIMENT='test_imputation')
    env.update(options)
    return subprocess.run([bash_bin(), f'closedset_scripts/{SCRIPT.name}'], cwd=tmp_path,
                          env=env, capture_output=True, text=True)


def test_four_tasks_chain_pretrain_then_a_then_b_from_same_checkpoints(tmp_path):
    setup_run(tmp_path)
    result = run_script(tmp_path)
    assert result.returncode == 0, result.stderr
    for task, tile in zip(TASKS, TILES):
        logs = [(tmp_path / f'logs/test_imputation/{task}_{stage}.log').read_text()
                for stage in ('imputer', 'A', 'B')]
        source = f'outputs/timematch_historical_rerun_seed1/source/closedset_source_{tile}_seed1'
        checkpoint = f'outputs/test_imputation/imputer/{task}/fold_0/imputer_epoch20.pt'
        assert '--stage pretrain' in logs[0]
        for variant, command in zip(('A', 'B'), logs[1:]):
            assert f'--stage uda --variant {variant}' in command
            assert f'--imputer_checkpoint {checkpoint}' in command
            assert f'--output_dir outputs/test_imputation/uda_{variant}/{task}' in command
            assert '--steps_per_epoch 500' in command
        for command in logs:
            assert f'--weights {source}' in command
            assert '--epochs 20' in command
            assert '--seed 1 --num_workers 8' in command
    launcher = (tmp_path / 'logs/test_imputation/launcher.log').read_text()
    for gpu, task in zip(('3', '1', '2', '0'), TASKS):
        assert f'TASK_START|gpu={gpu}|task={task}' in launcher


def test_missing_source_fails_before_any_task(tmp_path):
    setup_run(tmp_path)
    (tmp_path / f'outputs/timematch_historical_rerun_seed1/source/closedset_source_{TILES[-1]}_seed1/fold_0/model.pt').unlink()
    assert run_script(tmp_path).returncode != 0
    assert not (tmp_path / 'logs/test_imputation/AT1_DK1_imputer.log').exists()


def test_failure_stops_task_chain_and_propagates_status(tmp_path):
    setup_run(tmp_path)
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "$*"\nif [[ "$*" == *"--experiment_name FR1_FR2"* && "$*" == *"--stage pretrain"* ]]; then exit 23; fi\n')
    fake.chmod(0o755)
    assert run_script(tmp_path, PYTHON_BIN='./fake-python').returncode != 0
    assert not (tmp_path / 'logs/test_imputation/FR1_FR2_A.log').exists()
    assert (tmp_path / 'logs/test_imputation/AT1_DK1_B.log').is_file()


def test_existing_output_is_preserved(tmp_path):
    setup_run(tmp_path)
    (tmp_path / 'outputs/test_imputation').mkdir()
    assert run_script(tmp_path).returncode != 0
    assert not (tmp_path / 'logs/test_imputation/AT1_DK1_imputer.log').exists()


def test_dry_run_is_lightweight_and_does_not_allocate(tmp_path):
    output = tmp_path / 'absent'
    result = subprocess.run([sys.executable, str(ROOT / 'train_imputed_timematch.py'),
                             '--stage', 'uda', '--variant', 'B', '--source', 'austria/33UVP/2017',
                             '--target', 'denmark/32VNH/2017', '--weights', 'source-leaf',
                             '--imputer_checkpoint', 'imputer_epoch20.pt', '--output_dir', str(output),
                             '--dry_run'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)
    assert cfg['output_dir'] == str(output)
    assert cfg['mode'] is None and cfg['m1'] is False
    assert cfg['run_validation'] is False and cfg['log_target_labels'] is False
    assert cfg['strict_final_student'] is True
    assert cfg['lr'] == cfg['imputer_lr'] == 1e-4
    assert cfg['reconstruction_weight'] == .1
    assert not output.exists()


def test_bash_syntax():
    subprocess.run([bash_bin(), '-n', str(SCRIPT)], check=True)


def load_entry():
    spec = importlib.util.spec_from_file_location('imputation_entry', ROOT / 'train_imputed_timematch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_config():
    return dict(source='austria/33UVP/2017', target='austria/33UVP/2017', seed=1,
                num_workers=8, epochs=100, model='pseltae', closed_set=True,
                num_folds=1, num_blocks=100, val_ratio=.1, test_ratio=.2,
                sample_pixels_val=True, batch_size=128, seq_length=30, num_pixels=64,
                input_dim=10, combine_spring_and_winter=False, with_extra=False,
                with_shift_aug=False, lr=.001, weight_decay=.0001,
                focal_loss_gamma=1., shift_aug_p=1., max_shift_aug=60)


@pytest.mark.parametrize('override', [dict(mode='M1'), dict(m1=True), dict(method='timematch'),
                                     dict(lr=.0001), dict(epochs=20), dict(seed=2)])
def test_original_source_provenance_rejects_other_training(tmp_path, override):
    module = load_entry()
    leaf = tmp_path / 'source'
    (leaf / 'fold_0').mkdir(parents=True)
    (leaf / 'fold_0/model.pt').write_bytes(b'source')
    (leaf / 'train_config.json').write_text(json.dumps(dict(source_config(), **override)))
    cfg = module.build_config(['--stage', 'pretrain', '--source', 'austria/33UVP/2017',
                               '--target', 'denmark/32VNH/2017', '--weights', str(leaf),
                               '--output_dir', str(tmp_path / 'output')])
    with pytest.raises(ValueError):
        module.validate_source(cfg)


def test_dispatch_writes_provenance_and_refuses_existing_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = load_entry()
    leaf = tmp_path / 'source'
    (leaf / 'fold_0').mkdir(parents=True)
    (leaf / 'fold_0/model.pt').write_bytes(b'source')
    (leaf / 'train_config.json').write_text(json.dumps(source_config()))
    received = []
    monkeypatch.setitem(sys.modules, 'utils.imputation_training',
                        SimpleNamespace(pretrain_imputer=received.append, run_imputation_uda=None))
    output = tmp_path / 'output'
    args = ['--stage', 'pretrain', '--source', 'austria/33UVP/2017',
            '--target', 'denmark/32VNH/2017', '--weights', str(leaf), '--output_dir', str(output)]
    module.main(args)
    assert len(received) == 1
    assert (output / 'fold_0').is_dir()
    saved = json.loads((output / 'config.json').read_text())
    assert saved['reconstruction_weight'] == 1.
    assert len(saved['source_checkpoint_sha256']) == 64
    assert 'train_imputed_timematch.py' in json.loads((output / 'code_provenance.json').read_text())['sha256']
    with pytest.raises(FileExistsError):
        module.main(args)
    assert len(received) == 1
