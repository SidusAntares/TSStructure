"""Smoke-only entry and three-GPU queue protocol checks, without dataset access."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_segment_launcher import bash_bin, source_config

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'closedset_scripts/run_segment_smoke_8jobs_3gpu.sh'
TASKS = ('FR1_FR2', 'FR2_DK1', 'DK1_AT1', 'AT1_DK1')
TILES = ('30TXT', '31TCJ', '32VNH', '33UVP')


def entry():
    import importlib.util
    spec = importlib.util.spec_from_file_location('smoke_entry', ROOT / 'train_segment_smoke.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(tmp_path):
    return ['--variant', 'inshape', '--source', 'austria/33UVP/2017',
            '--target', 'denmark/32VNH/2017', '--weights', str(tmp_path / 'source'),
            '--output_dir', str(tmp_path / 'output')]


def write_source(tmp_path, **changes):
    leaf = tmp_path / 'source'
    (leaf / 'fold_0').mkdir(parents=True)
    (leaf / 'fold_0/model.pt').write_bytes(b'original')
    (leaf / 'train_config.json').write_text(json.dumps(dict(source_config(), **changes)))


def test_dry_run_is_stdlib_only_and_has_fixed_probe_protocol(tmp_path):
    script = ('import runpy,sys;sys.argv=' + repr(['train_segment_smoke.py', *args(tmp_path), '--dry_run'])
              + ';runpy.run_path(' + repr(str(ROOT / 'train_segment_smoke.py')) + ',run_name="__main__");'
              + 'assert not any(n in sys.modules for n in ("torch","numpy","utils.segment_smoke","utils.segment_training"))')
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)
    assert cfg['mode'] == cfg['stage'] == 'smoke'
    assert cfg['steps'] == 200 and cfg['epochs'] == cfg['scheduler_epochs'] == 20
    assert cfg['pelt_penalty'] == .8 and cfg['pelt_penalty_mode'] == 'absolute'
    assert cfg['pelt_min_size'] == 3
    assert cfg['probe_every'] == cfg['gradient_every'] == 25
    assert cfg['probe_samples'] == 128 and cfg['health_window'] == cfg['warmup_steps'] == 20
    assert cfg['seed'] == 1 and cfg['num_workers'] == 8 and cfg['batch_size'] == 128
    assert cfg['lr'] == cfg['selector_lr'] == cfg['weight_decay'] == 1e-4
    assert cfg['focal_loss_gamma'] == 1 and cfg['keep_weight'] == 1 and cfg['sparse_weight'] == .01
    assert cfg['evidence_mode'] == 'off' and cfg['structure_protocol']['imputation'] is False
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('flag,value', [('--steps', '0'), ('--steps', '1001'),
    ('--probe_every', '0'), ('--probe_samples', '0'), ('--gradient_every', '0'),
    ('--health_window', '0'), ('--warmup_steps', '-1'), ('--num_workers', '4'),
    ('--epochs', '1'), ('--pelt_penalty', '.1'), ('--pelt_penalty_mode', 'scaled'),
    ('--keep_weight', '.5'), ('--sparse_weight', 'nan')])
def test_protocol_and_probe_limits_are_enforced(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        entry().build_config([*args(tmp_path), flag, value])


def test_original_validation_precedes_output_creation(tmp_path, monkeypatch):
    write_source(tmp_path, mode='M1', m1=True)
    module = entry()
    with pytest.raises(ValueError):
        module.main(args(tmp_path))
    assert not (tmp_path / 'output').exists()


def test_dispatch_provenance_and_refuse_existing_leaf(tmp_path, monkeypatch):
    write_source(tmp_path)
    received = []
    monkeypatch.setitem(sys.modules, 'utils.segment_smoke', SimpleNamespace(run_smoke=received.append))
    module = entry()
    module.main(args(tmp_path))
    assert len(received) == 1
    saved = json.loads((tmp_path / 'output/config.json').read_text())
    assert saved['source_checkpoint_sha256'] == hashlib.sha256(b'original').hexdigest()
    assert Path(saved['fold_dir']) == tmp_path / 'output/fold_0'
    provenance = json.loads((tmp_path / 'output/code_provenance.json').read_text())['sha256']
    assert 'train_segment_smoke.py' in provenance
    assert 'closedset_scripts/run_segment_smoke_8jobs_3gpu.sh' in provenance
    with pytest.raises(FileExistsError):
        module.main(args(tmp_path))
    assert len(received) == 1


def setup_run(tmp_path):
    (tmp_path / 'closedset_scripts').mkdir()
    shutil.copyfile(SCRIPT, tmp_path / 'closedset_scripts' / SCRIPT.name)
    (tmp_path / 'data').mkdir()
    for tile in TILES:
        fold = tmp_path / f'outputs/timematch_historical_rerun_seed1/source/closedset_source_{tile}_seed1/fold_0'
        fold.mkdir(parents=True)
        (fold / 'model.pt').touch()


def run_script(tmp_path, **options):
    env = dict(os.environ, DATA_ROOT='data', PYTHON_BIN='/bin/echo', EXPERIMENT='test_smoke')
    env.update(options)
    return subprocess.run([bash_bin(), f'closedset_scripts/{SCRIPT.name}'], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=40)


def test_queue_emits_exactly_eight_real_parseable_smoke_commands(tmp_path):
    setup_run(tmp_path)
    result = run_script(tmp_path)
    assert result.returncode == 0, result.stderr
    import shlex
    for task, tile in zip(TASKS, TILES):
        for variant in ('inshape', 'attention'):
            command = (tmp_path / f'logs/test_smoke/{task}_{variant}.log').read_text().strip()
            tokens = shlex.split(command)
            assert tokens[:2] == ['-u', 'train_segment_smoke.py']
            cfg = entry().build_config(tokens[2:])
            assert cfg.variant == variant and cfg.steps == 200
            assert cfg.weights.endswith(f'closedset_source_{tile}_seed1')
            assert cfg.output_dir == f'outputs/test_smoke/{variant}/{task}'
    lines = (tmp_path / 'logs/test_smoke/launcher.log').read_text().splitlines()
    starts = [line for line in lines if line.startswith('TASK_START|')]
    assert len(starts) == 8
    assert 'gpu=0|task=FR1_FR2|variant=inshape' in starts[0]
    assert 'gpu=1|task=FR1_FR2|variant=attention' in starts[1]
    assert 'gpu=2|task=FR2_DK1|variant=inshape' in starts[2]
    assert sum(line.startswith('TASK_COMPLETE|') for line in lines) == 8
    assert len(list((tmp_path / 'logs/test_smoke').glob('*.log'))) == 9
    assert not list((tmp_path / 'logs/test_smoke').glob('queue.*'))
    assert 'ANALYSIS' not in '\n'.join(lines)


def test_failed_job_propagates_status_while_rest_of_queue_runs(tmp_path):
    setup_run(tmp_path)
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "$*"\nif [[ "$*" == *"--experiment_name FR1_FR2"* && "$*" == *"--variant inshape"* ]]; then exit 23; fi\n')
    fake.chmod(0o755)
    assert run_script(tmp_path, PYTHON_BIN='./fake-python').returncode != 0
    log = (tmp_path / 'logs/test_smoke/launcher.log').read_text()
    assert 'TASK_FAILED|gpu=0|task=FR1_FR2|variant=inshape|status=23' in log
    assert log.count('TASK_COMPLETE|') == 7
    assert (tmp_path / 'logs/test_smoke/AT1_DK1_attention.log').exists()


def test_free_gpu_takes_next_job_before_slow_initial_jobs_finish(tmp_path):
    setup_run(tmp_path)
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "$*"\nif [[ "$*" == *"--experiment_name FR1_FR2"* ]]; then sleep 0.4; fi\n')
    fake.chmod(0o755)
    assert run_script(tmp_path, PYTHON_BIN='./fake-python').returncode == 0
    lines = (tmp_path / 'logs/test_smoke/launcher.log').read_text().splitlines()
    assert next(i for i, l in enumerate(lines) if l.startswith('TASK_START|') and 'task=FR2_DK1|variant=attention' in l) < next(i for i, l in enumerate(lines) if l.startswith('TASK_COMPLETE|') and 'task=FR1_FR2' in l)


@pytest.mark.parametrize('failure', ['existing_output', 'existing_logs', 'missing_source', 'bad_steps', 'bad_workers', 'bad_gpus'])
def test_queue_preflight_refuses_unsafe_or_changed_runs(tmp_path, failure):
    setup_run(tmp_path)
    options = {}
    if failure == 'existing_output':
        (tmp_path / 'outputs/test_smoke').mkdir()
    elif failure == 'existing_logs':
        (tmp_path / 'logs/test_smoke').mkdir(parents=True)
    elif failure == 'missing_source':
        (tmp_path / f'outputs/timematch_historical_rerun_seed1/source/closedset_source_{TILES[-1]}_seed1/fold_0/model.pt').unlink()
    else:
        options[dict(bad_steps='SMOKE_STEPS', bad_workers='NUM_WORKERS', bad_gpus='GPUS')[failure]] = dict(bad_steps='1001', bad_workers='4', bad_gpus='0 1')[failure]
    assert run_script(tmp_path, **options).returncode != 0
    assert not list((tmp_path / 'logs/test_smoke').glob('*_inshape.log'))


def test_bash_syntax():
    subprocess.run([bash_bin(), '-n', str(SCRIPT)], check=True)


@pytest.mark.parametrize('weight', ['0.01', '0.03', '0.1'])
def test_smoke_accepts_approved_sparse_weights_only(tmp_path, weight):
    config = entry().build_config([*args(tmp_path), '--sparse_weight', weight])
    assert config.sparse_weight == float(weight)
    assert config.pelt_penalty == .8 and config.keep_weight == 1


@pytest.mark.parametrize('weight', ['0', '-1', '.02', 'inf'])
def test_smoke_rejects_unapproved_sparse_weights(tmp_path, weight):
    with pytest.raises(SystemExit):
        entry().build_config([*args(tmp_path), '--sparse_weight', weight])


def test_sparse_sweep_maps_all_24_jobs_to_distinct_paths(tmp_path):
    import shlex
    setup_run(tmp_path)
    assert run_script(tmp_path, SPARSE_SWEEP='1').returncode == 0
    for weight in ('0.01', '0.03', '0.1'):
        tag = 'sparse_' + weight.replace('.', 'p')
        for task, tile in zip(TASKS, TILES):
            for variant in ('inshape', 'attention'):
                path = tmp_path / f'logs/test_smoke/{task}_{variant}_{tag}.log'
                config = entry().build_config(shlex.split(path.read_text())[2:])
                assert config.sparse_weight == float(weight)
                assert config.weights.endswith(f'closedset_source_{tile}_seed1')
                assert config.output_dir == f'outputs/test_smoke/{tag}/{variant}/{task}'
    log = (tmp_path / 'logs/test_smoke/launcher.log').read_text()
    assert log.count('TASK_COMPLETE|') == 24
    assert len(list((tmp_path / 'logs/test_smoke').glob('*.log'))) == 25


def test_failed_sweep_job_does_not_skip_remaining_weights(tmp_path):
    setup_run(tmp_path)
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "$*"\nif [[ "$*" == *"--experiment_name FR1_FR2"* && "$*" == *"--variant inshape"* && "$*" == *"--sparse_weight 0.03"* ]]; then exit 23; fi\n')
    fake.chmod(0o755)
    assert run_script(tmp_path, SPARSE_SWEEP='1', PYTHON_BIN='./fake-python').returncode != 0
    log = (tmp_path / 'logs/test_smoke/launcher.log').read_text()
    assert log.count('TASK_COMPLETE|') == 23 and log.count('TASK_FAILED|') == 1


def test_retry_sweep_runs_only_missing_sixteen_jobs(tmp_path):
    setup_run(tmp_path)
    result = run_script(tmp_path, SPARSE_SWEEP='1', SPARSE_WEIGHTS='0.03 0.1')
    assert result.returncode == 0, result.stderr
    log = (tmp_path / 'logs/test_smoke/launcher.log').read_text()
    assert log.count('TASK_COMPLETE|') == 16
    assert not list((tmp_path / 'logs/test_smoke').glob('*_sparse_0p01.log'))
    assert len(list((tmp_path / 'logs/test_smoke').glob('*_sparse_0p03.log'))) == 8
    assert len(list((tmp_path / 'logs/test_smoke').glob('*_sparse_0p1.log'))) == 8


@pytest.mark.parametrize('weights', ['0.02', '0.03 0.03', ' ', '0.1 invalid'])
def test_retry_sweep_rejects_invalid_or_duplicate_weights(tmp_path, weights):
    setup_run(tmp_path)
    assert run_script(tmp_path, SPARSE_SWEEP='1', SPARSE_WEIGHTS=weights).returncode != 0
    assert not list((tmp_path / 'logs/test_smoke').glob('*_sparse_*.log'))


def test_inshape_only_queue_runs_four_source_tasks(tmp_path):
    setup_run(tmp_path)
    assert run_script(tmp_path, INSHAPE_ONLY='1').returncode == 0
    log = (tmp_path/'logs/test_smoke/launcher.log').read_text()
    assert log.count('TASK_COMPLETE|') == 4
    assert 'variant=attention' not in log
    assert 'gpu=0|task=FR1_FR2' in log
    assert 'gpu=1|task=FR2_DK1' in log
    assert 'gpu=2|task=DK1_AT1' in log
