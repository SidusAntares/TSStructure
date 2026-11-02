"""CPU checks for the isolated segment experiment entry and dynamic queue."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'closedset_scripts/run_segment_timematch_8jobs_3gpu.sh'
TASKS = ('FR1_FR2', 'FR2_DK1', 'DK1_AT1', 'AT1_DK1')
TILES = ('30TXT', '31TCJ', '32VNH', '33UVP')


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
    env = dict(os.environ, DATA_ROOT='data', PYTHON_BIN='/bin/echo', EXPERIMENT='test_segments')
    env.update(options)
    return subprocess.run([bash_bin(), f'closedset_scripts/{SCRIPT.name}'], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=40)


def test_eight_chains_use_original_source_then_own_finetune_and_final_analysis(tmp_path):
    setup_run(tmp_path)
    result = run_script(tmp_path)
    assert result.returncode == 0, result.stderr
    for task, tile in zip(TASKS, TILES):
        for variant in ('inshape', 'attention'):
            for stage in ('finetune', 'uda'):
                command = (tmp_path / f'logs/test_segments/{task}_{variant}_{stage}.log').read_text()
                assert f'--stage {stage} --variant {variant}' in command
                assert f'--weights outputs/timematch_historical_rerun_seed1/source/closedset_source_{tile}_seed1' in command
                assert f'--output_dir outputs/test_segments/{variant}/{task}/{stage}' in command
                assert '--seed 1 --num_workers 8 --epochs 20' in command
                if stage == 'uda':
                    assert f'--structure_checkpoint outputs/test_segments/{variant}/{task}/finetune/fold_0/model.pt' in command
                    assert '--steps_per_epoch 500' in command
    lines = (tmp_path / 'logs/test_segments/launcher.log').read_text().splitlines()
    starts = [line for line in lines if line.startswith('TASK_START|')]
    assert len(starts) == 8
    assert all(any(f'gpu={gpu}|' in line for line in starts) for gpu in (0, 1, 2))
    assert 'task=FR1_FR2|variant=inshape' in starts[0]
    assert 'task=FR1_FR2|variant=attention' in starts[1]
    assert 'task=AT1_DK1' in starts[-1]
    assert sum(line.startswith('TASK_COMPLETE|') for line in lines) == 8
    assert lines.index('ANALYSIS_START') > max(i for i, line in enumerate(lines) if line.startswith('TASK_COMPLETE|'))
    analysis = (tmp_path / 'logs/test_segments_visualization/analysis.log').read_text()
    assert '-m analysis.segment_structure_visualization' in analysis
    assert '--experiment_root outputs/test_segments' in analysis
    assert '--output_dir outputs/test_segments_visualization' in analysis


def test_failed_finetune_skips_its_uda_and_analysis_but_continues_queue(tmp_path):
    setup_run(tmp_path)
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "$*"\nif [[ "$*" == *"--stage finetune --variant inshape"* && "$*" == *"--experiment_name FR1_FR2"* ]]; then exit 23; fi\n')
    fake.chmod(0o755)
    result = run_script(tmp_path, PYTHON_BIN='./fake-python')
    assert result.returncode != 0
    assert not (tmp_path / 'logs/test_segments/FR1_FR2_inshape_uda.log').exists()
    assert (tmp_path / 'logs/test_segments/AT1_DK1_attention_uda.log').exists()
    assert not (tmp_path / 'logs/test_segments_visualization/analysis.log').exists()


def test_workers_reuse_free_gpu_without_round_barrier(tmp_path):
    setup_run(tmp_path)
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "$*"\nif [[ "$*" == *"--experiment_name FR1_FR2"* ]]; then sleep 0.4; fi\n')
    fake.chmod(0o755)
    assert run_script(tmp_path, PYTHON_BIN='./fake-python').returncode == 0
    lines = (tmp_path / 'logs/test_segments/launcher.log').read_text().splitlines()
    assert next(i for i, l in enumerate(lines) if l.startswith('TASK_START|') and 'task=FR2_DK1|variant=attention' in l) < next(i for i, l in enumerate(lines) if l.startswith('TASK_COMPLETE|') and 'task=FR1_FR2' in l)


@pytest.mark.parametrize('failure', ['missing_source', 'existing_output', 'bad_workers'])
def test_preflight_fails_before_starting_jobs(tmp_path, failure):
    setup_run(tmp_path)
    options = {}
    if failure == 'missing_source':
        (tmp_path / f'outputs/timematch_historical_rerun_seed1/source/closedset_source_{TILES[-1]}_seed1/fold_0/model.pt').unlink()
    elif failure == 'existing_output':
        (tmp_path / 'outputs/test_segments').mkdir()
    else:
        options['NUM_WORKERS'] = '4'
    assert run_script(tmp_path, **options).returncode != 0
    assert not list((tmp_path / 'logs/test_segments').glob('*_finetune.log'))


def load_entry():
    spec = importlib.util.spec_from_file_location('segment_entry', ROOT / 'train_segment_timematch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry_args(tmp_path, stage='finetune'):
    args = ['--stage', stage, '--variant', 'inshape', '--source', 'austria/33UVP/2017',
            '--target', 'denmark/32VNH/2017', '--weights', str(tmp_path / 'source'),
            '--output_dir', str(tmp_path / 'output')]
    if stage == 'uda':
        args += ['--structure_checkpoint', str(tmp_path / 'finetune/fold_0/model.pt')]
    return args


def source_config():
    return dict(source='austria/33UVP/2017', target='austria/33UVP/2017', seed=1,
                num_workers=8, epochs=100, model='pseltae', closed_set=True, num_folds=1,
                num_blocks=100, val_ratio=.1, test_ratio=.2, sample_pixels_val=True,
                batch_size=128, seq_length=30, num_pixels=64, input_dim=10,
                combine_spring_and_winter=False, with_extra=False, with_shift_aug=False,
                lr=.001, weight_decay=.0001, focal_loss_gamma=1., shift_aug_p=1.,
                max_shift_aug=60, mode=None, m1=False, method=None)


def write_source(tmp_path, **overrides):
    leaf = tmp_path / 'source'
    (leaf / 'fold_0').mkdir(parents=True)
    (leaf / 'fold_0/model.pt').write_bytes(b'original')
    (leaf / 'train_config.json').write_text(json.dumps(dict(source_config(), **overrides)))


@pytest.mark.parametrize('override', [dict(mode='M0'), dict(mode='M1'), dict(mode='anything'),
    dict(m1=True), dict(method='timematch'), dict(epochs=20), dict(target='denmark/32VNH/2017')])
def test_source_validation_refuses_changed_original_protocol(tmp_path, override):
    write_source(tmp_path, **override)
    module = load_entry()
    with pytest.raises(ValueError):
        module.validate_source(module.build_config(entry_args(tmp_path)))


@pytest.mark.parametrize('stage', ['finetune', 'uda'])
def test_entry_dispatch_isolated_output_and_provenance(tmp_path, monkeypatch, stage):
    write_source(tmp_path)
    if stage == 'uda':
        path = tmp_path / 'finetune/fold_0/model.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'fine')
    received = []
    monkeypatch.setitem(sys.modules, 'utils.segment_training', SimpleNamespace(
        finetune_structure=received.append, run_structure_uda=received.append))
    module = load_entry()
    module.main(entry_args(tmp_path, stage))
    assert len(received) == 1
    saved = json.loads((tmp_path / 'output/config.json').read_text())
    assert len(saved['source_checkpoint_sha256']) == 64
    assert saved['weights'] == str(tmp_path / 'source')
    assert (tmp_path / 'output/fold_0').is_dir()
    assert 'train_segment_timematch.py' in json.loads((tmp_path / 'output/code_provenance.json').read_text())['sha256']
    with pytest.raises(FileExistsError):
        module.main(entry_args(tmp_path, stage))
    assert len(received) == 1


def test_dry_run_has_protocol_and_no_heavy_imports_or_outputs(tmp_path):
    script = ('import sys,runpy; sys.argv=' + repr(['train_segment_timematch.py', *entry_args(tmp_path, 'uda'), '--dry_run'])
              + ';runpy.run_path(' + repr(str(ROOT / 'train_segment_timematch.py')) + ',run_name="__main__");'
              + 'assert not any(n in sys.modules for n in ("torch","numpy","utils.segment_training","utils.imputation_training"))')
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)
    assert cfg['mode'] is None and cfg['m1'] is False
    assert cfg['epochs'] == 20 and cfg['steps_per_epoch'] == 500
    assert cfg['lr'] == cfg['selector_lr'] == cfg['weight_decay'] == 1e-4
    assert cfg['keep_weight'] == 1 and cfg['sparse_weight'] == .01
    assert cfg['pelt_penalty'] == .8 and cfg['pelt_min_size'] == 3
    assert cfg['pelt_penalty_mode'] == 'absolute'
    assert cfg['selector_dropout'] == 0.
    assert cfg['run_validation'] is False and cfg['strict_final_student'] is True
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('flag,value', [('--num_workers', '4'), ('--epochs', '100'),
    ('--keep_weight', '.5'), ('--pelt_min_size', '1'), ('--sparse_weight', 'nan'),
    ('--domain_specific_bn', 'false'), ('--run_validation', 'true')])
def test_entry_refuses_protocol_drift(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        load_entry().build_config([*entry_args(tmp_path), flag, value])


def test_missing_finetune_checkpoint_does_not_allocate_output(tmp_path):
    write_source(tmp_path)
    with pytest.raises(FileNotFoundError, match='finetuning checkpoint'):
        load_entry().main(entry_args(tmp_path, 'uda'))
    assert not (tmp_path / 'output').exists()


def test_bad_original_source_does_not_allocate_output(tmp_path):
    write_source(tmp_path, mode='M0')
    with pytest.raises(ValueError):
        load_entry().main(entry_args(tmp_path))
    assert not (tmp_path / 'output').exists()


def test_analysis_failure_propagates_and_queue_temporary_files_are_removed(tmp_path):
    setup_run(tmp_path)
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "$*"\nif [[ "$*" == *"-m analysis.segment_structure_visualization"* ]]; then exit 7; fi\n')
    fake.chmod(0o755)
    assert run_script(tmp_path, PYTHON_BIN='./fake-python').returncode == 7
    assert not list((tmp_path / 'logs/test_segments').glob('queue.*'))


def test_bash_syntax():
    subprocess.run([bash_bin(), '-n', str(SCRIPT)], check=True)
