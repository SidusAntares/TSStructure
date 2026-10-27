"""Compare dry-run training arguments directly with the archived Namespace log."""

import ast
import argparse
import gzip
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from test_timematch_diagnostic_launcher import bash_bin


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'closedset_scripts/run_closedset_historical_4tasks_seed1.sh'
ARCHIVE = ROOT / 'logs/closedset_12tasks_3seeds_performance.log.gz'
TASKS = ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1')
TILES = ('33UVP', '30TXT', '31TCJ', '32VNH')
TARGETS = ('32VNH', '31TCJ', '32VNH', '33UVP')


def setup_script(tmp_path):
    assert SCRIPT.exists(), 'Historical rerun launcher is missing'
    (tmp_path / 'closedset_scripts').mkdir()
    shutil.copyfile(SCRIPT, tmp_path / 'closedset_scripts' / SCRIPT.name)
    (tmp_path / 'data').mkdir()


def launch(tmp_path, python='/bin/echo', **options):
    env = dict(os.environ, DATA_ROOT='data', PYTHON_BIN=python, EXPERIMENT='test_historical',
               TASKS=' '.join(TASKS), GPUS='3 1 2 0', SEED='1', NUM_WORKERS='8')
    env.update(options)
    return subprocess.run([bash_bin(), f'closedset_scripts/{SCRIPT.name}'], cwd=tmp_path,
                          env=env, capture_output=True, text=True)


def arguments(line):
    tokens = shlex.split(line)
    # Execute the real CLI parser without optional training imports or main().
    tree = ast.parse((ROOT / 'train.py').read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare))
    parser_nodes = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'cfg'
                                               for target in node.targets):
            break
        parser_nodes.append(node)
    helpers = ast.parse((ROOT / 'utils/train_utils.py').read_text(encoding='utf-8'))
    bool_flag = next(node for node in helpers.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'bool_flag')
    scope = dict(argparse=argparse, FALSY_STRINGS={'off', 'false', '0'}, TRUTHY_STRINGS={'on', 'true', '1'})
    exec(compile(ast.Module(body=[bool_flag]+parser_nodes, type_ignores=[]), '<training-cli>', 'exec'), scope)
    return vars(scope['parser'].parse_args(tokens[tokens.index('train.py')+1:]))


def test_launcher_matches_archived_seed1_training_configuration(tmp_path):
    setup_script(tmp_path)
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ''
    log_root = tmp_path / 'logs/test_historical'
    for task, tile, target in zip(TASKS, TILES, TARGETS):
        source = arguments((log_root / f'{task}_source.log').read_text())
        uda = arguments((log_root / f'{task}.log').read_text())
        assert source['seed'] == uda['seed'] == 1
        assert source['num_workers'] == uda['num_workers'] == 8
        assert source['epochs'] == 100 and uda['epochs'] == 20
        assert uda['steps_per_epoch'] == 500 and uda['evidence_mode'] == 'off'
        assert uda['weights'] == f'outputs/test_historical/source/closedset_source_{tile}_seed1'
        assert source['experiment_name'] == f'closedset_source_{tile}_seed1'
        assert uda['experiment_name'] == f'closedset_timematch_{tile}_to_{target}_seed1'
        assert uda['diagnostic_dir'] is None
        assert 'timematch' not in (log_root / f'{task}_source.log').read_text().split()
    if not ARCHIVE.is_file():
        return  # Server-independent command checks above always run.
    expected = {}
    with gzip.open(ARCHIVE, 'rt', encoding='utf-8', errors='replace') as stream:
        for line in stream:
            start = line.find('Namespace(')
            if start < 0:
                continue
            node = ast.parse(line[start:].strip(), mode='eval').body
            record = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
            if record['seed'] == 1:
                expected[record['experiment_name']] = record
    ignored = {'data_root', 'output_dir', 'tensorboard_log_dir', 'weights'}
    for path in log_root.glob('*.log'):
        if path.name == 'launcher.log':
            continue
        actual = arguments(path.read_text())
        historical = expected[actual['experiment_name']]
        for key, value in historical.items():
            if key not in ignored:
                assert actual.get(key) == value, (path.name, key, actual.get(key), value)


@pytest.mark.parametrize('failed_stage', ['source', 'uda'])
def test_source_then_uda_order_failure_and_gpu_isolation(tmp_path, failed_stage):
    setup_script(tmp_path)
    fake = tmp_path / 'fake-python'
    condition = ('"$*" == *"closedset_source_30TXT_seed1"* && "$*" != *" timematch "*'
                 if failed_stage == 'source' else '"$*" == *" timematch "* && "$*" == *"30TXT_to_31TCJ_seed1"*')
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "gpu=$CUDA_VISIBLE_DEVICES $*"\n'
                    f'if [[ {condition} ]]; then exit 23; fi\n',
                    encoding='utf-8')
    fake.chmod(0o755)
    assert launch(tmp_path, python='./fake-python', REUSE_SOURCE='1').returncode == 1
    logs = tmp_path / 'logs/test_historical'
    assert (logs / 'FR1_FR2.log').exists() == (failed_stage == 'uda')
    for gpu, task in zip((3, 1, 2, 0), TASKS):
        assert f'gpu={gpu} ' in (logs / f'{task}_source.log').read_text()
        if task != 'FR1_FR2':
            assert f'gpu={gpu} ' in (logs / f'{task}.log').read_text()
    assert 'FAILED|task=FR1_FR2' in (logs / 'launcher.log').read_text()


def test_refuses_existing_results_and_nonhistorical_seed_or_workers(tmp_path):
    setup_script(tmp_path)
    assert launch(tmp_path, NUM_WORKERS='4').returncode != 0
    assert launch(tmp_path, SEED='2').returncode != 0
    (tmp_path / 'outputs/test_historical').mkdir(parents=True)
    assert launch(tmp_path).returncode != 0
    assert not (tmp_path / 'logs/test_historical/AT1_DK1_source.log').exists()


def test_each_gpu_starts_uda_without_waiting_for_other_sources(tmp_path):
    setup_script(tmp_path)
    fake = tmp_path / 'fake-python'
    fake.write_text(
        '#!/usr/bin/env bash\n/bin/echo "$*"\n'
        'if [[ "$*" == *" timematch "* && "$*" == *"30TXT_to_31TCJ_seed1"* ]]; then\n'
        '  touch fr1_uda_started\n'
        'elif [[ "$*" == *"closedset_source_33UVP_seed1"* && "$*" != *" timematch "* ]]; then\n'
        '  for attempt in {1..60}; do\n'
        '    [[ -f fr1_uda_started ]] && exit 0\n'
        '    sleep 0.05\n'
        '  done\n'
        '  exit 31\n'
        'fi\n', encoding='utf-8')
    fake.chmod(0o755)
    assert launch(tmp_path, python='./fake-python').returncode == 0
    assert (tmp_path / 'fr1_uda_started').is_file()


def test_bash_syntax():
    assert SCRIPT.exists()
    subprocess.run([bash_bin(), '-n', str(SCRIPT)], check=True)
