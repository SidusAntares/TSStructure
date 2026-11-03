import shutil
import subprocess
import os
import shlex
import pytest
from pathlib import Path
from test_segment_launcher import bash_bin, setup_run, entry_args
from train_segment_timematch import build_config


def test_joint_source_and_uda_endpoint_configuration(tmp_path):
    config = build_config(entry_args(tmp_path, 'joint_source'))
    assert config.epochs == 100 and config.source_init == 'scratch'
    assert config.lr == .001 and config.selector_lr == .0005 and config.keep_weight == 1
    uda = build_config([*entry_args(tmp_path, 'uda'), '--source_training_epochs','100'])
    assert uda.epochs == 20 and uda.source_training_epochs == 100 and uda.keep_weight == 1


@pytest.mark.parametrize('fail', [False, True])
def test_eight_comparison_chains_and_logs_are_parseable(tmp_path, fail):
    setup_run(tmp_path)
    script = 'run_segment_compare_8jobs_4gpu.sh'
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root/'closedset_scripts'/script, tmp_path/'closedset_scripts'/script)
    subprocess.run([bash_bin(), '-n', str(root/'closedset_scripts'/script)], check=True)
    fake = tmp_path/'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "$*"\n'+
        ('if [[ "$*" == *"--stage joint_source"* && "$*" == *"--experiment_name FR1_FR2"* ]]; then exit 19; fi\n' if fail else ''))
    fake.chmod(0o755)
    result = subprocess.run([bash_bin(), 'closedset_scripts/'+script], cwd=tmp_path,
        env=dict(os.environ, DATA_ROOT='data', PYTHON_BIN='./fake-python', EXPERIMENT='compare'),
        capture_output=True,text=True,timeout=40)
    assert (result.returncode != 0) == fail, result.stderr
    for task in ['AT1_DK1','FR1_FR2','FR2_DK1','DK1_AT1']:
        for route, epochs in [('finetune',20),('joint',100)]:
            for stage in ['source','uda']:
                if fail and task == 'FR1_FR2' and route == 'joint' and stage == 'uda':
                    assert not (tmp_path/f'logs/compare/{task}_{route}_{stage}.log').exists()
                    continue
                cmd=(tmp_path/f'logs/compare/{task}_{route}_{stage}.log').read_text()
                config=build_config(shlex.split(cmd)[2:])
                assert config.variant == 'inshape' and config.keep_weight == 1
                assert config.epochs == (epochs if stage == 'source' else 20)
                if stage == 'uda':
                    assert config.source_training_epochs == epochs
                    assert config.structure_checkpoint == f'outputs/compare/{route}/{task}/source/fold_0/model.pt'
    assert (tmp_path/'logs/compare/launcher.log').read_text().count('TASK_COMPLETE|') == (7 if fail else 8)
