import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from test_segment_ranking import Classifier


def test_length_control_uses_exact_observation_counts_and_excludes_identical_subset():
    from analysis.segment_length_diagnostic import batch_length_controls
    info = dict(probabilities=torch.tensor([[[.9, .2, .1]]]),
        candidate_count=torch.tensor([3]), segments=torch.tensor([[0, 1, 2, 2, -1]]),
        attention=torch.tensor([[[.25, .25, .25, .25, 0.]]]),
        values=torch.tensor([[[[2.], [0.], [0.], [0.], [999.]]]]))
    model = Classifier().eval()
    rows, deletions = batch_length_controls(model, info, torch.tensor([0]), 1.)
    one, half = rows
    assert one['exact_match_alternatives'] == 1
    assert one['selector_observations'] == one['matched_observations'] == 1
    assert one['gain_vs_matched_random'] > 0 and one['gain_vs_longest'] > 0
    assert one['selector_is_longest'] is False
    assert half['exact_match_alternatives'] == 0 and half['gain_vs_matched_random'] is None
    assert [row['observation_count'] for row in deletions] == [1, 1, 2]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_frozen_checkpoint_analysis_reuses_original_probe_and_does_not_write_checkpoint(tmp_path, monkeypatch):
    import dataset
    import numpy as np
    from models.stclassifier import PseLTae
    from models.segment_selector import SegmentSelector
    from analysis.segment_length_diagnostic import analyze
    leaf = tmp_path/'inshape/AT1_DK1'
    (leaf/'fold_0').mkdir(parents=True)
    source = 'austria/33UVP/2017'
    config = dict(source=source, classes=['a', 'b', 'c'], seed=1, input_dim=10,
        with_extra=False, selector_width=16, selector_heads=2, selector_dropout=0.,
        num_pixels=8, seq_length=30, focal_loss_gamma=1., pelt_penalty=.8,
        pelt_min_size=3, pelt_penalty_mode='absolute', original_source_checkpoint_sha256='original')
    (leaf/'resolved_config.json').write_text(json.dumps(config))
    (leaf/'probe_parcel_ids.json').write_text(json.dumps(dict(split='SourceTrain only', parcel_ids=[1,2,3], seed=12001)))
    (leaf/'split_indices.json').write_text(json.dumps({source:dict(train=[1,2,3],val=[4],test=[5])}))
    checkpoint = leaf/'fold_0/smoke_checkpoint.pt'
    torch.save(dict(role='source_structure_smoke', status='completed', completed_smoke_epochs=5.,
        classification_path='keep_only_unified', variant='inshape',
        original_source_checkpoint_sha256='original',
        state_dict=PseLTae(num_classes=3, with_extra=False).state_dict(),
        selector_state_dict=SegmentSelector(width=16, heads=2).state_dict()), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    used = []
    class Data:
        def __init__(self, root, domain, classes, transform, **kwargs):
            assert domain == source and kwargs['indices'] == {1,2,3}
            self.transform = transform
            used.append(domain)
        def __len__(self): return 3
        def __getitem__(self, index):
            sample = dict(pixels=np.random.default_rng(index).integers(0, 65000, (40,10,8), dtype=np.uint16),
                positions=np.arange(40)*8, extra=np.zeros(4), label=index, parcel_index=index+1)
            return self.transform(sample)
    monkeypatch.setattr(dataset, 'PixelSetData', Data)
    args = SimpleNamespace(experiment_root=str(tmp_path), task='AT1_DK1', data_root='unused', device='cpu')
    result = analyze(args)
    assert used == [source] and result['frozen'] and result['completed_smoke_epochs'] == 5
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    output = tmp_path/'diagnostics/length_control/AT1_DK1'
    assert (output/'summary.json').exists() and (output/'budgets.csv').exists()
    assert (output/'deletions.csv').exists()
    assert not list(output.glob('*.pt'))
    with pytest.raises(FileExistsError): analyze(args)


@pytest.mark.parametrize('fail', [False, True])
def test_four_gpu_frozen_launcher_maps_logs_and_propagates_failure(tmp_path, fail):
    import os
    import shutil
    import subprocess
    from pathlib import Path
    from test_segment_launcher import bash_bin
    root = Path(__file__).resolve().parents[1]
    script = 'run_segment_length_diagnostic_4gpu.sh'
    (tmp_path/'closedset_scripts').mkdir()
    shutil.copyfile(root/'closedset_scripts'/script, tmp_path/'closedset_scripts'/script)
    for task in ('FR1_FR2','FR2_DK1','DK1_AT1','AT1_DK1'):
        fold = tmp_path/f'outputs/test_length/inshape/{task}/fold_0'
        fold.mkdir(parents=True)
        (fold/'smoke_checkpoint.pt').touch()
    fake = tmp_path/'fake-python'
    fake.write_text('#!/usr/bin/env bash\n/bin/echo "$CUDA_VISIBLE_DEVICES|$*"\n'+
        ('if [[ "$*" == *"--task FR2_DK1"* ]]; then exit 17; fi\n' if fail else ''))
    fake.chmod(0o755)
    subprocess.run([bash_bin(), '-n', str(root/'closedset_scripts'/script)], check=True)
    env = dict(os.environ, EXPERIMENT='test_length', DATA_ROOT='unused', GPUS='0 1 2 3', PYTHON_BIN='./fake-python')
    result = subprocess.run([bash_bin(), f'closedset_scripts/{script}'], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30)
    assert (result.returncode != 0) == fail
    for gpu, task in enumerate(('FR1_FR2','FR2_DK1','DK1_AT1','AT1_DK1')):
        text = (tmp_path/f'logs/test_length/{task}_length_control.log').read_text()
        assert text.startswith(f'{gpu}|-u -m analysis.segment_length_diagnostic')
        assert '--experiment_root outputs/test_length' in text and f'--task {task}' in text
    control = (tmp_path/'logs/test_length/length_control_launcher.log').read_text()
    assert control.count('DIAGNOSTIC_COMPLETE|') == (3 if fail else 4)
    assert control.count('DIAGNOSTIC_FAILED|') == (1 if fail else 0)
    assert not (tmp_path/'outputs/test_length/diagnostics').exists()
