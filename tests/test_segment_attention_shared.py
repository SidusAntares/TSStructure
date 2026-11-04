import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from models.segment_selector import SegmentSelector
from train_segment_timematch import build_config
from test_segment_launcher import entry_args, setup_run, bash_bin


def test_shared_attention_changes_only_detached_score_conditions():
    torch.manual_seed(9)
    baseline = SegmentSelector(feature_dim=8, width=16, heads=4, protocol='inshape_coverage_v1').eval()
    torch.manual_seed(9)
    selector = SegmentSelector(feature_dim=8, width=16, heads=4,
                               protocol='inshape_attention_coverage_v1').eval()
    features = torch.randn(2, 9, 8, requires_grad=True)
    attention = torch.randn(2, 16, 9).softmax(-1).requires_grad_()
    segments = torch.arange(3).repeat_interleave(3)[None].expand(2, -1)
    a = baseline(features, attention, segments, variant='inshape', deterministic=True)
    b = selector(features, attention, segments, variant='attention_shared', deterministic=True)
    torch.testing.assert_close(a['probabilities'], b['probabilities'], rtol=0, atol=0)
    assert torch.equal(b['gate'][:, :1].expand_as(b['gate']), b['gate'])
    # Once the extra input weights learn, attention can affect selection.
    with torch.no_grad():
        selector.attention_scorer.weight[1, 0] = 2.
    c = selector(features, attention, segments, variant='attention_shared', deterministic=True)
    assert not torch.equal(c['probabilities'], b['probabilities'])
    c['probabilities'].sum().backward()
    assert attention.grad is None
    assert features.grad is not None and features.grad.abs().sum() > 0
    selector.zero_grad()
    features.grad = None
    result = selector(features, attention, segments, variant='attention_shared')
    result['sparse_loss'].backward()
    assert features.grad is None and attention.grad is None
    assert selector.attention_scorer.weight.grad.abs().sum() > 0


def test_attention_finetune_matches_content_baseline_parameters(tmp_path):
    base = build_config(entry_args(tmp_path, 'finetune'))
    argv = entry_args(tmp_path, 'finetune')
    argv[argv.index('--variant')+1] = 'attention_shared'
    config = build_config(argv)
    assert config.selector_protocol == 'inshape_attention_coverage_v1'
    for key in ('epochs', 'lr', 'selector_lr', 'sparse_weight', 'reinmax_tau',
                'selector_width', 'selector_dropout', 'keep_weight', 'num_workers', 'seed'):
        assert getattr(config, key) == getattr(base, key)


def test_classification_updates_attention_scorer_through_shared_gate(monkeypatch):
    from models.stclassifier import PseLTae
    from utils import segment_structure
    torch.manual_seed(4)
    model = PseLTae(num_classes=3, with_extra=False)
    selector = SegmentSelector(width=32, heads=4, dropout=.1,
                               protocol='inshape_attention_coverage_v1')
    monkeypatch.setattr(segment_structure, 'segment_pelt', lambda features, *a, **kw:
        torch.arange(3).repeat_interleave(4)[None].expand(features.size(0), -1))
    _, logits, _ = segment_structure.structure_forward(model, selector,
        torch.rand(4, 12, 10, 8), torch.ones(4, 12, 8),
        torch.arange(12)[None].expand(4, -1)*20, torch.zeros(4, 4),
        variant='attention_shared', full_loss=False, unified=True)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0])).backward()
    assert selector.attention_scorer.weight.grad.abs().sum() > 0
    assert model.decoder[-1].weight.grad is not None


@pytest.mark.parametrize('failure', ['none', 'source', 'analysis'])
def test_four_attention_chains_then_separate_visualization(tmp_path, failure):
    setup_run(tmp_path)
    script = 'run_segment_attention_finetune_4gpu.sh'
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root/'closedset_scripts'/script, tmp_path/'closedset_scripts'/script)
    subprocess.run([bash_bin(), '-n', str(root/'closedset_scripts'/script)], check=True)
    # Existing baseline artifacts must remain untouched.
    base = tmp_path/'outputs/compare/finetune/AT1_DK1/uda'
    base.mkdir(parents=True)
    (base/'baseline').write_text('keep')
    plots = tmp_path/'outputs/compare_visualization'
    plots.mkdir()
    (plots/'manifest.json').write_text('baseline manifest')
    fake = tmp_path/'fake-python'
    body = '#!/usr/bin/env bash\n/bin/echo "$*"\n'
    if failure == 'source':
        body += 'if [[ "$*" == *"--stage finetune"* && "$*" == *"--experiment_name AT1_DK1"* ]]; then exit 17; fi\n'
    if failure == 'analysis':
        body += 'if [[ "$*" == *"-m analysis.segment_structure_visualization"* ]]; then exit 18; fi\n'
    fake.write_text(body)
    fake.chmod(0o755)
    result = subprocess.run([bash_bin(), 'closedset_scripts/'+script], cwd=tmp_path,
        env=dict(os.environ, DATA_ROOT='data', PYTHON_BIN='./fake-python', EXPERIMENT='compare'),
        capture_output=True, text=True, timeout=40)
    assert (result.returncode == 0) == (failure == 'none')
    for task in ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1'):
        for stage in ('source', 'uda'):
            log = tmp_path/f'logs/compare/{task}_finetune_attention_{stage}.log'
            if failure == 'source' and task == 'AT1_DK1' and stage == 'uda':
                assert not log.exists()
                continue
            config = build_config(shlex.split(log.read_text())[2:])
            assert config.epochs == 20 and config.variant == 'attention_shared'
            assert config.selector_protocol == 'inshape_attention_coverage_v1'
            if stage == 'uda':
                assert config.structure_checkpoint == f'outputs/compare/finetune_attention/{task}/source/fold_0/model.pt'
    analysis = tmp_path/'logs/compare_visualization/attention_analysis.log'
    assert analysis.exists() == (failure != 'source')
    if analysis.exists():
        assert '--layout attention_finetune' in analysis.read_text()
        assert '--output_dir outputs/compare_visualization' in analysis.read_text()
    assert (plots/'manifest.json').read_text() == 'baseline manifest'
    assert (base/'baseline').read_text() == 'keep'
