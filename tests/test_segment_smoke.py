from types import SimpleNamespace

import json
import pytest
import torch

from models.stclassifier import PseLTae
from models.segment_selector import SegmentSelector
from utils.segment_smoke import train_smoke_steps, probe_statistics, inspect_health


def batch():
    return dict(pixels=torch.rand(4, 30, 10, 8), valid_pixels=torch.ones(4, 30, 8),
                positions=torch.arange(30)[None].expand(4, -1)*10,
                extra=torch.zeros(4, 4), label=torch.tensor([0, 1, 2, 0]))


def cfg(path):
    return SimpleNamespace(device='cpu', steps=3, scheduler_epochs=20, lr=1e-4, selector_lr=1e-4,
        weight_decay=1e-4, focal_loss_gamma=1., keep_weight=.5, sparse_weight=.01,
        pelt_penalty=.8, pelt_penalty_mode='absolute', pelt_min_size=3, variant='attention',
        seed=1, fold_dir=str(path), probe_every=2, gradient_every=2, health_window=2,
        warmup_steps=20, original_source_checkpoint_sha256='original', source='source')


@pytest.mark.parametrize('sparse_weight', [.01, .03, .1])
def test_smoke_updates_jointly_without_exporting_twenty_epoch_checkpoint(tmp_path, sparse_weight):
    torch.manual_seed(5)
    model, selector = PseLTae(num_classes=3, with_extra=False), SegmentSelector(width=16, heads=2)
    old = {k: v.clone() for k, v in model.state_dict().items()}
    config = cfg(tmp_path)
    config.sparse_weight = sparse_weight
    result = train_smoke_steps(model, selector, [batch()], [batch()], config)
    assert result['completed_steps'] == 3 and result['status'] == 'completed'
    packet = torch.load(tmp_path/'smoke_checkpoint.pt', weights_only=False)
    assert packet['role'] == 'source_structure_smoke' and packet['completed_steps'] == 3
    assert 'completed_epochs' not in packet and not (tmp_path/'model.pt').exists()
    assert any(not torch.equal(v, old[k]) for k, v in packet['state_dict'].items())
    torch.testing.assert_close(packet['state_dict']['temporal_encoder.positional_enc.weight'],
                               old['temporal_encoder.positional_enc.weight'], rtol=0, atol=0)
    assert (tmp_path/'steps.csv').is_file() and (tmp_path/'probes.csv').is_file()
    report = json.loads((tmp_path/'health_summary.json').read_text())
    assert report['scheduler_total_steps'] == 20
    assert report['last_learning_rate'] > 0
    assert report['sparse_weight'] == sparse_weight and packet['sparse_weight'] == sparse_weight
    import csv
    with (tmp_path/'keep_gradients.csv').open() as stream:
        gradients = list(csv.DictReader(stream))
    assert len(gradients) == 3
    assert all(float(row['sparse_selector_grad_norm']) > 0 for row in gradients)
    assert 'weighted_keep_selector_grad_norm' in gradients[0]
    assert report['final_probe']['all_selected_rate'] >= 0
    if report['final_probe']['all_selected_rate'] >= .95:
        assert any('retain all segments' in warning for warning in report['warnings'])


def test_probe_restores_rng_and_every_model_mode_and_bn_buffer():
    model = PseLTae(num_classes=3, with_extra=False).train()
    selector = SegmentSelector(width=16, heads=2).train()
    selector.encoder.eval()
    config = cfg('.')
    sample = batch()
    modes = [m.training for root in (model, selector) for m in root.modules()]
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    state = torch.get_rng_state().clone()
    first = probe_statistics(model, selector, [sample], config)
    second = probe_statistics(model, selector, [sample], config)
    assert first == second and torch.equal(state, torch.get_rng_state())
    assert modes == [m.training for root in (model, selector) for m in root.modules()]
    assert all(torch.equal(v, buffers[name]) for name, v in model.named_buffers())


def test_health_checks_do_not_confuse_one_flat_sample_with_global_collapse():
    config = cfg('.')
    config.warmup_steps = 0
    assert inspect_health([dict(single_segment_rate=.1, fallback_rate=.1)]*2, config) is None
    assert inspect_health([dict(single_segment_rate=1., fallback_rate=0.)]*2, config) == 'failed_candidate_collapse'
    assert inspect_health([dict(single_segment_rate=0., fallback_rate=1.)]*2, config) == 'failed_gate_collapse'


def test_smoke_consumes_actual_numpy_transform_and_collated_batches(tmp_path):
    import numpy as np
    from torchvision.transforms import Compose
    from transforms import RandomSamplePixels, RandomSampleTimeSteps, Normalize, ToTensor
    from utils.segment_smoke import FixedProbeTransform
    transform = FixedProbeTransform(Compose([RandomSamplePixels(64),
        RandomSampleTimeSteps(30), Normalize(), ToTensor()]), seed=12001)
    samples = [transform(dict(pixels=np.random.default_rng(i).integers(0, 65535,
        size=(40, 10, 8), dtype=np.uint16), positions=np.arange(40)*8,
        extra=np.zeros(4), label=i % 3, parcel_index=i)) for i in range(4)]
    loader = torch.utils.data.DataLoader(samples, batch_size=4)
    config = cfg(tmp_path)
    result = train_smoke_steps(PseLTae(num_classes=3, with_extra=False),
        SegmentSelector(width=16, heads=2), loader, loader, config)
    assert result['completed_steps'] == 3 and result['no_uda']
    assert result['final_probe']['candidate_count'] >= 1


def test_loss_gradient_balance_records_actual_weights_without_accumulation():
    from utils.segment_smoke import selector_loss_gradients
    selector = torch.nn.Linear(2, 1, bias=False)
    parameter = selector.weight
    keep = parameter.sum()*2
    sparse = -parameter.sum()*3
    report = selector_loss_gradients(keep, sparse, selector, .5, .1)
    assert abs(report['selector_gradient_cosine'] + 1) < 1e-6
    assert abs(report['sparse_to_keep_gradient_ratio'] - .3) < 1e-6
    assert abs(report['weighted_keep_selector_grad_norm'] - 2**.5) < 1e-6
    assert abs(report['weighted_sparse_selector_grad_norm'] - .3*2**.5) < 1e-6
    assert parameter.grad is None
    (.5*keep+.1*sparse).backward()
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, .7))


def test_zero_classification_gradient_is_explicit_in_balance_report():
    from utils.segment_smoke import selector_loss_gradients
    selector = torch.nn.Linear(2, 1)
    keep, sparse = selector.weight.sum()*0, selector.weight.sum()
    report = selector_loss_gradients(keep, sparse, selector, .5, .03)
    assert report['selector_gradient_cosine'] is None
    assert report['sparse_to_keep_gradient_ratio'] is None
    assert report['weighted_sparse_selector_grad_norm'] > 0
    assert all(p.grad is None for p in selector.parameters())
