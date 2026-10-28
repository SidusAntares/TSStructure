import copy

import pytest
import torch


def test_soft_top50_valid_shapes_budget_and_uniform_escape():
    from utils.soft_top50 import soft_top50
    q = torch.zeros(2, 16, 7, requires_grad=True)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    info = soft_top50(q, valid, tau=.2)
    assert info['selection'].shape == (2, 16, 7)
    assert ((info['selection'] >= 0) & (info['selection'] <= 1)).all()
    assert not info['selection'][0, :, 4:].any()
    assert info['hard_target'].sum(-1)[0].tolist() == [2]*16
    assert info['hard_target'].sum(-1)[1].tolist() == [4]*16
    assert info['ratio_loss'].item() == 0
    info['sharpness_loss'].backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
    assert not q.grad[0, :, 4:].any()


def test_keep_normalization_and_keep_all_recovers_original_pooling():
    from utils.soft_top50 import keep_head_embeddings
    weights = torch.softmax(torch.randn(3, 16, 17), -1)
    values = torch.randn(3, 16, 17, 16)
    pooled, normalized, mass = keep_head_embeddings(weights, values, torch.ones_like(weights))
    assert torch.allclose(pooled, (weights[..., None]*values).sum(-2).flatten(1), atol=1e-6)
    assert torch.allclose(normalized.sum(-1), torch.ones(3, 16))
    assert torch.allclose(mass, torch.ones(3, 16))


@pytest.mark.parametrize('steps', [15, 30, 47])
def test_m1_full_compatibility_and_bn_updates_once(steps):
    from models.stclassifier import PseLTae
    torch.manual_seed(2)
    ordinary = PseLTae(num_classes=4, with_extra=False).train()
    m1 = copy.deepcopy(ordinary)
    pixels, mask = torch.rand(4, steps, 10, 8), torch.ones(4, steps, 8)
    positions = torch.arange(steps)[None].repeat(4, 1)*5
    extra = torch.zeros(4, 4)
    torch.manual_seed(1)
    expected = ordinary(pixels, mask, positions, extra)
    torch.manual_seed(1)
    full, keep, info = m1.forward_m1(pixels, mask, positions, extra, tau=.2)
    assert torch.equal(full, expected)
    assert info['selection'].shape == (4, 16, steps)
    assert torch.isfinite(keep).all()
    assert all(torch.equal(value, m1.state_dict()[key]) for key, value in ordinary.state_dict().items())
    (torch.nn.functional.cross_entropy(full, torch.arange(4))
     + torch.nn.functional.cross_entropy(keep, torch.arange(4))
     + info['ratio_loss'] + info['sharpness_loss']).backward()
    assert all(torch.isfinite(p.grad).all() for p in m1.parameters() if p.grad is not None)


def test_keep_loss_alone_reaches_query_key_and_pse():
    from models.stclassifier import PseLTae
    torch.manual_seed(7)
    model = PseLTae(num_classes=4, with_extra=False).train()
    full, keep, info = model.forward_m1(torch.rand(4, 30, 10, 8), torch.ones(4, 30, 8),
                                       torch.arange(30)[None].repeat(4, 1)*10, torch.zeros(4, 4), tau=.3)
    torch.nn.functional.cross_entropy(keep, torch.arange(4)).backward()
    for name in ('temporal_encoder.attention_heads.query', 'temporal_encoder.attention_heads.key.weight',
                 'spatial_encoder.mlp1.0.linear.weight'):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0, name


def test_invalid_times_never_enter_attention_keep_or_selection():
    from models.stclassifier import PseLTae
    model = PseLTae(num_classes=3, with_extra=False).eval()
    pixels, mask = torch.rand(2, 10, 10, 8), torch.ones(2, 10, 8)
    mask[0, 6:] = 0
    positions = torch.arange(10)[None].repeat(2, 1)
    positions[0, 6:] = 9999
    full, keep, info = model.forward_m1(pixels, mask, positions, torch.zeros(2, 4), tau=.2)
    assert torch.isfinite(full).all() and torch.isfinite(keep).all()
    assert not info['selection'][0, :, 6:].any()
    assert not info['keep_attention'][0, :, 6:].any()
    assert not info['attention'][0, :, 6:].any()


def test_short_optimization_sharpens_uniform_gate_and_checkpoint_roundtrip(tmp_path):
    from utils.soft_top50 import soft_top50
    from models.stclassifier import PseLTae
    scores = torch.nn.Parameter(torch.zeros(2, 16, 30))
    optimizer = torch.optim.Adam([scores], lr=.05)
    for _ in range(25):
        optimizer.zero_grad()
        info = soft_top50(scores, torch.ones(2, 30, dtype=torch.bool), tau=.2)
        (info['ratio_loss'] + info['sharpness_loss']).backward()
        optimizer.step()
    assert info['selection'].std(-1).mean() > .3
    assert (info['selection'].mean(-1)-.5).abs().max() < .02
    model = PseLTae(num_classes=3, with_extra=False)
    path = tmp_path / 'model.pt'
    torch.save({'state_dict': model.state_dict()}, path)
    restored = PseLTae(num_classes=3, with_extra=False)
    restored.load_state_dict(torch.load(path, weights_only=False)['state_dict'])
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
