import copy
import itertools

import numpy as np
import pytest
import torch


def test_pelt_matches_exhaustive_partitions():
    from utils.segment_structure import segment_pelt, _rbf_cost
    rng = np.random.default_rng(8)
    for n in range(3, 16):
        for minimum in (1, 2, 3):
            x = rng.normal(size=(n, 4))
            cost = _rbf_cost(x)
            beta = .13 * n
            best = float('inf')
            for bits in itertools.product((0, 1), repeat=n-1):
                boundaries = [0] + [i+1 for i, bit in enumerate(bits) if bit] + [n]
                if min(np.diff(boundaries)) < minimum:
                    continue
                best = min(best, sum(cost(a, b) for a, b in zip(boundaries, boundaries[1:])) + beta*(len(boundaries)-2))
            labels = segment_pelt(torch.tensor(x)[None], penalty=.13, min_size=minimum)[0].numpy()
            cuts = [0] + (np.flatnonzero(np.diff(labels))+1).tolist() + [n]
            actual = sum(cost(a, b) for a, b in zip(cuts, cuts[1:])) + beta*(len(cuts)-2)
            assert actual == pytest.approx(best)


def test_selector_hard_mask_sparse_gradient_and_padding():
    from models.segment_selector import SegmentSelector
    features = torch.randn(2, 9, 8, requires_grad=True)
    attention = torch.randn(2, 4, 9).softmax(-1).requires_grad_()
    segments = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 0, 1, 1, 1, -1, -1, -1]])
    selector = SegmentSelector(feature_dim=8, width=16, heads=4)
    result = selector(features, attention, segments)
    assert set(result['gate'].detach().unique().tolist()) <= {0., 1.}
    assert (result['gate'].sum(-1) >= 1).all()
    assert result['point_mask'][1, :, 6:].eq(0).all()
    assert result['gate'][1, :, 2:].eq(0).all()
    assert torch.allclose(result['sparse_loss'], result['probabilities'].sum(-1).mean())
    (result['point_mask'].square().mean() + result['sparse_loss']).backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert attention.grad is None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in selector.parameters())
    a = selector(features, attention, segments, variant='inshape', deterministic=True)
    b = selector(features, attention, segments, variant='inshape', deterministic=True)
    assert torch.equal(a['gate'], b['gate'])
    assert torch.equal(a['probabilities'][:, 0], a['probabilities'][:, 1])


def _model():
    from models.stclassifier import PseLTae
    return PseLTae(input_dim=3, mlp1=[3, 8], pooling='mean_std', mlp2=[16, 8],
                   with_extra=False, n_head=4, d_k=2, d_model=16,
                   mlp3=[16, 8], mlp4=[8, 8], num_classes=3, dropout=.2)


def test_variants_control_length_metadata_and_change_only_attention_inputs():
    from models.segment_selector import SegmentSelector
    selector = SegmentSelector(feature_dim=4, width=8, heads=2).eval()
    features = torch.randn(1, 6, 4)
    attention = torch.randn(1, 2, 6).softmax(-1)
    segments = torch.tensor([[0, 0, 1, 1, 1, 1]])
    inputs = []
    hook = selector.scorer[0].register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    selector(features, attention, segments, variant='inshape', deterministic=True)
    selector(features, attention, segments, variant='attention', deterministic=True)
    hook.remove()
    assert inputs[0][..., -3:-1].eq(0).all()
    assert inputs[1][..., -3:-1].gt(0).all()
    torch.testing.assert_close(inputs[0][..., :-3], inputs[1][..., :-3], rtol=0, atol=0)
    torch.testing.assert_close(inputs[0][..., -1], inputs[1][..., -1], rtol=0, atol=0)


def test_absolute_penalty_matches_diagnostic_without_changing_legacy_default():
    from utils.pelt_diagnostic import make_cost, pelt_boundaries
    from utils.segment_structure import segment_pelt
    for length in (12, 30, 58):
        features = torch.randn(2, length, 8)
        got = segment_pelt(features, penalty=.8, penalty_mode='absolute')
        legacy = segment_pelt(features, penalty=.1)
        explicit = segment_pelt(features, penalty=.1, penalty_mode='per_observation')
        assert torch.equal(legacy, explicit)
        for b in range(2):
            cost, _ = make_cost(features[b].numpy())
            bounds = pelt_boundaries(cost, length, .8, 3)
            expected = torch.empty(length, dtype=torch.long)
            for segment, (start, end) in enumerate(zip(bounds, bounds[1:])):
                expected[start:end] = segment
            assert torch.equal(got[b], expected)


def test_structure_full_matches_original_and_keep_gradients():
    from models.segment_selector import SegmentSelector
    from utils.segment_structure import structure_forward
    torch.manual_seed(7)
    model = _model().train()
    baseline = copy.deepcopy(model)
    selector = SegmentSelector(feature_dim=8, width=16, heads=4)
    pixels = torch.randn(3, 12, 3, 5)
    mask = torch.ones(3, 12, 5)
    positions = torch.arange(12)[None].expand(3, -1)
    torch.manual_seed(13)
    expected = baseline(pixels, mask, positions, None)
    torch.manual_seed(13)
    full, keep, info = structure_forward(model, selector, pixels, mask, positions, None, min_size=2, penalty=.01)
    assert torch.equal(full, expected)
    for name, buffer in model.named_buffers():
        if 'running_' in name or 'num_batches_tracked' in name:
            assert torch.equal(buffer, dict(baseline.named_buffers())[name]), name
    (keep.square().mean() + .01*info['sparse_loss']).backward()
    for parameter in (model.temporal_encoder.attention_heads.query,
                      model.temporal_encoder.attention_heads.key.weight,
                      model.spatial_encoder.mlp1[0].linear.weight,
                      selector.scorer[0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert info['keep_attention'][info['point_mask'].detach() == 0].eq(0).all()


def test_target_only_one_decoder_bn_update_and_padding():
    from models.segment_selector import SegmentSelector
    from utils.segment_structure import structure_forward
    model = _model().train()
    selector = SegmentSelector(feature_dim=8, width=16, heads=4)
    pixels = torch.randn(3, 12, 3, 5)
    mask = torch.ones(3, 12, 5)
    mask[0, 8:] = 0
    full, keep, info = structure_forward(model, selector, pixels, mask,
        torch.arange(12)[None].expand(3, -1), None, full_loss=False)
    assert full is None and keep.shape == (3, 3)
    assert info['point_mask'][0, :, 8:].eq(0).all()
    for root in (model.temporal_encoder.mlp, model.decoder):
        for module in root.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                assert module.num_batches_tracked == 1


def test_pelt_matches_dynamic_programming_and_handles_invalid_tail():
    from utils.segment_structure import _pelt_boundaries, _rbf_cost, segment_pelt
    rng = np.random.default_rng(18)
    for _ in range(150):
        n = int(rng.integers(6, 40))
        minimum = int(rng.integers(1, 5))
        x = rng.normal(size=(n, 3))
        beta = float(rng.uniform(0, 2))
        cost = _rbf_cost(x)
        objective = np.full(n+1, np.inf)
        objective[0] = -beta
        for end in range(minimum, n+1):
            objective[end] = min(objective[start]+cost(start, end)+beta
                                 for start in range(end-minimum+1))
        cuts = _pelt_boundaries(x, beta, minimum)
        actual = sum(cost(a, b) for a, b in zip(cuts, cuts[1:])) + beta*(len(cuts)-2)
        assert actual == pytest.approx(objective[n])
    x = torch.randn(2, 9, 8, requires_grad=True)
    valid = torch.tensor([[1, 1, 0, 1, 1, 1, 0, 0, 0], [1]*9], dtype=torch.bool)
    labels = segment_pelt(x, valid, min_size=3)
    assert labels[~valid].eq(-1).all()
    assert not labels.requires_grad
    for row, observed in zip(labels, valid):
        values = row[observed]
        assert values[0] == 0
        assert ((values[1:]-values[:-1]) <= 1).all()
        assert (values[1:] >= values[:-1]).all()


def test_zero_selection_repair_is_shared_and_sparse_loss_still_live():
    from models.segment_selector import SegmentSelector
    selector = SegmentSelector(feature_dim=8, width=16, heads=4)
    with torch.no_grad():
        selector.scorer[-1].weight.zero_()
        selector.scorer[-1].bias.fill_(-5)
    features = torch.randn(2, 6, 8)
    attention = torch.full((2, 4, 6), 1/6)
    segments = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]])
    for variant in ('inshape', 'attention'):
        result = selector(features, attention, segments, variant, deterministic=True)
        assert result['fallback_rate'] == 1
        assert result['gate'].sum(-1).eq(1).all()
        assert result['gate'][..., 0].eq(1).all()
        result['sparse_loss'].backward()
        assert selector.scorer[-1].bias.grad.abs().sum() > 0
        selector.zero_grad()


def test_reinmax_backward_matches_official_tau_one_equations():
    from models.segment_selector import _ReinMax
    logits = torch.randn(5, 2, requires_grad=True)
    hard, soft = _ReinMax.apply(logits)
    grad_hard, grad_soft = torch.randn_like(hard), torch.randn_like(soft)
    (hard*grad_hard + soft*grad_soft).sum().backward()
    shifted = .5*(soft.detach()+hard.detach())
    first = 2*grad_hard*shifted
    first = first-shifted*first.sum(-1, keepdim=True)
    zeroth = (-.5*grad_hard+grad_soft)*soft.detach()
    zeroth = zeroth-soft.detach()*zeroth.sum(-1, keepdim=True)
    expected = first+zeroth
    expected = expected-expected.mean(-1, keepdim=True)
    assert torch.allclose(logits.grad, expected)


@pytest.mark.parametrize('training', [False, True])
def test_unified_keep_all_matches_original_logits_gradients_and_bn(training):
    from models.segment_selector import SegmentSelector
    from utils.segment_structure import structure_forward
    torch.manual_seed(19)
    model = _model().train(training)
    baseline = copy.deepcopy(model)
    selector = SegmentSelector(feature_dim=8, width=16, heads=4)
    with torch.no_grad():
        selector.scorer[-1].weight.zero_()
        selector.scorer[-1].bias.fill_(10)
    pixels = torch.randn(4, 12, 3, 5)
    mask = torch.ones(4, 12, 5)
    positions = torch.arange(12)[None].expand(4, -1)
    torch.manual_seed(33)
    expected = baseline(pixels, mask, positions, None)
    torch.manual_seed(33)
    full, keep, info = structure_forward(model, selector, pixels, mask, positions, None,
        full_loss=False, unified=True, deterministic=True, penalty=.01)
    assert full is None and info['coverage'].eq(1).all()
    torch.testing.assert_close(keep, expected, atol=2e-6, rtol=2e-5)
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, dict(baseline.named_buffers())[name], atol=2e-6, rtol=2e-5)
    expected.square().mean().backward()
    (keep.square().mean()+.1*info['sparse_loss']).backward()
    for (name, parameter), (_, reference) in zip(model.named_parameters(), baseline.named_parameters()):
        if reference.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-5, rtol=2e-4, msg=name)


def test_sparse_loss_updates_selector_without_backbone_gradient():
    from models.segment_selector import SegmentSelector
    features = torch.randn(2, 9, 8, requires_grad=True)
    attention = torch.randn(2, 4, 9, requires_grad=True).softmax(-1)
    selector = SegmentSelector(feature_dim=8, width=16, heads=4)
    info = selector(features, attention, torch.arange(9)[None].expand(2, -1)//3)
    parameters = list(selector.parameters())
    expected = torch.autograd.grad(info['probabilities'].sum(-1).mean(), parameters, retain_graph=True)
    actual = torch.autograd.grad(info['sparse_loss'], parameters, retain_graph=True)
    for reference, restricted in zip(expected, actual):
        torch.testing.assert_close(restricted, reference)
    info['sparse_loss'].backward()
    assert features.grad is None
    assert selector.scorer[-1].bias.grad.abs().sum() > 0


def test_unified_partial_keep_classification_updates_backbone_and_selector():
    from models.segment_selector import SegmentSelector
    from utils.segment_structure import structure_forward
    torch.manual_seed(8)
    model = _model().train()
    selector = SegmentSelector(feature_dim=8, width=16, heads=4)
    _, keep, info = structure_forward(model, selector, torch.randn(4, 18, 3, 5),
        torch.ones(4, 18, 5), torch.arange(18)[None].expand(4, -1), None,
        full_loss=False, unified=True, penalty=0, min_size=3)
    assert (info['coverage'] < 1).any()
    keep.square().mean().backward()
    assert selector.scorer[0].weight.grad.abs().sum() > 0
    assert model.spatial_encoder.mlp1[0].linear.weight.grad.abs().sum() > 0
