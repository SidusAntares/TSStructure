import torch
from models.segment_selector import SegmentSelector


def test_restored_pure_content_shared_gates_coverage_and_gradient_routing():
    torch.manual_seed(1)
    selector = SegmentSelector(feature_dim=8, width=16, heads=4, protocol='inshape_coverage_v1')
    assert selector.scorer[0].in_features == 16
    features = torch.randn(4, 12, 8, requires_grad=True)
    attention = torch.ones(4, 16, 12)/12
    segments = torch.arange(12)[None].expand(4, -1)//3
    info = selector(features, attention, segments, variant='inshape')
    assert torch.equal(info['gate'], info['gate'][:, :1].expand_as(info['gate']))
    r = info['raw_coverage']
    torch.testing.assert_close(info['sparse_loss'].detach(), (2*r-1)*r)
    info['sparse_loss'].backward()
    assert features.grad is None
    assert selector.scorer[0].bias.grad.abs().sum() > 0


def test_restored_initial_bias_and_empty_repair_do_not_change_sparse_coverage():
    selector = SegmentSelector(feature_dim=8, width=16, heads=4, protocol='inshape_coverage_v1')
    with torch.no_grad():
        selector.scorer[0].weight.zero_()
    x = torch.randn(2, 6, 8)
    attention = torch.ones(2, 16, 6)/6
    segments = torch.tensor([[0,0,0,1,1,1]]*2)
    result = selector(x, attention, segments, 'inshape', True)
    torch.testing.assert_close(result['probabilities'], torch.full_like(result['probabilities'], .8))
    with torch.no_grad(): selector.scorer[0].bias.copy_(torch.tensor([5., -5.]))
    result = selector(x, attention, segments, 'inshape', True)
    assert result['raw_coverage'] == 0 and result['coverage'].gt(0).all()


def test_restored_five_epoch_training_is_finite_and_keeps_sparse_off_pse(tmp_path):
    from models.stclassifier import PseLTae
    from utils.segment_smoke import train_smoke_steps
    from test_segment_smoke import batch, cfg
    config = cfg(tmp_path)
    config.smoke_epochs, config.steps, config.variant = 5, None, 'inshape'
    config.sparse_weight, config.selector_protocol = 3., 'inshape_coverage_v1'
    model = PseLTae(num_classes=3, with_extra=False)
    selector = SegmentSelector(width=32, heads=4, dropout=.1, protocol=config.selector_protocol)
    result = train_smoke_steps(model, selector, [batch()], [batch()], config)
    assert result['status'] == 'completed' and result['completed_source_epochs'] == 5
    assert result['selector_protocol'] == config.selector_protocol
    assert result['sparse_weight'] == 3
