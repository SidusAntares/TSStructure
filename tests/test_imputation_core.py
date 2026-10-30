import pytest
import torch
from models.timeperceiver_imputer import TimePerceiverImputer
from utils.imputation import linear_interpolation, select_gap_queries, select_recovery_mask, reconstruction_loss, probe_attention


def test_linear_visible_only_and_no_extrapolation():
    x = torch.tensor([[[0.], [999.], [10.]]])
    t = torch.tensor([[0., 4., 10.]])
    valid = torch.tensor([[True, False, True]])
    assert linear_interpolation(x, t, torch.tensor([[5.]]), valid).item() == 5
    with pytest.raises(ValueError, match='bracket'):
        linear_interpolation(x, t, torch.tensor([[11.]]), valid)


def test_gap_threshold_dedup_and_padding():
    t = torch.tensor([[0., 20., 60., 101.], [0., 20., 0., 0.]])
    a = torch.ones(2, 2, 4)
    v = torch.tensor([[1,1,1,1], [1,1,0,0]], dtype=torch.bool)
    result = select_gap_queries(t, a, v, top_fraction=1., generator=torch.Generator().manual_seed(4))
    q = result['dates'][0, result['valid'][0]].tolist()
    assert len(q) == len(set(q)) and len(q) <= 8
    assert not result['valid'][1].any()
    assert all(20 < d < 101 for d in q)
    assert 40. in q or any(60 < d < 101 for d in q)
    strict = select_gap_queries(torch.tensor([[0.,40.]]), torch.ones(1,1,2), top_fraction=1.)
    assert strict['dates'][strict['valid']].tolist() == [20.]


def test_recovery_both_sides_and_exterior_brackets():
    dates = torch.arange(9.).unsqueeze(0) * 10
    attention = torch.zeros(1, 2, 9)
    attention[:,:,4] = 10
    mask = select_recovery_mask(dates, attention, generator=torch.Generator().manual_seed(5))
    assert mask.sum() in (2,3)
    assert mask[0,:4].any() and mask[0,5:].any()
    assert not mask[0,4] and not mask[0,0] and not mask[0,-1]
    assert not select_recovery_mask(dates[:,:3],attention[:,:,:3]).any()


def test_imputer_padding_gradient_and_empty_queries():
    torch.manual_seed(2)
    model = TimePerceiverImputer(feature_dim=4, d_model=16, num_latents=3, rounds=2, self_blocks=1, heads=2, dropout=0)
    x = torch.randn(2,4,4,requires_grad=True)
    dates = torch.tensor([[0.,10.,20.,999.],[0.,5.,10.,20.]])
    valid = torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.bool)
    queries = torch.tensor([[5.,15.],[2.5,15.]])
    result = model(x, dates, queries, valid)
    assert result['prediction'].shape == (2,2,4)
    assert result['log_variance'].min() >= -6 and result['log_variance'].max() <= 4
    changed = x.detach().clone(); changed[0,3] = 1e8
    altered = dates.clone(); altered[0,3] = -1e8
    assert torch.allclose(result['prediction'],model(changed,altered,queries,valid)['prediction'])
    loss = reconstruction_loss(result['prediction'],torch.zeros_like(result['prediction']), result['log_variance'])
    loss.backward()
    assert x.grad[0,3].abs().sum() == 0
    assert model.latents.grad.abs().sum() > 0
    assert model(x,dates,queries[:,:0],valid)['prediction'].shape == (2,0,4)


def test_probe_restores_modes_and_uses_shift():
    from models.ltae import LTAE
    encoder = LTAE(in_channels=4,n_head=2,d_k=2,n_neurons=[8,4],d_model=8,dropout=.9)
    encoder.train(); encoder.mlp.eval()
    x = torch.randn(2,3,4,requires_grad=True); dates=torch.tensor([[0,10,30],[1,11,31]])
    modes=[m.training for m in encoder.modules()]
    got=probe_attention(encoder,x,dates,shift=2)
    encoder.eval()
    with torch.no_grad(): expected=encoder(x,dates+2,return_head_parts=True)[1]
    for m, mode in zip(encoder.modules(),modes): m.training=mode
    assert torch.allclose(got,expected) and not got.requires_grad
    assert [m.training for m in encoder.modules()] == modes

def test_padded_sequence_matches_unpadded_and_query_padding():
    torch.manual_seed(3)
    model=TimePerceiverImputer(feature_dim=4,d_model=16,num_latents=3,rounds=1,self_blocks=1,heads=2,dropout=0).eval()
    x=torch.randn(1,3,4)
    dates=torch.tensor([[0.,10.,30.]])
    query=torch.tensor([[5.5]])
    expected=model(x,dates,query)
    xp=torch.cat([x,torch.full((1,2,4),float('nan'))],dim=1)
    dp=torch.cat([dates,torch.full((1,2),float('nan'))],dim=1)
    got=model(xp,dp,torch.tensor([[5.5,float('nan')]]),
              valid=torch.tensor([[1,1,1,0,0]],dtype=torch.bool),query_valid=torch.tensor([[1,0]],dtype=torch.bool))
    assert torch.allclose(got['prediction'][:,:1],expected['prediction'],atol=1e-6)
    assert got['prediction'][:,1:].eq(0).all() and torch.isfinite(got['log_variance']).all()


def test_recovery_seeded_and_gap_selector_ignores_padding_attention():
    dates=torch.tensor([[0.,10.,20.,70.,80.,90.,100.,110.,120.]])
    attention=torch.zeros(1,4,9); attention[:,:,3]=10
    one=select_recovery_mask(dates,attention,generator=torch.Generator().manual_seed(8))
    two=select_recovery_mask(dates,attention,generator=torch.Generator().manual_seed(8))
    assert torch.equal(one,two)
    assert not one[0,3] and one.sum() in (2,3)
    selected=select_gap_queries(torch.tensor([[0.,20.,999.]]),torch.tensor([[[1.,2.,1000.]]]),torch.tensor([[1,1,0]],dtype=torch.bool))
    assert selected['dates'].shape==(1,0)


def test_gap_selector_two_anchors_eight_queries_and_date_union():
    dates=torch.tensor([[0.,60.,120.,180.,240.,300.,360.]])
    attention=torch.zeros(1,2,7); attention[0,0,1]=10;attention[0,1,4]=10
    result=select_gap_queries(dates,attention,max_anchors=2,top_fraction=.1)
    assert result['anchor_count'].item()==2 and result['query_count'].item()==8
    assert set(result['dates'][0].tolist())=={20.,40.,80.,100.,200.,220.,260.,280.}


def test_probe_padding_excluded_and_batchnorm_unchanged():
    from models.ltae import LTAE
    encoder=LTAE(in_channels=4,n_head=2,d_k=2,n_neurons=[8,4],d_model=8).train()
    states={name:value.clone() for name,value in encoder.named_buffers()}
    x=torch.randn(2,4,4); dates=torch.tensor([[0,10,20,0],[0,10,20,30]])
    valid=torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.bool)
    result=probe_attention(encoder,x,dates,valid=valid)
    assert result[0,:,3].eq(0).all() and torch.allclose(result.sum(-1),torch.ones(2,2))
    assert all(torch.equal(value,states[name]) for name,value in encoder.named_buffers())

def test_query_anchor_context_uses_nearest_two_anchors_and_safe_fallback():
    model=TimePerceiverImputer(feature_dim=4,d_model=16,num_latents=3,rounds=1,self_blocks=1,heads=2,dropout=0)
    assert hasattr(model,'anchor_cross'), 'Queries require explicit local anchor attention'
    captured=[]
    hook=model.anchor_cross.register_forward_pre_hook(lambda module,args: captured.append(args[1].detach().clone()))
    tokens=torch.arange(5.).reshape(1,5,1).expand(1,5,16)
    dates=torch.tensor([[0.,10.,20.,30.,40.]])
    queries=torch.tensor([[25.]])
    valid=torch.ones_like(dates,dtype=torch.bool)
    anchors=torch.tensor([[1,0,1,0,1]],dtype=torch.bool)
    result=model._anchor_context(torch.zeros(1,1,16),tokens,dates,queries,valid,torch.ones(1,1,dtype=torch.bool),anchors)
    assert set(captured[-1][0,:,0].tolist())=={2.,4.}
    assert torch.isfinite(result).all()
    result=model._anchor_context(torch.zeros(1,1,16),tokens,dates,queries,valid,torch.ones(1,1,dtype=torch.bool),torch.zeros_like(anchors))
    assert set(captured[-1][0,:,0].tolist())=={2.,3.}
    assert torch.isfinite(result).all()
    hook.remove()
