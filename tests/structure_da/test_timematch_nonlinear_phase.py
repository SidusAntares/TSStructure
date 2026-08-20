import numpy as np
import pytest
import torch
from models.stclassifier import PseLTae
from methods.structure_da.original_timematch import (
    FrozenGeometryCopy,
    OriginalTimeMatchModel,
    module_state_hash,
)

from methods.structure_da.timematch_nonlinear_phase import (
    aggregate_class_residual_phases,
    am_rows_from_probabilities,
    candidate_phase_grid,
    canonical_grid,
    evaluate_candidate_positions,
    evaluate_phase_grid,
    inverse_phase_on_canonical_grid,
    select_alpha_from_probabilities,
    translate_grid_function,
)


def _rho(k=129):
    u=canonical_grid(k)
    return u + 0.04*torch.sin(torch.pi*u)*u*(1-u)


def test_alpha_zero_is_exact_timematch_translation():
    rho=_rho()
    pos=torch.tensor([[0.1,0.3,0.8]],dtype=torch.float64)
    mask=torch.ones_like(pos,dtype=torch.bool)
    got=evaluate_candidate_positions(pos,delta_days=-21,alpha=0.0,rho_dom=rho,time_scale_days=365.0,time_mask=mask)
    expected=pos-21.0/365.0
    assert torch.equal(got,expected)


def test_alpha_one_matches_declared_additive_candidate():
    rho=_rho();u=canonical_grid(rho.numel())
    got=candidate_phase_grid(delta_days=14,alpha=1.0,rho_dom=rho,time_scale_days=365.0)
    assert torch.allclose(got,u+14.0/365.0+(rho-u),atol=0,rtol=0)


def test_candidate_remains_strictly_monotone_for_bank():
    rho=_rho()
    for alpha in (0,.25,.5,.75,1):
        gamma=candidate_phase_grid(delta_days=30,alpha=alpha,rho_dom=rho)
        assert torch.all(gamma[1:]>gamma[:-1])


def test_inverse_of_alpha_zero_is_exact_opposite_scalar_shift():
    rho=canonical_grid(129)
    forward=candidate_phase_grid(delta_days=28,alpha=0,rho_dom=rho)
    inverse=inverse_phase_on_canonical_grid(forward)
    expected=canonical_grid(129)-28.0/365.0
    assert torch.allclose(inverse,expected,atol=1e-12,rtol=0)


def test_phase_grid_evaluation_preserves_padding_zero():
    rho=_rho();gamma=candidate_phase_grid(delta_days=7,alpha=.5,rho_dom=rho)
    pos=torch.tensor([[.2,.6,0.]],dtype=torch.float64);mask=torch.tensor([[1,1,0]],dtype=torch.bool)
    out=evaluate_phase_grid(pos,gamma,time_mask=mask)
    assert out[0,2].item()==0.0
    assert out[0,1]>out[0,0]


def test_class_aggregation_is_equal_weight_not_sample_weighted():
    u=canonical_grid(65)
    a=u+0.02*torch.sin(torch.pi*u)*u*(1-u)
    b=u-0.01*torch.sin(torch.pi*u)*u*(1-u)
    got=aggregate_class_residual_phases([a,b])
    from methods.structure_da.phase_geometry import sqrt_mean_gamma
    assert torch.allclose(got, sqrt_mean_gamma(torch.stack([a,b])))


def test_translate_grid_function_implements_f_corrected_v_equals_f_raw_v_minus_delta():
    u=canonical_grid(101)
    values=u[:,None]
    shifted,support=translate_grid_function(values,delta_days=36.5,time_scale_days=365.0,support=torch.ones_like(u))
    # delta=0.1; at corrected v=.5, raw coordinate=.4.
    assert shifted[50,0].item()==pytest.approx(.4,abs=2e-3)
    assert support[0].item()==0.0
    assert support[-1].item()==1.0


def test_am_alpha_scoring_matches_manual_time_match_formula():
    p=np.asarray([
        [[.9,.1],[.6,.4]],
        [[.2,.8],[.55,.45]],
        [[.8,.2],[.45,.55]],
        [[.1,.9],[.4,.6]],
    ],dtype=np.float64)
    c=np.asarray([.5,.5])
    rows=am_rows_from_probabilities(p,[0,.5],c)
    assert len(rows)==2
    selected=select_alpha_from_probabilities(p,[0,.5],c)
    expected=min(rows,key=lambda r:(r['am_score'],r['alpha']))['alpha']
    assert selected.alpha==expected
    assert selected.margin>=0


def test_invalid_nonmonotone_residual_rejected():
    rho=canonical_grid(9);rho[4]=rho[3]-0.1
    with pytest.raises(ValueError,match="strictly increasing"):
        candidate_phase_grid(delta_days=0,alpha=.5,rho_dom=rho)


def test_original_classifier_is_pse_ltae_decoder_without_decomposition():
    model = OriginalTimeMatchModel(PseLTae(input_dim=10, num_classes=4, with_extra=False))
    pixels = torch.randn(2, 5, 10, 6)
    valid = torch.ones(2, 5, 6)
    positions = torch.arange(5).repeat(2, 1).long() + 100
    output = model(pixels, valid, positions)
    assert output.logits.shape == (2, 4)
    with pytest.raises(AssertionError, match="never invokes decomposition"):
        model.forward_backbone(pixels, valid, positions, compute_decomposition=True)


def test_original_wrapper_matches_upstream_logits_at_integer_positions():
    upstream = PseLTae(input_dim=10, num_classes=4, with_extra=False).eval()
    wrapped = OriginalTimeMatchModel(upstream).eval()
    pixels = torch.randn(2, 5, 10, 6)
    valid = torch.ones(2, 5, 6)
    positions = torch.tensor([[12, 45, 96, 180, 300], [7, 61, 130, 240, 350]])
    with torch.inference_mode():
        expected = upstream(pixels, valid, positions, None)
        actual = wrapped(pixels, valid, positions).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_geometry_pse_is_independent_frozen_eval_and_hash_stable():
    semantic = OriginalTimeMatchModel(PseLTae(input_dim=10, num_classes=4, with_extra=False))
    geometry = FrozenGeometryCopy(semantic)
    assert geometry.geometry_pse is not semantic.spatial_encoder
    assert not geometry.training and not geometry.geometry_pse.training
    assert all(not p.requires_grad for p in geometry.parameters())
    before = module_state_hash(geometry.geometry_pse)
    with torch.no_grad():
        next(semantic.spatial_encoder.parameters()).add_(1.0)
    geometry.train(True)
    geometry.assert_frozen()
    assert module_state_hash(geometry.geometry_pse) == before


def test_alpha_zero_positions_match_original_shifted_days_exactly():
    rho = canonical_grid(129)
    days = torch.tensor([[30.0, 120.0, 280.0]])
    normalized = days / 365.0
    transformed = evaluate_candidate_positions(
        normalized, delta_days=-17, alpha=0.0, rho_dom=rho,
        time_scale_days=365.0,
    ) * 365.0
    assert torch.allclose(transformed, days - 17.0, atol=1e-7, rtol=0)


def test_geometry_is_required_only_for_a_nonzero_alpha_candidate():
    from methods.structure_da.timematch_nonlinear_phase import requires_nonlinear_phase

    assert requires_nonlinear_phase((0.0,)) is False
    assert requires_nonlinear_phase((0.0, 0.25, 0.5, 0.75, 1.0)) is True
