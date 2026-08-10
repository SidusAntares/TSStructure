from __future__ import annotations

import pytest
import torch

from methods.structure_da import FunctionalGeometryOutput, TSStructureForwardOutput, TSStructureModel


def _model(**overrides) -> TSStructureModel:
    options = dict(
        num_classes=3,input_dim=2,mlp1=(2,4,4),mlp2=(8,4),
        time_reference=0.0,time_scale=365.0,
        trend_num_basis=4,structure_num_basis=4,canonical_grid_size=5,
        roughness_grid_size=64,trend_smoothing=1e-2,structure_smoothing=1e-3,
        n_head=1,d_k=2,d_model=8,ltae_mlp=(8,4),dropout=0.0,
        classifier_hidden=(4,),max_initial_frequency=4.0,
    )
    options.update(overrides)
    return TSStructureModel(**options)


def _inputs(batch=3,length=5):
    torch.manual_seed(901+length)
    pixels=torch.randn(batch,length,2,4)
    valid=torch.ones(batch,length,4,dtype=torch.bool)
    positions=torch.linspace(0,300,length).round().long()
    return pixels,valid,positions


def test_task_forward_is_pse_to_single_ltae() -> None:
    model=_model().eval(); pixels,valid,positions=_inputs()
    output=model(pixels,valid,positions,return_geometry=False)
    assert isinstance(output,TSStructureForwardOutput)
    assert output.latent.shape==(3,5,4)
    assert output.fused_repr.shape==(3,4)
    assert output.logits.shape==(3,3)
    assert output.geometry is None
    assert output.trend is None and output.structure is None
    assert output.dynamics is None and output.residual is None
    assert not hasattr(output,'trend_repr') and not hasattr(output,'structure_repr')



def test_phase_only_freezes_decomposition_parameters() -> None:
    model = _model()
    assert list(model.backbone.decomposition.parameters())
    assert all(not parameter.requires_grad for parameter in model.backbone.decomposition.parameters())
    assert any(parameter.requires_grad for parameter in model.backbone.pixel_set_encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.temporal_module.raw_encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.classifier.parameters())

def test_geometry_forward_retains_decomposition_only_as_side_path() -> None:
    model=_model().eval(); pixels,valid,positions=_inputs()
    output=model(pixels,valid,positions,return_geometry=True)
    assert output.trend.shape==(3,5,4)
    assert output.structure.shape==(3,5,4)
    assert isinstance(output.geometry,FunctionalGeometryOutput)
    assert output.geometry.trend_srvf.shape==(3,5,4)
    assert output.geometry.structure_srvf.shape==(3,5,4)
    assert output.fused_repr.shape==(3,4)


def test_ce_does_not_use_decomposition(monkeypatch) -> None:
    model=_model(); pixels,valid,positions=_inputs()
    calls=0
    original=model.backbone.decomposition.forward
    def counted(*args,**kwargs):
        nonlocal calls
        calls+=1
        return original(*args,**kwargs)
    monkeypatch.setattr(model.backbone.decomposition,'forward',counted)
    output=model(pixels,valid,positions,return_geometry=False)
    torch.nn.functional.cross_entropy(output.logits,torch.tensor([0,1,2])).backward()
    assert calls==0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.pixel_set_encoder.parameters())
    assert all(p.grad is None for p in model.backbone.decomposition.parameters())


def test_geometry_path_does_not_change_task_embedding() -> None:
    model=_model().eval(); pixels,valid,positions=_inputs()
    task=model(pixels,valid,positions,return_geometry=False)
    with_geometry=model(pixels,valid,positions,return_geometry=True)
    torch.testing.assert_close(task.fused_repr,with_geometry.fused_repr,rtol=0,atol=0)
    torch.testing.assert_close(task.logits,with_geometry.logits,rtol=0,atol=0)


def test_temporal_position_override_changes_task_path_not_geometry() -> None:
    model=_model().eval(); pixels,valid,positions=_inputs()
    backbone=model.forward_backbone(pixels,valid,positions,None,compute_decomposition=True)
    baseline=model.forward_from_backbone(backbone,positions,return_geometry=True)
    override=(backbone.normalized_positions*0.8).clamp(0,1)
    changed=model.forward_from_backbone(backbone,positions,temporal_positions_override=override,return_geometry=True)
    torch.testing.assert_close(baseline.geometry.structure_srvf,changed.geometry.structure_srvf,rtol=0,atol=0)
    assert not torch.allclose(baseline.fused_repr,changed.fused_repr)


def test_padding_and_mask_do_not_change_valid_outputs() -> None:
    model=_model().eval(); pixels,valid,positions=_inputs(batch=2)
    mask=torch.tensor([[True,True,True,False,False],[True,True,True,True,False]])
    changed_pixels=torch.where(mask[:,:,None,None],pixels,torch.full_like(pixels,999.0))
    changed_positions=torch.where(mask,positions.float().expand(2,-1),torch.full((2,5),-1e9))
    base=model(pixels,valid,positions.float(),time_mask=mask,return_geometry=False)
    changed=model(changed_pixels,valid,changed_positions,time_mask=mask,return_geometry=False)
    torch.testing.assert_close(base.logits,changed.logits,rtol=0,atol=0)


def test_encode_geometry_returns_geometry() -> None:
    model=_model().eval(); pixels,valid,positions=_inputs()
    geometry=model.encode_geometry(pixels,valid,positions)
    assert isinstance(geometry,FunctionalGeometryOutput)


def test_forward_accepts_no_domain_arguments() -> None:
    model=_model(); pixels,valid,positions=_inputs()
    with pytest.raises(TypeError):
        model(pixels,valid,positions,source_labels=torch.zeros(3,dtype=torch.long))
