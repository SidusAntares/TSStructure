from __future__ import annotations

import pytest
import torch

from methods.structure_da import (
    FunctionalGeometryOutput,
    TSStructureForwardOutput,
    TSStructureModel,
)


def _model(**overrides) -> TSStructureModel:
    options = dict(
        num_classes=3,
        input_dim=2,
        mlp1=(2, 4, 4),
        mlp2=(8, 4),
        time_reference=0.0,
        time_scale=365.0,
        trend_num_basis=4,
        structure_num_basis=4,
        canonical_grid_size=5,
        roughness_grid_size=64,
        trend_smoothing=1e-2,
        structure_smoothing=1e-3,
        n_head=1,
        d_k=2,
        d_model=8,
        ltae_mlp=(8, 4),
        dropout=0.0,
        classifier_hidden=(4,),
        max_initial_frequency=4.0,
    )
    options.update(overrides)
    return TSStructureModel(**options)


def _inputs(batch: int = 3, length: int = 5, dtype=torch.float32):
    torch.manual_seed(901 + length)
    pixels = torch.randn(batch, length, 2, 4, dtype=dtype)
    valid = torch.ones(batch, length, 4, dtype=torch.bool)
    positions = torch.linspace(0, 300, length).round().long()
    return pixels, valid, positions


def test_model_forward_output_shapes_and_types() -> None:
    model = _model().eval()
    pixels, valid, positions = _inputs()
    output = model(pixels, valid, positions, return_geometry=True)

    assert isinstance(output, TSStructureForwardOutput)
    assert output.trend.shape == (3, 5, 4)
    assert output.structure.shape == (3, 5, 4)
    assert output.latent.shape == (3, 5, 4)
    assert output.trend_repr.shape == (3, 4)
    assert output.structure_repr.shape == (3, 4)
    assert output.fused_repr.shape == (3, 8)
    assert output.logits.shape == (3, 3)
    assert output.mask.shape == (3, 5)
    assert output.positions.shape == (3, 5)
    assert output.dynamics.shape == (3, 5, 4)
    assert output.residual.shape == (3, 5, 4)

    geometry = output.geometry
    assert isinstance(geometry, FunctionalGeometryOutput)
    assert geometry.trend_srvf.shape == (3, 5, 4)
    assert geometry.structure_srvf.shape == (3, 5, 4)
    assert geometry.trend_support.shape == (3, 5)
    assert geometry.structure_support.shape == (3, 5)
    assert geometry.canonical_grid.shape == (5,)
    assert geometry.trend_valid.shape == (3,)
    assert geometry.structure_valid.shape == (3,)
    assert geometry.trend_valid.dtype == torch.bool


def test_fused_repr_is_exact_concat_of_components() -> None:
    model = _model().eval()
    pixels, valid, positions = _inputs()
    output = model(pixels, valid, positions, return_geometry=False)

    torch.testing.assert_close(
        output.fused_repr,
        torch.cat([output.trend_repr, output.structure_repr], dim=-1),
        rtol=0,
        atol=0,
    )


def test_raw_path_uses_normalized_physical_positions() -> None:
    model = _model().eval()
    pixels, valid, _ = _inputs(length=3)
    physical = torch.tensor([0.0, 182.5, 365.0])

    output = model(pixels, valid, physical, return_geometry=False)

    expected = torch.tensor([0.0, 0.5, 1.0]).expand(3, -1)
    torch.testing.assert_close(output.positions, expected)
    assert output.geometry is None


def test_shared_ltae_parameter_identity() -> None:
    model = _model()
    ltae = model.temporal_module.raw_encoder.shared_ltae
    assert model.temporal_module.raw_encoder.shared_ltae is ltae
    assert ltae.trend_input_norm is not ltae.structure_input_norm
    assert ltae.trend_output_norm is not ltae.structure_output_norm
    assert (
        ltae.trend_input_projection is ltae.structure_input_projection
    )
    assert hasattr(ltae, "shared_time_encoder")
    assert hasattr(ltae, "attention_heads")
    assert hasattr(ltae, "shared_projection")


def test_geometry_is_differentiable_to_backbone_and_decomposition() -> None:
    model = _model()
    pixels, valid, positions = _inputs()
    output = model(pixels, valid, positions, return_geometry=True)
    valid_mask = output.geometry.structure_valid
    assert valid_mask.any().item()
    model.zero_grad()
    output.geometry.structure_srvf[valid_mask].sum().backward()

    pse_params = list(model.backbone.pixel_set_encoder.parameters())
    decomp_params = list(model.backbone.decomposition.parameters())
    assert any(
        parameter.grad is not None
        and parameter.grad.abs().sum().item() > 0
        for parameter in pse_params
    )
    assert any(
        parameter.grad is not None
        and parameter.grad.abs().sum().item() > 0
        for parameter in decomp_params
    )
    # Fixed B-spline basis must not be registered as trainable parameters.
    assert not any(
        "canonical_basis" in name or "knots" in name
        for name, _ in model.named_parameters()
    )


def test_ce_backpropagates_to_all_raw_path_components() -> None:
    model = _model()
    pixels, valid, positions = _inputs()
    output = model(pixels, valid, positions, return_geometry=False)
    labels = torch.tensor([0, 1, 2])
    model.zero_grad()
    torch.nn.functional.cross_entropy(output.logits, labels).backward()

    def has_grad(module) -> bool:
        return any(
            parameter.grad is not None and parameter.grad.abs().sum().item() > 0
            for parameter in module.parameters()
        )

    assert has_grad(model.backbone.pixel_set_encoder)
    assert has_grad(model.backbone.decomposition)
    ltae = model.temporal_module.raw_encoder.shared_ltae
    assert has_grad(ltae.shared_time_encoder)
    assert has_grad(ltae.shared_input_projection)
    assert has_grad(ltae.attention_heads)
    assert has_grad(ltae.trend_input_norm)
    assert has_grad(ltae.structure_input_norm)
    assert has_grad(ltae.trend_output_norm)
    assert has_grad(ltae.structure_output_norm)
    assert has_grad(model.classifier)


def test_padding_and_mask_do_not_change_valid_outputs() -> None:
    model = _model().eval()
    pixels, valid, positions = _inputs(batch=2)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]],
        dtype=torch.bool,
    )
    changed_pixels = pixels.clone()
    changed_pixels = torch.where(
        mask[:, :, None, None], changed_pixels, torch.full_like(changed_pixels, 999.0)
    )
    changed_positions = positions.clone().float()
    changed_positions = torch.where(
        mask, changed_positions, torch.full_like(changed_positions, -1e9)
    )

    baseline = model(pixels, valid, positions.float(), time_mask=mask, return_geometry=False)
    changed = model(
        changed_pixels, valid, changed_positions, time_mask=mask, return_geometry=False
    )

    torch.testing.assert_close(changed.logits, baseline.logits, rtol=0, atol=0)
    torch.testing.assert_close(
        changed.fused_repr, baseline.fused_repr, rtol=0, atol=0
    )


def test_model_state_contains_no_old_components() -> None:
    model = _model()
    state_keys = set(model.state_dict())
    forbidden = (
        "warp",
        "candidate_base",
        "shape_encoder",
        "z_shape",
        "quality",
        "domain_discriminator",
        "running_srvf",
        "running_support",
        "accepted_gamma",
        "phase_center",
    )
    assert not any(name in key for key in state_keys for name in forbidden)
    # Ensure a legitimate component name is not falsely flagged.
    assert "decomposition" in "|".join(state_keys)


def test_return_geometry_false_skips_functional_fit(monkeypatch) -> None:
    model = _model()
    pixels, valid, positions = _inputs()
    calls = 0
    original = model.temporal_module.trend_geometry.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.temporal_module.trend_geometry, "forward", counted)
    model(pixels, valid, positions, return_geometry=False)
    assert calls == 0

    model(pixels, valid, positions, return_geometry=True)
    assert calls == 1


def test_encode_geometry_returns_only_geometry() -> None:
    model = _model().eval()
    pixels, valid, positions = _inputs()
    geometry = model.encode_geometry(pixels, valid, positions)
    assert isinstance(geometry, FunctionalGeometryOutput)
    assert geometry.structure_srvf.shape == (3, 5, 4)


def test_forward_accepts_no_domain_arguments() -> None:
    model = _model()
    pixels, valid, positions = _inputs()
    with pytest.raises(TypeError):
        model(pixels, valid, positions, source_labels=torch.zeros(3, dtype=torch.long))
    with pytest.raises(TypeError):
        model(pixels, valid, positions, domain_score_weight=1.0)
