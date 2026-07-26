"""End-to-end tests for the assembled Structure DA model."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from methods.structure_da import (
    ComponentStructureClassifier,
    JointStructuralSpaceBuilder,
    SDADiscriminator,
    StructuralAdversarialAdapter,
    StructureDAModel,
    StructureDAForwardOutput,
    classification_loss,
    quality_domain_loss,
    structural_adversarial_loss,
)
from models.ltae import LTAE
from models.pse import PixelSetEncoder


def _make_model(with_extra=False, pse_mlp2=(8, 6)):
    return StructureDAModel(
        num_classes=3,
        input_dim=3,
        with_extra=with_extra,
        extra_size=2,
        pse_mlp1=(3, 4),
        pse_pooling="mean_std",
        pse_mlp2=pse_mlp2,
        n_head=2,
        d_k=2,
        d_model=8,
        ltae_mlp=(8, 6),
        dropout=0.0,
        max_position=64,
        max_temporal_shift=4,
        classifier_hidden=(5,),
        quality_hidden_cap=5,
        quality_eta=0.2,
        sda_hidden_dim=8,
    )


def _batch(batch_size=3, requires_grad=False):
    torch.manual_seed(20 + batch_size)
    pixels = torch.randn(batch_size, 4, 3, 5, requires_grad=requires_grad)
    valid_pixels = torch.ones(batch_size, 4, 5)
    positions = torch.tensor([0, 2, 7, 12])
    extra = torch.randn(batch_size, 2)
    return pixels, valid_pixels, positions, extra


def _quality_modules(model):
    component = model.component_classifier
    return (
        component.trend_temporal_quality,
        component.dynamics_temporal_quality,
        component.dynamics_channel_quality,
        component.trend_component_quality,
        component.dynamics_component_quality,
        component.residual_component_quality,
    )


def _assert_finite_grads(module, require_nonzero=False):
    parameters = tuple(p for p in module.parameters() if p.requires_grad)
    assert parameters
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    if require_nonzero:
        assert any(torch.count_nonzero(p.grad) > 0 for p in parameters)


def _assert_no_grads(module):
    assert all(p.grad is None for p in module.parameters())


def _effective_gates(output):
    gates = output.component.effective_gates
    return (
        gates.beta_trend_temporal,
        gates.beta_dynamics_temporal,
        gates.beta_dynamics_channel,
        gates.q_trend,
        gates.q_dynamics,
        gates.q_residual,
    )


def _raw_gates(output):
    component = output.component
    return (
        component.structural_quality.trend_temporal.raw_gate,
        component.structural_quality.dynamics_temporal.raw_gate,
        component.structural_quality.dynamics_channel.raw_gate,
        component.component_quality.trend.raw_gate,
        component.component_quality.dynamics.raw_gate,
        component.component_quality.residual.raw_gate,
    )


def test_forward_and_details_run_real_pse_to_logits():
    model = _make_model().eval()
    batch = _batch()

    details = model.forward_details(*batch)
    logits = model(*batch)

    assert isinstance(details, StructureDAForwardOutput)
    assert details.pse_features.shape == (3, 4, 6)
    assert details.component.logits.shape == (3, 3)
    assert details.logits.shape == (3, 3)
    torch.testing.assert_close(logits, details.logits)
    with pytest.raises(FrozenInstanceError):
        details.pse_features = torch.zeros_like(details.pse_features)


def test_default_pse_configuration_has_expected_output_dimension():
    model = StructureDAModel(num_classes=3)

    assert model.spatial_encoder.output_dim == 128
    assert model.component_classifier.feature_dim == 128
    assert model.joint_builder.feature_dim == 128


def test_with_extra_adjusts_a_copy_of_pse_mlp2_and_uses_extra():
    requested_mlp2 = [8, 6]
    model = _make_model(with_extra=True, pse_mlp2=requested_mlp2).eval()
    pixels, valid, positions, extra = _batch()

    first = model.forward_details(pixels, valid, positions, extra)
    second = model.forward_details(pixels, valid, positions, extra + 3)

    assert requested_mlp2 == [8, 6]
    assert model.spatial_encoder.mlp2_dim == [10, 6]
    assert not torch.allclose(first.pse_features, second.pse_features)


def test_with_extra_false_accepts_and_ignores_extra_tensor():
    model = _make_model(with_extra=False).eval()
    pixels, valid, positions, extra = _batch()

    first = model.forward_details(pixels, valid, positions, extra)
    second = model.forward_details(pixels, valid, positions, extra + 1000)

    torch.testing.assert_close(first.pse_features, second.pse_features)
    torch.testing.assert_close(first.logits, second.logits)


def test_invalid_pixels_valid_pixels_or_positions_type_is_rejected():
    model = _make_model()
    pixels, valid, positions, extra = _batch()

    for arguments in (
        ([1], valid, positions, extra),
        (pixels, [1], positions, extra),
        (pixels, valid, [0, 1, 2, 3], extra),
    ):
        with pytest.raises(ValueError):
            model.forward_details(*arguments)


def test_masked_pixel_values_do_not_affect_pse_features_or_logits():
    model = _make_model().eval()
    pixels, valid, positions, extra = _batch()
    valid[..., -1] = 0
    changed = pixels.clone()
    changed[..., -1] = 1e6

    original = model.forward_details(pixels, valid, positions, extra)
    modified = model.forward_details(changed, valid, positions, extra)

    torch.testing.assert_close(original.pse_features, modified.pse_features)
    torch.testing.assert_close(original.logits, modified.logits)


def test_valid_pixels_is_never_forwarded_as_temporal_mask(monkeypatch):
    model = _make_model().eval()
    pixels, valid, positions, extra = _batch()
    valid[..., 1:] = 0
    captured = []
    original = model.component_classifier.forward

    def spy(features, component_positions, time_mask=None, quality_progress=1.0):
        captured.append(time_mask)
        return original(
            features,
            component_positions,
            time_mask=time_mask,
            quality_progress=quality_progress,
        )

    monkeypatch.setattr(model.component_classifier, "forward", spy)

    model.forward_details(pixels, valid, positions, extra)

    assert captured == [None]


def test_quality_progress_is_forwarded_without_recalculation():
    model = _make_model().eval()
    batch = _batch()

    zero = model.forward_details(*batch, quality_progress=0.0)
    full = model.forward_details(*batch, quality_progress=1.0)

    assert all(torch.equal(gate, torch.ones_like(gate)) for gate in _effective_gates(zero))
    for effective, raw in zip(_effective_gates(full), _raw_gates(full)):
        torch.testing.assert_close(effective, raw.detach())


def test_build_joint_structure_exactly_delegates_to_owned_builder():
    model = _make_model().eval()
    details = model.forward_details(*_batch())
    component = details.component

    actual = model.build_joint_structure(details)
    expected = model.joint_builder(
        component.trend_temporal.statistic,
        component.dynamics_temporal.statistic,
        component.dynamics_channel.statistic,
        component.effective_gates.beta_trend_temporal,
        component.effective_gates.beta_dynamics_temporal,
        component.effective_gates.beta_dynamics_channel,
    )

    torch.testing.assert_close(actual.joint, expected.joint)
    torch.testing.assert_close(actual.trend_temporal, expected.trend_temporal)
    torch.testing.assert_close(actual.dynamics_temporal, expected.dynamics_temporal)
    torch.testing.assert_close(actual.dynamics_channel, expected.dynamics_channel)


def test_adapt_supports_unequal_batches_with_owned_shared_discriminator():
    model = _make_model().eval()
    source = model.forward_details(*_batch(3))
    target = model.forward_details(*_batch(5))
    calls = []
    hook = model.adversarial_adapter.discriminator.register_forward_hook(
        lambda _module, arguments, _output: calls.append(arguments[0].shape)
    )

    output = model.adapt(source, target, grl_coefficient=0.6)
    hook.remove()

    assert output.source_logits.shape == (3,)
    assert output.target_logits.shape == (5,)
    assert calls == [torch.Size([3, model.joint_builder.joint_dim]), torch.Size([5, model.joint_builder.joint_dim])]


def test_module_ownership_state_dict_and_parameter_registration_are_complete():
    model = _make_model()

    assert sum(isinstance(m, PixelSetEncoder) for m in model.modules()) == 1
    assert sum(isinstance(m, ComponentStructureClassifier) for m in model.modules()) == 1
    assert sum(isinstance(m, JointStructuralSpaceBuilder) for m in model.modules()) == 1
    assert sum(isinstance(m, StructuralAdversarialAdapter) for m in model.modules()) == 1
    assert sum(isinstance(m, SDADiscriminator) for m in model.modules()) == 1
    assert sum(isinstance(m, LTAE) for m in model.modules()) == 1
    keys = tuple(model.state_dict())
    assert any(key.startswith("spatial_encoder.") for key in keys)
    assert any(key.startswith("component_classifier.") for key in keys)
    assert any(key.startswith("adversarial_adapter.discriminator.") for key in keys)
    model_parameters = {id(parameter) for parameter in model.parameters()}
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
    assert all(
        id(parameter) in model_parameters
        for parameter in model.adversarial_adapter.discriminator.parameters()
    )


def test_source_and_target_forwards_do_not_create_duplicate_modules():
    model = _make_model().eval()
    module_ids = {id(module) for module in model.modules()}

    model.forward_details(*_batch(3))
    model.forward_details(*_batch(5))

    assert {id(module) for module in model.modules()} == module_ids


def test_end_to_end_classification_gradient_route_from_pixels():
    model = _make_model().eval()
    pixels, valid, positions, extra = _batch(4, requires_grad=True)
    details = model.forward_details(pixels, valid, positions, extra)

    classification_loss(details.component, torch.tensor([0, 1, 2, 0])).backward()

    assert pixels.grad is not None and torch.count_nonzero(pixels.grad) > 0
    _assert_finite_grads(model.spatial_encoder, require_nonzero=True)
    _assert_finite_grads(model.component_classifier.decomposition, require_nonzero=True)
    _assert_finite_grads(model.component_classifier.shared_ltae, require_nonzero=True)
    _assert_finite_grads(model.component_classifier.classifier, require_nonzero=True)
    for module in _quality_modules(model):
        _assert_no_grads(module)
    _assert_no_grads(model.adversarial_adapter.discriminator)


def test_end_to_end_quality_domain_gradient_stops_before_pse():
    model = _make_model().eval()
    source_pixels, valid, positions, extra = _batch(3, requires_grad=True)
    target_pixels, target_valid, target_positions, target_extra = _batch(5, requires_grad=True)
    source = model.forward_details(source_pixels, valid, positions, extra)
    target = model.forward_details(target_pixels, target_valid, target_positions, target_extra)

    quality_domain_loss(source.component, target.component).backward()

    assert source_pixels.grad is None and target_pixels.grad is None
    _assert_no_grads(model.spatial_encoder)
    _assert_no_grads(model.component_classifier.decomposition)
    _assert_no_grads(model.component_classifier.shared_ltae)
    _assert_no_grads(model.component_classifier.classifier)
    for module in _quality_modules(model):
        _assert_finite_grads(module.transferability, require_nonzero=True)
        _assert_no_grads(module.discriminability)
        if hasattr(module, "diversity"):
            _assert_no_grads(module.diversity)
    _assert_no_grads(model.adversarial_adapter.discriminator)


@pytest.mark.parametrize("coefficient", [1.0, 0.0])
def test_end_to_end_sda_gradient_route_from_pixels(coefficient):
    model = _make_model().eval()
    source_pixels, valid, positions, extra = _batch(3, requires_grad=True)
    target_pixels, target_valid, target_positions, target_extra = _batch(5, requires_grad=True)
    source = model.forward_details(source_pixels, valid, positions, extra)
    target = model.forward_details(target_pixels, target_valid, target_positions, target_extra)
    adaptation = model.adapt(source, target, grl_coefficient=coefficient)

    structural_adversarial_loss(adaptation).backward()

    _assert_finite_grads(model.adversarial_adapter.discriminator, require_nonzero=True)
    _assert_no_grads(model.component_classifier.shared_ltae)
    _assert_no_grads(model.component_classifier.classifier)
    for module in _quality_modules(model):
        _assert_no_grads(module)
    if coefficient == 1.0:
        assert source_pixels.grad is not None and torch.count_nonzero(source_pixels.grad) > 0
        assert target_pixels.grad is not None and torch.count_nonzero(target_pixels.grad) > 0
        _assert_finite_grads(model.spatial_encoder, require_nonzero=True)
        _assert_finite_grads(model.component_classifier.decomposition, require_nonzero=True)
    else:
        for gradient in (source_pixels.grad, target_pixels.grad):
            assert gradient is None or torch.count_nonzero(gradient) == 0
        assert all(
            parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
            for parameter in model.spatial_encoder.parameters()
        )
        assert all(
            parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
            for parameter in model.component_classifier.decomposition.parameters()
        )
