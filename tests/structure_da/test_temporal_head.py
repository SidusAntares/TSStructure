from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from methods.structure_da import (
    PhaseCoordinateEncoder,
    ShapeCoordinateEncoder,
    TemporalCoordinateOutput,
    TemporalRegistrationOutput,
    TemporalShapePhaseCoordinates,
    TemporalStructureEncoder,
    TemporalStructureFeatureOutput,
    TemporalStructureOutputHead,
)
from methods.structure_da.temporal_head import ShapeFeatureEncoder, ShapeFeatureOutput


def _coordinate_output(
    *,
    batch_size: int = 3,
    num_shape_basis: int = 4,
    attribute_dim: int = 5,
    num_phase_basis: int = 3,
    grid_size: int = 7,
    dtype: torch.dtype = torch.float32,
    valid: torch.Tensor | None = None,
    requires_grad: bool = False,
) -> TemporalCoordinateOutput:
    shape = torch.randn(
        batch_size,
        num_shape_basis,
        attribute_dim,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    phase = torch.randn(
        batch_size,
        num_phase_basis + 1,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    if valid is None:
        valid = torch.ones(batch_size, dtype=torch.bool)
    return TemporalCoordinateOutput(
        shape_coordinates=shape,
        phase_coordinates=phase,
        shape_time_coefficients=torch.randn(
            batch_size, num_shape_basis, attribute_dim, dtype=dtype
        ),
        phase_basis_coefficients=torch.randn(
            batch_size, num_phase_basis, dtype=dtype
        ),
        phase_magnitude=torch.randn(batch_size, dtype=dtype),
        shape_support=torch.rand(batch_size, grid_size, dtype=dtype),
        shape_basis_support=torch.rand(
            batch_size, num_shape_basis, dtype=dtype
        ),
        shape_residual=torch.randn(
            batch_size, grid_size, attribute_dim, dtype=dtype
        ),
        phase_tangent=torch.randn(batch_size, grid_size - 1, dtype=dtype),
        valid=valid,
    )


def _encoder(*, dropout: float = 0.0) -> TemporalStructureEncoder:
    return TemporalStructureEncoder(
        num_shape_basis=4,
        attribute_projection_dim=5,
        num_phase_basis=3,
        coordinate_hidden_dim=8,
        structure_dim=12,
        dropout=dropout,
    )


def test_encoder_networks_have_exact_required_layer_order() -> None:
    encoder = _encoder(dropout=0.25)

    assert [type(layer) for layer in encoder.shape_encoder.network] == [
        nn.Linear,
        nn.GELU,
        nn.Dropout,
        nn.Linear,
    ]
    assert [type(layer) for layer in encoder.phase_encoder.network] == [
        nn.Linear,
        nn.GELU,
        nn.Dropout,
        nn.Linear,
    ]
    assert encoder.shape_encoder.network[0].in_features == 20
    assert encoder.phase_encoder.network[0].in_features == 4
    assert encoder.shape_encoder.network[2].p == 0.25
    assert encoder.phase_encoder.network[2].p == 0.25


def test_output_head_has_exact_required_layer_structure() -> None:
    head = TemporalStructureOutputHead(16, 12, dropout=0.2)

    assert isinstance(head.projection, nn.Linear)
    assert isinstance(head.activation, nn.GELU)
    assert isinstance(head.dropout, nn.Dropout)
    assert isinstance(head.normalization, nn.LayerNorm)
    assert head.projection.in_features == 16
    assert head.projection.out_features == 12
    assert head.dropout.p == 0.2
    assert head.normalization.normalized_shape == (12,)
    assert head.normalization.elementwise_affine
    assert len([m for m in head.modules() if isinstance(m, nn.Linear)]) == 1
    assert len([m for m in head.modules() if isinstance(m, nn.LayerNorm)]) == 1


def test_complete_encoder_contains_no_forbidden_layers_or_names() -> None:
    encoder = _encoder()

    assert not any(
        isinstance(module, (nn.Sigmoid, nn.Softmax, nn.MultiheadAttention))
        for module in encoder.modules()
    )
    assert all(
        forbidden not in name.lower()
        for name, _ in encoder.named_modules()
        for forbidden in ("shared_projector", "attention", "gate", "classifier")
    )


def test_output_shapes_and_valid_identity() -> None:
    encoder = _encoder()
    coordinates = _coordinate_output()

    output = encoder(coordinates)

    assert isinstance(output, TemporalStructureFeatureOutput)
    assert output.shape_embedding.shape == (3, 8)
    assert output.phase_embedding.shape == (3, 8)
    assert output.joint_embedding.shape == (3, 16)
    assert output.feature.shape == (3, 12)
    assert output.valid.shape == (3,)
    assert output.valid is coordinates.valid


def test_shape_phase_and_output_parameters_are_independent() -> None:
    encoder = _encoder()
    shape_parameters = list(encoder.shape_encoder.parameters())
    phase_parameters = list(encoder.phase_encoder.parameters())
    output_parameters = list(encoder.output_head.parameters())

    shape_ids = {id(parameter) for parameter in shape_parameters}
    phase_ids = {id(parameter) for parameter in phase_parameters}
    output_ids = {id(parameter) for parameter in output_parameters}
    assert shape_ids.isdisjoint(phase_ids)
    assert shape_ids.isdisjoint(output_ids)
    assert phase_ids.isdisjoint(output_ids)

    phase_before = [parameter.detach().clone() for parameter in phase_parameters]
    with torch.no_grad():
        for parameter in shape_parameters:
            parameter.add_(1.0)
    for before, after in zip(phase_before, phase_parameters):
        torch.testing.assert_close(after, before)


def test_parameter_scope_is_exactly_three_branches() -> None:
    encoder = _encoder()

    assert set(dict(encoder.named_parameters())) == {
        "shape_encoder.network.0.weight",
        "shape_encoder.network.0.bias",
        "shape_encoder.network.3.weight",
        "shape_encoder.network.3.bias",
        "phase_encoder.network.0.weight",
        "phase_encoder.network.0.bias",
        "phase_encoder.network.3.weight",
        "phase_encoder.network.3.bias",
        "output_head.projection.weight",
        "output_head.projection.bias",
        "output_head.normalization.weight",
        "output_head.normalization.bias",
    }


def test_invalid_rows_are_zero_after_all_bias_and_normalization_layers() -> None:
    encoder = _encoder(dropout=0.0)
    with torch.no_grad():
        for module in encoder.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.fill_(0.75)
            if isinstance(module, nn.LayerNorm):
                module.bias.fill_(0.5)
    coordinates = _coordinate_output(
        valid=torch.tensor([True, False, True])
    )

    output = encoder(coordinates)

    for tensor in (
        output.shape_embedding,
        output.phase_embedding,
        output.joint_embedding,
        output.feature,
    ):
        torch.testing.assert_close(
            tensor[1], torch.zeros_like(tensor[1]), atol=0, rtol=0
        )


def test_manual_forward_matches_all_required_operations() -> None:
    torch.manual_seed(17)
    encoder = _encoder(dropout=0.0).eval()
    coordinates = _coordinate_output()

    output = encoder(coordinates)
    shape_flat = coordinates.shape_coordinates.reshape(3, 20)
    expected_shape = F.linear(
        shape_flat,
        encoder.shape_encoder.network[0].weight,
        encoder.shape_encoder.network[0].bias,
    )
    expected_shape = F.gelu(expected_shape)
    expected_shape = F.linear(
        expected_shape,
        encoder.shape_encoder.network[3].weight,
        encoder.shape_encoder.network[3].bias,
    )
    expected_phase = F.linear(
        coordinates.phase_coordinates,
        encoder.phase_encoder.network[0].weight,
        encoder.phase_encoder.network[0].bias,
    )
    expected_phase = F.gelu(expected_phase)
    expected_phase = F.linear(
        expected_phase,
        encoder.phase_encoder.network[3].weight,
        encoder.phase_encoder.network[3].bias,
    )
    expected_joint = torch.cat([expected_shape, expected_phase], dim=-1)
    expected_feature = F.linear(
        expected_joint,
        encoder.output_head.projection.weight,
        encoder.output_head.projection.bias,
    )
    expected_feature = F.gelu(expected_feature)
    expected_feature = F.layer_norm(
        expected_feature,
        (12,),
        encoder.output_head.normalization.weight,
        encoder.output_head.normalization.bias,
        encoder.output_head.normalization.eps,
    )

    torch.testing.assert_close(output.shape_embedding, expected_shape)
    torch.testing.assert_close(output.phase_embedding, expected_phase)
    torch.testing.assert_close(output.joint_embedding, expected_joint)
    torch.testing.assert_close(output.feature, expected_feature)


def test_diagnostic_fields_do_not_change_feature() -> None:
    encoder = _encoder().eval()
    coordinates = _coordinate_output()
    baseline = encoder(coordinates)
    changed = replace(
        coordinates,
        shape_time_coefficients=torch.full_like(
            coordinates.shape_time_coefficients, 1e7
        ),
        phase_basis_coefficients=torch.full_like(
            coordinates.phase_basis_coefficients, -1e7
        ),
        phase_magnitude=torch.full_like(coordinates.phase_magnitude, 1e8),
        shape_support=torch.zeros_like(coordinates.shape_support),
        shape_basis_support=torch.full_like(
            coordinates.shape_basis_support, 1e6
        ),
        shape_residual=torch.full_like(coordinates.shape_residual, -1e6),
        phase_tangent=torch.full_like(coordinates.phase_tangent, 1e5),
    )

    modified = encoder(changed)

    torch.testing.assert_close(modified.shape_embedding, baseline.shape_embedding)
    torch.testing.assert_close(modified.phase_embedding, baseline.phase_embedding)
    torch.testing.assert_close(modified.joint_embedding, baseline.joint_embedding)
    torch.testing.assert_close(modified.feature, baseline.feature)


def test_shape_and_phase_branches_are_directly_separate() -> None:
    torch.manual_seed(29)
    encoder = _encoder().eval()
    coordinates = _coordinate_output()
    baseline = encoder(coordinates)

    shape_changed = encoder(
        replace(
            coordinates,
            shape_coordinates=coordinates.shape_coordinates + 2.0,
        )
    )
    assert not torch.allclose(
        shape_changed.shape_embedding, baseline.shape_embedding
    )
    torch.testing.assert_close(
        shape_changed.phase_embedding, baseline.phase_embedding
    )
    assert not torch.allclose(shape_changed.feature, baseline.feature)

    phase_changed = encoder(
        replace(
            coordinates,
            phase_coordinates=coordinates.phase_coordinates - 2.0,
        )
    )
    torch.testing.assert_close(
        phase_changed.shape_embedding, baseline.shape_embedding
    )
    assert not torch.allclose(
        phase_changed.phase_embedding, baseline.phase_embedding
    )
    assert not torch.allclose(phase_changed.feature, baseline.feature)


def test_gradients_cover_inputs_and_every_trainable_layer() -> None:
    encoder = _encoder(dropout=0.0)
    coordinates = _coordinate_output(
        valid=torch.tensor([True, False, True]), requires_grad=True
    )
    output = encoder(coordinates)
    feature_weights = torch.arange(1, 13, dtype=output.feature.dtype)

    (output.feature * feature_weights).sum().backward()

    assert coordinates.shape_coordinates.grad is not None
    assert coordinates.phase_coordinates.grad is not None
    assert torch.isfinite(coordinates.shape_coordinates.grad).all()
    assert torch.isfinite(coordinates.phase_coordinates.grad).all()
    assert coordinates.shape_coordinates.grad[0].abs().sum().item() > 0
    assert coordinates.phase_coordinates.grad[0].abs().sum().item() > 0
    torch.testing.assert_close(
        coordinates.shape_coordinates.grad[1],
        torch.zeros_like(coordinates.shape_coordinates.grad[1]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        coordinates.phase_coordinates.grad[1],
        torch.zeros_like(coordinates.phase_coordinates.grad[1]),
        atol=0,
        rtol=0,
    )
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_dropout_zero_matches_between_train_and_eval() -> None:
    encoder = _encoder(dropout=0.0)
    coordinates = _coordinate_output()

    encoder.train()
    train_output = encoder(coordinates).feature
    encoder.eval()
    eval_output = encoder(coordinates).feature

    torch.testing.assert_close(train_output, eval_output)


def test_nonzero_dropout_is_deterministic_in_eval() -> None:
    encoder = _encoder(dropout=0.4).eval()
    coordinates = _coordinate_output()

    first = encoder(coordinates).feature
    second = encoder(coordinates).feature

    torch.testing.assert_close(first, second)
    assert encoder.shape_encoder.network[2].p == 0.4
    assert encoder.phase_encoder.network[2].p == 0.4
    assert encoder.output_head.dropout.p == 0.4


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype: torch.dtype) -> None:
    encoder = _encoder().to(dtype=dtype)
    coordinates = _coordinate_output(dtype=dtype)

    output = encoder(coordinates)

    for tensor in (
        output.shape_embedding,
        output.phase_embedding,
        output.joint_embedding,
        output.feature,
    ):
        assert tensor.dtype == dtype
        assert torch.isfinite(tensor).all()
    for parameter in encoder.parameters():
        assert parameter.dtype == dtype


def test_cpu_bfloat16_autocast_supports_fp32_master_parameters() -> None:
    encoder = _encoder(dropout=0.0)
    coordinates = _coordinate_output(requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = encoder(coordinates)
        loss = (
            output.feature.square().mean()
            + output.shape_embedding.square().mean()
            + output.phase_embedding.square().mean()
        )
    loss.backward()

    for tensor in (
        output.feature,
        output.shape_embedding,
        output.phase_embedding,
        output.joint_embedding,
    ):
        assert torch.isfinite(tensor).all()
    for module in (
        encoder.shape_encoder,
        encoder.phase_encoder,
        encoder.output_head,
    ):
        for parameter in module.parameters():
            assert parameter.dtype == torch.float32
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_precision_input_is_rejected_without_autocast(
    dtype: torch.dtype,
) -> None:
    coordinates = _coordinate_output(dtype=dtype)

    with pytest.raises(ValueError, match="autocast is disabled"):
        _encoder()(coordinates)


@pytest.mark.parametrize(
    "factory,kwargs,match",
    [
        (ShapeCoordinateEncoder, {"num_shape_basis": 0, "attribute_projection_dim": 5}, "num_shape_basis"),
        (ShapeCoordinateEncoder, {"num_shape_basis": 4, "attribute_projection_dim": 0}, "attribute_projection_dim"),
        (ShapeCoordinateEncoder, {"num_shape_basis": 4, "attribute_projection_dim": 5, "hidden_dim": 0}, "hidden_dim"),
        (PhaseCoordinateEncoder, {"num_phase_basis": 0}, "num_phase_basis"),
        (PhaseCoordinateEncoder, {"num_phase_basis": 3, "hidden_dim": 0}, "hidden_dim"),
        (TemporalStructureOutputHead, {"joint_dim": 0, "structure_dim": 12}, "joint_dim"),
        (TemporalStructureOutputHead, {"joint_dim": 16, "structure_dim": 0}, "structure_dim"),
        (TemporalStructureEncoder, {"num_shape_basis": 0, "attribute_projection_dim": 5, "num_phase_basis": 3}, "num_shape_basis"),
        (TemporalStructureEncoder, {"num_shape_basis": 4, "attribute_projection_dim": 0, "num_phase_basis": 3}, "attribute_projection_dim"),
        (TemporalStructureEncoder, {"num_shape_basis": 4, "attribute_projection_dim": 5, "num_phase_basis": 0}, "num_phase_basis"),
        (TemporalStructureEncoder, {"num_shape_basis": 4, "attribute_projection_dim": 5, "num_phase_basis": 3, "coordinate_hidden_dim": 0}, "coordinate_hidden_dim"),
        (TemporalStructureEncoder, {"num_shape_basis": 4, "attribute_projection_dim": 5, "num_phase_basis": 3, "structure_dim": 0}, "structure_dim"),
    ],
)
def test_nonpositive_constructor_dimensions_raise(factory, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, float("nan"), float("inf")])
@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (ShapeCoordinateEncoder, {"num_shape_basis": 4, "attribute_projection_dim": 5}),
        (PhaseCoordinateEncoder, {"num_phase_basis": 3}),
        (TemporalStructureOutputHead, {"joint_dim": 16, "structure_dim": 12}),
        (TemporalStructureEncoder, {"num_shape_basis": 4, "attribute_projection_dim": 5, "num_phase_basis": 3}),
    ],
)
def test_invalid_dropout_raises(factory, kwargs, dropout) -> None:
    with pytest.raises(ValueError, match="dropout"):
        factory(dropout=dropout, **kwargs)


@pytest.mark.parametrize(
    "coordinates,match",
    [
        (torch.ones(3, 20), "shape_coordinates"),
        (torch.ones(3, 3, 5), "shape_coordinates"),
        (torch.ones(3, 4, 4), "shape_coordinates"),
        (torch.ones(3, 4, 5, dtype=torch.int64), "floating"),
        (torch.full((3, 4, 5), float("nan")), "finite"),
        (torch.full((3, 4, 5), float("inf")), "finite"),
    ],
)
def test_shape_encoder_rejects_invalid_coordinates(coordinates, match) -> None:
    encoder = ShapeCoordinateEncoder(4, 5)
    with pytest.raises(ValueError, match=match):
        encoder(coordinates, torch.ones(3, dtype=torch.bool))


@pytest.mark.parametrize(
    "coordinates,match",
    [
        (torch.ones(3, 4, 1), "phase_coordinates"),
        (torch.ones(3, 3), "phase_coordinates"),
        (torch.ones(3, 4, dtype=torch.int64), "floating"),
        (torch.full((3, 4), float("nan")), "finite"),
        (torch.full((3, 4), float("inf")), "finite"),
    ],
)
def test_phase_encoder_rejects_invalid_coordinates(coordinates, match) -> None:
    encoder = PhaseCoordinateEncoder(3)
    with pytest.raises(ValueError, match=match):
        encoder(coordinates, torch.ones(3, dtype=torch.bool))


@pytest.mark.parametrize(
    "valid",
    [
        torch.ones(3),
        torch.ones(3, 1, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    ],
)
def test_coordinate_encoders_reject_invalid_valid_mask(valid) -> None:
    with pytest.raises(ValueError, match="valid"):
        ShapeCoordinateEncoder(4, 5)(torch.ones(3, 4, 5), valid)
    with pytest.raises(ValueError, match="valid"):
        PhaseCoordinateEncoder(3)(torch.ones(3, 4), valid)


def test_coordinate_encoder_rejects_device_mismatch() -> None:
    valid = torch.ones(3, dtype=torch.bool, device="meta")
    with pytest.raises(ValueError, match="device"):
        ShapeCoordinateEncoder(4, 5)(torch.ones(3, 4, 5), valid)


def test_complete_encoder_rejects_wrong_object_type() -> None:
    with pytest.raises(ValueError, match="TemporalCoordinateOutput"):
        _encoder()(object())


def test_complete_encoder_rejects_batch_mismatch() -> None:
    coordinates = _coordinate_output()
    with pytest.raises(ValueError, match="batch"):
        _encoder()(
            replace(coordinates, phase_coordinates=torch.ones(2, 4))
        )


def test_complete_encoder_rejects_coordinate_dtype_mismatch() -> None:
    coordinates = _coordinate_output()
    with pytest.raises(ValueError, match="dtype"):
        _encoder()(
            replace(
                coordinates,
                phase_coordinates=coordinates.phase_coordinates.double(),
            )
        )


def test_complete_encoder_rejects_coordinate_device_mismatch() -> None:
    coordinates = _coordinate_output()
    with pytest.raises(ValueError, match="device"):
        _encoder()(
            replace(
                coordinates,
                phase_coordinates=torch.ones(3, 4, device="meta"),
            )
        )


def test_complete_encoder_rejects_module_dtype_mismatch() -> None:
    with pytest.raises(ValueError, match="dtype"):
        _encoder().double()(_coordinate_output(dtype=torch.float32))


def test_real_coordinate_module_integration_preserves_gradients() -> None:
    torch.manual_seed(43)
    registered_srvf = torch.randn(2, 7, 3, requires_grad=True)
    interval_widths = torch.tensor(
        [
            [0.08, 0.12, 0.18, 0.24, 0.20, 0.18],
            [0.18, 0.20, 0.24, 0.18, 0.12, 0.08],
        ],
        requires_grad=True,
    )
    cumulative = interval_widths.cumsum(dim=-1)
    warp = torch.cat(
        [
            torch.zeros_like(interval_widths[:, :1]),
            cumulative[:, :-1],
            torch.ones_like(interval_widths[:, :1]),
        ],
        dim=-1,
    )
    speed = interval_widths * 6
    warp_derivative = torch.cat(
        [
            speed[:, :1],
            0.5 * (speed[:, :-1] + speed[:, 1:]),
            speed[:, -1:],
        ],
        dim=-1,
    )
    registration = TemporalRegistrationOutput(
        srvf_output=None,
        template_srvf=torch.zeros(2, 7, 3),
        template_support=torch.ones(2, 7),
        template_initialized=torch.tensor(True),
        template_mean_support=torch.tensor(1.0),
        interval_logits=torch.zeros(2, 6),
        interval_widths=interval_widths,
        warp=warp,
        warp_derivative=warp_derivative,
        registered_srvf=registered_srvf,
        registered_support=torch.ones(2, 7),
        registration_valid=torch.ones(2, dtype=torch.bool),
    )
    coordinate_module = TemporalShapePhaseCoordinates(
        feature_dim=3,
        canonical_grid_size=7,
        num_shape_basis=4,
        num_phase_basis=3,
        attribute_projection_dim=5,
    )
    head = _encoder(dropout=0.0)

    coordinates = coordinate_module(registration)
    output = head(coordinates)
    weights = torch.arange(1, 13, dtype=output.feature.dtype)
    (output.feature * weights).sum().backward()

    assert output.feature.shape == (2, 12)
    assert output.valid is registration.registration_valid
    assert registered_srvf.grad is not None
    assert interval_widths.grad is not None
    assert torch.isfinite(registered_srvf.grad).all()
    assert torch.isfinite(interval_widths.grad).all()
    assert coordinate_module.attribute_projection.weight.grad is not None
    assert torch.isfinite(
        coordinate_module.attribute_projection.weight.grad
    ).all()
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_shape_feature_encoder_has_exact_required_network_and_no_phase_encoder() -> None:
    encoder = ShapeFeatureEncoder(
        num_shape_basis=4,
        attribute_projection_dim=3,
        output_dim=7,
        hidden_dim=5,
        dropout=0.2,
    )

    assert [type(layer) for layer in encoder.network] == [
        nn.Linear,
        nn.GELU,
        nn.Dropout,
        nn.Linear,
    ]
    assert encoder.network[0].in_features == 12
    assert encoder.network[-1].out_features == 7
    assert not hasattr(encoder, "phase_encoder")


def test_shape_feature_encoder_zeroes_invalid_rows_after_biases() -> None:
    encoder = ShapeFeatureEncoder(4, 3, output_dim=7, hidden_dim=5, dropout=0.0)
    with torch.no_grad():
        encoder.network[0].bias.fill_(3.0)
        encoder.network[-1].bias.fill_(2.0)
    coordinates = torch.randn(2, 4, 3)
    valid = torch.tensor([True, False])

    output = encoder(coordinates, valid)

    assert isinstance(output, ShapeFeatureOutput)
    assert output.feature.shape == (2, 7)
    assert output.valid is valid
    torch.testing.assert_close(output.feature[1], torch.zeros(7), atol=0, rtol=0)
    assert not hasattr(output, "phase_embedding")
    assert not hasattr(output, "joint_embedding")


def test_shape_feature_encoder_backpropagates_to_coordinates_and_all_parameters() -> None:
    encoder = ShapeFeatureEncoder(4, 3, output_dim=7, hidden_dim=5, dropout=0.0)
    coordinates = torch.randn(2, 4, 3, requires_grad=True)

    output = encoder(coordinates, torch.tensor([True, True]))
    output.feature.square().sum().backward()

    assert coordinates.grad is not None and torch.isfinite(coordinates.grad).all()
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
