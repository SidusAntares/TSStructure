from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from methods.structure_da import (
    TemporalCoordinateOutput,
    TemporalRegistrationOutput,
    TemporalShapePhaseCoordinates,
    TemporalSRVFRegistration,
    warp_to_identity_tangent,
)
from methods.structure_da.temporal_coordinates import (
    TrendStructureCoordinateOutput,
    TrendStructureCoordinates,
    _weighted_orthogonal_cosine_basis,
)


def _registration_output(
    *,
    registered_srvf: torch.Tensor | None = None,
    template_srvf: torch.Tensor | None = None,
    registered_support: torch.Tensor | None = None,
    template_support: torch.Tensor | None = None,
    interval_widths: torch.Tensor | None = None,
    registration_valid: torch.Tensor | None = None,
    batch_size: int = 2,
    grid_size: int = 7,
    feature_dim: int = 3,
    dtype: torch.dtype = torch.float32,
) -> TemporalRegistrationOutput:
    if registered_srvf is None:
        if template_srvf is not None:
            batch_size, grid_size, feature_dim = template_srvf.shape
        elif registered_support is not None:
            batch_size, grid_size = registered_support.shape
        elif template_support is not None:
            batch_size, grid_size = template_support.shape
        elif interval_widths is not None:
            batch_size = interval_widths.shape[0]
            grid_size = interval_widths.shape[1] + 1
        registered_srvf = torch.zeros(
            batch_size, grid_size, feature_dim, dtype=dtype
        )
    batch_size, grid_size, _ = registered_srvf.shape
    if template_srvf is None:
        template_srvf = torch.zeros_like(registered_srvf)
    if registered_support is None:
        registered_support = torch.ones(
            batch_size,
            grid_size,
            device=registered_srvf.device,
            dtype=registered_srvf.dtype,
        )
    if template_support is None:
        template_support = torch.ones_like(registered_support)
    if interval_widths is None:
        interval_widths = torch.full(
            (batch_size, grid_size - 1),
            1.0 / (grid_size - 1),
            device=registered_srvf.device,
            dtype=registered_srvf.dtype,
        )
    if registration_valid is None:
        registration_valid = torch.ones(
            batch_size,
            device=registered_srvf.device,
            dtype=torch.bool,
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
    speed = interval_widths * (grid_size - 1)
    derivative = torch.cat(
        [
            speed[:, :1],
            0.5 * (speed[:, :-1] + speed[:, 1:]),
            speed[:, -1:],
        ],
        dim=-1,
    )
    return TemporalRegistrationOutput(
        srvf_output=None,
        template_srvf=template_srvf,
        template_support=template_support,
        template_initialized=torch.tensor(True, device=registered_srvf.device),
        template_mean_support=template_support.mean(),
        interval_logits=torch.zeros_like(interval_widths),
        interval_widths=interval_widths,
        warp=warp,
        warp_derivative=derivative,
        registered_srvf=registered_srvf,
        registered_support=registered_support,
        registration_valid=registration_valid,
    )


def _coordinates(
    *,
    feature_dim: int = 3,
    grid_size: int = 7,
    num_shape_basis: int = 4,
    num_phase_basis: int = 3,
    attribute_projection_dim: int = 2,
    **kwargs,
) -> TemporalShapePhaseCoordinates:
    return TemporalShapePhaseCoordinates(
        feature_dim=feature_dim,
        canonical_grid_size=grid_size,
        num_shape_basis=num_shape_basis,
        num_phase_basis=num_phase_basis,
        attribute_projection_dim=attribute_projection_dim,
        **kwargs,
    )


def _widths_from_phase_direction(
    direction: torch.Tensor,
    angle: float,
) -> torch.Tensor:
    direction = direction / direction.square().mean().sqrt()
    angle_tensor = torch.tensor(angle, dtype=direction.dtype)
    warp_srvf = (
        torch.cos(angle_tensor)
        + torch.sin(angle_tensor) * direction
    )
    assert torch.all(warp_srvf > 0)
    return warp_srvf.square() / direction.numel()


def test_fixed_basis_shapes_weighted_orthogonality_and_phase_zero_mean() -> None:
    module = _coordinates()

    assert module.shape_time_basis.shape == (7, 4)
    assert module.phase_time_basis.shape == (6, 3)
    shape_gram = module.shape_time_basis.T @ (
        module.grid_integration_weights.unsqueeze(-1) * module.shape_time_basis
    )
    phase_gram = module.phase_time_basis.T @ (
        module.interval_integration_weights.unsqueeze(-1)
        * module.phase_time_basis
    )
    torch.testing.assert_close(shape_gram, torch.eye(4), atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(phase_gram, torch.eye(3), atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(
        (module.interval_integration_weights.unsqueeze(-1)
         * module.phase_time_basis).sum(dim=0),
        torch.zeros(3),
        atol=2e-6,
        rtol=0,
    )


def test_fixed_grid_and_basis_are_buffers_not_parameters_and_deterministic() -> None:
    torch.manual_seed(91)
    first = _coordinates()
    torch.manual_seed(1234)
    second = _coordinates()
    expected_buffers = {
        "canonical_grid",
        "grid_integration_weights",
        "interval_midpoints",
        "interval_integration_weights",
        "shape_time_basis",
        "phase_time_basis",
    }

    assert set(dict(first.named_buffers())) == expected_buffers
    assert set(dict(first.named_parameters())) == {"attribute_projection.weight"}
    for name in expected_buffers:
        torch.testing.assert_close(getattr(first, name), getattr(second, name))


def test_weighted_basis_helper_matches_module_basis() -> None:
    module = _coordinates()
    shape = _weighted_orthogonal_cosine_basis(
        module.canonical_grid,
        module.grid_integration_weights,
        4,
        exclude_constant=False,
        eps=1e-8,
    )
    phase = _weighted_orthogonal_cosine_basis(
        module.interval_midpoints,
        module.interval_integration_weights,
        3,
        exclude_constant=True,
        eps=1e-8,
    )

    torch.testing.assert_close(shape, module.shape_time_basis)
    torch.testing.assert_close(phase, module.phase_time_basis)


def test_identity_phase_has_exact_zero_coordinates() -> None:
    module = _coordinates()
    result = module(_registration_output())

    assert isinstance(result, TemporalCoordinateOutput)
    torch.testing.assert_close(result.phase_tangent, torch.zeros(2, 6))
    torch.testing.assert_close(result.phase_basis_coefficients, torch.zeros(2, 3))
    torch.testing.assert_close(result.phase_magnitude, torch.zeros(2))
    torch.testing.assert_close(result.phase_coordinates, torch.zeros(2, 4))


def test_known_phase_direction_matches_weighted_basis_projection() -> None:
    module = _coordinates()
    direction = module.phase_time_basis[:, 1].double()
    widths = _widths_from_phase_direction(direction, angle=0.2).unsqueeze(0)
    module = module.double()
    result = module(_registration_output(interval_widths=widths, dtype=torch.float64))
    phase = warp_to_identity_tangent(widths, eps=module.eps)
    expected = torch.einsum(
        "j,bj,jm->bm",
        module.interval_integration_weights,
        phase.tangent,
        module.phase_time_basis,
    )

    torch.testing.assert_close(result.phase_basis_coefficients, expected)
    assert result.phase_basis_coefficients[0, 1].abs() == (
        result.phase_basis_coefficients[0].abs().max()
    )
    torch.testing.assert_close(
        result.phase_coordinates[:, -1], phase.magnitude
    )


def test_zero_shape_residual_has_zero_shape_coordinates() -> None:
    module = _coordinates()
    srvf = torch.randn(2, 7, 3)
    result = module(
        _registration_output(
            registered_srvf=srvf,
            template_srvf=srvf.clone(),
        )
    )

    torch.testing.assert_close(result.shape_residual, torch.zeros_like(srvf))
    torch.testing.assert_close(result.shape_time_coefficients, torch.zeros(2, 4, 3))
    torch.testing.assert_close(result.shape_coordinates, torch.zeros(2, 4, 2))


def test_known_shape_basis_recovers_attribute_and_shared_linear_projection() -> None:
    module = _coordinates().double()
    basis_index = 2
    attribute = torch.tensor([0.7, -1.2, 0.4], dtype=torch.float64)
    residual = (
        module.shape_time_basis[:, basis_index].unsqueeze(-1)
        * attribute
    ).unsqueeze(0)
    result = module(_registration_output(registered_srvf=residual))
    expected_attribute = attribute / torch.sqrt(
        torch.tensor(1.0 + module.eps, dtype=torch.float64)
    )

    torch.testing.assert_close(
        result.shape_time_coefficients[0, basis_index],
        expected_attribute,
        atol=2e-6,
        rtol=2e-6,
    )
    other = result.shape_time_coefficients[0].clone()
    other[basis_index] = 0
    torch.testing.assert_close(other, torch.zeros_like(other), atol=2e-6, rtol=0)
    torch.testing.assert_close(
        result.shape_coordinates,
        F.linear(result.shape_time_coefficients, module.attribute_projection.weight),
    )


def test_zero_support_position_isolates_arbitrarily_large_residual() -> None:
    module = _coordinates()
    residual = torch.randn(1, 7, 3)
    support = torch.ones(1, 7)
    support[0, 3] = 0.0
    baseline = module(
        _registration_output(
            registered_srvf=residual,
            registered_support=support,
        )
    )
    changed = residual.clone()
    changed[0, 3] = 1e8
    isolated = module(
        _registration_output(
            registered_srvf=changed,
            registered_support=support,
        )
    )

    torch.testing.assert_close(
        isolated.shape_time_coefficients,
        baseline.shape_time_coefficients,
    )
    torch.testing.assert_close(isolated.shape_coordinates, baseline.shape_coordinates)


def test_reducing_support_decreases_affected_basis_support() -> None:
    module = _coordinates()
    full = module(_registration_output(batch_size=1))
    support = torch.ones(1, 7)
    basis_index = 1
    location = module.shape_time_basis[:, basis_index].square().argmax().item()
    support[0, location] = 0.0
    reduced = module(
        _registration_output(
            batch_size=1,
            registered_support=support,
        )
    )

    assert (
        reduced.shape_basis_support[0, basis_index]
        < full.shape_basis_support[0, basis_index]
    )


def test_basis_coefficient_is_zero_below_support_threshold() -> None:
    module = _coordinates(min_basis_support=0.2)
    residual = torch.randn(1, 7, 3)
    support = torch.full((1, 7), 0.1)
    result = module(
        _registration_output(
            registered_srvf=residual,
            registered_support=support,
        )
    )

    assert torch.all(result.shape_basis_support < 0.2)
    torch.testing.assert_close(
        result.shape_time_coefficients,
        torch.zeros_like(result.shape_time_coefficients),
    )


def test_shape_support_is_raw_product_without_softmax_or_time_normalization() -> None:
    module = _coordinates()
    sample = torch.tensor([[0.2, 0.4, 0.8, 0.3, 0.7, 0.6, 0.9]])
    template = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]])
    result = module(
        _registration_output(
            registered_support=sample,
            template_support=template,
        )
    )

    torch.testing.assert_close(result.shape_support, sample * template)
    assert not torch.isclose(result.shape_support.sum(), torch.tensor(1.0))


def test_invalid_registration_rows_are_strictly_zero_but_keep_support_diagnostics() -> None:
    module = _coordinates()
    registered = torch.randn(2, 7, 3)
    sample_support = torch.rand(2, 7)
    template_support = torch.rand(2, 7)
    widths = torch.tensor(
        [
            [0.08, 0.12, 0.18, 0.24, 0.20, 0.18],
            [0.18, 0.20, 0.24, 0.18, 0.12, 0.08],
        ]
    )
    result = module(
        _registration_output(
            registered_srvf=registered,
            registered_support=sample_support,
            template_support=template_support,
            interval_widths=widths,
            registration_valid=torch.tensor([True, False]),
        )
    )

    for value in (
        result.shape_coordinates,
        result.phase_coordinates,
        result.shape_time_coefficients,
        result.phase_basis_coefficients,
        result.phase_magnitude,
        result.shape_residual,
        result.phase_tangent,
    ):
        torch.testing.assert_close(value[1], torch.zeros_like(value[1]), atol=0, rtol=0)
    torch.testing.assert_close(result.shape_support, sample_support * template_support)
    assert result.shape_basis_support[1].abs().sum().item() > 0
    assert result.valid.tolist() == [True, False]


def test_only_attribute_projection_weight_is_trainable() -> None:
    module = _coordinates(attribute_projection_dim=5)

    parameters = dict(module.named_parameters())
    assert set(parameters) == {"attribute_projection.weight"}
    assert parameters["attribute_projection.weight"].shape == (5, 3)
    assert module.attribute_projection.bias is None


def test_coordinate_gradients_are_finite_and_invalid_input_gradients_are_zero() -> None:
    module = _coordinates()
    registered = torch.randn(2, 7, 3, requires_grad=True)
    raw_widths = torch.tensor(
        [
            [0.08, 0.12, 0.18, 0.24, 0.20, 0.18],
            [0.18, 0.20, 0.24, 0.18, 0.12, 0.08],
        ],
        requires_grad=True,
    )
    output = module(
        _registration_output(
            registered_srvf=registered,
            interval_widths=raw_widths,
            registration_valid=torch.tensor([True, False]),
        )
    )

    (output.shape_coordinates.square().mean()
     + output.phase_coordinates.square().mean()).backward()

    assert registered.grad is not None and torch.isfinite(registered.grad).all()
    assert raw_widths.grad is not None and torch.isfinite(raw_widths.grad).all()
    assert module.attribute_projection.weight.grad is not None
    assert torch.isfinite(module.attribute_projection.weight.grad).all()
    assert registered.grad[0].abs().sum().item() > 0
    assert raw_widths.grad[0].abs().sum().item() > 0
    torch.testing.assert_close(registered.grad[1], torch.zeros_like(registered.grad[1]), atol=0, rtol=0)
    torch.testing.assert_close(raw_widths.grad[1], torch.zeros_like(raw_widths.grad[1]), atol=0, rtol=0)
    for value in module.buffers():
        assert value.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_shapes_and_dtype(dtype: torch.dtype) -> None:
    module = _coordinates().to(dtype=dtype)
    output = module(_registration_output(dtype=dtype))

    assert output.shape_coordinates.shape == (2, 4, 2)
    assert output.phase_coordinates.shape == (2, 4)
    assert output.shape_time_coefficients.shape == (2, 4, 3)
    assert output.phase_basis_coefficients.shape == (2, 3)
    assert output.phase_magnitude.shape == (2,)
    assert output.shape_support.shape == (2, 7)
    assert output.shape_basis_support.shape == (2, 4)
    assert output.shape_residual.shape == (2, 7, 3)
    assert output.phase_tangent.shape == (2, 6)
    assert output.valid.shape == (2,)
    for value in (
        output.shape_coordinates,
        output.phase_coordinates,
        output.shape_time_coefficients,
        output.phase_basis_coefficients,
        output.phase_magnitude,
        output.shape_support,
        output.shape_basis_support,
        output.shape_residual,
        output.phase_tangent,
    ):
        assert value.dtype == dtype
        assert torch.isfinite(value).all()
    for value in module.buffers():
        assert value.dtype == dtype
    assert module.attribute_projection.weight.dtype == dtype


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"feature_dim": 0}, "feature_dim"),
        ({"grid_size": 2}, "canonical_grid_size"),
        ({"num_shape_basis": 0}, "num_shape_basis"),
        ({"num_shape_basis": 8}, "num_shape_basis"),
        ({"num_phase_basis": 0}, "num_phase_basis"),
        ({"num_phase_basis": 6}, "num_phase_basis"),
        ({"attribute_projection_dim": 0}, "attribute_projection_dim"),
        ({"min_basis_support": -0.1}, "min_basis_support"),
        ({"min_basis_support": float("nan")}, "min_basis_support"),
        ({"min_basis_support": float("inf")}, "min_basis_support"),
        ({"eps": 0.0}, "eps"),
        ({"eps": float("nan")}, "eps"),
    ],
)
def test_constructor_rejects_invalid_arguments(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        _coordinates(**kwargs)


def test_maximum_legal_phase_basis_is_k_minus_two() -> None:
    module = _coordinates(num_phase_basis=5)

    assert module.phase_time_basis.shape == (6, 5)


def test_object_rejects_wrong_registration_output_type() -> None:
    with pytest.raises(ValueError, match="TemporalRegistrationOutput"):
        _coordinates()(object())


def test_rejects_registered_srvf_shape_mismatch() -> None:
    output = _registration_output()
    with pytest.raises(ValueError, match="registered_srvf"):
        _coordinates()(replace(output, registered_srvf=torch.ones(2, 6, 3)))


def test_rejects_template_srvf_shape_mismatch() -> None:
    output = _registration_output()
    with pytest.raises(ValueError, match="template_srvf"):
        _coordinates()(replace(output, template_srvf=torch.ones(2, 6, 3)))


def test_rejects_feature_dimension_mismatch() -> None:
    output = _registration_output(feature_dim=4)
    with pytest.raises(ValueError, match="feature"):
        _coordinates(feature_dim=3)(output)


@pytest.mark.parametrize("field", ["registered_support", "template_support"])
def test_rejects_support_shape_mismatch(field) -> None:
    output = _registration_output()
    with pytest.raises(ValueError, match=field):
        _coordinates()(replace(output, **{field: torch.ones(2, 6)}))


@pytest.mark.parametrize("field", ["registered_support", "template_support"])
def test_rejects_support_outside_unit_interval(field) -> None:
    output = _registration_output()
    support = getattr(output, field).clone()
    support[0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _coordinates()(replace(output, **{field: support}))


@pytest.mark.parametrize(
    "widths,match",
    [
        (torch.tensor([[0.0, 0.2, 0.2, 0.2, 0.2, 0.2]]).expand(2, -1), "positive"),
        (torch.full((2, 6), 0.1), "sum"),
        (torch.full((2, 5), 0.2), "shape"),
    ],
)
def test_rejects_invalid_interval_widths(widths, match) -> None:
    with pytest.raises(ValueError, match=match):
        _coordinates()(replace(_registration_output(), interval_widths=widths))


def test_rejects_nonboolean_registration_valid() -> None:
    with pytest.raises(ValueError, match="boolean"):
        _coordinates()(
            replace(_registration_output(), registration_valid=torch.ones(2))
        )


def test_rejects_tensor_dtype_mismatch() -> None:
    output = _registration_output()
    with pytest.raises(ValueError, match="dtype"):
        _coordinates()(
            replace(output, template_srvf=output.template_srvf.double())
        )


def test_rejects_tensor_device_mismatch() -> None:
    output = _registration_output()
    meta_valid = torch.ones(2, dtype=torch.bool, device="meta")
    with pytest.raises(ValueError, match="device"):
        _coordinates()(replace(output, registration_valid=meta_valid))


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("registered_srvf", float("nan")),
        ("template_srvf", float("inf")),
        ("registered_support", float("nan")),
        ("template_support", float("inf")),
        ("interval_widths", float("nan")),
    ],
)
def test_rejects_nonfinite_inputs(field, bad_value) -> None:
    output = _registration_output()
    tensor = getattr(output, field).clone()
    tensor.reshape(-1)[0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        _coordinates()(replace(output, **{field: tensor}))


def _real_registration(dtype: torch.dtype = torch.float32):
    registration = TemporalSRVFRegistration(
        feature_dim=2,
        num_basis=6,
        canonical_grid_size=8,
        roughness_grid_size=64,
        min_mean_support=0.0,
        min_dynamic_energy=0.0,
        min_template_mean_support=0.0,
        warp_hidden_dim=8,
        warp_kernel_size=3,
    ).to(dtype=dtype)
    torch.manual_seed(94)
    tokens = torch.randn(2, 6, 2, dtype=dtype)
    positions = torch.tensor(
        [0.0, 39.0, 92.0, 157.0, 244.0, 345.0], dtype=dtype
    )
    mask = torch.tensor(
        [[True, True, True, True, True, True],
         [True, False, True, True, False, True]]
    )
    return registration, tokens, positions, mask


def test_real_registration_coordinates_backward_preserves_gradients() -> None:
    registration, tokens, positions, mask = _real_registration()
    registration.update_source_statistics(tokens, mask)
    initial = registration(tokens, positions, mask)
    registration.update_source_support_scale(initial.srvf_output.functional)
    bootstrap = registration(tokens, positions, mask)
    registration.update_source_template(bootstrap)
    differentiable = (tokens + 0.1 * torch.randn_like(tokens)).requires_grad_()
    registration_output = registration(differentiable, positions, mask)
    coordinates = TemporalShapePhaseCoordinates(
        feature_dim=2,
        canonical_grid_size=8,
        num_shape_basis=4,
        num_phase_basis=3,
        attribute_projection_dim=3,
    )

    output = coordinates(registration_output)
    (output.shape_coordinates.square().mean()
     + output.phase_coordinates.square().mean()).backward()

    assert output.shape_coordinates.shape == (2, 4, 3)
    assert output.phase_coordinates.shape == (2, 4)
    assert registration_output.registration_valid.all().item()
    assert differentiable.grad is not None and torch.isfinite(differentiable.grad).all()
    assert differentiable.grad[mask].abs().sum().item() > 0
    torch.testing.assert_close(
        differentiable.grad[~mask],
        torch.zeros_like(differentiable.grad[~mask]),
        atol=0,
        rtol=0,
    )
    last = registration.warp_estimator.network[-1]
    assert last.weight.grad is not None and torch.isfinite(last.weight.grad).all()
    assert last.bias.grad is not None and torch.isfinite(last.bias.grad).all()
    assert coordinates.attribute_projection.weight.grad is not None
    assert torch.isfinite(coordinates.attribute_projection.weight.grad).all()
    assert not dict(registration.source_template.named_parameters())
    for value in registration.source_template.buffers():
        assert value.grad is None


def _trend_structure_coordinates(**kwargs) -> TrendStructureCoordinates:
    values = dict(
        feature_dim=3,
        canonical_grid_size=7,
        num_shape_basis=4,
        num_phase_basis=3,
        attribute_projection_dim=2,
    )
    values.update(kwargs)
    return TrendStructureCoordinates(**values)


def test_trend_structure_shape_coordinates_use_complete_aligned_structure() -> None:
    module = _trend_structure_coordinates().double()
    aligned = torch.randn(2, 7, 3, dtype=torch.float64)
    support = torch.rand(2, 7, dtype=torch.float64)
    widths = torch.full((2, 6), 1.0 / 6, dtype=torch.float64)
    output = module(
        aligned_structure_srvf=aligned,
        aligned_structure_support=support,
        interval_widths=widths,
        shape_valid=torch.tensor([True, True]),
        phase_valid=torch.tensor([True, True]),
    )

    basis_support = torch.einsum(
        "k,bk,km->bm",
        module.grid_integration_weights,
        support,
        module.shape_time_basis.square(),
    )
    numerator = torch.einsum(
        "k,bk,km,bkd->bmd",
        module.grid_integration_weights,
        support,
        module.shape_time_basis,
        aligned,
    )
    expected = numerator / (basis_support + module.eps).unsqueeze(-1)

    assert isinstance(output, TrendStructureCoordinateOutput)
    torch.testing.assert_close(output.shape_coordinates_fixed, expected)
    assert not torch.allclose(
        output.shape_coordinates_fixed,
        numerator / torch.sqrt(basis_support + module.eps).unsqueeze(-1),
    )
    torch.testing.assert_close(output.shape_support, support)
    assert output.shape_coordinates_fixed.shape == (2, 4, 3)
    assert output.shape_coordinates.shape == (2, 4, 2)


def test_trend_structure_shape_and_phase_validity_are_independent() -> None:
    module = _trend_structure_coordinates()
    output = module(
        aligned_structure_srvf=torch.randn(2, 7, 3),
        aligned_structure_support=torch.ones(2, 7),
        interval_widths=torch.tensor(
            [
                [0.10, 0.12, 0.18, 0.20, 0.22, 0.18],
                [0.18, 0.22, 0.20, 0.18, 0.12, 0.10],
            ]
        ),
        shape_valid=torch.tensor([True, False]),
        phase_valid=torch.tensor([False, True]),
    )

    assert output.shape_valid.tolist() == [True, False]
    assert output.phase_valid.tolist() == [False, True]
    for value in (
        output.shape_coordinates_fixed,
        output.shape_coordinates,
        output.shape_basis_support,
        output.shape_support,
    ):
        torch.testing.assert_close(value[1], torch.zeros_like(value[1]))
    for value in (
        output.phase_coordinates,
        output.phase_basis_coefficients,
        output.phase_magnitude,
        output.phase_tangent,
    ):
        torch.testing.assert_close(value[0], torch.zeros_like(value[0]))


def test_trend_structure_attribute_projection_has_gradient() -> None:
    module = _trend_structure_coordinates()
    aligned = torch.randn(2, 7, 3, requires_grad=True)
    output = module(
        aligned_structure_srvf=aligned,
        aligned_structure_support=torch.ones(2, 7),
        interval_widths=torch.full((2, 6), 1.0 / 6),
        shape_valid=torch.tensor([True, True]),
        phase_valid=torch.tensor([True, True]),
    )

    output.shape_coordinates.square().sum().backward()

    assert aligned.grad is not None and torch.isfinite(aligned.grad).all()
    gradient = module.attribute_projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
