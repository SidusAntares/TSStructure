import pytest
import torch

from methods.structure_da import (
    SourceRunningStandardizer,
    TemporalFunctionalLift,
    TemporalFunctionalOutput,
)
from methods.structure_da.temporal_functional import _evaluate_cubic_bspline


def _make_lift(**kwargs) -> TemporalFunctionalLift:
    parameters = {
        "feature_dim": 12,
        "num_basis": 8,
        "canonical_grid_size": 24,
        "roughness_grid_size": 96,
    }
    parameters.update(kwargs)
    return TemporalFunctionalLift(**parameters)


def test_first_source_update_uses_only_valid_first_and_second_moments() -> None:
    standardizer = SourceRunningStandardizer(feature_dim=2)
    tokens = torch.tensor(
        [
            [[1.0, 2.0], [1000.0, -1000.0], [3.0, 6.0]],
            [[5.0, 10.0], [-1000.0, 1000.0], [7.0, 14.0]],
        ]
    )
    time_mask = torch.tensor(
        [[True, False, True], [True, False, True]]
    )

    standardizer.update(tokens, time_mask)

    valid = tokens[time_mask]
    torch.testing.assert_close(standardizer.running_mean, valid.mean(dim=0))
    torch.testing.assert_close(
        standardizer.running_second_moment, valid.square().mean(dim=0)
    )
    assert standardizer.num_updates.item() == 1


def test_second_source_update_uses_configured_ema() -> None:
    standardizer = SourceRunningStandardizer(feature_dim=2, momentum=0.75)
    first = torch.tensor([[[1.0, 2.0], [3.0, 6.0]]])
    second = torch.tensor([[[5.0, 10.0], [9.0, 18.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    standardizer.update(first, mask)
    previous_mean = standardizer.running_mean.clone()
    previous_second = standardizer.running_second_moment.clone()

    standardizer.update(second, mask)

    torch.testing.assert_close(
        standardizer.running_mean,
        0.75 * previous_mean + 0.25 * second.reshape(-1, 2).mean(dim=0),
    )
    torch.testing.assert_close(
        standardizer.running_second_moment,
        0.75 * previous_second
        + 0.25 * second.reshape(-1, 2).square().mean(dim=0),
    )
    assert standardizer.num_updates.item() == 2


def test_standardizer_forward_is_identity_before_update_and_preserves_gradient() -> None:
    standardizer = SourceRunningStandardizer(feature_dim=3)
    tokens = torch.randn(2, 4, 3, requires_grad=True)
    buffers_before = {
        name: value.clone() for name, value in standardizer.named_buffers()
    }

    standardized = standardizer(tokens)
    standardized.square().sum().backward()

    torch.testing.assert_close(standardized, tokens)
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    for name, value in standardizer.named_buffers():
        torch.testing.assert_close(value, buffers_before[name])


def test_standardizer_forward_uses_running_scale_without_updating_buffers() -> None:
    standardizer = SourceRunningStandardizer(
        feature_dim=2, min_scale=1e-4, eps=1e-8
    )
    source = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]])
    standardizer.update(source, torch.ones(1, 3, dtype=torch.bool))
    tokens = torch.tensor([[[2.0, 8.0]]], requires_grad=True)
    buffers_before = {
        name: value.clone() for name, value in standardizer.named_buffers()
    }

    output = standardizer(tokens)
    variance = (
        standardizer.running_second_moment
        - standardizer.running_mean.square()
    )
    scale = torch.sqrt(variance.clamp_min(standardizer.min_scale**2))
    expected = (tokens - standardizer.running_mean) / scale.clamp_min(
        standardizer.eps
    )
    torch.testing.assert_close(output, expected)
    output.sum().backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    for name, value in standardizer.named_buffers():
        torch.testing.assert_close(value, buffers_before[name])


def test_standardizer_statistics_are_buffers_not_parameters() -> None:
    standardizer = SourceRunningStandardizer(feature_dim=5)

    assert set(dict(standardizer.named_buffers())) == {
        "running_mean",
        "running_second_moment",
        "num_updates",
    }
    assert not dict(standardizer.named_parameters())


def test_cubic_basis_partition_and_derivative_partition_include_endpoints() -> None:
    lift = _make_lift()
    points = torch.tensor([0.0, 0.13, 0.5, 0.91, 1.0], dtype=torch.float64)

    basis, derivative, _ = _evaluate_cubic_bspline(points, lift.knots)

    torch.testing.assert_close(
        basis.sum(dim=-1), torch.ones_like(points), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        derivative.sum(dim=-1), torch.zeros_like(points), atol=1e-11, rtol=0
    )


def test_roughness_matrix_is_finite_symmetric_and_positive_semidefinite() -> None:
    lift = _make_lift()
    omega = lift.roughness_matrix

    assert torch.isfinite(omega).all()
    torch.testing.assert_close(omega, omega.T, atol=1e-12, rtol=0)
    assert torch.linalg.eigvalsh(omega).min().item() >= -1e-9


def test_spline_state_is_registered_as_non_parameter_buffers() -> None:
    lift = _make_lift()
    buffers = dict(lift.named_buffers())
    parameters = dict(lift.named_parameters())

    for name in (
        "knots",
        "canonical_grid",
        "canonical_basis",
        "canonical_basis_derivative",
        "roughness_matrix",
    ):
        assert name in buffers
        assert name not in parameters
        assert buffers[name].grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_shapes_and_dtype(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    tokens = torch.randn(2, 7, 12, dtype=dtype)
    positions = torch.tensor([0, 17, 51, 103, 166, 244, 330])
    time_mask = torch.ones(2, 7, dtype=torch.bool)

    output = _make_lift()(tokens, positions, time_mask)

    assert isinstance(output, TemporalFunctionalOutput)
    assert output.standardized_tokens.shape == (2, 7, 12)
    assert output.normalized_positions.shape == (2, 7)
    assert output.coefficients.shape == (2, 8, 12)
    assert output.function.shape == (2, 24, 12)
    assert output.derivative.shape == (2, 24, 12)
    assert output.information_variance.shape == (2, 24)
    assert output.time_mask.shape == (2, 7)
    assert output.time_mask.dtype == torch.bool
    assert output.solve_valid.shape == (2,)
    for value in (
        output.standardized_tokens,
        output.normalized_positions,
        output.coefficients,
        output.function,
        output.derivative,
        output.information_variance,
        output.time_span,
        output.max_internal_gap,
    ):
        assert value.dtype == dtype
    assert output.num_valid_observations.shape == (2,)
    assert output.num_distinct_observations.shape == (2,)
    assert torch.isfinite(output.information_variance).all()
    assert torch.all(output.information_variance >= 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_information_variance_matches_batched_cholesky_reference(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(45)
    lift = _make_lift(smoothing_weight=2e-3, eps=1e-7)
    tokens = torch.randn(2, 6, 12, dtype=dtype)
    positions = torch.tensor(
        [[0.0, 41.0, 97.0, 153.0, 242.0, 340.0],
         [13.0, 62.0, 118.0, 191.0, 280.0, 355.0]],
        dtype=dtype,
    )
    time_mask = torch.tensor(
        [[True, True, True, True, True, True],
         [True, True, False, True, False, True]]
    )

    output = lift(tokens, positions, time_mask)

    fitting_positions = torch.where(
        time_mask, output.normalized_positions, torch.zeros_like(positions)
    )
    basis, _, _ = _evaluate_cubic_bspline(fitting_positions, lift.knots)
    weighted_basis = basis * time_mask.unsqueeze(-1).to(dtype=dtype)
    gram = basis.transpose(1, 2) @ weighted_basis
    roughness = lift.roughness_matrix.to(dtype=dtype)
    identity = torch.eye(lift.num_basis, dtype=dtype)
    gram = gram + lift.smoothing_weight * roughness + lift.eps * identity
    cholesky = torch.linalg.cholesky(gram)
    canonical_basis = lift.canonical_basis.to(dtype=dtype)
    canonical_rhs = canonical_basis.T.unsqueeze(0).expand(2, -1, -1)
    solved_basis = torch.cholesky_solve(canonical_rhs, cholesky)
    expected = (
        canonical_basis.T.unsqueeze(0) * solved_basis
    ).sum(dim=1).clamp_min(0.0)

    torch.testing.assert_close(
        output.information_variance,
        expected,
        atol=2e-5 if dtype == torch.float32 else 1e-10,
        rtol=2e-5 if dtype == torch.float32 else 1e-10,
    )


def test_spline_reconstructs_smooth_polynomial_on_canonical_grid() -> None:
    u = torch.tensor(
        [0.0, 0.04, 0.11, 0.19, 0.31, 0.46, 0.58, 0.73, 0.88, 1.0],
        dtype=torch.float64,
    )
    values = (0.7 * u.square() - 0.4 * u + 1.2).reshape(1, -1, 1)
    lift = TemporalFunctionalLift(
        feature_dim=1,
        num_basis=8,
        canonical_grid_size=41,
        roughness_grid_size=128,
        smoothing_weight=1e-8,
        eps=1e-8,
    )

    output = lift(values, u * 365.0, torch.ones(len(u), dtype=torch.bool))
    expected = (
        0.7 * lift.canonical_grid.square()
        - 0.4 * lift.canonical_grid
        + 1.2
    )

    assert output.solve_valid.item()
    torch.testing.assert_close(
        output.function[0, :, 0], expected, atol=2e-3, rtol=2e-3
    )


def test_linear_function_has_constant_canonical_derivative() -> None:
    u = torch.tensor(
        [0.0, 0.07, 0.18, 0.35, 0.54, 0.79, 1.0], dtype=torch.float64
    )
    slope, intercept = 1.75, -0.3
    values = (slope * u + intercept).reshape(1, -1, 1)
    lift = TemporalFunctionalLift(
        feature_dim=1,
        num_basis=7,
        canonical_grid_size=31,
        roughness_grid_size=96,
        smoothing_weight=1e-8,
        eps=1e-8,
    )

    output = lift(values, u * 365.0, torch.ones(len(u), dtype=torch.bool))

    expected = torch.full_like(output.derivative[0, 2:-2, 0], slope)
    torch.testing.assert_close(
        output.derivative[0, 2:-2, 0], expected, atol=3e-3, rtol=2e-3
    )


def test_common_physical_coordinates_do_not_apply_per_sample_min_max() -> None:
    tokens = torch.tensor(
        [[[0.0], [1.0], [0.0]], [[0.0], [1.0], [0.0]]]
    )
    positions = torch.tensor(
        [[0.0, 73.2, 146.4], [73.2, 146.4, 219.6]]
    )
    time_mask = torch.ones(2, 3, dtype=torch.bool)
    lift = TemporalFunctionalLift(
        feature_dim=1,
        num_basis=6,
        canonical_grid_size=32,
        roughness_grid_size=96,
    )

    output = lift(tokens, positions, time_mask)

    assert not torch.allclose(
        output.normalized_positions[0], output.normalized_positions[1]
    )
    assert not torch.allclose(output.function[0], output.function[1])


def test_physical_time_descriptors_ignore_masked_positions() -> None:
    tokens = torch.randn(1, 4, 1)
    positions = torch.tensor([0.0, 50.0, 350.0, 150.0])
    time_mask = torch.tensor([True, True, False, True])
    lift = TemporalFunctionalLift(
        feature_dim=1,
        num_basis=5,
        canonical_grid_size=16,
        roughness_grid_size=64,
    )

    output = lift(tokens, positions, time_mask)

    assert output.num_valid_observations.item() == 3
    assert output.num_distinct_observations.item() == 3
    torch.testing.assert_close(
        output.time_span, torch.tensor([150.0 / 365.0])
    )
    torch.testing.assert_close(
            output.max_internal_gap, torch.tensor([100.0 / 365.0])
    )


def test_masked_tokens_do_not_affect_fit_and_receive_zero_gradient() -> None:
    torch.manual_seed(43)
    tokens = torch.randn(1, 5, 2)
    changed = tokens.clone()
    time_mask = torch.tensor([True, False, True, True, False])
    changed[:, ~time_mask] = 1e9
    positions = torch.tensor([0.0, 30.0, 90.0, 180.0, 300.0])
    lift = TemporalFunctionalLift(
        feature_dim=2,
        num_basis=6,
        canonical_grid_size=20,
        roughness_grid_size=80,
    )

    expected = lift(tokens, positions, time_mask)
    actual = lift(changed, positions, time_mask)
    torch.testing.assert_close(actual.coefficients, expected.coefficients)
    torch.testing.assert_close(actual.function, expected.function)
    torch.testing.assert_close(actual.derivative, expected.derivative)

    differentiable_tokens = tokens.clone().requires_grad_()
    output = lift(differentiable_tokens, positions, time_mask)
    loss = output.function.square().mean() + output.derivative.square().mean()
    loss.backward()

    assert differentiable_tokens.grad is not None
    assert torch.isfinite(differentiable_tokens.grad).all()
    torch.testing.assert_close(
        differentiable_tokens.grad[:, ~time_mask],
        torch.zeros_like(differentiable_tokens.grad[:, ~time_mask]),
        atol=0,
        rtol=0,
    )
    assert differentiable_tokens.grad[:, time_mask].abs().sum().item() > 0


def test_solve_valid_requires_at_least_two_distinct_observations() -> None:
    tokens = torch.randn(3, 4, 1)
    positions = torch.tensor([0.0, 40.0, 120.0, 260.0])
    time_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, False, False, False],
            [False, False, False, False],
        ]
    )
    lift = TemporalFunctionalLift(
        feature_dim=1,
        num_basis=5,
        canonical_grid_size=16,
        roughness_grid_size=64,
    )

    output = lift(tokens, positions, time_mask)

    torch.testing.assert_close(
        output.solve_valid, torch.tensor([True, False, False])
    )
    assert output.num_valid_observations.tolist() == [4, 1, 0]
    assert output.num_distinct_observations.tolist() == [4, 1, 0]
    for value in (output.coefficients, output.function, output.derivative):
        torch.testing.assert_close(value[1:], torch.zeros_like(value[1:]))
    torch.testing.assert_close(
        output.information_variance[1:],
        torch.zeros_like(output.information_variance[1:]),
    )


def test_duplicate_valid_positions_raise_clear_error() -> None:
    tokens = torch.randn(1, 3, 1)

    with pytest.raises(ValueError, match="strictly increasing"):
        TemporalFunctionalLift(1)(
            tokens,
            torch.tensor([0.0, 0.0, 100.0]),
            torch.ones(3, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    "tokens,match",
    [
        (torch.randn(2, 3, 3, 4), "three-dimensional"),
        (torch.randn(2, 3, 11), "feature_dim=12"),
        (torch.ones(2, 3, 12, dtype=torch.long), "floating-point"),
    ],
)
def test_invalid_component_token_shape_or_dtype_raises_value_error(
    tokens: torch.Tensor, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _make_lift()(
            tokens,
            torch.tensor([0.0, 50.0, 100.0]),
            torch.ones(3, dtype=torch.bool),
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_valid_token_raises_value_error(invalid: float) -> None:
    tokens = torch.randn(1, 3, 12)
    tokens[0, 1, 7] = invalid

    with pytest.raises(ValueError, match="valid component tokens must be finite"):
        _make_lift()(
            tokens,
            torch.tensor([0.0, 50.0, 100.0]),
            torch.ones(3, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    "positions,match",
    [
        (torch.zeros(2, 4), "positions"),
        (torch.tensor([0.0, float("nan"), 100.0]), "finite"),
        (torch.tensor([0.0, 100.0, 50.0]), "strictly increasing"),
        (torch.tensor([0.0, 100.0, 400.0]), r"\[0, 1\]"),
    ],
)
def test_invalid_positions_raise_value_error(
    positions: torch.Tensor, match: str
) -> None:
    tokens = torch.randn(1, 3, 12)

    with pytest.raises(ValueError, match=match):
        _make_lift()(tokens, positions, torch.ones(3, dtype=torch.bool))


@pytest.mark.parametrize(
    "time_mask",
    [torch.ones(2, 4, dtype=torch.bool), torch.tensor([1.0, 2.0, 0.0])],
)
def test_invalid_time_mask_raises_value_error(time_mask: torch.Tensor) -> None:
    tokens = torch.randn(1, 3, 12)

    with pytest.raises(ValueError, match="time_mask"):
        _make_lift()(
            tokens, torch.tensor([0.0, 50.0, 100.0]), time_mask
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_dim": 0},
        {"num_basis": 3},
        {"canonical_grid_size": 1},
        {"canonical_grid_size": 80, "roughness_grid_size": 64},
        {"smoothing_weight": -1.0},
        {"time_reference": float("nan")},
        {"time_scale": 0.0},
        {"eps": 0.0},
    ],
)
def test_invalid_construction_parameters_raise_value_error(kwargs: dict) -> None:
    parameters = {"feature_dim": 12}
    parameters.update(kwargs)

    with pytest.raises(ValueError):
        TemporalFunctionalLift(**parameters)
