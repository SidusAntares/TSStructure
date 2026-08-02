from __future__ import annotations

import pytest
import torch
from torch import nn

from methods.structure_da import (
    SourceRunningSupportScale,
    TemporalFunctionalOutput,
    TemporalSRVFExtractor,
    TemporalSRVFOutput,
)


def _make_extractor(**kwargs) -> TemporalSRVFExtractor:
    parameters = {
        "feature_dim": 2,
        "num_basis": 6,
        "canonical_grid_size": 16,
        "roughness_grid_size": 64,
        "min_mean_support": 0.0,
        "min_dynamic_energy": 0.0,
    }
    parameters.update(kwargs)
    return TemporalSRVFExtractor(**parameters)


def _functional_output(
    *,
    information_variance: torch.Tensor,
    derivative: torch.Tensor,
    solve_valid: torch.Tensor | None = None,
) -> TemporalFunctionalOutput:
    batch_size, grid_size = information_variance.shape
    feature_dim = derivative.shape[-1]
    if solve_valid is None:
        solve_valid = torch.ones(batch_size, dtype=torch.bool)
    dtype = derivative.dtype
    return TemporalFunctionalOutput(
        standardized_tokens=torch.zeros(batch_size, 3, feature_dim, dtype=dtype),
        normalized_positions=torch.zeros(batch_size, 3, dtype=dtype),
        coefficients=torch.zeros(batch_size, 4, feature_dim, dtype=dtype),
        function=torch.zeros(batch_size, grid_size, feature_dim, dtype=dtype),
        derivative=derivative,
        information_variance=information_variance,
        time_mask=torch.ones(batch_size, 3, dtype=torch.bool),
        solve_valid=solve_valid,
        num_valid_observations=torch.full((batch_size,), 3, dtype=torch.long),
        num_distinct_observations=torch.full((batch_size,), 3, dtype=torch.long),
        time_span=torch.ones(batch_size, dtype=dtype),
        max_internal_gap=torch.full((batch_size,), 0.5, dtype=dtype),
    )


class _FixedFunctionalLift(nn.Module):
    def __init__(self, output: TemporalFunctionalOutput) -> None:
        super().__init__()
        self.output = output

    def forward(self, component_tokens, positions, time_mask):
        return self.output


def _forward_fixed(
    extractor: TemporalSRVFExtractor,
    functional: TemporalFunctionalOutput,
) -> TemporalSRVFOutput:
    extractor.functional_lift = _FixedFunctionalLift(functional)
    batch_size = functional.derivative.shape[0]
    tokens = torch.zeros(
        batch_size,
        3,
        extractor.feature_dim,
        dtype=functional.derivative.dtype,
    )
    return extractor(
        tokens,
        torch.tensor([0.0, 100.0, 200.0], dtype=tokens.dtype),
        torch.ones(batch_size, 3, dtype=torch.bool),
    )


def test_support_scale_first_update_uses_valid_mean_only() -> None:
    scale = SourceRunningSupportScale()
    variance = torch.tensor([[1.0, 3.0], [100.0, 200.0], [5.0, 7.0]])
    valid = torch.tensor([True, False, True])

    scale.update(variance, valid)

    torch.testing.assert_close(scale.running_scale, torch.tensor(4.0))
    assert scale.num_updates.item() == 1


def test_support_scale_second_update_uses_ema() -> None:
    scale = SourceRunningSupportScale(momentum=0.75)
    scale.update(torch.tensor([[2.0, 4.0]]), torch.tensor([True]))
    scale.update(torch.tensor([[8.0, 12.0]]), torch.tensor([True]))

    torch.testing.assert_close(
        scale.running_scale, torch.tensor(0.75 * 3.0 + 0.25 * 10.0)
    )
    assert scale.num_updates.item() == 2


def test_support_scale_no_valid_samples_does_not_update() -> None:
    scale = SourceRunningSupportScale(initial_scale=2.5)
    before = {name: value.clone() for name, value in scale.named_buffers()}

    scale.update(torch.ones(2, 4), torch.tensor([False, False]))

    for name, value in scale.named_buffers():
        torch.testing.assert_close(value, before[name])
    torch.testing.assert_close(
        scale(device=torch.device("cpu"), dtype=torch.float64),
        torch.tensor(2.5, dtype=torch.float64),
    )


def test_support_scale_forward_is_read_only_and_uses_requested_dtype() -> None:
    scale = SourceRunningSupportScale()
    scale.update(torch.tensor([[2.0, 6.0]]), torch.tensor([True]))
    before = {name: value.clone() for name, value in scale.named_buffers()}

    result = scale(device=torch.device("cpu"), dtype=torch.float64)

    assert result.dtype == torch.float64
    torch.testing.assert_close(result, torch.tensor(4.0, dtype=torch.float64))
    for name, value in scale.named_buffers():
        torch.testing.assert_close(value, before[name])


def test_support_scale_minimum_and_buffer_registration() -> None:
    scale = SourceRunningSupportScale(min_scale=0.25)
    scale.update(torch.zeros(1, 3), torch.tensor([True]))

    torch.testing.assert_close(scale.running_scale, torch.tensor(0.25))
    assert set(dict(scale.named_buffers())) == {"running_scale", "num_updates"}
    assert not dict(scale.named_parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_extractor_output_shapes_ranges_and_dtype(dtype: torch.dtype) -> None:
    torch.manual_seed(51)
    extractor = _make_extractor()
    tokens = torch.randn(2, 7, 2, dtype=dtype)
    positions = torch.tensor([0, 31, 75, 128, 190, 271, 355], dtype=dtype)
    mask = torch.ones(2, 7, dtype=torch.bool)

    output = extractor(tokens, positions, mask)

    assert isinstance(output, TemporalSRVFOutput)
    assert output.support_confidence.shape == (2, 16)
    assert output.mean_support.shape == (2,)
    assert output.derivative_norm.shape == (2, 16)
    assert output.dynamic_energy.shape == (2,)
    assert output.srvf.shape == (2, 16, 2)
    assert output.structure_valid.shape == (2,)
    assert output.structure_valid.dtype == torch.bool
    for value in (
        output.support_confidence,
        output.mean_support,
        output.derivative_norm,
        output.dynamic_energy,
        output.srvf,
    ):
        assert value.dtype == dtype
        assert torch.isfinite(value).all()
    assert torch.all((output.support_confidence >= 0) & (output.support_confidence <= 1))


def test_integration_weights_are_normalized_trapezoidal_buffer() -> None:
    extractor = _make_extractor(canonical_grid_size=4, roughness_grid_size=64)
    weights = extractor.integration_weights

    assert weights.shape == (4,)
    assert weights.sum().item() == 1.0
    torch.testing.assert_close(weights[0], weights[-1])
    torch.testing.assert_close(weights[1], 2 * weights[0])
    assert "integration_weights" in dict(extractor.named_buffers())
    assert "integration_weights" not in dict(extractor.named_parameters())


def test_support_confidence_is_monotone_in_variance_and_not_softmax() -> None:
    extractor = _make_extractor(canonical_grid_size=4, roughness_grid_size=64)
    functional = _functional_output(
        information_variance=torch.tensor([[0.0, 1.0, 3.0, 9.0]]),
        derivative=torch.ones(1, 4, 2),
    )

    output = _forward_fixed(extractor, functional)

    assert torch.all(output.support_confidence[:, 1:] < output.support_confidence[:, :-1])
    assert not torch.isclose(output.support_confidence.sum(), torch.tensor(1.0))


def test_larger_source_support_scale_increases_confidence() -> None:
    functional = _functional_output(
        information_variance=torch.full((1, 4), 2.0),
        derivative=torch.ones(1, 4, 2),
    )
    small = _make_extractor(canonical_grid_size=4, roughness_grid_size=64)
    large = _make_extractor(canonical_grid_size=4, roughness_grid_size=64)
    small.support_scale.update(torch.ones(1, 2), torch.tensor([True]))
    large.support_scale.update(torch.full((1, 2), 8.0), torch.tensor([True]))

    small_output = _forward_fixed(small, functional)
    large_output = _forward_fixed(large, functional)

    assert torch.all(large_output.support_confidence > small_output.support_confidence)


def test_invalid_solve_forces_confidence_and_srvf_to_zero() -> None:
    functional = _functional_output(
        information_variance=torch.ones(2, 4),
        derivative=torch.ones(2, 4, 2),
        solve_valid=torch.tensor([True, False]),
    )
    output = _forward_fixed(
        _make_extractor(canonical_grid_size=4, roughness_grid_size=64), functional
    )

    torch.testing.assert_close(output.support_confidence[1], torch.zeros(4))
    torch.testing.assert_close(output.srvf[1], torch.zeros(4, 2))
    assert output.structure_valid.tolist() == [True, False]


def test_dense_observations_have_more_mean_support_than_sparse_boundaries() -> None:
    extractor = _make_extractor(canonical_grid_size=32, roughness_grid_size=96)
    tokens = torch.randn(2, 9, 2)
    positions = torch.linspace(0.0, 366.0, 9)
    mask = torch.tensor(
        [[True] * 9, [True, False, False, False, False, False, False, False, True]]
    )

    output = extractor(tokens, positions, mask)

    assert output.mean_support[0] > output.mean_support[1]


def test_vector_srvf_uses_square_root_of_l2_norm_without_support_weighting() -> None:
    eps = 1e-6
    derivative = torch.tensor([[[3.0, 4.0]]] * 4).reshape(1, 4, 2)
    functional = _functional_output(
        information_variance=torch.full((1, 4), 9.0), derivative=derivative
    )
    extractor = _make_extractor(
        canonical_grid_size=4, roughness_grid_size=64, srvf_eps=eps
    )

    output = _forward_fixed(extractor, functional)

    expected = derivative / torch.sqrt(torch.tensor(5.0 + eps))
    torch.testing.assert_close(output.srvf, expected)
    assert torch.all(output.support_confidence < 0.2)


def test_zero_derivative_is_invalid_and_has_zero_srvf() -> None:
    functional = _functional_output(
        information_variance=torch.ones(1, 4), derivative=torch.zeros(1, 4, 2)
    )
    extractor = _make_extractor(
        canonical_grid_size=4,
        roughness_grid_size=64,
        min_dynamic_energy=1e-5,
    )

    output = _forward_fixed(extractor, functional)

    assert not output.structure_valid.item()
    torch.testing.assert_close(output.srvf, torch.zeros_like(output.srvf))


def test_reliability_thresholds_control_structure_validity() -> None:
    functional = _functional_output(
        information_variance=torch.ones(1, 4), derivative=torch.ones(1, 4, 2)
    )
    permissive = _make_extractor(
        canonical_grid_size=4,
        roughness_grid_size=64,
        min_mean_support=0.49,
        min_dynamic_energy=0.70,
    )
    strict_support = _make_extractor(
        canonical_grid_size=4,
        roughness_grid_size=64,
        min_mean_support=0.51,
        min_dynamic_energy=0.0,
    )
    strict_energy = _make_extractor(
        canonical_grid_size=4,
        roughness_grid_size=64,
        min_mean_support=0.0,
        min_dynamic_energy=0.72,
    )

    assert _forward_fixed(permissive, functional).structure_valid.item()
    assert not _forward_fixed(strict_support, functional).structure_valid.item()
    assert not _forward_fixed(strict_energy, functional).structure_valid.item()


def test_masked_values_do_not_change_outputs_and_have_zero_gradient() -> None:
    torch.manual_seed(52)
    extractor = _make_extractor(canonical_grid_size=20, roughness_grid_size=80)
    tokens = torch.randn(1, 6, 2)
    changed = tokens.clone()
    mask = torch.tensor([True, False, True, True, False, True])
    changed[:, ~mask] = 1e8
    positions = torch.tensor([0.0, 35.0, 88.0, 151.0, 240.0, 330.0])

    expected = extractor(tokens, positions, mask)
    actual = extractor(changed, positions, mask)

    for name in (
        "information_variance",
        "support_confidence",
        "mean_support",
        "dynamic_energy",
        "srvf",
    ):
        torch.testing.assert_close(
            getattr(actual.functional, name)
            if name == "information_variance"
            else getattr(actual, name),
            getattr(expected.functional, name)
            if name == "information_variance"
            else getattr(expected, name),
        )
    torch.testing.assert_close(actual.structure_valid, expected.structure_valid)

    differentiable = tokens.clone().requires_grad_()
    output = extractor(differentiable, positions, mask)
    (output.srvf.square().mean() + output.dynamic_energy.mean()).backward()
    assert differentiable.grad is not None
    assert torch.isfinite(differentiable.grad).all()
    assert differentiable.grad[:, mask].abs().sum().item() > 0
    torch.testing.assert_close(
        differentiable.grad[:, ~mask],
        torch.zeros_like(differentiable.grad[:, ~mask]),
        atol=0,
        rtol=0,
    )


def test_forward_does_not_update_source_state_and_explicit_updates_do() -> None:
    extractor = _make_extractor()
    tokens = torch.randn(1, 5, 2)
    positions = torch.linspace(0.0, 300.0, 5)
    mask = torch.ones(1, 5, dtype=torch.bool)

    output = extractor(tokens, positions, mask)
    assert extractor.functional_lift.standardizer.num_updates.item() == 0
    assert extractor.support_scale.num_updates.item() == 0

    extractor.update_source_statistics(tokens, mask)
    extractor.update_source_support_scale(output.functional)
    assert extractor.functional_lift.standardizer.num_updates.item() == 1
    assert extractor.support_scale.num_updates.item() == 1


def test_support_update_uses_only_functional_solve_valid_rows() -> None:
    extractor = _make_extractor(canonical_grid_size=4, roughness_grid_size=64)
    functional = _functional_output(
        information_variance=torch.tensor([[1.0, 3.0, 5.0, 7.0], [100.0] * 4]),
        derivative=torch.ones(2, 4, 2),
        solve_valid=torch.tensor([True, False]),
    )

    extractor.update_source_support_scale(functional)

    torch.testing.assert_close(extractor.support_scale.running_scale, torch.tensor(4.0))


def test_running_support_scale_has_no_gradient_and_is_not_parameter() -> None:
    extractor = _make_extractor()
    tokens = torch.randn(1, 6, 2, requires_grad=True)
    output = extractor(
        tokens, torch.linspace(0.0, 350.0, 6), torch.ones(1, 6, dtype=torch.bool)
    )
    (output.srvf.square().mean() + output.dynamic_energy.mean()).backward()

    assert extractor.support_scale.running_scale.grad is None
    assert "support_scale.running_scale" not in dict(extractor.named_parameters())
    assert not output.support_confidence.requires_grad
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (SourceRunningSupportScale, {"momentum": -0.1}),
        (SourceRunningSupportScale, {"momentum": 1.0}),
        (SourceRunningSupportScale, {"initial_scale": 0.0}),
        (SourceRunningSupportScale, {"min_scale": 0.0}),
        (_make_extractor, {"min_mean_support": -0.1}),
        (_make_extractor, {"min_mean_support": 1.1}),
        (_make_extractor, {"min_dynamic_energy": -1.0}),
        (_make_extractor, {"srvf_eps": 0.0}),
        (_make_extractor, {"derivative_norm_threshold": -1.0}),
    ],
)
def test_invalid_construction_parameters_raise_value_error(factory, kwargs) -> None:
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize(
    "variance,valid,match",
    [
        (torch.ones(2, 3, 1), torch.ones(2, dtype=torch.bool), "shape"),
        (torch.ones(2, 3), torch.ones(3, dtype=torch.bool), "shape"),
        (torch.tensor([[1.0, -1.0]]), torch.tensor([True]), "non-negative"),
        (torch.tensor([[1.0, float("nan")]]), torch.tensor([True]), "finite"),
        (torch.tensor([[1.0, float("inf")]]), torch.tensor([True]), "finite"),
        (torch.ones(1, 2), torch.tensor([1]), "boolean"),
        (torch.ones(1, 2), torch.tensor([2.0]), "boolean"),
    ],
)
def test_support_scale_rejects_invalid_update_inputs(variance, valid, match) -> None:
    with pytest.raises(ValueError, match=match):
        SourceRunningSupportScale().update(variance, valid)


def test_support_update_rejects_wrong_canonical_grid_size() -> None:
    extractor = _make_extractor(canonical_grid_size=4, roughness_grid_size=64)
    functional = _functional_output(
        information_variance=torch.ones(1, 5), derivative=torch.ones(1, 5, 2)
    )

    with pytest.raises(ValueError, match="canonical grid"):
        extractor.update_source_support_scale(functional)


def test_functional_output_is_not_mutated_by_srvf_extraction() -> None:
    functional = _functional_output(
        information_variance=torch.ones(1, 4), derivative=torch.ones(1, 4, 2)
    )
    original_derivative = functional.derivative.clone()
    output = _forward_fixed(
        _make_extractor(canonical_grid_size=4, roughness_grid_size=64), functional
    )

    torch.testing.assert_close(functional.derivative, original_derivative)
    assert output.functional is functional
