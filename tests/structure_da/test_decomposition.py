from __future__ import annotations

import pytest
import torch

from methods.structure_da.decomposition import (
    DecompositionOutput,
    SymmetricTimeKernelDecomposition,
)


def _inputs(batch=2, length=6, dim=5):
    torch.manual_seed(7)
    return torch.randn(batch, length, dim), torch.tensor([0, 3, 9, 20, 41, 75])[:length]


def test_outputs_reconstruct_three_dimensional_input() -> None:
    features, positions = _inputs()
    output = SymmetricTimeKernelDecomposition()(features, positions)
    assert isinstance(output, DecompositionOutput)
    assert output.trend.shape == output.dynamics.shape == output.residual.shape == features.shape
    torch.testing.assert_close(output.trend + output.dynamics + output.residual, features)


def test_masked_outputs_are_zero_and_valid_values_reconstruct() -> None:
    features, positions = _inputs()
    mask = torch.tensor([[True, True, False, True, False, True], [True, False, True, True, True, False]])
    output = SymmetricTimeKernelDecomposition()(features, positions, mask)
    expanded = mask.unsqueeze(-1)
    for component in (output.trend, output.dynamics, output.residual):
        assert torch.count_nonzero(component.masked_select(~expanded)) == 0
    expected = torch.where(expanded, features, torch.zeros_like(features))
    torch.testing.assert_close(output.trend + output.dynamics + output.residual, expected)


def test_masked_nonfinite_values_are_isolated() -> None:
    features, positions = _inputs()
    mask = torch.tensor([[True, False, True, True, True, True], [True] * 6])
    changed = features.clone()
    changed[0, 1] = float("nan")
    decomposition = SymmetricTimeKernelDecomposition()
    expected = decomposition(features, positions, mask)
    actual = decomposition(changed, positions, mask)
    for name in ("trend", "dynamics", "residual"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))


def test_time_scales_are_positive_ordered_and_receive_finite_gradients() -> None:
    features, positions = _inputs()
    features.requires_grad_()
    decomposition = SymmetricTimeKernelDecomposition()
    output = decomposition(features, positions)
    (output.trend.square().mean() + output.dynamics.square().mean()).backward()
    assert decomposition.tau_slow > decomposition.tau_fast > 0
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in decomposition.parameters())


def test_constant_sequence_has_only_trend() -> None:
    features = torch.full((2, 5, 4), 3.25)
    output = SymmetricTimeKernelDecomposition()(features, torch.arange(5))
    torch.testing.assert_close(output.trend, features)
    torch.testing.assert_close(output.dynamics, torch.zeros_like(features), atol=1e-6, rtol=0)
    torch.testing.assert_close(output.residual, torch.zeros_like(features), atol=1e-6, rtol=0)


def test_four_dimensional_input_is_rejected_explicitly() -> None:
    with pytest.raises(ValueError, match=r"\[B, L, D\]"):
        SymmetricTimeKernelDecomposition()(torch.randn(2, 5, 3, 4), torch.arange(5))


@pytest.mark.parametrize(
    "features,positions,mask,match",
    [
        (torch.randn(2, 5), torch.arange(5), None, "H must have shape"),
        (torch.randn(2, 5, 3), torch.arange(4), None, "positions"),
        (torch.randn(2, 5, 3), torch.arange(5), torch.ones(2, 4), "time_mask"),
        (torch.empty(2, 5, 0), torch.arange(5), None, "non-empty"),
    ],
)
def test_invalid_inputs_raise_clear_errors(features, positions, mask, match) -> None:
    with pytest.raises(ValueError, match=match):
        SymmetricTimeKernelDecomposition()(features, positions, mask)
