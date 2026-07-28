import pytest
import torch

from methods.structure_da import (
    DecompositionOutput,
    SymmetricTimeKernelDecomposition,
)


def _assert_outputs_close(
    first: DecompositionOutput,
    second: DecompositionOutput,
    **kwargs,
) -> None:
    torch.testing.assert_close(first.trend, second.trend, **kwargs)
    torch.testing.assert_close(first.dynamics, second.dynamics, **kwargs)
    torch.testing.assert_close(first.residual, second.residual, **kwargs)


def test_output_shapes_match_input() -> None:
    torch.manual_seed(0)
    H = torch.randn(3, 5, 7)
    positions = torch.arange(5).expand(3, -1)

    output = SymmetricTimeKernelDecomposition()(H, positions)

    assert isinstance(output, DecompositionOutput)
    assert output.trend.shape == H.shape
    assert output.dynamics.shape == H.shape
    assert output.residual.shape == H.shape


def test_exact_reconstruction_without_mask() -> None:
    torch.manual_seed(1)
    H = torch.randn(2, 6, 4)
    positions = torch.tensor([[0, 2, 5, 9, 14, 20], [1, 3, 8, 12, 17, 25]])

    output = SymmetricTimeKernelDecomposition()(H, positions)
    reconstruction = output.trend + output.dynamics + output.residual

    torch.testing.assert_close(reconstruction, H, atol=1e-6, rtol=1e-5)


def test_masked_exact_reconstruction_and_zero_outputs() -> None:
    torch.manual_seed(2)
    H = torch.randn(2, 5, 3)
    positions = torch.arange(5)
    time_mask = torch.tensor(
        [[True, True, False, True, False], [True, False, True, True, True]]
    )

    output = SymmetricTimeKernelDecomposition()(H, positions, time_mask)
    reconstruction = output.trend + output.dynamics + output.residual
    H_valid = H * time_mask.unsqueeze(-1)

    torch.testing.assert_close(reconstruction, H_valid, atol=1e-6, rtol=1e-5)
    for component in (output.trend, output.dynamics, output.residual):
        torch.testing.assert_close(
            component[~time_mask], torch.zeros_like(component[~time_mask])
        )


def test_masked_value_cannot_leak_into_valid_outputs() -> None:
    H_first = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    H_second = H_first.clone()
    H_second[:, 2] = 1e6
    positions = torch.tensor([0, 2, 7, 15])
    time_mask = torch.tensor([True, True, False, True])
    decomposition = SymmetricTimeKernelDecomposition()

    first = decomposition(H_first, positions, time_mask)
    second = decomposition(H_second, positions, time_mask)

    for first_component, second_component in zip(
        (first.trend, first.dynamics, first.residual),
        (second.trend, second.dynamics, second.residual),
    ):
        torch.testing.assert_close(
            first_component[:, time_mask], second_component[:, time_mask]
        )


@pytest.mark.parametrize("masked_value", [float("nan"), float("inf")])
def test_masked_nonfinite_value_cannot_leak_into_valid_outputs(
    masked_value: float,
) -> None:
    H = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    positions = torch.tensor([0, 2, 7, 15])
    time_mask = torch.tensor([True, True, False, True])
    H[:, 2] = masked_value

    output = SymmetricTimeKernelDecomposition()(H, positions, time_mask)

    for component in (output.trend, output.dynamics, output.residual):
        assert torch.isfinite(component).all()
        torch.testing.assert_close(
            component[:, ~time_mask], torch.zeros_like(component[:, ~time_mask])
        )


def test_nonfinite_value_at_valid_position_raises_value_error() -> None:
    H = torch.tensor([[[1.0], [float("nan")], [3.0]]])
    positions = torch.arange(3)

    with pytest.raises(ValueError, match="valid H values must be finite"):
        SymmetricTimeKernelDecomposition()(H, positions)


def test_all_masked_half_precision_outputs_are_finite_zero() -> None:
    H = torch.randn(2, 4, 3, dtype=torch.float16)
    positions = torch.arange(4)
    time_mask = torch.zeros(2, 4, dtype=torch.bool)

    output = SymmetricTimeKernelDecomposition()(H, positions, time_mask)

    for component in (output.trend, output.dynamics, output.residual):
        assert component.dtype == H.dtype
        assert torch.isfinite(component).all()
        torch.testing.assert_close(component, torch.zeros_like(component))


def test_constant_sequence_has_only_trend_on_valid_positions() -> None:
    constant = torch.tensor([2.5, -1.0, 4.0])
    H = constant.view(1, 1, -1).expand(2, 5, -1).clone()
    positions = torch.tensor([0, 3, 8, 15, 24])
    time_mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, True]]
    )

    output = SymmetricTimeKernelDecomposition()(H, positions, time_mask)

    expected = H * time_mask.unsqueeze(-1)
    torch.testing.assert_close(output.trend, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        output.dynamics, torch.zeros_like(H), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        output.residual, torch.zeros_like(H), atol=1e-6, rtol=1e-5
    )


def test_true_timestamp_spacing_changes_decomposition() -> None:
    H = torch.tensor([[[0.0], [1.0], [-1.0], [2.0]]])
    positions_regular = torch.tensor([0, 1, 2, 3])
    positions_irregular = torch.tensor([0, 1, 10, 30])
    decomposition = SymmetricTimeKernelDecomposition()

    regular = decomposition(H, positions_regular)
    irregular = decomposition(H, positions_irregular)

    assert not torch.allclose(regular.trend, irregular.trend, atol=1e-4, rtol=1e-4)


def test_temporal_shift_does_not_change_decomposition() -> None:
    H = torch.tensor([[[0.5, -1.0], [2.0, 0.0], [-0.5, 1.5], [3.0, 2.0]]])
    positions = torch.tensor([0, 2, 11, 30])
    decomposition = SymmetricTimeKernelDecomposition()

    original = decomposition(H, positions)
    shifted = decomposition(H, positions + 1000)

    _assert_outputs_close(original, shifted, atol=1e-6, rtol=1e-5)


def test_large_temporal_shift_preserves_high_precision_timestamp_differences() -> None:
    H = torch.tensor([[[0.5], [2.0], [-0.5], [3.0]]], dtype=torch.float32)
    positions = torch.tensor([0.0, 1.0, 10.0, 30.0], dtype=torch.float64)
    decomposition = SymmetricTimeKernelDecomposition()

    original = decomposition(H, positions)
    shifted = decomposition(H, positions + 1e9)

    _assert_outputs_close(original, shifted, atol=1e-6, rtol=1e-5)


def test_time_scale_is_applied_before_low_precision_distance_conversion() -> None:
    H = torch.tensor([[[0.0], [1.0]]], dtype=torch.float16)

    large_units = SymmetricTimeKernelDecomposition(time_scale=1e6)(
        H, torch.tensor([0, 100_000])
    )
    small_units = SymmetricTimeKernelDecomposition(time_scale=10.0)(
        H, torch.tensor([0, 1])
    )

    _assert_outputs_close(large_units, small_units, atol=1e-3, rtol=1e-3)


def test_fixed_time_scale_prevents_per_sample_min_max_equivalence() -> None:
    H = torch.tensor([[[0.0], [1.0], [-1.0], [2.0]]])
    positions_short = torch.tensor([0, 1, 2, 3])
    positions_long = torch.tensor([0, 10, 20, 30])
    decomposition = SymmetricTimeKernelDecomposition()

    short = decomposition(H, positions_short)
    long = decomposition(H, positions_long)

    assert not torch.allclose(short.trend, long.trend, atol=1e-4, rtol=1e-4)


def test_tau_initialization_matches_targets_and_is_ordered() -> None:
    decomposition = SymmetricTimeKernelDecomposition(
        tau_fast_init=0.07,
        tau_slow_init=0.31,
        tau_min=1e-4,
        delta_tau_min=2e-4,
    )

    torch.testing.assert_close(decomposition.tau_fast, torch.tensor(0.07))
    torch.testing.assert_close(decomposition.tau_slow, torch.tensor(0.31))
    assert 0 < decomposition.tau_fast.item() < decomposition.tau_slow.item()


@pytest.mark.parametrize(
    ("fast_unconstrained", "gap_unconstrained"),
    [(-80.0, 80.0), (80.0, -80.0), (-80.0, -80.0), (80.0, 80.0)],
)
def test_tau_ordering_survives_extreme_unconstrained_parameters(
    fast_unconstrained: float, gap_unconstrained: float
) -> None:
    decomposition = SymmetricTimeKernelDecomposition()
    parameters = list(decomposition.parameters())

    with torch.no_grad():
        parameters[0].fill_(fast_unconstrained)
        parameters[1].fill_(gap_unconstrained)

    assert 0 < decomposition.tau_fast.item() < decomposition.tau_slow.item()


def test_gradients_flow_to_input_and_both_time_scales() -> None:
    torch.manual_seed(3)
    H = torch.randn(2, 5, 3, requires_grad=True)
    positions = torch.tensor([0.0, 2.0, 7.0, 15.0, 28.0])
    decomposition = SymmetricTimeKernelDecomposition()

    output = decomposition(H, positions)
    loss = (
        output.trend.square().mean()
        + output.dynamics.square().mean()
        + output.residual.square().mean()
    )
    loss.backward()

    assert H.grad is not None
    assert torch.isfinite(H.grad).all()
    for parameter in decomposition.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_single_time_point_is_finite_and_reconstructs() -> None:
    H = torch.tensor([[[1.0, -2.0]], [[3.0, 4.0]]])
    positions = torch.tensor([42])

    output = SymmetricTimeKernelDecomposition()(H, positions)
    reconstruction = output.trend + output.dynamics + output.residual

    for component in (output.trend, output.dynamics, output.residual):
        assert torch.isfinite(component).all()
    torch.testing.assert_close(reconstruction, H, atol=1e-6, rtol=1e-5)


def test_one_dimensional_positions_broadcast_across_batch() -> None:
    torch.manual_seed(4)
    H = torch.randn(4, 6, 2)
    positions = torch.tensor([0, 1, 4, 9, 16, 25])

    output = SymmetricTimeKernelDecomposition()(H, positions)

    assert output.trend.shape == H.shape
    assert output.dynamics.shape == H.shape
    assert output.residual.shape == H.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_scale": 0.0},
        {"time_scale": -1.0},
        {"tau_min": 0.0},
        {"delta_tau_min": 0.0},
        {"tau_fast_init": 1e-4},
        {"tau_fast_init": 0.05, "tau_slow_init": 0.0501, "delta_tau_min": 1e-4},
        {"eps": 0.0},
    ],
)
def test_invalid_initialization_raises_value_error(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SymmetricTimeKernelDecomposition(**kwargs)


@pytest.mark.parametrize(
    ("H", "positions", "time_mask"),
    [
        (torch.randn(2, 3), torch.arange(3), None),
        (torch.randn(2, 0, 3), torch.empty(0), None),
        (torch.randn(2, 3, 4), torch.arange(4), None),
        (torch.randn(2, 3, 4), torch.zeros(1, 3), None),
        (torch.randn(2, 3, 4), torch.arange(3), torch.ones(4)),
        (torch.randn(2, 3, 4), torch.arange(3), torch.ones(1, 3)),
        (torch.randn(2, 3, 4), torch.arange(3), torch.tensor([1, 2, 0])),
    ],
)
def test_invalid_forward_shapes_or_mask_values_raise_value_error(
    H: torch.Tensor, positions: torch.Tensor, time_mask: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        SymmetricTimeKernelDecomposition()(H, positions, time_mask)


def test_float_and_equivalent_long_positions_match() -> None:
    torch.manual_seed(5)
    H = torch.randn(2, 4, 3)
    long_positions = torch.tensor([0, 2, 9, 21], dtype=torch.long)
    float_positions = long_positions.to(torch.float32)
    decomposition = SymmetricTimeKernelDecomposition()

    long_output = decomposition(H, long_positions)
    float_output = decomposition(H, float_positions)

    _assert_outputs_close(long_output, float_output)


def test_all_valid_mask_matches_no_mask() -> None:
    torch.manual_seed(6)
    H = torch.randn(3, 5, 2)
    positions = torch.tensor([0, 1, 5, 12, 20])
    decomposition = SymmetricTimeKernelDecomposition()

    unmasked = decomposition(H, positions)
    all_valid = decomposition(H, positions, torch.ones(5, dtype=torch.bool))

    _assert_outputs_close(unmasked, all_valid)


def test_four_dimensional_output_shapes_match_input() -> None:
    torch.manual_seed(7)
    H = torch.randn(2, 6, 10, 16)
    positions = torch.tensor([0, 3, 9, 20, 41, 75])

    output = SymmetricTimeKernelDecomposition()(H, positions)

    assert output.trend.shape == H.shape
    assert output.dynamics.shape == H.shape
    assert output.residual.shape == H.shape


def test_four_dimensional_exact_reconstruction_without_mask() -> None:
    torch.manual_seed(8)
    H = torch.randn(2, 6, 3, 4)
    positions = torch.tensor([0, 3, 9, 20, 41, 75])

    output = SymmetricTimeKernelDecomposition()(H, positions)
    reconstruction = output.trend + output.dynamics + output.residual
    max_abs_error = (reconstruction - H).abs().max()

    torch.testing.assert_close(reconstruction, H, atol=1e-6, rtol=1e-5)
    assert max_abs_error.item() < 1e-5


def test_four_dimensional_masked_reconstruction_and_zero_outputs() -> None:
    torch.manual_seed(9)
    H = torch.randn(2, 6, 3, 4)
    positions = torch.tensor([0, 3, 9, 20, 41, 75])
    time_mask = torch.tensor(
        [
            [True, True, False, True, False, True],
            [True, False, True, True, True, False],
        ]
    )
    expanded_mask = time_mask.reshape(2, 6, 1, 1)
    H_valid = torch.where(expanded_mask, H, torch.zeros_like(H))

    output = SymmetricTimeKernelDecomposition()(H, positions, time_mask)
    reconstruction = output.trend + output.dynamics + output.residual

    torch.testing.assert_close(reconstruction, H_valid, atol=1e-6, rtol=1e-5)
    for component in (output.trend, output.dynamics, output.residual):
        torch.testing.assert_close(
            component[~time_mask], torch.zeros_like(component[~time_mask])
        )


def test_four_dimensional_masked_nonfinite_values_are_isolated() -> None:
    torch.manual_seed(10)
    H = torch.randn(2, 5, 3, 4)
    positions = torch.tensor([0, 2, 7, 15, 28])
    time_mask = torch.tensor(
        [[True, False, True, True, False], [True, True, False, True, True]]
    )
    changed = H.clone()
    changed[0, 1, 0, 0] = float("nan")
    changed[0, 1, 2, 3] = float("inf")
    changed[1, 2] = float("nan")
    decomposition = SymmetricTimeKernelDecomposition()

    expected = decomposition(H, positions, time_mask)
    actual = decomposition(changed, positions, time_mask)

    for expected_component, actual_component in zip(
        (expected.trend, expected.dynamics, expected.residual),
        (actual.trend, actual.dynamics, actual.residual),
    ):
        assert torch.isfinite(actual_component).all()
        torch.testing.assert_close(actual_component, expected_component)


def test_four_dimensional_valid_nonfinite_value_raises_value_error() -> None:
    H = torch.randn(1, 4, 3, 2)
    H[0, 2, 1, 0] = float("nan")
    positions = torch.tensor([0, 2, 8, 19])

    with pytest.raises(ValueError, match="valid H values must be finite"):
        SymmetricTimeKernelDecomposition()(H, positions)


def test_four_dimensional_decomposition_matches_each_scalar_slot() -> None:
    torch.manual_seed(11)
    H = torch.randn(1, 5, 3, 4)
    positions = torch.tensor([0, 3, 10, 24, 51])
    decomposition = SymmetricTimeKernelDecomposition()

    full_output = decomposition(H, positions)

    for channel in range(H.shape[2]):
        for feature in range(H.shape[3]):
            scalar = H[:, :, channel, feature].unsqueeze(-1)
            scalar_output = decomposition(scalar, positions)
            torch.testing.assert_close(
                full_output.trend[:, :, channel, feature],
                scalar_output.trend.squeeze(-1),
            )
            torch.testing.assert_close(
                full_output.dynamics[:, :, channel, feature],
                scalar_output.dynamics.squeeze(-1),
            )
            torch.testing.assert_close(
                full_output.residual[:, :, channel, feature],
                scalar_output.residual.squeeze(-1),
            )


def test_four_dimensional_single_slot_change_does_not_leak() -> None:
    torch.manual_seed(12)
    H_first = torch.randn(1, 5, 3, 4)
    H_second = H_first.clone()
    H_second[:, :, 1, 2] += torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    positions = torch.tensor([0, 1, 6, 18, 39])
    decomposition = SymmetricTimeKernelDecomposition()
    unchanged_slots = torch.ones(3, 4, dtype=torch.bool)
    unchanged_slots[1, 2] = False

    first = decomposition(H_first, positions)
    second = decomposition(H_second, positions)

    for first_component, second_component in zip(
        (first.trend, first.dynamics, first.residual),
        (second.trend, second.dynamics, second.residual),
    ):
        torch.testing.assert_close(
            first_component[:, :, unchanged_slots],
            second_component[:, :, unchanged_slots],
        )


def test_four_dimensional_gradients_reach_input_and_time_scales() -> None:
    torch.manual_seed(13)
    H = torch.randn(2, 5, 3, 4, requires_grad=True)
    positions = torch.tensor([0.0, 2.0, 7.0, 15.0, 28.0])
    decomposition = SymmetricTimeKernelDecomposition()

    output = decomposition(H, positions)
    loss = (
        output.trend.square().mean()
        + output.dynamics.square().mean()
        + output.residual.square().mean()
    )
    loss.backward()

    assert H.grad is not None
    assert torch.isfinite(H.grad).all()
    assert decomposition._tau_fast_unconstrained.grad is not None
    assert torch.isfinite(decomposition._tau_fast_unconstrained.grad).all()
    assert decomposition._tau_gap_unconstrained.grad is not None
    assert torch.isfinite(decomposition._tau_gap_unconstrained.grad).all()


def test_four_dimensional_constant_sequence_has_only_trend() -> None:
    constants = torch.arange(12, dtype=torch.float32).reshape(3, 4) - 5.5
    H = constants.reshape(1, 1, 3, 4).expand(2, 5, -1, -1).clone()
    positions = torch.tensor([0, 3, 8, 19, 47])

    output = SymmetricTimeKernelDecomposition()(H, positions)

    torch.testing.assert_close(output.trend, H, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        output.dynamics, torch.zeros_like(H), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        output.residual, torch.zeros_like(H), atol=1e-6, rtol=1e-5
    )


def test_four_dimensional_single_time_point_is_exact_and_finite() -> None:
    torch.manual_seed(14)
    H = torch.randn(2, 1, 3, 4)
    positions = torch.tensor([42])

    output = SymmetricTimeKernelDecomposition()(H, positions)
    reconstruction = output.trend + output.dynamics + output.residual

    assert torch.isfinite(output.trend).all()
    assert torch.isfinite(output.dynamics).all()
    assert torch.isfinite(output.residual).all()
    torch.testing.assert_close(output.trend, H, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        output.dynamics, torch.zeros_like(H), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        output.residual, torch.zeros_like(H), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(reconstruction, H, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "H",
    [
        torch.randn(2, 3),
        torch.randn(2, 3, 4, 5, 6),
    ],
)
def test_only_three_and_four_dimensional_inputs_are_supported(H: torch.Tensor) -> None:
    with pytest.raises(
        ValueError,
        match=r"H must have shape \[B, L, D\] or \[B, L, C, P\]",
    ):
        SymmetricTimeKernelDecomposition()(H, torch.arange(3))


@pytest.mark.parametrize(
    "H",
    [
        torch.randn(2, 0, 3),
        torch.randn(2, 0, 3, 4),
        torch.randn(2, 3, 0, 4),
        torch.randn(2, 3, 4, 0),
    ],
)
def test_all_temporal_feature_dimensions_must_be_non_empty(H: torch.Tensor) -> None:
    positions = torch.empty(H.shape[1])

    with pytest.raises(ValueError, match="H feature dimensions must be non-empty"):
        SymmetricTimeKernelDecomposition()(H, positions)
