import pytest
import torch

from methods.structure_da import (
    ChannelRelationOperator,
    StructureOutput,
    TemporalRelationOperator,
    vectorize_channel_statistic,
)


def _assert_structure_close(
    first: StructureOutput,
    second: StructureOutput,
    **kwargs,
) -> None:
    torch.testing.assert_close(first.local, second.local, **kwargs)
    torch.testing.assert_close(first.statistic, second.statistic, **kwargs)
    torch.testing.assert_close(first.valid, second.valid)


def test_temporal_output_shapes() -> None:
    H = torch.randn(3, 5, 4)
    positions = torch.arange(5)

    output = TemporalRelationOperator()(H, positions, 0.05, 0.2)

    assert isinstance(output, StructureOutput)
    assert output.local.shape == (3, 5, 8)
    assert output.statistic.shape == (3, 16)
    assert output.valid.shape == (3,)
    assert output.valid.dtype == torch.bool


def test_temporal_constant_sequence_has_zero_structure() -> None:
    H = torch.tensor([2.0, -1.0, 4.0]).view(1, 1, 3).expand(2, 5, 3)
    positions = torch.tensor([0, 2, 7, 15, 30])

    output = TemporalRelationOperator()(H, positions, 0.05, 0.2)

    torch.testing.assert_close(output.local, torch.zeros_like(output.local))
    torch.testing.assert_close(output.statistic, torch.zeros_like(output.statistic))
    assert output.valid.tolist() == [True, True]


def test_temporal_linear_real_time_sequence_recovers_rate() -> None:
    time_scale = 10.0
    positions = torch.tensor([0.0, 2.0, 5.0, 9.0])
    rate = torch.tensor([1.5, -2.0])
    H = (positions / time_scale).view(1, -1, 1) * rate.view(1, 1, -1)

    output = TemporalRelationOperator(time_scale=time_scale)(
        H, positions, 0.08, 0.3
    )

    expected_local = rate.view(1, 1, -1).expand(1, 4, -1)
    torch.testing.assert_close(output.local[..., :2], expected_local, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(output.local[..., 2:], expected_local, atol=1e-5, rtol=1e-5)
    expected_statistic = torch.cat(
        [rate, torch.zeros_like(rate), rate, torch.zeros_like(rate)]
    ).unsqueeze(0)
    torch.testing.assert_close(
        output.statistic, expected_statistic, atol=1e-5, rtol=1e-5
    )


def test_temporal_true_timestamp_spacing_changes_output() -> None:
    H = torch.tensor([[[0.0], [1.0], [-1.0], [2.0]]])
    operator = TemporalRelationOperator()

    regular = operator(H, torch.tensor([0, 1, 2, 3]), 0.05, 0.2)
    irregular = operator(H, torch.tensor([0, 1, 10, 30]), 0.05, 0.2)

    assert not torch.allclose(regular.local, irregular.local)
    assert not torch.allclose(regular.statistic, irregular.statistic)


def test_temporal_translation_invariance_with_large_offset() -> None:
    H = torch.tensor([[[0.5], [2.0], [-0.5], [3.0]]])
    positions = torch.tensor([0.0, 1.0, 10.0, 30.0], dtype=torch.float64)
    operator = TemporalRelationOperator()

    original = operator(H, positions, 0.05, 0.2)
    shifted = operator(H, positions + 1e9, 0.05, 0.2)

    _assert_structure_close(original, shifted, atol=1e-6, rtol=1e-5)


def test_temporal_permutation_equivariance_and_statistic_invariance() -> None:
    H = torch.tensor([[[0.0, 1.0], [2.0, -1.0], [1.0, 3.0], [4.0, 0.5]]])
    positions = torch.tensor([0, 8, 2, 20])
    time_mask = torch.tensor([True, True, False, True])
    permutation = torch.tensor([2, 0, 3, 1])
    operator = TemporalRelationOperator()

    original = operator(H, positions, 0.05, 0.2, time_mask)
    permuted = operator(
        H[:, permutation],
        positions[permutation],
        0.05,
        0.2,
        time_mask[permutation],
    )

    torch.testing.assert_close(permuted.local, original.local[:, permutation])
    torch.testing.assert_close(permuted.statistic, original.statistic)
    torch.testing.assert_close(permuted.valid, original.valid)


def test_temporal_duplicate_timestamps_remain_permutation_invariant() -> None:
    torch.manual_seed(7)
    H = torch.randn(1, 4, 3)
    positions = torch.tensor([0, 0, 10, 20])
    permutation = torch.tensor([1, 0, 2, 3])
    operator = TemporalRelationOperator()

    original = operator(H, positions, 0.05, 0.2)
    permuted = operator(H[:, permutation], positions[permutation], 0.05, 0.2)

    torch.testing.assert_close(permuted.local, original.local[:, permutation])
    torch.testing.assert_close(permuted.statistic, original.statistic)


@pytest.mark.parametrize("masked_value", [1e6, float("nan"), float("inf")])
def test_temporal_masked_values_do_not_leak(masked_value: float) -> None:
    H = torch.tensor([[[0.0], [1.0], [2.0], [4.0]]])
    positions = torch.tensor([0, 2, 7, 15])
    time_mask = torch.tensor([True, True, False, True])
    changed = H.clone()
    changed[:, 2] = masked_value
    operator = TemporalRelationOperator()

    reference = operator(H, positions, 0.05, 0.2, time_mask)
    output = operator(changed, positions, 0.05, 0.2, time_mask)

    _assert_structure_close(reference, output)
    assert torch.isfinite(output.local).all()
    assert torch.isfinite(output.statistic).all()


def test_temporal_single_observation_is_invalid_and_zero() -> None:
    H = torch.randn(2, 3, 4)
    positions = torch.arange(3)
    time_mask = torch.tensor([[True, False, False], [False, True, False]])

    output = TemporalRelationOperator()(H, positions, 0.05, 0.2, time_mask)

    assert output.valid.tolist() == [False, False]
    torch.testing.assert_close(output.local, torch.zeros_like(output.local))
    torch.testing.assert_close(output.statistic, torch.zeros_like(output.statistic))


def test_temporal_mixed_batch_validity() -> None:
    H = torch.randn(2, 4, 3)
    positions = torch.arange(4)
    time_mask = torch.tensor(
        [[True, False, False, False], [True, True, False, True]]
    )

    output = TemporalRelationOperator()(H, positions, 0.05, 0.2, time_mask)

    assert output.valid.tolist() == [False, True]
    torch.testing.assert_close(output.local[0], torch.zeros_like(output.local[0]))
    torch.testing.assert_close(
        output.statistic[0], torch.zeros_like(output.statistic[0])
    )
    assert torch.count_nonzero(output.local[1]) > 0


def test_temporal_preserves_input_and_tau_gradients() -> None:
    H = torch.randn(2, 5, 3, requires_grad=True)
    positions = torch.tensor([0, 2, 7, 15, 30])
    tau_fast = torch.tensor(0.05, requires_grad=True)
    tau_slow = torch.tensor(0.2, requires_grad=True)

    output = TemporalRelationOperator()(H, positions, tau_fast, tau_slow)
    loss = output.local.square().mean() + output.statistic.square().mean()
    loss.backward()

    for gradient in (H.grad, tau_fast.grad, tau_slow.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_temporal_operator_has_no_trainable_parameters() -> None:
    assert list(TemporalRelationOperator().parameters()) == []


@pytest.mark.parametrize(
    ("tau_fast", "tau_slow"),
    [
        (0.0, 0.2),
        (-0.1, 0.2),
        (0.2, 0.2),
        (0.3, 0.2),
        (float("nan"), 0.2),
        (0.05, float("nan")),
        (0.05, float("inf")),
    ],
)
def test_temporal_invalid_tau_raises_value_error(
    tau_fast: float, tau_slow: float
) -> None:
    with pytest.raises(ValueError):
        TemporalRelationOperator()(
            torch.randn(1, 3, 2), torch.arange(3), tau_fast, tau_slow
        )


def test_temporal_irregular_coverage_changes_statistic() -> None:
    H = torch.tensor([[[0.0], [1.0], [4.0], [-2.0], [3.0]]])
    operator = TemporalRelationOperator(time_scale=30.0)

    regular = operator(H, torch.tensor([0, 5, 10, 15, 20]), 0.1, 0.5)
    irregular = operator(H, torch.tensor([0, 1, 2, 3, 20]), 0.1, 0.5)

    assert not torch.allclose(regular.statistic, irregular.statistic)


def test_temporal_identical_timestamps_use_finite_uniform_coverage() -> None:
    H = torch.tensor([[[0.0], [1.0], [2.0]]])
    positions = torch.tensor([5, 5, 5])

    output = TemporalRelationOperator()(H, positions, 0.05, 0.2)

    assert output.valid.item()
    assert torch.isfinite(output.local).all()
    assert torch.isfinite(output.statistic).all()


def test_channel_output_shapes_symmetry_and_zero_diagonal() -> None:
    H = torch.randn(3, 5, 4)
    positions = torch.arange(5)

    output = ChannelRelationOperator()(H, positions)

    assert isinstance(output, StructureOutput)
    assert output.local.shape == (3, 5, 4)
    assert output.statistic.shape == (3, 4, 4)
    assert output.valid.shape == (3,)
    torch.testing.assert_close(
        output.statistic, output.statistic.transpose(-1, -2)
    )
    assert torch.count_nonzero(torch.diagonal(output.statistic, dim1=-2, dim2=-1)) == 0


def test_channel_known_positive_and_negative_correlations() -> None:
    base = torch.tensor([-2.0, -1.0, 1.0, 3.0])
    H = torch.stack([base, base, -base], dim=-1).unsqueeze(0)
    positions = torch.arange(4)

    statistic = ChannelRelationOperator()(H, positions).statistic

    torch.testing.assert_close(statistic[:, 0, 1], torch.ones(1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(statistic[:, 0, 2], -torch.ones(1), atol=1e-5, rtol=1e-5)


def test_channel_constant_channel_is_finite() -> None:
    varying = torch.tensor([0.0, 1.0, 3.0, 8.0])
    constant = torch.ones_like(varying) * 5
    H = torch.stack([varying, constant, -varying], dim=-1).unsqueeze(0)

    output = ChannelRelationOperator()(H, torch.arange(4))

    assert torch.isfinite(output.local).all()
    assert torch.isfinite(output.statistic).all()


def test_channel_timestamp_coverage_changes_statistic() -> None:
    H = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 2.0, 0.0], [3.0, -1.0, 2.0], [5.0, 4.0, -2.0]]]
    )
    operator = ChannelRelationOperator(time_scale=30.0)

    regular = operator(H, torch.tensor([0, 5, 10, 15]))
    irregular = operator(H, torch.tensor([0, 1, 2, 15]))

    assert not torch.allclose(regular.statistic, irregular.statistic)


def test_channel_translation_invariance_with_large_offset() -> None:
    H = torch.randn(2, 5, 3)
    positions = torch.tensor([0.0, 1.0, 4.0, 10.0, 25.0], dtype=torch.float64)
    operator = ChannelRelationOperator()

    original = operator(H, positions)
    shifted = operator(H, positions + 1e9)

    _assert_structure_close(original, shifted, atol=1e-6, rtol=1e-5)


def test_channel_permutation_equivariance_and_statistic_invariance() -> None:
    H = torch.randn(2, 5, 3)
    positions = torch.tensor([0, 8, 2, 20, 13])
    time_mask = torch.tensor([True, True, False, True, True])
    permutation = torch.tensor([2, 4, 0, 3, 1])
    operator = ChannelRelationOperator()

    original = operator(H, positions, time_mask)
    permuted = operator(
        H[:, permutation], positions[permutation], time_mask[permutation]
    )

    torch.testing.assert_close(permuted.local, original.local[:, permutation])
    torch.testing.assert_close(permuted.statistic, original.statistic)
    torch.testing.assert_close(permuted.valid, original.valid)


def test_channel_duplicate_timestamps_remain_permutation_invariant() -> None:
    torch.manual_seed(8)
    H = torch.randn(1, 4, 3)
    positions = torch.tensor([0, 0, 10, 20])
    permutation = torch.tensor([1, 0, 2, 3])
    operator = ChannelRelationOperator()

    original = operator(H, positions)
    permuted = operator(H[:, permutation], positions[permutation])

    torch.testing.assert_close(permuted.local, original.local[:, permutation])
    torch.testing.assert_close(permuted.statistic, original.statistic)


@pytest.mark.parametrize("masked_value", [1e6, float("nan"), float("inf")])
def test_channel_masked_values_do_not_leak(masked_value: float) -> None:
    H = torch.tensor(
        [[[0.0, 1.0], [1.0, 2.0], [2.0, -1.0], [4.0, 3.0]]]
    )
    positions = torch.tensor([0, 2, 7, 15])
    time_mask = torch.tensor([True, True, False, True])
    changed = H.clone()
    changed[:, 2] = masked_value
    operator = ChannelRelationOperator()

    reference = operator(H, positions, time_mask)
    output = operator(changed, positions, time_mask)

    _assert_structure_close(reference, output)
    assert torch.isfinite(output.local).all()
    assert torch.isfinite(output.statistic).all()


def test_channel_invalid_for_fewer_than_two_observations() -> None:
    H = torch.randn(2, 3, 4)
    positions = torch.arange(3)
    time_mask = torch.tensor([[True, False, False], [False, False, False]])

    output = ChannelRelationOperator()(H, positions, time_mask)

    assert output.valid.tolist() == [False, False]
    torch.testing.assert_close(output.local, torch.zeros_like(output.local))
    torch.testing.assert_close(output.statistic, torch.zeros_like(output.statistic))


def test_channel_single_dimension_is_invalid_and_zero() -> None:
    H = torch.randn(2, 4, 1)

    output = ChannelRelationOperator()(H, torch.arange(4))

    assert output.valid.tolist() == [False, False]
    torch.testing.assert_close(output.local, torch.zeros_like(output.local))
    torch.testing.assert_close(output.statistic, torch.zeros_like(output.statistic))


def test_vectorize_channel_statistic_uses_strict_upper_triangle_order() -> None:
    statistic = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 4.0, 5.0],
                [2.0, 4.0, 0.0, 6.0],
                [3.0, 5.0, 6.0, 0.0],
            ]
        ]
    )

    vector = vectorize_channel_statistic(statistic)

    assert vector.shape == (1, 6)
    torch.testing.assert_close(vector, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]))


def test_channel_gradients_flow_to_input() -> None:
    H = torch.randn(2, 5, 4, requires_grad=True)
    positions = torch.tensor([0, 2, 7, 15, 30])

    output = ChannelRelationOperator()(H, positions)
    loss = output.local.square().mean() + output.statistic.square().mean()
    loss.backward()

    assert H.grad is not None
    assert torch.isfinite(H.grad).all()


def test_channel_operator_has_no_trainable_parameters() -> None:
    assert list(ChannelRelationOperator().parameters()) == []


def test_channel_identical_timestamps_use_finite_uniform_coverage() -> None:
    H = torch.randn(2, 4, 3)
    positions = torch.full((4,), 7)

    output = ChannelRelationOperator()(H, positions)

    assert output.valid.tolist() == [True, True]
    assert torch.isfinite(output.local).all()
    assert torch.isfinite(output.statistic).all()


def test_structure_operators_broadcast_positions_and_mask() -> None:
    H = torch.randn(3, 4, 3)
    positions = torch.tensor([0, 2, 8, 20])
    time_mask = torch.tensor([True, True, False, True])

    temporal = TemporalRelationOperator()(H, positions, 0.05, 0.2, time_mask)
    channel = ChannelRelationOperator()(H, positions, time_mask)

    assert temporal.local.shape == (3, 4, 6)
    assert channel.local.shape == H.shape
    assert temporal.valid.tolist() == [True, True, True]
    assert channel.valid.tolist() == [True, True, True]


@pytest.mark.parametrize("operator", [TemporalRelationOperator, ChannelRelationOperator])
def test_invalid_constructor_values_raise_value_error(operator) -> None:
    with pytest.raises(ValueError):
        operator(time_scale=0.0)


@pytest.mark.parametrize(
    ("H", "positions", "time_mask"),
    [
        (torch.randn(2, 3), torch.arange(3), None),
        (torch.randn(2, 0, 3), torch.empty(0), None),
        (torch.ones(2, 3, 4, dtype=torch.long), torch.arange(3), None),
        (torch.randn(2, 3, 4), torch.arange(4), None),
        (torch.randn(2, 3, 4), torch.full((3,), float("nan")), None),
        (torch.randn(2, 3, 4), torch.arange(3), torch.tensor([1, 2, 0])),
    ],
)
def test_invalid_common_inputs_raise_value_error(
    H: torch.Tensor, positions: torch.Tensor, time_mask: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        ChannelRelationOperator()(H, positions, time_mask)


def test_nonfinite_valid_H_raises_value_error() -> None:
    H = torch.randn(1, 3, 2)
    H[:, 1] = float("nan")

    with pytest.raises(ValueError, match="valid H values must be finite"):
        ChannelRelationOperator()(H, torch.arange(3))


def test_half_precision_outputs_are_finite_and_match_input_dtype() -> None:
    H = torch.randn(2, 4, 3, dtype=torch.float16)
    positions = torch.tensor([0, 10, 30, 60])

    temporal = TemporalRelationOperator()(H, positions, 0.05, 0.2)
    channel = ChannelRelationOperator()(H, positions)

    for component in (
        temporal.local,
        temporal.statistic,
        channel.local,
        channel.statistic,
    ):
        assert component.dtype == H.dtype
        assert torch.isfinite(component).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_extreme_finite_inputs_remain_finite(
    dtype: torch.dtype,
) -> None:
    maximum = torch.finfo(dtype).max
    temporal_H = torch.tensor([[[0.0], [maximum]]], dtype=dtype)
    temporal = TemporalRelationOperator(time_scale=1.0)(
        temporal_H, torch.tensor([0.0, 1e-4]), 0.05, 0.2
    )
    channel_H = torch.tensor(
        [[[maximum, maximum], [0.0, 0.0], [0.0, 0.0]]],
        dtype=dtype,
    )
    channel = ChannelRelationOperator(time_scale=1.0)(
        channel_H, torch.tensor([0.0, 1.0, 1e20], dtype=torch.float64)
    )

    for component in (
        temporal.local,
        temporal.statistic,
        channel.local,
        channel.statistic,
    ):
        assert torch.isfinite(component).all()
