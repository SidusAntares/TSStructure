from __future__ import annotations

import math

import pytest
import torch

from methods.structure_da.diagnostics import (
    compute_decomposition_diagnostics,
    compute_structure_contribution_diagnostics,
    merge_contribution_diagnostics,
    merge_decomposition_diagnostics,
    summarize_contribution_diagnostics,
    summarize_decomposition_diagnostics,
)


def _decomposition_inputs(dtype: torch.dtype = torch.float32):
    trend = torch.tensor(
        [[[[1.0]], [[2.0]], [[4.0]], [[7.0]]]], dtype=dtype
    )
    dynamics = torch.tensor(
        [[[[0.0]], [[1.0]], [[0.0]], [[-1.0]]]], dtype=dtype
    )
    residual = torch.tensor(
        [[[[1.0]], [[0.5]], [[-0.5]], [[0.0]]]], dtype=dtype
    )
    signal = trend + dynamics + residual
    timestamps = torch.tensor([[0.0, 1.0, 3.0, 6.0]], dtype=dtype)
    mask = torch.ones(1, 4, dtype=torch.bool)
    return signal, trend, dynamics, residual, timestamps, mask


def _summary(*inputs):
    return summarize_decomposition_diagnostics(
        compute_decomposition_diagnostics(*inputs)
    )


def test_complete_energy_identity_closes_with_all_cross_terms() -> None:
    signal, trend, dynamics, residual, timestamps, mask = _decomposition_inputs()
    summary = _summary(signal, trend, dynamics, residual, timestamps, mask)

    assert summary["energy_closure_relative_error"].item() < 1e-12
    naive_rhs = (
        trend.square().mean()
        + dynamics.square().mean()
        + residual.square().mean()
    )
    assert not torch.isclose(signal.square().mean(), naive_rhs)
    assert summary["cos_TD_valid_rate"].item() == 1.0
    assert summary["cos_TR_valid_rate"].item() == 1.0
    assert summary["cos_DR_valid_rate"].item() == 1.0


def test_reconstruction_and_closure_are_independent_diagnostics() -> None:
    signal, trend, dynamics, residual, timestamps, mask = _decomposition_inputs()
    exact = _summary(signal, trend, dynamics, residual, timestamps, mask)
    broken = signal.clone()
    broken[:, 1] += 0.75
    inexact = _summary(broken, trend, dynamics, residual, timestamps, mask)

    assert exact["reconstruction_relative_error"].item() < 1e-7
    assert exact["energy_closure_relative_error"].item() < 1e-12
    assert inexact["reconstruction_relative_error"].item() > 0
    assert inexact["energy_closure_relative_error"].item() > 0
    assert not torch.isclose(
        inexact["reconstruction_relative_error"],
        inexact["energy_closure_relative_error"],
    )


def test_constant_sequence_has_zero_centered_energy_and_roughness() -> None:
    trend = torch.full((2, 4, 2, 3), 5.0)
    zero = torch.zeros_like(trend)
    timestamps = torch.tensor([[0.0, 1.0, 3.0, 6.0]]).expand(2, -1)
    mask = torch.ones(2, 4, dtype=torch.bool)
    summary = _summary(trend, trend, zero, zero, timestamps, mask)

    assert summary["centered_energy_T"].item() == pytest.approx(0.0, abs=1e-12)
    assert summary["roughness_T"].item() == pytest.approx(0.0, abs=1e-12)
    assert summary["roughness_valid_rate_T"].item() == 1.0
    assert summary["normalized_roughness_T"].item() == 0.0
    assert summary["normalized_roughness_valid_rate_T"].item() == 0.0


def test_roughness_orders_low_mid_and_high_frequency_signals() -> None:
    timestamps = torch.linspace(0.0, 2.0 * math.pi, 65).unsqueeze(0)
    mask = torch.ones(1, 65, dtype=torch.bool)
    signals = [
        torch.sin(frequency * timestamps).view(1, 65, 1, 1)
        for frequency in (1.0, 3.0, 7.0)
    ]
    roughness = []
    for signal in signals:
        zero = torch.zeros_like(signal)
        result = _summary(signal, zero, signal, zero, timestamps, mask)
        roughness.append(result["roughness_D"].item())

    assert roughness[0] < roughness[1] < roughness[2]


def test_irregular_roughness_matches_hand_calculation() -> None:
    signal = torch.tensor([0.0, 1.0, 3.0]).view(1, 3, 1, 1)
    zero = torch.zeros_like(signal)
    timestamps = torch.tensor([[0.0, 1.0, 3.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    summary = _summary(signal, zero, signal, zero, timestamps, mask)

    # ((1^2 / 1) + (2^2 / 2)) / (1 + 2) = 1
    assert summary["roughness_D"].item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "timestamps",
    [
        torch.tensor([[0.0, 1.0, 1.0, 3.0]]),
        torch.tensor([[0.0, 2.0, 1.0, 3.0]]),
        torch.tensor([[0.0, float("nan"), 2.0, 3.0]]),
    ],
)
def test_invalid_time_intervals_are_ignored_and_outputs_stay_finite(
    timestamps: torch.Tensor,
) -> None:
    signal = torch.arange(4.0).view(1, 4, 1, 1)
    zero = torch.zeros_like(signal)
    mask = torch.ones(1, 4, dtype=torch.bool)
    summary = _summary(signal, signal, zero, zero, timestamps, mask)

    assert all(torch.isfinite(value).all() for value in summary.values())
    assert summary["roughness_valid_rate_H"].item() == 1.0


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros(1, 4, dtype=torch.bool),
        torch.tensor([[True, False, False, False]]),
        torch.tensor([[True, False, True, False]]),
    ],
)
def test_padding_and_insufficient_intervals_never_create_nan(mask) -> None:
    signal = torch.randn(1, 4, 2, 2)
    zero = torch.zeros_like(signal)
    timestamps = torch.tensor([[0.0, 1.0, 1.0, float("nan")]])
    summary = _summary(signal, signal, zero, zero, timestamps, mask)

    assert all(torch.isfinite(value).all() for value in summary.values())


def test_zero_energy_component_excludes_cosine_instead_of_averaging_zero() -> None:
    signal = torch.randn(2, 4, 1, 1)
    zero = torch.zeros_like(signal)
    timestamps = torch.arange(4.0).expand(2, -1)
    mask = torch.ones(2, 4, dtype=torch.bool)
    summary = _summary(signal, signal, zero, zero, timestamps, mask)

    assert summary["cos_TD"].item() == 0.0
    assert summary["cos_TD_valid_rate"].item() == 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_decomposition_diagnostics_promote_inputs_and_detach(dtype) -> None:
    signal, trend, dynamics, residual, timestamps, mask = _decomposition_inputs(dtype)
    signal.requires_grad_(True)
    result = compute_decomposition_diagnostics(
        signal, trend, dynamics, residual, timestamps, mask
    )
    summary = summarize_decomposition_diagnostics(result)

    assert all(torch.isfinite(value).all() for value in summary.values())
    assert all(not value.requires_grad for value in summary.values())


def test_decomposition_sufficient_statistics_merge_exactly() -> None:
    inputs = _decomposition_inputs()
    duplicated = tuple(
        value.expand(2, *value.shape[1:]).clone()
        if value.ndim > 1 else value
        for value in inputs
    )
    whole = compute_decomposition_diagnostics(*duplicated)
    first = compute_decomposition_diagnostics(
        *(value[:1] for value in duplicated)
    )
    second = compute_decomposition_diagnostics(
        *(value[1:] for value in duplicated)
    )
    merged = merge_decomposition_diagnostics(first, second)

    whole_summary = summarize_decomposition_diagnostics(whole)
    merged_summary = summarize_decomposition_diagnostics(merged)
    for name in whole_summary:
        torch.testing.assert_close(merged_summary[name], whole_summary[name])


def _contribution_inputs(dtype=torch.float32):
    coefficients = {
        "alpha_T": torch.tensor([0.5, 0.0], dtype=dtype),
        "alpha_D": torch.tensor([0.25, 0.5], dtype=dtype),
        "alpha_R": torch.tensor([0.25, 0.5], dtype=dtype),
        "beta_T_temporal": torch.tensor([0.4, 0.8], dtype=dtype),
        "beta_D_temporal": torch.tensor([0.2, 0.0], dtype=dtype),
        "beta_T_channel": torch.tensor([0.6, 0.7], dtype=dtype),
        "beta_D_channel": torch.tensor([0.3, 0.4], dtype=dtype),
    }
    structures = {
        "temporal_T": torch.ones(2, 3, dtype=dtype),
        "temporal_D": 2.0 * torch.ones(2, 3, dtype=dtype),
        "channel_T": 3.0 * torch.ones(2, 3, dtype=dtype),
        "channel_D": 4.0 * torch.ones(2, 3, dtype=dtype),
    }
    fusion = {
        "raw_fusion": torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=dtype),
        "temporal_fusion": torch.tensor([[0.0, 2.0], [0.0, 0.0]], dtype=dtype),
        "channel_fusion": torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=dtype),
    }
    valid = {
        "temporal_T_valid": torch.tensor([True, True]),
        "temporal_D_valid": torch.tensor([True, True]),
        "channel_T_valid": torch.tensor([True, True]),
        "channel_D_valid": torch.tensor([True, True]),
    }
    return {**coefficients, **structures, **fusion, **valid}


def test_zero_alpha_or_beta_zeroes_exact_gate_and_effective_norm() -> None:
    summary = summarize_contribution_diagnostics(
        compute_structure_contribution_diagnostics(**_contribution_inputs())
    )

    assert summary["gate_T_temporal_mean"].item() == pytest.approx(0.1)
    assert summary["gate_D_temporal_mean"].item() == pytest.approx(0.025)
    assert summary["effective_T_temporal_norm"].item() > 0
    assert summary["effective_D_temporal_norm"].item() > 0
    inputs = _contribution_inputs()
    inputs["alpha_T"].zero_()
    inputs["beta_D_channel"].zero_()
    zeroed = summarize_contribution_diagnostics(
        compute_structure_contribution_diagnostics(**inputs)
    )
    assert zeroed["gate_T_temporal_mean"].item() == 0.0
    assert zeroed["effective_T_temporal_norm"].item() == 0.0
    assert zeroed["gate_D_channel_mean"].item() == 0.0
    assert zeroed["effective_D_channel_norm"].item() == 0.0


def test_fusion_shares_sum_to_one_and_zero_energy_is_invalid() -> None:
    summary = summarize_contribution_diagnostics(
        compute_structure_contribution_diagnostics(**_contribution_inputs())
    )

    total = sum(
        summary[name]
        for name in (
            "fusion_share_raw",
            "fusion_share_temporal",
            "fusion_share_channel",
        )
    )
    assert total.item() == pytest.approx(1.0)
    assert summary["fusion_share_valid_rate"].item() == pytest.approx(0.5)
    assert all(torch.isfinite(value) for value in summary.values())


def test_invalid_structure_is_excluded_from_effective_norm_count() -> None:
    inputs = _contribution_inputs()
    inputs["temporal_T_valid"] = torch.tensor([False, True])
    result = compute_structure_contribution_diagnostics(**inputs)

    assert result.effective_norms["T_temporal"].count.item() == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_contribution_diagnostics_are_finite_detached_and_mergeable(dtype) -> None:
    inputs = _contribution_inputs(dtype)
    inputs["temporal_T"].requires_grad_(True)
    first = compute_structure_contribution_diagnostics(**inputs)
    second = compute_structure_contribution_diagnostics(**inputs)
    merged = merge_contribution_diagnostics(first, second)
    summary = summarize_contribution_diagnostics(merged)

    assert all(torch.isfinite(value).all() for value in summary.values())
    assert all(not value.requires_grad for value in summary.values())
    torch.testing.assert_close(
        summary["gate_T_temporal_std"],
        summarize_contribution_diagnostics(first)["gate_T_temporal_std"],
    )
