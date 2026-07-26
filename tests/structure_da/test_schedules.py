"""Tests for detached quality-gate warm-up utilities."""

import math

import pytest
import torch

from methods.structure_da import (
    apply_quality_warmup,
    grl_coefficient,
    grl_progress,
    quality_gate_progress,
)


def test_quality_gate_progress_starts_at_zero():
    progress = quality_gate_progress(step=0, warmup_steps=10)

    assert isinstance(progress, float)
    assert progress == 0.0


def test_quality_gate_progress_is_linear_at_midpoint():
    assert quality_gate_progress(step=5, warmup_steps=10) == 0.5


@pytest.mark.parametrize("step", [10, 11, 100])
def test_quality_gate_progress_saturates_after_warmup(step):
    assert quality_gate_progress(step=step, warmup_steps=10) == 1.0


@pytest.mark.parametrize(
    "step, warmup_steps",
    [
        (-1, 10),
        (0, 0),
        (0, -1),
        (float("nan"), 10),
        (0, float("inf")),
        (True, 10),
        (0, False),
    ],
)
def test_quality_gate_progress_rejects_invalid_arguments(step, warmup_steps):
    with pytest.raises(ValueError):
        quality_gate_progress(step=step, warmup_steps=warmup_steps)


def test_apply_quality_warmup_returns_ones_at_zero_progress():
    raw_gate = torch.tensor([0.1, 0.4, 0.9], requires_grad=True)

    effective = apply_quality_warmup(raw_gate, progress=0.0)

    torch.testing.assert_close(effective, torch.ones_like(raw_gate))


def test_apply_quality_warmup_returns_detached_gate_at_full_progress():
    raw_gate = torch.tensor([0.1, 0.4, 0.9], requires_grad=True)

    effective = apply_quality_warmup(raw_gate, progress=1.0)

    torch.testing.assert_close(effective, raw_gate)
    assert not effective.requires_grad


def test_apply_quality_warmup_uses_exact_midpoint_formula():
    raw_gate = torch.tensor([0.2, 0.6, 1.0], requires_grad=True)

    effective = apply_quality_warmup(raw_gate, progress=0.5)

    torch.testing.assert_close(effective, 0.5 + 0.5 * raw_gate.detach())


def test_apply_quality_warmup_never_backpropagates_to_raw_gate():
    raw_gate = torch.tensor([0.2, 0.6, 1.0], requires_grad=True)

    effective = apply_quality_warmup(raw_gate, progress=0.3)

    assert not effective.requires_grad
    assert effective.grad_fn is None
    assert raw_gate.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_apply_quality_warmup_preserves_dtype_and_device(dtype):
    raw_gate = torch.tensor([0.2, 0.8], dtype=dtype)

    effective = apply_quality_warmup(raw_gate, progress=0.4)

    assert effective.dtype == raw_gate.dtype
    assert effective.device == raw_gate.device


@pytest.mark.parametrize(
    "progress", [-0.1, 1.1, float("nan"), float("inf"), True]
)
def test_apply_quality_warmup_rejects_invalid_progress(progress):
    with pytest.raises(ValueError, match="progress"):
        apply_quality_warmup(torch.tensor([0.5]), progress)


def test_apply_quality_warmup_rejects_non_tensor_or_non_floating_gate():
    with pytest.raises(ValueError, match="raw_gate"):
        apply_quality_warmup([0.5], 0.5)
    with pytest.raises(ValueError, match="floating"):
        apply_quality_warmup(torch.tensor([1], dtype=torch.long), 0.5)


def test_grl_progress_starts_at_zero_and_is_a_python_float():
    progress = grl_progress(step=0, warmup_steps=10)

    assert isinstance(progress, float)
    assert progress == 0.0


def test_grl_progress_is_linear_at_midpoint():
    assert grl_progress(step=5, warmup_steps=10) == 0.5


@pytest.mark.parametrize("step", [10, 11, 100])
def test_grl_progress_saturates_after_warmup(step):
    assert grl_progress(step=step, warmup_steps=10) == 1.0


@pytest.mark.parametrize(
    "step, warmup_steps",
    [
        (-1, 10),
        (0, 0),
        (0, -1),
        (float("nan"), 10),
        (0, float("inf")),
        (True, 10),
        (0, False),
    ],
)
def test_grl_progress_rejects_invalid_arguments(step, warmup_steps):
    with pytest.raises(ValueError):
        grl_progress(step=step, warmup_steps=warmup_steps)


def test_grl_coefficient_starts_at_zero():
    assert grl_coefficient(step=0, warmup_steps=10) == 0.0


def test_grl_coefficient_matches_logistic_formula_at_midpoint_and_completion():
    midpoint = grl_coefficient(step=5, warmup_steps=10, gamma=10.0)
    completed = grl_coefficient(step=10, warmup_steps=10, gamma=10.0)

    assert midpoint == pytest.approx(2 / (1 + math.exp(-5.0)) - 1)
    assert completed == pytest.approx(2 / (1 + math.exp(-10.0)) - 1)
    assert 0 < midpoint < completed < 1


def test_grl_coefficient_is_monotonic_and_finite():
    values = [
        grl_coefficient(step=step, warmup_steps=10, gamma=3.0)
        for step in range(11)
    ]

    assert values == sorted(values)
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in values)


@pytest.mark.parametrize("gamma", [0, -1, float("nan"), float("inf"), True])
def test_grl_coefficient_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError, match="gamma"):
        grl_coefficient(step=1, warmup_steps=10, gamma=gamma)
