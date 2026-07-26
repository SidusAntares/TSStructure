"""Training schedules for detached quality gates."""

import math

import torch


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


def quality_gate_progress(step: float, warmup_steps: float) -> float:
    """Return the clipped linear quality-gate warm-up progress."""

    step = _finite_float("step", step)
    warmup_steps = _finite_float("warmup_steps", warmup_steps)
    if step < 0:
        raise ValueError("step must be greater than or equal to zero")
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be greater than zero")
    return float(min(1.0, step / warmup_steps))


def apply_quality_warmup(
    raw_gate: torch.Tensor, progress: float
) -> torch.Tensor:
    """Blend from an all-one gate to a detached measured quality gate."""

    if not isinstance(raw_gate, torch.Tensor):
        raise ValueError("raw_gate must be a torch.Tensor")
    if not raw_gate.is_floating_point():
        raise ValueError("raw_gate must use a floating-point dtype")
    progress = _finite_float("progress", progress)
    if not 0 <= progress <= 1:
        raise ValueError("progress must be in [0, 1]")
    return 1.0 - progress + progress * raw_gate.detach()


def grl_progress(step: float, warmup_steps: float) -> float:
    """Return the clipped linear progress used only by the GRL schedule."""

    step = _finite_float("step", step)
    warmup_steps = _finite_float("warmup_steps", warmup_steps)
    if step < 0:
        raise ValueError("step must be greater than or equal to zero")
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be greater than zero")
    return float(min(1.0, step / warmup_steps))


def grl_coefficient(
    step: float, warmup_steps: float, gamma: float = 10.0
) -> float:
    """Return the logistic gradient-reversal coefficient for one step."""

    gamma = _finite_float("gamma", gamma)
    if gamma <= 0:
        raise ValueError("gamma must be greater than zero")
    progress = grl_progress(step, warmup_steps)
    return float(2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)
