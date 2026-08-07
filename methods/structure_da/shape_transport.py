"""Pure Domain-Shape transport and target-style source synthesis helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .domain_phase_state import DomainPhaseState, PhaseGroup, PhaseGroupStatus
from .domain_shape_state import DomainShapeState, DomainShapeStatus
from .prototype_bank import SourcePrototypeBank, support_aware_q_distance


@dataclass(frozen=True)
class SyntheticSourceExample:
    source_sample_id: int
    class_id: int
    group_id: int
    trend_tokens: Tensor
    structure_tokens: Tensor
    target_style_positions: Tensor
    mask: Tensor
    q_shape: Tensor
    q_support: Tensor
    lambda_delta: float


@dataclass(frozen=True)
class SyntheticSourceDiagnostics:
    finite: bool
    valid_support: bool
    label_preserved: bool
    target_shape_distance_before: float | None
    target_shape_distance_after: float | None
    target_shape_improved: bool | None
    phase_leakage: float | None
    shape_class_separation_margin: float | None
    classifier_margin: float | None


def _validate_lambda_delta(lambda_delta: float) -> float:
    try:
        value = float(lambda_delta)
    except (TypeError, ValueError) as error:
        raise ValueError("lambda_delta must be a finite real number") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("lambda_delta must lie in [0, 1]")
    return value


def _confirmed_delta(state: DomainShapeState, reference: Tensor) -> Tensor:
    if not isinstance(state, DomainShapeState):
        raise TypeError("domain_shape_state must be a DomainShapeState")
    if state.status is not DomainShapeStatus.CONFIRMED or state.delta is None:
        raise ValueError("Domain Shape transport requires a confirmed state")
    delta = state.delta.detach().to(device=reference.device, dtype=reference.dtype)
    if delta.shape != reference.shape[-2:]:
        raise ValueError("Domain Shape delta must match the SRVF [K,D] shape")
    if not torch.isfinite(delta).all().item():
        raise ValueError("Domain Shape delta must be finite")
    return delta


def apply_domain_shape_effect(
    source_q: Tensor,
    delta_shape: Tensor,
    lambda_delta: float,
) -> Tensor:
    """Algebraic ``q_source + lambda_delta * Delta`` helper.

    This function is intentionally state-free so it can be unit-tested as a
    numerical primitive. Formal Stage-2 construction must use
    :func:`synthesize_source_shape_to_target`, which enforces a confirmed
    :class:`DomainShapeState`.
    """
    lambda_delta = _validate_lambda_delta(lambda_delta)
    if not isinstance(source_q, Tensor) or source_q.ndim not in (2, 3):
        raise ValueError("source_q must have shape [K,D] or [B,K,D]")
    if not source_q.is_floating_point() or not torch.isfinite(source_q).all().item():
        raise ValueError("source_q must be a finite floating-point tensor")
    if not isinstance(delta_shape, Tensor) or delta_shape.shape != source_q.shape[-2:]:
        raise ValueError("delta_shape must have shape [K,D] matching source_q")
    if not delta_shape.is_floating_point() or not torch.isfinite(delta_shape).all().item():
        raise ValueError("delta_shape must be a finite floating-point tensor")
    delta = delta_shape.detach().to(device=source_q.device, dtype=source_q.dtype)
    return source_q + lambda_delta * delta


def correct_target_shape_to_source(
    aligned_target_q: Tensor,
    domain_shape_state: DomainShapeState,
) -> Tensor:
    """Apply the confirmed target-to-source Shape correction ``q_t - Delta``."""
    if not isinstance(aligned_target_q, Tensor) or aligned_target_q.ndim not in (2, 3):
        raise ValueError("aligned_target_q must have shape [K,D] or [B,K,D]")
    if not aligned_target_q.is_floating_point():
        raise ValueError("aligned_target_q must use a floating-point dtype")
    delta = _confirmed_delta(domain_shape_state, aligned_target_q)
    return aligned_target_q - delta


def synthesize_source_shape_to_target(
    source_q: Tensor,
    domain_shape_state: DomainShapeState,
    lambda_delta: float,
) -> Tensor:
    """Apply a confirmed shared Domain Shape effect to source SRVFs."""
    delta = _confirmed_delta(domain_shape_state, source_q)
    return apply_domain_shape_effect(source_q, delta, lambda_delta)


def inverse_vector_srvf(
    q: Tensor,
    initial_value: Tensor,
    *,
    grid: Tensor | None = None,
) -> Tensor:
    """Invert a vector-valued SRVF using ``velocity = q * ||q||_2``.

    Trapezoidal integration is used on the supplied grid. No scalar fdasrsf
    inverse is called.
    """
    if not isinstance(q, Tensor) or q.ndim not in (2, 3):
        raise ValueError("q must have shape [K,D] or [B,K,D]")
    if not q.is_floating_point() or not torch.isfinite(q).all().item():
        raise ValueError("q must be a finite floating-point tensor")
    squeeze = q.ndim == 2
    q_b = q.unsqueeze(0) if squeeze else q
    batch, grid_size, dim = q_b.shape
    if grid_size < 2:
        raise ValueError("q grid must contain at least two points")

    if not isinstance(initial_value, Tensor) or not initial_value.is_floating_point():
        raise ValueError("initial_value must be a floating-point tensor")
    initial = initial_value.to(device=q.device, dtype=q.dtype)
    if initial.ndim == 1:
        if initial.shape != (dim,):
            raise ValueError("initial_value must have shape [D] or [B,D]")
        initial = initial.unsqueeze(0).expand(batch, -1)
    elif initial.shape != (batch, dim):
        raise ValueError("initial_value must have shape [D] or [B,D]")
    if not torch.isfinite(initial).all().item():
        raise ValueError("initial_value must be finite")

    if grid is None:
        grid_t = torch.linspace(
            0.0, 1.0, grid_size, device=q.device, dtype=q.dtype
        )
    else:
        if not isinstance(grid, Tensor) or grid.shape != (grid_size,):
            raise ValueError("grid must have shape [K]")
        if not grid.is_floating_point():
            raise ValueError("grid must use a floating-point dtype")
        grid_t = grid.to(device=q.device, dtype=q.dtype)
        if not torch.isfinite(grid_t).all().item() or not torch.all(
            grid_t[1:] > grid_t[:-1]
        ).item():
            raise ValueError("grid must be finite and strictly increasing")

    speed = torch.linalg.vector_norm(q_b, ord=2, dim=-1)
    velocity = q_b * speed.unsqueeze(-1)
    widths = (grid_t[1:] - grid_t[:-1]).view(1, -1, 1)
    increments = 0.5 * (velocity[:, :-1] + velocity[:, 1:]) * widths
    cumulative = torch.cumsum(increments, dim=1)
    curve = torch.cat(
        [initial.unsqueeze(1), initial.unsqueeze(1) + cumulative], dim=1
    )
    return curve.squeeze(0) if squeeze else curve


def _evaluate_piecewise_linear(values: Tensor, query: Tensor) -> Tensor:
    """Evaluate values sampled on a uniform [0,1] grid at arbitrary queries."""
    if values.ndim not in (1, 2):
        raise ValueError("values must have shape [K] or [K,D]")
    grid_size = values.shape[0]
    if grid_size < 2:
        raise ValueError("values must contain at least two grid points")
    if query.ndim != 1:
        raise ValueError("query must have shape [L]")
    if torch.any((query < 0.0) | (query > 1.0)).item():
        raise ValueError("query must lie in [0,1]")
    scaled = query * (grid_size - 1)
    lower = torch.floor(scaled).to(dtype=torch.long).clamp(0, grid_size - 2)
    upper = lower + 1
    fraction = (scaled - lower.to(dtype=query.dtype)).to(dtype=values.dtype)
    if values.ndim == 1:
        result = values[lower] + fraction * (values[upper] - values[lower])
    else:
        result = values[lower] + fraction.unsqueeze(-1) * (
            values[upper] - values[lower]
        )
    return torch.where(
        (query == 1.0).reshape(-1, *([1] * (result.ndim - 1))),
        values[-1].expand_as(result),
        result,
    )


def map_source_positions_to_target(
    source_positions: Tensor,
    mask: Tensor,
    center_gamma: Tensor,
) -> Tensor:
    """Map positions in the source coordinate by ``gamma(u_source)``."""
    if not isinstance(source_positions, Tensor) or source_positions.ndim != 1:
        raise ValueError("source_positions must have shape [L]")
    if not source_positions.is_floating_point():
        raise ValueError("source_positions must use a floating-point dtype")
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != source_positions.shape:
        raise ValueError("mask must be boolean with shape [L]")
    if not isinstance(center_gamma, Tensor) or center_gamma.ndim != 1:
        raise ValueError("center_gamma must have shape [K_gamma]")
    gamma = center_gamma.detach().to(
        device=source_positions.device, dtype=source_positions.dtype
    )
    if not torch.isfinite(gamma).all().item() or torch.any(
        gamma[1:] <= gamma[:-1]
    ).item():
        raise ValueError("center_gamma must be finite and strictly increasing")
    tolerance = 1e-6
    if (
        abs(float(gamma[0].item())) > tolerance
        or abs(float(gamma[-1].item()) - 1.0) > tolerance
    ):
        raise ValueError("center_gamma must preserve endpoints")
    valid = source_positions[mask]
    if not torch.isfinite(valid).all().item() or torch.any(
        (valid < -tolerance) | (valid > 1.0 + tolerance)
    ).item():
        raise ValueError("valid source positions must be finite and lie in [0,1]")
    safe = torch.where(mask, source_positions.clamp(0.0, 1.0), torch.zeros_like(source_positions))
    mapped = _evaluate_piecewise_linear(gamma, safe)
    mapped = torch.where(mask, mapped, torch.zeros_like(mapped))
    mapped_valid = mapped[mask]
    if mapped_valid.numel() > 1 and not torch.all(
        mapped_valid[1:] > mapped_valid[:-1]
    ).item():
        raise ValueError("mapped valid positions must remain strictly increasing")
    return mapped.detach()


def _confirmed_group_for_class(
    phase_state: DomainPhaseState,
    class_id: int,
) -> PhaseGroup | None:
    if not isinstance(phase_state, DomainPhaseState) or phase_state.m == 0:
        return None
    matches = [
        group
        for group in phase_state.groups
        if group.status is PhaseGroupStatus.CONFIRMED
        and class_id in group.member_classes
    ]
    if len(matches) > 1:
        raise ValueError(f"class {class_id} belongs to multiple confirmed phase groups")
    return matches[0] if matches else None


def _sample_curve_at_positions(curve: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
    safe = torch.where(mask, positions.clamp(0.0, 1.0), torch.zeros_like(positions))
    sampled = _evaluate_piecewise_linear(curve, safe)
    return torch.where(mask.unsqueeze(-1), sampled, torch.zeros_like(sampled))


def _frozen_slow_curve(
    structure_curve: Tensor,
    decomposition: nn.Module,
) -> Tensor:
    """Apply exactly the frozen decomposition slow kernel on the canonical grid."""
    if not hasattr(decomposition, "forward"):
        raise TypeError("decomposition must be a temporal decomposition module")
    grid = torch.linspace(
        0.0,
        1.0,
        structure_curve.shape[0],
        device=structure_curve.device,
        dtype=structure_curve.dtype,
    )
    with torch.no_grad():
        output = decomposition(
            structure_curve.detach().unsqueeze(0),
            grid,
            torch.ones(
                1,
                structure_curve.shape[0],
                dtype=torch.bool,
                device=structure_curve.device,
            ),
        )
    if not hasattr(output, "trend"):
        raise ValueError("decomposition output must expose the slow trend")
    return output.trend.squeeze(0).detach()


def build_synthetic_source_example(
    *,
    source_sample_id: int,
    class_id: int,
    source_structure_function: Tensor,
    source_q_shape: Tensor,
    source_q_support: Tensor,
    source_positions: Tensor,
    mask: Tensor,
    phase_state: DomainPhaseState,
    domain_shape_state: DomainShapeState,
    decomposition: nn.Module,
    lambda_delta: float,
) -> SyntheticSourceExample | None:
    """Construct one phase+Shape target-style source example.

    The function returns ``None`` when either the Domain Shape state or the
    source class phase group is not confirmed. Values are synthesized in the
    source coordinate, while token positions are relabelled in the required
    source-to-target direction ``gamma(source_position)``.
    """
    lambda_delta = _validate_lambda_delta(lambda_delta)
    if domain_shape_state.status is not DomainShapeStatus.CONFIRMED:
        return None
    group = _confirmed_group_for_class(phase_state, class_id)
    if group is None:
        return None
    if not isinstance(source_q_shape, Tensor) or source_q_shape.ndim != 2:
        raise ValueError("source_q_shape must have shape [K,D]")
    if not isinstance(source_structure_function, Tensor) or source_structure_function.shape != source_q_shape.shape:
        raise ValueError("source_structure_function must match source_q_shape [K,D]")
    if not isinstance(source_q_support, Tensor) or source_q_support.shape != source_q_shape.shape[:1]:
        raise ValueError("source_q_support must have shape [K]")
    if not torch.isfinite(source_structure_function).all().item():
        raise ValueError("source_structure_function must be finite")
    if not torch.isfinite(source_q_support).all().item() or torch.any(
        (source_q_support < 0.0) | (source_q_support > 1.0)
    ).item():
        raise ValueError("source_q_support must be finite and lie in [0,1]")
    if not isinstance(source_positions, Tensor) or source_positions.ndim != 1:
        raise ValueError("source_positions must have shape [L]")
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != source_positions.shape:
        raise ValueError("mask must be boolean with shape [L]")

    q_target_style = synthesize_source_shape_to_target(
        source_q_shape.detach(), domain_shape_state, lambda_delta
    ).detach()
    grid = torch.linspace(
        0.0,
        1.0,
        source_q_shape.shape[0],
        device=source_q_shape.device,
        dtype=source_q_shape.dtype,
    )
    structure_curve = inverse_vector_srvf(
        q_target_style,
        source_structure_function[0].detach().to(source_q_shape),
        grid=grid,
    ).detach()
    trend_curve = _frozen_slow_curve(structure_curve, decomposition)
    positions = source_positions.detach().to(
        device=source_q_shape.device, dtype=source_q_shape.dtype
    )
    mask_local = mask.detach().to(device=source_q_shape.device)
    structure_tokens = _sample_curve_at_positions(
        structure_curve, positions, mask_local
    ).detach()
    trend_tokens = _sample_curve_at_positions(
        trend_curve, positions, mask_local
    ).detach()
    target_style_positions = map_source_positions_to_target(
        positions, mask_local, group.center_gamma
    )
    return SyntheticSourceExample(
        source_sample_id=int(source_sample_id),
        class_id=int(class_id),
        group_id=int(group.group_id),
        trend_tokens=trend_tokens,
        structure_tokens=structure_tokens,
        target_style_positions=target_style_positions,
        mask=mask_local,
        q_shape=q_target_style,
        q_support=source_q_support.detach().to(source_q_shape),
        lambda_delta=lambda_delta,
    )


def _integration_weights(grid_size: int, reference: Tensor) -> Tensor:
    weights = torch.full(
        (grid_size,),
        1.0 / (grid_size - 1),
        device=reference.device,
        dtype=reference.dtype,
    )
    weights[[0, -1]] *= 0.5
    return weights


def _single_q_distance(
    q: Tensor,
    support: Tensor,
    prototype_q: Tensor,
    prototype_support: Tensor,
) -> float | None:
    weights = _integration_weights(q.shape[0], q)
    output = support_aware_q_distance(
        q.unsqueeze(0),
        prototype_q.to(q).unsqueeze(0),
        support.to(q).unsqueeze(0),
        prototype_support.to(q).unsqueeze(0),
        weights,
    )
    if not bool(output.valid[0, 0].item()):
        return None
    return float(output.distance[0, 0].item())


def _classifier_margin(logits: Tensor, class_id: int) -> float:
    probabilities = torch.softmax(logits.detach(), dim=-1)
    positive = probabilities[class_id]
    negative = torch.cat((probabilities[:class_id], probabilities[class_id + 1 :]))
    if negative.numel() == 0:
        return float("nan")
    return float((positive - negative.max()).item())


def evaluate_synthetic_source_diagnostics(
    *,
    example: SyntheticSourceExample,
    source_q_shape: Tensor,
    source_logits: Tensor,
    synthetic_logits: Tensor,
    source_positions: Tensor,
    phase_state: DomainPhaseState,
    domain_shape_state: DomainShapeState,
    source_prototype_bank: SourcePrototypeBank,
) -> SyntheticSourceDiagnostics:
    """Compute non-gating diagnostics for one synthetic source example."""
    tensors = (
        example.trend_tokens,
        example.structure_tokens,
        example.target_style_positions,
        example.q_shape,
        example.q_support,
        source_q_shape,
        source_logits,
        synthetic_logits,
    )
    finite = all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)
    valid_support = bool(
        torch.isfinite(example.q_support).all().item()
        and torch.all((example.q_support >= 0.0) & (example.q_support <= 1.0)).item()
        and torch.any(example.q_support > 0.0).item()
    )
    class_id = example.class_id
    source_pred = int(source_logits.detach().argmax().item())
    synthetic_pred = int(synthetic_logits.detach().argmax().item())
    label_preserved = source_pred == class_id and synthetic_pred == class_id

    center = next(
        (
            item
            for item in domain_shape_state.class_centers
            if item.class_id == class_id and item.valid
        ),
        None,
    )
    before = after = None
    improved = None
    if center is not None and valid_support:
        before = _single_q_distance(
            source_q_shape.detach().to(example.q_shape),
            example.q_support,
            center.center_q,
            center.center_support,
        )
        after = _single_q_distance(
            example.q_shape,
            example.q_support,
            center.center_q,
            center.center_support,
        )
        if before is not None and after is not None:
            improved = after <= before

    group = _confirmed_group_for_class(phase_state, class_id)
    phase_leakage = None
    if group is not None:
        expected = map_source_positions_to_target(
            source_positions.detach().to(example.target_style_positions),
            example.mask,
            group.center_gamma,
        )
        if torch.any(example.mask).item():
            phase_leakage = float(
                (
                    example.target_style_positions[example.mask]
                    - expected[example.mask]
                )
                .abs()
                .max()
                .item()
            )
        else:
            phase_leakage = 0.0

    shape_margin = None
    if (
        domain_shape_state.status is DomainShapeStatus.CONFIRMED
        and domain_shape_state.delta is not None
        and valid_support
    ):
        bank = source_prototype_bank
        ready_indices = torch.nonzero(bank.ready, as_tuple=False).flatten()
        if class_id in ready_indices.tolist() and ready_indices.numel() >= 2:
            delta = domain_shape_state.delta.detach().to(example.q_shape)
            prototypes = bank.shape_srvf[ready_indices].detach().to(example.q_shape)
            prototypes = prototypes + example.lambda_delta * delta.unsqueeze(0)
            prototype_support = bank.shape_support[ready_indices].detach().to(example.q_shape)
            distances = support_aware_q_distance(
                example.q_shape.unsqueeze(0),
                prototypes,
                example.q_support.unsqueeze(0),
                prototype_support,
                _integration_weights(example.q_shape.shape[0], example.q_shape),
            )
            local = int(torch.nonzero(ready_indices == class_id, as_tuple=False)[0].item())
            if bool(distances.valid[0, local].item()):
                negative_mask = distances.valid[0].clone()
                negative_mask[local] = False
                if torch.any(negative_mask).item():
                    positive = distances.distance[0, local]
                    negative = distances.distance[0, negative_mask].min()
                    shape_margin = float((negative - positive).item())

    return SyntheticSourceDiagnostics(
        finite=finite,
        valid_support=valid_support,
        label_preserved=label_preserved,
        target_shape_distance_before=before,
        target_shape_distance_after=after,
        target_shape_improved=improved,
        phase_leakage=phase_leakage,
        shape_class_separation_margin=shape_margin,
        classifier_margin=_classifier_margin(synthetic_logits, class_id),
    )
