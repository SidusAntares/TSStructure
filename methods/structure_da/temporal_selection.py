"""Trend-led candidate scoring with stop-gradient structure disambiguation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .temporal_geometry import warp_to_identity_tangent
from .temporal_registration import (
    MonotoneWarpCandidatesOutput,
    MonotoneWarpOutput,
    _apply_srvf_group_action,
    _warp_sequence,
    select_warp_candidate,
)


def _finite_real(name: str, value: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


@dataclass(frozen=True)
class TrendStructureSelectionConfig:
    gain_weight: float = 1.0
    identity_weight: float = 1.0
    roughness_weight: float = 1.0
    unsupported_weight: float = 1.0

    gain_temperature: float = 0.05
    candidate_temperature: float = 0.05

    min_common_support: float = 0.05
    max_gain_ratio: float = 1.0

    ambiguity_relative_tolerance: float = 0.05
    ambiguity_absolute_tolerance: float = 1e-6

    structure_veto_ratio: float = 1.05
    structure_tie_tolerance: float = 1e-6

    min_interval_speed: float | None = None
    max_interval_speed: float | None = None
    max_phase_magnitude: float | None = None
    max_roughness: float | None = None

    identity_tolerance: float = 1e-4
    candidate_unique_tolerance: float = 1e-4

    eps: float = 1e-8

    def __post_init__(self) -> None:
        nonnegative = (
            "gain_weight",
            "identity_weight",
            "roughness_weight",
            "unsupported_weight",
            "ambiguity_relative_tolerance",
            "ambiguity_absolute_tolerance",
            "structure_tie_tolerance",
            "identity_tolerance",
        )
        positive = (
            "gain_temperature",
            "candidate_temperature",
            "max_gain_ratio",
            "structure_veto_ratio",
            "eps",
            "candidate_unique_tolerance",
        )
        for name in nonnegative:
            value = _finite_real(name, getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in positive:
            value = _finite_real(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
            object.__setattr__(self, name, value)

        min_common_support = _finite_real(
            "min_common_support", self.min_common_support
        )
        if not 0.0 <= min_common_support <= 1.0:
            raise ValueError("min_common_support must be in [0, 1]")
        object.__setattr__(self, "min_common_support", min_common_support)

        for name in ("min_interval_speed", "max_interval_speed"):
            value = getattr(self, name)
            if value is not None:
                converted = _finite_real(name, value)
                if converted <= 0:
                    raise ValueError(f"{name} must be greater than zero")
                object.__setattr__(self, name, converted)
        if (
            self.min_interval_speed is not None
            and self.max_interval_speed is not None
            and self.min_interval_speed >= self.max_interval_speed
        ):
            raise ValueError(
                "min_interval_speed must be less than max_interval_speed"
            )
        for name in ("max_phase_magnitude", "max_roughness"):
            value = getattr(self, name)
            if value is not None:
                converted = _finite_real(name, value)
                if converted < 0:
                    raise ValueError(f"{name} must be non-negative")
                object.__setattr__(self, name, converted)


@dataclass(frozen=True)
class TrendStructurePhaseSelectionOutput:
    candidates: MonotoneWarpCandidatesOutput

    candidate_trend_registered_srvf: Tensor
    candidate_trend_registered_support: Tensor
    candidate_structure_registered_srvf: Tensor
    candidate_structure_registered_support: Tensor

    candidate_trend_identity_error: Tensor
    candidate_trend_registered_error: Tensor
    candidate_trend_gain_ratio: Tensor
    candidate_trend_score: Tensor

    candidate_structure_identity_error: Tensor
    candidate_structure_registered_error: Tensor
    candidate_structure_ratio: Tensor

    candidate_common_support: Tensor
    candidate_phase_magnitude: Tensor
    candidate_roughness: Tensor
    candidate_unsupported_error: Tensor
    candidate_speed_min: Tensor
    candidate_speed_max: Tensor

    candidate_trainable_mask: Tensor
    candidate_acceptable_mask: Tensor
    candidate_near_optimal_mask: Tensor
    candidate_softmin_score: Tensor

    candidate_pairwise_distance: Tensor
    candidate_unique_count: Tensor
    candidate_collapse_mask: Tensor
    candidate_selected_is_identity: Tensor

    trend_preferred_candidate_index: Tensor
    selected_candidate_index: Tensor

    trend_candidate_ambiguous: Tensor
    structure_disambiguation_used: Tensor
    structure_candidate_vetoed: Tensor

    accepted_warp: MonotoneWarpOutput
    accepted_inverse_warp: Tensor

    accepted_trend_registered_srvf: Tensor
    accepted_trend_registered_support: Tensor
    accepted_structure_registered_srvf: Tensor
    accepted_structure_registered_support: Tensor

    phase_status: Tensor
    phase_valid: Tensor
    identity_accepted: Tensor
    identity_fallback: Tensor
    structure_shape_valid: Tensor

    @property
    def candidate_selected_index(self) -> Tensor:
        return self.selected_candidate_index


def _resolve_initialized(value: Tensor, batch_size: int, device: torch.device, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean scalar or [B] tensor")
    if value.ndim == 0:
        return value.to(device=device).expand(batch_size)
    if value.shape == (batch_size,):
        if value.device != device:
            raise ValueError(f"{name} device must match SRVF inputs")
        return value
    raise ValueError(f"{name} must be a boolean scalar or [B] tensor")


def _validate_selection_inputs(
    *,
    trend_srvf: Tensor,
    trend_support: Tensor,
    trend_valid: Tensor,
    trend_template_srvf: Tensor,
    trend_template_support: Tensor,
    structure_srvf: Tensor,
    structure_support: Tensor,
    structure_valid: Tensor,
    structure_template_srvf: Tensor,
    structure_template_support: Tensor,
    candidates: MonotoneWarpCandidatesOutput,
    integration_weights: Tensor,
) -> tuple[int, int, int, int]:
    if not isinstance(trend_srvf, Tensor) or trend_srvf.ndim != 3:
        raise ValueError("trend_srvf must have shape [B, K, D]")
    if not trend_srvf.is_floating_point() or not torch.isfinite(trend_srvf).all().item():
        raise ValueError("trend_srvf must contain only finite floating values")
    batch_size, grid_size, feature_dim = trend_srvf.shape
    if grid_size < 2:
        raise ValueError("canonical grid size must be at least 2")
    for name, tensor in (
        ("trend_template_srvf", trend_template_srvf),
        ("structure_srvf", structure_srvf),
        ("structure_template_srvf", structure_template_srvf),
    ):
        if not isinstance(tensor, Tensor) or tensor.shape != trend_srvf.shape:
            raise ValueError(f"{name} must have shape [B, K, D]")
        if tensor.device != trend_srvf.device or tensor.dtype != trend_srvf.dtype:
            raise ValueError(f"{name} must match trend_srvf device and dtype")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values")
    for name, tensor in (
        ("trend_support", trend_support),
        ("trend_template_support", trend_template_support),
        ("structure_support", structure_support),
        ("structure_template_support", structure_template_support),
    ):
        if not isinstance(tensor, Tensor) or tensor.shape != (batch_size, grid_size):
            raise ValueError(f"{name} must have shape [B, K]")
        if tensor.device != trend_srvf.device or tensor.dtype != trend_srvf.dtype:
            raise ValueError(f"{name} must match trend_srvf device and dtype")
        if not torch.isfinite(tensor).all().item() or torch.any((tensor < 0) | (tensor > 1)).item():
            raise ValueError(f"{name} must contain finite values in [0, 1]")
    for name, tensor in (("trend_valid", trend_valid), ("structure_valid", structure_valid)):
        if not isinstance(tensor, Tensor) or tensor.dtype != torch.bool or tensor.shape != (batch_size,):
            raise ValueError(f"{name} must be a boolean tensor with shape [B]")
        if tensor.device != trend_srvf.device:
            raise ValueError(f"{name} device must match trend_srvf")
    if not isinstance(candidates, MonotoneWarpCandidatesOutput):
        raise ValueError("candidates must be a MonotoneWarpCandidatesOutput")
    if not isinstance(candidates.warp, Tensor) or candidates.warp.ndim != 3:
        raise ValueError("candidate warp must have shape [B, G, K]")
    if candidates.warp.shape[0] != batch_size or candidates.warp.shape[2] != grid_size:
        raise ValueError("candidate warp must have shape [B, G, K]")
    num_candidates = candidates.warp.shape[1]
    if num_candidates < 1:
        raise ValueError("candidates must contain at least one candidate")
    expected_grid = (batch_size, num_candidates, grid_size)
    expected_intervals = (batch_size, num_candidates, grid_size - 1)
    for name, tensor, shape in (
        ("interval_logits", candidates.interval_logits, expected_intervals),
        ("interval_widths", candidates.interval_widths, expected_intervals),
        ("warp", candidates.warp, expected_grid),
        ("warp_derivative", candidates.warp_derivative, expected_grid),
        ("inverse_warp", candidates.inverse_warp, expected_grid),
    ):
        if not isinstance(tensor, Tensor) or tensor.shape != shape:
            raise ValueError(f"candidate {name} has invalid shape")
        if tensor.device != trend_srvf.device or tensor.dtype != trend_srvf.dtype:
            raise ValueError(f"candidate {name} must match trend_srvf device and dtype")
    if (
        not isinstance(integration_weights, Tensor)
        or integration_weights.shape != (grid_size,)
        or not integration_weights.is_floating_point()
    ):
        raise ValueError("integration_weights must have shape [K]")
    if integration_weights.device != trend_srvf.device or integration_weights.dtype != trend_srvf.dtype:
        raise ValueError("integration_weights must match trend_srvf device and dtype")
    if not torch.isfinite(integration_weights).all().item() or torch.any(integration_weights < 0).item():
        raise ValueError("integration_weights must be finite and non-negative")
    if integration_weights.sum().item() <= 0:
        raise ValueError("integration_weights must have positive total weight")
    return batch_size, num_candidates, grid_size, feature_dim


def _candidate_weighted_error(
    sample: Tensor,
    template: Tensor,
    support: Tensor,
    weights: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    squared_error = (sample - template).square().sum(dim=-1)
    denominator = (weights * support).sum(dim=-1)
    numerator = (weights * support * squared_error).sum(dim=-1)
    connected_zero = sample.sum(dim=(-1, -2)) * 0.0
    error = torch.where(
        denominator > eps,
        numerator / denominator.clamp_min(eps),
        connected_zero,
    )
    return error, denominator


def _gather_candidate(value: Tensor, index: Tensor) -> Tensor:
    expanded = index.reshape(-1, 1, *([1] * (value.ndim - 2))).expand(
        -1, 1, *value.shape[2:]
    )
    return torch.gather(value, 1, expanded).squeeze(1)


def _candidate_diversity_diagnostics(
    candidate_warp: Tensor, tolerance: float
) -> tuple[Tensor, Tensor, Tensor]:
    differences = candidate_warp[:, :, None, :] - candidate_warp[:, None, :, :]
    pairwise = differences.square().mean(dim=-1).sqrt()
    batch_size, num_candidates, _ = pairwise.shape
    unique_count = torch.ones(
        batch_size, dtype=torch.long, device=candidate_warp.device
    )
    for candidate in range(1, num_candidates):
        matches_prior = (pairwise[:, candidate, :candidate] <= tolerance).any(
            dim=-1
        )
        unique_count = unique_count + (~matches_prior).long()
    return pairwise, unique_count, unique_count <= 1


def select_trend_structure_phase(
    *,
    trend_srvf: Tensor,
    trend_support: Tensor,
    trend_valid: Tensor,
    trend_template_srvf: Tensor,
    trend_template_support: Tensor,
    trend_template_initialized: Tensor,
    structure_srvf: Tensor,
    structure_support: Tensor,
    structure_valid: Tensor,
    structure_template_srvf: Tensor,
    structure_template_support: Tensor,
    structure_template_initialized: Tensor,
    candidates: MonotoneWarpCandidatesOutput,
    integration_weights: Tensor,
    config: TrendStructureSelectionConfig,
) -> TrendStructurePhaseSelectionOutput:
    """Choose one T-generated warp, using S only to resolve T ambiguity."""

    if not isinstance(config, TrendStructureSelectionConfig):
        raise ValueError("config must be a TrendStructureSelectionConfig")
    batch_size, num_candidates, grid_size, feature_dim = _validate_selection_inputs(
        trend_srvf=trend_srvf,
        trend_support=trend_support,
        trend_valid=trend_valid,
        trend_template_srvf=trend_template_srvf,
        trend_template_support=trend_template_support,
        structure_srvf=structure_srvf,
        structure_support=structure_support,
        structure_valid=structure_valid,
        structure_template_srvf=structure_template_srvf,
        structure_template_support=structure_template_support,
        candidates=candidates,
        integration_weights=integration_weights,
    )
    device, dtype = trend_srvf.device, trend_srvf.dtype
    trend_initialized = _resolve_initialized(
        trend_template_initialized, batch_size, device, "trend_template_initialized"
    )
    structure_initialized = _resolve_initialized(
        structure_template_initialized,
        batch_size,
        device,
        "structure_template_initialized",
    )
    template_trend_mean_support = (
        trend_template_support * integration_weights
    ).sum(dim=-1) / integration_weights.sum()
    template_structure_mean_support = (
        structure_template_support * integration_weights
    ).sum(dim=-1) / integration_weights.sum()
    trend_eligible = (
        trend_valid
        & trend_initialized
        & (template_trend_mean_support >= config.min_common_support)
    )
    structure_eligible = (
        structure_valid
        & structure_initialized
        & (template_structure_mean_support >= config.min_common_support)
    )

    candidate_finite = (
        torch.isfinite(candidates.interval_logits).all(dim=-1)
        & torch.isfinite(candidates.interval_widths).all(dim=-1)
        & torch.isfinite(candidates.warp).all(dim=-1)
        & torch.isfinite(candidates.warp_derivative).all(dim=-1)
        & torch.isfinite(candidates.inverse_warp).all(dim=-1)
    )
    widths_positive = (candidates.interval_widths > 0).all(dim=-1)
    derivative_positive = (candidates.warp_derivative > 0).all(dim=-1)
    warp_in_range = ((candidates.warp >= 0) & (candidates.warp <= 1)).all(dim=-1)
    warp_increasing = (candidates.warp[..., 1:] > candidates.warp[..., :-1]).all(dim=-1)
    endpoint_valid = (candidates.warp[..., 0] == 0) & (
        candidates.warp[..., -1] == 1
    )
    candidate_action_valid = (
        candidate_finite
        & derivative_positive
        & warp_in_range
        & warp_increasing
        & endpoint_valid
    )

    identity = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    identity_warp = identity.expand(batch_size, num_candidates, -1)
    identity_widths = torch.full_like(
        candidates.interval_widths, 1.0 / (grid_size - 1)
    )
    safe_warp = torch.where(
        candidate_action_valid.unsqueeze(-1), candidates.warp, identity_warp
    )
    safe_derivative = torch.where(
        candidate_action_valid.unsqueeze(-1),
        candidates.warp_derivative,
        torch.ones_like(candidates.warp_derivative),
    )
    widths_usable = candidate_finite & widths_positive
    normalized_widths = candidates.interval_widths / candidates.interval_widths.sum(
        dim=-1, keepdim=True
    ).clamp_min(config.eps)
    safe_widths = torch.where(
        widths_usable.unsqueeze(-1),
        normalized_widths,
        identity_widths,
    )

    def expand_srvf(value: Tensor) -> Tensor:
        return value[:, None].expand(-1, num_candidates, -1, -1).reshape(
            batch_size * num_candidates, grid_size, feature_dim
        )

    flat_warp = safe_warp.reshape(batch_size * num_candidates, grid_size)
    flat_derivative = safe_derivative.reshape(batch_size * num_candidates, grid_size)
    candidate_trend_registered_srvf = _apply_srvf_group_action(
        expand_srvf(trend_srvf), flat_warp, flat_derivative, config.eps
    ).reshape(batch_size, num_candidates, grid_size, feature_dim)
    candidate_structure_registered_srvf = _apply_srvf_group_action(
        expand_srvf(structure_srvf), flat_warp, flat_derivative, config.eps
    ).reshape(batch_size, num_candidates, grid_size, feature_dim)

    def warp_support(value: Tensor) -> Tensor:
        flat = value[:, None].expand(-1, num_candidates, -1).reshape(
            batch_size * num_candidates, grid_size, 1
        )
        return _warp_sequence(flat, flat_warp).reshape(
            batch_size, num_candidates, grid_size
        )

    candidate_trend_registered_support = warp_support(trend_support)
    candidate_structure_registered_support = warp_support(structure_support)
    weights = integration_weights.reshape(1, 1, grid_size)

    trend_common_support = trend_template_support[:, None] * torch.minimum(
        trend_support[:, None], candidate_trend_registered_support
    )
    candidate_common_support = (weights * trend_common_support).sum(dim=-1)
    trend_template = trend_template_srvf[:, None].expand(-1, num_candidates, -1, -1)
    trend_unregistered = trend_srvf[:, None].expand(-1, num_candidates, -1, -1)
    candidate_trend_identity_error, trend_denominator = _candidate_weighted_error(
        trend_unregistered, trend_template, trend_common_support, weights, config.eps
    )
    candidate_trend_registered_error, _ = _candidate_weighted_error(
        candidate_trend_registered_srvf,
        trend_template,
        trend_common_support,
        weights,
        config.eps,
    )
    candidate_trend_gain_ratio = candidate_trend_registered_error / (
        candidate_trend_identity_error.detach().clamp_min(config.eps)
    )
    positive_support = trend_denominator > config.eps
    candidate_trend_gain_ratio = torch.where(
        positive_support,
        candidate_trend_gain_ratio,
        torch.full_like(candidate_trend_gain_ratio, torch.inf),
    )

    phase = warp_to_identity_tangent(
        safe_widths.reshape(batch_size * num_candidates, grid_size - 1),
        eps=config.eps,
    )
    candidate_phase_magnitude = phase.magnitude.reshape(batch_size, num_candidates)
    speed = candidates.interval_widths * (grid_size - 1)
    candidate_speed_min = speed.amin(dim=-1)
    candidate_speed_max = speed.amax(dim=-1)
    interval_support = 0.5 * (
        candidate_trend_registered_support[..., :-1]
        + candidate_trend_registered_support[..., 1:]
    )
    log_speed = torch.log(speed.clamp_min(config.eps))
    if grid_size - 1 == 1:
        candidate_roughness = log_speed.sum(dim=-1) * 0.0
    else:
        roughness_support = 0.5 * (
            interval_support[..., 1:] + interval_support[..., :-1]
        )
        roughness_numerator = (
            roughness_support * torch.diff(log_speed, dim=-1).square()
        ).sum(dim=-1)
        candidate_roughness = roughness_numerator / roughness_support.sum(
            dim=-1
        ).clamp_min(config.eps)
    unsupported = 1.0 - interval_support
    interval_weights = torch.full_like(unsupported, 1.0 / (grid_size - 1))
    candidate_unsupported_error = (
        interval_weights * unsupported * log_speed.square()
    ).sum(dim=-1) / (interval_weights * unsupported).sum(dim=-1).clamp_min(config.eps)

    gain_loss = config.gain_temperature * F.softplus(
        (candidate_trend_gain_ratio - 1.0) / config.gain_temperature
    )
    raw_trend_score = (
        config.gain_weight * gain_loss
        + config.identity_weight * candidate_phase_magnitude.square()
        + config.roughness_weight * candidate_roughness
        + config.unsupported_weight * candidate_unsupported_error
    )
    diagnostic_finite = (
        torch.isfinite(candidate_trend_gain_ratio)
        & torch.isfinite(raw_trend_score)
        & torch.isfinite(candidate_phase_magnitude)
        & torch.isfinite(candidate_roughness)
        & torch.isfinite(candidate_unsupported_error)
        & torch.isfinite(candidate_speed_min)
        & torch.isfinite(candidate_speed_max)
    )
    score_valid = (
        candidate_action_valid
        & widths_positive
        & diagnostic_finite
        & positive_support
    )
    candidate_trend_score = torch.where(
        score_valid,
        raw_trend_score,
        torch.full_like(raw_trend_score, torch.inf),
    )
    candidate_trainable_mask = (
        trend_eligible[:, None]
        & candidate_finite
        & widths_positive
        & (candidate_speed_min > 0)
        & (candidate_common_support >= config.min_common_support)
        & torch.isfinite(candidate_trend_score)
    )
    if config.min_interval_speed is not None:
        candidate_trainable_mask &= candidate_speed_min >= config.min_interval_speed
    if config.max_interval_speed is not None:
        candidate_trainable_mask &= candidate_speed_max <= config.max_interval_speed
    if config.max_phase_magnitude is not None:
        candidate_trainable_mask &= candidate_phase_magnitude <= config.max_phase_magnitude
    if config.max_roughness is not None:
        candidate_trainable_mask &= candidate_roughness <= config.max_roughness
    candidate_acceptable_mask = candidate_trainable_mask & (
        candidate_trend_gain_ratio <= config.max_gain_ratio
    )
    has_trainable = candidate_trainable_mask.any(dim=1)
    has_acceptable = candidate_acceptable_mask.any(dim=1)
    acceptable_trend_score = torch.where(
        candidate_acceptable_mask,
        candidate_trend_score,
        torch.full_like(candidate_trend_score, torch.inf),
    )
    best_score, preferred = acceptable_trend_score.min(dim=1)
    trend_preferred_candidate_index = torch.where(
        has_acceptable, preferred, torch.zeros_like(preferred)
    )
    finite_best = torch.where(
        has_acceptable, best_score, torch.zeros_like(best_score)
    )
    near_threshold = (
        finite_best
        + config.ambiguity_absolute_tolerance
        + config.ambiguity_relative_tolerance * finite_best.abs()
    )
    candidate_near_optimal_mask = candidate_acceptable_mask & (
        candidate_trend_score <= near_threshold[:, None]
    )
    trend_candidate_ambiguous = candidate_near_optimal_mask.sum(dim=1) >= 2
    trainable_count = candidate_trainable_mask.sum(dim=1)
    softmin_logits = torch.where(
            candidate_trainable_mask,
            -candidate_trend_score / config.candidate_temperature,
            torch.full_like(candidate_trend_score, -torch.inf),
        )
    softmin_logits = torch.where(
        has_trainable[:, None], softmin_logits, torch.zeros_like(softmin_logits)
    )
    logsum = torch.logsumexp(
        softmin_logits,
        dim=1,
    )
    candidate_softmin_score = (
        -config.candidate_temperature * logsum
        + config.candidate_temperature
        * torch.log(trainable_count.clamp_min(1).to(dtype))
    )
    connected_zero = raw_trend_score.nan_to_num().sum(dim=1) * 0.0
    candidate_softmin_score = torch.where(
        has_trainable, candidate_softmin_score, connected_zero
    )

    structure_input = structure_srvf.detach()
    structure_template_input = structure_template_srvf.detach()
    structure_support_input = structure_support.detach()
    structure_template_support_input = structure_template_support.detach()
    structure_registered = candidate_structure_registered_srvf.detach()
    structure_registered_support = candidate_structure_registered_support.detach()
    structure_common_support = structure_template_support_input[:, None] * torch.minimum(
        structure_support_input[:, None], structure_registered_support
    )
    structure_template = structure_template_input[:, None].expand(-1, num_candidates, -1, -1)
    structure_unregistered = structure_input[:, None].expand(-1, num_candidates, -1, -1)
    candidate_structure_identity_error, structure_denominator = _candidate_weighted_error(
        structure_unregistered,
        structure_template,
        structure_common_support,
        weights,
        config.eps,
    )
    candidate_structure_registered_error, _ = _candidate_weighted_error(
        structure_registered,
        structure_template,
        structure_common_support,
        weights,
        config.eps,
    )
    candidate_structure_ratio = candidate_structure_registered_error / (
        candidate_structure_identity_error.clamp_min(config.eps)
    )
    candidate_structure_ratio = torch.where(
        structure_denominator > config.eps,
        candidate_structure_ratio,
        torch.full_like(candidate_structure_ratio, torch.inf),
    )
    structure_common_amount = (weights * structure_common_support).sum(dim=-1)
    structure_pass_mask = (
        candidate_near_optimal_mask
        & torch.isfinite(candidate_structure_ratio)
        & (structure_common_amount >= config.min_common_support)
        & (candidate_structure_ratio <= config.structure_veto_ratio)
    )

    with torch.no_grad():
        use_structure = trend_candidate_ambiguous & structure_eligible
        has_structure_pass = structure_pass_mask.any(dim=1)
        structure_disambiguation_used = use_structure
        structure_candidate_vetoed = use_structure & ~has_structure_pass

        best_structure_ratio = torch.where(
            structure_pass_mask,
            candidate_structure_ratio,
            torch.full_like(candidate_structure_ratio, torch.inf),
        ).amin(dim=1)
        structure_ties = structure_pass_mask & (
            (candidate_structure_ratio - best_structure_ratio[:, None]).abs()
            <= config.structure_tie_tolerance
        )
        tie_trend_scores = torch.where(
            structure_ties,
            candidate_trend_score,
            torch.full_like(candidate_trend_score, torch.inf),
        )
        structure_choice = tie_trend_scores.argmin(dim=1)
        selected_candidate_index = torch.where(
            has_acceptable,
            trend_preferred_candidate_index,
            torch.full_like(trend_preferred_candidate_index, -1),
        )
        selected_candidate_index = torch.where(
            use_structure & has_structure_pass,
            structure_choice,
            selected_candidate_index,
        )
        selected_candidate_index = torch.where(
            structure_candidate_vetoed,
            torch.full_like(selected_candidate_index, -1),
            selected_candidate_index,
        )

        safe_index = selected_candidate_index.clamp_min(0)
        selected_warp = select_warp_candidate(candidates, safe_index)
        identity_logits = torch.zeros(batch_size, grid_size - 1, device=device, dtype=dtype)
        identity_width = torch.full_like(identity_logits, 1.0 / (grid_size - 1))
        identity_batch = identity.expand(batch_size, -1)
        candidate_selected = selected_candidate_index >= 0
        use = candidate_selected[:, None]
        accepted_warp = MonotoneWarpOutput(
            interval_logits=torch.where(use, selected_warp.interval_logits, identity_logits),
            interval_widths=torch.where(use, selected_warp.interval_widths, identity_width),
            warp=torch.where(use, selected_warp.warp, identity_batch),
            warp_derivative=torch.where(use, selected_warp.warp_derivative, torch.ones_like(identity_batch)),
        )
        selected_inverse = _gather_candidate(candidates.inverse_warp, safe_index)
        accepted_inverse_warp = torch.where(use, selected_inverse, identity_batch)
        accepted_trend_registered_srvf = torch.where(
            use.unsqueeze(-1),
            _gather_candidate(candidate_trend_registered_srvf, safe_index),
            trend_srvf,
        )
        accepted_trend_registered_support = torch.where(
            use,
            _gather_candidate(candidate_trend_registered_support, safe_index),
            trend_support,
        )
        accepted_structure_registered_srvf = torch.where(
            use.unsqueeze(-1),
            _gather_candidate(candidate_structure_registered_srvf, safe_index),
            structure_srvf,
        )
        accepted_structure_registered_support = torch.where(
            use,
            _gather_candidate(candidate_structure_registered_support, safe_index),
            structure_support,
        )
        max_abs_deviation = (accepted_warp.warp - identity_batch).abs().amax(
            dim=-1
        )
        accepted_is_identity = max_abs_deviation <= config.identity_tolerance
        candidate_selected_is_identity = candidate_selected & accepted_is_identity
        phase_status = torch.where(
            ~trend_eligible,
            torch.zeros_like(selected_candidate_index),
            torch.where(
                accepted_is_identity,
                torch.ones_like(selected_candidate_index),
                torch.full_like(selected_candidate_index, 2),
            ),
        )
        phase_valid = phase_status > 0
        identity_accepted = phase_status == 1
        identity_fallback = phase_status == 0
        structure_shape_valid = phase_valid & structure_valid

    (
        candidate_pairwise_distance,
        candidate_unique_count,
        candidate_collapse_mask,
    ) = _candidate_diversity_diagnostics(
        safe_warp, config.candidate_unique_tolerance
    )

    return TrendStructurePhaseSelectionOutput(
        candidates=candidates,
        candidate_trend_registered_srvf=candidate_trend_registered_srvf,
        candidate_trend_registered_support=candidate_trend_registered_support,
        candidate_structure_registered_srvf=candidate_structure_registered_srvf,
        candidate_structure_registered_support=candidate_structure_registered_support,
        candidate_trend_identity_error=candidate_trend_identity_error,
        candidate_trend_registered_error=candidate_trend_registered_error,
        candidate_trend_gain_ratio=candidate_trend_gain_ratio,
        candidate_trend_score=candidate_trend_score,
        candidate_structure_identity_error=candidate_structure_identity_error,
        candidate_structure_registered_error=candidate_structure_registered_error,
        candidate_structure_ratio=candidate_structure_ratio,
        candidate_common_support=candidate_common_support,
        candidate_phase_magnitude=candidate_phase_magnitude,
        candidate_roughness=candidate_roughness,
        candidate_unsupported_error=candidate_unsupported_error,
        candidate_speed_min=candidate_speed_min,
        candidate_speed_max=candidate_speed_max,
        candidate_trainable_mask=candidate_trainable_mask,
        candidate_acceptable_mask=candidate_acceptable_mask,
        candidate_near_optimal_mask=candidate_near_optimal_mask,
        candidate_softmin_score=candidate_softmin_score,
        candidate_pairwise_distance=candidate_pairwise_distance,
        candidate_unique_count=candidate_unique_count,
        candidate_collapse_mask=candidate_collapse_mask,
        candidate_selected_is_identity=candidate_selected_is_identity,
        trend_preferred_candidate_index=trend_preferred_candidate_index,
        selected_candidate_index=selected_candidate_index,
        trend_candidate_ambiguous=trend_candidate_ambiguous,
        structure_disambiguation_used=structure_disambiguation_used,
        structure_candidate_vetoed=structure_candidate_vetoed,
        accepted_warp=accepted_warp,
        accepted_inverse_warp=accepted_inverse_warp,
        accepted_trend_registered_srvf=accepted_trend_registered_srvf,
        accepted_trend_registered_support=accepted_trend_registered_support,
        accepted_structure_registered_srvf=accepted_structure_registered_srvf,
        accepted_structure_registered_support=accepted_structure_registered_support,
        phase_status=phase_status,
        phase_valid=phase_valid,
        identity_accepted=identity_accepted,
        identity_fallback=identity_fallback,
        structure_shape_valid=structure_shape_valid,
    )
