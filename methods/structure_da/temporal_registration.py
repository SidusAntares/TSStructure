"""Source-template registration for support-aware temporal SRVFs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .temporal_functional import TemporalFunctionalOutput, _finite_float
from .temporal_srvf import TemporalSRVFExtractor, TemporalSRVFOutput


def _positive_integer(name: str, value: int, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _calibrate_candidate_profile_scale(
    profile: Tensor,
    target_max_deviation: float,
    min_increment: float,
) -> float:
    """Map a desired warp displacement to a low-frequency logit scale."""

    if target_max_deviation == 0.0 or profile.numel() < 2:
        return 0.0
    identity = torch.linspace(
        0.0, 1.0, profile.numel() + 1, dtype=torch.float64
    )

    def deviation(scale: float) -> float:
        increments = F.softplus(profile * scale) + min_increment
        widths = increments / increments.sum()
        warp = torch.cat(
            [
                identity.new_zeros(1),
                widths.cumsum(0)[:-1],
                identity.new_ones(1),
            ]
        )
        return float((warp - identity).abs().amax().item())

    lower, upper = 0.0, 1.0
    for _ in range(32):
        if deviation(upper) >= target_max_deviation:
            break
        upper *= 2.0
    else:
        raise ValueError(
            "candidate_init_warp_amplitude cannot be reached safely on this grid"
        )
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        if deviation(middle) < target_max_deviation:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _initial_candidate_base_logits(
    num_candidates: int,
    canonical_grid_size: int,
    target_max_deviation: float,
    min_increment: float,
) -> Tensor:
    intervals = canonical_grid_size - 1
    positions = (torch.arange(intervals, dtype=torch.float64) + 0.5) / intervals
    profile = torch.cos(torch.pi * positions)
    logits = torch.zeros(num_candidates, intervals, dtype=torch.float64)
    if num_candidates == 1 or target_max_deviation == 0.0:
        return logits.float()
    pair_count = (num_candidates - 1 + 1) // 2
    for candidate in range(1, num_candidates):
        pair = (candidate + 1) // 2
        amplitude = target_max_deviation * pair / pair_count
        scale = _calibrate_candidate_profile_scale(
            profile, amplitude, min_increment
        )
        sign = 1.0 if candidate % 2 == 1 else -1.0
        logits[candidate] = sign * scale * profile
    return logits.float()


@dataclass(frozen=True)
class SourceSRVFTemplateOutput:
    srvf: Tensor
    support: Tensor
    initialized: Tensor


class SourceRunningSRVFTemplate(nn.Module):
    """Maintain a support-weighted source SRVF template as running buffers."""

    def __init__(
        self,
        canonical_grid_size: int,
        feature_dim: int,
        momentum: float = 0.99,
        min_grid_weight: float = 1e-6,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        canonical_grid_size = _positive_integer(
            "canonical_grid_size", canonical_grid_size, minimum=2
        )
        feature_dim = _positive_integer("feature_dim", feature_dim)
        momentum = _finite_float("momentum", momentum)
        min_grid_weight = _finite_float("min_grid_weight", min_grid_weight)
        eps = _finite_float("eps", eps)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if min_grid_weight <= 0:
            raise ValueError("min_grid_weight must be greater than zero")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")

        self.canonical_grid_size = canonical_grid_size
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.min_grid_weight = min_grid_weight
        self.eps = eps
        self.register_buffer(
            "running_srvf", torch.zeros(canonical_grid_size, feature_dim)
        )
        self.register_buffer(
            "running_support", torch.zeros(canonical_grid_size)
        )
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def _validate_update_inputs(
        self,
        srvf: Tensor,
        support_confidence: Tensor,
        sample_valid: Tensor,
    ) -> None:
        if (
            not isinstance(srvf, Tensor)
            or srvf.ndim != 3
            or srvf.shape[1:] != (
                self.canonical_grid_size,
                self.feature_dim,
            )
        ):
            raise ValueError("srvf must have shape [B, K, D]")
        if not srvf.is_floating_point():
            raise ValueError("srvf must use a floating-point dtype")
        if not torch.isfinite(srvf).all().item():
            raise ValueError("srvf must contain only finite values")
        if (
            not isinstance(support_confidence, Tensor)
            or support_confidence.shape
            != (srvf.shape[0], self.canonical_grid_size)
        ):
            raise ValueError("support_confidence must have shape [B, K]")
        if not support_confidence.is_floating_point():
            raise ValueError("support_confidence must use a floating-point dtype")
        if not torch.isfinite(support_confidence).all().item():
            raise ValueError("support_confidence must contain only finite values")
        if torch.any(
            (support_confidence < 0) | (support_confidence > 1)
        ).item():
            raise ValueError("support_confidence must lie in [0, 1]")
        if not isinstance(sample_valid, Tensor) or sample_valid.dtype != torch.bool:
            raise ValueError("sample_valid must be a boolean tensor with shape [B]")
        if sample_valid.shape != (srvf.shape[0],):
            raise ValueError("sample_valid must be a boolean tensor with shape [B]")

    @torch.no_grad()
    def update(
        self,
        srvf: Tensor,
        support_confidence: Tensor,
        sample_valid: Tensor,
    ) -> None:
        self._validate_update_inputs(srvf, support_confidence, sample_valid)
        support = support_confidence.to(device=srvf.device, dtype=srvf.dtype)
        valid = sample_valid.to(device=srvf.device)
        if not torch.any(valid).item():
            return

        weights = support * valid.unsqueeze(-1)
        weight_sum = weights.sum(dim=0)
        grid_valid = weight_sum >= self.min_grid_weight
        if not torch.any(grid_valid).item():
            return
        batch_srvf = (
            weights.unsqueeze(-1) * srvf
        ).sum(dim=0) / weight_sum.clamp_min(self.eps).unsqueeze(-1)
        batch_support = support[valid].mean(dim=0)

        batch_srvf = batch_srvf.to(
            device=self.running_srvf.device,
            dtype=self.running_srvf.dtype,
        )
        batch_support = batch_support.to(
            device=self.running_support.device,
            dtype=self.running_support.dtype,
        )
        grid_valid = grid_valid.to(device=self.running_support.device)
        if self.num_updates.item() == 0:
            updated_srvf = batch_srvf
            updated_support = batch_support
        else:
            updated_srvf = (
                self.momentum * self.running_srvf
                + (1.0 - self.momentum) * batch_srvf
            )
            updated_support = (
                self.momentum * self.running_support
                + (1.0 - self.momentum) * batch_support
            )
        self.running_srvf.copy_(
            torch.where(
                grid_valid.unsqueeze(-1),
                updated_srvf,
                self.running_srvf,
            )
        )
        self.running_support.copy_(
            torch.where(grid_valid, updated_support, self.running_support)
        )
        self.num_updates.add_(1)

    def forward(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SourceSRVFTemplateOutput:
        if not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point dtype")
        return SourceSRVFTemplateOutput(
            srvf=self.running_srvf.to(device=device, dtype=dtype),
            support=self.running_support.to(device=device, dtype=dtype),
            initialized=(self.num_updates > 0).to(device=device),
        )


@dataclass(frozen=True)
class MonotoneWarpOutput:
    interval_logits: Tensor
    interval_widths: Tensor
    warp: Tensor
    warp_derivative: Tensor


@dataclass(frozen=True)
class MonotoneWarpCandidatesOutput:
    interval_logits: Tensor
    interval_widths: Tensor
    warp: Tensor
    warp_derivative: Tensor
    inverse_warp: Tensor


def invert_monotone_warp(
    warp: Tensor,
    query: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Invert endpoint-preserving piecewise-linear monotone warps."""

    eps = _finite_float("eps", eps)
    if eps <= 0:
        raise ValueError("eps must be greater than zero")
    if not isinstance(warp, Tensor) or warp.ndim < 1 or warp.shape[-1] < 2:
        raise ValueError("warp must have shape [..., K] with K >= 2")
    if not warp.is_floating_point():
        raise ValueError("warp must use a floating-point dtype")
    if not torch.isfinite(warp).all().item():
        raise ValueError("warp must contain only finite values")
    if torch.any((warp < 0) | (warp > 1)).item():
        raise ValueError("warp must lie in [0, 1]")
    if torch.any(warp[..., 1:] <= warp[..., :-1]).item():
        raise ValueError("warp must be strictly increasing")
    if not torch.allclose(
        warp[..., 0], torch.zeros_like(warp[..., 0]), atol=eps, rtol=0.0
    ):
        raise ValueError("warp must start at 0")
    if not torch.allclose(
        warp[..., -1], torch.ones_like(warp[..., -1]), atol=eps, rtol=0.0
    ):
        raise ValueError("warp must end at 1")

    grid_size = warp.shape[-1]
    leading_shape = warp.shape[:-1]
    if query is None:
        query_tensor = torch.linspace(
            0.0, 1.0, grid_size, device=warp.device, dtype=warp.dtype
        ).expand(*leading_shape, grid_size)
    else:
        try:
            query_tensor = torch.as_tensor(query, device=warp.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                "query must be a real tensor convertible to warp dtype"
            ) from error
        if query_tensor.is_complex() or query_tensor.dtype == torch.bool:
            raise ValueError("query must contain real numeric values")
        query_tensor = query_tensor.to(dtype=warp.dtype)
        if query_tensor.ndim < 1:
            raise ValueError("query must have shape [L] or [..., L]")
        if query_tensor.ndim == 1:
            query_tensor = query_tensor.expand(*leading_shape, query_tensor.shape[0])
        elif query_tensor.shape[:-1] != leading_shape:
            raise ValueError("query leading dimensions must match warp")
        if not torch.isfinite(query_tensor).all().item():
            raise ValueError("query must contain only finite values")
        if torch.any((query_tensor < 0) | (query_tensor > 1)).item():
            raise ValueError("query must lie in [0, 1]")

    output_length = query_tensor.shape[-1]
    flat_warp = warp.reshape(-1, grid_size).contiguous()
    flat_query = query_tensor.reshape(-1, output_length).contiguous()
    upper = torch.searchsorted(flat_warp, flat_query, right=True).clamp(
        min=1, max=grid_size - 1
    )
    lower = upper - 1
    lower_value = torch.gather(flat_warp, 1, lower)
    upper_value = torch.gather(flat_warp, 1, upper)
    fraction = (flat_query - lower_value) / (
        upper_value - lower_value
    ).clamp_min(eps)
    inverse = (lower.to(dtype=warp.dtype) + fraction) / (grid_size - 1)
    inverse = torch.where(flat_query == 0, torch.zeros_like(inverse), inverse)
    inverse = torch.where(flat_query == 1, torch.ones_like(inverse), inverse)
    return inverse.clamp(0.0, 1.0).reshape(*leading_shape, output_length)


def select_warp_candidate(
    candidates: MonotoneWarpCandidatesOutput,
    candidate_index: Tensor,
) -> MonotoneWarpOutput:
    """Select one warp candidate independently for every batch row."""

    if not isinstance(candidates, MonotoneWarpCandidatesOutput):
        raise ValueError("candidates must be a MonotoneWarpCandidatesOutput")
    if candidates.warp.ndim != 3:
        raise ValueError("candidate warp must have shape [B, G, K]")
    batch_size, num_candidates, grid_size = candidates.warp.shape
    expected_shapes = {
        "interval_logits": (batch_size, num_candidates, grid_size - 1),
        "interval_widths": (batch_size, num_candidates, grid_size - 1),
        "warp_derivative": (batch_size, num_candidates, grid_size),
        "inverse_warp": (batch_size, num_candidates, grid_size),
    }
    for name, expected_shape in expected_shapes.items():
        value = getattr(candidates, name)
        if not isinstance(value, Tensor) or value.shape != expected_shape:
            raise ValueError(f"candidate {name} has an invalid shape")
    if not isinstance(candidate_index, Tensor) or candidate_index.dtype != torch.long:
        raise ValueError("candidate_index must use torch.long dtype")
    if candidate_index.shape != (batch_size,):
        raise ValueError("candidate_index must have shape [B]")
    if candidate_index.device != candidates.warp.device:
        raise ValueError("candidate_index must be on the candidates device")
    if torch.any((candidate_index < 0) | (candidate_index >= num_candidates)).item():
        raise ValueError("candidate_index values must lie in the candidate range")

    def gather(value: Tensor) -> Tensor:
        index = candidate_index[:, None, None].expand(-1, 1, value.shape[-1])
        return torch.gather(value, 1, index).squeeze(1)

    return MonotoneWarpOutput(
        interval_logits=gather(candidates.interval_logits),
        interval_widths=gather(candidates.interval_widths),
        warp=gather(candidates.warp),
        warp_derivative=gather(candidates.warp_derivative),
    )


class MonotoneWarpEstimator(nn.Module):
    """Estimate a strictly monotone endpoint-preserving temporal warp."""

    def __init__(
        self,
        feature_dim: int,
        canonical_grid_size: int,
        hidden_dim: int = 64,
        kernel_size: int = 5,
        min_increment: float = 1e-4,
        num_candidates: int = 3,
        candidate_init_warp_amplitude: float = 0.015,
    ) -> None:
        super().__init__()
        feature_dim = _positive_integer("feature_dim", feature_dim)
        canonical_grid_size = _positive_integer(
            "canonical_grid_size", canonical_grid_size, minimum=2
        )
        hidden_dim = _positive_integer("hidden_dim", hidden_dim)
        kernel_size = _positive_integer("kernel_size", kernel_size)
        num_candidates = _positive_integer("num_candidates", num_candidates)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        min_increment = _finite_float("min_increment", min_increment)
        if min_increment <= 0:
            raise ValueError("min_increment must be greater than zero")
        candidate_init_warp_amplitude = _finite_float(
            "candidate_init_warp_amplitude", candidate_init_warp_amplitude
        )
        if not 0.0 <= candidate_init_warp_amplitude < 0.5:
            raise ValueError(
                "candidate_init_warp_amplitude must lie in [0, 0.5)"
            )

        self.feature_dim = feature_dim
        self.canonical_grid_size = canonical_grid_size
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.min_increment = min_increment
        self.num_candidates = num_candidates
        self.candidate_init_warp_amplitude = candidate_init_warp_amplitude
        self.network = nn.Sequential(
            nn.Conv1d(
                2 * feature_dim + 2,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
            nn.Conv1d(hidden_dim, num_candidates, kernel_size=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.candidate_base_logits = nn.Parameter(
            _initial_candidate_base_logits(
                num_candidates,
                canonical_grid_size,
                candidate_init_warp_amplitude,
                min_increment,
            )
        )

    def _validate_inputs(
        self,
        sample_srvf: Tensor,
        template_srvf: Tensor,
        sample_support: Tensor,
        template_support: Tensor,
        registration_valid: Tensor,
    ) -> None:
        expected_srvf_shape = (
            sample_srvf.shape[0],
            self.canonical_grid_size,
            self.feature_dim,
        ) if isinstance(sample_srvf, Tensor) and sample_srvf.ndim > 0 else None
        if (
            not isinstance(sample_srvf, Tensor)
            or sample_srvf.ndim != 3
            or sample_srvf.shape[1:]
            != (self.canonical_grid_size, self.feature_dim)
        ):
            raise ValueError("sample_srvf must have shape [B, K, D]")
        if not sample_srvf.is_floating_point():
            raise ValueError("sample_srvf must use a floating-point dtype")
        if not torch.isfinite(sample_srvf).all().item():
            raise ValueError("sample_srvf must contain only finite values")
        if (
            not isinstance(template_srvf, Tensor)
            or template_srvf.shape != expected_srvf_shape
        ):
            raise ValueError("template_srvf must have shape [B, K, D]")
        if (
            not template_srvf.is_floating_point()
            or not torch.isfinite(template_srvf).all().item()
        ):
            raise ValueError("template_srvf must contain only finite floating values")
        expected_support_shape = (
            sample_srvf.shape[0], self.canonical_grid_size
        )
        for name, support in (
            ("sample_support", sample_support),
            ("template_support", template_support),
        ):
            if not isinstance(support, Tensor) or support.shape != expected_support_shape:
                raise ValueError(f"{name} must have shape [B, K]")
            if not support.is_floating_point() or not torch.isfinite(support).all().item():
                raise ValueError(f"{name} must contain only finite floating values")
            if torch.any((support < 0) | (support > 1)).item():
                raise ValueError(f"{name} must lie in [0, 1]")
        if (
            not isinstance(registration_valid, Tensor)
            or registration_valid.dtype != torch.bool
        ):
            raise ValueError("registration_valid must be a boolean tensor with shape [B]")
        if registration_valid.shape != (sample_srvf.shape[0],):
            raise ValueError("registration_valid must be a boolean tensor with shape [B]")
        for name, tensor in (
            ("template_srvf", template_srvf),
            ("sample_support", sample_support),
            ("template_support", template_support),
        ):
            if tensor.device != sample_srvf.device or tensor.dtype != sample_srvf.dtype:
                raise ValueError(f"{name} must match sample_srvf device and dtype")

    def forward(
        self,
        sample_srvf: Tensor,
        template_srvf: Tensor,
        sample_support: Tensor,
        template_support: Tensor,
        registration_valid: Tensor,
    ) -> MonotoneWarpOutput:
        candidates = self.forward_candidates(
            sample_srvf,
            template_srvf,
            sample_support,
            template_support,
            registration_valid,
        )
        candidate_index = torch.zeros(
            candidates.warp.shape[0],
            dtype=torch.long,
            device=candidates.warp.device,
        )
        return select_warp_candidate(candidates, candidate_index)

    def forward_candidates(
        self,
        sample_srvf: Tensor,
        template_srvf: Tensor,
        sample_support: Tensor,
        template_support: Tensor,
        registration_valid: Tensor,
    ) -> MonotoneWarpCandidatesOutput:
        device_type = (
            sample_srvf.device.type if isinstance(sample_srvf, Tensor) else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            if isinstance(sample_srvf, Tensor) and sample_srvf.dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                sample_srvf = sample_srvf.float()
            if isinstance(template_srvf, Tensor) and template_srvf.dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                template_srvf = template_srvf.float()
            if isinstance(sample_support, Tensor) and sample_support.dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                sample_support = sample_support.float()
            if isinstance(template_support, Tensor) and template_support.dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                template_support = template_support.float()
            return self._forward_candidates_float32(
                sample_srvf,
                template_srvf,
                sample_support,
                template_support,
                registration_valid,
            )

    def _forward_candidates_float32(
        self,
        sample_srvf: Tensor,
        template_srvf: Tensor,
        sample_support: Tensor,
        template_support: Tensor,
        registration_valid: Tensor,
    ) -> MonotoneWarpCandidatesOutput:
        self._validate_inputs(
            sample_srvf,
            template_srvf,
            sample_support,
            template_support,
            registration_valid,
        )
        batch_size = sample_srvf.shape[0]
        warp_input = torch.cat(
            [
                sample_srvf,
                template_srvf,
                sample_support.unsqueeze(-1),
                template_support.unsqueeze(-1),
            ],
            dim=-1,
        ).transpose(1, 2)
        point_logits = self.network(warp_input)
        interval_logits = 0.5 * (
            point_logits[..., :-1] + point_logits[..., 1:]
        )
        interval_logits = interval_logits + self.candidate_base_logits.to(
            device=interval_logits.device, dtype=interval_logits.dtype
        ).unsqueeze(0)
        positive_increments = F.softplus(interval_logits) + self.min_increment
        interval_widths = positive_increments / positive_increments.sum(
            dim=-1, keepdim=True
        )
        zero = torch.zeros(
            batch_size,
            self.num_candidates,
            1,
            device=sample_srvf.device,
            dtype=sample_srvf.dtype,
        )
        one = torch.ones_like(zero)
        cumulative = interval_widths.cumsum(dim=-1)
        warp = torch.cat([zero, cumulative[..., :-1], one], dim=-1)

        delta_u = 1.0 / (self.canonical_grid_size - 1)
        interval_derivative = interval_widths / delta_u
        middle_derivative = 0.5 * (
            interval_derivative[..., :-1] + interval_derivative[..., 1:]
        )
        warp_derivative = torch.cat(
            [
                interval_derivative[..., :1],
                middle_derivative,
                interval_derivative[..., -1:],
            ],
            dim=-1,
        )

        valid = registration_valid.to(device=sample_srvf.device)
        identity_widths = torch.full_like(
            interval_widths, 1.0 / (self.canonical_grid_size - 1)
        )
        identity_warp = torch.linspace(
            0.0,
            1.0,
            self.canonical_grid_size,
            device=sample_srvf.device,
            dtype=sample_srvf.dtype,
        ).expand(batch_size, self.num_candidates, -1)
        interval_logits = torch.where(
            valid[:, None, None], interval_logits, torch.zeros_like(interval_logits)
        )
        interval_widths = torch.where(
            valid[:, None, None], interval_widths, identity_widths
        )
        warp = torch.where(valid[:, None, None], warp, identity_warp)
        warp_derivative = torch.where(
            valid[:, None, None],
            warp_derivative,
            torch.ones_like(warp_derivative),
        )
        return MonotoneWarpCandidatesOutput(
            interval_logits=interval_logits,
            interval_widths=interval_widths,
            warp=warp,
            warp_derivative=warp_derivative,
            inverse_warp=invert_monotone_warp(warp),
        )


def _warp_sequence(sequence: Tensor, warp: Tensor) -> Tensor:
    device_type = sequence.device.type if isinstance(sequence, Tensor) else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        if isinstance(sequence, Tensor) and sequence.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            sequence = sequence.float()
        if isinstance(warp, Tensor) and warp.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            warp = warp.float()
        return _warp_sequence_float32(sequence, warp)


def _warp_sequence_float32(sequence: Tensor, warp: Tensor) -> Tensor:
    if not isinstance(sequence, Tensor) or sequence.ndim != 3:
        raise ValueError("sequence must have shape [B, K, D]")
    if not sequence.is_floating_point():
        raise ValueError("sequence must use a floating-point dtype")
    if sequence.shape[1] < 2:
        raise ValueError("sequence grid size must be at least 2")
    if not torch.isfinite(sequence).all().item():
        raise ValueError("sequence must contain only finite values")
    if not isinstance(warp, Tensor) or warp.shape != sequence.shape[:2]:
        raise ValueError("warp must have shape [B, K]")
    if not warp.is_floating_point():
        raise ValueError("warp must use a floating-point dtype")
    if warp.device != sequence.device or warp.dtype != sequence.dtype:
        raise ValueError("warp must match sequence device and dtype")
    if not torch.isfinite(warp).all().item():
        raise ValueError("warp must contain only finite values")
    if torch.any((warp < 0) | (warp > 1)).item():
        raise ValueError("warp must lie in [0, 1]")
    if torch.any(warp[:, 1:] <= warp[:, :-1]).item():
        raise ValueError("warp must be strictly increasing")

    input_tensor = sequence.transpose(1, 2).unsqueeze(2)
    grid_x = 2.0 * warp - 1.0
    grid_y = torch.zeros_like(grid_x)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)
    warped = F.grid_sample(
        input_tensor,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.squeeze(2).transpose(1, 2)


def _apply_srvf_group_action(
    srvf: Tensor,
    warp: Tensor,
    warp_derivative: Tensor,
    eps: float,
) -> Tensor:
    device_type = srvf.device.type if isinstance(srvf, Tensor) else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        if isinstance(srvf, Tensor) and srvf.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            srvf = srvf.float()
        if isinstance(warp, Tensor) and warp.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            warp = warp.float()
        if isinstance(warp_derivative, Tensor) and warp_derivative.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            warp_derivative = warp_derivative.float()
        return _apply_srvf_group_action_float32(
            srvf, warp, warp_derivative, eps
        )


def _apply_srvf_group_action_float32(
    srvf: Tensor,
    warp: Tensor,
    warp_derivative: Tensor,
    eps: float,
) -> Tensor:
    eps = _finite_float("eps", eps)
    if eps <= 0:
        raise ValueError("eps must be greater than zero")
    if not isinstance(warp_derivative, Tensor) or warp_derivative.shape != warp.shape:
        raise ValueError("warp_derivative must have shape [B, K]")
    if not warp_derivative.is_floating_point():
        raise ValueError("warp_derivative must use a floating-point dtype")
    if (
        warp_derivative.device != srvf.device
        or warp_derivative.dtype != srvf.dtype
    ):
        raise ValueError("warp_derivative must match srvf device and dtype")
    if not torch.isfinite(warp_derivative).all().item():
        raise ValueError("warp_derivative must contain only finite values")
    if torch.any(warp_derivative <= 0).item():
        raise ValueError("warp_derivative must be strictly positive")
    warped_srvf = _warp_sequence_float32(srvf, warp)
    return warped_srvf * torch.sqrt(
        warp_derivative.clamp_min(eps)
    ).unsqueeze(-1)


@dataclass(frozen=True)
class TemporalRegistrationOutput:
    srvf_output: TemporalSRVFOutput

    template_srvf: Tensor
    template_support: Tensor
    template_initialized: Tensor
    template_mean_support: Tensor

    interval_logits: Tensor
    interval_widths: Tensor
    warp: Tensor
    warp_derivative: Tensor

    registered_srvf: Tensor
    registered_support: Tensor
    registration_valid: Tensor


class TemporalSRVFRegistration(nn.Module):
    """Register temporal SRVFs to an explicitly updated source template."""

    def __init__(
        self,
        feature_dim: int,
        num_basis: int = 12,
        canonical_grid_size: int = 64,
        roughness_grid_size: int = 256,
        smoothing_weight: float = 1e-3,
        time_reference: float = 0.0,
        time_scale: float = 366.0,
        statistics_momentum: float = 0.99,
        min_feature_scale: float = 1e-3,
        support_scale_momentum: float = 0.99,
        initial_support_scale: float = 1.0,
        min_support_scale: float = 1e-6,
        min_mean_support: float = 0.05,
        min_dynamic_energy: float = 1e-4,
        srvf_eps: float = 1e-8,
        derivative_norm_threshold: float = 1e-8,
        template_momentum: float = 0.99,
        min_template_grid_weight: float = 1e-6,
        min_template_mean_support: float = 0.05,
        warp_hidden_dim: int = 64,
        warp_kernel_size: int = 5,
        warp_min_increment: float = 1e-4,
        warp_num_candidates: int = 3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        min_template_mean_support = _finite_float(
            "min_template_mean_support", min_template_mean_support
        )
        eps = _finite_float("eps", eps)
        if not 0.0 <= min_template_mean_support <= 1.0:
            raise ValueError("min_template_mean_support must be in [0, 1]")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")

        self.srvf_extractor = TemporalSRVFExtractor(
            feature_dim=feature_dim,
            num_basis=num_basis,
            canonical_grid_size=canonical_grid_size,
            roughness_grid_size=roughness_grid_size,
            smoothing_weight=smoothing_weight,
            time_reference=time_reference,
            time_scale=time_scale,
            statistics_momentum=statistics_momentum,
            min_feature_scale=min_feature_scale,
            support_scale_momentum=support_scale_momentum,
            initial_support_scale=initial_support_scale,
            min_support_scale=min_support_scale,
            min_mean_support=min_mean_support,
            min_dynamic_energy=min_dynamic_energy,
            srvf_eps=srvf_eps,
            derivative_norm_threshold=derivative_norm_threshold,
            eps=eps,
        )
        self.source_template = SourceRunningSRVFTemplate(
            canonical_grid_size=canonical_grid_size,
            feature_dim=feature_dim,
            momentum=template_momentum,
            min_grid_weight=min_template_grid_weight,
            eps=eps,
        )
        self.warp_estimator = MonotoneWarpEstimator(
            feature_dim=feature_dim,
            canonical_grid_size=canonical_grid_size,
            hidden_dim=warp_hidden_dim,
            kernel_size=warp_kernel_size,
            min_increment=warp_min_increment,
            num_candidates=warp_num_candidates,
        )
        self.min_template_mean_support = min_template_mean_support
        self.eps = eps

    @torch.no_grad()
    def update_source_statistics(
        self,
        component_tokens: Tensor,
        time_mask: Tensor,
    ) -> None:
        self.srvf_extractor.update_source_statistics(component_tokens, time_mask)

    @torch.no_grad()
    def update_source_support_scale(
        self,
        functional_output: TemporalFunctionalOutput,
    ) -> None:
        self.srvf_extractor.update_source_support_scale(functional_output)

    @torch.no_grad()
    def update_source_template(
        self,
        registration_output: TemporalRegistrationOutput,
    ) -> None:
        if not isinstance(registration_output, TemporalRegistrationOutput):
            raise ValueError(
                "registration_output must be a TemporalRegistrationOutput"
            )
        if self.source_template.num_updates.item() == 0:
            self.source_template.update(
                registration_output.srvf_output.srvf,
                registration_output.srvf_output.support_confidence,
                registration_output.srvf_output.structure_valid,
            )
        else:
            self.source_template.update(
                registration_output.registered_srvf,
                registration_output.registered_support,
                registration_output.registration_valid,
            )

    def forward(
        self,
        component_tokens: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> TemporalRegistrationOutput:
        srvf_output = self.srvf_extractor(
            component_tokens,
            positions,
            time_mask,
        )
        batch_size = component_tokens.shape[0]
        template_output = self.source_template(
            device=component_tokens.device,
            dtype=component_tokens.dtype,
        )
        template_srvf = template_output.srvf.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        template_support = template_output.support.unsqueeze(0).expand(
            batch_size, -1
        )
        integration_weights = self.srvf_extractor.integration_weights.to(
            device=component_tokens.device,
            dtype=component_tokens.dtype,
        )
        template_mean_support = (
            template_output.support * integration_weights
        ).sum()
        registration_valid = (
            srvf_output.structure_valid
            & template_output.initialized
            & (template_mean_support >= self.min_template_mean_support)
        )
        warp_output = self.warp_estimator(
            srvf_output.srvf.detach(),
            template_srvf.detach(),
            srvf_output.support_confidence.detach(),
            template_support.detach(),
            registration_valid,
        )
        registered_srvf = _apply_srvf_group_action(
            srvf_output.srvf,
            warp_output.warp,
            warp_output.warp_derivative,
            self.eps,
        )
        registered_support = _warp_sequence(
            srvf_output.support_confidence.unsqueeze(-1),
            warp_output.warp,
        ).squeeze(-1)
        return TemporalRegistrationOutput(
            srvf_output=srvf_output,
            template_srvf=template_srvf,
            template_support=template_support,
            template_initialized=template_output.initialized,
            template_mean_support=template_mean_support,
            interval_logits=warp_output.interval_logits,
            interval_widths=warp_output.interval_widths,
            warp=warp_output.warp,
            warp_derivative=warp_output.warp_derivative,
            registered_srvf=registered_srvf,
            registered_support=registered_support,
            registration_valid=registration_valid,
        )
