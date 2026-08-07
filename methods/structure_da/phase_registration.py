"""Curve-DP T registration adapter and gamma utility helpers.

Round 3.5 delegates the class-conditioned T registration of a vector-valued
T-SRVF to ``fdasrsf.curve_functions.optimum_reparam_curve``. That API takes a
single ``n``-dimensional curve SRVF as ``[n, T]`` (for this project ``[D,
K_reg]``) and returns one shared gamma ``[K_reg]`` via ``method="DP"``. The
solver aligns q2 (target) to q1 (source):

    gamma(u_source) = u_target

so applying ``warp_q_gamma(target_q, gamma)`` gives the target SRVF expressed
in source coordinates. This module owns the fdasrsf call, the legality
diagnostics of the returned gamma, and the small pure helpers needed to
resample gamma and warp a scalar support (which is *not* an SRVF and must not
use the SRVF group action).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .registration_geometry import (
    SourceRegistrationPrototypeBank,
    evaluate_registration_geometry,
)
from .temporal_registration import _warp_sequence
from .temporal_srvf import TemporalSRVFExtractor


class FdasrsfCurveRegistrationAdapter(nn.Module):
    """Wrap ``fdasrsf.curve_functions.optimum_reparam_curve`` with the DP solver.

    The adapter is deliberately narrow: it computes exactly one shared gamma
    per call for a vector-valued curve, with the target SRVF as q2. It never
    computes Shape distances, empirical CDFs, class selection or phase groups.
    """

    def __init__(
        self,
        registration_lambda: float,
    ) -> None:
        super().__init__()
        self.registration_lambda = float(registration_lambda)
        try:
            from fdasrsf import curve_functions as cf
        except ImportError as error:  # pragma: no cover - import guard
            raise RuntimeError(
                "fdasrsf is required for curve registration; install it first"
            ) from error
        self._cf = cf

    def register(
        self,
        source_trend_srvf: Tensor,
        target_trend_srvf: Tensor,
    ) -> Tensor:
        """Return the curve-DP gamma aligning the target T-SRVF to the source one.

        Args:
            source_trend_srvf: Class source T prototype, ``[K_reg, D]``.
            target_trend_srvf: Target sample T-SRVF, ``[K_reg, D]``.

        Returns:
            gamma ``[K_reg]`` (requires_grad=False). The direction is
            ``gamma(u_source) = u_target``.
        """
        if source_trend_srvf.ndim != 2 or target_trend_srvf.ndim != 2:
            raise ValueError("SRVFs must have shape [K_reg, D]")
        if source_trend_srvf.shape != target_trend_srvf.shape:
            raise ValueError("source and target SRVFs must share shape")

        source_np = (
            source_trend_srvf.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .numpy()
            .T
        )  # [D, K_reg]
        target_np = (
            target_trend_srvf.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .numpy()
            .T
        )  # [D, K_reg]
        gamma_np = self._cf.optimum_reparam_curve(
            q1=source_np,
            q2=target_np,
            lam=self.registration_lambda,
            method="DP",
        )
        gamma = torch.as_tensor(gamma_np, dtype=torch.float64, device="cpu")
        if gamma.ndim != 1:
            raise ValueError(
                "curve DP must return a single one-dimensional gamma"
            )
        return gamma.contiguous()


def warp_q_gamma(q: Tensor, gamma: Tensor) -> Tensor:
    """Apply the SRVF group action of ``gamma`` to a curve ``[B, K, D]`` or ``[K, D]``.

    Uses the retained pure SRVF warping (grid-sample reparametrisation) and
    the analytic derivative of the piecewise-linear gamma.
    """
    if q.ndim == 2:
        q = q.unsqueeze(0)
    batch, grid, _dim = q.shape
    if gamma.ndim == 1:
        gamma = gamma.unsqueeze(0).expand(batch, -1)
    gamma = gamma.to(dtype=q.dtype, device=q.device)
    if gamma.shape != (batch, grid):
        raise ValueError("gamma must have shape [K] or [B, K]")
    delta = gamma[:, 1:] - gamma[:, :-1]
    local_speed = delta / (1.0 / (grid - 1))
    derivative = torch.cat(
        [
            local_speed[:, :1],
            0.5 * (local_speed[:, :-1] + local_speed[:, 1:]),
            local_speed[:, -1:],
        ],
        dim=-1,
    )
    warped = _warp_sequence(q, gamma)
    return warped * torch.sqrt(derivative.clamp_min(1e-8)).unsqueeze(-1)


def warp_support_gamma(support: Tensor, gamma: Tensor, grid: Tensor) -> Tensor:
    """Warp a scalar support by a gamma: ``warped(u) = support(gamma(u))``.

    Support is an ordinary scalar function of time, so it is resampled but is
    NOT multiplied by ``sqrt(gamma')``.
    """
    if support.ndim != 1:
        raise ValueError("support must have shape [K]")
    if gamma.ndim != 1 or gamma.shape != support.shape:
        raise ValueError("gamma must have shape [K] matching support")
    if grid.ndim != 1 or grid.shape != support.shape:
        raise ValueError("grid must have shape [K] matching support")
    gamma = gamma.to(dtype=support.dtype, device=support.device)
    # Monotone linear resampling of support onto gamma(u).
    support_2d = support.unsqueeze(0).unsqueeze(-1)  # [1, K, 1]
    gamma_2d = gamma.unsqueeze(0)                    # [1, K]
    warped = _warp_sequence(support_2d, gamma_2d)
    return warped.squeeze(0).squeeze(-1)


def resample_gamma(gamma_reg: Tensor, reg_grid: Tensor, target_grid: Tensor) -> Tensor:
    """Monotonically resample a gamma function onto a different grid."""
    if gamma_reg.ndim != 1 or gamma_reg.shape != reg_grid.shape:
        raise ValueError("gamma_reg must have shape [K_reg]")
    if target_grid.ndim != 1:
        raise ValueError("target_grid must be one-dimensional")
    # invert_monotone_warp evaluates a piecewise-linear warp at arbitrary
    # queries. gamma_reg maps reg_grid -> reg_grid; evaluating at target_grid
    # points gives the resampled gamma.
    from .temporal_registration import invert_monotone_warp

    gamma_reg = gamma_reg.to(device=target_grid.device, dtype=target_grid.dtype)
    return invert_monotone_warp(gamma_reg, query=target_grid)


@dataclass(frozen=True)
class GammaLegalityOutput:
    finite: bool
    endpoint_error: float
    strictly_increasing: bool
    min_increment: float
    max_local_speed: float
    roughness: float
    phase_deviation: float
    legal: bool


def check_gamma_legality(
    gamma: Tensor,
    grid: Tensor,
    *,
    endpoint_tolerance: float = 1e-6,
    registration_min_increment: float = 1e-6,
    registration_max_local_speed: float = 50.0,
    registration_max_roughness: float = 1e4,
    registration_max_deviation: float = 2.0,
) -> GammaLegalityOutput:
    """Validate a curve-DP gamma and compute its diagnostic statistics.

    Roughness uses the stable log-speed finite-difference definition applied
    as an external gate after the solver (curve DP does not expose a roughness
    penalty); the phase deviation is the L2 norm
    ``sqrt(mean((gamma-u)^2))``.
    """
    if not isinstance(gamma, Tensor) or not torch.isfinite(gamma).all().item():
        return GammaLegalityOutput(
            finite=False, endpoint_error=float("nan"), strictly_increasing=False,
            min_increment=float("nan"), max_local_speed=float("nan"),
            roughness=float("nan"), phase_deviation=float("nan"), legal=False,
        )
    gamma = gamma.to(torch.float32)
    grid = grid.to(torch.float32)
    endpoint_error = max(
        float(abs(gamma[0] - grid[0]).item()),
        float(abs(gamma[-1] - grid[-1]).item()),
    )
    delta = gamma[1:] - gamma[:-1]
    increments = delta
    min_increment = float(increments.min().item())
    strictly_increasing = bool(increments.min().item() > 0)
    step = grid[1] - grid[0]
    local_speed = increments / step
    max_local_speed = float(local_speed.max().item())
    log_speed = torch.log(local_speed + 1e-8)
    d_log = log_speed[1:] - log_speed[:-1]
    roughness = float(((d_log / step) ** 2).mean().item())
    phase_deviation = float(torch.sqrt(((gamma - grid) ** 2).mean()).item())

    legal = (
        endpoint_error <= endpoint_tolerance
        and strictly_increasing
        and min_increment >= registration_min_increment
        and max_local_speed <= registration_max_local_speed
        and roughness <= registration_max_roughness
        and phase_deviation <= registration_max_deviation
    )
    return GammaLegalityOutput(
        finite=True,
        endpoint_error=endpoint_error,
        strictly_increasing=strictly_increasing,
        min_increment=min_increment,
        max_local_speed=max_local_speed,
        roughness=roughness,
        phase_deviation=phase_deviation,
        legal=legal,
    )


def build_source_registration_prototypes(
    model: nn.Module,
    source_scan_loader,
    num_classes: int,
    *,
    device: torch.device,
    reg_extractor: TemporalSRVFExtractor | None = None,
    eps: float = 1e-8,
    min_mean_support: float = 0.0,
) -> SourceRegistrationPrototypeBank:
    """Build per-class source T-SRVF prototypes on the K_reg grid.

    Runs one deterministic full-source scan. q_T at K_reg is obtained by
    re-evaluating the trend functional fit on the registration grid. Labels
    are the true source labels.
    """
    from .temporal_srvf import TemporalSRVFExtractor

    if reg_extractor is None:
        feature_dim = model.backbone.feature_dim
        structure_geometry = model.temporal_module.structure_geometry
        functional = structure_geometry.functional_lift
        reg_extractor = TemporalSRVFExtractor(
            feature_dim=feature_dim,
            num_basis=functional.num_basis,
            canonical_grid_size=128,
            roughness_grid_size=functional.roughness_grid_size,
            smoothing_weight=structure_geometry.functional_lift.smoothing_weight,
            time_reference=0.0,
            time_scale=1.0,
            min_mean_support=0.0,
            min_dynamic_energy=0.0,
        )
    reg_extractor = reg_extractor.to(device=device)

    was_training = model.training
    grid: Tensor | None = None
    srvf_sum: list[Tensor | None] = [None] * num_classes
    support_sum: list[Tensor | None] = [None] * num_classes
    counts = [0] * num_classes
    try:
        model.eval()
        with torch.inference_mode():
            for batch in source_scan_loader:
                output = model(
                    batch["pixels"],
                    batch["valid_pixels"],
                    batch["positions"],
                    batch.get("extra"),
                    return_geometry=True,
                )
                labels = batch["label"].to(device=device, dtype=torch.long)
                trend = output.trend
                mask = output.mask
                positions = output.positions
                reg = evaluate_registration_geometry(
                    trend, positions, mask, reg_extractor
                )
                if grid is None:
                    grid = reg.registration_grid
                for class_id in range(num_classes):
                    sample_mask = (labels == class_id) & reg.trend_valid
                    if not torch.any(sample_mask).item():
                        continue
                    srvf = reg.trend_srvf[sample_mask]
                    support = reg.trend_support[sample_mask]
                    count = int(sample_mask.sum().item())
                    w = support.unsqueeze(-1)
                    batch_sum = (srvf * w).sum(dim=0)
                    batch_support_sum = support.sum(dim=0)
                    if srvf_sum[class_id] is None:
                        srvf_sum[class_id] = batch_sum
                        support_sum[class_id] = batch_support_sum
                        counts[class_id] = count
                    else:
                        srvf_sum[class_id] = srvf_sum[class_id] + batch_sum
                        support_sum[class_id] = support_sum[class_id] + batch_support_sum
                        counts[class_id] += count
        if grid is None:
            raise RuntimeError("source scan loader produced no batches")

        dtype = next(model.parameters()).dtype
        trend_out: list[Tensor] = []
        support_out: list[Tensor] = []
        ready: list[bool] = []
        for class_id in range(num_classes):
            if srvf_sum[class_id] is None:
                trend_out.append(torch.zeros(grid.shape[0], dtype=dtype, device=device))
                support_out.append(torch.zeros(grid.shape[0], dtype=dtype, device=device))
                ready.append(False)
                continue
            p = (srvf_sum[class_id] / (support_sum[class_id].unsqueeze(-1) + eps)).to(dtype=dtype)
            sup = (support_sum[class_id] / counts[class_id]).to(dtype=dtype)
            trend_out.append(p)
            support_out.append(sup)
            ready.append(True)
        return SourceRegistrationPrototypeBank(
            trend_srvf=torch.stack(trend_out),
            trend_support=torch.stack(support_out),
            class_counts=torch.tensor(counts, device=device, dtype=torch.long),
            ready=torch.tensor(ready, device=device, dtype=torch.bool),
            registration_grid=grid.to(device=device, dtype=dtype),
        )
    finally:
        model.train(was_training)
