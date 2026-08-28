import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ltae import LTAE


def _inverse_softplus(value):
    """Numerically stable inverse of softplus for a positive scalar."""
    return value + math.log(-math.expm1(-value))


def _normalized_pairwise_time_distance(h, positions, time_scale_days):
    normalized_positions = positions.to(device=h.device, dtype=h.dtype)
    normalized_positions = normalized_positions / time_scale_days.to(
        device=h.device, dtype=h.dtype
    )
    return torch.abs(
        normalized_positions.unsqueeze(-1) - normalized_positions.unsqueeze(-2)
    )


def _as_scalar_tensor(value, reference):
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _laplace_temporal_weights(
    h,
    pairwise_distance,
    tau_days,
    time_scale_days,
    time_mask,
    eps,
    remove_diagonal=False,
):
    tau = _as_scalar_tensor(tau_days, h) / _as_scalar_tensor(time_scale_days, h)
    kernel = torch.exp(-pairwise_distance / tau)
    if time_mask is not None:
        key_mask = time_mask.to(device=h.device, dtype=h.dtype).unsqueeze(1)
        kernel = kernel * key_mask
    if remove_diagonal:
        off_diagonal = 1.0 - torch.eye(
            kernel.shape[-1], device=h.device, dtype=h.dtype
        )
        kernel = kernel * off_diagonal.unsqueeze(0)
    return kernel / (kernel.sum(dim=-1, keepdim=True) + eps)


def _laplace_temporal_smooth(
    h, pairwise_distance, tau_days, time_scale_days, time_mask, eps
):
    weights = _laplace_temporal_weights(
        h,
        pairwise_distance,
        tau_days,
        time_scale_days,
        time_mask,
        eps,
    )
    smoothed = torch.bmm(weights, h)
    if time_mask is not None:
        query_mask = time_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
        smoothed = smoothed * query_mask
    return smoothed


class MultiScaleTemporalKernelDecomposition(nn.Module):
    """Smooth temporal features with ordered fast and slow exponential kernels."""

    def __init__(
        self,
        time_scale_days=365.0,
        tau_fast_init_days=30.0,
        tau_slow_init_days=90.0,
        tau_min_days=1.0,
        delta_tau_min_days=1.0,
        learnable_tau=True,
        eps=1e-8,
    ):
        super().__init__()
        if time_scale_days <= 0:
            raise ValueError("time_scale_days must be positive")
        if tau_min_days <= 0:
            raise ValueError("tau_min_days must be positive")
        if delta_tau_min_days <= 0:
            raise ValueError("delta_tau_min_days must be positive")
        if tau_fast_init_days <= tau_min_days:
            raise ValueError("tau_fast_init_days must be greater than tau_min_days")
        if tau_slow_init_days <= tau_fast_init_days + delta_tau_min_days:
            raise ValueError(
                "tau_slow_init_days must be greater than "
                "tau_fast_init_days + delta_tau_min_days"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.register_buffer("time_scale_days", torch.tensor(float(time_scale_days)))
        self.register_buffer("tau_min_days", torch.tensor(float(tau_min_days)))
        self.register_buffer(
            "delta_tau_min_days", torch.tensor(float(delta_tau_min_days))
        )
        self.eps = float(eps)

        a_init = _inverse_softplus(tau_fast_init_days - tau_min_days)
        b_init = _inverse_softplus(
            tau_slow_init_days - tau_fast_init_days - delta_tau_min_days
        )
        a = torch.tensor(a_init)
        b = torch.tensor(b_init)
        if learnable_tau:
            self.a = nn.Parameter(a)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("a", a)
            self.register_buffer("b", b)

    def get_tau_days(self):
        """Return differentiable scalar tensors for the current ordered scales."""
        tau_fast_days = self.tau_min_days + F.softplus(self.a)
        tau_slow_days = (
            tau_fast_days + self.delta_tau_min_days + F.softplus(self.b)
        )
        return tau_fast_days, tau_slow_days

    def _smooth(self, h, pairwise_distance, tau_days, time_mask):
        return _laplace_temporal_smooth(
            h,
            pairwise_distance,
            tau_days,
            self.time_scale_days,
            time_mask,
            self.eps,
        )

    def forward(self, h, positions, time_mask=None):
        if h.ndim != 3:
            raise ValueError("h must have shape [B, L, C]")
        if positions.shape != h.shape[:2]:
            raise ValueError("positions must have shape [B, L]")
        if time_mask is not None and time_mask.shape != h.shape[:2]:
            raise ValueError("time_mask must have shape [B, L]")

        pairwise_distance = _normalized_pairwise_time_distance(
            h, positions, self.time_scale_days
        )
        tau_fast_days, tau_slow_days = self.get_tau_days()
        s = self._smooth(h, pairwise_distance, tau_fast_days, time_mask)
        t = self._smooth(h, pairwise_distance, tau_slow_days, time_mask)
        return t, s

    def debug_components(self, h, positions, time_mask=None):
        """Return the full algebraic decomposition for diagnostics only."""
        t, s = self(h, positions, time_mask=time_mask)
        d = s - t
        r = h - s
        return {"T": t, "S": s, "D": d, "R": r}


class SingleScaleTemporalKernel(nn.Module):
    """Fixed-bandwidth Laplace temporal smoothing using the MTKD kernel math."""

    def __init__(self, tau_days, time_scale_days=365.0, eps=1e-8):
        super().__init__()
        if tau_days <= 0:
            raise ValueError("tau_days must be positive")
        if time_scale_days <= 0:
            raise ValueError("time_scale_days must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.register_buffer("tau_days", torch.tensor(float(tau_days)))
        self.register_buffer("time_scale_days", torch.tensor(float(time_scale_days)))
        self.eps = float(eps)

    def forward(self, h, positions, time_mask=None):
        if h.ndim != 3:
            raise ValueError("h must have shape [B, L, C]")
        if positions.shape != h.shape[:2]:
            raise ValueError("positions must have shape [B, L]")
        if time_mask is not None and time_mask.shape != h.shape[:2]:
            raise ValueError("time_mask must have shape [B, L]")

        pairwise_distance = _normalized_pairwise_time_distance(
            h, positions, self.time_scale_days
        )
        return _laplace_temporal_smooth(
            h,
            pairwise_distance,
            self.tau_days,
            self.time_scale_days,
            time_mask,
            self.eps,
        )


class LearnableSingleScaleTemporalKernel(nn.Module):
    """One positive learnable bandwidth with the shared Laplace kernel."""

    def __init__(
        self,
        tau_init_days=75.0,
        tau_min_days=1.0,
        time_scale_days=365.0,
        eps=1e-8,
    ):
        super().__init__()
        if tau_min_days <= 0:
            raise ValueError("tau_min_days must be positive")
        if tau_init_days <= tau_min_days:
            raise ValueError("tau_init_days must be greater than tau_min_days")
        if time_scale_days <= 0:
            raise ValueError("time_scale_days must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.register_buffer("tau_min_days", torch.tensor(float(tau_min_days)))
        self.register_buffer("time_scale_days", torch.tensor(float(time_scale_days)))
        self.a = nn.Parameter(
            torch.tensor(_inverse_softplus(tau_init_days - tau_min_days))
        )
        self.eps = float(eps)

    def get_tau_days(self):
        return self.tau_min_days + F.softplus(self.a)

    def forward(self, h, positions, time_mask=None, detach_tau=False):
        if h.ndim != 3:
            raise ValueError("h must have shape [B, L, C]")
        if positions.shape != h.shape[:2]:
            raise ValueError("positions must have shape [B, L]")
        if time_mask is not None and time_mask.shape != h.shape[:2]:
            raise ValueError("time_mask must have shape [B, L]")
        tau_days = self.get_tau_days()
        if detach_tau:
            tau_days = tau_days.detach()
        pairwise_distance = _normalized_pairwise_time_distance(
            h, positions, self.time_scale_days
        )
        return _laplace_temporal_smooth(
            h,
            pairwise_distance,
            tau_days,
            self.time_scale_days,
            time_mask,
            self.eps,
        )


def leave_one_out_smooth_l1(
    h,
    positions,
    tau_days,
    time_scale_days=365.0,
    time_mask=None,
    eps=1e-8,
    beta=1.0,
):
    """Vectorized leave-one-out Laplace prediction loss and valid query count."""
    if h.ndim != 3:
        raise ValueError("h must have shape [B, L, C]")
    if positions.shape != h.shape[:2]:
        raise ValueError("positions must have shape [B, L]")
    if time_mask is None:
        time_mask = torch.ones(h.shape[:2], device=h.device, dtype=torch.bool)
    elif time_mask.shape != h.shape[:2]:
        raise ValueError("time_mask must have shape [B, L]")
    else:
        time_mask = time_mask.to(device=h.device, dtype=torch.bool)

    pairwise_distance = _normalized_pairwise_time_distance(
        h, positions, _as_scalar_tensor(time_scale_days, h)
    )
    weights = _laplace_temporal_weights(
        h,
        pairwise_distance,
        tau_days,
        time_scale_days,
        time_mask,
        eps,
        remove_diagonal=True,
    )
    prediction = torch.bmm(weights, h)
    sample_is_valid = time_mask.sum(dim=1) >= 2
    valid_queries = time_mask & sample_is_valid.unsqueeze(1)
    valid_query_count = int(valid_queries.sum().item())
    if valid_query_count == 0:
        tau_tensor = _as_scalar_tensor(tau_days, h)
        return h.sum() * 0.0 + tau_tensor.sum() * 0.0, 0

    element_loss = F.smooth_l1_loss(prediction, h, beta=beta, reduction="none")
    loss = element_loss[valid_queries].mean()
    return loss, valid_query_count


def compute_feature_curve_diagnostics(
    h, positions, time_mask=None, labels=None, eps=1e-8
):
    """Detached temporal smoothness, scale, and optional source separation metrics."""
    with torch.no_grad():
        h = h.detach()
        positions = positions.detach().to(device=h.device, dtype=h.dtype)
        if time_mask is None:
            time_mask = torch.ones(h.shape[:2], device=h.device, dtype=torch.bool)
        else:
            time_mask = time_mask.detach().to(device=h.device, dtype=torch.bool)
        mask_f = time_mask.to(dtype=h.dtype)
        counts = mask_f.sum(dim=1).clamp_min(1.0)
        temporal_mean = (h * mask_f.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
        centered = h - temporal_mean.unsqueeze(1)
        temporal_var = (
            centered.square() * mask_f.unsqueeze(-1)
        ).sum(dim=1) / counts.unsqueeze(-1)
        valid_samples = time_mask.any(dim=1)
        valid_means = temporal_mean[valid_samples]
        if valid_means.shape[0] > 0:
            cross_sample_var = valid_means.var(dim=0, unbiased=False).mean()
        else:
            cross_sample_var = h.sum() * 0.0

        pair_mask = time_mask[:, 1:] & time_mask[:, :-1]
        adjacent = h[:, 1:] - h[:, :-1]
        adjacent_l2 = torch.linalg.vector_norm(adjacent, dim=-1) / math.sqrt(h.shape[-1])
        delta_days = torch.abs(positions[:, 1:] - positions[:, :-1])
        if pair_mask.any():
            h_adjacent_l2 = adjacent_l2[pair_mask].mean()
            h_adjacent_slope = (
                adjacent_l2 / (delta_days + eps)
            )[pair_mask].mean()
        else:
            h_adjacent_l2 = h.sum() * 0.0
            h_adjacent_slope = h.sum() * 0.0
        latent_rms = torch.linalg.vector_norm(h, dim=-1) / math.sqrt(h.shape[-1])
        if time_mask.any():
            h_latent_rms = latent_rms[time_mask].mean()
        else:
            h_latent_rms = h.sum() * 0.0

        diagnostics = {
            "h_temporal_var": temporal_var[valid_samples].mean()
            if valid_samples.any()
            else h.sum() * 0.0,
            "h_cross_sample_var": cross_sample_var,
            "h_adjacent_l2": h_adjacent_l2,
            "h_adjacent_slope_per_day": h_adjacent_slope,
            "h_latent_rms": h_latent_rms,
        }
        if labels is not None:
            labels = labels.detach().to(device=h.device)[valid_samples]
            z = valid_means
            within = h.new_zeros(())
            between = h.new_zeros(())
            if z.shape[0] > 0:
                overall_mean = z.mean(dim=0)
                for class_id in torch.unique(labels):
                    class_z = z[labels == class_id]
                    class_mean = class_z.mean(dim=0)
                    within = within + (class_z - class_mean).square().sum()
                    between = between + class_z.shape[0] * (
                        class_mean - overall_mean
                    ).square().sum()
                within = within / z.shape[0]
                between = between / z.shape[0]
            diagnostics.update(
                {
                    "h_source_within_class_scatter": within,
                    "h_source_between_class_scatter": between,
                    "h_source_fisher": between / (within + eps),
                }
            )
        return {name: value.detach() for name, value in diagnostics.items()}


class MTKDEarlyConcatLTAE(nn.Module):
    """MTKD T/S early concatenation followed by one LTAE."""

    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=8,
        d_model=256,
        n_neurons=(256, 128),
        dropout=0.2,
        T=1000,
        max_temporal_shift=100,
        time_scale_days=365.0,
        tau_fast_init_days=30.0,
        tau_slow_init_days=90.0,
        tau_min_days=1.0,
        delta_tau_min_days=1.0,
        learnable_tau=True,
    ):
        super().__init__()
        self.mtkd = MultiScaleTemporalKernelDecomposition(
            time_scale_days=time_scale_days,
            tau_fast_init_days=tau_fast_init_days,
            tau_slow_init_days=tau_slow_init_days,
            tau_min_days=tau_min_days,
            delta_tau_min_days=delta_tau_min_days,
            learnable_tau=learnable_tau,
        )
        self.ltae = LTAE(
            in_channels=2 * in_channels,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=list(n_neurons),
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
        )
        self.reset_diagnostics()

    @property
    def max_temporal_shift(self):
        return self.ltae.max_temporal_shift

    @property
    def positional_enc(self):
        return self.ltae.positional_enc

    def reset_diagnostics(self):
        """Clear non-persistent, detached per-epoch component statistics."""
        self._diagnostic_totals = {}
        self._diagnostic_sample_count = 0

    def record_diagnostics(self, t, s):
        """Accumulate sample-weighted component diagnostics without gradients."""
        with torch.no_grad():
            flat_t = t.detach().reshape(t.shape[0], -1)
            flat_s = s.detach().reshape(s.shape[0], -1)
            t_norm = torch.linalg.vector_norm(flat_t, dim=1)
            s_norm = torch.linalg.vector_norm(flat_s, dim=1)
            difference_norm = torch.linalg.vector_norm(flat_s - flat_t, dim=1)
            values = {
                "ts_relative_difference": difference_norm / (s_norm + self.mtkd.eps),
                "ts_cosine_similarity": F.cosine_similarity(
                    flat_t, flat_s, dim=1, eps=self.mtkd.eps
                ),
                "t_to_s_norm_ratio": t_norm / (s_norm + self.mtkd.eps),
            }
            for name, value in values.items():
                batch_total = value.sum().detach()
                if name in self._diagnostic_totals:
                    self._diagnostic_totals[name].add_(batch_total)
                else:
                    self._diagnostic_totals[name] = batch_total
            self._diagnostic_sample_count += t.shape[0]

    def get_diagnostics(self):
        """Read scalar tau and sample-averaged component diagnostics."""
        with torch.no_grad():
            tau_fast, tau_slow = self.mtkd.get_tau_days()
            fast_days = tau_fast.detach().cpu().item()
            slow_days = tau_slow.detach().cpu().item()
            diagnostics = {
                "tau_fast_days": fast_days,
                "tau_slow_days": slow_days,
                "delta_tau_days": slow_days - fast_days,
            }
            for name in (
                "ts_relative_difference",
                "ts_cosine_similarity",
                "t_to_s_norm_ratio",
            ):
                if self._diagnostic_sample_count:
                    diagnostics[name] = (
                        self._diagnostic_totals[name]
                        / self._diagnostic_sample_count
                    ).detach().cpu().item()
                else:
                    diagnostics[name] = float("nan")
            return diagnostics

    def forward(self, spatial_feats, positions):
        t, s = self.mtkd(spatial_feats, positions)
        if self.training:
            self.record_diagnostics(t, s)
        early_concat = torch.cat([t, s], dim=-1)
        return self.ltae(early_concat, positions)


class MTKDSOnlyLTAE(MTKDEarlyConcatLTAE):
    """MTKD followed by one LTAE over the smoothed structure S only."""

    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=8,
        d_model=256,
        n_neurons=(256, 128),
        dropout=0.2,
        T=1000,
        max_temporal_shift=100,
        time_scale_days=365.0,
        tau_fast_init_days=30.0,
        tau_slow_init_days=90.0,
        tau_min_days=1.0,
        delta_tau_min_days=1.0,
        learnable_tau=True,
    ):
        nn.Module.__init__(self)
        self.mtkd = MultiScaleTemporalKernelDecomposition(
            time_scale_days=time_scale_days,
            tau_fast_init_days=tau_fast_init_days,
            tau_slow_init_days=tau_slow_init_days,
            tau_min_days=tau_min_days,
            delta_tau_min_days=delta_tau_min_days,
            learnable_tau=learnable_tau,
        )
        self.ltae = LTAE(
            in_channels=in_channels,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=list(n_neurons),
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
        )
        self.reset_diagnostics()

    def forward(self, spatial_feats, positions):
        t, s = self.mtkd(spatial_feats, positions)
        if self.training:
            self.record_diagnostics(t, s)
        return self.ltae(s, positions)


class MTKDMidConcatLTAE(nn.Module):
    """MTKD T/S branches encoded independently, then concatenated."""

    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=8,
        d_model=256,
        n_neurons=(256, 128),
        dropout=0.2,
        T=1000,
        max_temporal_shift=100,
        time_scale_days=365.0,
        tau_fast_init_days=30.0,
        tau_slow_init_days=90.0,
        tau_min_days=1.0,
        delta_tau_min_days=1.0,
        learnable_tau=True,
    ):
        super().__init__()
        self.mtkd = MultiScaleTemporalKernelDecomposition(
            time_scale_days=time_scale_days,
            tau_fast_init_days=tau_fast_init_days,
            tau_slow_init_days=tau_slow_init_days,
            tau_min_days=tau_min_days,
            delta_tau_min_days=delta_tau_min_days,
            learnable_tau=learnable_tau,
        )
        ltae_kwargs = dict(
            in_channels=in_channels,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=list(n_neurons),
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
        )
        self.ltae_t = LTAE(**ltae_kwargs)
        self.ltae_s = LTAE(**ltae_kwargs)
        self.reset_diagnostics()

    @property
    def max_temporal_shift(self):
        return self.ltae_t.max_temporal_shift

    @property
    def positional_enc(self):
        return self.ltae_t.positional_enc

    def reset_diagnostics(self):
        """Clear non-persistent, detached per-epoch component statistics."""
        self._diagnostic_totals = {}
        self._diagnostic_sample_count = 0

    def record_diagnostics(self, t, s):
        """Accumulate sample-weighted component diagnostics without gradients."""
        with torch.no_grad():
            flat_t = t.detach().reshape(t.shape[0], -1)
            flat_s = s.detach().reshape(s.shape[0], -1)
            t_norm = torch.linalg.vector_norm(flat_t, dim=1)
            s_norm = torch.linalg.vector_norm(flat_s, dim=1)
            difference_norm = torch.linalg.vector_norm(flat_s - flat_t, dim=1)
            values = {
                "ts_relative_difference": difference_norm / (s_norm + self.mtkd.eps),
                "ts_cosine_similarity": F.cosine_similarity(
                    flat_t, flat_s, dim=1, eps=self.mtkd.eps
                ),
                "t_to_s_norm_ratio": t_norm / (s_norm + self.mtkd.eps),
            }
            for name, value in values.items():
                batch_total = value.sum().detach()
                if name in self._diagnostic_totals:
                    self._diagnostic_totals[name].add_(batch_total)
                else:
                    self._diagnostic_totals[name] = batch_total
            self._diagnostic_sample_count += t.shape[0]

    def get_diagnostics(self):
        """Read scalar tau and sample-averaged component diagnostics."""
        with torch.no_grad():
            tau_fast, tau_slow = self.mtkd.get_tau_days()
            fast_days = tau_fast.detach().cpu().item()
            slow_days = tau_slow.detach().cpu().item()
            diagnostics = {
                "tau_fast_days": fast_days,
                "tau_slow_days": slow_days,
                "delta_tau_days": slow_days - fast_days,
            }
            for name in (
                "ts_relative_difference",
                "ts_cosine_similarity",
                "t_to_s_norm_ratio",
            ):
                if self._diagnostic_sample_count:
                    diagnostics[name] = (
                        self._diagnostic_totals[name] / self._diagnostic_sample_count
                    ).detach().cpu().item()
                else:
                    diagnostics[name] = float("nan")
            return diagnostics

    def forward(self, spatial_feats, positions):
        t, s = self.mtkd(spatial_feats, positions)
        if self.training:
            self.record_diagnostics(t, s)
        z_t = self.ltae_t(t, positions)
        z_s = self.ltae_s(s, positions)
        return torch.cat([z_t, z_s], dim=-1)


class MTKDTDMidConcatLTAE(MTKDMidConcatLTAE):
    """MTKD T/D branches encoded independently, then concatenated."""

    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=8,
        d_model=256,
        n_neurons=(256, 128),
        dropout=0.2,
        T=1000,
        max_temporal_shift=100,
        time_scale_days=365.0,
        tau_fast_init_days=30.0,
        tau_slow_init_days=90.0,
        tau_min_days=1.0,
        delta_tau_min_days=1.0,
        learnable_tau=True,
    ):
        super().__init__(
            in_channels=in_channels,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=n_neurons,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
            time_scale_days=time_scale_days,
            tau_fast_init_days=tau_fast_init_days,
            tau_slow_init_days=tau_slow_init_days,
            tau_min_days=tau_min_days,
            delta_tau_min_days=delta_tau_min_days,
            learnable_tau=learnable_tau,
        )
        self.ltae_d = self.ltae_s
        del self.ltae_s

    def record_diagnostics(self, t, s):
        """Accumulate existing T/S metrics and detached T/D metrics."""
        super().record_diagnostics(t, s)
        with torch.no_grad():
            flat_t = t.detach().reshape(t.shape[0], -1)
            flat_s = s.detach().reshape(s.shape[0], -1)
            flat_d = flat_s - flat_t
            d_norm = torch.linalg.vector_norm(flat_d, dim=1)
            s_norm = torch.linalg.vector_norm(flat_s, dim=1)
            values = {
                "d_to_s_norm_ratio": d_norm / (s_norm + self.mtkd.eps),
                "td_cosine_similarity": F.cosine_similarity(
                    flat_t, flat_d, dim=1, eps=self.mtkd.eps
                ),
            }
            for name, value in values.items():
                batch_total = value.sum().detach()
                if name in self._diagnostic_totals:
                    self._diagnostic_totals[name].add_(batch_total)
                else:
                    self._diagnostic_totals[name] = batch_total

    def get_diagnostics(self):
        diagnostics = super().get_diagnostics()
        for name in ("d_to_s_norm_ratio", "td_cosine_similarity"):
            if self._diagnostic_sample_count:
                diagnostics[name] = (
                    self._diagnostic_totals[name] / self._diagnostic_sample_count
                ).detach().cpu().item()
            else:
                diagnostics[name] = float("nan")
        return diagnostics

    def forward(self, spatial_feats, positions):
        t, s = self.mtkd(spatial_feats, positions)
        if self.training:
            self.record_diagnostics(t, s)
        d = s - t
        z_t = self.ltae_t(t, positions)
        z_d = self.ltae_d(d, positions)
        return torch.cat([z_t, z_d], dim=-1)


class MTKDTQMidConcatLTAE(MTKDMidConcatLTAE):
    """MTKD T/Q branches encoded independently, then concatenated."""

    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=8,
        d_model=256,
        n_neurons=(256, 128),
        dropout=0.2,
        T=1000,
        max_temporal_shift=100,
        time_scale_days=365.0,
        tau_fast_init_days=30.0,
        tau_slow_init_days=90.0,
        tau_min_days=1.0,
        delta_tau_min_days=1.0,
        learnable_tau=True,
    ):
        super().__init__(
            in_channels=in_channels,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=n_neurons,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
            time_scale_days=time_scale_days,
            tau_fast_init_days=tau_fast_init_days,
            tau_slow_init_days=tau_slow_init_days,
            tau_min_days=tau_min_days,
            delta_tau_min_days=delta_tau_min_days,
            learnable_tau=learnable_tau,
        )
        self.ltae_q = self.ltae_s
        del self.ltae_s

    def record_diagnostics(self, t, s, h):
        """Accumulate existing T/S metrics and detached T/Q metrics."""
        super().record_diagnostics(t, s)
        with torch.no_grad():
            flat_t = t.detach().reshape(t.shape[0], -1)
            flat_h = h.detach().reshape(h.shape[0], -1)
            flat_q = flat_h - flat_t
            q_norm = torch.linalg.vector_norm(flat_q, dim=1)
            h_norm = torch.linalg.vector_norm(flat_h, dim=1)
            values = {
                "q_to_h_norm_ratio": q_norm / (h_norm + self.mtkd.eps),
                "tq_cosine_similarity": F.cosine_similarity(
                    flat_t, flat_q, dim=1, eps=self.mtkd.eps
                ),
            }
            for name, value in values.items():
                batch_total = value.sum().detach()
                if name in self._diagnostic_totals:
                    self._diagnostic_totals[name].add_(batch_total)
                else:
                    self._diagnostic_totals[name] = batch_total

    def get_diagnostics(self):
        diagnostics = super().get_diagnostics()
        for name in ("q_to_h_norm_ratio", "tq_cosine_similarity"):
            if self._diagnostic_sample_count:
                diagnostics[name] = (
                    self._diagnostic_totals[name] / self._diagnostic_sample_count
                ).detach().cpu().item()
            else:
                diagnostics[name] = float("nan")
        return diagnostics

    def forward(self, spatial_feats, positions):
        t, s = self.mtkd(spatial_feats, positions)
        if self.training:
            self.record_diagnostics(t, s, spatial_feats)
        q = spatial_feats - t
        z_t = self.ltae_t(t, positions)
        z_q = self.ltae_q(q, positions)
        return torch.cat([z_t, z_q], dim=-1)


class MTKDTQSingleLTAE(nn.Module):
    """Fixed single-scale T/Q branches encoded independently, then concatenated."""

    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=8,
        d_model=256,
        n_neurons=(256, 128),
        dropout=0.2,
        T=1000,
        max_temporal_shift=100,
        tau_days=60.0,
        time_scale_days=365.0,
    ):
        super().__init__()
        self.smoother = SingleScaleTemporalKernel(
            tau_days=tau_days,
            time_scale_days=time_scale_days,
        )
        ltae_kwargs = dict(
            in_channels=in_channels,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=list(n_neurons),
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
        )
        self.ltae_t = LTAE(**ltae_kwargs)
        self.ltae_q = LTAE(**ltae_kwargs)
        self.reset_diagnostics()

    @property
    def max_temporal_shift(self):
        return self.ltae_t.max_temporal_shift

    @property
    def positional_enc(self):
        return self.ltae_t.positional_enc

    def reset_diagnostics(self):
        self._diagnostic_totals = {}
        self._diagnostic_sample_count = 0

    def record_diagnostics(self, t, h):
        with torch.no_grad():
            flat_t = t.detach().reshape(t.shape[0], -1)
            flat_h = h.detach().reshape(h.shape[0], -1)
            flat_q = flat_h - flat_t
            q_norm = torch.linalg.vector_norm(flat_q, dim=1)
            h_norm = torch.linalg.vector_norm(flat_h, dim=1)
            values = {
                "q_to_h_norm_ratio": q_norm / (h_norm + self.smoother.eps),
                "tq_cosine_similarity": F.cosine_similarity(
                    flat_t, flat_q, dim=1, eps=self.smoother.eps
                ),
            }
            for name, value in values.items():
                batch_total = value.sum().detach()
                if name in self._diagnostic_totals:
                    self._diagnostic_totals[name].add_(batch_total)
                else:
                    self._diagnostic_totals[name] = batch_total
            self._diagnostic_sample_count += t.shape[0]

    def get_diagnostics(self):
        diagnostics = {"tau_days": self.smoother.tau_days.detach().cpu().item()}
        for name in ("q_to_h_norm_ratio", "tq_cosine_similarity"):
            if self._diagnostic_sample_count:
                diagnostics[name] = (
                    self._diagnostic_totals[name] / self._diagnostic_sample_count
                ).detach().cpu().item()
            else:
                diagnostics[name] = float("nan")
        return diagnostics

    def forward(self, spatial_feats, positions):
        t = self.smoother(spatial_feats, positions)
        if self.training:
            self.record_diagnostics(t, spatial_feats)
        q = spatial_feats - t
        z_t = self.ltae_t(t, positions)
        z_q = self.ltae_q(q, positions)
        return torch.cat([z_t, z_q], dim=-1)


class MTKDTQLOOLTAE(nn.Module):
    """Single-scale T/Q encoder with auditable LOO gradient routing."""

    VARIANTS = {"loo_tau_only", "pse_loo_fixed75", "split_loo_tau_pse"}

    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=8,
        d_model=256,
        n_neurons=(256, 128),
        dropout=0.2,
        T=1000,
        max_temporal_shift=100,
        variant="split_loo_tau_pse",
        tau_init_days=75.0,
        tau_min_days=1.0,
        time_scale_days=365.0,
    ):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError("unknown TQ LOO variant: {}".format(variant))
        self.variant = variant
        self.tau_init_days = 75.0 if variant == "pse_loo_fixed75" else float(tau_init_days)
        if variant == "pse_loo_fixed75":
            self.smoother = SingleScaleTemporalKernel(
                tau_days=75.0,
                time_scale_days=time_scale_days,
            )
        else:
            self.smoother = LearnableSingleScaleTemporalKernel(
                tau_init_days=tau_init_days,
                tau_min_days=tau_min_days,
                time_scale_days=time_scale_days,
            )
            self.smoother.a.register_hook(self._record_tau_gradient)
        ltae_kwargs = dict(
            in_channels=in_channels,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=list(n_neurons),
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
        )
        self.ltae_t = LTAE(**ltae_kwargs)
        self.ltae_q = LTAE(**ltae_kwargs)
        self.reset_diagnostics()

    @property
    def max_temporal_shift(self):
        return self.ltae_t.max_temporal_shift

    @property
    def positional_enc(self):
        return self.ltae_t.positional_enc

    @property
    def tau_is_trainable(self):
        return isinstance(self.smoother, LearnableSingleScaleTemporalKernel)

    def get_tau_days(self):
        if self.tau_is_trainable:
            return self.smoother.get_tau_days()
        return self.smoother.tau_days

    def _record_tau_gradient(self, gradient):
        if hasattr(self, "_tau_grad_total"):
            self._tau_grad_total += gradient.detach().abs().cpu().item()
            self._tau_grad_count += 1
        return gradient

    def reset_diagnostics(self):
        self._diagnostic_totals = {}
        self._diagnostic_counts = {}
        self._tau_grad_total = 0.0
        self._tau_grad_count = 0

    def _smooth_task(self, h, positions, time_mask=None):
        if self.tau_is_trainable:
            return self.smoother(h, positions, time_mask=time_mask, detach_tau=True)
        return self.smoother(h, positions, time_mask=time_mask)

    def forward(self, spatial_feats, positions, time_mask=None):
        t = self._smooth_task(spatial_feats, positions, time_mask=time_mask)
        q = spatial_feats - t
        z_t = self.ltae_t(t, positions)
        z_q = self.ltae_q(q, positions)
        return torch.cat([z_t, z_q], dim=-1)

    def compute_loo_losses(self, h, positions, time_mask=None):
        tau_days = self.get_tau_days()
        zero = h.sum() * 0.0 + tau_days.sum() * 0.0
        losses = {
            "loo_tau": zero,
            "loo_pse": zero,
            "loo_valid_query_count": 0,
        }
        if self.variant in ("loo_tau_only", "split_loo_tau_pse"):
            losses["loo_tau"], count = leave_one_out_smooth_l1(
                h.detach(),
                positions,
                tau_days,
                time_scale_days=self.smoother.time_scale_days,
                time_mask=time_mask,
                eps=self.smoother.eps,
            )
            losses["loo_valid_query_count"] = count
        if self.variant in ("pse_loo_fixed75", "split_loo_tau_pse"):
            losses["loo_pse"], count = leave_one_out_smooth_l1(
                h,
                positions,
                tau_days.detach(),
                time_scale_days=self.smoother.time_scale_days,
                time_mask=time_mask,
                eps=self.smoother.eps,
            )
            losses["loo_valid_query_count"] = count
        return losses

    def record_domain_diagnostics(
        self,
        prefix,
        h,
        positions,
        loo_error,
        valid_query_count,
        time_mask=None,
        labels=None,
    ):
        metrics = compute_feature_curve_diagnostics(
            h, positions, time_mask=time_mask, labels=labels
        )
        metrics["loo_error"] = loo_error.detach()
        metrics["loo_valid_query_count"] = h.new_tensor(float(valid_query_count))
        for name, value in metrics.items():
            full_name = "{}_{}".format(prefix, name) if prefix else name
            scalar = value.detach().cpu().item()
            self._diagnostic_totals[full_name] = (
                self._diagnostic_totals.get(full_name, 0.0) + scalar
            )
            self._diagnostic_counts[full_name] = (
                self._diagnostic_counts.get(full_name, 0) + 1
            )

    def get_diagnostics(self):
        tau_days = self.get_tau_days().detach().cpu().item()
        diagnostics = {
            "tau_days": tau_days,
            "tau_trainable": float(self.tau_is_trainable),
        }
        if self.tau_is_trainable:
            diagnostics["tau_delta_from_init"] = tau_days - self.tau_init_days
            diagnostics["tau_grad_abs"] = (
                self._tau_grad_total / self._tau_grad_count
                if self._tau_grad_count
                else float("nan")
            )
        for name, total in self._diagnostic_totals.items():
            diagnostics[name] = total / self._diagnostic_counts[name]
        return diagnostics


def optimizer_parameter_groups(model, weight_decay):
    """Keep the single learnable bandwidth scalar out of weight decay."""
    temporal_encoder = getattr(model, "temporal_encoder", None)
    smoother = getattr(temporal_encoder, "smoother", None)
    tau_parameter = getattr(smoother, "a", None)
    if not isinstance(tau_parameter, nn.Parameter) or not tau_parameter.requires_grad:
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]

    tau_parameter_id = id(tau_parameter)
    main_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) != tau_parameter_id
    ]
    return [
        {"params": main_parameters, "weight_decay": weight_decay},
        {"params": [tau_parameter], "weight_decay": 0.0},
    ]


def reset_mtkd_diagnostics(model):
    """Reset diagnostics when the model uses an MTKD temporal encoder."""
    temporal_encoder = getattr(model, "temporal_encoder", None)
    if isinstance(
        temporal_encoder,
        (
            MTKDEarlyConcatLTAE,
            MTKDMidConcatLTAE,
            MTKDTQSingleLTAE,
            MTKDTQLOOLTAE,
        ),
    ):
        temporal_encoder.reset_diagnostics()


def log_mtkd_diagnostics(model, stage, epoch, writer=None):
    """Print and optionally publish one non-training MTKD epoch summary."""
    temporal_encoder = getattr(model, "temporal_encoder", None)
    if not isinstance(
        temporal_encoder,
        (
            MTKDEarlyConcatLTAE,
            MTKDMidConcatLTAE,
            MTKDTQSingleLTAE,
            MTKDTQLOOLTAE,
        ),
    ):
        return None

    diagnostics = temporal_encoder.get_diagnostics()
    if isinstance(temporal_encoder, MTKDTQLOOLTAE):
        lines = [
            "=" * 60,
            "TQ LOO DIAGNOSTICS",
            "stage                  : {}".format(stage),
            "epoch                  : {}".format(epoch),
            "variant                : {}".format(temporal_encoder.variant),
        ]
        for name, value in diagnostics.items():
            lines.append("{:<23}: {:.6f}".format(name, value))
        lines.append("=" * 60)
        print("\n\n" + "\n".join(lines) + "\n\n", flush=True)
        if writer is not None:
            for name, value in diagnostics.items():
                writer.add_scalar("mtkd/{}".format(name), value, epoch)
        return diagnostics

    if isinstance(temporal_encoder, MTKDTQSingleLTAE):
        lines = [
            "=" * 60,
            "SINGLE-TAU TQ DIAGNOSTICS",
            "stage                  : {}".format(stage),
            "epoch                  : {}".format(epoch),
            "",
            "tau_days               : {:.6f}".format(diagnostics["tau_days"]),
            "q_to_h_norm_ratio      : {:.6f}".format(
                diagnostics["q_to_h_norm_ratio"]
            ),
            "tq_cosine_similarity   : {:.6f}".format(
                diagnostics["tq_cosine_similarity"]
            ),
            "=" * 60,
        ]
        print("\n\n" + "\n".join(lines) + "\n\n", flush=True)
        if writer is not None:
            for name, value in diagnostics.items():
                writer.add_scalar("mtkd/{}".format(name), value, epoch)
        return diagnostics

    lines = [
        "=" * 60,
        "MTKD DIAGNOSTICS",
        "stage                  : {}".format(stage),
        "epoch                  : {}".format(epoch),
        "",
        "tau_fast_days          : {:.6f}".format(diagnostics["tau_fast_days"]),
        "tau_slow_days          : {:.6f}".format(diagnostics["tau_slow_days"]),
        "delta_tau_days         : {:.6f}".format(diagnostics["delta_tau_days"]),
        "",
        "ts_relative_difference : {:.6f}".format(
            diagnostics["ts_relative_difference"]
        ),
        "ts_cosine_similarity   : {:.6f}".format(
            diagnostics["ts_cosine_similarity"]
        ),
        "t_to_s_norm_ratio      : {:.6f}".format(
            diagnostics["t_to_s_norm_ratio"]
        ),
        "=" * 60,
    ]
    if "d_to_s_norm_ratio" in diagnostics:
        lines[-1:-1] = [
            "d_to_s_norm_ratio      : {:.6f}".format(
                diagnostics["d_to_s_norm_ratio"]
            ),
            "td_cosine_similarity   : {:.6f}".format(
                diagnostics["td_cosine_similarity"]
            ),
        ]
    if "q_to_h_norm_ratio" in diagnostics:
        lines[-1:-1] = [
            "q_to_h_norm_ratio      : {:.6f}".format(
                diagnostics["q_to_h_norm_ratio"]
            ),
            "tq_cosine_similarity   : {:.6f}".format(
                diagnostics["tq_cosine_similarity"]
            ),
        ]
    print("\n\n" + "\n".join(lines) + "\n\n", flush=True)
    if writer is not None:
        for name, value in diagnostics.items():
            writer.add_scalar("mtkd/{}".format(name), value, epoch)
    return diagnostics
