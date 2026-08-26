import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ltae import LTAE


def _inverse_softplus(value):
    """Numerically stable inverse of softplus for a positive scalar."""
    return value + math.log(-math.expm1(-value))


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
        tau = tau_days.to(dtype=h.dtype) / self.time_scale_days.to(dtype=h.dtype)
        kernel = torch.exp(-pairwise_distance / tau)
        if time_mask is not None:
            key_mask = time_mask.to(device=h.device, dtype=h.dtype).unsqueeze(1)
            kernel = kernel * key_mask
        weights = kernel / (kernel.sum(dim=-1, keepdim=True) + self.eps)
        smoothed = torch.bmm(weights, h)
        if time_mask is not None:
            query_mask = time_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
            smoothed = smoothed * query_mask
        return smoothed

    def forward(self, h, positions, time_mask=None):
        if h.ndim != 3:
            raise ValueError("h must have shape [B, L, C]")
        if positions.shape != h.shape[:2]:
            raise ValueError("positions must have shape [B, L]")
        if time_mask is not None and time_mask.shape != h.shape[:2]:
            raise ValueError("time_mask must have shape [B, L]")

        normalized_positions = positions.to(device=h.device, dtype=h.dtype)
        normalized_positions = normalized_positions / self.time_scale_days.to(
            dtype=h.dtype
        )
        pairwise_distance = torch.abs(
            normalized_positions.unsqueeze(-1) - normalized_positions.unsqueeze(-2)
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


def reset_mtkd_diagnostics(model):
    """Reset diagnostics when the model uses an MTKD temporal encoder."""
    temporal_encoder = getattr(model, "temporal_encoder", None)
    if isinstance(temporal_encoder, (MTKDEarlyConcatLTAE, MTKDMidConcatLTAE)):
        temporal_encoder.reset_diagnostics()


def log_mtkd_diagnostics(model, stage, epoch, writer=None):
    """Print and optionally publish one non-training MTKD epoch summary."""
    temporal_encoder = getattr(model, "temporal_encoder", None)
    if not isinstance(temporal_encoder, (MTKDEarlyConcatLTAE, MTKDMidConcatLTAE)):
        return None

    diagnostics = temporal_encoder.get_diagnostics()
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
    print("\n\n" + "\n".join(lines) + "\n\n", flush=True)
    if writer is not None:
        for name, value in diagnostics.items():
            writer.add_scalar("mtkd/{}".format(name), value, epoch)
    return diagnostics
