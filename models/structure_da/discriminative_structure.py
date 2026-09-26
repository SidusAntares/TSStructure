"""Differentiable discriminative structure representation for irregular SITS."""

import torch
from torch import nn
from torch.nn import functional as F

from models.fourier_reconstruction import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
    _batched_fourier_matrix,
    positions_to_periodic_points,
)


class FourierStructureExposer(nn.Module):
    """Expose a smooth structure view on a fixed periodic grid."""

    def __init__(self, num_modes=13, grid_points=64, period_days=365.0, reg=1e-3):
        super().__init__()
        self.analyzer = BatchedDirectFourierAnalyzer(num_modes, period_days, reg)
        self.synthesizer = BatchedDirectFourierSynthesizer(num_modes, period_days)
        self.grid_points = int(grid_points)
        self.period_days = float(period_days)
        self.register_buffer(
            "canonical_grid",
            torch.arange(self.grid_points, dtype=torch.float32)
            * (self.period_days / self.grid_points),
        )
        points = positions_to_periodic_points(self.canonical_grid, self.period_days)
        self.register_buffer(
            "canonical_synthesis_matrix",
            _batched_fourier_matrix(
                points, num_modes, self.synthesizer.synthesis_isign,
                torch.complex64,
            ),
            persistent=False,
        )
        double_points = positions_to_periodic_points(
            self.canonical_grid.double(), self.period_days,
        )
        self.register_buffer(
            "canonical_synthesis_matrix_double",
            _batched_fourier_matrix(
                double_points, num_modes, self.synthesizer.synthesis_isign,
                torch.complex128,
            ),
            persistent=False,
        )

    def synthesize_canonical(self, coefficients):
        """Evaluate only the fixed canonical grid using its cached Fourier basis."""
        cached = (
            self.canonical_synthesis_matrix_double
            if coefficients.dtype == torch.complex128
            else self.canonical_synthesis_matrix
        )
        matrix = cached.to(
            device=coefficients.device, dtype=coefficients.dtype,
        )
        return torch.matmul(matrix.unsqueeze(0), coefficients).real

    def forward(self, features, positions):
        coefficients, diagnostics = self.analyzer(features, positions)
        if torch.any(diagnostics["solver_info"] != 0):
            raise FloatingPointError("non-finite Fourier solve in structure-shapelet training")
        if not torch.isfinite(coefficients).all():
            raise FloatingPointError("non-finite Fourier coefficients in structure-shapelet training")
        grid = self.canonical_grid.to(device=features.device, dtype=features.dtype)
        grid = grid.unsqueeze(0).expand(features.shape[0], -1)
        exposed = self.synthesize_canonical(coefficients)
        if not torch.isfinite(exposed).all():
            raise FloatingPointError("non-finite Fourier reconstruction in structure-shapelet training")
        return exposed, grid


class MultiScaleWindowExtractor(nn.Module):
    """Extract fixed, dense, non-extrema local windows in deterministic order."""

    def __init__(self, scales=(8, 16, 24), stride=4, grid_points=64):
        super().__init__()
        scales = tuple(int(value) for value in scales)
        if not scales or min(scales) < 2 or stride < 1 or grid_points < 2:
            raise ValueError("window scales must be >=2 and stride must be positive")
        if max(scales) > grid_points:
            raise ValueError("window scales cannot exceed the exposed curve length")
        self.scales = scales
        self.stride = int(stride)
        self.grid_points = int(grid_points)
        for index, scale in enumerate(scales):
            starts = torch.arange(0, self.grid_points, self.stride)
            offsets = torch.arange(scale)
            self.register_buffer(
                f"indices_{index}",
                (starts[:, None] + offsets[None, :]) % self.grid_points,
                persistent=False,
            )

    def forward(self, curve, return_centers=False):
        if curve.ndim != 3:
            raise ValueError("curve must be [B,G,D]")
        _, points, _ = curve.shape
        if points != self.grid_points:
            raise ValueError(
                f"expected exposed curve length {self.grid_points}, got {points}"
            )
        windows, scale_ids, centers = [], [], []
        for index, scale in enumerate(self.scales):
            indices = getattr(self, f"indices_{index}")
            windows.append(curve[:, indices])
            scale_ids.extend([scale] * indices.shape[0])
            starts = torch.arange(
                0, self.grid_points, self.stride,
                device=curve.device, dtype=curve.dtype,
            )
            centers.append((starts + (scale - 1) / 2.) % self.grid_points)
        if not windows:
            raise ValueError("no window scale fits the exposed curve")
        scales = torch.tensor(scale_ids, device=curve.device)
        if return_centers:
            return windows, scales, torch.cat(centers)
        return windows, scales


class _ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden, dilation, dropout=.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, values):
        return self.norm((values + self.network(values)).transpose(1, 2)).transpose(1, 2)


class _VariableLengthTemporalEncoder(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.input_projection = nn.Conv1d(channels, hidden, 1)
        self.multiscale = nn.ModuleList(
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
            for kernel in (3, 5, 7)
        )
        self.multiscale_projection = nn.Conv1d(3 * hidden, hidden, 1)
        self.blocks = nn.ModuleList(
            _ResidualTemporalBlock(hidden, dilation) for dilation in (1, 2, 4)
        )
        self.temporal_score = nn.Conv1d(hidden, 1, 1)

    def forward(self, values, mask=None):
        batch, tokens, points, channels = values.shape
        flat = values.reshape(batch * tokens, points, channels).transpose(1, 2)
        encoded = self.input_projection(flat)
        encoded = self.multiscale_projection(torch.cat([
            F.gelu(layer(encoded)) for layer in self.multiscale
        ], dim=1))
        for block in self.blocks:
            encoded = block(encoded)
        scores = self.temporal_score(encoded).squeeze(1)
        if mask is not None:
            flat_mask = mask.reshape(batch * tokens, points)
            scores = scores.masked_fill(~flat_mask, -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        pooled = (encoded * attention.unsqueeze(1)).sum(-1)
        return pooled.reshape(batch, tokens, -1)


class ShapeTokenGenerator(nn.Module):
    """Encode normalized morphology/dynamics and absolute level/variation."""

    def __init__(self, channels, shape_dim=128, branch_dim=32, eps=1e-6, resample_length=16):
        super().__init__()
        self.eps = eps
        self.resample_length = int(resample_length)
        if self.resample_length < 2:
            raise ValueError("shape resample length must be at least 2")
        self.raw_encoder = _VariableLengthTemporalEncoder(channels, branch_dim)
        self.diff_encoder = _VariableLengthTemporalEncoder(channels, branch_dim)
        self.mean_encoder = nn.Sequential(nn.Linear(channels, branch_dim), nn.GELU(), nn.Linear(branch_dim, branch_dim))
        self.std_encoder = nn.Sequential(nn.Linear(channels, branch_dim), nn.GELU(), nn.Linear(branch_dim, branch_dim))
        self.fusion = nn.Sequential(
            nn.Linear(4 * branch_dim, shape_dim),
            nn.LayerNorm(shape_dim),
            nn.GELU(),
        )

    def components(self, windows, mask=None):
        if mask is None:
            mask = torch.ones(
                windows.shape[:3], dtype=torch.bool, device=windows.device,
            )
        weights = mask.to(windows.dtype).unsqueeze(-1)
        count = weights.sum(2).clamp_min(1)
        mean = (windows * weights).sum(2) / count
        centered = windows - mean.unsqueeze(2)
        variance = (centered.square() * weights).sum(2) / count
        variance = variance.clamp_min(0)
        std = torch.sqrt(variance + self.eps)
        normalized = centered / std.unsqueeze(2)
        normalized = normalized * weights
        batch, tokens, _, channels = normalized.shape
        normalized = F.interpolate(
            normalized.reshape(batch * tokens, -1, channels).transpose(1, 2),
            size=self.resample_length, mode="linear", align_corners=True,
        ).transpose(1, 2).reshape(batch, tokens, self.resample_length, channels)
        difference = torch.zeros_like(normalized)
        difference[:, :, 1:] = normalized[:, :, 1:] - normalized[:, :, :-1]
        return {"normalized": normalized, "difference": difference, "mean": mean, "std": std}

    def forward(self, windows, mask=None, return_encoded_components=False):
        parts = self.components(windows, mask)
        raw = self.raw_encoder(parts["normalized"])
        difference = self.diff_encoder(parts["difference"])
        mean = self.mean_encoder(parts["mean"])
        std = self.std_encoder(parts["std"])
        token = self.fusion(torch.cat((raw, difference, mean, std), dim=-1))
        if return_encoded_components:
            return {
                "raw_encoded": raw,
                "diff_encoded": difference,
                "mean_encoded": mean,
                "std_encoded": std,
                "shape_token": token,
            }
        return token


def normalized_candidate_concentration(weights, candidate_mask=None, eps=1e-12):
    """Return one minus candidate entropy, normalized by valid candidate count."""
    if weights.ndim != 3:
        raise ValueError("weights must be [B,N,M]")
    if candidate_mask is None:
        candidate_mask = torch.ones(
            weights.shape[:2], dtype=torch.bool, device=weights.device,
        )
    else:
        candidate_mask = torch.as_tensor(
            candidate_mask, dtype=torch.bool, device=weights.device,
        )
        if candidate_mask.ndim == 1:
            candidate_mask = candidate_mask.unsqueeze(0).expand(weights.shape[0], -1)
    if candidate_mask.shape != weights.shape[:2]:
        raise ValueError("candidate_mask must be [N] or [B,N]")
    valid_count = candidate_mask.sum(1).to(weights.dtype)
    if torch.any(valid_count == 0):
        raise ValueError("every sample must retain at least one candidate")
    entropy = -(weights * weights.clamp_min(eps).log()).sum(1)
    denominator = valid_count.log()
    normalized = torch.where(
        valid_count[:, None] > 1,
        entropy / denominator[:, None].clamp_min(eps),
        torch.zeros_like(entropy),
    )
    return (1. - normalized).clamp(0., 1.)


class ShapeletDictionary(nn.Module):
    """Shared learned morphology anchors with candidate-wise soft assignment."""

    def __init__(self, shape_dim=128, count=16, beta=5.):
        super().__init__()
        if count < 1 or beta <= 0:
            raise ValueError("shapelet count and beta must be positive")
        self.anchors = nn.Parameter(torch.randn(int(count), int(shape_dim)))
        self.beta = float(beta)

    def compute_similarity(self, tokens):
        return F.normalize(tokens, dim=-1) @ F.normalize(self.anchors, dim=-1).T

    def compute_response(self, tokens, candidate_mask=None, return_details=False):
        similarity = self.compute_similarity(tokens)
        if candidate_mask is None:
            candidate_mask = torch.ones(
                similarity.shape[:2], dtype=torch.bool, device=similarity.device,
            )
        else:
            candidate_mask = torch.as_tensor(
                candidate_mask, dtype=torch.bool, device=similarity.device,
            )
            if candidate_mask.ndim == 1:
                candidate_mask = candidate_mask.unsqueeze(0).expand(similarity.shape[0], -1)
            if candidate_mask.shape != similarity.shape[:2]:
                raise ValueError("candidate_mask must be [N] or [B,N]")
        if not candidate_mask.any(dim=1).all():
            raise ValueError("every sample must retain at least one candidate")
        scores = (self.beta * similarity).masked_fill(
            ~candidate_mask.unsqueeze(-1), -torch.inf,
        )
        weights = torch.softmax(scores, dim=1)
        response = (weights * similarity).sum(dim=1)
        if return_details:
            return {
                "response": response,
                "similarity": similarity,
                "weights": weights,
                "candidate_mask": candidate_mask,
            }
        return response

    def forward(self, tokens):
        return self.compute_response(tokens)


class InvariantProjector(nn.Module):
    """Residual projector used by the formal V4 structure representation."""

    def __init__(self, feature_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.GELU(), nn.Linear(128, feature_dim),
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, response):
        return self.norm(response + self.mlp(response))


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, alpha):
        ctx.alpha = float(alpha)
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.alpha * gradient, None


def gradient_reverse(features, alpha):
    return _GradientReversal.apply(features, alpha)


class StructureDomainClassifier(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.GELU(), nn.Dropout(.1),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2),
        )

    def forward(self, features):
        return self.network(features)


class DomainProjector(nn.Module):
    def __init__(self, input_dim=64, domain_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, domain_dim),
            nn.LayerNorm(domain_dim),
        )

    def forward(self, features):
        return self.network(features)


class PrivateDomainClassifier(nn.Module):
    def __init__(self, feature_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.GELU(), nn.Dropout(.1),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2),
        )

    def forward(self, features):
        return self.network(features)


def initialize_shapelet_dictionary_from_tokens(dictionary, tokens, seed):
    """Copy deterministic spherical K-means centers into an existing dictionary."""
    from sklearn.cluster import KMeans

    if tokens.ndim != 2 or tokens.shape[1] != dictionary.anchors.shape[1]:
        raise ValueError("tokens must be [N, shape_dim]")
    count = int(dictionary.anchors.shape[0])
    if tokens.shape[0] < count:
        raise ValueError(f"need at least {count} source tokens, got {tokens.shape[0]}")
    normalized = F.normalize(tokens.detach().float().cpu(), dim=-1, eps=1e-8)
    estimator = KMeans(n_clusters=count, random_state=int(seed), n_init=10)
    centers = torch.from_numpy(estimator.fit(normalized.numpy()).cluster_centers_)
    centers = F.normalize(centers, dim=-1, eps=1e-8)
    with torch.no_grad():
        dictionary.anchors.copy_(centers.to(
            device=dictionary.anchors.device,
            dtype=dictionary.anchors.dtype,
        ))
    anchors = F.normalize(dictionary.anchors.detach(), dim=-1, eps=1e-8)
    pairwise = anchors @ anchors.T
    off_diagonal = pairwise[~torch.eye(count, dtype=torch.bool, device=pairwise.device)]
    return {
        "tokens": int(tokens.shape[0]),
        "anchors": count,
        "pairwise_cos_mean": float(off_diagonal.mean().cpu()) if off_diagonal.numel() else 0.,
        "pairwise_cos_max": float(off_diagonal.max().cpu()) if off_diagonal.numel() else 0.,
    }


class DiscriminativeStructureBranch(nn.Module):
    def __init__(
        self, channels, shape_dim=128, num_modes=13, grid_points=64,
        period_days=365.0, reg=1e-3, window_scales=(24,), window_stride=8,
        shapelet_count=16, shapelet_beta=5., shape_resample_length=16,
    ):
        super().__init__()
        self.exposer = FourierStructureExposer(num_modes, grid_points, period_days, reg)
        self.window_extractor = MultiScaleWindowExtractor(
            window_scales, window_stride, grid_points=grid_points,
        )
        self.token_generator = ShapeTokenGenerator(
            channels, shape_dim, resample_length=shape_resample_length,
        )
        self.shapelet_dictionary = ShapeletDictionary(shape_dim, shapelet_count, shapelet_beta)
        response_dim = 2 * shapelet_count
        self.response_to_query = nn.Sequential(
            nn.Linear(response_dim, 64), nn.GELU(),
            nn.Linear(64, shape_dim), nn.LayerNorm(shape_dim),
        )

    def compose_rich_response(self, details):
        strength = details["response"]
        concentration = normalized_candidate_concentration(
            details["weights"], details["candidate_mask"],
        )
        return torch.cat((strength, concentration), dim=-1)

    def compute_rich_response(self, tokens, candidate_mask=None, return_details=False):
        details = self.shapelet_dictionary.compute_response(
            tokens, candidate_mask=candidate_mask, return_details=True,
        )
        rich = self.compose_rich_response(details)
        if return_details:
            return {**details, "strength": details["response"], "rich_response": rich}
        return rich

    def forward(self, features, positions):
        exposed, grid = self.exposer(features, positions)
        window_groups, scales = self.window_extractor(exposed)
        encoded = [
            self.token_generator(windows, return_encoded_components=True)
            for windows in window_groups
        ]
        tokens = torch.cat([value["shape_token"] for value in encoded], dim=1)
        stats_tokens = torch.cat([
            torch.cat((value["mean_encoded"], value["std_encoded"]), dim=-1)
            for value in encoded
        ], dim=1)
        details = self.shapelet_dictionary.compute_response(tokens, return_details=True)
        strength = details["response"]
        response = self.compose_rich_response(details)
        concentration = response[:, strength.shape[1]:]
        return {
            "shape_tokens": tokens,
            "shapelet_strength": strength,
            "shapelet_concentration": concentration,
            "shapelet_response": response,
            "shape_stats_feature": stats_tokens.mean(dim=1),
            "shape_class_token": self.response_to_query(response),
            "shape_scales": scales,
            "exposed_curve": exposed,
            "exposed_grid": grid,
        }
