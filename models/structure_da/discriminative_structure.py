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

    def analyze(self, features, positions):
        coefficients, diagnostics = self.analyzer(features, positions)
        if torch.any(diagnostics["solver_info"] != 0):
            raise FloatingPointError("non-finite Fourier solve in structure-shapelet training")
        if not torch.isfinite(coefficients).all():
            raise FloatingPointError("non-finite Fourier coefficients in structure-shapelet training")
        return coefficients

    def synthesize_shifted(self, coefficients, temporal_shift=0):
        batch = coefficients.shape[0]
        shift = torch.as_tensor(
            temporal_shift, device=coefficients.device,
            dtype=self.canonical_grid.dtype,
        )
        if shift.ndim == 0:
            shift = shift.reshape(1, 1).expand(batch, 1)
        elif shift.ndim == 1:
            if shift.numel() == 1:
                shift = shift.reshape(1, 1).expand(batch, 1)
            elif shift.numel() == batch:
                shift = shift[:, None]
            else:
                raise ValueError("temporal shift vector must match batch size")
        elif shift.shape != (batch, 1):
            raise ValueError("temporal shift must be scalar, [B], or [B,1]")
        grid = self.canonical_grid.to(
            device=coefficients.device, dtype=shift.dtype,
        )[None].expand(batch, -1)
        exposed = (
            self.synthesize_canonical(coefficients)
            if torch.count_nonzero(shift) == 0
            else self.synthesizer(coefficients, grid - shift)
        )
        if not torch.isfinite(exposed).all():
            raise FloatingPointError("non-finite Fourier reconstruction in structure-shapelet training")
        return exposed, grid

    def forward(self, features, positions):
        coefficients = self.analyze(features, positions)
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


class StateShapeTokenGenerator(nn.Module):
    """Encode standardized local state and its physical-time derivative."""

    def __init__(
        self, channels, shape_dim=128, period_days=365., grid_points=64,
        eps=1e-6,
    ):
        super().__init__()
        self.eps = float(eps)
        self.delta_t = float(period_days) / int(grid_points)
        self.state_encoder = _VariableLengthTemporalEncoder(
            2 * int(channels), int(shape_dim),
        )

    def standardized_state(self, windows, mask=None):
        if windows.ndim != 4:
            raise ValueError("state windows must be [B,N,T,C]")
        if mask is None:
            mask = torch.ones(
                windows.shape[:3], dtype=torch.bool, device=windows.device,
            )
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=windows.device)
        if mask.shape != windows.shape[:3]:
            raise ValueError("state window mask must be [B,N,T]")
        weights = mask.to(windows.dtype).unsqueeze(-1)
        count = weights.sum(2).clamp_min(1.)
        mean = (windows * weights).sum(2) / count
        centered = windows - mean.unsqueeze(2)
        variance = (centered.square() * weights).sum(2) / count
        standardized = centered / torch.sqrt(
            variance.clamp_min(0.).unsqueeze(2) + self.eps
        )
        standardized = standardized * weights
        derivative = torch.zeros_like(standardized)
        derivative[:, :, 1:] = (
            standardized[:, :, 1:] - standardized[:, :, :-1]
        ) / self.delta_t
        derivative = derivative * weights
        return torch.cat((standardized, derivative), dim=-1), mask

    def forward(self, windows, mask=None):
        state, mask = self.standardized_state(windows, mask)
        return self.state_encoder(state, mask)


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


def sorted_anchor_profile(similarity):
    """Keep each anchor's full response distribution without window order."""
    if similarity.ndim != 3:
        raise ValueError("similarity must be [B,N,M]")
    return similarity.sort(dim=1, descending=True).values.transpose(1, 2).flatten(1)


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
        shape_representation="current", state_org_readout="full",
    ):
        super().__init__()
        if shape_representation not in (
            "current", "sorted_profile", "set_response", "residual_response",
            "phase_moment", "state_org",
        ):
            raise ValueError(f"unknown shape representation: {shape_representation}")
        self.shape_representation = shape_representation
        if state_org_readout not in ("full", "composition", "presence"):
            raise ValueError("state_org_readout must be full, composition, or presence")
        if shape_representation != "state_org" and state_org_readout != "full":
            raise ValueError("non-full state_org_readout requires state_org representation")
        self.state_org_readout = state_org_readout
        self.exposer = FourierStructureExposer(num_modes, grid_points, period_days, reg)
        self.window_extractor = MultiScaleWindowExtractor(
            window_scales, window_stride, grid_points=grid_points,
        )
        self.token_generator = (
            StateShapeTokenGenerator(
                channels, shape_dim, period_days=period_days,
                grid_points=grid_points,
            )
            if shape_representation == "state_org"
            else ShapeTokenGenerator(
                channels, shape_dim, resample_length=shape_resample_length,
            )
        )
        self.shapelet_dictionary = ShapeletDictionary(shape_dim, shapelet_count, shapelet_beta)
        if shape_representation == "state_org" and state_org_readout == "full":
            self.organization_encoder = nn.Sequential(
                nn.Conv1d(
                    shapelet_count, 32, kernel_size=3, padding=1,
                    padding_mode="circular",
                ),
                nn.GELU(),
                nn.Conv1d(
                    32, 32, kernel_size=3, padding=1,
                    padding_mode="circular",
                ),
                nn.GELU(),
            )
            response_dim = shapelet_count + 32
        elif shape_representation == "state_org" and state_org_readout == "composition":
            self.composition_encoder = nn.Sequential(
                nn.Linear(shapelet_count, 96), nn.GELU(),
                nn.Linear(96, 32), nn.GELU(),
            )
            response_dim = shapelet_count + 32
        elif shape_representation == "state_org":
            response_dim = shapelet_count + 32
        elif shape_representation == "set_response":
            self.window_set_encoder = nn.Sequential(
                nn.Linear(shapelet_count, 64), nn.GELU(), nn.Linear(64, 32),
            )
            response_dim = 32
        elif shape_representation == "phase_moment":
            response_dim = 6 * shapelet_count
        else:
            response_dim = 2 * shapelet_count
        if shape_representation == "residual_response":
            self.anchor_context_weight = nn.Parameter(
                torch.zeros(shapelet_count, shapelet_count)
            )
        self.response_to_query = (
            None if shape_representation == "state_org" else nn.Sequential(
                nn.Linear(response_dim, 64), nn.GELU(),
                nn.Linear(64, shape_dim), nn.LayerNorm(shape_dim),
            )
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

    def compose_set_response(self, similarity, candidate_mask):
        if self.shape_representation != "set_response":
            raise RuntimeError("set response is only available in set_response mode")
        if similarity.ndim != 3:
            raise ValueError("similarity must be [B,N,M]")
        candidate_mask = torch.as_tensor(
            candidate_mask, dtype=torch.bool, device=similarity.device,
        )
        if candidate_mask.ndim == 1:
            candidate_mask = candidate_mask.unsqueeze(0).expand(similarity.shape[0], -1)
        if candidate_mask.shape != similarity.shape[:2]:
            raise ValueError("candidate_mask must be [N] or [B,N]")
        valid_count = candidate_mask.sum(dim=1, keepdim=True)
        if not (valid_count > 0).all():
            raise ValueError("every sample must retain at least one candidate")
        encoded = self.window_set_encoder(similarity)
        return (
            encoded * candidate_mask.unsqueeze(-1).to(encoded.dtype)
        ).sum(dim=1) / valid_count.to(encoded.dtype)

    def compose_state_org_response(self, similarity, candidate_mask=None):
        if self.shape_representation != "state_org":
            raise RuntimeError("state organization is only available in state_org mode")
        if similarity.ndim != 3:
            raise ValueError("similarity must be [B,N,M]")
        if similarity.shape[-1] != self.shapelet_dictionary.anchors.shape[0]:
            raise ValueError("similarity anchor dimension does not match dictionary")
        if candidate_mask is None:
            candidate_mask = torch.ones(
                similarity.shape[:2], dtype=torch.bool, device=similarity.device,
            )
        else:
            candidate_mask = torch.as_tensor(
                candidate_mask, dtype=torch.bool, device=similarity.device,
            )
            if candidate_mask.ndim == 1:
                candidate_mask = candidate_mask.unsqueeze(0).expand(
                    similarity.shape[0], -1,
                )
        if candidate_mask.shape != similarity.shape[:2]:
            raise ValueError("candidate_mask must be [N] or [B,N]")
        valid_count = candidate_mask.sum(1, keepdim=True)
        if not (valid_count > 0).all():
            raise ValueError("every sample must retain at least one candidate")
        presence_weights = torch.softmax(
            (self.shapelet_dictionary.beta * similarity).masked_fill(
                ~candidate_mask.unsqueeze(-1), -torch.inf,
            ),
            dim=1,
        )
        presence = (presence_weights * similarity).sum(1)
        state_distribution = torch.softmax(
            self.shapelet_dictionary.beta * similarity, dim=-1,
        )
        if self.state_org_readout == "full":
            organization_sequence = self.organization_encoder(
                state_distribution.transpose(1, 2)
            ).transpose(1, 2)
            organization = (
                organization_sequence
                * candidate_mask.unsqueeze(-1).to(organization_sequence.dtype)
            ).sum(1) / valid_count.to(organization_sequence.dtype)
        elif self.state_org_readout == "composition":
            composition_sequence = self.composition_encoder(state_distribution)
            organization = (
                composition_sequence
                * candidate_mask.unsqueeze(-1).to(composition_sequence.dtype)
            ).sum(1) / valid_count.to(composition_sequence.dtype)
        else:
            organization = similarity.new_zeros(similarity.shape[0], 32)
        response = torch.cat((presence, organization), dim=-1)
        return {
            "presence": presence,
            "presence_weights": presence_weights,
            "state_distribution": state_distribution,
            "organization": organization,
            "shapelet_response": response,
        }

    def contextualize_similarity(self, similarity):
        if self.shape_representation != "residual_response":
            raise RuntimeError(
                "contextual similarity is only available in residual_response mode"
            )
        if similarity.ndim != 3:
            raise ValueError("similarity must be [B,N,M]")
        count = similarity.shape[-1]
        if self.anchor_context_weight.shape != (count, count):
            raise ValueError("similarity anchor dimension does not match context weight")
        off_diagonal = 1. - torch.eye(
            count, dtype=similarity.dtype, device=similarity.device,
        )
        weight = self.anchor_context_weight * off_diagonal
        modulation = torch.tanh(similarity @ weight.T)
        return similarity * (1. + modulation), modulation

    def _residual_response_details(self, details):
        contextual, modulation = self.contextualize_similarity(details["similarity"])
        scores = (self.shapelet_dictionary.beta * contextual).masked_fill(
            ~details["candidate_mask"].unsqueeze(-1), -torch.inf,
        )
        weights = torch.softmax(scores, dim=1)
        strength = (weights * contextual).sum(dim=1)
        return {
            **details,
            "response": strength,
            "weights": weights,
            "contextual_similarity": contextual,
            "modulation": modulation,
        }

    def compose_phase_moments(self, weights, centers, phase_shift=0):
        """Return k=1,2 circular occurrence moments in fixed cos/sin block order."""
        batch, candidates, _ = weights.shape
        centers = torch.as_tensor(
            centers, device=weights.device, dtype=weights.dtype,
        )
        if centers.shape != (candidates,):
            raise ValueError("window centers must align with candidate weights")
        shift = torch.as_tensor(
            phase_shift, device=weights.device, dtype=weights.dtype,
        )
        if shift.ndim == 0:
            shift = shift.expand(batch)
        elif shift.ndim == 2 and shift.shape == (batch, 1):
            shift = shift[:, 0]
        elif shift.ndim != 1 or shift.shape[0] not in (1, batch):
            raise ValueError("phase shift must be scalar, [B], or [B,1]")
        if shift.shape[0] == 1:
            shift = shift.expand(batch)
        center_days = (
            centers * self.exposer.period_days / self.window_extractor.grid_points
        )
        theta = 2. * torch.pi * (
            center_days[None] + shift[:, None]
        ) / self.exposer.period_days
        blocks = []
        for harmonic in (1, 2):
            blocks.extend((
                (weights * torch.cos(harmonic * theta).unsqueeze(-1)).sum(1),
                (weights * torch.sin(harmonic * theta).unsqueeze(-1)).sum(1),
            ))
        return torch.cat(blocks, dim=-1)

    def prepare_context(self, features, positions):
        return {"coefficients": self.exposer.analyze(features, positions)}

    def forward_from_context(
        self, context, structure_aug_shift=0, phase_shift=0,
        include_legacy_query=True, temporal_shift=None,
    ):
        if temporal_shift is not None:
            structure_aug_shift = temporal_shift
        exposed, grid = self.exposer.synthesize_shifted(
            context["coefficients"], structure_aug_shift,
        )
        if self.shape_representation == "phase_moment":
            window_groups, scales, centers = self.window_extractor(
                exposed, return_centers=True,
            )
        else:
            window_groups, scales = self.window_extractor(exposed)
            centers = None
        if self.shape_representation == "state_org":
            tokens = torch.cat([
                self.token_generator(windows) for windows in window_groups
            ], dim=1)
            stats_tokens = None
        else:
            encoded = [
                self.token_generator(windows, return_encoded_components=True)
                for windows in window_groups
            ]
            tokens = torch.cat([value["shape_token"] for value in encoded], dim=1)
            stats_tokens = torch.cat([
                torch.cat((value["mean_encoded"], value["std_encoded"]), dim=-1)
                for value in encoded
            ], dim=1)
        if self.shape_representation == "state_org":
            similarity = self.shapelet_dictionary.compute_similarity(tokens)
            candidate_mask = torch.ones(
                similarity.shape[:2], dtype=torch.bool, device=similarity.device,
            )
            state = self.compose_state_org_response(
                similarity, candidate_mask,
            )
            return {
                "shape_tokens": tokens,
                "shapelet_similarity": similarity,
                "shapelet_strength": state["presence"],
                "shapelet_presence": state["presence"],
                "state_distribution": state["state_distribution"],
                "shape_organization": state["organization"],
                "shapelet_response": state["shapelet_response"],
                "shape_class_token": None,
                "shape_scales": scales,
                "exposed_curve": exposed,
                "exposed_grid": grid,
            }
        details = self.shapelet_dictionary.compute_response(tokens, return_details=True)
        response_details = (
            self._residual_response_details(details)
            if self.shape_representation == "residual_response" else details
        )
        strength = response_details["response"]
        rich_response = self.compose_rich_response(response_details)
        concentration = rich_response[:, strength.shape[1]:]
        phase_moments = None
        if self.shape_representation == "set_response":
            response = self.compose_set_response(
                details["similarity"], details["candidate_mask"],
            )
        elif self.shape_representation == "phase_moment":
            phase_moments = self.compose_phase_moments(
                response_details["weights"], centers, phase_shift,
            )
            response = torch.cat((rich_response, phase_moments), dim=-1)
        else:
            response = rich_response
        profile = sorted_anchor_profile(details["similarity"])
        result = {
            "shape_tokens": tokens,
            "shapelet_similarity": details["similarity"],
            "shapelet_strength": strength,
            "shapelet_concentration": concentration,
            "shapelet_response": response,
            "sorted_anchor_profile": profile,
            "shape_stats_feature": stats_tokens.mean(dim=1),
            "shape_class_token": (
                self.response_to_query(response) if include_legacy_query else None
            ),
            "shape_scales": scales,
            "exposed_curve": exposed,
            "exposed_grid": grid,
        }
        if self.shape_representation == "residual_response":
            result.update({
                "shapelet_context_score": response_details["contextual_similarity"],
                "shapelet_modulation": response_details["modulation"],
            })
        if phase_moments is not None:
            result["shapelet_phase_moments"] = phase_moments
        return result

    def forward(
        self, features, positions, include_legacy_query=True,
        temporal_shift=0, phase_shift=0,
    ):
        return self.forward_from_context(
            self.prepare_context(features, positions),
            structure_aug_shift=temporal_shift,
            phase_shift=phase_shift,
            include_legacy_query=include_legacy_query,
        )
