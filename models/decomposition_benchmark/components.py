from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

from models.decoder import get_decoder
from models.ltae import LTAE
from models.pse import PixelSetEncoder


@dataclass
class TemporalComponentBatch:
    """A decomposition component that still has an explicit time axis."""

    name: str
    pixels: torch.Tensor
    mask: torch.Tensor
    positions: torch.Tensor


class ChannelStatsAccumulator:
    """Streaming masked channel statistics for [B, T, C, S] tensors."""

    def __init__(self, num_channels: int):
        self.num_channels = int(num_channels)
        self.sum = torch.zeros(self.num_channels, dtype=torch.float64)
        self.sumsq = torch.zeros(self.num_channels, dtype=torch.float64)
        self.count = torch.zeros(self.num_channels, dtype=torch.float64)

    @torch.no_grad()
    def update(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        if values.ndim != 4:
            raise ValueError("values must have shape [B, T, C, S]")
        if mask.ndim != 3:
            raise ValueError("mask must have shape [B, T, S]")
        if values.shape[0] != mask.shape[0] or values.shape[1] != mask.shape[1] or values.shape[3] != mask.shape[2]:
            raise ValueError("mask shape is incompatible with values")
        if values.shape[2] != self.num_channels:
            raise ValueError("unexpected number of channels")

        weights = mask.to(dtype=values.dtype).unsqueeze(2)
        weighted = values * weights
        self.sum += weighted.sum(dim=(0, 1, 3)).detach().cpu().double()
        self.sumsq += (values.square() * weights).sum(dim=(0, 1, 3)).detach().cpu().double()
        count = weights.sum(dim=(0, 1, 3)).detach().cpu().double()
        self.count += count.expand(self.num_channels)

    def finalize(self, eps: float = 1e-6):
        if torch.any(self.count <= 0):
            raise RuntimeError("cannot finalize component statistics with zero valid observations")
        mean = self.sum / self.count
        var = self.sumsq / self.count - mean.square()
        std = torch.sqrt(torch.clamp(var, min=eps * eps))
        return mean.float(), std.float()


class FixedChannelStandardizer(nn.Module):
    """Fixed per-channel standardization fitted from the supervised training split."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(self.num_channels))
        self.register_buffer("std", torch.ones(self.num_channels))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != self.num_channels or std.numel() != self.num_channels:
            raise ValueError("component statistics have the wrong channel dimension")
        self.mean.copy_(mean.reshape_as(self.mean).to(device=self.mean.device, dtype=self.mean.dtype))
        self.std.copy_(std.reshape_as(self.std).to(device=self.std.device, dtype=self.std.dtype).clamp_min(self.eps))
        self.fitted.fill_(True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError(
                "component normalizer has not been fitted; source training must fit component statistics before forward"
            )
        if values.ndim != 4 or values.shape[2] != self.num_channels:
            raise ValueError("expected component tensor with shape [B, T, C, S]")
        mean = self.mean.view(1, 1, -1, 1)
        std = self.std.view(1, 1, -1, 1)
        return (values - mean) / std


class AdditiveLogitFusion(nn.Module):
    """DLinear-style output-level addition of component logits."""

    def forward(self, logits: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(logits) == 0:
            raise ValueError("at least one component logit tensor is required")
        out = logits[0]
        for value in logits[1:]:
            out = out + value
        return out


def component_normalization_dataset_name(model, source_name: str, training_name: str) -> str:
    """Keep benchmark component statistics source-only, including target upper bounds."""
    if hasattr(model, "fit_component_normalizers"):
        return source_name
    return training_name


class RawComponentPseLtaeClassifier(nn.Module):
    """
    Shared-PSE / component-specific-LTAE classifier for raw decomposition components.

    The decomposer must return a fixed set of TemporalComponentBatch objects whose
    names match ``component_names``. Component normalization is fitted once on the
    supervised training split and stored in the checkpoint. TimeMatch adaptation
    therefore reuses source-fitted statistics instead of updating them on target data.
    """

    def __init__(
        self,
        decomposer: nn.Module,
        component_names: Sequence[str],
        input_dim: int = 10,
        mlp1: Sequence[int] = (10, 32, 64),
        pooling: str = "mean_std",
        mlp2: Sequence[int] = (128, 128),
        with_extra: bool = True,
        extra_size: int = 4,
        n_head: int = 16,
        d_k: int = 8,
        d_model: int = 256,
        mlp3: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        T: int = 1000,
        mlp4: Sequence[int] = (128, 64, 32),
        num_classes: int = 20,
        max_temporal_shift: int = 100,
        fusion_mode: str = "additive_logits",
    ):
        super().__init__()
        self.decomposer = decomposer
        self.component_names = tuple(component_names)
        if len(self.component_names) == 0:
            raise ValueError("component_names must not be empty")
        if len(set(self.component_names)) != len(self.component_names):
            raise ValueError("component_names must be unique")

        mlp1 = list(mlp1)
        mlp2 = list(mlp2)
        mlp3 = list(mlp3)
        mlp4 = list(mlp4)
        if mlp1[0] != input_dim:
            raise ValueError("mlp1[0] must equal input_dim")

        pse_mlp2 = deepcopy(mlp2)
        if with_extra:
            pse_mlp2[0] += extra_size

        self.spatial_encoder = PixelSetEncoder(
            input_dim=input_dim,
            mlp1=mlp1,
            pooling=pooling,
            mlp2=pse_mlp2,
            with_extra=with_extra,
            extra_size=extra_size,
        )
        self.component_normalizers = nn.ModuleDict(
            {
                name: FixedChannelStandardizer(input_dim)
                for name in self.component_names
            }
        )
        self.temporal_encoders = nn.ModuleDict(
            {
                name: LTAE(
                    in_channels=mlp2[-1],
                    n_head=n_head,
                    d_k=d_k,
                    d_model=d_model,
                    n_neurons=mlp3,
                    dropout=dropout,
                    T=T,
                    max_temporal_shift=max_temporal_shift,
                )
                for name in self.component_names
            }
        )
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in ("additive_logits", "concat_embeddings"):
            raise ValueError(
                "fusion_mode must be 'additive_logits' or 'concat_embeddings'"
            )

        if self.fusion_mode == "additive_logits":
            self.decoders = nn.ModuleDict(
                {
                    name: get_decoder(mlp4, num_classes)
                    for name in self.component_names
                }
            )
            self.final_decoder = None
            self.fusion = AdditiveLogitFusion()
        else:
            self.decoders = nn.ModuleDict()
            concat_dim = len(self.component_names) * mlp3[-1]
            final_mlp = [concat_dim] + mlp4[1:]
            self.final_decoder = get_decoder(final_mlp, num_classes)
            self.fusion = None

    @property
    def temporal_encoder(self):
        """Compatibility view used by TimeMatch temporal-index range checks."""
        return self.temporal_encoders[self.component_names[0]]

    @property
    def component_normalizers_fitted(self) -> bool:
        return all(bool(self.component_normalizers[name].fitted.item()) for name in self.component_names)

    def _decompose(self, pixels, mask, positions) -> List[TemporalComponentBatch]:
        components = list(self.decomposer(pixels, mask, positions))
        names = [component.name for component in components]
        if tuple(names) != self.component_names:
            raise RuntimeError(
                "decomposer returned unexpected components: "
                f"expected {self.component_names}, got {tuple(names)}"
            )
        return components

    @torch.no_grad()
    def fit_component_normalizers(self, data_loader: Iterable[Dict[str, torch.Tensor]], device) -> None:
        accumulators = {
            name: ChannelStatsAccumulator(self.component_normalizers[name].num_channels)
            for name in self.component_names
        }
        was_training = self.training
        self.eval()
        for sample in data_loader:
            pixels = sample["pixels"].to(device=device, non_blocking=True)
            mask = sample["valid_pixels"].to(device=device, non_blocking=True)
            positions = sample["positions"].to(device=device, non_blocking=True)
            for component in self._decompose(pixels, mask, positions):
                accumulators[component.name].update(component.pixels, component.mask)

        for name in self.component_names:
            mean, std = accumulators[name].finalize(self.component_normalizers[name].eps)
            self.component_normalizers[name].set_stats(mean, std)
        self.train(was_training)

    def _encode_components(self, pixels, mask, positions, extra):
        if not self.component_normalizers_fitted:
            raise RuntimeError(
                "component normalizers are not fitted; train the source model first or load a source checkpoint"
            )
        embeddings = {}
        spatial_features = {}
        components = self._decompose(pixels, mask, positions)
        for component in components:
            normalized = self.component_normalizers[component.name](component.pixels)
            spatial = self.spatial_encoder(normalized, component.mask, extra)
            temporal = self.temporal_encoders[component.name](spatial, component.positions)
            spatial_features[component.name] = (spatial, component.positions)
            embeddings[component.name] = temporal
        return embeddings, spatial_features

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        embeddings, _ = self._encode_components(pixels, mask, positions, extra)
        fused, features = self._classify_embeddings(embeddings)
        if return_feats:
            return fused, features
        return fused

    def _classify_embeddings(self, embeddings):
        features = torch.cat(
            [embeddings[name] for name in self.component_names], dim=1
        )
        if self.fusion_mode == "additive_logits":
            logits = [
                self.decoders[name](embeddings[name])
                for name in self.component_names
            ]
            return self.fusion(logits), features
        return self.final_decoder(features), features

    def forward_shift_candidates(self, pixels, mask, positions, extra, shifts):
        """
        Compute TimeMatch shift candidates without assuming one PSE/LTAE/decoder chain.

        Decomposition, normalization and the shared PSE are evaluated once per
        component; only the temporal encoder and classifier are rerun for each shift.
        """
        if not self.component_normalizers_fitted:
            raise RuntimeError(
                "component normalizers are not fitted; load the source-trained checkpoint before TimeMatch adaptation"
            )
        components = self._decompose(pixels, mask, positions)
        branch_inputs = {}
        for component in components:
            normalized = self.component_normalizers[component.name](component.pixels)
            spatial = self.spatial_encoder(normalized, component.mask, extra)
            branch_inputs[component.name] = (spatial, component.positions)

        all_shift_logits = []
        for shift in shifts:
            embeddings = {}
            for name in self.component_names:
                spatial, component_positions = branch_inputs[name]
                embeddings[name] = self.temporal_encoders[name](
                    spatial, component_positions + int(shift)
                )
            fused, _ = self._classify_embeddings(embeddings)
            all_shift_logits.append(fused)
        return torch.stack(all_shift_logits, dim=1)
