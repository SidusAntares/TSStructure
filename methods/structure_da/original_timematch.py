"""Original TimeMatch semantic model and its frozen geometry side branch."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from models.stclassifier import PseLTae
from .decomposition import SymmetricTimeKernelDecomposition


@dataclass(frozen=True)
class OriginalTimeMatchOutput:
    logits: Tensor
    fused_repr: Tensor


@dataclass(frozen=True)
class OriginalTimeMatchBackboneOutput:
    tokens: Tensor
    normalized_positions: Tensor
    time_mask: Tensor


def _continuous_original_encoding(encoder: nn.Module, positions: Tensor, *, period: float = 1000.0) -> Tensor:
    """Evaluate the frozen upstream sinusoidal table formula at float days."""
    dim = int(encoder.positional_enc.embedding_dim)
    table = encoder.positional_enc.weight
    frequency = torch.exp(
        torch.arange(0, dim, 2, device=positions.device, dtype=table.dtype)
        * (-torch.log(torch.tensor(float(period), device=positions.device, dtype=table.dtype)) / dim)
    )
    shifted = positions.to(table.dtype) + float(encoder.max_temporal_shift)
    phase = shifted.unsqueeze(-1) * frequency
    result = torch.empty(*shifted.shape, dim, device=positions.device, dtype=table.dtype)
    result[..., 0::2] = torch.sin(phase)
    result[..., 1::2] = torch.cos(phase)
    return result


class _RawEncoderView:
    def __init__(self, owner: "OriginalTimeMatchModel") -> None:
        self.owner = owner
        self.time_encoder = owner.temporal_encoder.positional_enc

    def __call__(self, *, latent: Tensor, positions: Tensor, mask: Tensor):
        days = positions * self.owner.time_scale
        return SimpleNamespace(fused_repr=self.owner.encode_temporal(latent, days, mask))


class OriginalTimeMatchModel(nn.Module):
    """Compatibility shell whose trainable graph remains PSE -> LTAE -> decoder."""

    def __init__(self, semantic_model: PseLTae, *, time_scale: float = 365.0) -> None:
        super().__init__()
        self.semantic_model = semantic_model
        self.time_scale = float(time_scale)

    @property
    def spatial_encoder(self):
        return self.semantic_model.spatial_encoder

    @property
    def temporal_encoder(self):
        return self.semantic_model.temporal_encoder

    @property
    def decoder(self):
        return self.semantic_model.decoder

    @property
    def classifier(self):
        return self.decoder

    @property
    def backbone(self):
        return SimpleNamespace(time_scale=self.time_scale, feature_dim=128)

    @property
    def temporal_module(self):
        return SimpleNamespace(raw_encoder=_RawEncoderView(self))

    def encode_temporal(self, tokens: Tensor, positions_days: Tensor, time_mask: Tensor | None = None) -> Tensor:
        encoder = self.temporal_encoder
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device) if time_mask is None else time_mask.bool()
        x = torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        if encoder.inconv is not None:
            x = encoder.inconv(x)
        encoded = x + _continuous_original_encoding(encoder, positions_days)
        encoded, _ = encoder.attention_heads(encoded, time_mask=mask)
        encoded = encoder.dropout(encoder.mlp(encoded))
        return torch.where(mask.any(-1, keepdim=True), encoded, torch.zeros_like(encoded))

    def forward_backbone(self, pixels, valid_pixels, positions, extra=None, *, time_mask=None, compute_decomposition=False):
        if compute_decomposition:
            raise AssertionError("Original TimeMatch classifier never invokes decomposition")
        mask = valid_pixels.bool().any(dim=-1) if time_mask is None else time_mask.bool()
        tokens = self.spatial_encoder(pixels, valid_pixels, extra)
        normalized = torch.where(mask, positions.to(tokens.dtype) / self.time_scale, torch.zeros_like(positions, dtype=tokens.dtype))
        return OriginalTimeMatchBackboneOutput(tokens, normalized, mask)

    def forward_from_backbone(self, backbone, positions, extra=None, *, temporal_positions_override=None, return_geometry=False):
        if return_geometry:
            raise AssertionError("Original TimeMatch classifier has no geometry output")
        temporal_positions = backbone.normalized_positions if temporal_positions_override is None else temporal_positions_override
        features = self.encode_temporal(backbone.tokens, temporal_positions * self.time_scale, backbone.time_mask)
        return OriginalTimeMatchOutput(self.decoder(features), features)

    def forward(self, pixels, valid_pixels, positions, extra=None, *, time_mask=None, return_geometry=False):
        backbone = self.forward_backbone(pixels, valid_pixels, positions, extra, time_mask=time_mask)
        return self.forward_from_backbone(backbone, positions, extra, return_geometry=return_geometry)


def build_original_timematch_model(runtime: dict, checkpoint: dict, device: torch.device) -> OriginalTimeMatchModel:
    model = PseLTae(
        input_dim=int(runtime.get("input_dim", 10)),
        num_classes=len(runtime["classes"]),
        with_extra=bool(runtime.get("with_extra", False)),
    )
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    if state is None:
        raise ValueError("Original TimeMatch source checkpoint must contain state_dict")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint is not an Original TimeMatch PseLTae checkpoint") from error
    return OriginalTimeMatchModel(model, time_scale=float(runtime.get("time_scale", 365.0))).to(device)


def module_state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenGeometryCopy(nn.Module):
    """Independent, immutable PSE copy used only for no-grad Phase geometry."""

    def __init__(self, semantic_model: OriginalTimeMatchModel) -> None:
        super().__init__()
        self.geometry_pse = deepcopy(semantic_model.spatial_encoder)
        self.decomposition = SymmetricTimeKernelDecomposition(time_scale=semantic_model.time_scale)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        super().train(False)
        self._initial_hash = module_state_hash(self.geometry_pse)
        self.assert_frozen()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def assert_frozen(self) -> None:
        if self.training or self.geometry_pse.training:
            raise RuntimeError("geometry_pse must remain in eval mode")
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("geometry branch parameters must remain frozen")
        if module_state_hash(self.geometry_pse) != self._initial_hash:
            raise RuntimeError("geometry_pse state changed after Stage2 bootstrap")

    def forward(self, pixels, valid_pixels, positions, extra=None, *, return_geometry=True, **_):
        if not return_geometry:
            raise ValueError("FrozenGeometryCopy only provides geometry")
        self.assert_frozen()
        with torch.inference_mode():
            mask = valid_pixels.bool().any(dim=-1)
            tokens = self.geometry_pse(pixels, valid_pixels, extra)
            decomposition = self.decomposition(tokens, positions, mask)
            normalized = torch.where(mask, positions.to(tokens.dtype) / self.decomposition.time_scale, torch.zeros_like(positions, dtype=tokens.dtype))
        return SimpleNamespace(trend=decomposition.trend, mask=mask, positions=normalized)
