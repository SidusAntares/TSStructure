"""
Pixel-Set encoder module

author: Vivien Sainte Fare Garnot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from models.layers import LinearLayer


class PixelSetEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp1=[10, 32, 64],
        pooling="mean_std",
        mlp2=[64, 128],
        with_extra=True,
        extra_size=4,
    ):
        """
        Pixel-set encoder.
        Args:
            input_dim (int): Number of channels of the input tensors
            mlp1 (list):  Dimensions of the successive feature spaces of MLP1
            pooling (str): Pixel-embedding pooling strategy, can be chosen in ('mean','std','max,'min')
                or any underscore-separated combination thereof.
            mlp2 (list): Dimensions of the successive feature spaces of MLP2
            with_extra (bool): Whether additional pre-computed features are passed between the two MLPs
            extra_size (int, optional): Number of channels of the additional features, if any.
        """

        super(PixelSetEncoder, self).__init__()

        self.input_dim = input_dim
        self.mlp1_dim = copy.deepcopy(mlp1)
        self.mlp2_dim = copy.deepcopy(mlp2)
        self.pooling = pooling

        self.with_extra = with_extra
        self.extra_size = extra_size

        self.output_dim = (
            input_dim * len(pooling.split("_"))
            if len(self.mlp2_dim) == 0
            else self.mlp2_dim[-1]
        )

        inter_dim = self.mlp1_dim[-1] * len(pooling.split("_"))
        if self.with_extra:
            inter_dim += self.extra_size

        assert input_dim == mlp1[0]
        assert inter_dim == mlp2[0]
        # Feature extraction
        layers = []
        for i in range(len(self.mlp1_dim) - 1):
            layers.append(LinearLayer(self.mlp1_dim[i], self.mlp1_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        # MLP after pooling
        layers = []
        for i in range(len(self.mlp2_dim) - 1):
            layers.append(LinearLayer(self.mlp2_dim[i], self.mlp2_dim[i + 1]))
        self.mlp2 = nn.Sequential(*layers)

    def forward(self, pixels, mask, extra):
        """
        The input of the PSE is a tuple of tensors as yielded by the PixelSetData class:
          (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
        Pixel-Set : Batch_size x (Sequence length) x Channel x Number of pixels
        Pixel-Mask : Batch_size x (Sequence length) x Number of pixels
        Extra-features : Batch_size x (Sequence length) x Number of features

        If the input tensors have a temporal dimension, it will be combined with the batch dimension so that the
        complete sequences are processed at once. Then the temporal dimension is separated back to produce a tensor of
        shape Batch_size x Sequence length x Embedding dimension
        """
        out = pixels

        batch, temp = out.shape[:2]

        out = out.view(batch * temp, *out.shape[2:]).transpose(1, 2)  # (B*T, S, C)
        mask = mask.view(batch * temp, -1)

        out = self.mlp1(out).transpose(1, 2)
        out = torch.cat(
            [pooling_methods[n](out, mask) for n in self.pooling.split("_")], dim=1
        )

        if self.with_extra:
            extra = extra.unsqueeze(1).repeat(1, temp, 1)
            extra = extra.view(batch * temp, -1)
            out = torch.cat([out, extra], dim=1)
        out = self.mlp2(out)
        out = out.view(batch, temp, -1)
        return out


class _TokenMLP(nn.Module):
    """Apply an MLP independently on the final feature dimension."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, tokens):
        return self.layers(tokens)


def _masked_mean_and_sample_std(tokens, valid_pixels, eps=1e-8):
    """Pool ``[B,L,S,C,D]`` tokens over only the pixel-set axis ``S``."""

    mask = valid_pixels.unsqueeze(-1).unsqueeze(-1)
    count = mask.sum(dim=2, keepdim=True)
    mean = (tokens * mask).sum(dim=2, keepdim=True) / count

    centered = (tokens - mean) * mask
    variance = centered.square().sum(dim=2, keepdim=True) / (count - 1).clamp_min(1)
    std = torch.where(
        count > 1,
        torch.sqrt(variance + eps),
        torch.zeros_like(variance),
    )
    return mean.squeeze(2), std.squeeze(2)


class ChannelPreservingPixelSetEncoder(nn.Module):
    """Encode each physical variable's parcel pixel set independently.

    Inputs are pixel values with shape ``[B,L,C,S]`` and a valid-pixel mask
    with shape ``[B,L,S]``. The output token ``Z[b,t,c,:]`` is the sum of a
    shared Pixel-Set Encoder applied only to variable ``c`` and the learnable
    identity vector for variable ``c``. The output has shape ``[B,L,C,p]``:
    ``C`` remains the fixed original-variable axis, while ``p`` is each
    variable's internal feature dimension. Pooling acts only over the unordered
    within-parcel pixel-set axis ``S``; pixels are not independent samples.
    Cross-variable relations are left to later explicit channel-structure
    operators, and this encoder performs no channel mixing.
    """

    def __init__(
        self,
        num_channels: int,
        channel_feature_dim: int = 16,
        pixel_hidden_dim: int = 16,
        eps: float = 1e-8,
    ):
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        for name, value in (
            ("channel_feature_dim", channel_feature_dim),
            ("pixel_hidden_dim", pixel_hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.num_channels = num_channels
        self.channel_feature_dim = channel_feature_dim
        self.output_dim = channel_feature_dim
        self.eps = eps

        self.pixel_value_encoder = _TokenMLP(1, pixel_hidden_dim)
        self.post_pool_encoder = _TokenMLP(
            2 * pixel_hidden_dim, self.channel_feature_dim
        )
        self.channel_embedding = nn.Embedding(
            self.num_channels, self.channel_feature_dim
        )
        nn.init.normal_(self.channel_embedding.weight, mean=0.0, std=0.02)

    def _validate_inputs(self, pixels, valid_pixels):
        if pixels.ndim != 4:
            raise ValueError("pixels must be 4D with shape [B,L,C,S]")
        if valid_pixels.ndim != 3:
            raise ValueError("valid_pixels must be 3D with shape [B,L,S]")
        if not torch.is_floating_point(pixels):
            raise TypeError("pixels must be a floating point tensor")
        if valid_pixels.dtype != torch.bool and not torch.is_floating_point(
            valid_pixels
        ):
            raise TypeError("valid_pixels must be a boolean or floating point tensor")

        batch, time, channels, num_pixels = pixels.shape
        if valid_pixels.shape != (batch, time, num_pixels):
            raise ValueError(
                "pixels and valid_pixels batch, time, and pixel dimensions must match"
            )
        if channels != self.num_channels:
            raise ValueError(
                f"pixels channel dimension must equal num_channels={self.num_channels}"
            )

        if valid_pixels.dtype == torch.bool:
            mask_bool = valid_pixels
        else:
            if not torch.isfinite(valid_pixels).all().item() or not torch.all(
                (valid_pixels == 0) | (valid_pixels == 1)
            ).item():
                raise ValueError("valid_pixels must contain only finite 0/1 values")
            mask_bool = valid_pixels != 0

        mask_bool = mask_bool.to(device=pixels.device)
        if torch.any(mask_bool.sum(dim=-1) == 0):
            raise ValueError("each [B,L] date must contain at least one valid pixel")
        return mask_bool, mask_bool.to(dtype=pixels.dtype)

    def forward(
        self,
        pixels: torch.Tensor,
        valid_pixels: torch.Tensor,
    ) -> torch.Tensor:
        """Return channel-preserving parcel tokens with shape ``[B,L,C,p]``."""

        mask_bool, mask = self._validate_inputs(pixels, valid_pixels)
        channels = pixels.shape[2]
        values = pixels.permute(0, 1, 3, 2).unsqueeze(-1)
        if not torch.isfinite(values[mask_bool]).all().item():
            raise ValueError("valid pixel values must be finite")
        values = torch.where(
            mask_bool.unsqueeze(-1).unsqueeze(-1),
            values,
            torch.zeros_like(values),
        )

        pixel_features = self.pixel_value_encoder(values)
        mean, std = _masked_mean_and_sample_std(pixel_features, mask, self.eps)
        base_tokens = self.post_pool_encoder(torch.cat([mean, std], dim=-1))
        channel_indices = torch.arange(channels, device=pixels.device)
        channel_identity = self.channel_embedding(channel_indices).reshape(
            1, 1, channels, self.channel_feature_dim
        )
        return base_tokens + channel_identity


def masked_mean(x, mask):
    out = x.permute((1, 0, 2))
    out = out * mask
    out = out.sum(dim=-1) / mask.sum(dim=-1)
    out = out.permute((1, 0))
    return out

def masked_std(x, mask):
    m = masked_mean(x, mask)

    out = x.permute((2, 0, 1))
    out = out - m
    out = out.permute((2, 1, 0))

    out = out * mask
    d = mask.sum(dim=-1)
    d[d == 1] = 2

    out = (out ** 2).sum(dim=-1) / (d - 1)
    out = torch.sqrt(out + 10e-32) # To ensure differentiability
    out = out.permute(1, 0)
    return out

def maximum(x, mask):
    return x.max(dim=-1)[0].squeeze()

def minimum(x, mask):
    return x.min(dim=-1)[0].squeeze()


pooling_methods = {
    "mean": masked_mean,
    "std": masked_std,
    "max": maximum,
    "min": minimum,
}
