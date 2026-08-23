"""Timestamp-based patching and recursive temporal fusion for ReIMTS."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from models.reimts.mtan_encoder import ReIMTSMTANEncoders


@dataclass(frozen=True)
class PeriodPatchBatch:
    """A padded batch of timestamp-defined period patches."""

    features: Tensor
    split_positions: Tensor
    encoder_positions: Tensor
    valid: Tensor
    patch_ids: Tensor


@dataclass(frozen=True)
class RecursiveScaleOutput:
    e: Tensor
    h: Optional[Tensor]
    alpha: Optional[Tensor]
    g: Tensor
    valid: Tensor


@dataclass(frozen=True)
class RecursiveTemporalOutput:
    lowest: Tensor
    lowest_valid: Tensor
    scales: Tuple[RecursiveScaleOutput, ...]


def period_patch_ids(positions: Tensor, patches: int, period: int) -> Tensor:
    """Assign each timestamp to exactly one equal-width period patch."""
    if patches < 1:
        raise ValueError("patches must be positive")
    if period < 1:
        raise ValueError("period must be positive")
    return torch.div(
        positions.clamp(min=0, max=period - 1) * patches,
        period,
        rounding_mode="floor",
    ).long()


def gather_period_patches(
    features: Tensor,
    split_positions: Tensor,
    encoder_positions: Tensor,
    patches: int,
    period: int,
) -> PeriodPatchBatch:
    """Gather observations by raw-time membership and retain encoder time."""
    if features.ndim != 3:
        raise ValueError("features must have shape [B,T,D]")
    if split_positions.shape != features.shape[:2]:
        raise ValueError("split_positions must have shape [B,T]")
    if encoder_positions.shape != split_positions.shape:
        raise ValueError("encoder_positions must match split_positions")

    batch_size, _, feature_dim = features.shape
    patch_ids = period_patch_ids(split_positions, patches, period)
    counts = torch.stack(
        [(patch_ids == patch).sum(dim=1) for patch in range(patches)], dim=1
    )
    padded_length = max(1, int(counts.max().item()))
    gathered_features = features.new_zeros(
        batch_size, patches, padded_length, feature_dim
    )
    gathered_split = split_positions.new_zeros(batch_size, patches, padded_length)
    gathered_encoder = encoder_positions.new_zeros(
        batch_size, patches, padded_length
    )
    valid = torch.zeros(
        batch_size,
        patches,
        padded_length,
        dtype=torch.bool,
        device=features.device,
    )

    for batch_index in range(batch_size):
        for patch_index in range(patches):
            selected = patch_ids[batch_index] == patch_index
            length = int(selected.sum().item())
            if length == 0:
                continue
            gathered_features[batch_index, patch_index, :length] = features[
                batch_index, selected
            ]
            gathered_split[batch_index, patch_index, :length] = split_positions[
                batch_index, selected
            ]
            gathered_encoder[batch_index, patch_index, :length] = encoder_positions[
                batch_index, selected
            ]
            valid[batch_index, patch_index, :length] = True

    return PeriodPatchBatch(
        features=gathered_features,
        split_positions=gathered_split,
        encoder_positions=gathered_encoder,
        valid=valid,
        patch_ids=patch_ids,
    )


def split_temporal_representation(
    representation: Tensor, valid: Tensor, factor: int = 2
):
    """Split each parent reference sequence into ordered child sequences."""
    if representation.ndim != 4:
        raise ValueError("representation must have shape [B,P,R,D]")
    if valid.shape != representation.shape[:3]:
        raise ValueError("valid must have shape [B,P,R]")
    if factor < 1 or representation.shape[2] % factor:
        raise ValueError("reference count must be divisible by factor")
    batch_size, parents, ref_points, latent_dim = representation.shape
    child_refs = ref_points // factor
    children = representation.reshape(
        batch_size, parents, factor, child_refs, latent_dim
    ).reshape(batch_size, parents * factor, child_refs, latent_dim)
    child_valid = valid.reshape(
        batch_size, parents, factor, child_refs
    ).reshape(batch_size, parents * factor, child_refs)
    return children, child_valid


class IARFFusion(nn.Module):
    """Temporal-only Irregularity-Aware Representation Fusion (IARF)."""

    def __init__(self, latent_dim, parent_ref_points, ref_points):
        super().__init__()
        self.temporal_mapping = nn.Parameter(
            1 - 2 * torch.rand(parent_ref_points, ref_points)
        )
        self.feed_forward = nn.Linear(latent_dim, latent_dim)

    def forward(self, e: Tensor, parent_h: Tensor, parent_valid: Tensor):
        if parent_h.shape[:2] != parent_valid.shape:
            raise ValueError("parent_valid must match parent representation")
        masked_parent = parent_h * parent_valid.unsqueeze(-1).to(parent_h.dtype)
        aligned_h = (
            masked_parent.transpose(1, 2) @ self.temporal_mapping
        ).transpose(1, 2)
        if aligned_h.shape != e.shape:
            raise ValueError("aligned H and E must have identical shapes")
        patch_valid = parent_valid.any(dim=1, keepdim=True).unsqueeze(-1)
        aligned_h = aligned_h * patch_valid.to(aligned_h.dtype)
        alpha = torch.relu(self.feed_forward(aligned_h))
        alpha = alpha * patch_valid.to(alpha.dtype)
        # PyOmniTS ReIMTS+mTAN uses subtractive global-to-local correction:
        #     G = E - alpha * H
        # whereas Eq. (6) in the ICLR 2026 paper is printed with '+'. We
        # follow the released implementation for reproducibility.
        g = e - alpha * aligned_h
        return alpha, g, aligned_h


class RecursiveTemporalEncoder(nn.Module):
    """Three-level timestamp-split ReIMTS+mTAN temporal encoder."""

    def __init__(
        self,
        input_dim=128,
        latent_dim=128,
        levels=3,
        scale_factor=2,
        period=365,
        num_ref_points=8,
        num_heads=1,
    ):
        super().__init__()
        if levels != 3 or scale_factor != 2:
            raise ValueError("the first implementation supports only 3 levels and factor 2")
        if num_ref_points % scale_factor:
            raise ValueError("num_ref_points must be divisible by scale_factor")
        self.levels = levels
        self.scale_factor = scale_factor
        self.period = period
        self.num_ref_points = num_ref_points
        self.mtan = ReIMTSMTANEncoders(
            levels=levels,
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_ref_points=num_ref_points,
            num_heads=num_heads,
            period=period,
        )
        self.fusions = nn.ModuleList(
            [
                IARFFusion(
                    latent_dim=latent_dim,
                    parent_ref_points=num_ref_points // scale_factor,
                    ref_points=num_ref_points,
                )
                for _ in range(levels - 1)
            ]
        )

    @property
    def scale_encoders(self):
        return self.mtan.scale_encoders

    @property
    def reference_points(self):
        return self.mtan.reference_points

    def forward(
        self,
        spatial_features: Tensor,
        split_positions: Tensor,
        encoder_positions: Tensor,
    ) -> RecursiveTemporalOutput:
        batch_size = spatial_features.shape[0]
        scale_outputs = []
        previous_g = None
        previous_valid = None

        for level, mtan_encoder in enumerate(self.scale_encoders):
            patches = self.scale_factor**level
            gathered = gather_period_patches(
                spatial_features,
                split_positions,
                encoder_positions,
                patches=patches,
                period=self.period,
            )
            _, _, padded_length, feature_dim = gathered.features.shape
            flat_features = gathered.features.reshape(
                batch_size * patches, padded_length, feature_dim
            )
            flat_positions = gathered.encoder_positions.reshape(
                batch_size * patches, padded_length
            )
            flat_valid = gathered.valid.reshape(
                batch_size * patches, padded_length
            )
            e = mtan_encoder(flat_features, flat_positions, flat_valid).reshape(
                batch_size, patches, self.num_ref_points, -1
            )
            # mTAN already consumes observation timestamps and validity. This
            # is only a patch-nonempty flag, not reference-level missingness.
            patch_nonempty = gathered.valid.any(dim=-1)
            reference_valid = patch_nonempty.unsqueeze(-1).expand(
                -1, -1, self.num_ref_points
            )
            e = e * reference_valid.unsqueeze(-1).to(e.dtype)

            if previous_g is None:
                h = None
                alpha = None
                g = e
            else:
                split_h, split_valid = split_temporal_representation(
                    previous_g, previous_valid, factor=self.scale_factor
                )
                flat_e = e.reshape(batch_size * patches, self.num_ref_points, -1)
                flat_h = split_h.reshape(
                    batch_size * patches, split_h.shape[2], split_h.shape[3]
                )
                flat_h_valid = split_valid.reshape(
                    batch_size * patches, split_valid.shape[2]
                )
                flat_alpha, flat_g, flat_aligned_h = self.fusions[level - 1](
                    flat_e, flat_h, flat_h_valid
                )
                alpha = flat_alpha.reshape_as(e)
                g = flat_g.reshape_as(e)
                h = flat_aligned_h.reshape_as(e)
                g = g * reference_valid.unsqueeze(-1).to(g.dtype)

            scale_outputs.append(
                RecursiveScaleOutput(
                    e=e, h=h, alpha=alpha, g=g, valid=reference_valid
                )
            )
            previous_g = g
            previous_valid = reference_valid

        return RecursiveTemporalOutput(
            lowest=scale_outputs[-1].g,
            lowest_valid=scale_outputs[-1].valid,
            scales=tuple(scale_outputs),
        )
