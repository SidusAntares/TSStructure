"""PSE → ReIMTS+mTAN → shared LTAE → shared classifier."""

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.decoder import get_decoder
from models.ltae import LTAE
from models.pse import PixelSetEncoder
from models.reimts.recursive_temporal import RecursiveTemporalEncoder


@dataclass(frozen=True)
class ReIMTSClassificationOutput:
    """Training output; inference continues to expose only sample logits."""

    sample_logits: torch.Tensor
    patch_logits: torch.Tensor
    patch_valid: torch.Tensor


class PatchOccupancyMeter:
    """Aggregate lowest-scale patch occupancy across representative batches."""

    def __init__(self):
        self.samples = 0
        self.valid_patches = 0
        self.patch_slots = 0
        self.empty_by_patch = None

    def update(self, patch_valid):
        valid = patch_valid.detach().bool().cpu()
        if valid.ndim != 2:
            raise ValueError("patch_valid must have shape [B, P]")
        if self.empty_by_patch is None:
            self.empty_by_patch = torch.zeros(valid.shape[1], dtype=torch.long)
        elif self.empty_by_patch.numel() != valid.shape[1]:
            raise ValueError("patch count changed while aggregating occupancy")
        self.samples += valid.shape[0]
        self.valid_patches += int(valid.sum().item())
        self.patch_slots += valid.numel()
        self.empty_by_patch += (~valid).sum(dim=0)

    def summary(self):
        patch_count = 0 if self.empty_by_patch is None else self.empty_by_patch.numel()
        return {
            "samples": self.samples,
            "patches": patch_count,
            "valid_per_sample_mean": (
                self.valid_patches / self.samples if self.samples else 0.0
            ),
            "empty_patch_rate": (
                1.0 - self.valid_patches / self.patch_slots
                if self.patch_slots
                else 0.0
            ),
            "quarter_empty_rates": (
                (self.empty_by_patch.float() / self.samples).tolist()
                if self.samples
                else [0.0] * patch_count
            ),
        }

    def format_lines(self, title="ReIMTS patches:"):
        summary = self.summary()
        lines = [
            title,
            "  valid patches/sample mean: "
            f"{summary['valid_per_sample_mean']:.3f}",
            f"  empty patch rate: {summary['empty_patch_rate']:.2%}",
        ]
        lines.extend(
            f"  Q{index} empty rate: {rate:.2%}"
            for index, rate in enumerate(summary["quarter_empty_rates"], 1)
        )
        return lines


def format_patch_diagnostics(patch_valid, label):
    """Format lowest-scale patch occupancy without changing aggregation."""
    valid = patch_valid.detach().bool().cpu()
    if valid.ndim != 2:
        raise ValueError("patch_valid must have shape [B, P]")
    batch_size, patches = valid.shape
    counts = valid.sum(dim=1)
    lines = [
        f"[ReIMTS patch diagnostics] {label}",
        f"lowest scale = {patches} patches",
        "valid patch count per sample:",
    ]
    for count in range(patches + 1):
        samples = int((counts == count).sum().item())
        ratio = samples / batch_size if batch_size else 0.0
        lines.append(f"  {count}: {samples} ({ratio:.2%})")
    empty = ~valid
    empty_rate = float(empty.float().mean().item()) if valid.numel() else 0.0
    lines.append(f"empty patch rate: {empty_rate:.2%}")
    for patch in range(patches):
        rate = (
            float(empty[:, patch].float().mean().item())
            if batch_size
            else 0.0
        )
        lines.append(f"Q{patch + 1} empty rate: {rate:.2%}")
    return "\n".join(lines)


def reimts_classification_loss(output, targets, criterion, mode="patch"):
    """Compute patch-direct or Round 1 sample-level classification loss."""
    if mode == "sample":
        return criterion(output.sample_logits, targets)
    if mode != "patch":
        raise ValueError("reimts loss mode must be 'patch' or 'sample'")
    repeated_targets = targets.unsqueeze(1).expand(
        -1, output.patch_logits.shape[1]
    )
    if not output.patch_valid.any():
        raise ValueError("patch loss requires at least one non-empty patch")
    return criterion(
        output.patch_logits[output.patch_valid],
        repeated_targets[output.patch_valid],
    )


class PseReIMTSMTANLTAE(nn.Module):
    """Three-level ReIMTS+mTAN classifier for TimeMatch SITS batches."""

    def __init__(
        self,
        input_dim=10,
        num_classes=20,
        with_extra=True,
        extra_size=4,
        mlp1=None,
        pooling="mean_std",
        mlp2=None,
        latent_dim=128,
        num_ref_points=8,
        mtan_heads=1,
        reimts_levels=3,
        reimts_scale_factor=2,
        reimts_period=365,
        ltae_heads=16,
        ltae_key_dim=8,
        ltae_model_dim=256,
        ltae_mlp=None,
        classifier_mlp=None,
        dropout=0.2,
        max_temporal_shift=100,
    ):
        super().__init__()
        mlp1 = [input_dim, 32, 64] if mlp1 is None else deepcopy(mlp1)
        mlp2 = [128 + (extra_size if with_extra else 0), 128] if mlp2 is None else deepcopy(mlp2)
        ltae_mlp = [ltae_model_dim, 128] if ltae_mlp is None else deepcopy(ltae_mlp)
        classifier_mlp = [ltae_mlp[-1], 64, 32] if classifier_mlp is None else deepcopy(classifier_mlp)
        self.reimts_period = reimts_period
        self.max_temporal_shift = max_temporal_shift
        self.spatial_encoder = PixelSetEncoder(
            input_dim=input_dim,
            mlp1=mlp1,
            pooling=pooling,
            mlp2=mlp2,
            with_extra=with_extra,
            extra_size=extra_size,
        )
        self.reimts_encoder = RecursiveTemporalEncoder(
            input_dim=mlp2[-1],
            latent_dim=latent_dim,
            levels=reimts_levels,
            scale_factor=reimts_scale_factor,
            period=reimts_period,
            num_ref_points=num_ref_points,
            num_heads=mtan_heads,
        )
        self.temporal_decoder = LTAE(
            in_channels=latent_dim,
            n_head=ltae_heads,
            d_k=ltae_key_dim,
            d_model=ltae_model_dim,
            n_neurons=ltae_mlp,
            dropout=dropout,
            max_temporal_shift=max_temporal_shift,
            max_position=reimts_period,
        )
        self.classifier = get_decoder(classifier_mlp, num_classes)
        decoder_positions = torch.linspace(
            0, reimts_period - 1, num_ref_points
        ).round().long()
        self.register_buffer(
            "decoder_positions", decoder_positions, persistent=True
        )

    def _decode_lowest(self, lowest, lowest_valid):
        batch_size, patches, ref_points, latent_dim = lowest.shape
        flat_lowest = lowest.reshape(
            batch_size * patches, ref_points, latent_dim
        )
        positions = self.decoder_positions.unsqueeze(0).expand(
            batch_size * patches, -1
        )
        flat_features = self.temporal_decoder(flat_lowest, positions)
        patch_features = flat_features.reshape(batch_size, patches, -1)
        patch_logits = self.classifier(
            patch_features.reshape(batch_size * patches, -1)
        ).reshape(batch_size, patches, -1)
        sample_logits = patch_logits.mean(dim=1)
        return sample_logits, patch_features, patch_logits

    def _forward_from_spatial_output(self, spatial_feats, positions, shift=0):
        split_positions = positions
        encoder_positions = positions + shift
        recursive = self.reimts_encoder(
            spatial_feats,
            split_positions=split_positions,
            encoder_positions=encoder_positions,
        )
        sample_logits, patch_features, patch_logits = self._decode_lowest(
            recursive.lowest, recursive.lowest_valid
        )
        patch_valid = recursive.lowest_valid.any(dim=-1)
        output = ReIMTSClassificationOutput(
            sample_logits=sample_logits,
            patch_logits=patch_logits,
            patch_valid=patch_valid,
        )
        return output, patch_features

    def forward_from_spatial_for_loss_with_shift(
        self, spatial_feats, positions, shift=0
    ):
        output, _ = self._forward_from_spatial_output(
            spatial_feats, positions, shift
        )
        return output

    def forward_from_spatial_with_shift(
        self, spatial_feats, positions, shift=0, return_feats=False
    ):
        output, patch_features = self._forward_from_spatial_output(
            spatial_feats, positions, shift
        )
        if return_feats:
            # Unweighted mean of lowest-scale patch features. Empty-patch
            # policy will follow sample aggregation after real-data smoke.
            return output.sample_logits, patch_features.mean(dim=1)
        return output.sample_logits

    def forward_for_loss(
        self, pixels, mask, positions, extra, shift=0
    ):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        return self.forward_from_spatial_for_loss_with_shift(
            spatial_feats, positions, shift
        )

    def forward_with_shift(
        self, pixels, mask, positions, extra, shift, return_feats=False
    ):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        return self.forward_from_spatial_with_shift(
            spatial_feats,
            positions,
            shift=shift,
            return_feats=return_feats,
        )

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        return self.forward_with_shift(
            pixels,
            mask,
            positions,
            extra,
            shift=0,
            return_feats=return_feats,
        )
