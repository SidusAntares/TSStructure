"""PSE → ReIMTS+mTAN → shared LTAE → shared classifier."""

from copy import deepcopy
from dataclasses import dataclass
import time

import torch
import torch.nn as nn

from models.decoder import get_decoder
from models.ltae import LTAE
from models.pse import PixelSetEncoder
from models.reimts.recursive_temporal import RecursiveTemporalEncoder


@dataclass(frozen=True)
class ReIMTSClassificationOutput:
    """Sample-level training output plus occupancy-only diagnostics."""

    sample_logits: torch.Tensor
    patch_valid: torch.Tensor


@dataclass(frozen=True)
class ReIMTSShiftFeatures:
    """Pre-LTAE whole-sample features cached across TimeMatch candidates."""

    tokens: torch.Tensor
    positions: torch.Tensor
    patch_valid: torch.Tensor
    spatial_encoder_time: float = 0.0
    reimts_mtan_time: float = 0.0

    @property
    def total_feature_preparation_time(self):
        return self.spatial_encoder_time + self.reimts_mtan_time


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


def reimts_classification_loss(output, targets, criterion, mode="sample"):
    """Compute the formal Round 4 sample-level classification loss."""
    if mode != "sample":
        raise ValueError(
            "Round 4 ReIMTS uses sample-level loss; patch mode is incompatible"
        )
    return criterion(output.sample_logits, targets)


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

    def _flatten_lowest(
        self, lowest, lowest_reference_positions, lowest_valid
    ):
        """Concatenate patch blocks into one chronological token sequence."""
        if lowest.ndim != 4:
            raise ValueError("lowest tokens must have shape [B,P,R,D]")
        if lowest_reference_positions.shape != lowest.shape[:3]:
            raise ValueError("reference positions must match lowest tokens")
        if lowest_valid.shape != lowest.shape[:3]:
            raise ValueError("lowest validity must match lowest tokens")
        batch_size, patches, ref_points, latent_dim = lowest.shape
        return ReIMTSShiftFeatures(
            tokens=lowest.reshape(batch_size, patches * ref_points, latent_dim),
            positions=lowest_reference_positions.reshape(
                batch_size, patches * ref_points
            ),
            patch_valid=lowest_valid.any(dim=-1),
        )

    def prepare_shift_features_from_spatial(self, spatial_feats, positions):
        """Run shift-invariant ReIMTS/mTAN/IARF once on real timestamps."""
        if spatial_feats.is_cuda:
            torch.cuda.synchronize(spatial_feats.device)
        reimts_started = time.perf_counter()
        recursive = self.reimts_encoder(spatial_feats, positions)
        flattened = self._flatten_lowest(
            recursive.lowest,
            recursive.lowest_reference_positions,
            recursive.lowest_valid,
        )
        if flattened.tokens.is_cuda:
            torch.cuda.synchronize(flattened.tokens.device)
        reimts_seconds = time.perf_counter() - reimts_started
        return ReIMTSShiftFeatures(
            tokens=flattened.tokens,
            positions=flattened.positions,
            patch_valid=flattened.patch_valid,
            reimts_mtan_time=reimts_seconds,
        )

    def prepare_shift_features(self, pixels, mask, positions, extra):
        """Cache PSE and all ReIMTS work before candidate-specific LTAE."""
        if pixels.is_cuda:
            torch.cuda.synchronize(pixels.device)
        spatial_started = time.perf_counter()
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        if spatial_feats.is_cuda:
            torch.cuda.synchronize(spatial_feats.device)
        spatial_seconds = time.perf_counter() - spatial_started
        features = self.prepare_shift_features_from_spatial(
            spatial_feats, positions
        )
        return ReIMTSShiftFeatures(
            tokens=features.tokens,
            positions=features.positions,
            patch_valid=features.patch_valid,
            spatial_encoder_time=spatial_seconds,
            reimts_mtan_time=features.reimts_mtan_time,
        )

    def _whole_sample_shift(self, shift, features):
        shift = torch.as_tensor(
            shift, device=features.positions.device, dtype=features.positions.dtype
        )
        batch_size = features.positions.shape[0]
        if shift.ndim == 0:
            return shift.expand(batch_size)
        if shift.ndim == 1:
            if shift.numel() == 1:
                return shift.expand(batch_size)
            if shift.numel() != batch_size:
                raise ValueError("shift must provide one value per sample")
            return shift
        if shift.shape[0] != batch_size:
            raise ValueError("shift batch dimension must match features")
        flat = shift.reshape(batch_size, -1)
        if not torch.equal(flat, flat[:, :1].expand_as(flat)):
            raise ValueError("shift must be one whole-sample translation")
        return flat[:, 0]

    def _decode_whole(self, features, shift):
        sample_shift = self._whole_sample_shift(shift, features)
        shifted_positions = features.positions + sample_shift.unsqueeze(1)
        sample_features = self.temporal_decoder(
            features.tokens, shifted_positions
        )
        sample_logits = self.classifier(sample_features)
        return sample_logits, sample_features

    def forward_from_shift_features(
        self, features, shift, return_feats=False
    ):
        """Evaluate one global translation with one LTAE and classifier."""
        sample_logits, sample_features = self._decode_whole(features, shift)
        if return_feats:
            return sample_logits, sample_features
        return sample_logits

    def _output_from_shift_features(self, features, shift):
        sample_logits = self.forward_from_shift_features(features, shift)
        return ReIMTSClassificationOutput(
            sample_logits=sample_logits,
            patch_valid=features.patch_valid,
        )

    def forward_from_spatial_for_loss_with_shift(
        self, spatial_feats, positions, shift=0
    ):
        features = self.prepare_shift_features_from_spatial(
            spatial_feats, positions
        )
        return self._output_from_shift_features(features, shift)

    def forward_from_spatial_with_shift(
        self, spatial_feats, positions, shift=0, return_feats=False
    ):
        features = self.prepare_shift_features_from_spatial(
            spatial_feats, positions
        )
        return self.forward_from_shift_features(
            features, shift, return_feats=return_feats
        )

    def forward_for_loss(
        self, pixels, mask, positions, extra, shift=0
    ):
        features = self.prepare_shift_features(pixels, mask, positions, extra)
        return self._output_from_shift_features(features, shift)

    def forward_with_shift(
        self, pixels, mask, positions, extra, shift, return_feats=False
    ):
        features = self.prepare_shift_features(pixels, mask, positions, extra)
        return self.forward_from_shift_features(
            features, shift=shift, return_feats=return_feats
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
