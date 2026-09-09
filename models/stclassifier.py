from copy import deepcopy

import torch
import torch.nn as nn

from models.competings import GRU, TempConv
from models.decoder import get_decoder
from models.fourier_reconstruction import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
)
from models.ltae import LTAE
from models.pse import PixelSetEncoder
from models.tae import TemporalAttentionEncoder


class PseFourierReconLTae(nn.Module):
    """PSE followed by parameter-free finite Fourier reconstruction and LTAE."""

    def __init__(
        self,
        input_dim=10,
        mlp1=[10, 32, 64],
        pooling="mean_std",
        mlp2=[128, 128],
        with_extra=True,
        extra_size=4,
        n_head=16,
        d_k=8,
        d_model=256,
        mlp3=[256, 128],
        dropout=0.2,
        T=1000,
        mlp4=[128, 64, 32],
        num_classes=20,
        max_temporal_shift=100,
        fourier_num_modes=13,
        fourier_reg=1e-3,
        fourier_period_days=365.0,
        fourier_solver="dense_direct",
    ):
        super().__init__()
        if fourier_solver != "dense_direct":
            raise ValueError("fourier_solver must be 'dense_direct'")
        spatial_mlp2 = deepcopy(mlp2)
        if with_extra:
            spatial_mlp2[0] += extra_size
        self.spatial_encoder = PixelSetEncoder(
            input_dim,
            mlp1=mlp1,
            pooling=pooling,
            mlp2=spatial_mlp2,
            with_extra=with_extra,
            extra_size=extra_size,
        )
        channels = spatial_mlp2[-1]
        self.fourier_analyzer = BatchedDirectFourierAnalyzer(
            num_modes=fourier_num_modes,
            period_days=fourier_period_days,
            reg=fourier_reg,
        )
        self.fourier_synthesizer = BatchedDirectFourierSynthesizer(
            num_modes=fourier_num_modes,
            period_days=fourier_period_days,
        )
        self.temporal_encoder = LTAE(
            in_channels=channels,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
        )
        self.decoder = get_decoder(mlp4, num_classes)
        self.fourier_num_modes = fourier_num_modes
        self.fourier_reg = fourier_reg
        self.fourier_period_days = fourier_period_days
        self.fourier_solver = fourier_solver
        print(
            "FOURIER_RECON_CONFIG|"
            f"num_modes={fourier_num_modes}|solver={fourier_solver}|"
            f"period_days={float(fourier_period_days)}|reg={fourier_reg}|"
            "learned_mask=false|decomposition=false|branches=1"
        )

    def get_temporal_encoders(self):
        return (self.temporal_encoder,)

    def prepare_temporal_features(self, spatial_feats, positions):
        fourier_coeffs, _ = self.fourier_analyzer(spatial_feats, positions)
        return self.fourier_synthesizer(fourier_coeffs, positions)

    def classify_prepared(
        self,
        prepared,
        positions,
        temporal_shift=0,
        return_feats=False,
    ):
        temporal_feats = self.temporal_encoder(prepared, positions + temporal_shift)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        return logits

    def forward_with_temporal_shift(
        self,
        pixels,
        mask,
        positions,
        extra,
        temporal_shift=0,
        return_feats=False,
    ):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        reconstructed_features = self.prepare_temporal_features(spatial_feats, positions)
        return self.classify_prepared(
            reconstructed_features,
            positions,
            temporal_shift=temporal_shift,
            return_feats=return_feats,
        )

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        return self.forward_with_temporal_shift(
            pixels,
            mask,
            positions,
            extra,
            temporal_shift=0,
            return_feats=return_feats,
        )


class PseLTae(nn.Module):
    """
    Pixel-Set encoder + Lightweight Temporal Attention Encoder sequence classifier
    """

    def __init__(
        self,
        input_dim=10,
        mlp1=[10, 32, 64],
        pooling="mean_std",
        mlp2=[128, 128],
        with_extra=True,
        extra_size=4,
        n_head=16,
        d_k=8,
        d_model=256,
        mlp3=[256, 128],
        dropout=0.2,
        T=1000,
        mlp4=[128, 64, 32],
        num_classes=20,
        max_temporal_shift=100,
    ):
        super(PseLTae, self).__init__()
        if with_extra:
            mlp2 = deepcopy(mlp2)
            mlp2[0] += extra_size

        self.spatial_encoder = PixelSetEncoder(
            input_dim,
            mlp1=mlp1,
            pooling=pooling,
            mlp2=mlp2,
            with_extra=with_extra,
            extra_size=extra_size,
        )
        self.temporal_encoder = LTAE(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
        )
        self.decoder = get_decoder(mlp4, num_classes)

    def get_temporal_encoders(self):
        return (self.temporal_encoder,)

    def prepare_temporal_features(self, spatial_feats, positions):
        return spatial_feats

    def classify_prepared(
        self,
        prepared,
        positions,
        temporal_shift=0,
        return_feats=False,
    ):
        temporal_feats = self.temporal_encoder(
            prepared,
            positions + temporal_shift,
        )
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        return logits

    def forward_with_temporal_shift(
        self,
        pixels,
        mask,
        positions,
        extra,
        temporal_shift=0,
        return_feats=False,
    ):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        prepared = self.prepare_temporal_features(spatial_feats, positions)
        return self.classify_prepared(
            prepared,
            positions,
            temporal_shift=temporal_shift,
            return_feats=return_feats,
        )

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        """
        Args:
           input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
           Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
           Pixel-Mask : Batch_size x Sequence length x Number of pixels
           Positions : Batch_size x Sequence length
           Extra-features : Batch_size x Sequence length x Number of features
        """
        return self.forward_with_temporal_shift(
            pixels,
            mask,
            positions,
            extra,
            temporal_shift=0,
            return_feats=return_feats,
        )

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                s / total * 100, t / total * 100, c / total * 100
            )
        )

        return total


class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(
        self,
        input_dim=10,
        mlp1=[10, 32, 64],
        pooling="mean_std",
        mlp2=[128, 128],
        with_extra=True,
        extra_size=4,
        n_head=4,
        d_k=32,
        d_model=None,
        mlp3=[512, 128, 128],
        dropout=0.2,
        T=1000,
        mlp4=[128, 64, 32],
        num_classes=20,
        max_temporal_shift=100,
        max_position=365,
    ):
        super(PseTae, self).__init__()
        if with_extra:
            mlp2 = deepcopy(mlp2)
            mlp2[0] += 4
        self.spatial_encoder = PixelSetEncoder(
            input_dim,
            mlp1=mlp1,
            pooling=pooling,
            mlp2=mlp2,
            with_extra=with_extra,
            extra_size=extra_size,
        )
        self.temporal_encoder = TemporalAttentionEncoder(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_position=max_position,
            max_temporal_shift=max_temporal_shift,
        )
        self.decoder = get_decoder(mlp4, num_classes)

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        """
        Args:
           input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
           Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
           Pixel-Mask : Batch_size x Sequence length x Number of pixels
           Positions : Batch_size x Sequence length
           Extra-features : Batch_size x Sequence length x Number of features
        """
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        else:
            return logits

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                s / total * 100, t / total * 100, c / total * 100
            )
        )
        return total


class PseGru(nn.Module):
    """
    Pixel-Set encoder + GRU
    """

    def __init__(
        self,
        input_dim=10,
        mlp1=[10, 32, 64],
        pooling="mean_std",
        mlp2=[128, 128],
        with_extra=True,
        extra_size=4,
        hidden_dim=128,
        mlp4=[128, 64, 32],
        num_classes=20,
        max_temporal_shift=100,
        max_position=365,
    ):
        super(PseGru, self).__init__()
        if with_extra:
            mlp2 = deepcopy(mlp2)
            mlp2[0] += 4
        self.spatial_encoder = PixelSetEncoder(
            input_dim,
            mlp1=mlp1,
            pooling=pooling,
            mlp2=mlp2,
            with_extra=with_extra,
            extra_size=extra_size,
        )
        self.temporal_encoder = GRU(
            in_channels=mlp2[-1],
            hidden_dim=hidden_dim,
            max_position=max_position,
            max_temporal_shift=max_temporal_shift,
        )
        self.decoder = get_decoder(mlp4, num_classes)

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        """
        Args:
           input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
           Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
           Pixel-Mask : Batch_size x Sequence length x Number of pixels
           Positions : Batch_size x Sequence length
           Extra-features : Batch_size x Sequence length x Number of features
        """
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        else:
            return logits

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                s / total * 100, t / total * 100, c / total * 100
            )
        )
        return total


class PseTempCNN(nn.Module):
    """
    Pixel-Set encoder + GRU
    """

    def __init__(
        self,
        input_dim=10,
        mlp1=[10, 32, 64],
        pooling="mean_std",
        mlp2=[128, 128],
        with_extra=True,
        extra_size=4,
        nker=[32, 32, 128],
        mlp3=[128, 128],
        seq_len=24,
        mlp4=[128, 64, 32],
        num_classes=20,
        max_temporal_shift=100,
        max_position=365,
    ):
        super(PseTempCNN, self).__init__()
        if with_extra:
            mlp2 = deepcopy(mlp2)
            mlp2[0] += 4

        self.spatial_encoder = PixelSetEncoder(
            input_dim,
            mlp1=mlp1,
            pooling=pooling,
            mlp2=mlp2,
            with_extra=with_extra,
            extra_size=extra_size,
        )
        self.temporal_encoder = TempConv(
            input_size=mlp2[-1],
            nker=nker,
            seq_len=seq_len,
            nfc=mlp3,
            max_position=max_position,
            max_temporal_shift=max_temporal_shift,
        )
        self.decoder = get_decoder(mlp4, num_classes)

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        """
        Args:
           input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
           Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
           Pixel-Mask : Batch_size x Sequence length x Number of pixels
           Positions : Batch_size x Sequence length
           Extra-features : Batch_size x Sequence length x Number of features
        """
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        else:
            return logits

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                s / total * 100, t / total * 100, c / total * 100
            )
        )
        return total


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
