from copy import deepcopy

import torch.nn as nn

from models.competings import GRU, TempConv
from models.decoder import MTKDLateLogitDecoder, get_decoder
from models.ltae import LTAE
from models.mtkd import (
    MTKDEarlyConcatLTAE,
    MTKDMidConcatLTAE,
    MTKDSOnlyLTAE,
    MTKDTDMidConcatLTAE,
    MTKDTQMidConcatLTAE,
    MTKDTQSingleLTAE,
    MTKDTQLOOLTAE,
)
from models.pse import PixelSetEncoder
from models.tae import TemporalAttentionEncoder


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


class PseMTKDLtae(nn.Module):
    """Pixel-Set encoder + MTKD T/S early concat + one LTAE classifier."""

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
        mtkd_time_scale_days=365.0,
        mtkd_tau_fast_init_days=30.0,
        mtkd_tau_slow_init_days=90.0,
        mtkd_tau_min_days=1.0,
        mtkd_delta_tau_min_days=1.0,
        mtkd_learnable_tau=True,
    ):
        super().__init__()
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
        self.temporal_encoder = MTKDEarlyConcatLTAE(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
            time_scale_days=mtkd_time_scale_days,
            tau_fast_init_days=mtkd_tau_fast_init_days,
            tau_slow_init_days=mtkd_tau_slow_init_days,
            tau_min_days=mtkd_tau_min_days,
            delta_tau_min_days=mtkd_delta_tau_min_days,
            learnable_tau=mtkd_learnable_tau,
        )
        self.decoder = get_decoder(mlp4, num_classes)

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        return logits

    def param_ratio(self):
        total = get_ntrainparams(self)
        spatial = get_ntrainparams(self.spatial_encoder)
        temporal = get_ntrainparams(self.temporal_encoder)
        classifier = get_ntrainparams(self.decoder)
        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                spatial / total * 100,
                temporal / total * 100,
                classifier / total * 100,
            )
        )
        return total


class PseMTKDMidLtae(nn.Module):
    """Pixel-Set encoder + MTKD T/S mid concat + classifier."""

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
        mtkd_time_scale_days=365.0,
        mtkd_tau_fast_init_days=30.0,
        mtkd_tau_slow_init_days=90.0,
        mtkd_tau_min_days=1.0,
        mtkd_delta_tau_min_days=1.0,
        mtkd_learnable_tau=True,
    ):
        super().__init__()
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
        self.temporal_encoder = MTKDMidConcatLTAE(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
            time_scale_days=mtkd_time_scale_days,
            tau_fast_init_days=mtkd_tau_fast_init_days,
            tau_slow_init_days=mtkd_tau_slow_init_days,
            tau_min_days=mtkd_tau_min_days,
            delta_tau_min_days=mtkd_delta_tau_min_days,
            learnable_tau=mtkd_learnable_tau,
        )
        mid_mlp4 = deepcopy(mlp4)
        mid_mlp4[0] = 2 * mlp3[-1]
        self.decoder = get_decoder(mid_mlp4, num_classes)

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        return logits

    def param_ratio(self):
        total = get_ntrainparams(self)
        spatial = get_ntrainparams(self.spatial_encoder)
        temporal = get_ntrainparams(self.temporal_encoder)
        classifier = get_ntrainparams(self.decoder)
        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                spatial / total * 100,
                temporal / total * 100,
                classifier / total * 100,
            )
        )
        return total


class PseMTKDLateLtae(nn.Module):
    """Pixel-Set encoder + MTKD dual LTAE + independent logit classifiers."""

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
        mtkd_time_scale_days=365.0,
        mtkd_tau_fast_init_days=30.0,
        mtkd_tau_slow_init_days=90.0,
        mtkd_tau_min_days=1.0,
        mtkd_delta_tau_min_days=1.0,
        mtkd_learnable_tau=True,
    ):
        super().__init__()
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
        self.temporal_encoder = MTKDMidConcatLTAE(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
            time_scale_days=mtkd_time_scale_days,
            tau_fast_init_days=mtkd_tau_fast_init_days,
            tau_slow_init_days=mtkd_tau_slow_init_days,
            tau_min_days=mtkd_tau_min_days,
            delta_tau_min_days=mtkd_delta_tau_min_days,
            learnable_tau=mtkd_learnable_tau,
        )
        self.decoder = MTKDLateLogitDecoder(mlp4, num_classes)

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        return logits

    def param_ratio(self):
        total = get_ntrainparams(self)
        spatial = get_ntrainparams(self.spatial_encoder)
        temporal = get_ntrainparams(self.temporal_encoder)
        classifier = get_ntrainparams(self.decoder)
        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                spatial / total * 100,
                temporal / total * 100,
                classifier / total * 100,
            )
        )
        return total


class PseMTKDSOnlyLtae(nn.Module):
    """Pixel-Set encoder + MTKD smoothed structure S + one LTAE classifier."""

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
        mtkd_time_scale_days=365.0,
        mtkd_tau_fast_init_days=30.0,
        mtkd_tau_slow_init_days=90.0,
        mtkd_tau_min_days=1.0,
        mtkd_delta_tau_min_days=1.0,
        mtkd_learnable_tau=True,
    ):
        super().__init__()
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
        self.temporal_encoder = MTKDSOnlyLTAE(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
            time_scale_days=mtkd_time_scale_days,
            tau_fast_init_days=mtkd_tau_fast_init_days,
            tau_slow_init_days=mtkd_tau_slow_init_days,
            tau_min_days=mtkd_tau_min_days,
            delta_tau_min_days=mtkd_delta_tau_min_days,
            learnable_tau=mtkd_learnable_tau,
        )
        self.decoder = get_decoder(mlp4, num_classes)

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        return logits

    def param_ratio(self):
        total = get_ntrainparams(self)
        spatial = get_ntrainparams(self.spatial_encoder)
        temporal = get_ntrainparams(self.temporal_encoder)
        classifier = get_ntrainparams(self.decoder)
        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                spatial / total * 100,
                temporal / total * 100,
                classifier / total * 100,
            )
        )
        return total


class PseMTKDTDMidLtae(nn.Module):
    """Pixel-Set encoder + MTKD T/D mid concat + classifier."""

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
        mtkd_time_scale_days=365.0,
        mtkd_tau_fast_init_days=30.0,
        mtkd_tau_slow_init_days=90.0,
        mtkd_tau_min_days=1.0,
        mtkd_delta_tau_min_days=1.0,
        mtkd_learnable_tau=True,
    ):
        super().__init__()
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
        self.temporal_encoder = MTKDTDMidConcatLTAE(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
            time_scale_days=mtkd_time_scale_days,
            tau_fast_init_days=mtkd_tau_fast_init_days,
            tau_slow_init_days=mtkd_tau_slow_init_days,
            tau_min_days=mtkd_tau_min_days,
            delta_tau_min_days=mtkd_delta_tau_min_days,
            learnable_tau=mtkd_learnable_tau,
        )
        td_mlp4 = deepcopy(mlp4)
        td_mlp4[0] = 2 * mlp3[-1]
        self.decoder = get_decoder(td_mlp4, num_classes)

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        return logits

    def param_ratio(self):
        total = get_ntrainparams(self)
        spatial = get_ntrainparams(self.spatial_encoder)
        temporal = get_ntrainparams(self.temporal_encoder)
        classifier = get_ntrainparams(self.decoder)
        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                spatial / total * 100,
                temporal / total * 100,
                classifier / total * 100,
            )
        )
        return total


class PseMTKDTQMidLtae(nn.Module):
    """Pixel-Set encoder + MTKD T/Q mid concat + classifier."""

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
        mtkd_time_scale_days=365.0,
        mtkd_tau_fast_init_days=30.0,
        mtkd_tau_slow_init_days=90.0,
        mtkd_tau_min_days=1.0,
        mtkd_delta_tau_min_days=1.0,
        mtkd_learnable_tau=True,
    ):
        super().__init__()
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
        self.temporal_encoder = MTKDTQMidConcatLTAE(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
            time_scale_days=mtkd_time_scale_days,
            tau_fast_init_days=mtkd_tau_fast_init_days,
            tau_slow_init_days=mtkd_tau_slow_init_days,
            tau_min_days=mtkd_tau_min_days,
            delta_tau_min_days=mtkd_delta_tau_min_days,
            learnable_tau=mtkd_learnable_tau,
        )
        tq_mlp4 = deepcopy(mlp4)
        tq_mlp4[0] = 2 * mlp3[-1]
        self.decoder = get_decoder(tq_mlp4, num_classes)

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        return logits

    def param_ratio(self):
        total = get_ntrainparams(self)
        spatial = get_ntrainparams(self.spatial_encoder)
        temporal = get_ntrainparams(self.temporal_encoder)
        classifier = get_ntrainparams(self.decoder)
        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                spatial / total * 100,
                temporal / total * 100,
                classifier / total * 100,
            )
        )
        return total


class PseMTKDTQSingleLtae(nn.Module):
    """Pixel-Set encoder + fixed single-scale T/Q mid concat + classifier."""

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
        tq_tau_days=60.0,
        time_scale_days=365.0,
    ):
        super().__init__()
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
        self.temporal_encoder = MTKDTQSingleLTAE(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
            tau_days=tq_tau_days,
            time_scale_days=time_scale_days,
        )
        tq_mlp4 = deepcopy(mlp4)
        tq_mlp4[0] = 2 * mlp3[-1]
        self.decoder = get_decoder(tq_mlp4, num_classes)

    def forward(self, pixels, mask, positions, extra, return_feats=False):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        return logits

    def param_ratio(self):
        total = get_ntrainparams(self)
        spatial = get_ntrainparams(self.spatial_encoder)
        temporal = get_ntrainparams(self.temporal_encoder)
        classifier = get_ntrainparams(self.decoder)
        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                spatial / total * 100,
                temporal / total * 100,
                classifier / total * 100,
            )
        )
        return total


class PseMTKDTQLOOLtae(nn.Module):
    """Pixel-Set encoder + T/Q mid classifier with routed LOO objectives."""

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
        tq_loo_variant="split_loo_tau_pse",
        tq_tau_init_days=75.0,
        tq_tau_min_days=1.0,
        tq_loo_pse_weight=0.1,
        time_scale_days=365.0,
    ):
        super().__init__()
        if with_extra:
            mlp2 = deepcopy(mlp2)
            mlp2[0] += extra_size
        self.loo_variant = tq_loo_variant
        self.loo_pse_weight = float(tq_loo_pse_weight)

        self.spatial_encoder = PixelSetEncoder(
            input_dim,
            mlp1=mlp1,
            pooling=pooling,
            mlp2=mlp2,
            with_extra=with_extra,
            extra_size=extra_size,
        )
        self.temporal_encoder = MTKDTQLOOLTAE(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
            variant=tq_loo_variant,
            tau_init_days=tq_tau_init_days,
            tau_min_days=tq_tau_min_days,
            time_scale_days=time_scale_days,
        )
        tq_mlp4 = deepcopy(mlp4)
        tq_mlp4[0] = 2 * mlp3[-1]
        self.decoder = get_decoder(tq_mlp4, num_classes)

    def forward_from_spatial(self, spatial_feats, positions, return_feats=False):
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        return logits

    def compute_loo_losses(self, spatial_feats, positions, time_mask=None):
        return self.temporal_encoder.compute_loo_losses(
            spatial_feats, positions, time_mask=time_mask
        )

    def forward(
        self,
        pixels,
        mask,
        positions,
        extra,
        return_feats=False,
        return_spatial=False,
    ):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        output = self.forward_from_spatial(
            spatial_feats, positions, return_feats=return_feats
        )
        if return_spatial:
            return output, spatial_feats
        return output

    def param_ratio(self):
        total = get_ntrainparams(self)
        spatial = get_ntrainparams(self.spatial_encoder)
        temporal = get_ntrainparams(self.temporal_encoder)
        classifier = get_ntrainparams(self.decoder)
        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                spatial / total * 100,
                temporal / total * 100,
                classifier / total * 100,
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
