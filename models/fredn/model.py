"""Historical FreDN classifier retained for offline audit and probes."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from models.decoder import get_decoder
from models.fredn.disentangler import FrequencyDisentangler, ReImSpectralEncoder
from models.fredn.nufft import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
    IrregularFourierAnalyzer,
    IrregularFourierSynthesizer,
    centered_modes,
)
from models.ltae import LTAE
from models.pse import PixelSetEncoder

@dataclass
class FreDNPreparedFeatures:
    trend: torch.Tensor
    seasonal_coeffs: torch.Tensor
    diagnostics: Dict[str, object]


class PseFreDNLTae(nn.Module):
    """PSE + FreDN decomposition + trend LTAE and seasonal ReIm encoder."""

    supports_fredn_diagnostics = True

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
        max_position=365,
        fredn_num_modes=9,
        fredn_nufft_reg=1e-3,
        fredn_nufft_tol=1e-5,
        fredn_nufft_max_iter=20,
        fredn_period_days=365.0,
        fredn_fourier_solver="dense_direct",
        nufft_backend=None,
    ):
        super().__init__()
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
        if fredn_fourier_solver == "dense_direct":
            self.fourier_analyzer = BatchedDirectFourierAnalyzer(
                num_modes=fredn_num_modes,
                period_days=fredn_period_days,
                reg=fredn_nufft_reg,
            )
            self.fourier_synthesizer = BatchedDirectFourierSynthesizer(
                num_modes=fredn_num_modes,
                period_days=fredn_period_days,
            )
        elif fredn_fourier_solver == "nufft_cg":
            self.fourier_analyzer = IrregularFourierAnalyzer(
                num_modes=fredn_num_modes,
                period_days=fredn_period_days,
                reg=fredn_nufft_reg,
                tol=fredn_nufft_tol,
                max_iter=fredn_nufft_max_iter,
                backend=nufft_backend,
            )
            self.fourier_synthesizer = IrregularFourierSynthesizer(
                num_modes=fredn_num_modes,
                period_days=fredn_period_days,
                backend=self.fourier_analyzer.backend,
            )
        else:
            raise ValueError(
                "fredn_fourier_solver must be 'dense_direct' or 'nufft_cg'"
            )
        self.fredn_fourier_solver = fredn_fourier_solver
        self.fredn_num_modes = fredn_num_modes
        self.fredn_period_days = fredn_period_days
        self.synthesis_isign = self.fourier_synthesizer.synthesis_isign
        self.register_buffer(
            "fredn_modes",
            centered_modes(fredn_num_modes),
            persistent=False,
        )
        self.frequency_disentangler = FrequencyDisentangler(
            num_modes=fredn_num_modes,
            channels=channels,
        )

        ltae_kwargs = dict(
            in_channels=channels,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            max_temporal_shift=max_temporal_shift,
            max_position=max_position,
        )
        self.trend_temporal_encoder = LTAE(**ltae_kwargs)
        temporal_dim = mlp3[-1]
        if mlp4[0] != temporal_dim:
            raise ValueError("mlp4 input must match the branch feature dimension")
        self.trend_norm = nn.LayerNorm(temporal_dim)
        self.seasonal_spectral_encoder = ReImSpectralEncoder(
            num_modes=fredn_num_modes,
            channels=channels,
            output_dim=temporal_dim,
            dropout=dropout,
        )
        self.trend_classifier = get_decoder(mlp4, num_classes)
        self.seasonal_classifier = get_decoder(mlp4, num_classes)
        self.last_diagnostics: Dict[str, object] = {}

    def get_temporal_encoders(self):
        return (self.trend_temporal_encoder,)

    def prepare_temporal_features(
        self,
        spatial_feats,
        positions,
        collect_diagnostics=False,
    ):
        if self.fredn_fourier_solver == "dense_direct":
            coeffs, analysis_diagnostics = self.fourier_analyzer(
                spatial_feats,
                positions,
                collect_diagnostics=collect_diagnostics,
            )
        else:
            coeffs, analysis_diagnostics = self.fourier_analyzer(
                spatial_feats,
                positions,
            )
        trend_coeffs, seasonal_coeffs, mask = self.frequency_disentangler(coeffs)

        trend_complex = self.fourier_synthesizer.synthesize_complex(
            trend_coeffs,
            positions,
        )
        trend = trend_complex.real
        diagnostics = {}
        if collect_diagnostics:
            seasonal_complex = self.fourier_synthesizer.synthesize_complex(
                seasonal_coeffs,
                positions,
            )
            reconstructed_complex = self.fourier_synthesizer.synthesize_complex(
                coeffs,
                positions,
            )
            reconstructed = reconstructed_complex.real
            epsilon = torch.finfo(spatial_feats.dtype).eps
            additivity_error = torch.linalg.vector_norm(
                trend + seasonal_complex.real - reconstructed
            ) / (torch.linalg.vector_norm(reconstructed) + epsilon)
            reconstruction_error = torch.linalg.vector_norm(
                reconstructed - spatial_feats
            ) / (torch.linalg.vector_norm(spatial_feats) + epsilon)
            imaginary_residual = torch.linalg.vector_norm(
                reconstructed_complex.imag
            ) / (torch.linalg.vector_norm(reconstructed_complex.real) + epsilon)

            diagnostics = dict(analysis_diagnostics)
            if "condition_numbers" in diagnostics:
                condition_numbers = diagnostics["condition_numbers"]
                diagnostics.update(
                    {
                        "fourier_condition_mean": condition_numbers.mean().detach(),
                        "fourier_condition_median": condition_numbers.median().detach(),
                        "fourier_condition_p95": torch.quantile(
                            condition_numbers,
                            0.95,
                        ).detach(),
                        "fourier_condition_max": condition_numbers.max().detach(),
                    }
                )
            if "solver_info" in diagnostics:
                solver_info = diagnostics["solver_info"]
                diagnostics.update(
                    {
                        "solver_iterations": torch.zeros(
                            (), device=solver_info.device, dtype=torch.long
                        ),
                        "solver_converged": (solver_info == 0).all().detach(),
                    }
                )
            diagnostics.update(
                self.frequency_disentangler.diagnostics(
                    coeffs,
                    trend_coeffs,
                    seasonal_coeffs,
                    mask,
                )
            )
            diagnostics.update(
                {
                    "additivity_error": additivity_error.detach(),
                    "reconstruction_error": reconstruction_error.detach(),
                    "imaginary_residual": imaginary_residual.detach(),
                }
            )
        self.last_diagnostics = diagnostics
        return FreDNPreparedFeatures(
            trend=trend,
            seasonal_coeffs=seasonal_coeffs,
            diagnostics=diagnostics,
        )

    def _shift_seasonal_coefficients(
        self,
        seasonal_coeffs,
        temporal_shift,
    ):
        if seasonal_coeffs.ndim != 3:
            raise ValueError("seasonal_coeffs must be [B,F,D]")
        batch_size = seasonal_coeffs.shape[0]
        shift = torch.as_tensor(
            temporal_shift,
            device=seasonal_coeffs.device,
            dtype=seasonal_coeffs.real.dtype,
        )
        if shift.ndim == 0:
            shift = shift.reshape(1, 1, 1)
        elif shift.ndim == 1 and shift.shape[0] in (1, batch_size):
            shift = shift.reshape(shift.shape[0], 1, 1)
        elif (
            shift.ndim == 2
            and shift.shape[1] == 1
            and shift.shape[0] in (1, batch_size)
        ):
            shift = shift.reshape(shift.shape[0], 1, 1)
        else:
            raise ValueError(
                "temporal_shift must be scalar, [1], [B], [1,1], or [B,1]"
            )

        modes = self.fredn_modes.to(
            device=seasonal_coeffs.device,
            dtype=seasonal_coeffs.real.dtype,
        ).reshape(1, -1, 1)
        phase_angle = (
            -self.synthesis_isign
            * 2.0
            * torch.pi
            * shift
            * modes
            / self.fredn_period_days
        )
        phase = torch.polar(torch.ones_like(phase_angle), phase_angle)
        return seasonal_coeffs * phase

    def classify_prepared(
        self,
        prepared,
        positions,
        temporal_shift=0,
        return_feats=False,
    ):
        shifted_positions = positions + temporal_shift
        trend_embedding = self.trend_temporal_encoder(
            prepared.trend,
            shifted_positions,
        )
        trend_features = self.trend_norm(trend_embedding)
        shifted_seasonal_coeffs = self._shift_seasonal_coefficients(
            prepared.seasonal_coeffs,
            temporal_shift,
        )
        seasonal_features = self.seasonal_spectral_encoder(
            shifted_seasonal_coeffs
        )
        trend_logits = self.trend_classifier(trend_features)
        seasonal_logits = self.seasonal_classifier(seasonal_features)
        logits = trend_logits + seasonal_logits
        temporal_feats = trend_features + seasonal_features
        if prepared.diagnostics:
            prepared.diagnostics.update(
                {
                    "trend_logit_rms": trend_logits.square().mean().sqrt().detach(),
                    "seasonal_logit_rms": seasonal_logits.square().mean().sqrt().detach(),
                    "trend_feature_rms": trend_features.square().mean().sqrt().detach(),
                    "seasonal_feature_rms": seasonal_features.square().mean().sqrt().detach(),
                }
            )
            self.last_diagnostics = prepared.diagnostics
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
        collect_diagnostics=False,
    ):
        spatial_feats = self.spatial_encoder(pixels, mask, extra)
        prepared = self.prepare_temporal_features(
            spatial_feats,
            positions,
            collect_diagnostics=collect_diagnostics,
        )
        return self.classify_prepared(
            prepared,
            positions,
            temporal_shift=temporal_shift,
            return_feats=return_feats,
        )

    def forward(
        self,
        pixels,
        mask,
        positions,
        extra,
        return_feats=False,
        collect_diagnostics=False,
    ):
        return self.forward_with_temporal_shift(
            pixels,
            mask,
            positions,
            extra,
            temporal_shift=0,
            return_feats=return_feats,
            collect_diagnostics=collect_diagnostics,
        )


