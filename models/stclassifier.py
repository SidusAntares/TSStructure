from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.competings import GRU, TempConv
from models.decoder import get_decoder
from models.fourier_reconstruction import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
)
from models.ltae import LTAE
from models.pse import PixelSetEncoder
from models.tae import TemporalAttentionEncoder
from models.structure_da.discriminative_structure import DiscriminativeStructureBranch


class FrozenStateOrgReference(nn.Module):
    """Source checkpoint measurement basis kept outside the UDA optimizer."""

    def __init__(
        self, spatial_encoder, structure_branch, response_norm,
        trainable_anchors=False,
    ):
        super().__init__()
        self.spatial_encoder = deepcopy(spatial_encoder)
        self.structure_branch = deepcopy(structure_branch)
        self.shape_response_norm = deepcopy(response_norm)
        self.requires_grad_(False)
        self.structure_branch.shapelet_dictionary.anchors.requires_grad_(
            bool(trainable_anchors)
        )
        self.eval()

    @classmethod
    def from_source_model(cls, model, trainable_anchors=False):
        if getattr(model, "shape_representation", None) != "state_org":
            raise ValueError("frozen source basis requires state_org")
        return cls(
            model.spatial_encoder, model.structure_branch, model.shape_response_norm,
            trainable_anchors=trainable_anchors,
        )

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, pixels, mask, positions, extra):
        spatial = self.spatial_encoder(pixels, mask, extra)
        structure = self.structure_branch(
            spatial, positions, include_legacy_query=False,
        )
        return self.shape_response_norm(structure["shapelet_response"])

    def forward_with_details(self, pixels, mask, positions, extra):
        spatial = self.spatial_encoder(pixels, mask, extra)
        structure = self.structure_branch(
            spatial, positions, include_legacy_query=False,
        )
        return self.shape_response_norm(structure["shapelet_response"]), structure


class PseStructureProtoLTae(nn.Module):
    """PSE classifier with discriminative Fourier structure as the LTAE query."""

    def __init__(
        self, input_dim=10, mlp1=[10, 32, 64], pooling="mean_std",
        mlp2=[128, 128], with_extra=True, extra_size=4,
        n_head=16, d_k=8, d_model=256, mlp3=[256, 128], dropout=.2,
        T=1000, mlp4=[128, 64, 32], num_classes=20,
        max_temporal_shift=100, shape_dim=128,
        shape_window_scales=(24,), shape_window_stride=8,
        shapelet_count=16, shapelet_beta=5., shape_resample_length=16,
        fourier_num_modes=13, fourier_reg=1e-3, fourier_period_days=365.,
        shape_representation="current", shape_injection="current_query",
        structure_shift_mode="none", state_org_readout="full",
    ):
        super().__init__()
        if (
            shape_injection == "direct_response_query"
            and shape_representation not in ("current", "state_org")
        ):
            raise ValueError(
                "direct_response_query injection requires current or state_org representation"
            )
        if shape_representation == "current" and shape_injection not in (
            "current_query", "direct_response_query", "local_query", "local_query_only",
        ):
            raise ValueError(
                "current representation requires current_query, direct_response_query, "
                "local_query, or local_query_only injection"
            )
        if shape_representation == "sorted_profile" and shape_injection not in (
            "direct_query", "late_fusion",
        ):
            raise ValueError(
                "sorted_profile representation requires direct_query or late_fusion injection"
            )
        if shape_representation == "set_response" and shape_injection != "current_query":
            raise ValueError("set_response representation requires current_query injection")
        if shape_representation == "residual_response" and shape_injection != "current_query":
            raise ValueError("residual_response representation requires current_query injection")
        if shape_representation == "phase_moment" and shape_injection != "current_query":
            raise ValueError("phase_moment representation requires current_query injection")
        if shape_representation == "state_org" and shape_injection != "direct_response_query":
            raise ValueError("state_org representation requires direct_response_query injection")
        if shape_representation not in (
            "current", "sorted_profile", "set_response", "residual_response",
            "phase_moment", "state_org",
        ):
            raise ValueError(f"unknown shape representation: {shape_representation}")
        self.shape_representation = shape_representation
        self.shape_injection = shape_injection
        if structure_shift_mode not in ("none", "timematch"):
            raise ValueError("structure_shift_mode must be none or timematch")
        self.structure_shift_mode = structure_shift_mode
        spatial_mlp2 = deepcopy(mlp2)
        if with_extra:
            spatial_mlp2[0] += extra_size
        self.spatial_encoder = PixelSetEncoder(
            input_dim, mlp1=mlp1, pooling=pooling, mlp2=spatial_mlp2,
            with_extra=with_extra, extra_size=extra_size,
        )
        channels = spatial_mlp2[-1]
        self.structure_branch = DiscriminativeStructureBranch(
            channels, shape_dim=shape_dim, num_modes=fourier_num_modes,
            period_days=fourier_period_days, reg=fourier_reg,
            window_scales=tuple(shape_window_scales), window_stride=shape_window_stride,
            shapelet_count=shapelet_count, shapelet_beta=shapelet_beta,
            shape_resample_length=shape_resample_length,
            shape_representation=shape_representation,
            state_org_readout=state_org_readout,
        )
        candidates_per_scale = (64 + int(shape_window_stride) - 1) // int(shape_window_stride)
        sorted_profile_dim = (
            int(shapelet_count) * candidates_per_scale * len(tuple(shape_window_scales))
        )
        if shape_representation in ("current", "residual_response"):
            evidence_dim = 2 * int(shapelet_count)
        elif shape_representation == "phase_moment":
            evidence_dim = 6 * int(shapelet_count)
        elif shape_representation == "set_response":
            evidence_dim = 32
        elif shape_representation == "state_org":
            evidence_dim = int(shapelet_count) + 32
        else:
            evidence_dim = sorted_profile_dim
        external_query_dim = (
            shape_dim if shape_injection == "current_query"
            else evidence_dim if shape_injection in (
                "direct_query", "direct_response_query",
            )
            else None
        )
        self.temporal_encoder = LTAE(
            in_channels=channels, n_head=n_head, d_k=d_k, d_model=d_model,
            n_neurons=mlp3, dropout=dropout, T=T,
            max_temporal_shift=max_temporal_shift,
            external_query_dim=external_query_dim,
        )
        self.decoder = get_decoder(mlp4, num_classes)
        if shape_representation == "state_org":
            self.shape_response_norm = nn.LayerNorm(evidence_dim)
        self.shape_classifier = nn.Linear(evidence_dim, num_classes)
        if shape_injection == "late_fusion":
            self.late_fusion_projection = nn.Linear(
                evidence_dim, mlp3[-1], bias=False,
            )
            nn.init.zeros_(self.late_fusion_projection.weight)
        if shape_injection in ("local_query", "local_query_only"):
            self.local_query_projection = nn.Linear(
                int(shapelet_count), n_head * d_k, bias=False,
            )
            if shape_injection == "local_query":
                nn.init.zeros_(self.local_query_projection.weight)
            else:
                nn.init.xavier_uniform_(self.local_query_projection.weight)
            self.local_order_scorer = nn.Conv1d(
                mlp3[-1], 1, kernel_size=3, padding=1,
            )
            if shape_injection == "local_query":
                self.raw_structure_gamma = nn.Parameter(torch.logit(torch.tensor(.2)))
            self.local_query_heads = int(n_head)
            self.local_query_dim = int(d_k)
        self.shape_dim = shape_dim
        self.shape_evidence_dim = evidence_dim
        self.instance_dim = mlp3[-1]
        self.state_org_readout = state_org_readout

    def configure_state_org_query(self, query_view="full", query_scale=1., reference=None):
        if self.shape_representation != "state_org":
            if query_view != "full" or float(query_scale) != 1. or reference is not None:
                raise ValueError("state-org query controls require state_org")
            return
        self.select_state_org_query(torch.zeros(1, self.shape_evidence_dim), query_view)
        object.__setattr__(self, "_state_org_query_view", query_view)
        object.__setattr__(self, "_shape_query_scale", float(query_scale))
        object.__setattr__(self, "_frozen_state_org_reference", reference)

    def get_temporal_encoders(self):
        return (self.temporal_encoder,)

    def prepare_temporal_features(self, spatial_feats, positions):
        return spatial_feats

    def prepare_structure_context(self, prepared, positions):
        return self.structure_branch.prepare_context(prepared, positions)

    def prepare_structure_from_context(
        self, context, temporal_shift=0, structure_aug_shift=None, phase_shift=None,
    ):
        if self.shape_representation == "phase_moment":
            structure_shift = 0 if structure_aug_shift is None else structure_aug_shift
            calendar_shift = temporal_shift if phase_shift is None else phase_shift
        else:
            structure_shift = (
                temporal_shift if self.structure_shift_mode == "timematch" else 0
            ) if structure_aug_shift is None else structure_aug_shift
            calendar_shift = 0 if phase_shift is None else phase_shift
        return self.structure_branch.forward_from_context(
            context, structure_aug_shift=structure_shift,
            phase_shift=calendar_shift,
            include_legacy_query=self.shape_injection == "current_query",
        )

    def prepare_structure(self, prepared, positions, temporal_shift=0):
        if (
            self.shape_injection in ("local_query", "local_query_only")
            or self.shape_representation == "phase_moment"
        ):
            return self.prepare_structure_from_context(
                self.prepare_structure_context(prepared, positions), temporal_shift,
            )
        return self.structure_branch(
            prepared, positions,
            include_legacy_query=self.shape_injection == "current_query",
        )

    def _shape_evidence(self, structure):
        if self.shape_representation in (
            "current", "set_response", "residual_response", "phase_moment",
            "state_org",
        ):
            evidence = structure["shapelet_response"]
            return (
                self.shape_response_norm(evidence)
                if self.shape_representation == "state_org" else evidence
            )
        return structure["sorted_anchor_profile"]

    @staticmethod
    def _parameter_free_layer_norm(evidence):
        return F.layer_norm(evidence, evidence.shape[-1:])

    @property
    def structure_gamma(self):
        if self.shape_injection != "local_query":
            raise AttributeError("structure gamma is only defined for local_query")
        return .5 * torch.sigmoid(self.raw_structure_gamma)

    def _project_local_queries(self, similarity):
        batch, windows, _ = similarity.shape
        return self.local_query_projection(similarity).view(
            batch, windows, self.local_query_heads, self.local_query_dim,
        )

    def _encode_instance(
        self, prepared, shifted_positions, structure, return_details=False,
        detach_structure_query=False, query_view="full", query_scale=1.,
    ):
        evidence = self._shape_evidence(structure)
        if self.shape_injection == "current_query":
            instance = self.temporal_encoder(
                prepared, shifted_positions,
                external_query=structure["shape_class_token"],
            )
            return (instance, {}) if return_details else instance
        if self.shape_injection == "direct_response_query":
            evidence = self.select_state_org_query(evidence, query_view)
            instance = self.temporal_encoder(
                prepared, shifted_positions, external_query=evidence,
                detach_external_query_correction=detach_structure_query,
                query_scale=query_scale,
            )
            return (instance, {}) if return_details else instance
        normalized = self._parameter_free_layer_norm(evidence)
        if self.shape_injection == "direct_query":
            instance = self.temporal_encoder(
                prepared, shifted_positions, external_query=normalized,
            )
            return (instance, {}) if return_details else instance
        if self.shape_injection == "late_fusion":
            temporal = self.temporal_encoder(
                prepared, shifted_positions, external_query=None,
            )
            instance = temporal + self.late_fusion_projection(normalized)
            return (instance, {}) if return_details else instance
        similarity = structure["shapelet_similarity"]
        correction = self._project_local_queries(similarity)
        if self.shape_injection == "local_query_only":
            local = self.temporal_encoder.forward_with_explicit_queries(
                prepared, shifted_positions, correction,
            )
            attention = torch.softmax(
                self.local_order_scorer(local.transpose(1, 2)).squeeze(1), dim=1,
            )
            instance = (attention.unsqueeze(-1) * local).sum(1)
            details = {
                "local_query": correction,
                "local_structure_readout": local,
                "structure_attention": attention,
                "structure_feature": instance,
            }
            return (instance, details) if return_details else instance
        temporal, local = self.temporal_encoder.forward_with_local_queries(
            prepared, shifted_positions, correction,
        )
        residual = local - temporal[:, None]
        attention = torch.softmax(
            self.local_order_scorer(residual.transpose(1, 2)).squeeze(1), dim=1,
        )
        structure_feature = (attention.unsqueeze(-1) * residual).sum(1)
        gamma = self.structure_gamma
        instance = temporal + gamma * structure_feature
        details = {
            "base_instance_feature": temporal,
            "local_query_correction": correction,
            "local_structure_readout": local,
            "structure_residual": residual,
            "structure_attention": attention,
            "structure_feature": structure_feature,
            "structure_gamma": gamma,
        }
        return (instance, details) if return_details else instance

    def structure_usage_diagnostics(self, output):
        if self.shape_injection == "local_query_only":
            return {}
        if self.shape_injection == "local_query":
            base_norm = output["base_instance_feature"].norm(dim=-1).clamp_min(1e-12)
            residual_norm = (
                output["structure_gamma"] * output["structure_feature"]
            ).norm(dim=-1)
            return {
                "gamma": output["structure_gamma"],
                "structure_residual_norm_ratio": (residual_norm / base_norm).mean(),
            }
        if self.shape_representation == "residual_response":
            modulation = output["shapelet_modulation"]
            similarity = output["shapelet_similarity"]
            contextual = output["shapelet_context_score"]
            return {
                "mean_abs_modulation": modulation.abs().mean(),
                "max_abs_modulation": modulation.abs().max(),
                "relative_context_correction": (
                    (contextual - similarity).norm()
                    / similarity.norm().clamp_min(1e-12)
                ),
            }
        if self.shape_representation == "phase_moment":
            moments = output["shapelet_phase_moments"]
            count = moments.shape[-1] // 4
            blocks = moments.reshape(moments.shape[0], 2, 2, count)
            magnitude = blocks.square().sum(dim=2).sqrt()
            return {
                "phase_k1_mean_magnitude": magnitude[:, 0].mean(),
                "phase_k2_mean_magnitude": magnitude[:, 1].mean(),
            }
        if self.shape_representation != "sorted_profile":
            return {}
        diagnostics = {
            "sorted_profile_norm_mean": output["sorted_anchor_profile"].norm(dim=-1).mean(),
        }
        if self.shape_injection == "direct_query":
            projection = self.temporal_encoder.attention_heads.external_query_projection
            diagnostics["query_projection_weight_norm"] = projection.weight.norm()
        else:
            diagnostics["late_fusion_weight_norm"] = self.late_fusion_projection.weight.norm()
        return diagnostics

    def classify_prepared(
        self, prepared, positions, temporal_shift=0, return_feats=False,
        prepared_structure=None,
    ):
        shifted_positions = positions + temporal_shift
        structure = (
            self.prepare_structure(prepared, positions, temporal_shift)
            if prepared_structure is None else prepared_structure
        )
        instance = self._encode_instance(prepared, shifted_positions, structure)
        logits = self.decoder(instance)
        if return_feats:
            return logits, instance
        return logits

    def _output_from_prepared_structure(
        self, prepared, shifted_positions, structure, detach_structure_query=False,
        query_view="full", query_scale=1.,
    ):
        evidence = self._shape_evidence(structure)
        instance, usage = self._encode_instance(
            prepared, shifted_positions, structure, return_details=True,
            detach_structure_query=detach_structure_query,
            query_view=query_view, query_scale=query_scale,
        )
        return {
            "logits": self.decoder(instance),
            "shape_logits": self.shape_classifier(evidence),
            "shape_evidence": evidence,
            "instance_feature": instance,
            **usage,
            **structure,
        }

    def select_state_org_query(self, normalized_evidence, query_view="full"):
        if query_view not in ("full", "presence"):
            raise ValueError("state-org query view must be full or presence")
        if query_view == "full":
            return normalized_evidence
        if self.shape_representation != "state_org":
            raise ValueError("presence query view requires state_org")
        count = self.structure_branch.shapelet_dictionary.anchors.shape[0]
        return torch.cat((
            normalized_evidence[:, :count],
            torch.zeros_like(normalized_evidence[:, count:]),
        ), dim=1)

    def forward_with_external_shape_evidence(
        self, pixels, mask, positions, extra, shape_evidence,
        temporal_shift=0, query_view="full", query_scale=1.,
        detach_projected_query_correction=False,
    ):
        if self.shape_representation != "state_org":
            raise ValueError("external shape evidence requires state_org")
        spatial = self.spatial_encoder(pixels, mask, extra)
        query = self.select_state_org_query(shape_evidence, query_view)
        instance = self.temporal_encoder(
            spatial, positions + temporal_shift, external_query=query,
            query_scale=query_scale,
            detach_external_query_correction=detach_projected_query_correction,
        )
        return {
            "logits": self.decoder(instance),
            "shape_logits": self.shape_classifier(shape_evidence),
            "shape_evidence": shape_evidence,
            "shapelet_response": shape_evidence,
            "instance_feature": instance,
        }

    def forward_phase_equivariance_target(
        self, pixels, mask, positions, extra, structure_aug_shift,
    ):
        """Share one spatial feature and Fourier context across E's three consumers."""
        if self.shape_representation != "phase_moment":
            raise RuntimeError("phase equivariance requires phase_moment representation")
        spatial = self.spatial_encoder(pixels, mask, extra)
        context = self.prepare_structure_context(spatial, positions)
        base = self.prepare_structure_from_context(
            context, structure_aug_shift=0, phase_shift=0,
        )
        shifted = self.prepare_structure_from_context(
            context, structure_aug_shift=structure_aug_shift, phase_shift=0,
        )
        output = self._output_from_prepared_structure(spatial, positions, base)
        return output, base, shifted

    def forward_with_temporal_shift(
        self, pixels, mask, positions, extra, temporal_shift=0,
        return_feats=False, return_dict=False, collect_diagnostics=False,
        detach_structure_query=False, query_view=None, query_scale=None,
        structure_anchor_grad=False,
    ):
        del collect_diagnostics
        if detach_structure_query and self.shape_representation != "state_org":
            raise ValueError("detach_structure_query is only valid for state_org")
        query_view = query_view or getattr(self, "_state_org_query_view", "full")
        query_scale = (
            getattr(self, "_shape_query_scale", 1.)
            if query_scale is None else float(query_scale)
        )
        reference = getattr(self, "_frozen_state_org_reference", None)
        if reference is not None:
            if structure_anchor_grad:
                evidence, reference_structure = reference.forward_with_details(
                    pixels, mask, positions, extra,
                )
            else:
                with torch.no_grad():
                    evidence, reference_structure = reference.forward_with_details(
                        pixels, mask, positions, extra,
                    )
            output = self.forward_with_external_shape_evidence(
                pixels, mask, positions, extra, evidence,
                temporal_shift=temporal_shift, query_view=query_view,
                query_scale=query_scale,
                detach_projected_query_correction=detach_structure_query,
            )
            output.update({
                key: value for key, value in reference_structure.items()
                if key not in output
            })
            if return_dict:
                return output
            if return_feats:
                return output["logits"], output["instance_feature"]
            return output["logits"]
        spatial = self.spatial_encoder(pixels, mask, extra)
        shifted_positions = positions + temporal_shift
        structure = self.prepare_structure(spatial, positions, temporal_shift)
        output = self._output_from_prepared_structure(
            spatial, shifted_positions, structure,
            detach_structure_query=detach_structure_query,
            query_view=query_view, query_scale=query_scale,
        )
        if return_dict:
            return output
        if return_feats:
            return output["logits"], output["instance_feature"]
        return output["logits"]

    def forward(self, pixels, mask, positions, extra, return_feats=False, return_dict=False):
        return self.forward_with_temporal_shift(
            pixels, mask, positions, extra,
            return_feats=return_feats, return_dict=return_dict,
        )


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
