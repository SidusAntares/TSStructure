"""End-to-end assembly of the Structure DA model."""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from models.pse import PixelSetEncoder

from .adaptation import (
    JointStructuralOutput,
    JointStructuralSpaceBuilder,
    StructuralAdversarialAdapter,
    StructuralAdversarialOutput,
)
from .model import ComponentStructureClassifier, ComponentStructureOutput


@dataclass(frozen=True)
class StructureDAForwardOutput:
    """PSE features and the complete component-classifier output."""

    pse_features: torch.Tensor
    component: ComponentStructureOutput

    @property
    def logits(self) -> torch.Tensor:
        return self.component.logits

    @property
    def fused_embedding(self) -> torch.Tensor:
        return self.component.fused_embedding


class StructureDAModel(nn.Module):
    """Assemble shared PSE, component classifier, joint space, and SDA."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 10,
        with_extra: bool = False,
        extra_size: int = 4,
        pse_mlp1: Sequence[int] = (10, 32, 64),
        pse_pooling: str = "mean_std",
        pse_mlp2: Sequence[int] = (128, 128),
        time_scale: float = 365.0,
        tau_fast_init: float = 0.05,
        tau_slow_init: float = 0.20,
        tau_min: float = 1e-4,
        delta_tau_min: float = 1e-4,
        n_head: int = 16,
        d_k: int = 8,
        d_model: int = 256,
        ltae_mlp: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        positional_period: int = 1000,
        max_position: int = 365,
        max_temporal_shift: int = 100,
        classifier_hidden: Sequence[int] = (64, 32),
        quality_hidden_cap: int = 128,
        quality_eta: float = 0.1,
        sda_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        mlp1 = list(pse_mlp1)
        mlp2 = list(pse_mlp2)
        if with_extra:
            if not mlp2:
                raise ValueError("pse_mlp2 must not be empty when with_extra=True")
            mlp2[0] += extra_size

        self.spatial_encoder = PixelSetEncoder(
            input_dim=input_dim,
            mlp1=mlp1,
            pooling=pse_pooling,
            mlp2=mlp2,
            with_extra=with_extra,
            extra_size=extra_size,
        )
        feature_dim = self.spatial_encoder.output_dim
        self.component_classifier = ComponentStructureClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            time_scale=time_scale,
            tau_fast_init=tau_fast_init,
            tau_slow_init=tau_slow_init,
            tau_min=tau_min,
            delta_tau_min=delta_tau_min,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            ltae_mlp=ltae_mlp,
            dropout=dropout,
            positional_period=positional_period,
            max_position=max_position,
            max_temporal_shift=max_temporal_shift,
            classifier_hidden=classifier_hidden,
            quality_hidden_cap=quality_hidden_cap,
            quality_eta=quality_eta,
        )
        self.joint_builder = JointStructuralSpaceBuilder(feature_dim=feature_dim)
        self.adversarial_adapter = StructuralAdversarialAdapter(
            joint_dim=self.joint_builder.joint_dim,
            hidden_dim=sda_hidden_dim,
        )

    def forward(
        self,
        pixels: torch.Tensor,
        valid_pixels: torch.Tensor,
        positions: torch.Tensor,
        extra: torch.Tensor,
        quality_progress: float = 1.0,
    ) -> torch.Tensor:
        """Return evaluation-compatible classification logits."""

        return self.forward_details(
            pixels,
            valid_pixels,
            positions,
            extra,
            quality_progress=quality_progress,
        ).logits

    def forward_details(
        self,
        pixels: torch.Tensor,
        valid_pixels: torch.Tensor,
        positions: torch.Tensor,
        extra: torch.Tensor,
        quality_progress: float = 1.0,
    ) -> StructureDAForwardOutput:
        """Run the shared PSE and component classifier for one domain batch."""

        self._validate_tensor("pixels", pixels)
        self._validate_tensor("valid_pixels", valid_pixels)
        self._validate_tensor("positions", positions)
        pse_features = self.spatial_encoder(pixels, valid_pixels, extra)
        component = self.component_classifier(
            pse_features,
            positions,
            time_mask=None,
            quality_progress=quality_progress,
        )
        return StructureDAForwardOutput(
            pse_features=pse_features,
            component=component,
        )

    def build_joint_structure(
        self, output: StructureDAForwardOutput
    ) -> JointStructuralOutput:
        """Build the quality-weighted joint structure via the owned builder."""

        if not isinstance(output, StructureDAForwardOutput):
            raise ValueError("output must be a StructureDAForwardOutput")
        component = output.component
        return self.joint_builder(
            component.trend_temporal.statistic,
            component.dynamics_temporal.statistic,
            component.dynamics_channel.statistic,
            component.effective_gates.beta_trend_temporal,
            component.effective_gates.beta_dynamics_temporal,
            component.effective_gates.beta_dynamics_channel,
        )

    def adapt(
        self,
        source_output: StructureDAForwardOutput,
        target_output: StructureDAForwardOutput,
        grl_coefficient: float,
    ) -> StructuralAdversarialOutput:
        """Apply the owned shared SDA adapter to two detailed outputs."""

        source_joint = self.build_joint_structure(source_output)
        target_joint = self.build_joint_structure(target_output)
        return self.adversarial_adapter(
            source_joint.joint,
            target_joint.joint,
            grl_coefficient=grl_coefficient,
        )

    @staticmethod
    def _validate_tensor(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
