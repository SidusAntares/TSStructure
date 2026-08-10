"""Explicit Stage-2 trainable/frozen parameter ownership for Phase-only TSStructure."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class Stage2ParameterPolicy:
    trainable_parameter_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]


def configure_stage2_parameter_policy(model: nn.Module) -> Stage2ParameterPolicy:
    """Freeze PSE/decomposition/geometry; train only single-stream LTAE + classifier."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    try:
        raw = model.temporal_module.raw_encoder
        classifier = model.classifier
        model.backbone.pixel_set_encoder
        model.backbone.decomposition
        model.temporal_module.trend_geometry
        model.temporal_module.structure_geometry
    except AttributeError as error:
        raise ValueError("model does not expose the Phase-only TSStructure modules") from error

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable_modules = (
        raw.time_encoder,
        raw.input_projection,
        raw.input_norm,
        raw.attention_heads,
        raw.projection,
        raw.output_norm,
        classifier,
    )
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        (trainable if parameter.requires_grad else frozen).append(name)
    all_names = {name for name, _ in model.named_parameters()}
    if set(trainable) & set(frozen) or set(trainable) | set(frozen) != all_names:
        raise RuntimeError("Stage-2 parameter policy does not partition model parameters")
    return Stage2ParameterPolicy(tuple(trainable), tuple(frozen))
