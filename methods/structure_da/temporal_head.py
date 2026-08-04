"""Shape-only feature encoding for the V3 temporal path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F

def _positive_integer(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dropout_probability(value: float) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("dropout must be a finite number in [0, 1)")
    converted = float(value)
    if not 0.0 <= converted < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return converted


def _validate_valid_mask(
    valid: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    if (
        not isinstance(valid, Tensor)
        or valid.dtype != torch.bool
        or valid.shape != (batch_size,)
    ):
        raise ValueError("valid must be a boolean tensor with shape [B]")
    if valid.device != device:
        raise ValueError("valid device must match coordinate device")


def _autocast_enabled_for(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.is_autocast_enabled()
    if device.type == "cpu":
        return torch.is_autocast_cpu_enabled()
    return False


def _validate_floating_tensor(
    name: str,
    tensor: Tensor,
    *,
    expected_tail: tuple[int, ...],
    parameter: Tensor,
) -> None:
    expected_ndim = len(expected_tail) + 1
    if not isinstance(tensor, Tensor) or tensor.ndim != expected_ndim:
        shape = ", ".join(["B", *(str(value) for value in expected_tail)])
        raise ValueError(f"{name} must have shape [{shape}]")
    if tensor.shape[1:] != expected_tail:
        shape = ", ".join(["B", *(str(value) for value in expected_tail)])
        raise ValueError(f"{name} must have shape [{shape}]")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if tensor.device != parameter.device:
        raise ValueError(f"{name} device must match module parameter device")
    if tensor.dtype != parameter.dtype:
        if not _autocast_enabled_for(tensor.device):
            raise ValueError(
                f"{name} dtype must match module parameter dtype when "
                "autocast is disabled"
            )
        if parameter.dtype != torch.float32:
            raise ValueError(
                f"{name} mixed-precision input requires float32 master "
                "parameters"
            )
        if tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(f"{name} uses an unsupported autocast dtype")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class ShapeFeatureOutput:
    feature: Tensor
    valid: Tensor


class ShapeFeatureEncoder(nn.Module):
    """Encode Shape-only coordinates directly into the final shape feature."""

    def __init__(
        self,
        num_shape_basis: int,
        attribute_projection_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_shape_basis = _positive_integer(
            "num_shape_basis", num_shape_basis
        )
        self.attribute_projection_dim = _positive_integer(
            "attribute_projection_dim", attribute_projection_dim
        )
        self.output_dim = _positive_integer("output_dim", output_dim)
        self.hidden_dim = _positive_integer("hidden_dim", hidden_dim)
        self.dropout = _dropout_probability(dropout)
        self.network = nn.Sequential(
            nn.Linear(
                self.num_shape_basis * self.attribute_projection_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(
        self,
        shape_coordinates: Tensor,
        valid: Tensor,
        *,
        deterministic: bool = False,
    ) -> ShapeFeatureOutput:
        _validate_floating_tensor(
            "shape_coordinates",
            shape_coordinates,
            expected_tail=(
                self.num_shape_basis,
                self.attribute_projection_dim,
            ),
            parameter=self.network[0].weight,
        )
        _validate_valid_mask(
            valid,
            batch_size=shape_coordinates.shape[0],
            device=shape_coordinates.device,
        )
        flattened = shape_coordinates.reshape(shape_coordinates.shape[0], -1)
        feature = self.network[0](flattened)
        feature = self.network[1](feature)
        feature = F.dropout(
            feature,
            p=self.dropout,
            training=self.training and not deterministic,
        )
        feature = self.network[3](feature)
        feature = torch.where(
            valid.unsqueeze(-1), feature, torch.zeros_like(feature)
        )
        return ShapeFeatureOutput(feature=feature, valid=valid)
