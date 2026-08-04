"""Independent shape/phase encoders and temporal structure output head."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .temporal_coordinates import TemporalCoordinateOutput


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
class TemporalStructureFeatureOutput:
    feature: Tensor
    shape_embedding: Tensor
    phase_embedding: Tensor
    joint_embedding: Tensor
    valid: Tensor


@dataclass(frozen=True)
class ShapeFeatureOutput:
    feature: Tensor
    valid: Tensor


class ShapeCoordinateEncoder(nn.Module):
    """Encode flattened shape coordinates with an independent two-layer MLP."""

    def __init__(
        self,
        num_shape_basis: int,
        attribute_projection_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_shape_basis = _positive_integer(
            "num_shape_basis", num_shape_basis
        )
        self.attribute_projection_dim = _positive_integer(
            "attribute_projection_dim", attribute_projection_dim
        )
        self.hidden_dim = _positive_integer("hidden_dim", hidden_dim)
        self.dropout = _dropout_probability(dropout)
        self.network = nn.Sequential(
            nn.Linear(
                self.num_shape_basis * self.attribute_projection_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(
        self,
        shape_coordinates: Tensor,
        valid: Tensor,
    ) -> Tensor:
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
        flattened = shape_coordinates.reshape(
            shape_coordinates.shape[0],
            self.num_shape_basis * self.attribute_projection_dim,
        )
        embedding = self.network(flattened)
        return torch.where(
            valid.unsqueeze(-1),
            embedding,
            torch.zeros_like(embedding),
        )


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


class PhaseCoordinateEncoder(nn.Module):
    """Encode phase coordinates with a separate two-layer MLP."""

    def __init__(
        self,
        num_phase_basis: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_phase_basis = _positive_integer(
            "num_phase_basis", num_phase_basis
        )
        self.hidden_dim = _positive_integer("hidden_dim", hidden_dim)
        self.dropout = _dropout_probability(dropout)
        self.network = nn.Sequential(
            nn.Linear(self.num_phase_basis + 1, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(
        self,
        phase_coordinates: Tensor,
        valid: Tensor,
    ) -> Tensor:
        _validate_floating_tensor(
            "phase_coordinates",
            phase_coordinates,
            expected_tail=(self.num_phase_basis + 1,),
            parameter=self.network[0].weight,
        )
        _validate_valid_mask(
            valid,
            batch_size=phase_coordinates.shape[0],
            device=phase_coordinates.device,
        )
        embedding = self.network(phase_coordinates)
        return torch.where(
            valid.unsqueeze(-1),
            embedding,
            torch.zeros_like(embedding),
        )


class TemporalStructureOutputHead(nn.Module):
    """Project a joint coordinate embedding to the temporal structure space."""

    def __init__(
        self,
        joint_dim: int,
        structure_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.joint_dim = _positive_integer("joint_dim", joint_dim)
        self.structure_dim = _positive_integer(
            "structure_dim", structure_dim
        )
        dropout = _dropout_probability(dropout)
        self.projection = nn.Linear(self.joint_dim, self.structure_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(self.structure_dim)

    def forward(
        self,
        joint_embedding: Tensor,
        valid: Tensor,
    ) -> Tensor:
        _validate_floating_tensor(
            "joint_embedding",
            joint_embedding,
            expected_tail=(self.joint_dim,),
            parameter=self.projection.weight,
        )
        _validate_valid_mask(
            valid,
            batch_size=joint_embedding.shape[0],
            device=joint_embedding.device,
        )
        feature = self.projection(joint_embedding)
        feature = self.activation(feature)
        feature = self.dropout(feature)
        feature = self.normalization(feature)
        return torch.where(
            valid.unsqueeze(-1),
            feature,
            torch.zeros_like(feature),
        )


class TemporalStructureEncoder(nn.Module):
    """Encode explicit shape and phase coordinates into one temporal feature."""

    def __init__(
        self,
        num_shape_basis: int,
        attribute_projection_dim: int,
        num_phase_basis: int,
        coordinate_hidden_dim: int = 64,
        structure_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_shape_basis = _positive_integer(
            "num_shape_basis", num_shape_basis
        )
        self.attribute_projection_dim = _positive_integer(
            "attribute_projection_dim", attribute_projection_dim
        )
        self.num_phase_basis = _positive_integer(
            "num_phase_basis", num_phase_basis
        )
        self.coordinate_hidden_dim = _positive_integer(
            "coordinate_hidden_dim", coordinate_hidden_dim
        )
        self.structure_dim = _positive_integer(
            "structure_dim", structure_dim
        )
        dropout = _dropout_probability(dropout)

        self.shape_encoder = ShapeCoordinateEncoder(
            num_shape_basis=self.num_shape_basis,
            attribute_projection_dim=self.attribute_projection_dim,
            hidden_dim=self.coordinate_hidden_dim,
            dropout=dropout,
        )
        self.phase_encoder = PhaseCoordinateEncoder(
            num_phase_basis=self.num_phase_basis,
            hidden_dim=self.coordinate_hidden_dim,
            dropout=dropout,
        )
        self.output_head = TemporalStructureOutputHead(
            joint_dim=2 * self.coordinate_hidden_dim,
            structure_dim=self.structure_dim,
            dropout=dropout,
        )

    def forward(
        self,
        coordinates: TemporalCoordinateOutput,
    ) -> TemporalStructureFeatureOutput:
        if not isinstance(coordinates, TemporalCoordinateOutput):
            raise ValueError(
                "coordinates must be a TemporalCoordinateOutput"
            )
        shape_coordinates = coordinates.shape_coordinates
        phase_coordinates = coordinates.phase_coordinates
        valid = coordinates.valid

        _validate_floating_tensor(
            "shape_coordinates",
            shape_coordinates,
            expected_tail=(
                self.num_shape_basis,
                self.attribute_projection_dim,
            ),
            parameter=self.shape_encoder.network[0].weight,
        )
        _validate_floating_tensor(
            "phase_coordinates",
            phase_coordinates,
            expected_tail=(self.num_phase_basis + 1,),
            parameter=self.phase_encoder.network[0].weight,
        )
        if shape_coordinates.shape[0] != phase_coordinates.shape[0]:
            raise ValueError(
                "shape and phase coordinate batch dimensions must match"
            )
        if shape_coordinates.dtype != phase_coordinates.dtype:
            raise ValueError("shape and phase coordinate dtype must match")
        if shape_coordinates.device != phase_coordinates.device:
            raise ValueError("shape and phase coordinate device must match")
        _validate_valid_mask(
            valid,
            batch_size=shape_coordinates.shape[0],
            device=shape_coordinates.device,
        )

        shape_embedding = self.shape_encoder(shape_coordinates, valid)
        phase_embedding = self.phase_encoder(phase_coordinates, valid)
        if shape_embedding.dtype != phase_embedding.dtype:
            raise RuntimeError(
                "shape and phase embeddings must use the same dtype"
            )
        joint_embedding = torch.cat(
            [shape_embedding, phase_embedding], dim=-1
        )
        joint_embedding = torch.where(
            valid.unsqueeze(-1),
            joint_embedding,
            torch.zeros_like(joint_embedding),
        )
        feature = self.output_head(joint_embedding, valid)
        return TemporalStructureFeatureOutput(
            feature=feature,
            shape_embedding=shape_embedding,
            phase_embedding=phase_embedding,
            joint_embedding=joint_embedding,
            valid=valid,
        )
