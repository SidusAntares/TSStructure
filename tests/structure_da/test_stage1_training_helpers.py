from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from methods.structure_da import (
    SourceClassificationTrainer,
    SourcePrototypeBank,
    Stage1Objective,
    TSStructureModel,
    build_source_prototype_bank,
)


class TinySourceDataset(Dataset):
    """Deterministic tiny source dataset with 3 classes."""

    def __init__(self, n: int = 24, length: int = 5, *, num_pixels: int = 4) -> None:
        self.n = n
        self.length = length
        self.num_pixels = num_pixels
        self.labels = torch.arange(n) % 3

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(1000 + index)
        return {
            "pixels": torch.randn(
                self.length, 2, self.num_pixels, generator=generator
            ),
            "valid_pixels": torch.ones(self.length, self.num_pixels),
            "positions": torch.linspace(0, 300, self.length).round().long(),
            "label": int(self.labels[index]),
            "parcel_index": index,
        }


def _model(**overrides) -> TSStructureModel:
    options = dict(
        num_classes=3,
        input_dim=2,
        mlp1=(2, 4, 4),
        mlp2=(8, 4),
        trend_num_basis=4,
        structure_num_basis=4,
        canonical_grid_size=5,
        roughness_grid_size=64,
        n_head=1,
        d_k=2,
        d_model=8,
        ltae_mlp=(8, 4),
        dropout=0.0,
        classifier_hidden=(4,),
        max_initial_frequency=4.0,
    )
    options.update(overrides)
    return TSStructureModel(**options)


def _batch(batch_size: int = 6, length: int = 5) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    return {
        "pixels": torch.randn(batch_size, length, 2, 4),
        "valid_pixels": torch.ones(batch_size, length, 4, dtype=torch.bool),
        "positions": torch.linspace(0, 300, length).round().long().expand(batch_size, -1),
        "label": torch.tensor([i % 3 for i in range(batch_size)]),
    }


def _objective() -> Stage1Objective:
    return Stage1Objective(
        num_classes=3,
        lambda_q=0.1,
        lambda_f=0.1,
        lambda_q_to_cls=0.1,
        margin_q=0.1,
        margin_f=0.1,
        tau_q=0.1,
    )


def _bank() -> SourcePrototypeBank:
    return SourcePrototypeBank(
        trend_srvf=torch.zeros(3, 5, 4),
        shape_srvf=torch.zeros(3, 5, 4),
        trend_support=torch.ones(3, 5),
        shape_support=torch.ones(3, 5),
        fused=torch.zeros(3, 8),
        class_counts=torch.tensor([8, 8, 8]),
        ready=torch.ones(3, dtype=torch.bool),
        q_distance_samples=(torch.zeros(0), torch.zeros(0), torch.zeros(0)),
        f_distance_samples=(torch.zeros(0), torch.zeros(0), torch.zeros(0)),
        q_quantiles=torch.zeros(3, 3),
        f_quantiles=torch.zeros(3, 3),
        version=1,
    )
