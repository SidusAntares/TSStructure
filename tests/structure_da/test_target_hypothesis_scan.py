from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from methods.structure_da import (
    FdasrsfCurveRegistrationAdapter,
    PhaseHypothesisScanConfig,
    SourcePrototypeBank,
    SourceRegistrationPrototypeBank,
    TSStructureModel,
    build_source_registration_prototypes,
    scan_target_class_phase_hypotheses,
)


class TinyParcelDataset(Dataset):
    """Tiny dataset of (pixels, valid, positions, [label]) parcels."""

    def __init__(self, n: int, length: int = 5, *, labels: bool = True) -> None:
        self.n = n
        self.length = length
        self.labels = labels

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(2000 + index)
        sample = {
            "pixels": torch.randn(
                self.length, 2, 4, generator=generator
            ),
            "valid_pixels": torch.ones(self.length, 4),
            "positions": torch.linspace(0, 300, self.length).round().long(),
            "parcel_index": torch.tensor(index),
        }
        if self.labels:
            sample["label"] = torch.tensor(index % 3)
        return sample


def _model(**overrides) -> TSStructureModel:
    options = dict(
        num_classes=3,
        input_dim=2,
        mlp1=(2, 4, 4),
        mlp2=(8, 4),
        trend_num_basis=4,
        structure_num_basis=4,
        canonical_grid_size=64,
        roughness_grid_size=256,
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


def _scan_config(**overrides) -> PhaseHypothesisScanConfig:
    values = dict(
        registration_lambda=0.0,
        registration_gain_ratio_max=2.0,
        registration_min_common_support=1e-4,
        registration_max_roughness=1e6,
        registration_min_increment=1e-8,
        registration_max_local_speed=1e3,
        registration_max_deviation=2.0,
        class_hypothesis_margin=0.2,
        k_reg=128,
        class_hypothesis_max=2,
    )
    values.update(overrides)
    return PhaseHypothesisScanConfig(**values)


def _stage1_bank() -> SourcePrototypeBank:
    return SourcePrototypeBank(
        trend_srvf=torch.zeros(3, 64, 4),
        shape_srvf=torch.zeros(3, 64, 4),
        trend_support=torch.ones(3, 64),
        shape_support=torch.ones(3, 64),
        fused=torch.zeros(3, 8),
        class_counts=torch.tensor([4, 4, 4]),
        ready=torch.ones(3, dtype=torch.bool),
        q_distance_samples=(
            torch.tensor([0.1, 0.2, 0.4, 0.8]),
            torch.tensor([0.1, 0.2, 0.4, 0.8]),
            torch.tensor([0.1, 0.2, 0.4, 0.8]),
        ),
        f_distance_samples=(torch.zeros(0), torch.zeros(0), torch.zeros(0)),
        q_quantiles=torch.tensor([[0.3, 0.5, 0.9]] * 3),
        f_quantiles=torch.zeros(3, 3),
        version=1,
    )


def _reg_bank() -> SourceRegistrationPrototypeBank:
    return SourceRegistrationPrototypeBank(
        trend_srvf=torch.zeros(3, 128, 4),
        trend_support=torch.ones(3, 128),
        class_counts=torch.tensor([4, 4, 4]),
        ready=torch.ones(3, dtype=torch.bool),
        registration_grid=torch.linspace(0, 1, 128),
    )


def _extractors(model):
    structure = model.temporal_module.structure_geometry
    shape_extractor = structure
    reg_extractor = type(structure)(
        feature_dim=model.backbone.feature_dim,
        num_basis=structure.functional_lift.num_basis,
        canonical_grid_size=128,
        roughness_grid_size=structure.functional_lift.roughness_grid_size,
        smoothing_weight=structure.functional_lift.smoothing_weight,
        time_reference=0.0,
        time_scale=1.0,
        min_mean_support=0.0,
        min_dynamic_energy=0.0,
    )
    return shape_extractor, reg_extractor


def test_scan_uses_k_reg_128_and_k_shape_64() -> None:
    model = _model().eval()
    shape_extractor, reg_extractor = _extractors(model)
    loader = DataLoader(TinyParcelDataset(n=6, labels=False), batch_size=2)
    config = _scan_config()

    result = scan_target_class_phase_hypotheses(
        model,
        loader,
        _stage1_bank(),
        _reg_bank(),
        config,
        device=torch.device("cpu"),
        shape_extractor=shape_extractor,
        reg_extractor=reg_extractor,
    )

    assert result.num_samples == 6
    for hypothesis in result.hypotheses:
        assert hypothesis.gamma.shape == (128,)
    # At most two hypotheses per sample.
    assert result.samples_with_two_hypotheses <= result.num_samples
    assert len(result.hypotheses) <= 2 * result.num_samples


def test_scan_is_no_grad_and_no_backward() -> None:
    model = _model().eval()
    shape_extractor, reg_extractor = _extractors(model)
    loader = DataLoader(TinyParcelDataset(n=4, labels=False), batch_size=2)
    result = scan_target_class_phase_hypotheses(
        model,
        loader,
        _stage1_bank(),
        _reg_bank(),
        _scan_config(),
        device=torch.device("cpu"),
        shape_extractor=shape_extractor,
        reg_extractor=reg_extractor,
    )
    for hypothesis in result.hypotheses:
        assert hypothesis.gamma.requires_grad is False
    for parameter in model.parameters():
        assert parameter.grad is None


def test_scan_ignores_target_labels() -> None:
    # Two loaders with identical features/times/masks but different labels.
    dataset_a = TinyParcelDataset(n=6, labels=True)
    dataset_b = TinyParcelDataset(n=6, labels=True)
    # Force different labels: copy dataset_a and flip labels deterministically.
    class _Flipped(Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, index):
            item = dict(self.base[index])
            item["label"] = torch.tensor((int(item["label"].item()) + 1) % 3)
            return item

    loader_a = DataLoader(dataset_a, batch_size=2)
    loader_b = DataLoader(_Flipped(dataset_b), batch_size=2)

    model = _model().eval()
    shape_extractor, reg_extractor = _extractors(model)

    result_a = scan_target_class_phase_hypotheses(
        model, loader_a, _stage1_bank(), _reg_bank(), _scan_config(),
        device=torch.device("cpu"), shape_extractor=shape_extractor,
        reg_extractor=reg_extractor,
    )
    result_b = scan_target_class_phase_hypotheses(
        model, loader_b, _stage1_bank(), _reg_bank(), _scan_config(),
        device=torch.device("cpu"), shape_extractor=shape_extractor,
        reg_extractor=reg_extractor,
    )
    assert len(result_a.hypotheses) == len(result_b.hypotheses)
    for ha, hb in zip(result_a.hypotheses, result_b.hypotheses):
        assert ha.class_id == hb.class_id
        torch.testing.assert_close(ha.gamma, hb.gamma, rtol=0, atol=0)


def test_solver_failure_isolated_to_one_pair() -> None:
    model = _model().eval()
    shape_extractor, reg_extractor = _extractors(model)
    loader = DataLoader(TinyParcelDataset(n=4, labels=False), batch_size=2)

    class _FlakyAdapter(FdasrsfCurveRegistrationAdapter):
        def __init__(self):
            super().__init__(registration_lambda=0.0)
            self.calls = 0
        def register(self, source, target):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic solver failure")
            return super().register(source, target)

    adapter = _FlakyAdapter()
    result = scan_target_class_phase_hypotheses(
        model, loader, _stage1_bank(), _reg_bank(), _scan_config(),
        device=torch.device("cpu"), shape_extractor=shape_extractor,
        reg_extractor=reg_extractor, adapter=adapter,
    )
    # The scan survives the one failing pair.
    assert result.num_solver_failed == 1
    assert isinstance(result.hypotheses, tuple)


def test_scan_restores_training_state() -> None:
    model = _model().train()
    shape_extractor, reg_extractor = _extractors(model)
    loader = DataLoader(TinyParcelDataset(n=4, labels=False), batch_size=2)
    assert model.training is True
    scan_target_class_phase_hypotheses(
        model, loader, _stage1_bank(), _reg_bank(), _scan_config(),
        device=torch.device("cpu"), shape_extractor=shape_extractor,
        reg_extractor=reg_extractor,
    )
    assert model.training is True


def test_build_source_registration_prototypes_k_reg() -> None:
    model = _model().eval()
    loader = DataLoader(TinyParcelDataset(n=12, labels=True), batch_size=4)
    shape_extractor, reg_extractor = _extractors(model)
    bank = build_source_registration_prototypes(
        model,
        loader,
        3,
        device=torch.device("cpu"),
        reg_extractor=reg_extractor,
    )
    assert bank.trend_srvf.shape == (3, 128, 4)
    assert bank.trend_support.shape == (3, 128)
    assert bank.ready.tolist() == [True, True, True]
    assert bank.class_counts.tolist() == [4, 4, 4]
