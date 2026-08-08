from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import methods.structure_da.target_hypothesis_scan as scan_module

from methods.structure_da import (
    DeviceBatchLoader,
    PairwiseClassAlignment,
    FdasrsfCurveRegistrationAdapter,
    PhaseHypothesisScanConfig,
    SourcePrototypeBank,
    SourceRegistrationPrototypeBank,
    TSStructureModel,
    TargetPhaseHypothesisScanner,
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


def _stage1_bank(num_classes: int = 3, *, q_outer: float = 0.9) -> SourcePrototypeBank:
    return SourcePrototypeBank(
        trend_srvf=torch.zeros(num_classes, 64, 4),
        shape_srvf=torch.zeros(num_classes, 64, 4),
        trend_support=torch.ones(num_classes, 64),
        shape_support=torch.ones(num_classes, 64),
        fused=torch.zeros(num_classes, 8),
        class_counts=torch.full((num_classes,), 4, dtype=torch.long),
        ready=torch.ones(num_classes, dtype=torch.bool),
        q_distance_samples=tuple(
            torch.tensor([0.1, 0.2, 0.4, 0.8]) for _ in range(num_classes)
        ),
        f_distance_samples=tuple(torch.zeros(0) for _ in range(num_classes)),
        q_quantiles=torch.tensor([[0.3, 0.5, q_outer]] * num_classes),
        f_quantiles=torch.zeros(num_classes, 3),
        version=1,
    )


def _reg_bank(num_classes: int = 3, *, trend_srvf: torch.Tensor | None = None) -> SourceRegistrationPrototypeBank:
    if trend_srvf is None:
        trend_srvf = torch.zeros(num_classes, 128, 4)
    return SourceRegistrationPrototypeBank(
        trend_srvf=trend_srvf,
        trend_support=torch.ones(num_classes, 128),
        class_counts=torch.full((num_classes,), 4, dtype=torch.long),
        ready=torch.ones(num_classes, dtype=torch.bool),
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


class _IdentityAdapter:
    def __init__(self):
        self.calls = 0

    def register(self, source, target):
        self.calls += 1
        return torch.linspace(0.0, 1.0, source.shape[0], dtype=torch.float64)


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
        adapter=_IdentityAdapter(),
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
        adapter=_IdentityAdapter(),
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
        reg_extractor=reg_extractor, adapter=_IdentityAdapter(),
    )
    result_b = scan_target_class_phase_hypotheses(
        model, loader_b, _stage1_bank(), _reg_bank(), _scan_config(),
        device=torch.device("cpu"), shape_extractor=shape_extractor,
        reg_extractor=reg_extractor, adapter=_IdentityAdapter(),
    )
    assert _candidate_signature(result_a) == _candidate_signature(result_b)
    assert len(result_a.pairwise_alignments) == len(result_b.pairwise_alignments)
    for aa, ab in zip(result_a.pairwise_alignments, result_b.pairwise_alignments):
        assert (aa.sample_id, aa.class_id) == (ab.sample_id, ab.class_id)
        if aa.gamma is not None and ab.gamma is not None:
            torch.testing.assert_close(aa.gamma, ab.gamma, rtol=0, atol=0)
    assert len(result_a.hypotheses) == len(result_b.hypotheses)
    for ha, hb in zip(result_a.hypotheses, result_b.hypotheses):
        assert ha.class_id == hb.class_id
        torch.testing.assert_close(ha.gamma, hb.gamma, rtol=0, atol=0)


def test_solver_failure_isolated_to_one_pair() -> None:
    model = _model().eval()
    shape_extractor, reg_extractor = _extractors(model)
    loader = DataLoader(TinyParcelDataset(n=4, labels=False), batch_size=2)

    class _FlakyAdapter:
        def __init__(self):
            self.calls = 0
        def register(self, source, target):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic solver failure")
            return torch.linspace(0.0, 1.0, source.shape[0], dtype=torch.float64)


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
        reg_extractor=reg_extractor, adapter=_IdentityAdapter(),
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


def test_scan_keeps_exact_dp_solver_inputs_cpu_native() -> None:
    device = torch.device("cpu")
    model = _model().to(device).eval()
    shape_extractor, reg_extractor = _extractors(model)
    reg_extractor = reg_extractor.to(device)
    loader = DeviceBatchLoader(
        DataLoader(TinyParcelDataset(n=4, labels=False), batch_size=2), device
    )

    class _CpuIdentityAdapter:
        def register(self, source, target):
            assert source.device.type == "cpu"
            assert target.device.type == "cpu"
            return torch.linspace(
                0.0, 1.0, source.shape[0], dtype=torch.float64, device="cpu"
            )

    result = scan_target_class_phase_hypotheses(
        model,
        loader,
        _stage1_bank(),
        _reg_bank(),
        _scan_config(),
        device=device,
        shape_extractor=shape_extractor,
        reg_extractor=reg_extractor,
        adapter=_CpuIdentityAdapter(),
    )

    assert result.num_samples == 4
    assert all(hypothesis.gamma.device.type == "cpu" for hypothesis in result.hypotheses)


def test_formal_scan_runs_exact_dp_for_every_ready_class() -> None:
    num_classes = 10
    model = _model(num_classes=num_classes).eval()
    shape_extractor, reg_extractor = _extractors(model)
    loader = DataLoader(TinyParcelDataset(n=4, labels=False), batch_size=2)
    adapter = _IdentityAdapter()
    result = scan_target_class_phase_hypotheses(
        model, loader, _stage1_bank(num_classes), _reg_bank(num_classes), _scan_config(),
        device=torch.device("cpu"), shape_extractor=shape_extractor,
        reg_extractor=reg_extractor, adapter=adapter,
    )
    assert result.num_all_class_pairs == 4 * 10
    assert result.num_pairwise_attempted == 4 * 10
    assert result.num_solver_calls == 4 * 10
    assert adapter.calls == 4 * 10
    assert len(result.pairwise_alignments) == 4 * 10
    assert result.pairwise_class_counts == (4,) * 10
    # Deprecated proposal diagnostics are aliases only; there is no pruning.
    assert result.num_proposal_pairs == result.num_all_class_pairs
    assert result.proposal_class_counts == result.pairwise_class_counts


def test_progressive_scanner_only_solves_new_nested_evidence() -> None:
    model = _model().eval()
    shape_extractor, reg_extractor = _extractors(model)
    loader = DataLoader(TinyParcelDataset(n=8, labels=False), batch_size=4)
    adapter = _IdentityAdapter()
    scanner = TargetPhaseHypothesisScanner(
        model,
        loader,
        _stage1_bank(),
        _reg_bank(),
        _scan_config(),
        device=torch.device("cpu"),
        shape_extractor=shape_extractor,
        reg_extractor=reg_extractor,
        evidence_seed=3,
        adapter=adapter,
    )

    first = scanner.scan_to_budget(4)
    calls_after_first = adapter.calls
    second = scanner.scan_to_budget(8)

    assert first.num_samples == 4
    assert second.num_samples == 8
    assert set(first.scanned_sample_ids).issubset(second.scanned_sample_ids)
    assert adapter.calls == second.num_solver_calls
    assert adapter.calls > calls_after_first
    assert second.num_solver_calls - first.num_solver_calls == 4 * 3


def _alignment(
    *,
    class_id: int,
    q_distance: float,
    q_percentile: float,
    eligible: bool = True,
    sample_id: int = 0,
    reasons: tuple[str, ...] = (),
) -> PairwiseClassAlignment:
    gamma = torch.linspace(0.0, 1.0, 128, dtype=torch.float64)
    return PairwiseClassAlignment(
        sample_id=sample_id,
        class_id=class_id,
        gamma=gamma,
        t_identity_error=1.0,
        t_registered_error=0.5,
        t_gain_ratio=0.5,
        pre_common_support_t=1.0,
        common_support_t=1.0,
        gamma_finite=True,
        gamma_endpoint_error=0.0,
        gamma_strictly_increasing=True,
        gamma_min_increment=1.0 / 127.0,
        gamma_max_local_speed=1.0,
        gamma_roughness=0.0,
        phase_deviation=0.0,
        q_shape_distance=q_distance,
        q_distance_percentile=q_percentile,
        common_support_shape=1.0,
        numerically_valid=True,
        phase_evidence_eligible=eligible,
        reject_reasons=reasons,
    )


def _candidate_signature(result):
    return tuple(
        (
            item.sample_id,
            item.class_id,
            item.secondary_class_id,
            item.ambiguous,
        )
        for item in result.candidate_pseudo_labels
    )


def test_candidate_pseudo_label_ignores_classifier_logits() -> None:
    model_a = _model().eval()
    model_b = copy.deepcopy(model_a).eval()
    with torch.no_grad():
        final = model_b.classifier[-1]
        final.weight.zero_()
        final.bias.copy_(torch.tensor([-100.0, -50.0, 100.0]))

    loader = DataLoader(TinyParcelDataset(n=5, labels=False), batch_size=5)
    shape_a, reg_a = _extractors(model_a)
    shape_b, reg_b = _extractors(model_b)
    result_a = scan_target_class_phase_hypotheses(
        model_a, loader, _stage1_bank(), _reg_bank(), _scan_config(),
        device=torch.device("cpu"), shape_extractor=shape_a, reg_extractor=reg_a,
        adapter=_IdentityAdapter(),
    )
    result_b = scan_target_class_phase_hypotheses(
        model_b, loader, _stage1_bank(), _reg_bank(), _scan_config(),
        device=torch.device("cpu"), shape_extractor=shape_b, reg_extractor=reg_b,
        adapter=_IdentityAdapter(),
    )
    assert _candidate_signature(result_a) == _candidate_signature(result_b)


def test_candidate_pseudo_label_does_not_use_identity_trend_ranking() -> None:
    model = _model().eval()
    shape_extractor, reg_extractor = _extractors(model)
    loader = DataLoader(TinyParcelDataset(n=5, labels=False), batch_size=5)
    normal = _reg_bank()
    distorted = _reg_bank(
        trend_srvf=torch.stack(
            [torch.full((128, 4), float(scale)) for scale in (100.0, -50.0, 7.0)]
        )
    )
    result_a = scan_target_class_phase_hypotheses(
        model, loader, _stage1_bank(), normal, _scan_config(),
        device=torch.device("cpu"), shape_extractor=shape_extractor, reg_extractor=reg_extractor,
        adapter=_IdentityAdapter(),
    )
    result_b = scan_target_class_phase_hypotheses(
        model, loader, _stage1_bank(), distorted, _scan_config(),
        device=torch.device("cpu"), shape_extractor=shape_extractor, reg_extractor=reg_extractor,
        adapter=_IdentityAdapter(),
    )
    assert _candidate_signature(result_a) == _candidate_signature(result_b)


def test_candidate_pseudo_label_uses_raw_q_distance_not_percentile() -> None:
    candidate, hypotheses = scan_module._select_candidate_and_hypotheses(
        [
            _alignment(class_id=0, q_distance=0.20, q_percentile=0.95),
            _alignment(class_id=1, q_distance=0.30, q_percentile=0.05),
        ],
        ambiguity_margin=0.05,
    )
    assert candidate is not None
    assert candidate.class_id == 0
    assert [item.class_id for item in hypotheses] == [0]


def test_clear_top1_only_contributes_phase_evidence() -> None:
    candidate, hypotheses = scan_module._select_candidate_and_hypotheses(
        [
            _alignment(class_id=0, q_distance=0.10, q_percentile=0.8),
            _alignment(class_id=1, q_distance=0.50, q_percentile=0.1),
        ],
        ambiguity_margin=0.20,
    )
    assert candidate is not None
    assert candidate.class_id == 0
    assert candidate.ambiguous is False
    assert candidate.secondary_class_id is None
    assert len(hypotheses) == 1
    assert hypotheses[0].class_id == 0
    assert hypotheses[0].preferred is True
    assert hypotheses[0].evidence_weight == pytest.approx(1.0)


def test_near_tie_keeps_at_most_two_phase_hypotheses() -> None:
    candidate, hypotheses = scan_module._select_candidate_and_hypotheses(
        [
            _alignment(class_id=0, q_distance=0.10, q_percentile=0.9),
            _alignment(class_id=1, q_distance=0.15, q_percentile=0.2),
            _alignment(class_id=2, q_distance=0.50, q_percentile=0.1),
        ],
        ambiguity_margin=0.10,
    )
    assert candidate is not None
    assert candidate.class_id == 0
    assert candidate.ambiguous is True
    assert candidate.secondary_class_id == 1
    assert [item.class_id for item in hypotheses] == [0, 1]
    assert all(item.ambiguous_class for item in hypotheses)
    assert [item.evidence_weight for item in hypotheses] == pytest.approx([0.5, 0.5])


def test_other_reliability_failure_keeps_candidate_pseudo_label_but_not_phase_evidence() -> None:
    candidate, hypotheses = scan_module._select_candidate_and_hypotheses(
        [
            _alignment(
                class_id=0,
                q_distance=0.10,
                q_percentile=1.0,
                eligible=False,
                reasons=("gain",),
            ),
            _alignment(class_id=1, q_distance=0.50, q_percentile=0.2),
        ],
        ambiguity_margin=0.05,
    )
    assert candidate is not None
    assert candidate.class_id == 0
    assert candidate.phase_evidence_eligible is False
    # Class 1 cannot replace the geometry argmin merely because class 0 failed
    # an evidence-reliability gate.
    assert hypotheses == ()


def test_all_successful_pairwise_gammas_are_cached() -> None:
    model = _model().eval()
    shape_extractor, reg_extractor = _extractors(model)
    loader = DataLoader(TinyParcelDataset(n=4, labels=False), batch_size=2)
    result = scan_target_class_phase_hypotheses(
        model, loader, _stage1_bank(), _reg_bank(), _scan_config(),
        device=torch.device("cpu"), shape_extractor=shape_extractor,
        reg_extractor=reg_extractor, adapter=_IdentityAdapter(),
    )
    assert len(result.pairwise_alignments) == 4 * 3
    assert all(item.gamma is not None for item in result.pairwise_alignments)
    assert all(item.gamma.shape == (128,) for item in result.pairwise_alignments if item.gamma is not None)
    assert {(item.sample_id, item.class_id) for item in result.pairwise_alignments} == {
        (sample_id, class_id) for sample_id in range(4) for class_id in range(3)
    }


def test_support_gate_does_not_prune_classes_before_pseudo_label() -> None:
    model = _model().eval()
    shape_extractor, reg_extractor = _extractors(model)
    loader = DataLoader(TinyParcelDataset(n=2, labels=False), batch_size=2)
    adapter = _IdentityAdapter()
    result = scan_target_class_phase_hypotheses(
        model, loader, _stage1_bank(), _reg_bank(),
        _scan_config(registration_min_common_support=2.0),
        device=torch.device("cpu"), shape_extractor=shape_extractor,
        reg_extractor=reg_extractor, adapter=adapter,
    )
    assert adapter.calls == 2 * 3
    assert len(result.candidate_pseudo_labels) == 2
    assert all(not item.phase_evidence_eligible for item in result.candidate_pseudo_labels)
    assert result.num_pre_support_rejected == 2 * 3


def test_source_outer_range_is_diagnostic_only_for_phase_discovery() -> None:
    model = _model().eval()
    shape_extractor, reg_extractor = _extractors(model)
    loader = DataLoader(TinyParcelDataset(n=3, labels=False), batch_size=3)
    result = scan_target_class_phase_hypotheses(
        model, loader, _stage1_bank(q_outer=-1.0), _reg_bank(), _scan_config(),
        device=torch.device("cpu"), shape_extractor=shape_extractor,
        reg_extractor=reg_extractor, adapter=_IdentityAdapter(),
    )
    assert len(result.candidate_pseudo_labels) == 3
    assert all(item.phase_evidence_eligible for item in result.candidate_pseudo_labels)
    assert len(result.hypotheses) >= 3
    assert result.num_outer_rejected == 3 * 3
    assert all(
        "shape_outer" not in item.reject_reasons
        for item in result.pairwise_alignments
        if item.q_shape_distance is not None
    )
