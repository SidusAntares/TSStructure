import csv
import inspect
import io
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _source_batch(dtype=torch.float64):
    positions = torch.tensor(
        [0, 17, 39, 62, 88, 117, 149, 184, 221, 259, 300, 342],
        dtype=dtype,
    ).repeat(8, 1)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    phase = 2 * torch.pi * positions / 365.0
    curves = []
    loadings = torch.tensor(
        [[1.0, -0.4, 0.7], [-0.3, 1.0, 0.5]], dtype=dtype
    )
    for index, label in enumerate(labels):
        signal = (
            torch.sin(phase[index] + 0.5 * label)
            + 0.35 * torch.cos(2 * phase[index] - 0.2 * index)
        )
        curves.append(signal[:, None] * loadings[label][None, :])
    return torch.stack(curves), positions, labels


def _bank(modes=(9, 13), prominence_rel=0.15):
    from models.shape_alignment import SourceShapeReferenceBank

    features, positions, labels = _source_batch()
    return SourceShapeReferenceBank.from_source_features(
        features,
        positions,
        labels,
        modes=modes,
        grid_points=32,
        period_days=365.0,
        reg=1e-3,
        prominence_rel=prominence_rel,
        min_distance_days=14.0,
    )


def test_reference_bank_is_source_only_and_frozen():
    bank = _bank((9,))

    assert bank.class_projections.shape == (2, 3)
    assert bank.grid.shape == (32,)
    assert not any(parameter.requires_grad for parameter in bank.parameters())
    assert all(not value.requires_grad for value in bank.prototypes.values())


def test_shape_loss_uses_pseudo_class_and_has_no_target_label_input():
    from models.shape_alignment import ShapeAlignment

    bank = _bank((9,))
    alignment = ShapeAlignment(bank, morph_weight=1.0, event_weight=0.5)
    features, positions, _ = _source_batch()
    target = features[:2].clone().requires_grad_(True)

    class_zero = alignment(target, positions[:2], torch.zeros(2, dtype=torch.long))
    class_one = alignment(target, positions[:2], torch.ones(2, dtype=torch.long))

    assert inspect.signature(alignment.forward).parameters.keys() == {
        "spatial_features",
        "positions",
        "pseudo_classes",
        "residual_shifts",
    }
    assert not torch.allclose(class_zero.loss, class_one.loss)


def test_target_true_labels_cannot_affect_loss_selection_or_gradient():
    from models.shape_alignment import ShapeAlignment

    features, positions, _ = _source_batch()
    pseudo_classes = torch.tensor([0, 1])
    target_a = features[:2].clone().requires_grad_(True)
    target_b = features[:2].clone().requires_grad_(True)
    target_true_a = torch.tensor([0, 1])
    target_true_b = torch.tensor([1, 0])
    alignment = ShapeAlignment(_bank((9,)))

    loss_a = alignment(target_a, positions[:2], pseudo_classes).loss
    loss_b = alignment(target_b, positions[:2], pseudo_classes).loss
    loss_a.backward()
    loss_b.backward()

    assert not torch.equal(target_true_a, target_true_b)
    assert torch.equal(loss_a, loss_b)
    assert torch.equal(target_a.grad, target_b.grad)


def test_combined_modes_are_averaged_and_share_one_input_tensor():
    from models.shape_alignment import ShapeAlignment

    features, positions, labels = _source_batch()
    target = features[:3].clone().requires_grad_(True)
    classes = labels[:3]
    bank = _bank((9, 13))
    combined = ShapeAlignment(bank)(target, positions[:3], classes)
    loss9 = ShapeAlignment(_bank((9,)))(target, positions[:3], classes)
    loss13 = ShapeAlignment(_bank((13,)))(target, positions[:3], classes)

    assert torch.allclose(
        combined.loss, (loss9.loss + loss13.loss) / 2, atol=1e-10
    )
    assert set(combined.mode_losses) == {9, 13}


def test_no_pseudo_samples_is_finite_connected_zero():
    from models.shape_alignment import ShapeAlignment

    alignment = ShapeAlignment(_bank((9,)))
    features = torch.empty((0, 12, 3), dtype=torch.float64, requires_grad=True)
    positions = torch.empty((0, 12), dtype=torch.float64)
    result = alignment(features, positions, torch.empty(0, dtype=torch.long))

    assert torch.isfinite(result.loss)
    assert result.loss.item() == 0
    result.loss.backward()
    assert features.grad is not None


def test_class_without_canonical_events_keeps_morphology_and_zero_event_loss():
    from models.shape_alignment import ShapeAlignment, SourceShapeReferenceBank

    features = torch.ones((4, 12, 3), dtype=torch.float64)
    positions = torch.arange(12, dtype=torch.float64).repeat(4, 1) * 20
    labels = torch.tensor([0, 0, 1, 1])
    bank = SourceShapeReferenceBank.from_source_features(
        features,
        positions,
        labels,
        modes=(9,),
        grid_points=24,
        prominence_rel=1e6,
    )
    result = ShapeAlignment(bank)(features[:2], positions[:2], labels[:2])

    assert torch.isfinite(result.morph_loss)
    assert result.event_loss.item() == 0
    assert result.no_event_reference_count.item() == 2


def test_shape_gradient_only_needs_target_spatial_features():
    from models.shape_alignment import ShapeAlignment

    features, positions, labels = _source_batch(torch.float32)
    spatial = torch.nn.Linear(3, 3, bias=False)
    teacher = torch.nn.Linear(3, 3, bias=False)
    target = spatial(features[:2])
    alignment = ShapeAlignment(_bank((9,)).float())
    result = alignment(
        target, positions[:2], labels[:2]
    )
    result.loss.backward()

    assert spatial.weight.grad is not None
    assert spatial.weight.grad.abs().sum() > 0
    assert teacher.weight.grad is None
    assert all(parameter.grad is None for parameter in alignment.reference.parameters())


def test_reference_build_restores_all_rng_states():
    from models.shape_alignment import preserve_rng_state

    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    with preserve_rng_state(seed=99):
        _ = (random.random(), np.random.rand(), torch.rand(20))
        _bank((9,))
    actual = (random.random(), np.random.rand(), torch.rand(1))

    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_shape_module_has_no_frequency_disentangler_dependency():
    import models.shape_alignment as shape_alignment

    source = inspect.getsource(shape_alignment)
    assert "FrequencyDisentangler" not in source


class _Spatial(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixels, mask, extra):
        self.calls += 1
        return pixels.mean(dim=-1)


class _Semantic(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.spatial_encoder = _Spatial()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.temporal_encoder = SimpleNamespace(
            max_temporal_shift=100,
            positional_enc=SimpleNamespace(num_embeddings=565),
        )
        self.fail = fail

    def forward_with_temporal_shift(
        self, pixels, mask, positions, extra, temporal_shift=0
    ):
        spatial = self.spatial_encoder(pixels, mask, extra)
        if self.fail:
            raise RuntimeError("semantic failure")
        score = spatial.mean(dim=(1, 2), keepdim=False) * self.scale
        return torch.stack([score, torch.zeros_like(score)], dim=1)


def _semantic_inputs(batch=3):
    pixels = torch.randn(batch, 3, 2, 4)
    mask = torch.ones(batch, 3, 4)
    positions = torch.arange(3).repeat(batch, 1)
    extra = torch.zeros(batch, 4)
    return pixels, mask, positions, extra


def test_temporary_hook_reuses_semantic_pse_forward_and_is_removed():
    import timematch

    model = _Semantic()
    inputs = _semantic_inputs()
    logits, captured = timematch._forward_with_spatial_capture(model, *inputs)

    assert logits.shape == (3, 2)
    assert captured.shape == (3, 3, 2)
    assert model.spatial_encoder.calls == 1
    assert len(model.spatial_encoder._forward_hooks) == 0


def test_temporary_hook_is_removed_when_semantic_forward_raises():
    import timematch

    model = _Semantic(fail=True)
    with pytest.raises(RuntimeError, match="semantic failure"):
        timematch._forward_with_spatial_capture(model, *_semantic_inputs())
    assert len(model.spatial_encoder._forward_hooks) == 0


def test_pseudo_mask_is_only_admission_gate_and_shift_is_added_once():
    import timematch

    class Recorder:
        def __call__(self, spatial_features, positions, pseudo_classes):
            self.spatial = spatial_features
            self.positions = positions
            self.classes = pseudo_classes
            return "result"

    recorder = Recorder()
    captured_selected = torch.randn(2, 4, 3)
    positions = torch.tensor([[10, 20, 30, 40], [1, 2, 3, 4], [7, 8, 9, 10]])
    pseudo = torch.tensor([1, 0, 1])
    mask = torch.tensor([True, False, True])
    result = timematch._shape_loss_from_capture(
        recorder,
        captured_selected,
        positions,
        pseudo,
        mask,
        target_to_source_shift=6,
    )

    assert result == "result"
    assert recorder.spatial.data_ptr() == captured_selected.data_ptr()
    assert torch.equal(recorder.classes, torch.tensor([1, 1]))
    assert torch.equal(
        recorder.positions,
        torch.tensor([[16, 26, 36, 46], [13, 14, 15, 16]]),
    )


def test_shape_disabled_helper_does_not_register_hook_or_change_loss():
    import timematch

    model = _Semantic()
    inputs = _semantic_inputs()
    baseline = timematch._forward_with_temporal_shift(model, *inputs)
    calls = model.spatial_encoder.calls
    tm_loss = baseline.square().mean()
    total = timematch._add_shape_loss(tm_loss, None, shape_lambda=0.1)

    assert total is tm_loss
    assert model.spatial_encoder.calls == calls
    assert len(model.spatial_encoder._forward_hooks) == 0


class _Loader:
    def __init__(self, batches, labels=(0, 1)):
        self.batches = batches
        self.dataset = SimpleNamespace(get_labels=lambda: np.asarray(labels))

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _Writer:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, step):
        self.scalars[name] = value


def _training_batch(batch_size, value):
    return {
        "pixels": torch.full((batch_size, 3, 2, 4), value),
        "valid_pixels": torch.ones(batch_size, 3, 4),
        "positions": torch.arange(3).repeat(batch_size, 1),
        "extra": torch.zeros(batch_size, 4),
        "label": torch.zeros(batch_size, dtype=torch.long),
    }


@pytest.mark.parametrize(
    (
        "domain_specific_bn,expected_student_pse_calls,shape_align,"
        "class_residual_phase,expected_shape_builds,expected_shape_csv_rows"
    ),
    [
        (True, 2, True, False, 1, 1),
        (False, 1, True, False, 1, 1),
        (True, 2, True, True, 1, 1),
        (False, 1, True, True, 1, 1),
        (True, 2, False, False, 0, 0),
    ],
)
def test_shape_training_reuses_the_existing_student_semantic_forward(
    monkeypatch,
    domain_specific_bn,
    expected_student_pse_calls,
    shape_align,
    class_residual_phase,
    expected_shape_builds,
    expected_shape_csv_rows,
):
    import timematch
    from models.shape_alignment import ClassResidualPhaseResult, ShapeAlignmentResult

    source = _training_batch(2, 1.0)
    target_weak = _training_batch(2, 5.0)
    target_strong = _training_batch(2, 2.0)
    monkeypatch.setattr(
        timematch,
        "get_data_loaders",
        lambda *args, **kwargs: (
            _Loader([source]),
            _Loader([]),
            _Loader([(target_weak, target_strong)]),
        ),
    )
    monkeypatch.setattr(
        timematch,
        "to_cuda",
        lambda sample, device: (
            sample["pixels"],
            sample["valid_pixels"],
            sample["positions"],
            sample["extra"],
        ),
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    shape_csv_rows = []
    monkeypatch.setattr(
        timematch,
        "_append_shape_training_metrics",
        lambda output_dir, metrics: shape_csv_rows.append((output_dir, metrics)),
    )
    monkeypatch.setattr(
        timematch, "_write_class_phase_csvs", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        timematch,
        "_maybe_build_class_phase",
        lambda *args, **kwargs: (
            (object(), object()) if class_residual_phase else (None, None)
        ),
    )
    monkeypatch.setattr(
        timematch,
        "_estimate_class_residual_phase",
        lambda *args, **kwargs: ClassResidualPhaseResult(
            accepted_shifts=torch.zeros(2), records=[]
        ),
    )
    model = _Semantic()
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"state_dict": deepcopy(model.state_dict())},
    )
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: None)
    build_calls = []

    class FakeAlignment:
        def __call__(self, spatial, positions, classes, residual_shifts=None):
            build_calls.append((spatial, positions, classes))
            loss = spatial.mean().square()
            zero = loss.detach() * 0
            return ShapeAlignmentResult(
                loss, loss, zero, {9: loss}, zero, zero, zero.long()
            )

    monkeypatch.setattr(
        timematch,
        "_build_source_shape_alignment",
        lambda *args, **kwargs: FakeAlignment(),
    )
    config = SimpleNamespace(
        balance_source=False,
        weights="weights",
        model="pseltae",
        use_focal_loss=False,
        focal_loss_gamma=1.0,
        steps_per_epoch=1,
        lr=1e-3,
        weight_decay=0.0,
        epochs=1,
        max_temporal_shift=60,
        num_classes=2,
        estimate_shift=False,
        pseudo_threshold=0.0,
        domain_specific_bn=domain_specific_bn,
        trade_off=2.0,
        ema_decay=0.99,
        log_step=1,
        run_validation=False,
        output_student=True,
        progress_bar="off",
        shape_align=shape_align,
        shape_lambda=0.1,
        shape_diag_batches=0,
        fold_dir="unused-fold",
        class_residual_phase=class_residual_phase,
        classes=["a", "b"],
    )
    timematch.train_timematch(
        model,
        config,
        _Writer(),
        None,
        "cpu",
        "unused.pt",
        0,
        {},
    )

    assert model.spatial_encoder.calls == expected_student_pse_calls
    assert len(build_calls) == expected_shape_builds
    if build_calls:
        assert build_calls[0][0].shape[0] == 2
    assert len(shape_csv_rows) == expected_shape_csv_rows
    if shape_csv_rows:
        assert shape_csv_rows[0][0] == "unused-fold"
        assert shape_csv_rows[0][1]["selected_target_count"] == 2


def test_shape_metric_observation_does_not_change_logits_loss_or_gradient():
    import timematch
    from models.shape_alignment import ShapeAlignmentResult

    logits_without = torch.tensor([[0.4, -0.2]], requires_grad=True)
    logits_with = logits_without.detach().clone().requires_grad_(True)

    def losses(logits):
        timematch_loss = logits.square().sum()
        morph_loss = (logits[:, 0] - logits[:, 1]).square().mean()
        zero = morph_loss.detach() * 0
        shape_result = ShapeAlignmentResult(
            morph_loss,
            morph_loss,
            zero,
            {13: morph_loss},
            1 - morph_loss.detach(),
            zero,
            zero.long(),
        )
        total = timematch._add_shape_loss(timematch_loss, shape_result, 0.1)
        return timematch_loss, shape_result, total

    _, _, total_without = losses(logits_without)
    timematch_loss, shape_result, total_with = losses(logits_with)
    observed = {
        "selected": 0,
        "target": 0,
        "morph": 0.0,
        "weighted": 0.0,
        "ratio": 0.0,
        "corr": 0.0,
    }
    timematch._observe_shape_training_step(
        observed,
        shape_result,
        torch.tensor([True]),
        timematch_loss,
        shape_lambda=0.1,
    )
    total_without.backward()
    total_with.backward()

    assert torch.equal(logits_without.detach(), logits_with.detach())
    assert torch.equal(total_without.detach(), total_with.detach())
    assert torch.equal(logits_without.grad, logits_with.grad)
    assert observed["selected"] == 1
    assert all(not isinstance(value, torch.Tensor) for value in observed.values())


@pytest.mark.parametrize("modes", [(9,), (13,), (9, 13)])
def test_synthetic_forward_backward_ema_and_checkpoint_round_trip(modes):
    from models.shape_alignment import ShapeAlignment
    from timematch import update_ema_variables

    bank = _bank(modes).float()
    alignment = ShapeAlignment(bank)
    features, positions, labels = _source_batch(torch.float32)
    student = torch.nn.Linear(3, 3, bias=False)
    teacher = torch.nn.Linear(3, 3, bias=False)
    teacher.load_state_dict(student.state_dict())
    transformed = student(features[:2])
    result = alignment(transformed, positions[:2], labels[:2])
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert student.weight.grad is not None
    update_ema_variables(student, teacher, decay=0.9)

    checkpoint = io.BytesIO()
    torch.save(bank.export_payload(), checkpoint)
    checkpoint.seek(0)
    restored = type(bank).from_payload(
        torch.load(checkpoint, weights_only=False)
    )
    restored_result = ShapeAlignment(restored)(
        transformed.detach(), positions[:2], labels[:2]
    )
    assert torch.allclose(result.loss.detach(), restored_result.loss, atol=1e-5)


def _phase_reference(dtype=torch.float64):
    from models.shape_alignment import SourceShapeReferenceBank, _robust_normalize

    grid = torch.linspace(20.0, 330.0, 64, dtype=dtype)
    omega = 2.0 * torch.pi / 365.0
    prototype_a = torch.sin(3 * omega * grid) + 0.35 * torch.cos(4 * omega * grid)
    prototype_b = torch.cos(omega * grid + 0.4) - 0.25 * torch.sin(3 * omega * grid)
    prototypes = torch.stack([prototype_a, prototype_b])
    prototypes = _robust_normalize(prototypes, detach_statistics=False)
    empty = torch.zeros((2, grid.numel()), dtype=torch.bool)
    return SourceShapeReferenceBank(
        class_projections=torch.ones((2, 1), dtype=dtype),
        grid=grid,
        prototypes={9: prototypes, 13: prototypes.clone()},
        event_masks={9: empty, 13: empty.clone()},
        peak_masks={9: empty.clone(), 13: empty.clone()},
        valley_masks={9: empty.clone(), 13: empty.clone()},
        sample_counts=torch.tensor([64, 64]),
        period_days=365.0,
        reg=1e-8,
        prominence_rel=0.15,
        min_distance_days=14.0,
    )


def _phase_target(shifts, samples_per_class=36, dtype=torch.float64):
    positions = torch.linspace(0.0, 350.0, 30, dtype=dtype)
    omega = 2.0 * torch.pi / 365.0
    features = []
    timestamps = []
    classes = []
    for class_index, shift in enumerate(shifts):
        for _ in range(samples_per_class):
            shifted = positions + shift
            if class_index == 0:
                curve = torch.sin(3 * omega * shifted) + 0.35 * torch.cos(4 * omega * shifted)
            else:
                curve = torch.cos(omega * shifted + 0.4) - 0.25 * torch.sin(3 * omega * shifted)
            features.append(curve[:, None])
            timestamps.append(positions)
            classes.append(class_index)
    return torch.stack(features), torch.stack(timestamps), torch.tensor(classes)


def test_fourier_residual_shift_sign_matches_reanalysis():
    from models.fredn.nufft import (
        BatchedDirectFourierAnalyzer,
        BatchedDirectFourierSynthesizer,
    )

    features, positions, _ = _phase_target((4,), samples_per_class=2)
    grid = torch.linspace(15.0, 340.0, 48, dtype=features.dtype).repeat(2, 1)
    analyzer = BatchedDirectFourierAnalyzer(9, 365.0, 1e-8)
    synthesizer = BatchedDirectFourierSynthesizer(9, 365.0)
    global_shift = 11.0
    residual_shift = 4.0

    base_coeffs, _ = analyzer(features, positions + global_shift)
    queried = synthesizer(base_coeffs, grid - residual_shift)
    shifted_coeffs, _ = analyzer(
        features, positions + global_shift + residual_shift
    )
    reanalyzed = synthesizer(shifted_coeffs, grid)

    assert torch.allclose(queried, reanalyzed, atol=2e-6, rtol=2e-6)


def test_known_per_class_residual_shifts_are_recovered():
    from models.shape_alignment import ClassResidualPhaseEstimator

    features, positions, pseudo_classes = _phase_target((3, -5))
    estimator = ClassResidualPhaseEstimator(
        _phase_reference(),
        mode=9,
        radius_days=7,
        step_days=1,
        min_samples=32,
        max_samples_per_class=128,
        min_corr_gain=0.005,
    )
    analysis_calls = []
    handle = estimator.analyzer.register_forward_hook(
        lambda module, inputs, output: analysis_calls.append(1)
    )

    try:
        result = estimator.estimate(
            features, positions, pseudo_classes, num_classes=2
        )
    finally:
        handle.remove()

    assert torch.equal(result.accepted_shifts.cpu(), torch.tensor([3.0, -5.0]))
    assert all(record.accepted for record in result.records)
    assert analysis_calls == [1]


def test_phase_estimator_falls_back_for_insufficient_samples():
    from models.shape_alignment import ClassResidualPhaseEstimator

    features, positions, pseudo_classes = _phase_target((4,), samples_per_class=31)
    result = ClassResidualPhaseEstimator(
        _phase_reference(), min_samples=32
    ).estimate(features, positions, pseudo_classes, num_classes=2)

    assert result.accepted_shifts.tolist() == [0.0, 0.0]
    assert result.records[0].reason == "insufficient_samples"


def test_phase_estimator_rejects_negligible_correlation_gain():
    from models.shape_alignment import ClassResidualPhaseEstimator

    features, positions, pseudo_classes = _phase_target((0,), samples_per_class=40)
    result = ClassResidualPhaseEstimator(
        _phase_reference(), min_samples=32, min_corr_gain=0.005
    ).estimate(features, positions, pseudo_classes, num_classes=2)

    assert result.accepted_shifts[0].item() == 0.0
    assert not result.records[0].accepted
    assert result.records[0].reason == "insufficient_corr_gain"


def test_mode9_estimates_phase_but_only_mode13_contributes_shape_loss():
    from models.shape_alignment import ShapeAlignment

    bank = _phase_reference()
    features, positions, pseudo_classes = _phase_target((2, -3), samples_per_class=2)
    alignment = ShapeAlignment(
        bank, morph_weight=1.0, event_weight=0.0, loss_modes=(13,)
    )
    per_sample_shift = torch.tensor([2.0, 2.0, -3.0, -3.0])
    shifted = alignment(
        features, positions, pseudo_classes, residual_shifts=per_sample_shift
    )

    assert set(shifted.mode_losses) == {13}
    assert torch.isfinite(shifted.loss)


def test_zero_residual_phase_is_exactly_previous_morph_only_loss():
    from models.shape_alignment import ShapeAlignment

    features, positions, pseudo_classes = _phase_target((2, -3), samples_per_class=2)
    alignment = ShapeAlignment(
        _phase_reference(), morph_weight=1.0, event_weight=0.0, loss_modes=(13,)
    )
    baseline = alignment(features, positions, pseudo_classes)
    explicit_zero = alignment(
        features,
        positions,
        pseudo_classes,
        residual_shifts=torch.zeros(features.shape[0], dtype=features.dtype),
    )

    assert torch.equal(baseline.loss, explicit_zero.loss)
    assert torch.equal(baseline.morph_loss, explicit_zero.morph_loss)


def test_per_sample_residual_phase_changes_shape_only_not_semantic_logits():
    import timematch
    from models.shape_alignment import ShapeAlignment

    model = _Semantic()
    inputs = _semantic_inputs(batch=3)
    semantic_before, captured = timematch._forward_with_spatial_capture(model, *inputs)
    alignment = ShapeAlignment(
        _phase_reference(dtype=torch.float32),
        morph_weight=1.0,
        event_weight=0.0,
        loss_modes=(13,),
    )
    pseudo_classes = torch.tensor([0, 1, 0])
    residual = torch.tensor([3.0, -5.0, 0.0])
    shape_result = alignment(
        captured,
        inputs[2].to(torch.float32),
        pseudo_classes,
        residual_shifts=residual,
    )
    semantic_after = timematch._forward_with_temporal_shift(model, *inputs)

    assert torch.equal(semantic_before, semantic_after)
    assert torch.isfinite(shape_result.loss)


def test_residual_shape_gradient_reaches_student_but_not_phase_or_reference():
    from models.shape_alignment import ShapeAlignment

    bank = _phase_reference(dtype=torch.float32)
    features, positions, pseudo_classes = _phase_target(
        (2, -3), samples_per_class=2, dtype=torch.float32
    )
    student = torch.nn.Linear(1, 1, bias=False)
    residual = torch.tensor([2.0, 2.0, -3.0, -3.0])
    result = ShapeAlignment(
        bank, morph_weight=1.0, event_weight=0.0, loss_modes=(13,)
    )(
        student(features),
        positions,
        pseudo_classes,
        residual_shifts=residual,
    )
    result.loss.backward()

    assert student.weight.grad is not None
    assert student.weight.grad.abs().sum() > 0
    assert not residual.requires_grad
    assert all(not value.requires_grad for value in bank.prototypes.values())


def test_shape_loss_selects_residual_shift_by_pseudo_class():
    import timematch

    class Recorder:
        def __call__(
            self,
            spatial_features,
            positions,
            pseudo_classes,
            residual_shifts=None,
        ):
            self.positions = positions
            self.classes = pseudo_classes
            self.residual_shifts = residual_shifts
            return "shape"

    recorder = Recorder()
    features = torch.randn(2, 4, 3)
    positions = torch.arange(12).reshape(3, 4)
    pseudo = torch.tensor([1, 0, 1])
    selected = torch.tensor([True, False, True])
    class_shifts = torch.tensor([0.0, -5.0])

    result = timematch._shape_loss_from_capture(
        recorder,
        features,
        positions,
        pseudo,
        selected,
        target_to_source_shift=6,
        class_residual_shifts=class_shifts,
    )

    assert result == "shape"
    assert torch.equal(recorder.classes, torch.tensor([1, 1]))
    assert torch.equal(recorder.residual_shifts, torch.tensor([-5.0, -5.0]))
    assert torch.equal(
        recorder.positions,
        torch.tensor([[6, 7, 8, 9], [14, 15, 16, 17]]),
    )


def test_epoch_zero_class_phase_returns_zero_without_scanning_loader():
    import timematch

    class ForbiddenLoader:
        def __iter__(self):
            raise AssertionError("epoch zero must not scan target data")

    config = SimpleNamespace(
        class_phase_start_epoch=1,
        class_phase_seed=1,
        num_classes=2,
        pseudo_threshold=0.9,
    )
    result = timematch._estimate_class_residual_phase(
        teacher=None,
        estimator=None,
        phase_loader=ForbiddenLoader(),
        device="cpu",
        target_to_source_shift=8,
        config=config,
        epoch=0,
    )

    assert result.accepted_shifts.tolist() == [0.0, 0.0]
    assert all(record.reason == "before_start_epoch" for record in result.records)


def test_phase_scan_uses_teacher_pseudo_labels_not_target_true_labels_and_restores_rng(
    monkeypatch,
):
    import timematch
    from models.shape_alignment import ClassResidualPhaseResult

    class RecordingEstimator:
        def estimate(self, features, positions, pseudo_classes, *, num_classes):
            self.features = features
            self.positions = positions
            self.pseudo_classes = pseudo_classes
            return ClassResidualPhaseResult(
                accepted_shifts=torch.zeros(num_classes), records=[]
            )

    sample = _training_batch(3, 2.0)
    sample["label"] = torch.tensor([1, 1, 1])
    loader = _Loader([sample])
    model = _Semantic()
    estimator = RecordingEstimator()
    monkeypatch.setattr(
        timematch,
        "to_cuda",
        lambda batch, device: (
            batch["pixels"],
            batch["valid_pixels"],
            batch["positions"],
            batch["extra"],
        ),
    )
    config = SimpleNamespace(
        class_phase_start_epoch=1,
        class_phase_seed=17,
        num_classes=2,
        pseudo_threshold=0.0,
    )
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)

    first_result = timematch._estimate_class_residual_phase(
        model,
        estimator,
        loader,
        "cpu",
        target_to_source_shift=4,
        config=config,
        epoch=1,
    )
    actual = (random.random(), np.random.rand(), torch.rand(1))
    first_pseudo = estimator.pseudo_classes.clone()
    sample["label"] = torch.tensor([0, 0, 0])
    second_estimator = RecordingEstimator()
    second_result = timematch._estimate_class_residual_phase(
        model,
        second_estimator,
        loader,
        "cpu",
        target_to_source_shift=4,
        config=config,
        epoch=1,
    )

    assert torch.equal(first_pseudo, torch.zeros(3, dtype=torch.long))
    assert torch.equal(second_estimator.pseudo_classes, first_pseudo)
    assert torch.equal(first_result.accepted_shifts, second_result.accepted_shifts)
    assert torch.equal(
        estimator.positions,
        sample["positions"] + 4,
    )
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_phase_and_shape_csvs_have_stable_audit_fields(monkeypatch):
    import csv
    import os
    import timematch
    from models.shape_alignment import ClassPhaseRecord, ClassResidualPhaseResult

    class NonClosingStringIO(io.StringIO):
        def close(self):
            pass

    files = {}

    def fake_open(path, mode="r", *args, **kwargs):
        path = os.fspath(path)
        stream = files.setdefault(path, NonClosingStringIO())
        if "a" in mode:
            stream.seek(0, io.SEEK_END)
        else:
            stream.seek(0)
        return stream

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(timematch.os.path, "exists", lambda path: os.fspath(path) in files)
    monkeypatch.setattr(timematch.os, "makedirs", lambda *args, **kwargs: None)
    output_dir = "audit-output"

    phase = ClassResidualPhaseResult(
        accepted_shifts=torch.tensor([4.0]),
        records=[
            ClassPhaseRecord(
                class_index=0,
                pseudo_count=87,
                corr_zero=0.721,
                best_corr=0.758,
                corr_gain=0.037,
                raw_delta=4.0,
                accepted_delta=4.0,
                accepted=True,
                reason="accepted",
            )
        ],
    )
    timematch._write_class_phase_csvs(
        output_dir,
        epoch=3,
        global_shift=11,
        class_names=["winter_wheat"],
        result=phase,
    )
    timematch._append_shape_training_metrics(
        output_dir,
        {
            "epoch": 3,
            "morph_loss": 0.2,
            "global_corr_mean": 0.8,
            "local_level_corr_mean": 0.7,
            "local_slope_corr_mean": 0.6,
            "local_shape_loss": 0.3,
            "weighted_shape_loss": 0.02,
            "shape_loss_ratio": 0.1,
            "selected_target_count": 10,
            "selected_target_rate": 0.5,
            "selected_class_count": 2,
            "morph_corr_mean": 0.8,
        },
    )
    timematch._append_shape_class_metrics(
        output_dir,
        [
            {
                "epoch": 3,
                "class_index": 0,
                "class_name": "winter_wheat",
                "selected_count": 4,
                "global_corr_mean": 0.8,
                "local_level_corr_mean": 0.7,
                "local_slope_corr_mean": 0.6,
                "local_shape_loss": 0.3,
                "source_activity_weight_min": 0.1,
                "source_activity_weight_max": 0.2,
            }
        ],
    )

    phase_stream = files[os.path.join(output_dir, "class_residual_phase.csv")]
    summary_stream = files[os.path.join(output_dir, "class_phase_epoch_summary.csv")]
    shape_stream = files[os.path.join(output_dir, "shape_training_metrics.csv")]
    shape_class_stream = files[
        os.path.join(output_dir, "shape_class_metrics.csv")
    ]
    phase_stream.seek(0)
    summary_stream.seek(0)
    shape_stream.seek(0)
    shape_class_stream.seek(0)
    phase_row = next(csv.DictReader(phase_stream))
    summary_row = next(csv.DictReader(summary_stream))
    shape_row = next(csv.DictReader(shape_stream))
    shape_class_row = next(csv.DictReader(shape_class_stream))

    assert phase_row == {
        "epoch": "3",
        "class_index": "0",
        "class_name": "winter_wheat",
        "class": "winter_wheat",
        "global_shift": "11",
        "pseudo_count": "87",
        "corr_zero": "0.721",
        "best_corr": "0.758",
        "corr_gain": "0.037",
        "raw_delta": "4.0",
        "accepted_delta": "4.0",
        "accepted": "True",
        "reason": "accepted",
    }
    assert summary_row["num_classes_nonzero_phase"] == "1"
    assert summary_row["mean_abs_delta"] == "4.0"
    assert shape_row["morph_corr_mean"] == "0.8"
    assert shape_row["global_corr_mean"] == "0.8"
    assert shape_row["local_level_corr_mean"] == "0.7"
    assert shape_row["local_slope_corr_mean"] == "0.6"
    assert shape_row["selected_class_count"] == "2"
    assert shape_class_row["class_name"] == "winter_wheat"
    assert shape_class_row["selected_count"] == "4"


def test_disabled_training_does_not_build_or_scan_class_phase(monkeypatch):
    import timematch

    monkeypatch.setattr(
        timematch,
        "_build_class_phase_loader",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("disabled path built phase loader")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        timematch,
        "_estimate_class_residual_phase",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("disabled path scanned phase")
        ),
        raising=False,
    )

    config = SimpleNamespace(class_residual_phase=False)
    loader, estimator = timematch._maybe_build_class_phase(
        None, None, config, {}, "cpu"
    )

    assert loader is None
    assert estimator is None


def _local_bank(dtype=torch.float64):
    from models.shape_alignment import SourceShapeReferenceBank

    features, positions, labels = _source_batch(dtype)
    return SourceShapeReferenceBank.from_source_features(
        features,
        positions,
        labels,
        modes=(13,),
        grid_points=64,
        period_days=365.0,
        reg=1e-3,
        prominence_rel=0.15,
        min_distance_days=14.0,
    )


def test_global_corr_explicit_configuration_preserves_loss_and_gradient_exactly():
    from models.shape_alignment import ShapeAlignment

    bank = _local_bank(torch.float64)
    features, positions, labels = _source_batch(torch.float64)
    legacy_input = features[:3].clone().requires_grad_(True)
    configured_input = features[:3].clone().requires_grad_(True)
    legacy = ShapeAlignment(bank, morph_weight=1.0, event_weight=0.0)
    configured = ShapeAlignment(
        bank,
        morph_weight=1.0,
        event_weight=0.0,
        loss_type="global_corr",
        local_window_points=16,
        local_stride_points=8,
        local_slope_weight=0.5,
        class_balanced=False,
    )

    legacy_result = legacy(legacy_input, positions[:3], labels[:3])
    configured_result = configured(configured_input, positions[:3], labels[:3])
    legacy_result.loss.backward()
    configured_result.loss.backward()

    assert torch.equal(legacy_result.loss, configured_result.loss)
    assert torch.equal(legacy_input.grad, configured_input.grad)
    assert configured_result.global_corr_mean == configured_result.morph_corr_mean


def test_local_window_bounds_are_seven_overlapping_sixteen_point_windows():
    from models.shape_alignment import _local_window_bounds

    assert _local_window_bounds(64, 16, 8) == (
        (0, 16),
        (8, 24),
        (16, 32),
        (24, 40),
        (32, 48),
        (40, 56),
        (48, 64),
    )


def test_source_activity_weights_are_frozen_normalized_and_flat_safe():
    from models.shape_alignment import _local_window_bounds, _source_activity_weights

    prototype = torch.ones((2, 64), dtype=torch.float64, requires_grad=True)
    weights = _source_activity_weights(
        prototype,
        _local_window_bounds(64, 16, 8),
    )

    assert weights.shape == (2, 7)
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(weights, torch.full_like(weights, 1 / 7))
    assert not weights.requires_grad


def test_flat_local_level_and_slope_correlations_are_finite():
    from models.shape_alignment import _local_morphology_terms

    flat = torch.ones((2, 16), dtype=torch.float64)
    loss, level_corr, slope_corr = _local_morphology_terms(
        flat,
        flat,
        torch.ones((2, 1), dtype=torch.float64),
        ((0, 16),),
        slope_weight=0.5,
    )

    assert torch.isfinite(loss).all()
    assert torch.isfinite(level_corr).all()
    assert torch.isfinite(slope_corr).all()


def test_local_loss_exposes_an_error_hidden_by_global_correlation():
    from models.shape_alignment import (
        _local_morphology_terms,
        _local_window_bounds,
        _stable_correlation,
    )

    x = torch.linspace(0, 4 * torch.pi, 64, dtype=torch.float64)
    prototype = (torch.sin(x) + 0.2 * torch.sin(3 * x)).unsqueeze(0)
    target = prototype.clone()
    target[:, 28:36] = -target[:, 28:36]
    windows = _local_window_bounds(64, 16, 8)
    weights = torch.full((1, 7), 1 / 7, dtype=torch.float64)

    global_loss = 1 - _stable_correlation(target, prototype)
    local_loss, _, _ = _local_morphology_terms(
        target, prototype, weights, windows, slope_weight=0.5
    )

    assert global_loss.item() < 0.5
    assert local_loss.item() > global_loss.item() + 0.1


def test_local_slope_uses_fifteen_differences_without_extra_normalization():
    from models.shape_alignment import _local_morphology_terms

    prototype = torch.arange(16, dtype=torch.float64).unsqueeze(0)
    target = prototype.clone()
    target[:, 6:10] = torch.tensor([7.5, 6.5, 9.5, 8.5], dtype=torch.float64)
    loss, level_corr, slope_corr = _local_morphology_terms(
        target,
        prototype,
        torch.ones((1, 1), dtype=torch.float64),
        ((0, 16),),
        slope_weight=0.5,
    )

    assert torch.isfinite(loss).all()
    assert level_corr.item() > 0.95
    assert slope_corr.item() < level_corr.item() - 0.2
    assert torch.diff(target[:, 0:16], dim=1).shape[1] == 15


def test_local_window_loss_is_normalized_by_one_plus_slope_weight():
    from models.shape_alignment import _combine_local_window_losses

    level_corr = torch.tensor([[0.4, 0.8]])
    slope_corr = torch.tensor([[0.1, 0.7]])
    actual = _combine_local_window_losses(
        level_corr, slope_corr, slope_weight=0.5
    )
    expected = ((1 - level_corr) + 0.5 * (1 - slope_corr)) / 1.5

    assert torch.equal(actual, expected)


def test_class_balanced_mean_weights_present_pseudo_classes_equally():
    from models.shape_alignment import _class_balanced_mean

    losses = torch.tensor([0.2] * 10 + [0.8])
    pseudo_classes = torch.tensor([0] * 10 + [1])

    assert torch.allclose(
        _class_balanced_mean(losses, pseudo_classes), torch.tensor(0.5)
    )


def test_local_morph_sample_mean_weights_every_selected_sample_equally(monkeypatch):
    import models.shape_alignment as shape_alignment

    losses = torch.tensor([0.2] * 10 + [0.8], dtype=torch.float64)

    def fixed_local_terms(target, prototype, activity_weights, windows, slope_weight):
        values = losses.to(device=target.device, dtype=target.dtype)
        zeros = torch.zeros_like(values)
        return values, zeros, zeros

    monkeypatch.setattr(
        shape_alignment, "_local_morphology_terms", fixed_local_terms
    )
    features, positions, _ = _source_batch(torch.float64)
    features = torch.cat((features, features[:3]), dim=0)
    positions = torch.cat((positions, positions[:3]), dim=0)
    pseudo_classes = torch.tensor([0] * 10 + [1])
    alignment = shape_alignment.ShapeAlignment(
        _local_bank(torch.float64),
        event_weight=0.0,
        loss_type="local_morph",
        class_balanced=False,
    )

    result = alignment(features, positions, pseudo_classes)

    assert torch.allclose(result.morph_loss, losses.mean())
    assert not torch.allclose(result.morph_loss, result.morph_loss.new_tensor(0.5))


def test_local_shape_uses_one_reconstruction_and_has_only_student_gradient():
    from models.shape_alignment import ShapeAlignment

    bank = _local_bank(torch.float32)
    features, positions, labels = _source_batch(torch.float32)
    student = torch.nn.Linear(3, 3, bias=False)
    teacher = torch.nn.Linear(3, 3, bias=False)
    target = student(features[:3])
    alignment = ShapeAlignment(
        bank,
        event_weight=0.0,
        loss_type="local_morph",
        local_window_points=16,
        local_stride_points=8,
        local_slope_weight=0.5,
        class_balanced=True,
    )
    analysis_calls = []
    synthesis_calls = []
    analysis_handle = alignment.analyzers["13"].register_forward_hook(
        lambda *_: analysis_calls.append(1)
    )
    synthesis_handle = alignment.synthesizers["13"].register_forward_hook(
        lambda *_: synthesis_calls.append(1)
    )
    try:
        result = alignment(target, positions[:3], labels[:3])
    finally:
        analysis_handle.remove()
        synthesis_handle.remove()
    result.loss.backward()

    assert analysis_calls == [1]
    assert synthesis_calls == [1]
    assert student.weight.grad is not None
    assert student.weight.grad.abs().sum() > 0
    assert teacher.weight.grad is None
    assert not any(parameter.requires_grad for parameter in alignment.reference.parameters())
    assert not getattr(alignment, "local_activity_weights_13").requires_grad


def test_local_morph_loss_never_reintroduces_event_loss():
    from models.shape_alignment import ShapeAlignment

    bank = _local_bank(torch.float64)
    features, positions, labels = _source_batch(torch.float64)
    without_event = ShapeAlignment(
        bank,
        event_weight=0.0,
        loss_type="local_morph",
        class_balanced=True,
    )(features[:3], positions[:3], labels[:3])
    configured_event = ShapeAlignment(
        bank,
        event_weight=10.0,
        loss_type="local_morph",
        class_balanced=True,
    )(features[:3], positions[:3], labels[:3])

    assert torch.equal(without_event.loss, configured_event.loss)


def test_local_diagnostics_reuses_one_mode13_analysis_and_synthesis():
    from models.shape_alignment import ShapeAlignment

    alignment = ShapeAlignment(
        _local_bank(torch.float64),
        event_weight=0.0,
        loss_type="local_morph",
        class_balanced=True,
    )
    features, positions, labels = _source_batch(torch.float64)
    analysis_calls = []
    synthesis_calls = []
    analysis_handle = alignment.analyzers["13"].register_forward_hook(
        lambda *_: analysis_calls.append(1)
    )
    synthesis_handle = alignment.synthesizers["13"].register_forward_hook(
        lambda *_: synthesis_calls.append(1)
    )
    try:
        metrics = alignment.diagnostics(features[:2], positions[:2], labels[:2])
    finally:
        analysis_handle.remove()
        synthesis_handle.remove()

    assert analysis_calls == [1]
    assert synthesis_calls == [1]
    assert all(torch.isfinite(value) for value in metrics.values())


def test_global_corr_does_not_validate_unused_local_window_configuration():
    from models.shape_alignment import ShapeAlignment, SourceShapeReferenceBank

    features, positions, labels = _source_batch(torch.float64)
    bank = SourceShapeReferenceBank.from_source_features(
        features,
        positions,
        labels,
        modes=(13,),
        grid_points=8,
        period_days=365.0,
        reg=1e-3,
        prominence_rel=0.15,
        min_distance_days=14.0,
    )

    alignment = ShapeAlignment(bank, loss_type="global_corr")
    result = alignment(features[:2], positions[:2], labels[:2])

    assert torch.isfinite(result.loss)


def test_local_shape_has_no_target_label_input_and_returns_pseudo_class_metrics():
    from models.shape_alignment import ShapeAlignment

    bank = _local_bank(torch.float64)
    features, positions, _ = _source_batch(torch.float64)
    pseudo_classes = torch.tensor([0, 1])
    target_a = features[:2].clone().requires_grad_(True)
    target_b = features[:2].clone().requires_grad_(True)
    target_true_a = torch.tensor([0, 1])
    target_true_b = torch.tensor([1, 0])
    alignment = ShapeAlignment(
        bank,
        event_weight=0.0,
        loss_type="local_morph",
        class_balanced=True,
    )

    result_a = alignment(target_a, positions[:2], pseudo_classes)
    result_b = alignment(target_b, positions[:2], pseudo_classes)
    result_a.loss.backward()
    result_b.loss.backward()

    assert not torch.equal(target_true_a, target_true_b)
    assert torch.equal(result_a.loss, result_b.loss)
    assert torch.equal(target_a.grad, target_b.grad)
    assert set(result_a.class_metrics) == {0, 1}
    assert result_a.class_metrics[0]["selected_count"] == 1
    assert result_a.class_metrics[1]["selected_count"] == 1


def test_local_manifest_records_window_activity_by_class():
    from models.shape_alignment import ShapeAlignment

    alignment = ShapeAlignment(
        _local_bank(torch.float64),
        event_weight=0.0,
        loss_type="local_morph",
        class_balanced=True,
    )
    metadata = alignment.manifest_metadata(["a", "b"])

    assert metadata["shape_loss_type"] == "local_morph"
    assert metadata["local_window_points"] == 16
    assert metadata["local_stride_points"] == 8
    assert metadata["per_class"]["a"]["window_count"] == 7
    assert len(metadata["per_class"]["a"]["window_activity_weights"]) == 7


def test_local_epoch_observation_reuses_result_and_accumulates_pseudo_classes():
    import timematch
    from models.shape_alignment import ShapeAlignmentResult

    scalar = lambda value: torch.tensor(value, dtype=torch.float64)
    result = ShapeAlignmentResult(
        loss=scalar(0.4),
        morph_loss=scalar(0.4),
        event_loss=scalar(0.0),
        mode_losses={13: scalar(0.4)},
        morph_corr_mean=scalar(0.75),
        event_amp_abs_gap=scalar(0.0),
        no_event_reference_count=torch.tensor(0),
        global_corr_mean=scalar(0.75),
        local_level_corr_mean=scalar(0.65),
        local_slope_corr_mean=scalar(0.55),
        local_shape_loss=scalar(0.4),
        class_metrics={
            0: {
                "selected_count": torch.tensor(2),
                "global_corr_mean": scalar(0.8),
                "local_level_corr_mean": scalar(0.7),
                "local_slope_corr_mean": scalar(0.6),
                "local_shape_loss": scalar(0.3),
                "source_activity_weight_min": scalar(0.1),
                "source_activity_weight_max": scalar(0.2),
            },
            1: {
                "selected_count": torch.tensor(1),
                "global_corr_mean": scalar(0.5),
                "local_level_corr_mean": scalar(0.4),
                "local_slope_corr_mean": scalar(0.3),
                "local_shape_loss": scalar(0.8),
                "source_activity_weight_min": scalar(0.05),
                "source_activity_weight_max": scalar(0.25),
            },
        },
    )
    epoch = timematch._new_shape_epoch_metrics()

    timematch._observe_shape_training_step(
        epoch,
        result,
        torch.tensor([True, True, True, False]),
        scalar(2.0),
        shape_lambda=0.1,
    )

    assert epoch["selected"] == 3
    assert epoch["global_corr"] == pytest.approx(0.75 * 3)
    assert epoch["local_level_corr"] == pytest.approx(0.65 * 3)
    assert epoch["local_slope_corr"] == pytest.approx(0.55 * 3)
    assert epoch["local_loss"] == pytest.approx(0.4 * 3)
    assert set(epoch["classes"]) == {0, 1}
    assert epoch["classes"][0]["selected"] == 2
    assert epoch["classes"][1]["selected"] == 1


def test_train_cli_declares_backward_compatible_local_shape_options():
    source = Path("train.py").read_text(encoding="utf-8")

    assert '"--shape_loss_type"' in source
    assert 'default="global_corr"' in source
    assert '"--shape_local_window_points"' in source
    assert '"--shape_local_stride_points"' in source
    assert '"--shape_local_slope_weight"' in source
    assert '"--shape_class_balanced"' in source


def test_local_morph_launcher_freezes_four_tasks_and_disables_class_phase():
    source = Path(
        "scripts/run_timematch_shape13_localmorph_4tasks_4gpu.sh"
    ).read_text(encoding="utf-8")

    assert 'launch "$GPU0" AT1 "$AT1" DK1 "$DK1" "$AT1_WEIGHTS"' in source
    assert 'launch "$GPU1" DK1 "$DK1" FR1 "$FR1" "$DK1_WEIGHTS"' in source
    assert 'launch "$GPU2" FR1 "$FR1" FR2 "$FR2" "$FR1_WEIGHTS"' in source
    assert 'launch "$GPU3" FR2 "$FR2" AT1 "$AT1" "$FR2_WEIGHTS"' in source
    assert "--shape_loss_type local_morph" in source
    assert "--shape_local_window_points 16" in source
    assert "--shape_local_stride_points 8" in source
    assert "--shape_local_slope_weight 0.5" in source
    assert "--shape_class_balanced true" in source
    assert "--shape_event_weight 0.0" in source
    assert "--class_residual_phase false" in source
    assert 'SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/' in source
    assert 'nohup bash "$SCRIPT_PATH" --worker' in source
    for forbidden in ("git ", "pip install", "conda ", "module load"):
        assert forbidden not in source


def _write_ablation_status(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("group", "task", "exit_code", "status")
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_ablation_log(log_root, group, task, seed=1, include_test=True):
    experiment = f"timematch_{group}_{task}_seed{seed}"
    path = log_root / group / f"{task}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "Validation result: loss=0.1, acc=90.00, f1=88.0000\n"
    if include_test:
        text += f"Test result for {experiment}: accuracy=91.0000, f1=87.2500\n"
    path.write_text(text, encoding="utf-8")


def test_ablation_summarizer_parses_exact_test_records(tmp_path):
    from scripts.summarize_timematch_ablation import TASKS, summarize

    log_root = tmp_path / "logs"
    status_path = tmp_path / "statuses.csv"
    output_path = tmp_path / "summary.csv"
    statuses = []
    for group in ("original", "local_sample"):
        for task, _, _ in TASKS:
            _write_ablation_log(log_root, group, task)
            statuses.append({"group": group, "task": task, "exit_code": 0})
    _write_ablation_status(status_path, statuses)

    result = summarize(log_root, status_path, output_path, seed=1)

    assert result == 0
    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert rows[0]["original_status"] == "SUCCESS"
    assert rows[0]["local_sample_status"] == "SUCCESS"
    assert float(rows[0]["local_minus_original"]) == pytest.approx(0.0)


def test_ablation_summarizer_reports_failed_child_process(tmp_path):
    from scripts.summarize_timematch_ablation import TASKS, summarize

    log_root = tmp_path / "logs"
    status_path = tmp_path / "statuses.csv"
    output_path = tmp_path / "summary.csv"
    statuses = []
    for task, _, _ in TASKS:
        _write_ablation_log(log_root, "original", task)
        statuses.append(
            {
                "group": "original",
                "task": task,
                "exit_code": 9 if task == "DK1_FR1" else 0,
                "status": (
                    "MISSING_SHAPE_ARTIFACT" if task == "FR1_FR2" else ""
                ),
            }
        )
    _write_ablation_status(status_path, statuses)

    result = summarize(
        log_root,
        status_path,
        output_path,
        seed=1,
        required_groups=("original",),
    )

    assert result == 1
    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = {row["task"]: row for row in csv.DictReader(stream)}
    assert rows["DK1_FR1"]["original_status"] == "PROCESS_FAILED(9)"
    assert rows["FR1_FR2"]["original_status"] == "MISSING_SHAPE_ARTIFACT"


def test_ablation_summarizer_rejects_missing_or_wrong_test_record(tmp_path):
    from scripts.summarize_timematch_ablation import TASKS, summarize

    log_root = tmp_path / "logs"
    status_path = tmp_path / "statuses.csv"
    output_path = tmp_path / "summary.csv"
    statuses = []
    for task, _, _ in TASKS:
        _write_ablation_log(
            log_root,
            "original",
            task,
            include_test=task != "FR1_FR2",
        )
        statuses.append({"group": "original", "task": task, "exit_code": 0})
    wrong_log = log_root / "original" / "FR1_FR2.log"
    wrong_log.write_text(
        "Test result for another_experiment: accuracy=99.0, f1=99.0\n",
        encoding="utf-8",
    )
    _write_ablation_status(status_path, statuses)

    result = summarize(
        log_root,
        status_path,
        output_path,
        seed=1,
        required_groups=("original",),
    )

    assert result == 1
    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = {row["task"]: row for row in csv.DictReader(stream)}
    assert rows["FR1_FR2"]["original_status"] == "MISSING_TEST_F1"


def test_ablation_launcher_freezes_two_waves_and_shape_boundaries():
    source = Path(
        "scripts/run_timematch_ablation_original_local_sample_4tasks.sh"
    ).read_text(encoding="utf-8")

    assert '--shape_align false' in source
    assert '--class_residual_phase false' in source
    assert '--shape_align true' in source
    assert '--shape_modes 13' in source
    assert '--shape_loss_type local_morph' in source
    assert '--shape_class_balanced false' in source
    assert '--shape_lambda 0.1' in source
    assert '--shape_event_weight 0.0' in source
    assert 'run_wave "original"' in source
    assert 'run_wave "local_sample"' in source
    assert 'if [[ "$FOLD" != "0" ]]' in source
    assert "check_clean_destinations" in source
    assert "audit_original_artifacts" in source
    assert "audit_local_sample_artifacts" in source
    assert "shape_training_metrics.csv" in source
    assert "shape_class_metrics.csv" in source
    assert "shape_reference_manifest.json" in source
    assert "shape_reference.pt" in source
    for forbidden in ("git ", "pip install", "wget ", "curl ", "conda "):
        assert forbidden not in source
