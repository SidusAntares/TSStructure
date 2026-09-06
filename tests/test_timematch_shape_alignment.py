import inspect
import io
import random
from copy import deepcopy
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
    "domain_specific_bn,expected_student_pse_calls", [(True, 2), (False, 1)]
)
def test_shape_training_reuses_the_existing_student_semantic_forward(
    monkeypatch, domain_specific_bn, expected_student_pse_calls
):
    import timematch
    from models.shape_alignment import ShapeAlignmentResult

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
    model = _Semantic()
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"state_dict": deepcopy(model.state_dict())},
    )
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: None)
    build_calls = []

    class FakeAlignment:
        def __call__(self, spatial, positions, classes):
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
        shape_align=True,
        shape_lambda=0.1,
        shape_diag_batches=0,
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
    assert len(build_calls) == 1
    assert build_calls[0][0].shape[0] == 2


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
