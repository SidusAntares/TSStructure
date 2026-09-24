import random

import numpy as np
import pytest
import torch

from methods.structure_da.prototype_losses import (
    compose_da_loss,
    compose_source_loss,
    initialize_instance_bank,
    instance_prototype_loss,
    prototype_contrastive_loss,
    shapelet_diversity_loss,
    shapelet_data_support_loss,
    update_instance_bank,
    ensure_finite_structure_loss,
)
from models.structure_da.prototype_bank import ClassPrototypeBank
from timematch import update_ema_variables
from models.stclassifier import PseStructureProtoLTae
import methods.structure_da.prototype_losses as prototype_losses


def _small_structure_model():
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=10, shapelet_count=4,
        fourier_num_modes=5, dropout=0,
    )


def _small_structure_batch():
    return {
        "pixels": torch.randn(4, 9, 3, 5),
        "valid_pixels": torch.ones(4, 9, 5),
        "positions": torch.arange(9).repeat(4, 1) * 25,
        "extra": torch.empty(4, 9, 0),
        "label": torch.tensor([0, 1, 2, 1]),
    }


def _gradient_norm(module):
    return torch.sqrt(sum(
        parameter.grad.detach().square().sum()
        for parameter in module.parameters() if parameter.grad is not None
    ))


def test_shape_classifier_directly_supervises_tokens_and_anchors_with_zero_query_projection():
    model = _small_structure_model().eval()
    batch = _small_structure_batch()
    projection = model.temporal_encoder.attention_heads.external_query_projection
    assert torch.count_nonzero(projection.weight) == 0
    output = model(**{key: batch[key] for key in ("pixels", "positions", "extra")},
                   mask=batch["valid_pixels"], return_dict=True)
    assert output["shape_logits"].shape == (4, 3)
    loss = torch.nn.functional.cross_entropy(output["shape_logits"], batch["label"])
    assert torch.isfinite(loss)
    loss.backward()
    assert _gradient_norm(model.structure_branch.token_generator) > 0
    assert _gradient_norm(model.structure_branch.shapelet_dictionary) > 0
    assert _gradient_norm(model.shape_classifier) > 0
    assert projection.weight.grad is None


def test_target_shape_loss_uses_only_existing_pseudo_mask_and_returns_zero_when_empty():
    from timematch import selected_shape_pseudo_loss

    criterion = torch.nn.CrossEntropyLoss()
    pseudo = torch.tensor([2, 0, 1, 2])
    mask = torch.tensor([True, False, True, False])
    selected_logits = torch.tensor([[0., 1., 3.], [0., 4., 1.]], requires_grad=True)
    loss, accuracy = selected_shape_pseudo_loss(
        selected_logits, pseudo, mask, criterion, minimum=2,
    )
    assert torch.allclose(loss, criterion(selected_logits, pseudo[mask]))
    assert accuracy == 1.
    empty_logits = selected_logits[:0]
    zero, empty_accuracy = selected_shape_pseudo_loss(
        empty_logits, pseudo, torch.zeros_like(mask), criterion, minimum=2,
    )
    assert zero.item() == 0.
    assert empty_accuracy == 0.


def test_new_structure_defaults_are_random_stride8_and_24_candidates():
    import argparse
    import train

    parser = train.add_model_arguments(argparse.ArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.shapelet_init == "random"
    assert defaults.shape_window_stride == 8
    explicit = parser.parse_args(["--shapelet-init", "kmeans"])
    assert explicit.shapelet_init == "kmeans"
    model = _small_structure_model()
    assert model.structure_branch.window_extractor.scales == (8, 16, 24)
    curve = torch.randn(2, 64, 8)
    groups, scales = model.structure_branch.window_extractor(curve)
    assert sum(group.shape[1] for group in groups) == 24
    assert scales.numel() == 24


def test_shape_health_snapshot_is_finite_and_detects_collapsed_response():
    from methods.structure_da.prototype_losses import shape_health_snapshot

    model = _small_structure_model()
    output = {
        "shapelet_response": torch.ones(4, 4),
        "shape_tokens": torch.randn(4, 24, 10),
    }
    health = shape_health_snapshot(model, output)
    assert health["shape_response_std_mean"] == 0.
    assert health["shape_response_effective_rank"] == pytest.approx(1.)
    assert health["shape_token_std"] > 0
    for key in (
        "shape_raw_encoder_param_norm", "shape_diff_encoder_param_norm",
        "shape_fusion_param_norm", "shape_anchor_param_norm",
        "shape_query_projection_norm",
    ):
        assert np.isfinite(health[key])


def test_bank_updates_source_classes_by_ema_and_leaves_absent_class():
    bank = ClassPrototypeBank(3, 2, momentum=.5)
    features = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    labels = torch.tensor([0, 0, 1])
    bank.update_source(features, labels)
    before = bank.prototypes[2].clone()
    bank.update_source(torch.tensor([[0., 1.]]), torch.tensor([0]))
    assert torch.equal(bank.prototypes[2], before)
    assert bank.initialized.tolist() == [True, True, False]
    assert torch.allclose(bank.prototypes[bank.initialized].norm(dim=-1), torch.ones(2))
    assert not any(parameter.requires_grad for parameter in bank.parameters())


def test_bank_initialization_requires_every_class():
    bank = ClassPrototypeBank(3, 2)
    with pytest.raises(RuntimeError, match="missing source classes.*2"):
        bank.initialize_source(torch.eye(2), torch.tensor([0, 1]))


def test_bank_has_no_target_update_api_and_roundtrips_state():
    bank = ClassPrototypeBank(2, 3)
    assert not hasattr(bank, "update_target")
    bank.update_source(torch.randn(4, 3), torch.tensor([0, 0, 1, 1]))
    restored = ClassPrototypeBank(2, 3)
    restored.load_state_dict(bank.state_dict())
    assert torch.equal(restored.prototypes, bank.prototypes)
    assert torch.equal(restored.update_count, bank.update_count)


def test_structure_model_has_only_source_only_instance_prototype_bank():
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=10, shape_window_scales=(8, 16),
        shape_window_stride=8, fourier_num_modes=5,
    )
    assert hasattr(model, "instance_prototype_bank")
    assert not hasattr(model, "shape_" + "prototype_bank")
    assert not hasattr(model, "target_instance_prototype_bank")


def test_centroid_alignment_uses_only_shared_classes_and_pseudo_labels():
    source_sum = torch.zeros(3, 2)
    source_count = torch.zeros(3)
    target_sum = torch.zeros(3, 2)
    target_count = torch.zeros(3)
    prototype_losses.accumulate_class_feature_sums(
        source_sum, source_count,
        torch.tensor([[2., 0.], [0., 3.], [1., 0.]]),
        torch.tensor([0, 1, 2]),
    )
    # These labels are teacher pseudo labels; the helper has no true-label input.
    prototype_losses.accumulate_class_feature_sums(
        target_sum, target_count,
        torch.tensor([[4., 0.], [5., 0.], [1., 0.]]),
        torch.tensor([0, 0, 1]),
    )
    summary = prototype_losses.centroid_alignment_summary(
        source_sum, source_count, target_sum, target_count,
    )
    assert summary["valid_classes"].tolist() == [0, 1]
    assert torch.allclose(summary["per_class"], torch.tensor([1., 0.]))
    assert torch.allclose(summary["macro_cos"], torch.tensor(.5))


def test_prototype_loss_and_composition_formulas_are_exact():
    features = torch.tensor([[1., 0.], [0., 1.]])
    prototypes = torch.eye(2)
    labels = torch.tensor([0, 1])
    assert prototype_contrastive_loss(features, labels, prototypes, .1) < .001
    source = compose_source_loss(
        torch.tensor(2.), torch.tensor(3.), torch.tensor(5.), torch.tensor(7.),
        .4, 2., .25, .5,
    )
    assert torch.allclose(source.total, torch.tensor(2.) + .4 * 2 * 3 + .25 * 5 + .5 * 7)
    da = compose_da_loss(
        torch.tensor(2.), torch.tensor(4.), 2.,
        torch.tensor(3.), torch.tensor(7.), torch.tensor(11.), torch.tensor(13.),
        .25, 2., .5, .75,
    )
    expected = 2 + 2 * 4 + 2 * 3 + .25 * 2 * 7 + .5 * 11 + .75 * 13
    assert torch.allclose(da, torch.tensor(expected))


def test_shapelet_diversity_penalizes_identical_anchors_more_than_dissimilar_anchors():
    identical = torch.ones(4, 3)
    dissimilar = torch.eye(4)
    assert shapelet_diversity_loss(identical, margin=.5) > 0
    assert shapelet_diversity_loss(dissimilar, margin=.5) == 0


def test_shapelet_data_support_prefers_near_anchor_and_detaches_source_tokens():
    tokens = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
    near = torch.tensor([[1., 0.]], requires_grad=True)
    far = torch.tensor([[-1., 0.]], requires_grad=True)
    near_loss = shapelet_data_support_loss(tokens, near, temperature=.1)
    far_loss = shapelet_data_support_loss(tokens, far, temperature=.1)
    assert near_loss < far_loss
    near_loss.backward()
    assert tokens.grad is None
    assert near.grad is not None and near.grad.abs().sum() > 0
    assert torch.isfinite(near.grad).all()


def test_shapelet_data_support_is_finite_and_invariant_to_token_duplication():
    torch.manual_seed(6)
    tokens = torch.randn(2, 5, 4, requires_grad=True)
    anchors = torch.randn(3, 4, requires_grad=True)
    loss = shapelet_data_support_loss(tokens, anchors, temperature=.1)
    duplicated = shapelet_data_support_loss(
        torch.cat((tokens, tokens), dim=1), anchors, temperature=.1,
    )
    assert torch.isfinite(loss)
    assert torch.allclose(loss, duplicated, atol=1e-6)
    loss.backward()
    assert tokens.grad is None
    assert anchors.grad is not None and torch.isfinite(anchors.grad).all()


def test_structure_loss_fails_fast_on_non_finite_values():
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        ensure_finite_structure_loss(torch.tensor(float("nan")))


def test_instance_bank_helpers_never_update_from_target_features():
    bank = ClassPrototypeBank(2, 3)
    source = torch.randn(4, 3)
    labels = torch.tensor([0, 0, 1, 1])
    initialize_instance_bank([(source, labels)], bank)
    before = bank.prototypes.clone()
    target = torch.randn(4, 3)
    loss = instance_prototype_loss(target, labels, bank, .1)
    assert torch.isfinite(loss)
    assert torch.equal(bank.prototypes, before)
    update_instance_bank(source, labels, bank)
    assert bank.update_count.sum() > 2


def test_prototype_ramp_starts_then_increases_linearly_to_one():
    ramp = prototype_losses.prototype_ramp
    assert ramp(0, 5, .1) == pytest.approx(.1)
    assert ramp(2, 5, .1) == pytest.approx(.46)
    assert ramp(5, 5, .1) == pytest.approx(1.)
    assert ramp(9, 5, .1) == pytest.approx(1.)


def test_teacher_ema_updates_parameters_but_not_prototype_banks():
    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.instance_prototype_bank = ClassPrototypeBank(2, 2)

    student, teacher = Holder(), Holder()
    teacher_bank = teacher.instance_prototype_bank.prototypes.clone()
    with torch.no_grad():
        student.encoder.weight.fill_(2)
        teacher.encoder.weight.zero_()
        student.instance_prototype_bank.prototypes.fill_(1)
    update_ema_variables(student, teacher, .5)
    assert torch.allclose(teacher.encoder.weight, torch.ones_like(teacher.encoder.weight))
    assert torch.equal(teacher.instance_prototype_bank.prototypes, teacher_bank)


def test_teacher_ema_covers_all_new_trainable_submodules():
    student = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=10, shape_window_scales=(8, 16),
        shape_window_stride=8, fourier_num_modes=5,
    )
    teacher = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=10, shape_window_scales=(8, 16),
        shape_window_stride=8, fourier_num_modes=5,
    )
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.fill_(2.)
        for parameter in teacher.parameters():
            parameter.zero_()
        student.instance_prototype_bank.prototypes.fill_(1.)
    bank_before = teacher.instance_prototype_bank.prototypes.clone()
    update_ema_variables(student, teacher, .5)
    named = dict(teacher.named_parameters())
    required = (
        "spatial_encoder", "structure_branch.token_generator.raw_encoder",
        "structure_branch.token_generator.diff_encoder",
        "structure_branch.token_generator.mean_encoder",
        "structure_branch.token_generator.std_encoder",
        "structure_branch.token_generator.fusion", "structure_branch.shapelet_dictionary",
        "structure_branch.response_to_query",
        "temporal_encoder.attention_heads.external_query_projection",
        "temporal_encoder.mlp", "decoder", "shape_classifier",
    )
    for prefix in required:
        assert any(name.startswith(prefix) and torch.allclose(value, torch.ones_like(value))
                   for name, value in named.items()), prefix
    assert torch.equal(teacher.instance_prototype_bank.prototypes, bank_before)


def test_source_shapelet_initialization_is_eval_no_grad_and_only_changes_anchors():
    import train

    torch.manual_seed(17)
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=6, shape_window_scales=(8,),
        shape_window_stride=8, shapelet_count=3, fourier_num_modes=5,
    )
    sample = {
        "pixels": torch.randn(4, 10, 3, 5),
        "valid_pixels": torch.ones(4, 10, 5),
        "positions": torch.arange(10).repeat(4, 1) * 20,
        "extra": torch.zeros(4, 4),
        "label": torch.tensor([0, 1, 2, 0]),
    }
    before_parameters = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    before_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    random.seed(29)
    np.random.seed(29)
    torch.manual_seed(29)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    model.train()
    diagnostics = train.initialize_shapelet_dictionary_from_source(
        model, [sample], device="cpu", seed=3, max_tokens=100,
    )
    assert model.training
    assert diagnostics["tokens"] == 32
    for name, value in model.named_parameters():
        if name == "structure_branch.shapelet_dictionary.anchors":
            assert not torch.equal(value, before_parameters[name])
        else:
            assert torch.equal(value, before_parameters[name]), name
            assert value.grad is None
    for name, value in model.named_buffers():
        assert torch.equal(value, before_buffers[name]), name
    assert random.getstate() == python_rng
    assert np.random.get_state()[0] == numpy_rng[0]
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert np.random.get_state()[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
