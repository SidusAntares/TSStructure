import pytest
import torch

from methods.structure_da.prototype_losses import (
    compose_da_loss,
    compose_source_loss,
    prototype_contrastive_loss,
    select_top_shape_tokens,
    shape_prototype_loss,
)
from models.structure_da.prototype_bank import ClassPrototypeBank
from timematch import update_ema_variables
from models.stclassifier import PseStructureProtoLTae
import methods.structure_da.prototype_losses as prototype_losses


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


def test_top_shape_selection_uses_teacher_attention_and_ceil_ratio():
    attention = torch.tensor([[.1, .6, .3, .9], [.9, .1, .8, .2]])
    valid = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    selected = select_top_shape_tokens(attention, valid, ratio=.6)
    assert selected.tolist() == [[False, True, True, False], [True, False, False, False]]


def test_shape_loss_is_sample_averaged_and_empty_target_is_differentiable_zero():
    tokens = torch.randn(2, 3, 4, requires_grad=True)
    selected = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    labels = torch.tensor([0, 1])
    prototypes = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    loss = shape_prototype_loss(tokens, selected, labels, prototypes, .1)
    assert torch.isfinite(loss)
    zero = shape_prototype_loss(tokens, selected & False, labels, prototypes, .1)
    assert zero.item() == 0
    zero.backward()
    assert tokens.grad is not None


def test_prototype_loss_and_composition_formulas_are_exact():
    features = torch.tensor([[1., 0.], [0., 1.]])
    prototypes = torch.eye(2)
    labels = torch.tensor([0, 1])
    assert prototype_contrastive_loss(features, labels, prototypes, .1) < .001
    source = compose_source_loss(
        torch.tensor(2.), torch.tensor(3.), torch.tensor(5.), .4, 2., 3.,
    )
    assert torch.allclose(source.total, torch.tensor(2.) + .4 * (2 * 3 + 3 * 5))
    da = compose_da_loss(
        torch.tensor(2.), torch.tensor(4.), 2.,
        torch.tensor(3.), torch.tensor(5.),
        torch.tensor(7.), torch.tensor(11.),
        .25, 2., 3.,
    )
    expected = 2 + 2 * 4 + 2 * 3 + 3 * 5 + .25 * (2 * 7 + 3 * 11)
    assert torch.allclose(da, torch.tensor(expected))


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
            self.shape_prototype_bank = ClassPrototypeBank(2, 2)

    student, teacher = Holder(), Holder()
    teacher_bank = teacher.shape_prototype_bank.prototypes.clone()
    with torch.no_grad():
        student.encoder.weight.fill_(2)
        teacher.encoder.weight.zero_()
        student.shape_prototype_bank.prototypes.fill_(1)
    update_ema_variables(student, teacher, .5)
    assert torch.allclose(teacher.encoder.weight, torch.ones_like(teacher.encoder.weight))
    assert torch.equal(teacher.shape_prototype_bank.prototypes, teacher_bank)


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
        student.shape_prototype_bank.prototypes.fill_(1.)
    bank_before = teacher.shape_prototype_bank.prototypes.clone()
    update_ema_variables(student, teacher, .5)
    named = dict(teacher.named_parameters())
    required = (
        "spatial_encoder", "structure_branch.token_generator.raw_encoder",
        "structure_branch.token_generator.diff_encoder",
        "structure_branch.token_generator.mean_encoder",
        "structure_branch.token_generator.std_encoder",
        "structure_branch.token_generator.fusion", "structure_branch.attention_pool",
        "temporal_encoder.attention_heads.external_query_projection",
        "temporal_encoder.mlp", "decoder",
    )
    for prefix in required:
        assert any(name.startswith(prefix) and torch.allclose(value, torch.ones_like(value))
                   for name, value in named.items()), prefix
    assert torch.equal(teacher.shape_prototype_bank.prototypes, bank_before)
