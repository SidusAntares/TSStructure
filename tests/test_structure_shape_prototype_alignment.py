import inspect
from pathlib import Path
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_structure_proto_v2clean_shapeproto_4tasks_4gpu_seed1.sh"


def _initialized_bank():
    from models.structure_da.prototype_bank import ClassPrototypeBank

    bank = ClassPrototypeBank(2, 2, momentum=.9)
    bank.initialize_source(torch.eye(2), torch.tensor([0, 1]))
    return bank


def test_source_shape_prototype_initialization_covers_classes_and_is_normalized():
    from methods.structure_da.prototype_losses import initialize_source_shape_prototypes
    from models.structure_da.prototype_bank import ClassPrototypeBank

    bank = ClassPrototypeBank(3, 2, momentum=.9)
    features = torch.tensor([
        [2., 0.], [1., 0.], [0., 3.], [0., 1.], [-2., 0.], [-1., 0.],
    ])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    diagnostics = initialize_source_shape_prototypes(bank, features, labels)

    assert diagnostics["initialized_classes"] == 3
    assert diagnostics["source_proto_accuracy"] == pytest.approx(1.)
    torch.testing.assert_close(bank.prototypes.norm(dim=-1), torch.ones(3))
    assert list(bank.parameters()) == []
    assert not hasattr(bank, "update_target")


def test_prototype_prediction_and_agreement_use_only_trusted_teacher_outputs():
    from methods.structure_da.prototype_losses import (
        prototype_agreement_mask,
        shape_prototype_predictions,
    )

    bank = _initialized_bank()
    teacher_shape = torch.tensor([[4., 1.], [1., 5.], [3., 1.]])
    pseudo = torch.tensor([0, 0, 1])
    trusted = torch.tensor([True, True, False])
    proto_pred, similarity = shape_prototype_predictions(teacher_shape, bank)
    agreement = prototype_agreement_mask(trusted, pseudo, proto_pred)

    assert similarity.shape == (3, 2)
    assert proto_pred.tolist() == [0, 1, 0]
    assert agreement.tolist() == [True, False, False]


def test_source_prototype_alignment_empty_mask_is_finite_zero():
    from methods.structure_da.prototype_losses import source_prototype_center_alignment

    target = torch.randn(4, 2, requires_grad=True)
    result = source_prototype_center_alignment(
        target, torch.tensor([0, 0, 1, 1]), torch.full((4,), .99),
        torch.zeros(4, dtype=torch.bool), _initialized_bank(),
        pseudo_threshold=.9,
    )
    assert result["valid_classes"] == 0
    assert result["total_loss"].item() == 0
    assert torch.isfinite(result["total_loss"])


def test_source_prototype_alignment_one_class_has_target_gradient_only():
    from methods.structure_da.prototype_losses import source_prototype_center_alignment

    bank = _initialized_bank()
    target = torch.tensor([[1., 1.], [1., 1.]], requires_grad=True)
    result = source_prototype_center_alignment(
        target, torch.zeros(2, dtype=torch.long), torch.full((2,), .95),
        torch.ones(2, dtype=torch.bool), bank, pseudo_threshold=.9,
    )
    assert result["valid_classes"] == 1
    assert result["total_loss"] > 0
    result["total_loss"].backward()
    assert target.grad is not None and target.grad.abs().sum() > 0
    assert bank.prototypes.grad is None


def test_source_prototype_alignment_multiclass_reliability_weighting_is_exact():
    from methods.structure_da.prototype_losses import source_prototype_center_alignment

    bank = _initialized_bank()
    target = torch.tensor([
        [1., 0.], [1., 0.],
        [1., 0.], [1., 0.], [1., 0.], [1., 0.],
    ], requires_grad=True)
    result = source_prototype_center_alignment(
        target, torch.tensor([0, 0, 1, 1, 1, 1]),
        torch.full((6,), .95), torch.ones(6, dtype=torch.bool), bank,
        pseudo_threshold=.9, min_target_support=2, support_saturation=4,
    )
    assert result["valid_classes"] == 2
    assert result["target_mean_cos"].item() == pytest.approx(.5)
    assert result["total_loss"].item() == pytest.approx(2. / 3., abs=1e-6)


def test_source_prototype_mode_does_not_call_batch_relative_alignment(monkeypatch):
    import timematch

    monkeypatch.setattr(
        timematch, "class_relative_domain_alignment",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("batch align called")),
    )
    bank = _initialized_bank()
    output = {"shapelet_response": torch.tensor([[1., 0.], [1., 0.]])}
    result = timematch.compute_shape_da_alignment(
        "source_prototype", output, torch.zeros(2, dtype=torch.long),
        output, torch.zeros(2, dtype=torch.long), torch.full((2,), .95),
        torch.ones(2, dtype=torch.bool), .9,
        prototype_bank=bank, prototype_agreement=torch.ones(2, dtype=torch.bool),
    )
    assert result["valid_classes"] == 1


def test_batch_align_dispatch_is_numerically_identical_to_existing_helper():
    import timematch
    from methods.structure_da.prototype_losses import class_relative_domain_alignment

    source = {"shapelet_response": torch.tensor([[1., 0.], [0., 1.]])}
    target = {"shapelet_response": torch.tensor([[.8, .2], [.2, .8]])}
    labels = torch.tensor([0, 1])
    confidence = torch.tensor([.95, .99])
    trusted = torch.ones(2, dtype=torch.bool)
    actual = timematch.compute_shape_da_alignment(
        "batch_align", source, labels, target, labels, confidence, trusted, .9,
    )
    expected = class_relative_domain_alignment(
        source["shapelet_response"], labels, target["shapelet_response"],
        labels, confidence, .9, distance="mse",
        min_target_support=2, support_saturation=4,
    )
    torch.testing.assert_close(actual["total_loss"], expected["total_loss"])
    assert actual["valid_classes"] == expected["valid_classes"]


def test_prototype_mode_reuses_v2clean_loss_formula_and_source_shape_classifier():
    from methods.structure_da.prototype_losses import compose_structure_v2clean_da_loss
    from models.stclassifier import PseStructureProtoLTae

    values = [torch.tensor(float(value)) for value in range(1, 6)]
    total = compose_structure_v2clean_da_loss(
        *values, ramp=.4, trade_off=2., shape_weight=.1,
        diversity_weight=.01, shape_align_weight=.05,
    )
    expected = values[0] + 2 * values[1] + .1 * values[2] + .01 * values[3] + .4 * .05 * values[4]
    torch.testing.assert_close(total, expected)

    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=10, shapelet_count=16,
        fourier_num_modes=5, dropout=0,
    )
    assert model.shape_classifier.in_features == 32
    assert model.shape_classifier.out_features == 3


def test_batch_align_remains_default_and_formal_masks_are_separate():
    import timematch
    import train

    trainer = inspect.getsource(timematch._train_structure_proto_timematch)
    parser_source = inspect.getsource(train)
    assert 'default="batch_align"' in parser_source
    assert 'teacher_output["shapelet_response"]' in trainer
    assert 'target_output["logits"], training_target_labels,\n                trusted_mask' in trainer
    assert 'prototype_agreement=proto_agree_mask' in trainer
    assert 'shape_prototype_bank.update_source(' in trainer
    assert trainer.index("optimizer.step()") < trainer.index("shape_prototype_bank.update_source(")


def test_existing_source_checkpoint_shape_still_loads_strictly():
    from models.stclassifier import PseStructureProtoLTae

    kwargs = dict(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=10, shapelet_count=16,
        fourier_num_modes=5, dropout=0,
    )
    source = PseStructureProtoLTae(**kwargs)
    target = PseStructureProtoLTae(**kwargs)
    incompatible = target.load_state_dict(source.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_shapeproto_launcher_reuses_source_and_freezes_protocol():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'SOURCE_ROOT="${SOURCE_ROOT:-outputs/structure_proto_v2clean_4tasks_seed1/source}"' in source
    assert source.count('--shape-da-mode source_prototype') == 1
    assert source.count('--fourier_num_modes 13') == 1
    assert '--shapelet-count 16' in source
    assert '--shape-window-scales 24 --shape-window-stride 8' in source
    assert '--pseudo_threshold 0.9' in source
    assert '--oracle-pseudo-labels false' in source
    assert '--adaptive-pseudo-selection false' in source
    assert '--epochs 20 --steps_per_epoch 500' in source
    assert '--epochs 100' not in source
    assert 'fold_0/model.pt' in source
    assert 'python -u train.py' in source
