from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from utils.focal_loss import FocalLoss


def test_weighted_focal_with_unit_weights_matches_existing_focal():
    from methods.structure_da.prototype_losses import weighted_focal_loss

    logits = torch.tensor([[2., -1.], [-.5, 1.5], [1., .5]])
    labels = torch.tensor([0, 1, 0])
    expected = FocalLoss(gamma=1.)(logits, labels)
    actual = weighted_focal_loss(logits, labels, torch.ones(2), gamma=1.)
    torch.testing.assert_close(actual, expected)


def test_minority_class_weights_follow_formula_and_normalize_mean():
    from methods.structure_da.prototype_losses import minority_class_weights

    weights = minority_class_weights(torch.tensor([100., 25., 4.]))
    assert float(weights.mean()) == pytest.approx(1.)
    assert float(weights.min()) >= .5 / 1.5
    assert float(weights.max()) <= 2. / .75
    raw = torch.tensor([100., 25., 4.]).rsqrt()
    expected = (raw / raw.mean()).clamp(.5, 2.)
    expected = expected / expected.mean()
    torch.testing.assert_close(weights, expected)


def test_support_aware_compactness_is_zero_for_perfect_matches():
    from methods.structure_da.prototype_losses import support_aware_prototype_compactness
    from models.structure_da.prototype_bank import ClassPrototypeBank

    bank = ClassPrototypeBank(2, 2)
    bank.prototypes.copy_(torch.eye(2))
    bank.initialized.fill_(True)
    result = support_aware_prototype_compactness(
        torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]),
        bank, torch.tensor([1., .5]),
    )
    assert float(result["loss"]) == pytest.approx(0.)


def test_compactness_class_average_is_not_sample_count_weighted():
    from methods.structure_da.prototype_losses import support_aware_prototype_compactness
    from models.structure_da.prototype_bank import ClassPrototypeBank

    bank = ClassPrototypeBank(2, 2)
    bank.prototypes.copy_(torch.eye(2))
    bank.initialized.fill_(True)
    labels = torch.tensor([0, 1])
    features = torch.tensor([[0., 1.], [0., 1.]])
    base = support_aware_prototype_compactness(
        features, labels, bank, torch.ones(2),
    )["loss"]
    duplicated = support_aware_prototype_compactness(
        torch.cat((features[:1].repeat(20, 1), features[1:])),
        torch.cat((labels[:1].repeat(20), labels[1:])), bank, torch.ones(2),
    )["loss"]
    torch.testing.assert_close(base, duplicated)


def test_source_loss_base_is_exact_v2clean_formula():
    from methods.structure_da.prototype_losses import (
        compose_minority_source_loss,
        compose_structure_v4_source_loss,
    )

    values = [torch.tensor(value) for value in (2., 3., 5.)]
    expected = compose_structure_v4_source_loss(*values, .1, .01)
    actual = compose_minority_source_loss(
        *values, values[0].new_zeros(()), mode="base",
        prototype_ramp_value=.1, shape_weight=.1, diversity_weight=.01,
    )
    torch.testing.assert_close(actual, expected)


def test_minority_mode_does_not_enter_v2clean_uda_loss():
    from methods.structure_da.prototype_losses import compose_structure_v2clean_da_loss

    values = [torch.tensor(value) for value in (1., 2., 3., 4., 5.)]
    expected = compose_structure_v2clean_da_loss(*values, ramp=.4)
    for _mode in ("base", "weighted", "weighted_proto"):
        actual = compose_structure_v2clean_da_loss(*values, ramp=.4)
        torch.testing.assert_close(actual, expected)


class _FakeBranch:
    def forward_from_context(self, context, temporal_shift, include_legacy_query):
        assert include_legacy_query is True
        return {"shape_class_token": context + float(temporal_shift)}


class _FakeModel:
    def __init__(self):
        self.structure_branch = _FakeBranch()
        self.positions_seen = []

    def prepare_structure_context(self, spatial, positions):
        return spatial

    def _encode_instance(self, spatial, shifted_positions, structure):
        self.positions_seen.append(shifted_positions.clone())
        return structure["shape_class_token"]

    @staticmethod
    def decoder(instance):
        return instance


def test_shift_audit_changes_query_while_holding_temporal_positions_fixed():
    from analysis.v2clean_shift_ab_audit import isolated_structure_shift_logits

    model = _FakeModel()
    positions = torch.tensor([[10, 20]], dtype=torch.long)
    base, shifted, _, _ = isolated_structure_shift_logits(
        model, torch.tensor([[1., 2.]]), positions, delta=15,
    )
    assert torch.equal(model.positions_seen[0], positions + 15)
    assert torch.equal(model.positions_seen[1], positions + 15)
    assert not torch.equal(base, shifted)


def test_launcher_has_s_then_four_ab_pipelines_and_uda_base_mode():
    source = Path("scripts/run_v2clean_shift_ab_2tasks_seed1.sh").read_text()
    assert source.index("run_shift_sensitivity") < source.index("run_pipeline")
    assert 'run_pipeline "$GPU0" A DK1' in source
    assert 'run_pipeline "$GPU1" A FR1' in source
    assert 'run_pipeline "$GPU2" B DK1' in source
    assert 'run_pipeline "$GPU3" B FR1' in source
    assert "--source-minority-mode base" in source


def test_launcher_does_not_expand_task_inside_its_declaring_local_statement():
    source = Path("scripts/run_v2clean_shift_ab_2tasks_seed1.sh").read_text()
    unsafe = 'local gpu="$1" source="$2" task="$3" log="$LOG_ROOT/Base_${task}.log"'
    assert unsafe not in source
    assert 'local log="$LOG_ROOT/Base_${task}.log"' in source
