import numpy as np
import pytest
import torch
from torch.nn import functional as F

from methods.structure_da.full_model import TSStructureModel
from methods.structure_da.seed_warmup_diagnostic import (
    class_balanced_seed_cross_entropy,
    configure_13b_semantic_student,
    normalize_seed_manifest_rows,
    seed_manifest_fingerprint,
    validate_checkpoint_epochs,
)


def _small_model():
    return TSStructureModel(
        num_classes=3,
        input_dim=10,
        mlp1=[10, 8, 8],
        mlp2=[16, 8],
        n_head=2,
        d_k=2,
        d_model=8,
        ltae_mlp=(8, 6),
        classifier_hidden=(4,),
        trend_num_basis=4,
        structure_num_basis=4,
        canonical_grid_size=8,
        roughness_grid_size=64,
    )


def test_13b_parameter_policy_trains_pse_raw_encoder_classifier_only():
    model = _small_model()
    policy = configure_13b_semantic_student(model)
    trainable = set(policy.trainable_parameter_names)
    assert any(name.startswith("backbone.pixel_set_encoder.") for name in trainable)
    assert any(name.startswith("temporal_module.raw_encoder.") for name in trainable)
    assert any(name.startswith("classifier.") for name in trainable)
    assert not any(name.startswith("backbone.decomposition.") for name in trainable)
    assert not any(name.startswith("temporal_module.trend_geometry.") for name in trainable)
    assert not any(name.startswith("temporal_module.structure_geometry.") for name in trainable)


def test_class_balanced_seed_ce_is_mean_of_present_class_means():
    logits = torch.tensor([
        [3.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 1.0, 0.0],
    ])
    labels = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    per = F.cross_entropy(logits, labels, reduction="none")
    expected = 0.5 * (per[:2].mean() + per[2:].mean())
    actual = class_balanced_seed_cross_entropy(logits, labels)
    torch.testing.assert_close(actual, expected)


def test_class_balanced_seed_ce_does_not_weight_by_seed_count():
    # Duplicate class-1 samples without changing their mean loss: class average
    # must remain unchanged rather than acquiring more weight from sample count.
    base_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    base_labels = torch.tensor([0, 1], dtype=torch.long)
    expanded_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
    expanded_labels = torch.tensor([0, 1, 1, 1], dtype=torch.long)
    torch.testing.assert_close(
        class_balanced_seed_cross_entropy(base_logits, base_labels),
        class_balanced_seed_cross_entropy(expanded_logits, expanded_labels),
    )


def test_seed_manifest_is_label_free_and_cannot_relabel_stage1_candidate():
    rows = [{"sample_id": "10", "pseudo_label": "2", "selection_evidence": "fixed upstream gate"}]
    records = normalize_seed_manifest_rows(rows, valid_sample_ids=[10, 11], initial_candidates={10: 2, 11: 1}, num_classes=3)
    assert [(r.sample_id, r.pseudo_label) for r in records] == [(10, 2)]
    with pytest.raises(ValueError, match="relabels Stage-1 candidate"):
        normalize_seed_manifest_rows([{"sample_id": 10, "pseudo_label": 1}], valid_sample_ids=[10], initial_candidates={10: 2}, num_classes=3)
    with pytest.raises(ValueError, match="forbidden oracle fields"):
        normalize_seed_manifest_rows([{"sample_id": 10, "pseudo_label": 2, "true_label": 2}], valid_sample_ids=[10], initial_candidates={10: 2}, num_classes=3)


def test_seed_manifest_fingerprint_is_deterministic():
    a = normalize_seed_manifest_rows(
        [{"sample_id": 2, "pseudo_label": 1}, {"sample_id": 1, "pseudo_label": 0}],
        valid_sample_ids=[1, 2], initial_candidates={1: 0, 2: 1}, num_classes=3,
    )
    b = tuple(reversed(a))
    assert seed_manifest_fingerprint(a) == seed_manifest_fingerprint(b)


def test_13b_checkpoint_schedule_requires_initial_early_middle_end():
    assert validate_checkpoint_epochs(5, (0, 1, 3, 5)) == (0, 1, 3, 5)
    with pytest.raises(ValueError):
        validate_checkpoint_epochs(5, (0, 5))
    with pytest.raises(ValueError):
        validate_checkpoint_epochs(5, (0, 1, 3, 4))
