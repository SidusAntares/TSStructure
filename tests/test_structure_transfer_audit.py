import copy

import numpy as np
import torch

from analysis.structure_transfer_audit import (
    deterministic_class_indices,
    forward_intervention,
    forward_prepared_intervention,
    gradient_conflict_metrics,
    prepare_intervention,
    query_geometry,
    remove_direction,
)
from models.stclassifier import PseStructureProtoLTae


def _model():
    torch.manual_seed(4)
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4, 8], mlp2=[16, 16], with_extra=False,
        n_head=2, d_k=4, d_model=16, mlp3=[16, 8], mlp4=[8, 4],
        num_classes=3, dropout=0.0, max_temporal_shift=10, shape_dim=8,
        shape_window_scales=(8, 16, 24), shape_window_stride=8,
        shapelet_count=4, shape_resample_length=8, fourier_num_modes=5,
    ).eval()
    return model


def _batch():
    torch.manual_seed(7)
    return {
        "pixels": torch.randn(3, 9, 3, 5),
        "valid_pixels": torch.ones(3, 9, 5),
        "positions": torch.tensor([
            [0, 35, 72, 110, 145, 180, 220, 270, 330],
            [3, 39, 75, 115, 150, 185, 225, 275, 335],
            [6, 42, 79, 119, 155, 190, 230, 280, 340],
        ]),
        "extra": torch.zeros(3, 4),
        "label": torch.tensor([0, 1, 2]),
    }


def test_query_alpha_one_matches_normal_forward():
    model, batch = _model(), _batch()
    with torch.no_grad():
        expected = model(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch["extra"], return_dict=True,
        )
        actual = forward_intervention(model, batch, query_alpha=1.0)
    assert torch.equal(expected["logits"], actual["logits"])
    assert torch.equal(expected["instance_feature"], actual["instance_feature"])


def test_full_component_and_scale_match_normal_forward():
    model, batch = _model(), _batch()
    with torch.no_grad():
        expected = model(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch["extra"], return_dict=True,
        )
        actual = forward_intervention(
            model, batch, component_mode="FULL", scale_mode="FULL",
        )
    for key in ("logits", "shape_logits", "shapelet_response", "shape_class_token"):
        assert torch.equal(expected[key], actual[key])


def test_prepared_intervention_matches_one_shot_intervention():
    model, batch = _model(), _batch()
    with torch.no_grad():
        expected = forward_intervention(model, batch, query_alpha=0.5)
        prepared = prepare_intervention(model, batch)
        actual = forward_prepared_intervention(
            model, prepared, batch["positions"], query_alpha=0.5,
        )
    assert torch.equal(expected["logits"], actual["logits"])
    assert torch.equal(expected["shapelet_response"], actual["shapelet_response"])


def test_gradient_audit_does_not_modify_parameters_or_grad_fields():
    model, batch = _model(), _batch()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    output = forward_intervention(model, batch)
    main = torch.nn.functional.cross_entropy(output["logits"], batch["label"])
    shape = 0.1 * torch.nn.functional.cross_entropy(output["shape_logits"], batch["label"])
    rows = gradient_conflict_metrics(
        main, shape, {"pse": tuple(model.spatial_encoder.parameters())},
    )
    assert rows["pse"]["main_grad_norm"] > 0
    assert all(value.grad is None for value in model.parameters())
    assert all(torch.equal(before[name], value) for name, value in model.named_parameters())


def test_direction_removal_preserves_shape_and_removes_projection():
    values = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 4.0, 2.0]])
    direction = torch.tensor([1.0, 0.0, 0.0])
    cleaned = remove_direction(values, direction)
    assert cleaned.shape == values.shape
    assert torch.allclose(cleaned[:, 0], torch.zeros(2))
    assert torch.equal(cleaned[:, 1:], values[:, 1:])


def test_fixed_seed_class_sampling_is_reproducible():
    labels = np.asarray([0] * 10 + [1] * 10 + [2] * 10)
    first = deterministic_class_indices(labels, limit=4, seed=19)
    second = deterministic_class_indices(labels, limit=4, seed=19)
    third = deterministic_class_indices(labels, limit=4, seed=20)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)
    assert all(np.sum(labels[first] == label) == 4 for label in (0, 1, 2))


def test_query_geometry_aggregates_at_cpu_logging_boundary():
    qmaster = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    qshape = torch.tensor([[[3.0, 4.0], [0.0, 5.0]], [[0.0, 5.0], [3.0, 4.0]]])
    result = query_geometry(qmaster, qshape)
    assert result["qmaster_norm"] == 5.0
    assert result["qshape_qmaster_norm_ratio"] == 1.0
    assert np.isfinite(result["qmaster_qshape_cosine"])
    if torch.cuda.is_available():
        mixed = query_geometry(qmaster.cuda(), qshape.cpu())
        assert mixed == result
