from pathlib import Path

import numpy as np
import pytest
import torch

from models.stclassifier import PseStructureProtoLTae


def _model(representation):
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation=representation, shape_injection="current_query",
        dropout=0.,
    ).eval()


def _structure(model):
    pixels = torch.randn(3, 8, 3, 4)
    valid = torch.ones(3, 8, 4)
    positions = torch.arange(8).repeat(3, 1) * 40
    spatial = model.spatial_encoder(pixels, valid, torch.zeros(3, 4))
    return model.prepare_structure(spatial, positions)


def test_four_boundary_stages_are_taken_from_real_forward_for_all_variants():
    from analysis.structure_layer_boundary_audit import extract_boundary_stages

    for representation in ("current", "set_response", "residual_response"):
        model = _model(representation)
        structure = _structure(model)
        stages = extract_boundary_stages(structure)
        assert set(stages) == {
            "shape_token_ordered", "anchor_similarity_ordered",
            "anchor_similarity_sorted", "shape_response",
        }
        torch.testing.assert_close(
            stages["shape_token_ordered"], structure["shape_tokens"].flatten(1),
        )
        torch.testing.assert_close(
            stages["anchor_similarity_ordered"],
            structure["shapelet_similarity"].flatten(1),
        )
        torch.testing.assert_close(
            stages["anchor_similarity_sorted"],
            structure["shapelet_similarity"].sort(dim=1).values.transpose(1, 2).flatten(1),
        )
        torch.testing.assert_close(
            stages["shape_response"], structure["shapelet_response"],
        )
        restored = _model(representation)
        restored.load_state_dict(model.state_dict(), strict=True)


def test_sorted_similarity_is_window_permutation_invariant_but_tokens_are_not():
    from analysis.structure_layer_boundary_audit import extract_boundary_stages

    tokens = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    similarity = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
    response = torch.randn(2, 4)
    permutation = torch.tensor([2, 0, 3, 1])
    original = extract_boundary_stages({
        "shape_tokens": tokens,
        "shapelet_similarity": similarity,
        "shapelet_response": response,
    })
    permuted = extract_boundary_stages({
        "shape_tokens": tokens[:, permutation],
        "shapelet_similarity": similarity[:, permutation],
        "shapelet_response": response,
    })
    assert not torch.equal(
        original["shape_token_ordered"], permuted["shape_token_ordered"],
    )
    torch.testing.assert_close(
        original["anchor_similarity_sorted"],
        permuted["anchor_similarity_sorted"],
    )


def test_class_distance_matrix_is_symmetric_with_zero_diagonal():
    from analysis.structure_layer_boundary_audit import class_distance_matrix

    centers = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    distance = class_distance_matrix(centers)
    np.testing.assert_allclose(distance, distance.T)
    np.testing.assert_allclose(np.diag(distance), 0.)


def test_identical_class_relations_score_one_without_nan():
    from analysis.structure_layer_boundary_audit import relation_metrics

    distance = np.array([
        [0., .2, .8], [.2, 0., .5], [.8, .5, 0.],
    ])
    result = relation_metrics(distance, distance, tie_tolerance=1e-6)
    assert result["triplet_order_agreement"] == pytest.approx(1.)
    assert result["distance_rank_correlation"] == pytest.approx(1.)
    assert result["nearest_class_agreement"] == pytest.approx(1.)
    assert result["valid_triplets"] > 0
    assert all(np.isfinite(value) for value in result.values())


def test_triplet_flip_decreases_agreement_and_ties_are_skipped():
    from analysis.structure_layer_boundary_audit import relation_metrics

    source = np.array([
        [0., .2, .8], [.2, 0., .5], [.8, .5, 0.],
    ])
    flipped = np.array([
        [0., .9, .1], [.9, 0., .5], [.1, .5, 0.],
    ])
    tied = source.copy()
    tied[0, 1] = tied[1, 0] = .5
    tied[0, 2] = tied[2, 0] = .5

    flipped_result = relation_metrics(source, flipped, tie_tolerance=1e-6)
    tied_result = relation_metrics(source, tied, tie_tolerance=1e-6)
    assert flipped_result["triplet_order_agreement"] < 1.
    assert tied_result["valid_triplets"] < relation_metrics(
        source, source, tie_tolerance=1e-6,
    )["valid_triplets"]


def test_audit_scope_is_three_variants_four_tasks_source_val_only():
    import analysis.structure_layer_boundary_audit as audit

    assert audit.VARIANTS == ("current", "set_response", "residual_response")
    assert set(audit.TASKS) == {"AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1"}
    assert audit.STAGES == (
        "shape_token_ordered", "anchor_similarity_ordered",
        "anchor_similarity_sorted", "shape_response",
    )
    source = Path("analysis/structure_layer_boundary_audit.py").read_text(
        encoding="utf-8",
    )
    assert "target_test" not in source
    assert "source_test" not in source
    assert "/uda/" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source


def test_residual_current_relation_deltas_use_shape_response_only():
    from analysis.structure_layer_boundary_audit import residual_current_deltas

    rows = []
    for variant, offset in (("current", 0.), ("residual_response", .1)):
        rows.append({
            "variant": variant, "task": "AT1_DK1", "stage": "shape_response",
            "triplet_order_agreement": .5 + offset,
            "distance_rank_correlation": .4 + offset,
            "nearest_class_agreement": .3 + offset,
        })
        rows.append({
            "variant": variant, "task": "AT1_DK1",
            "stage": "anchor_similarity_sorted",
            "triplet_order_agreement": 0.,
            "distance_rank_correlation": 0.,
            "nearest_class_agreement": 0.,
        })
    result = residual_current_deltas(rows)
    assert result == [{
        "task": "AT1_DK1",
        "triplet_order_agreement_delta": pytest.approx(.1),
        "distance_rank_correlation_delta": pytest.approx(.1),
        "nearest_class_agreement_delta": pytest.approx(.1),
    }]

