from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analysis.structure_representation_chain_audit import (
    ANCHOR_COVERAGE_FIELDS,
    REPRESENTATION_FIELDS,
    REPRESENTATION_GEOMETRY_FIELDS,
    REPRESENTATION_PER_CLASS_FIELDS,
    anchor_coverage_rows,
    audit_split_indices,
    common_class_names,
    compose_chain_representations,
    fit_source_probe,
    pixel_budget_batches,
    remap_to_classes,
    target_oracle_probe,
    extract_representation_stages,
    effective_rank,
    normalize_checkpoint_config,
)
from models.stclassifier import PseStructureProtoLTae
from models.structure_da.discriminative_structure import DiscriminativeStructureBranch


def _chain_inputs():
    torch.manual_seed(3)
    branch = DiscriminativeStructureBranch(
        channels=4, shape_dim=128, num_modes=13, grid_points=64,
        window_scales=(24,), window_stride=8, shapelet_count=16,
        shapelet_beta=5., shape_resample_length=16,
    ).eval()
    features = torch.randn(2, 30, 4)
    positions = torch.arange(30).repeat(2, 1) * 12
    return branch, features, positions


def test_chain_shapes_match_v2clean_q24_stride8_m16():
    branch, features, positions = _chain_inputs()
    with torch.no_grad():
        fourier, _ = branch.exposer(features, positions)
        groups, _ = branch.window_extractor(fourier)
        tokens = torch.cat([branch.token_generator(group) for group in groups], dim=1)
        similarity = branch.shapelet_dictionary.compute_similarity(tokens)
        representations = compose_chain_representations(
            fourier, tokens, similarity, branch.response_to_query,
            beta=branch.shapelet_dictionary.beta,
        )

    assert fourier.shape == (2, 64, 4)
    assert tokens.shape == (2, 8, 128)
    assert similarity.shape == (2, 8, 16)
    assert representations["fourier_flat"].shape == (2, 64 * 4)
    assert representations["shape_token_flat"].shape == (2, 8 * 128)
    assert representations["anchor_ordered"].shape == (2, 8 * 16)
    assert representations["anchor_sorted"].shape == (2, 8 * 16)
    assert representations["shape_strength"].shape == (2, 16)
    assert representations["shape_response"].shape == (2, 32)
    assert representations["qshape"].shape == (2, 128)


def test_candidate_permutation_changes_only_ordered_representation():
    torch.manual_seed(7)
    fourier = torch.randn(2, 64, 5)
    tokens = torch.randn(2, 8, 6)
    similarity = torch.randn(2, 8, 4)
    query = torch.nn.Linear(8, 3, bias=False)
    permutation = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4])

    original = compose_chain_representations(
        fourier, tokens, similarity, query, beta=5.,
    )
    permuted = compose_chain_representations(
        fourier, tokens[:, permutation], similarity[:, permutation], query,
        beta=5.,
    )

    assert not torch.equal(original["anchor_ordered"], permuted["anchor_ordered"])
    torch.testing.assert_close(original["anchor_sorted"], permuted["anchor_sorted"])
    torch.testing.assert_close(original["shape_strength"], permuted["shape_strength"])
    torch.testing.assert_close(original["shape_response"], permuted["shape_response"])
    torch.testing.assert_close(original["qshape"], permuted["qshape"])


def test_source_probe_scaler_and_classifier_fit_source_train_only():
    source_train = np.array([[-3., 0.], [-1., 0.], [1., 0.], [3., 0.]])
    source_labels = np.array([0, 0, 1, 1])
    source_val = np.array([[-2., 0.], [2., 0.]])
    target_val = np.array([[100., 0.], [200., 0.]])

    result = fit_source_probe(
        source_train, source_labels, source_val, np.array([0, 1]),
        target_val, np.array([0, 1]), class_ids=np.array([0, 1]),
    )

    np.testing.assert_allclose(result["estimator"].named_steps["scaler"].mean_, [0., 0.])
    assert result["source_val_predictions"].shape == (2,)
    assert result["target_predictions"].shape == (2,)


def test_target_oracle_probe_uses_only_target_val_and_reduces_folds():
    features = np.array([
        [-2., 0.], [-1., 0.], [-.5, 0.],
        [.5, 0.], [1., 0.], [2., 0.],
        [9., 9.],
    ])
    labels = np.array([0, 0, 0, 1, 1, 1, 2])
    result = target_oracle_probe(features, labels, class_ids=np.array([0, 1, 2]), seed=1)

    assert result["folds"] == 3
    assert result["available_classes"].tolist() == [0, 1]
    assert np.isnan(result["per_class_f1"][2])
    assert result["predictions"].shape == labels.shape


def test_split_replay_is_independent_and_never_returns_test_indices():
    calls = []

    def fold_creator(datasets, num_folds, eligible, val_ratio, test_ratio):
        calls.append(tuple(datasets))
        marker = 10 if datasets[0] == datasets[1] else 20
        return [{
            name: {
                "train": {marker + index},
                "val": {marker + index + 1},
                "test": {marker + index + 2},
            }
            for index, name in enumerate(dict.fromkeys(datasets))
        }]

    result = audit_split_indices(
        "source", "target", {"source": [1], "target": [2]},
        seed=1, val_ratio=.1, test_ratio=.2, fold_creator=fold_creator,
    )

    assert calls == [("source", "source"), ("source", "target")]
    assert result == {
        "source_train": {10}, "source_val": {11}, "target_val": {22},
    }
    assert all("test" not in key for key in result)


def test_common_classes_and_label_remap_are_name_based():
    common = common_class_names(
        ["corn", "meadow", "peas"],
        ["corn", "barley", "meadow"],
        ["meadow", "corn", "wheat"],
    )
    assert common == ["corn", "meadow"]

    features = np.arange(12).reshape(4, 3)
    labels = np.array([0, 2, 1, 0])
    remapped_features, remapped_labels = remap_to_classes(
        features, labels, ["corn", "peas", "meadow"], common,
    )
    np.testing.assert_array_equal(remapped_features, features[[0, 1, 3]])
    np.testing.assert_array_equal(remapped_labels, [0, 1, 0])


def test_anchor_coverage_and_margin_are_aggregated_exactly():
    similarity = np.array([[
        [.9, .4, .1], [.7, .6, .2],
    ], [
        [.8, .3, .2], [.5, .4, .1],
    ]])
    rows = anchor_coverage_rows(
        "AT1_DK1", "task_native", "target", similarity,
        np.array([0, 0]), ["corn"],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["support"] == 2
    assert row["mean_anchor_coverage"] == pytest.approx(.725)
    assert row["std_anchor_coverage"] == pytest.approx(np.std([.9, .7, .8, .5]))
    assert row["mean_anchor_margin"] == pytest.approx(.3)
    assert row["std_anchor_margin"] == pytest.approx(np.std([.5, .1, .5, .1]))


def test_pixel_budget_batches_prevent_padding_memory_explosion():
    pixel_counts = [64] * 128 + [4096] * 4
    batches = pixel_budget_batches(
        pixel_counts, max_batch_size=128, pixel_budget=8192,
    )

    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(len(pixel_counts)))
    assert len(flattened) == len(set(flattened))
    for batch in batches:
        padded_pixels = len(batch) * max(pixel_counts[index] for index in batch)
        assert padded_pixels <= 8192
    assert max(len(batch) for batch in batches if pixel_counts[batch[0]] == 4096) == 2


@pytest.mark.parametrize(
    "fields,required",
    [
        (REPRESENTATION_FIELDS, {
            "variant", "task", "class_protocol", "representation", "feature_dim",
            "effective_rank",
            "source_val_macro_f1", "target_oracle_macro_f1",
            "source_to_target_macro_f1", "source_to_target_knn_macro_f1",
        }),
        (REPRESENTATION_PER_CLASS_FIELDS, {
            "variant", "task", "class_protocol", "representation", "class",
            "source_support", "target_support", "source_val_f1",
            "target_oracle_f1", "source_to_target_f1",
            "source_to_target_knn_f1",
        }),
        (ANCHOR_COVERAGE_FIELDS, {
            "task", "class_protocol", "domain", "class", "support",
            "mean_anchor_coverage", "std_anchor_coverage",
            "mean_anchor_margin", "std_anchor_margin",
        }),
        (REPRESENTATION_GEOMETRY_FIELDS, {
            "variant", "task", "class_protocol", "representation", "class",
            "source_class_radius", "target_class_radius",
            "same_class_source_distance", "nearest_wrong_source_distance",
            "cross_domain_margin",
        }),
    ],
)
def test_csv_schema_is_fixed(fields, required):
    assert set(fields) == required


def test_launcher_is_four_task_two_variant_source_only_dry_run():
    launcher = Path("scripts/run_structure_representation_chain_audit_seed1.sh")
    source = launcher.read_text(encoding="utf-8")
    assert all(f"GPU{index}" in source for index in range(4))
    assert all(task in source for task in ("AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1"))
    assert "outputs/structure_proto_v2clean_4tasks_seed1/source" in source
    assert "outputs/structure_set_response_4tasks_seed1/source" in source
    assert "/uda/" not in source.lower()
    assert "uda_checkpoint" not in source.lower()
    assert "DRY_RUN" in source
    assert 'PIXEL_BUDGET="${PIXEL_BUDGET:-8192}"' in source
    assert '--pixel-budget "$PIXEL_BUDGET"' in source


def _audit_model(representation):
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation=representation, shape_injection="current_query",
    ).eval()


@pytest.mark.parametrize("representation", ["current", "set_response"])
def test_real_model_three_stage_chain_is_exact(representation):
    model = _audit_model(representation)
    pixels = torch.randn(2, 8, 3, 4)
    valid = torch.ones(2, 8, 4)
    positions = torch.arange(8).repeat(2, 1) * 40
    spatial = model.spatial_encoder(pixels, valid, torch.zeros(2, 4))
    structure = model.prepare_structure(spatial, positions)
    stages = extract_representation_stages(model, structure)
    projection = model.temporal_encoder.attention_heads.external_query_projection
    assert set(stages) == {"shape_response", "shape_query_feature", "query_delta"}
    assert stages["shape_response"].shape == (2, 32)
    assert stages["shape_query_feature"].shape == (2, 8)
    assert stages["query_delta"].shape == (2, 8)
    torch.testing.assert_close(
        stages["shape_query_feature"],
        model.structure_branch.response_to_query(stages["shape_response"]),
    )
    torch.testing.assert_close(
        stages["query_delta"], projection(stages["shape_query_feature"]).flatten(1),
    )
    assert not any(value.requires_grad for value in stages.values())


def test_effective_rank_uses_same_definition_for_every_stage():
    identity = np.eye(4, dtype=np.float32)
    assert effective_rank(identity) == pytest.approx(4.)


def test_chain_audit_is_four_task_two_variant_source_only():
    import analysis.structure_representation_chain_audit as audit

    assert set(audit.TASKS) == {"AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1"}
    assert audit.REPRESENTATIONS == (
        "shape_response", "shape_query_feature", "query_delta",
    )
    source = Path("scripts/run_structure_representation_chain_audit_seed1.sh").read_text()
    assert "outputs/structure_proto_v2clean_4tasks_seed1/source" in source
    assert "outputs/structure_set_response_4tasks_seed1/source" in source
    assert "outputs/structure_response_query_chain_audit_seed1" in source
    for task in audit.TASKS:
        assert task in source
    assert "/uda/" not in source.lower()
    script = Path("analysis/structure_representation_chain_audit.py").read_text()
    assert "target_test" not in script
    assert "source_test" not in script
    assert ".backward(" not in script
    assert "torch.optim" not in script


def test_legacy_current_checkpoint_config_is_adapted_without_weakening_variant_checks():
    legacy = {"model": "psestructureprotoltae"}
    adapted = normalize_checkpoint_config(legacy, "current")
    assert adapted["shape_representation"] == "current"
    assert adapted["shape_injection"] == "current_query"
    assert adapted["structure_shift_mode"] == "none"
    assert "shape_representation" not in legacy

    with pytest.raises(ValueError, match="missing shape_representation"):
        normalize_checkpoint_config(legacy, "set_response")

    with pytest.raises(ValueError, match="does not match"):
        normalize_checkpoint_config(
            {"shape_representation": "set_response"}, "current",
        )


def test_merge_initializes_all_five_output_collections(monkeypatch):
    import analysis.structure_representation_chain_audit as audit

    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: "{}")
    monkeypatch.setattr(Path, "write_text", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(audit, "read_csv", lambda path: [])
    monkeypatch.setattr(audit, "write_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit, "_print_summary", lambda *args: None)
    audit.merge_outputs(SimpleNamespace(
        output_root="unused", code_version="test", seed=1,
    ))
