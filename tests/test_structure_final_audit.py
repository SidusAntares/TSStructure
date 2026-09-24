import torch


def _model():
    from models.stclassifier import PseStructureProtoLTae

    torch.manual_seed(71)
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=10, shape_window_scales=(8, 16),
        shape_window_stride=8, shapelet_count=4, fourier_num_modes=5,
        dropout=0,
    ).eval()
    model.temporal_encoder.attention_heads.dropout.p = 0
    return model


def _batch():
    return {
        "pixels": torch.randn(4, 7, 3, 5),
        "valid_pixels": torch.ones(4, 7, 5),
        "positions": torch.arange(7).repeat(4, 1) * 30,
        "extra": torch.empty(4, 7, 0),
        "label": torch.tensor([0, 1, 2, 1]),
    }


def test_official_diagnostic_full_and_cached_paths_are_equivalent():
    from analysis.structure_final_audit import compare_forward_paths

    result = compare_forward_paths(_model(), _batch(), temporal_shift=7)
    for comparison in ("official_vs_full", "official_vs_cached"):
        assert set(result[comparison]) == {
            "logits", "instance_feature", "shapelet_response", "shape_class_token",
        }
        assert max(result[comparison].values()) < 1e-6


def test_response_interventions_only_replace_response_and_query():
    from analysis.structure_final_audit import response_intervention_outputs

    model, batch = _model(), _batch()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    result = response_intervention_outputs(model, batch, temporal_shift=5, seed=11)
    assert set(result) == {"FULL", "ZERO", "MEAN", "SHUFFLE"}
    assert torch.equal(result["FULL"]["spatial"], result["ZERO"]["spatial"])
    assert torch.equal(result["FULL"]["positions"], result["SHUFFLE"]["positions"])
    assert torch.count_nonzero(result["ZERO"]["qshape"]) == 0
    assert torch.allclose(
        result["MEAN"]["response"],
        result["MEAN"]["response"][0].expand_as(result["MEAN"]["response"]),
    )
    permutation = result["SHUFFLE"]["permutation"]
    assert torch.equal(
        result["SHUFFLE"]["response"], result["FULL"]["response"][permutation],
    )
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_beta_five_matches_dictionary_and_hard_max_selects_candidate_maximum():
    from analysis.structure_final_audit import response_from_similarity

    dictionary = _model().structure_branch.shapelet_dictionary
    tokens = torch.randn(3, 5, 10)
    similarity = dictionary.compute_similarity(tokens)
    expected = dictionary(tokens)
    actual, weights = response_from_similarity(similarity, beta=5)
    assert torch.allclose(actual, expected)
    hard, hard_weights = response_from_similarity(similarity, hard_max=True)
    assert torch.allclose(hard, similarity.max(dim=1).values)
    assert torch.equal(hard_weights.argmax(dim=1), similarity.argmax(dim=1))
    assert torch.allclose(hard_weights.sum(dim=1), torch.ones_like(hard_weights.sum(dim=1)))


def test_layer_statistics_report_candidate_and_sample_variance():
    from analysis.structure_final_audit import layer_statistics

    values = torch.randn(5, 7, 11)
    result = layer_statistics(values)
    assert result["sample_variance"] > 0
    assert result["candidate_variance"] > 0
    assert result["feature_norm_mean"] > 0
    assert 1 <= result["effective_rank"] <= 11


def test_checkpoint_paths_include_source_and_uda_complete_locations(tmp_path, capsys):
    from scripts.diagnose_structure_final_audit import resolve_checkpoints

    result = resolve_checkpoints(tmp_path, "AT1", "DK1")
    assert result == {"source": None, "uda": None}
    text = capsys.readouterr().out
    assert str(tmp_path / "source" / "source_AT1_seed1" / "fold_0" / "model.pt") in text
    assert str(tmp_path / "uda" / "AT1_DK1_seed1" / "fold_0" / "model.pt") in text

