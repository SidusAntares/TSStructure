import torch

from models.ltae import LTAE
from models.structure_da.discriminative_structure import (
    DiscriminativeStructureBranch,
    FourierStructureExposer,
    MultiScaleWindowExtractor,
    ShapeletDictionary,
    ShapeTokenGenerator,
    initialize_shapelet_dictionary_from_tokens,
)
from models.stclassifier import PseStructureProtoLTae
from models.fourier_reconstruction import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
)


def test_fourier_exposer_is_differentiable_and_finite():
    torch.manual_seed(0)
    features = torch.randn(2, 11, 6, requires_grad=True)
    positions = torch.arange(11).repeat(2, 1) * 20
    exposer = FourierStructureExposer(13, grid_points=64)
    exposed, grid = exposer(features, positions)
    assert exposed.shape == (2, 64, 6)
    assert grid.shape == (2, 64)
    assert torch.isfinite(exposed).all()
    exposed.square().mean().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_fourier_exposer_matches_existing_finite_fourier_operators():
    torch.manual_seed(1)
    features = torch.randn(1, 9, 4)
    positions = torch.arange(9).unsqueeze(0) * 31
    exposer = FourierStructureExposer(7, grid_points=64)
    actual, grid = exposer(features, positions)
    coefficients, _ = BatchedDirectFourierAnalyzer(7, 365., 1e-3)(features, positions)
    expected = BatchedDirectFourierSynthesizer(7, 365.)(coefficients, grid)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_fourier_exposer_canonical_buffer_matches_dynamic_synthesis_matrix():
    torch.manual_seed(101)
    features = torch.randn(2, 9, 4, dtype=torch.float64)
    positions = torch.arange(9).repeat(2, 1) * 31
    exposer = FourierStructureExposer(7, grid_points=64).double()
    coefficients, _ = exposer.analyzer(features, positions)
    dynamic = exposer.synthesizer(
        coefficients,
        exposer.canonical_grid[None].expand(features.shape[0], -1),
    )
    buffered = exposer.synthesize_canonical(coefficients)
    assert "canonical_synthesis_matrix" in dict(exposer.named_buffers())
    assert torch.allclose(buffered, dynamic, atol=1e-12, rtol=1e-10)


def test_multiscale_windows_are_circular_without_padding():
    curve = torch.arange(64, dtype=torch.float32)[None, :, None]
    groups, scales = MultiScaleWindowExtractor((8, 16, 24), 4)(curve)
    assert [group.shape for group in groups] == [
        (1, 16, 8, 1), (1, 16, 16, 1), (1, 16, 24, 1),
    ]
    assert scales.tolist() == [8] * 16 + [16] * 16 + [24] * 16
    expected_wrap = torch.cat((curve[:, 60:64], curve[:, 0:4]), dim=1)
    assert torch.equal(groups[0][:, -1], expected_wrap)
    extractor = MultiScaleWindowExtractor((8, 16, 24), 4, grid_points=64)
    buffered, buffered_scales = extractor(curve)
    for scale, actual in zip((8, 16, 24), buffered):
        indices = torch.stack([
            (torch.arange(scale) + start) % 64 for start in range(0, 64, 4)
        ])
        assert torch.equal(actual, curve[:, indices])
    assert torch.equal(buffered_scales, scales)
    assert len(dict(extractor.named_buffers())) == 3


def test_shape_components_preserve_level_and_amplitude_information():
    t = torch.linspace(-1, 1, 16)
    base = torch.stack((t, t.square()), dim=-1)[None, None]
    windows = torch.cat((base, base + 4, base * 3), dim=1)
    mask = torch.ones(1, 3, 16, dtype=torch.bool)
    generator = ShapeTokenGenerator(2, shape_dim=12, resample_length=16)
    components = generator.components(windows, mask)
    assert torch.allclose(components["normalized"][:, 0], components["normalized"][:, 1], atol=1e-5)
    assert torch.allclose(components["normalized"][:, 0], components["normalized"][:, 2], atol=1e-5)
    assert torch.allclose(components["difference"][:, 0], components["difference"][:, 1], atol=1e-5)
    assert not torch.allclose(components["mean"][:, 0], components["mean"][:, 1])
    assert not torch.allclose(components["std"][:, 0], components["std"][:, 2])
    tokens = generator(windows, mask)
    assert tokens.shape == (1, 3, 12)
    assert torch.isfinite(tokens).all()


def test_shared_shape_tokenizer_accepts_variable_window_lengths_with_gradient():
    generator = ShapeTokenGenerator(3, shape_dim=12, resample_length=16)
    windows = [torch.randn(2, 4, length, 3, requires_grad=True) for length in (8, 16, 24)]
    tokens = [generator(value) for value in windows]
    assert all(value.shape == (2, 4, 12) for value in tokens)
    sum(value.sum() for value in tokens).backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in windows)


def test_shape_components_distinguish_dynamics_and_handle_constants():
    rising = torch.arange(16, dtype=torch.float32)
    windows = torch.stack((rising, rising.flip(0), torch.ones(16)), dim=0)
    windows = windows[None, :, :, None].requires_grad_()
    mask = torch.ones(1, 3, 16, dtype=torch.bool)
    generator = ShapeTokenGenerator(1, shape_dim=8)
    parts = generator.components(windows, mask)
    assert not torch.allclose(parts["difference"][:, 0], parts["difference"][:, 1])
    assert torch.allclose(
        parts["std"][:, 2],
        torch.full_like(parts["std"][:, 2], generator.eps ** .5),
    )
    output = generator(windows, mask)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert windows.grad is not None and torch.isfinite(windows.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in generator.parameters())


def test_resampling_makes_same_morphology_comparable_across_scales():
    generator = ShapeTokenGenerator(1, shape_dim=8, resample_length=16)
    windows = []
    for length in (8, 16, 24):
        curve = torch.linspace(-1, 1, length).view(1, 1, length, 1)
        windows.append(generator.components(curve)["normalized"])
    assert all(value.shape[2] == 16 for value in windows)
    assert torch.allclose(windows[0], windows[1], atol=.15, rtol=.05)
    assert torch.allclose(windows[1], windows[2], atol=.15, rtol=.05)


def test_shapelet_dictionary_normalizes_over_candidate_tokens_and_has_gradients():
    torch.manual_seed(3)
    dictionary = ShapeletDictionary(shape_dim=8, count=5, beta=10.)
    tokens = torch.randn(2, 7, 8, requires_grad=True)
    response = dictionary(tokens)
    normalized_tokens = torch.nn.functional.normalize(tokens, dim=-1)
    normalized_anchors = torch.nn.functional.normalize(dictionary.anchors, dim=-1)
    similarity = normalized_tokens @ normalized_anchors.T
    weights = torch.softmax(dictionary.beta * similarity, dim=1)
    assert response.shape == (2, 5)
    assert similarity.shape == weights.shape == (2, 7, 5)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 5))
    assert torch.allclose(response, (weights * similarity).sum(dim=1))
    response.sum().backward()
    assert torch.isfinite(tokens.grad).all()
    assert torch.isfinite(dictionary.anchors.grad).all()


def test_shapelet_candidate_masks_renormalize_without_changing_all_candidate_result():
    torch.manual_seed(31)
    dictionary = ShapeletDictionary(shape_dim=8, count=5, beta=5.)
    tokens = torch.randn(2, 12, 8)
    baseline = dictionary(tokens)
    all_candidates = dictionary.compute_response(
        tokens, candidate_mask=torch.ones(12, dtype=torch.bool), return_details=True,
    )
    torch.testing.assert_close(all_candidates["response"], baseline)
    torch.testing.assert_close(
        all_candidates["weights"].sum(dim=1), torch.ones(2, 5),
    )
    for mask in (
        torch.arange(12) < 4,
        ~((torch.arange(12) >= 4) & (torch.arange(12) < 8)),
        torch.arange(12) % 2 == 0,
    ):
        details = dictionary.compute_response(tokens, candidate_mask=mask, return_details=True)
        assert torch.isfinite(details["response"]).all()
        assert details["weights"][:, ~mask].eq(0).all()
        torch.testing.assert_close(details["weights"].sum(dim=1), torch.ones(2, 5))


def test_cached_structure_classification_is_identical_to_uncached_path():
    torch.manual_seed(37)
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=6, shape_window_scales=(8,),
        shape_window_stride=8, shapelet_count=3, fourier_num_modes=5,
    ).eval()
    prepared = torch.randn(2, 10, 8)
    positions = torch.arange(10).repeat(2, 1) * 20
    cached = model.prepare_structure(prepared, positions)
    expected = model.classify_prepared(prepared, positions, temporal_shift=3)
    actual = model.classify_prepared(
        prepared, positions, temporal_shift=3, prepared_structure=cached,
    )
    torch.testing.assert_close(actual, expected)


def test_ltae_baseline_is_unchanged_when_external_query_is_none():
    torch.manual_seed(3)
    model = LTAE(in_channels=8, n_head=2, d_k=4, d_model=8, n_neurons=[8, 6], dropout=0)
    model.eval()
    x = torch.randn(3, 5, 8)
    positions = torch.arange(5).repeat(3, 1)
    projected = model.inconv(x)
    encoded = projected + model.positional_enc(positions + model.max_temporal_shift)
    expected = model.dropout(model.mlp(model.attention_heads(encoded)[0]))
    actual = model(x, positions, external_query=None)
    assert torch.allclose(actual, expected)


def test_external_query_projection_starts_zero_and_receives_gradient():
    torch.manual_seed(4)
    model = LTAE(
        in_channels=8, n_head=2, d_k=4, d_model=8,
        n_neurons=[8, 6], dropout=0, external_query_dim=10,
    ).eval()
    x = torch.randn(2, 5, 8)
    positions = torch.arange(5).repeat(2, 1)
    query = torch.randn(2, 10, requires_grad=True)
    baseline, baseline_attention = model(x, positions, external_query=None, return_att=True)
    output, attention = model(x, positions, external_query=query, return_att=True)
    projection = model.attention_heads.external_query_projection
    assert projection.bias is None
    assert torch.count_nonzero(projection.weight) == 0
    assert torch.allclose(output, baseline)
    assert torch.allclose(attention, baseline_attention)
    output.sum().backward()
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()
    assert projection.weight.grad.abs().sum() > 0
    assert torch.isfinite(attention).all()


def test_kmeans_shapelet_initialization_tracks_clusters_and_copies_parameter():
    torch.manual_seed(11)
    dictionary = ShapeletDictionary(shape_dim=3, count=3, beta=5.)
    parameter_identity = id(dictionary.anchors)
    before = dictionary.anchors.detach().clone()
    clusters = torch.cat((
        torch.tensor([1., 0., 0.]).repeat(30, 1) + .01 * torch.randn(30, 3),
        torch.tensor([0., 1., 0.]).repeat(30, 1) + .01 * torch.randn(30, 3),
        torch.tensor([0., 0., 1.]).repeat(30, 1) + .01 * torch.randn(30, 3),
    ))
    diagnostics = initialize_shapelet_dictionary_from_tokens(
        dictionary, clusters, seed=7,
    )
    anchors = torch.nn.functional.normalize(dictionary.anchors.detach(), dim=-1)
    assert torch.all((anchors @ torch.eye(3).T).max(0).values > .99)
    assert id(dictionary.anchors) == parameter_identity
    assert not torch.equal(before, dictionary.anchors)
    assert torch.isfinite(dictionary.anchors).all()
    assert torch.allclose(dictionary.anchors.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert diagnostics["tokens"] == 90
    assert diagnostics["anchors"] == 3


def test_zero_external_query_is_exactly_the_master_query_path():
    torch.manual_seed(5)
    model = LTAE(
        in_channels=8, n_head=2, d_k=4, d_model=8,
        n_neurons=[8, 6], dropout=0, external_query_dim=10,
    ).eval()
    model.attention_heads.dropout.p = 0
    x = torch.randn(3, 5, 8)
    positions = torch.arange(5).repeat(3, 1)
    baseline, baseline_attention = model(x, positions, return_att=True)
    zero, zero_attention = model(
        x, positions, return_att=True, external_query=torch.zeros(3, 10),
    )
    assert torch.equal(zero, baseline)
    assert torch.equal(zero_attention, baseline_attention)


def test_structure_branch_returns_single_forward_intermediates():
    branch = DiscriminativeStructureBranch(6, shape_dim=16)
    features = torch.randn(2, 20, 6, requires_grad=True)
    positions = torch.arange(20).repeat(2, 1) * 10
    result = branch(features, positions)
    assert set(result) >= {"shape_tokens", "shapelet_response", "shape_class_token"}
    assert "shape_" + "attention" not in result
    result["shape_class_token"].sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_v2_q24_branch_returns_rich_response_and_reuses_strength():
    torch.manual_seed(41)
    branch = DiscriminativeStructureBranch(
        6, shape_dim=16, window_scales=(24,), window_stride=8,
        shapelet_count=5,
    )
    features = torch.randn(2, 20, 6)
    positions = torch.arange(20).repeat(2, 1) * 10
    result = branch(features, positions)
    assert result["shape_tokens"].shape == (2, 8, 16)
    assert result["shapelet_strength"].shape == (2, 5)
    assert result["shapelet_concentration"].shape == (2, 5)
    assert result["shapelet_response"].shape == (2, 10)
    assert result["shape_stats_feature"].shape == (2, 64)
    assert torch.isfinite(result["shapelet_concentration"]).all()
    assert torch.all((result["shapelet_concentration"] >= 0)
                     & (result["shapelet_concentration"] <= 1))
    expected = branch.shapelet_dictionary.compute_response(result["shape_tokens"])
    torch.testing.assert_close(result["shapelet_strength"], expected)


def test_normalized_candidate_concentration_distinguishes_uniform_and_peaked_weights():
    from models.structure_da.discriminative_structure import normalized_candidate_concentration

    uniform = torch.full((2, 8, 3), 1 / 8)
    peaked = torch.zeros(2, 8, 3)
    peaked[:, 0] = 1
    torch.testing.assert_close(
        normalized_candidate_concentration(uniform), torch.zeros(2, 3),
        atol=1e-6, rtol=0,
    )
    assert torch.all(normalized_candidate_concentration(peaked) > .999)


def test_full_model_returns_all_training_intermediates_without_second_forward():
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=10, shape_window_scales=(8, 16),
        shape_window_stride=8, fourier_num_modes=5,
    )
    pixels = torch.randn(2, 6, 3, 5)
    mask = torch.ones(2, 6, 5)
    positions = torch.arange(6).repeat(2, 1) * 30
    extra = torch.empty(2, 6, 0)
    result = model(pixels, mask, positions, extra, return_dict=True)
    assert result["logits"].shape == (2, 3)
    assert result["shape_logits"].shape == (2, 3)
    assert result["instance_feature"].shape == (2, 6)
    assert result["shape_tokens"].shape[-1] == 10
    assert result["shapelet_response"].shape == (2, 32)
    assert "shape_" + "attention" not in result
    result["logits"].sum().backward()
    assert model.structure_branch.token_generator.raw_encoder.input_projection.weight.grad is not None


def test_structure_branch_is_decoupled_from_timematch_global_shift():
    torch.manual_seed(9)
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=10, shape_window_scales=(8, 16),
        shape_window_stride=8, fourier_num_modes=5,
    ).eval()
    pixels = torch.randn(2, 7, 3, 5)
    mask = torch.ones(2, 7, 5)
    positions = torch.arange(7).repeat(2, 1) * 30
    extra = torch.empty(2, 7, 0)
    output0 = model.forward_with_temporal_shift(
        pixels, mask, positions, extra, temporal_shift=0, return_dict=True,
    )
    output1 = model.forward_with_temporal_shift(
        pixels, mask, positions, extra, temporal_shift=10, return_dict=True,
    )
    for key in ("shape_tokens", "shapelet_response", "shape_class_token"):
        assert torch.equal(output0[key], output1[key])
    assert not torch.allclose(output0["instance_feature"], output1["instance_feature"])
