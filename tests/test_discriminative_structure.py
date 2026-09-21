import torch

from models.ltae import LTAE
from models.structure_da.discriminative_structure import (
    DiscriminativeStructureBranch,
    FourierStructureExposer,
    MultiScaleWindowExtractor,
    ShapeAttentionPool,
    ShapeTokenGenerator,
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


def test_multiscale_windows_are_circular_without_padding():
    curve = torch.arange(64, dtype=torch.float32)[None, :, None]
    groups, scales = MultiScaleWindowExtractor((16, 32), 8)(curve)
    assert [group.shape for group in groups] == [(1, 8, 16, 1), (1, 8, 32, 1)]
    assert scales.tolist() == [16] * 8 + [32] * 8
    expected_wrap = torch.cat((curve[:, 56:64], curve[:, 0:8]), dim=1)
    assert torch.equal(groups[0][:, -1], expected_wrap)


def test_shape_components_preserve_level_and_amplitude_information():
    t = torch.linspace(-1, 1, 16)
    base = torch.stack((t, t.square()), dim=-1)[None, None]
    windows = torch.cat((base, base + 4, base * 3), dim=1)
    mask = torch.ones(1, 3, 16, dtype=torch.bool)
    generator = ShapeTokenGenerator(2, shape_dim=12)
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
    generator = ShapeTokenGenerator(3, shape_dim=12)
    windows16 = torch.randn(2, 4, 16, 3, requires_grad=True)
    windows32 = torch.randn(2, 4, 32, 3, requires_grad=True)
    tokens16 = generator(windows16)
    tokens32 = generator(windows32)
    assert tokens16.shape == tokens32.shape == (2, 4, 12)
    (tokens16.sum() + tokens32.sum()).backward()
    assert windows16.grad is not None and windows16.grad.abs().sum() > 0
    assert windows32.grad is not None and windows32.grad.abs().sum() > 0


def test_shape_components_distinguish_dynamics_and_handle_constants():
    rising = torch.arange(16, dtype=torch.float32)
    windows = torch.stack((rising, rising.flip(0), torch.ones(16)), dim=0)
    windows = windows[None, :, :, None]
    mask = torch.ones(1, 3, 16, dtype=torch.bool)
    generator = ShapeTokenGenerator(1, shape_dim=8)
    parts = generator.components(windows, mask)
    assert not torch.allclose(parts["difference"][:, 0], parts["difference"][:, 1])
    assert torch.allclose(parts["std"][:, 2], torch.zeros_like(parts["std"][:, 2]))
    assert torch.isfinite(generator(windows, mask)).all()


def test_attention_pool_masks_tokens_and_sums_to_one():
    tokens = torch.randn(2, 4, 8)
    valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    pooled, attention, returned = ShapeAttentionPool(8)(tokens, valid)
    assert pooled.shape == (2, 8)
    assert torch.equal(valid, returned)
    assert torch.allclose(attention.sum(-1), torch.ones(2))
    assert attention[~valid].eq(0).all()
    assert isinstance(ShapeAttentionPool(8).score, torch.nn.Sequential)
    assert sum(isinstance(module, torch.nn.Tanh) for module in ShapeAttentionPool(8).score) == 1


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


def test_external_query_changes_output_and_receives_gradient():
    torch.manual_seed(4)
    model = LTAE(
        in_channels=8, n_head=2, d_k=4, d_model=8,
        n_neurons=[8, 6], dropout=0, external_query_dim=10,
    )
    x = torch.randn(2, 5, 8)
    positions = torch.arange(5).repeat(2, 1)
    query = torch.randn(2, 10, requires_grad=True)
    output = model(x, positions, external_query=query)
    other = model(x, positions, external_query=query + 1)
    assert not torch.allclose(output, other)
    output.sum().backward()
    assert query.grad is not None and query.grad.abs().sum() > 0


def test_structure_branch_returns_single_forward_intermediates():
    branch = DiscriminativeStructureBranch(6, shape_dim=16, window_scales=(16, 32), window_stride=8)
    features = torch.randn(2, 20, 6, requires_grad=True)
    positions = torch.arange(20).repeat(2, 1) * 10
    result = branch(features, positions)
    assert set(result) >= {"shape_tokens", "shape_attention", "shape_mask", "shape_class_token"}
    result["shape_class_token"].sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


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
    assert result["instance_feature"].shape == (2, 6)
    assert result["shape_tokens"].shape[-1] == 10
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
    for key in ("shape_tokens", "shape_attention", "shape_class_token"):
        assert torch.equal(output0[key], output1[key])
    assert not torch.allclose(output0["instance_feature"], output1["instance_feature"])
