import io

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from unittest.mock import patch

from models.reimts_classifier import (
    PseReIMTSMTANLTAE,
    ReIMTSClassificationOutput,
    reimts_classification_loss,
)
from models.reimts.mtan_encoder import MTANEncoder, ReIMTSMTANEncoders
from models.reimts.recursive_temporal import (
    IARFFusion,
    RecursiveTemporalEncoder,
    gather_period_patches,
    period_patch_ids,
    split_temporal_representation,
)


def test_period_patch_ids_use_timestamps_not_observation_counts():
    positions = torch.tensor([[1, 2, 200, 201]])

    patch_ids = period_patch_ids(positions, patches=2, period=365)

    assert patch_ids.tolist() == [[0, 0, 1, 1]]


def test_three_levels_assign_every_observation_to_one_of_1_2_4_patches():
    positions = torch.tensor([[0, 90, 91, 181, 182, 273, 364]])

    memberships = []
    for patches in (1, 2, 4):
        patch_ids = period_patch_ids(positions, patches=patches, period=365)
        one_hot = torch.nn.functional.one_hot(patch_ids, num_classes=patches)
        assert one_hot.sum(dim=-1).tolist() == [[1] * positions.shape[1]]
        memberships.append(torch.unique(patch_ids).numel())

    assert memberships == [1, 2, 4]


def test_gather_preserves_irregular_observation_timestamps_exactly():
    observation_positions = torch.tensor([[3, 17, 41, 96, 103, 188, 249, 360]])
    features = torch.arange(8, dtype=torch.float32).view(1, 8, 1)

    gathered = gather_period_patches(
        features,
        observation_positions,
        patches=4,
        period=365,
    )

    assert gathered.observation_valid.sum(dim=-1).tolist() == [[3, 2, 2, 1]]
    assert gathered.observation_positions[gathered.observation_valid].tolist() == [
        3, 17, 41, 96, 103, 188, 249, 360
    ]
    assert set(gathered.observation_positions[gathered.observation_valid].tolist()) <= set(
        observation_positions.flatten().tolist()
    )


def test_shift_does_not_change_patch_membership():
    positions = torch.tensor([[80, 100, 170, 190, 260, 280, 350]])
    features = torch.randn(1, positions.shape[1], 3)

    negative_candidate = gather_period_patches(features, positions, 4, 365)
    positive_candidate = gather_period_patches(features, positions, 4, 365)

    assert torch.equal(negative_candidate.patch_ids, positive_candidate.patch_ids)
    assert torch.equal(
        negative_candidate.observation_valid,
        positive_candidate.observation_valid,
    )
    assert torch.equal(
        negative_candidate.observation_positions,
        positive_candidate.observation_positions,
    )


def test_reimts_uses_independent_mtan_instances_with_official_reference_count():
    encoders = ReIMTSMTANEncoders(
        levels=3,
        input_dim=4,
        latent_dim=8,
        num_ref_points=8,
        num_heads=1,
    )

    assert len(encoders.scale_encoders) == 3
    assert len({id(module) for module in encoders.scale_encoders}) == 3
    assert encoders.reference_points == (8, 8, 8)


def test_mtan_returns_official_sampled_z0_shape_and_uses_observation_positions():
    encoder = MTANEncoder(
        input_dim=3,
        latent_dim=6,
        num_ref_points=8,
        num_heads=1,
    )
    values = torch.randn(2, 5, 3)
    positions = torch.tensor(
        [[10, 50, 100, 180, 300], [20, 60, 120, 220, 340]],
        dtype=torch.float32,
    )
    valid = torch.ones(2, 5, dtype=torch.bool)

    torch.manual_seed(7)
    base = encoder(values, positions, valid)
    torch.manual_seed(7)
    shifted = encoder(values, positions + 30, valid)

    assert base.shape == (2, 8, 6)
    assert not torch.allclose(base, shifted)


def test_mtan_eval_is_deterministic_without_resetting_random_seed():
    encoder = MTANEncoder(
        input_dim=3,
        latent_dim=6,
        num_ref_points=8,
        num_heads=1,
    ).eval()
    values = torch.randn(2, 5, 3)
    positions = torch.tensor(
        [[10, 50, 100, 180, 300], [20, 60, 120, 220, 340]],
        dtype=torch.float32,
    )
    valid = torch.ones(2, 5, dtype=torch.bool)

    first = encoder(values, positions, valid)
    second = encoder(values, positions, valid)

    assert torch.allclose(first, second)


def test_mtan_train_uses_reparameterization_noise():
    encoder = MTANEncoder(
        input_dim=2,
        latent_dim=4,
        num_ref_points=8,
        num_heads=1,
    ).train()
    values = torch.randn(1, 3, 2)
    positions = torch.tensor([[10, 100, 300]], dtype=torch.float32)
    valid = torch.ones(1, 3, dtype=torch.bool)

    with patch("torch.randn_like", side_effect=lambda value: torch.zeros_like(value)):
        zero_noise = encoder(values, positions, valid)
    with patch("torch.randn_like", side_effect=lambda value: torch.ones_like(value)):
        unit_noise = encoder(values, positions, valid)

    assert not torch.allclose(zero_noise, unit_noise)


def test_mtan_mask_prevents_padded_values_from_affecting_representation():
    encoder = MTANEncoder(
        input_dim=2,
        latent_dim=4,
        num_ref_points=8,
        num_heads=1,
    ).eval()
    positions = torch.tensor([[10, 100, 0]], dtype=torch.float32)
    valid = torch.tensor([[True, True, False]])
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, -99.0]]])
    changed_padding = values.clone()
    changed_padding[:, -1] = torch.tensor([-500.0, 500.0])

    first = encoder(values, positions, valid)
    second = encoder(changed_padding, positions, valid)

    assert torch.allclose(first, second)


def test_split_temporal_representation_halves_each_parent_reference_axis():
    parent = torch.arange(1 * 2 * 8 * 3, dtype=torch.float32).view(1, 2, 8, 3)
    valid = torch.ones(1, 2, 8, dtype=torch.bool)

    children, child_valid = split_temporal_representation(parent, valid, factor=2)

    assert children.shape == (1, 4, 4, 3)
    assert child_valid.shape == (1, 4, 4)
    assert torch.equal(children[0, 0], parent[0, 0, :4])
    assert torch.equal(children[0, 1], parent[0, 0, 4:])


def test_iarf_aligns_h_to_e_and_masks_padded_parent_values():
    fusion = IARFFusion(latent_dim=4, parent_ref_points=4, ref_points=8)
    e = torch.randn(2, 8, 4)
    h = torch.randn(2, 4, 4)
    valid = torch.tensor([[True, True, False, False], [False] * 4])
    changed_padding = h.clone()
    changed_padding[0, 2:] = 999.0
    changed_padding[1] = -999.0

    alpha, first_g, first_h = fusion(e, h, valid)
    second_alpha, second_g, second_h = fusion(e, changed_padding, valid)

    assert alpha.shape == first_g.shape == first_h.shape == e.shape
    assert torch.allclose(alpha, second_alpha)
    assert torch.allclose(first_g, second_g)
    assert torch.allclose(first_h, second_h)
    assert torch.equal(first_g[1], e[1])


def test_iarf_uses_subtractive_global_to_local_correction_algebra():
    fusion = IARFFusion(latent_dim=1, parent_ref_points=1, ref_points=1)
    with torch.no_grad():
        fusion.temporal_mapping.fill_(2.0)
        fusion.feed_forward.weight.fill_(3.0)
        fusion.feed_forward.bias.zero_()
    e = torch.tensor([[[5.0]]])
    parent_h = torch.tensor([[[2.0]]])
    parent_valid = torch.tensor([[True]])

    alpha, actual, aligned_h = fusion(e, parent_h, parent_valid)
    expected_h = torch.tensor([[[4.0]]])
    expected_alpha = torch.tensor([[[12.0]]])
    expected = e - expected_alpha * expected_h

    assert torch.equal(aligned_h, expected_h)
    assert torch.equal(alpha, expected_alpha)
    assert torch.equal(actual, expected)


def test_recursive_encoder_produces_matching_e_h_g_shapes_at_1_2_4_scales():
    encoder = RecursiveTemporalEncoder(
        input_dim=4,
        latent_dim=8,
        levels=3,
        scale_factor=2,
        period=365,
        num_ref_points=8,
        num_heads=1,
    )
    features = torch.randn(2, 7, 4)
    split_positions = torch.tensor(
        [[1, 40, 100, 190, 230, 300, 360], [5, 80, 170, 200, 260, 320, 364]]
    )

    output = encoder(features, split_positions)

    assert [scale.e.shape[:3] for scale in output.scales] == [
        (2, 1, 8),
        (2, 2, 8),
        (2, 4, 8),
    ]
    assert output.scales[0].h is None
    assert output.scales[0].g.shape == output.scales[0].e.shape
    for scale in output.scales[1:]:
        assert scale.e.shape == scale.h.shape == scale.g.shape
        assert scale.alpha.shape == scale.e.shape
    assert output.lowest.shape == (2, 4, 8, 8)
    assert output.lowest_reference_positions.shape == (2, 4, 8)
    assert torch.equal(
        output.scales[-1].observation_positions[
            output.scales[-1].observation_valid
        ],
        torch.tensor([1, 40, 100, 190, 230, 300, 360, 5, 80, 170, 200, 260, 320, 364]),
    )


def test_reference_valid_means_only_patch_nonempty():
    encoder = RecursiveTemporalEncoder(
        input_dim=4,
        latent_dim=8,
        levels=3,
        scale_factor=2,
        period=365,
        num_ref_points=8,
        num_heads=1,
    ).eval()
    features = torch.randn(1, 2, 4)
    positions = torch.tensor([[10, 200]])

    output = encoder(features, positions)

    assert output.lowest_valid.shape == (1, 4, 8)
    assert output.lowest_valid[0, 0].all()
    assert not output.lowest_valid[0, 1].any()
    assert output.lowest_valid[0, 2].all()
    assert not output.lowest_valid[0, 3].any()


class _MeanTemporalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.seen_values = None
        self.seen_positions = None

    def forward(self, values, positions):
        self.calls += 1
        self.seen_values = values.clone()
        self.seen_positions = positions.clone()
        return values.mean(dim=1)


class _NonlinearClassifier(nn.Module):
    def forward(self, features):
        return torch.stack([features[:, 0].square(), features[:, 1]], dim=1)


def test_four_patch_blocks_flatten_chronologically_and_decode_once():
    model = PseReIMTSMTANLTAE(
        input_dim=2,
        num_classes=2,
        with_extra=False,
        latent_dim=4,
        num_ref_points=8,
        mtan_heads=1,
        ltae_heads=1,
        ltae_key_dim=2,
        ltae_model_dim=4,
        ltae_mlp=[4, 4],
        classifier_mlp=[4],
    )
    decoder = _MeanTemporalDecoder()
    model.temporal_decoder = decoder
    model.classifier = _NonlinearClassifier()
    lowest = torch.zeros(1, 4, 8, 4)
    for patch_index in range(4):
        lowest[0, patch_index, :, 0] = patch_index + 1
    valid = torch.ones(1, 4, 8, dtype=torch.bool)
    reference_positions = torch.arange(32).reshape(1, 4, 8)
    features = model._flatten_lowest(lowest, reference_positions, valid)

    logits, sample_features = model._decode_whole(features, shift=0)

    assert features.tokens.shape == (1, 32, 4)
    assert features.positions.shape == (1, 32)
    assert features.tokens[0, :, 0].tolist() == [1] * 8 + [2] * 8 + [3] * 8 + [4] * 8
    assert features.positions.tolist() == [list(range(32))]
    assert decoder.calls == 1
    assert decoder.seen_values.shape == (1, 32, 4)
    assert sample_features.shape == (1, 4)
    assert logits.shape == (1, 2)


def test_patch_loss_mode_fails_fast_for_whole_sample_architecture():
    output = ReIMTSClassificationOutput(
        sample_logits=torch.randn(2, 3),
        patch_valid=torch.ones(2, 4, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="sample-level"):
        reimts_classification_loss(
            output, torch.tensor([0, 1]), nn.CrossEntropyLoss(), mode="patch"
        )


def test_sample_loss_consumes_one_logit_row_per_target_without_repeat():
    sample_logits = torch.randn(2, 3)
    targets = torch.tensor([1, 2])
    output = ReIMTSClassificationOutput(
        sample_logits=sample_logits,
        patch_valid=torch.ones(2, 4, dtype=torch.bool),
    )
    criterion = nn.CrossEntropyLoss()

    actual = reimts_classification_loss(output, targets, criterion, mode="sample")
    expected = criterion(sample_logits, targets)

    assert torch.allclose(actual, expected)


def test_full_model_returns_sample_logits_and_strict_checkpoint_round_trips():
    kwargs = dict(
        input_dim=2,
        num_classes=3,
        with_extra=False,
        latent_dim=8,
        num_ref_points=8,
        mtan_heads=1,
        ltae_heads=1,
        ltae_key_dim=2,
        ltae_model_dim=8,
        ltae_mlp=[8, 8],
        classifier_mlp=[8],
        dropout=0.0,
    )
    model = PseReIMTSMTANLTAE(**kwargs).eval()
    clone = PseReIMTSMTANLTAE(**kwargs)
    pixels = torch.randn(2, 6, 2, 5)
    valid_pixels = torch.ones(2, 6, 5)
    positions = torch.tensor([[1, 60, 120, 190, 260, 340]]).repeat(2, 1)

    with torch.no_grad():
        logits = model(pixels, valid_pixels, positions, extra=None)
    checkpoint = io.BytesIO()
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=False)
    clone.load_state_dict(restored["state_dict"], strict=True)
    clone.eval()
    with torch.no_grad():
        cloned_logits = clone(pixels, valid_pixels, positions, extra=None)

    assert logits.shape == (2, 3)
    assert torch.allclose(logits, cloned_logits)


def test_ordinary_forward_calls_whole_sample_ltae_and_classifier_once_and_returns_final_embedding():
    model = PseReIMTSMTANLTAE(
        input_dim=2, num_classes=3, with_extra=False, latent_dim=8,
        num_ref_points=8, mtan_heads=1, ltae_heads=1, ltae_key_dim=2,
        ltae_model_dim=8, ltae_mlp=[8, 8], classifier_mlp=[8], dropout=0.0,
    ).eval()
    decoder = _MeanTemporalDecoder()
    model.temporal_decoder = decoder
    classifier = nn.Linear(8, 3)
    model.classifier = classifier
    classifier_calls = []
    hook = classifier.register_forward_hook(
        lambda module, args, output: classifier_calls.append(output.shape)
    )
    pixels = torch.randn(2, 8, 2, 5)
    valid_pixels = torch.ones(2, 8, 5)
    positions = torch.tensor([[3, 17, 41, 96, 103, 188, 249, 360]]).repeat(2, 1)

    try:
        logits, sample_feature = model(
            pixels, valid_pixels, positions, None, return_feats=True
        )
    finally:
        hook.remove()

    assert decoder.calls == 1
    assert classifier_calls == [torch.Size([2, 3])]
    assert decoder.seen_values.shape == (2, 32, 8)
    assert sample_feature.shape == (2, 8)
    assert logits.shape == (2, 3)


def test_forward_for_loss_exposes_sample_logits_and_patch_occupancy_only():
    model = PseReIMTSMTANLTAE(
        input_dim=2,
        num_classes=3,
        with_extra=False,
        latent_dim=8,
        num_ref_points=8,
        mtan_heads=1,
        ltae_heads=1,
        ltae_key_dim=2,
        ltae_model_dim=8,
        ltae_mlp=[8, 8],
        classifier_mlp=[8],
        dropout=0.0,
    ).eval()
    pixels = torch.randn(2, 2, 2, 5)
    valid_pixels = torch.ones(2, 2, 5)
    positions = torch.tensor([[10, 200]]).repeat(2, 1)

    output = model.forward_for_loss(pixels, valid_pixels, positions, None)
    ordinary = model(pixels, valid_pixels, positions, None)

    assert output.sample_logits.shape == (2, 3)
    assert output.patch_valid.shape == (2, 4)
    assert output.patch_valid.tolist() == [
        [True, False, True, False],
        [True, False, True, False],
    ]
    assert torch.allclose(ordinary, output.sample_logits)


def test_shift_changes_only_whole_ltae_positions_and_cached_tokens_are_identical():
    model = PseReIMTSMTANLTAE(
        input_dim=2,
        num_classes=3,
        with_extra=False,
        latent_dim=8,
        num_ref_points=8,
        mtan_heads=1,
        ltae_heads=1,
        ltae_key_dim=2,
        ltae_model_dim=8,
        ltae_mlp=[8, 8],
        classifier_mlp=[8],
        dropout=0.0,
    ).eval()
    decoder = _MeanTemporalDecoder()
    model.temporal_decoder = decoder
    model.classifier = nn.Linear(8, 3)
    spatial = torch.randn(2, 8, 128)
    positions = torch.tensor([[3, 17, 41, 96, 103, 188, 249, 360]]).repeat(2, 1)

    features = model.prepare_shift_features_from_spatial(spatial, positions)
    original_tokens = features.tokens.clone()
    base_positions = features.positions.clone()
    model.forward_from_shift_features(features, shift=0)
    seen_zero = decoder.seen_positions.clone()
    model.forward_from_shift_features(features, shift=20)
    seen_twenty = decoder.seen_positions.clone()

    assert torch.equal(features.tokens, original_tokens)
    assert torch.equal(features.positions, base_positions)
    assert torch.equal(seen_zero, base_positions)
    assert torch.equal(seen_twenty, base_positions + 20)


def test_absolute_reference_positions_are_four_calendar_blocks_not_repeated_local_indices():
    encoder = RecursiveTemporalEncoder(
        input_dim=4, latent_dim=8, levels=3, scale_factor=2,
        period=365, num_ref_points=8, num_heads=1,
    ).eval()
    features = torch.randn(1, 8, 4)
    positions = torch.tensor([[3, 17, 41, 96, 103, 188, 249, 360]])

    output = encoder(features, positions)
    flattened = output.lowest_reference_positions.flatten().tolist()

    assert output.lowest_reference_positions.shape == (1, 4, 8)
    assert flattened == [
        0, 13, 26, 39, 52, 65, 78, 91,
        91, 104, 117, 130, 143, 156, 169, 182,
        182, 196, 209, 222, 235, 248, 261, 274,
        274, 287, 300, 313, 326, 339, 352, 364,
    ]


def test_global_shift_tensor_must_be_constant_within_each_sample():
    model = PseReIMTSMTANLTAE(
        input_dim=2, num_classes=3, with_extra=False, latent_dim=8,
        num_ref_points=8, mtan_heads=1, ltae_heads=1, ltae_key_dim=2,
        ltae_model_dim=8, ltae_mlp=[8, 8], classifier_mlp=[8], dropout=0.0,
    ).eval()
    features = SimpleNamespace(
        tokens=torch.randn(1, 32, 8),
        positions=torch.arange(32).unsqueeze(0),
        patch_valid=torch.ones(1, 4, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="whole-sample"):
        model.forward_from_shift_features(
            features, torch.tensor([[0, 0, 1, 1]])
        )


@pytest.mark.parametrize(
    ("shift", "expected_index_range"),
    [(-60, (40, 404)), (0, (100, 464)), (60, (160, 524))],
)
def test_real_ltae_embedding_accepts_full_year_reference_boundaries(
    shift, expected_index_range
):
    model = PseReIMTSMTANLTAE(
        input_dim=2,
        num_classes=3,
        with_extra=False,
        latent_dim=8,
        num_ref_points=8,
        mtan_heads=1,
        ltae_heads=1,
        ltae_key_dim=2,
        ltae_model_dim=8,
        ltae_mlp=[8, 8],
        classifier_mlp=[8],
        dropout=0.0,
        max_temporal_shift=100,
    ).eval()
    pixels = torch.randn(2, 8, 2, 5)
    valid_pixels = torch.ones(2, 8, 5)
    observation_positions = torch.tensor(
        [[3, 17, 41, 96, 103, 188, 249, 360]]
    ).repeat(2, 1)
    embedding_indices = []
    hook = model.temporal_decoder.positional_enc.register_forward_pre_hook(
        lambda module, args: embedding_indices.append(args[0].detach().clone())
    )

    try:
        with torch.no_grad():
            logits = model.forward_with_shift(
                pixels,
                valid_pixels,
                observation_positions,
                extra=None,
                shift=shift,
            )
    finally:
        hook.remove()

    assert logits.shape == (2, 3)
    assert len(embedding_indices) == 1
    actual_indices = embedding_indices[0]
    assert actual_indices.shape == (2, 32)
    assert (int(actual_indices.min()), int(actual_indices.max())) == (
        expected_index_range
    )
    assert 0 <= int(actual_indices.min())
    assert int(actual_indices.max()) < model.temporal_decoder.positional_enc.num_embeddings
