import torch
import torch.nn as nn
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


def test_gather_uses_unshifted_membership_and_keeps_shifted_encoder_time():
    features = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    split_positions = torch.tensor([[10, 100, 190, 280]])
    encoder_positions = split_positions + 30

    gathered = gather_period_patches(
        features,
        split_positions,
        encoder_positions,
        patches=4,
        period=365,
    )

    assert gathered.patch_ids.tolist() == [[0, 1, 2, 3]]
    assert gathered.valid.sum(dim=-1).tolist() == [[1, 1, 1, 1]]
    assert gathered.features[gathered.valid].flatten().tolist() == [0, 1, 2, 3]
    assert gathered.encoder_positions[gathered.valid].tolist() == [40, 130, 220, 310]


def test_shift_does_not_change_patch_membership():
    positions = torch.tensor([[80, 100, 170, 190, 260, 280, 350]])
    features = torch.randn(1, positions.shape[1], 3)

    base = gather_period_patches(features, positions, positions, 4, 365)
    shifted = gather_period_patches(features, positions, positions + 60, 4, 365)

    assert torch.equal(base.patch_ids, shifted.patch_ids)
    assert torch.equal(base.valid, shifted.valid)


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


def test_mtan_returns_official_sampled_z0_shape_and_uses_encoder_positions():
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

    output = encoder(features, split_positions, split_positions + 15)

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

    output = encoder(features, positions, positions)

    assert output.lowest_valid.shape == (1, 4, 8)
    assert output.lowest_valid[0, 0].all()
    assert not output.lowest_valid[0, 1].any()
    assert output.lowest_valid[0, 2].all()
    assert not output.lowest_valid[0, 3].any()


class _MeanTemporalDecoder(nn.Module):
    def forward(self, values, positions):
        return values.mean(dim=1)


class _NonlinearClassifier(nn.Module):
    def forward(self, features):
        return torch.stack([features[:, 0].square(), features[:, 1]], dim=1)


def test_decoder_shares_modules_and_averages_patch_logits_not_features():
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
    model.temporal_decoder = _MeanTemporalDecoder()
    model.classifier = _NonlinearClassifier()
    lowest = torch.zeros(1, 4, 8, 4)
    lowest[0, 0, :, :2] = torch.tensor([0.0, 1.0])
    lowest[0, 1, :, :2] = torch.tensor([2.0, 3.0])
    lowest[0, 2, :, :2] = torch.tensor([4.0, 5.0])
    lowest[0, 3, :, :2] = torch.tensor([6.0, 7.0])
    valid = torch.ones(1, 4, 8, dtype=torch.bool)

    logits, patch_features, patch_logits = model._decode_lowest(lowest, valid)

    assert patch_features.shape == (1, 4, 4)
    assert patch_logits.shape == (1, 4, 2)
    assert torch.allclose(logits, torch.tensor([[14.0, 4.0]]))
    assert not torch.allclose(logits[:, 0], patch_features[:, :, 0].mean(1).square())
    assert sum(1 for module in model.modules() if module is model.temporal_decoder) == 1
    assert sum(1 for module in model.modules() if module is model.classifier) == 1


def test_patch_loss_is_mean_of_valid_patch_individual_losses():
    patch_logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 3.0], [2.0, 1.0], [9.0, -9.0]],
            [[0.0, 2.0], [1.0, 2.0], [-5.0, 5.0], [4.0, 0.0]],
        ]
    )
    patch_valid = torch.tensor(
        [[True, True, True, False], [True, True, False, True]]
    )
    targets = torch.tensor([0, 1])
    output = ReIMTSClassificationOutput(
        sample_logits=patch_logits.mean(dim=1),
        patch_logits=patch_logits,
        patch_valid=patch_valid,
    )
    criterion = nn.CrossEntropyLoss()

    actual = reimts_classification_loss(output, targets, criterion, mode="patch")
    repeated_targets = targets.unsqueeze(1).expand(-1, 4)
    expected = criterion(patch_logits[patch_valid], repeated_targets[patch_valid])

    assert torch.allclose(actual, expected)


def test_sample_loss_matches_round_one_mean_logits_loss():
    patch_logits = torch.randn(2, 4, 3)
    targets = torch.tensor([1, 2])
    output = ReIMTSClassificationOutput(
        sample_logits=patch_logits.mean(dim=1),
        patch_logits=patch_logits,
        patch_valid=torch.ones(2, 4, dtype=torch.bool),
    )
    criterion = nn.CrossEntropyLoss()

    actual = reimts_classification_loss(output, targets, criterion, mode="sample")
    expected = criterion(patch_logits.mean(dim=1), targets)

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
    clone.load_state_dict(model.state_dict(), strict=True)

    assert logits.shape == (2, 3)


def test_forward_for_loss_exposes_patch_logits_and_patch_nonempty_mask():
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
    assert output.patch_logits.shape == (2, 4, 3)
    assert output.patch_valid.shape == (2, 4)
    assert output.patch_valid.tolist() == [
        [True, False, True, False],
        [True, False, True, False],
    ]
    assert torch.allclose(ordinary, output.sample_logits)


def test_shift_capability_eval_is_deterministic_and_changes_encoder_coordinates():
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
    spatial = torch.randn(2, 6, 128)
    positions = torch.tensor([[1, 60, 120, 190, 260, 340]]).repeat(2, 1)
    seen_encoder_positions = []

    hook = model.reimts_encoder.scale_encoders[0].register_forward_pre_hook(
        lambda module, args: seen_encoder_positions.append(args[1].clone())
    )
    try:
        first = model.forward_from_spatial_with_shift(spatial, positions, shift=15)
        second = model.forward_from_spatial_with_shift(spatial, positions, shift=15)
        model.forward_from_spatial_with_shift(spatial, positions, shift=30)
    finally:
        hook.remove()

    assert torch.allclose(first, second)
    assert torch.equal(seen_encoder_positions[0], seen_encoder_positions[1])
    assert not torch.equal(seen_encoder_positions[0], seen_encoder_positions[2])
