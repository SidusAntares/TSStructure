from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from analysis.state_org_feasibility_audit import (
    _loader,
    organization_counterfactuals,
    query_counterfactuals,
    rule_labels,
)
from models.stclassifier import PseStructureProtoLTae
from timematch import configure_structure_specific_training


def _model():
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=128, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, shape_resample_length=24,
        fourier_num_modes=5, shape_representation="state_org",
        shape_injection="direct_response_query", structure_shift_mode="none",
        dropout=0.,
    )


def _batch(batch=3):
    pixels = torch.randn(batch, 10, 3, 4)
    mask = torch.ones(batch, 10, 4)
    positions = torch.arange(10).repeat(batch, 1) * 30
    return pixels, mask, positions, torch.zeros(batch, 4)


class _VariablePixelDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.pixel_counts = (3, 5)

    def __len__(self):
        return len(self.pixel_counts)

    def __getitem__(self, index):
        pixels = self.pixel_counts[index]
        return {
            "pixels": torch.ones(4, 2, pixels),
            "valid_pixels": torch.ones(4, pixels),
            "positions": torch.arange(4),
            "extra": torch.zeros(4),
            "label": torch.tensor(index),
        }

    def get_shapes(self):
        return [(4, 2, pixels) for pixels in self.pixel_counts]


def test_audit_loader_pads_variable_pixel_parcels():
    batch = next(iter(_loader(_VariablePixelDataset(), batch_size=2)))
    assert batch["pixels"].shape == (2, 4, 2, 5)
    assert batch["valid_pixels"].shape == (2, 4, 5)
    assert torch.count_nonzero(batch["valid_pixels"][0, :, 3:]) == 0


def test_query_counterfactual_is_linear_after_one_shared_layer_norm():
    torch.manual_seed(601)
    model = _model().eval()
    projection = model.temporal_encoder.attention_heads.external_query_projection
    with torch.no_grad():
        projection.weight.normal_(std=.1)
    response = torch.randn(5, 48)
    result = query_counterfactuals(model, response)
    torch.testing.assert_close(
        result["full_delta"],
        result["presence_delta"] + result["organization_delta"],
    )
    assert torch.count_nonzero(result["master_external_query"]) == 0


def test_organization_roll_is_invariant_and_mean_repeat_preserves_presence():
    torch.manual_seed(607)
    model = _model().eval()
    similarity = torch.randn(4, 8, 16)
    base = model.structure_branch.compose_state_org_response(similarity)
    variants = organization_counterfactuals(model, base)
    torch.testing.assert_close(variants["roll_1"]["presence"], base["presence"])
    torch.testing.assert_close(variants["roll_2"]["presence"], base["presence"])
    torch.testing.assert_close(variants["mean_repeat"]["presence"], base["presence"])
    torch.testing.assert_close(
        variants["roll_1"]["organization"], base["organization"],
        atol=2e-6, rtol=2e-6,
    )
    assert not torch.allclose(
        variants["mean_repeat"]["organization"], base["organization"],
    )


def test_detached_projected_target_query_blocks_only_structure_query_gradients():
    torch.manual_seed(613)
    model = _model().train()
    projection = model.temporal_encoder.attention_heads.external_query_projection
    with torch.no_grad():
        projection.weight.normal_(std=.05)
    output = model.forward_with_temporal_shift(
        *_batch(), return_dict=True, detach_structure_query=True,
    )
    F.cross_entropy(output["logits"], torch.tensor([0, 1, 2])).backward()
    blocked = [
        model.structure_branch.token_generator.state_encoder.input_projection.weight,
        model.structure_branch.shapelet_dictionary.anchors,
        model.structure_branch.organization_encoder[0].weight,
        model.shape_response_norm.weight,
        projection.weight,
    ]
    assert all(parameter.grad is None or not parameter.grad.abs().any() for parameter in blocked)
    assert model.temporal_encoder.attention_heads.key.weight.grad.abs().sum() > 0
    assert next(model.decoder.parameters()).grad.abs().sum() > 0


def test_freeze_structure_specific_keeps_shared_pse_ltae_decoder_trainable():
    model = _model()
    frozen = configure_structure_specific_training(model, freeze=True)
    assert frozen
    assert all(not parameter.requires_grad for _, parameter in frozen)
    assert any(parameter.requires_grad for parameter in model.spatial_encoder.parameters())
    assert model.temporal_encoder.attention_heads.key.weight.requires_grad
    assert next(model.decoder.parameters()).requires_grad


def test_rule_labels_are_deterministic_and_evidence_only():
    labels = rule_labels({
        "target_oracle": .85, "source_to_target": .40,
        "presence_minus_master": .01, "organization_minus_master": -.02,
        "mean_repeat_minus_original": .03,
        "source_full_minus_master": .02, "last_full_minus_master": -.01,
        "no_shape_aux_peak_final_gain": .02,
        "detach_target_final_gain": .03,
    })
    assert "CROSS_DOMAIN_GEOMETRY_MISMATCH" in labels
    assert "ORGANIZATION_NEGATIVE_TRANSFER" in labels
    assert "UDA_STRUCTURE_DRIFT" in labels
    assert "SOURCE_AUX_NEGATIVE_TRANSFER" in labels
    assert "PSEUDO_STRUCTURE_CONTAMINATION" in labels


def test_launcher_contract():
    launcher = Path("scripts/run_state_org_feasibility_audit_seed1.sh")
    assert launcher.exists()
    text = launcher.read_text(encoding="utf-8")
    assert "outputs/structure_state_org_4tasks_seed1" in text
    assert "outputs/state_org_feasibility" in text
    assert "--epochs 8" in text and "--steps_per_epoch 500" in text
    assert "--uda-shape-class-weight 0" in text
    assert "--detach-target-structure true" in text
    assert "--freeze-structure-specific true" in text
    assert "GPU0" in text and "GPU1" in text and "GPU2" in text
    assert text.index("analysis/state_org_feasibility_audit.py") < text.index(
        "run_variant \"$gpu\""
    )


def test_frozen_audit_is_read_only_and_uses_fixed_logistic_probe():
    source = Path("analysis/state_org_feasibility_audit.py").read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "optimizer" not in source.lower()
    assert 'class_weight="balanced"' in source
    assert "random_state=int(seed)" in source
    assert "max_iter=1000" in source
    assert '"teacher_state_dict"' in source
    assert '"target_test_labels_used_for_training": False' in source
