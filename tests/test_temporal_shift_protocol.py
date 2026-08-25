import sys
import types

import pytest
import torch


class _Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


torchvision_stub = types.ModuleType("torchvision")
transforms_stub = types.ModuleType("torchvision.transforms")
transforms_stub.Compose = _Compose
transforms_stub.transforms = transforms_stub
torchvision_stub.transforms = transforms_stub
sys.modules.setdefault("torchvision", torchvision_stub)
sys.modules.setdefault("torchvision.transforms", transforms_stub)

import timematch
import evaluation as evaluation_module
from models.stclassifier import PseLTae


def _tiny_pseltae():
    return PseLTae(
        input_dim=2,
        mlp1=[2, 4],
        pooling="mean_std",
        mlp2=[8, 8],
        with_extra=False,
        n_head=2,
        d_k=2,
        d_model=8,
        mlp3=[8, 4],
        dropout=0.0,
        mlp4=[4, 3],
        num_classes=2,
    )


def _batch():
    torch.manual_seed(19)
    return (
        torch.randn(2, 4, 2, 3),
        torch.ones(2, 4, 3),
        torch.tensor([[10, 50, 100, 180], [12, 55, 105, 185]], dtype=torch.long),
        torch.zeros(2, 4),
    )


@pytest.mark.parametrize("shift", [0, 7, -5])
def test_pseltae_protocol_is_numerically_equivalent_to_old_shift_path(shift):
    model = _tiny_pseltae().eval()
    pixels, mask, positions, extra = _batch()
    with torch.no_grad():
        spatial = model.spatial_encoder(pixels, mask, extra)
        old_logits = model.decoder(model.temporal_encoder(spatial, positions + shift))
        protocol_logits = model.forward_with_temporal_shift(
            pixels,
            mask,
            positions,
            extra,
            temporal_shift=shift,
        )

    assert torch.equal(old_logits, protocol_logits)


def test_pseltae_protocol_preserves_mixed_per_sample_shift_path():
    model = _tiny_pseltae().eval()
    pixels, mask, positions, extra = _batch()
    shifts = torch.tensor([[7], [0]], dtype=positions.dtype)
    with torch.no_grad():
        spatial = model.spatial_encoder(pixels, mask, extra)
        old_logits = model.decoder(model.temporal_encoder(spatial, positions + shifts))
        protocol_logits = model.forward_with_temporal_shift(
            pixels,
            mask,
            positions,
            extra,
            temporal_shift=shifts,
        )

    assert torch.equal(old_logits, protocol_logits)


class _CountingSpatial(torch.nn.Module):
    def forward(self, pixels, mask, extra):
        return pixels


class _CountingProtocolModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_encoder = _CountingSpatial()
        self.prepare_calls = 0
        self.classify_calls = 0

    def prepare_temporal_features(self, spatial_feats, positions):
        self.prepare_calls += 1
        return spatial_feats.mean(dim=1)

    def classify_prepared(self, prepared, positions, temporal_shift=0):
        self.classify_calls += 1
        score = prepared.flatten(1).mean(dim=1) + float(temporal_shift) * 0.01
        return torch.stack([score, -score], dim=1)


def test_shift_sweep_prepares_temporal_features_once_per_batch(monkeypatch):
    model = _CountingProtocolModel()
    sample = {
        "pixels": torch.ones(2, 3, 1),
        "valid_pixels": torch.ones(2, 3),
        "positions": torch.tensor([[10, 20, 30], [11, 21, 31]]),
        "extra": torch.zeros(2, 4),
        "label": torch.tensor([0, 1]),
    }
    loader = [sample]
    monkeypatch.setattr(
        timematch,
        "to_cuda",
        lambda value, device: (
            value["pixels"],
            value["valid_pixels"],
            value["positions"],
            value["extra"],
        ),
    )

    timematch.estimate_temporal_shift(
        model,
        loader,
        "cpu",
        min_shift=-2,
        max_shift=2,
        sample_size=1,
        shift_estimator="IS",
        progress_bar="off",
    )

    assert model.prepare_calls == 1
    assert model.classify_calls == 5


class _RangeEncoder:
    def __init__(self, table_size):
        self.max_temporal_shift = 100
        self.positional_enc = torch.nn.Embedding(table_size, 1)


class _DualRangeModel:
    def __init__(self):
        self.encoders = (_RangeEncoder(565), _RangeEncoder(120))

    def get_temporal_encoders(self):
        return self.encoders


def test_temporal_range_check_validates_both_fredn_ltaes():
    positions = torch.tensor([[30, 40]], dtype=torch.long)

    with pytest.raises(ValueError, match="encoder=1"):
        timematch._check_temporal_index_range(
            _DualRangeModel(),
            positions,
            applied_shift=0,
            tag="source",
        )


class _EvaluationProtocolModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, *args, **kwargs):
        raise AssertionError("shifted positions must not use the legacy path")

    def forward_with_temporal_shift(
        self,
        pixels,
        mask,
        positions,
        extra,
        temporal_shift=0,
    ):
        self.calls.append((positions.detach().clone(), temporal_shift))
        return torch.tensor([[2.0, -1.0], [-1.0, 2.0]])


def test_evaluation_keeps_original_positions_outside_ltae_shift(monkeypatch):
    model = _EvaluationProtocolModel()
    positions = torch.tensor([[10, 20], [12, 22]], dtype=torch.long)
    sample = {
        "pixels": torch.ones(2, 2, 1),
        "valid_pixels": torch.ones(2, 2),
        "positions": positions,
        "extra": torch.zeros(2, 4),
        "label": torch.tensor([0, 1]),
    }
    monkeypatch.setattr(
        evaluation_module,
        "to_cuda",
        lambda value, device: (
            value["pixels"],
            value["valid_pixels"],
            value["positions"],
            value["extra"],
        ),
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)

    evaluation_module.evaluation(
        model,
        [sample],
        "cpu",
        ["a", "b"],
        temporal_shift=7,
        progress_bar="off",
    )

    assert len(model.calls) == 1
    assert torch.equal(model.calls[0][0], positions)
    assert model.calls[0][1] == 7
