"""Exercise actual NumPy-to-Torch transforms rather than prebuilt tensors."""
import numpy as np
import pytest
import torch

from transforms import Normalize, RandomSamplePixels, RandomSampleTimeSteps, ToTensor


@pytest.mark.parametrize('with_extra', [False, True])
def test_numpy_without_long_alias_keeps_date_values_and_int64_indices(monkeypatch, with_extra):
    monkeypatch.delitem(np.__dict__, 'long', raising=False)
    dates = np.array([-60, 0, 12, 365, 400], dtype=np.int32)
    sample = dict(pixels=np.ones((5, 10, 4), dtype=np.uint16),
                  valid_pixels=np.ones((5, 4), dtype=np.float32), positions=dates.copy(), label=1)
    if with_extra:
        sample['extra'] = np.zeros(4, dtype=np.float64)
    result = ToTensor()(sample)
    assert result['positions'].dtype == torch.int64
    np.testing.assert_array_equal(result['positions'].numpy(), dates)
    embedding = torch.nn.Embedding(565, 2)
    assert embedding(result['positions']+100).shape == (5, 2)
    assert result['pixels'].dtype == result['valid_pixels'].dtype == torch.float32


def test_real_training_transform_chain_collates_and_runs_pseltae(monkeypatch):
    from models.stclassifier import PseLTae
    monkeypatch.delitem(np.__dict__, 'long', raising=False)
    rows = []
    for label in (0, 1):
        item = dict(pixels=np.ones((40, 10, 8), dtype=np.uint16)*1000,
                    valid_pixels=np.ones((40, 8), dtype=np.float32),
                    positions=np.arange(40, dtype=np.int32)*9, label=label, extra=np.zeros(4))
        for transform in (RandomSamplePixels(64), RandomSampleTimeSteps(30), Normalize(), ToTensor()):
            item = transform(item)
        rows.append(item)
    batch = next(iter(torch.utils.data.DataLoader(rows, batch_size=2, num_workers=0)))
    assert batch['pixels'].shape == (2, 30, 10, 64)
    assert batch['positions'].dtype == torch.int64
    assert (batch['positions'][:, 1:] > batch['positions'][:, :-1]).all()
    assert (batch['positions'] % 9 == 0).all()
    model = PseLTae(num_classes=2, with_extra=False).eval()
    with torch.no_grad():
        logits = model(batch['pixels'], batch['valid_pixels'], batch['positions'], batch['extra'])
    assert logits.shape == (2, 2) and torch.isfinite(logits).all()
