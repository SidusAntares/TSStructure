from __future__ import annotations

import copy

import pytest
import torch

from models.ltae import LTAE, MultiHeadAttention


def _ltae(dtype: torch.dtype = torch.float32) -> LTAE:
    torch.manual_seed(301)
    return LTAE(
        in_channels=4,
        n_head=2,
        d_k=3,
        n_neurons=[6, 5],
        dropout=0.0,
        d_model=6,
        T=100,
        max_temporal_shift=2,
        max_position=12,
    ).to(dtype=dtype).eval()


def _inputs(dtype: torch.dtype = torch.float32):
    torch.manual_seed(302)
    x = torch.randn(3, 5, 4, dtype=dtype)
    positions = torch.tensor(
        [[0, 1, 3, 6, 9], [1, 2, 4, 7, 10], [0, 2, 5, 8, 11]]
    )
    mask = torch.tensor(
        [
            [True, True, False, True, True],
            [True, False, True, False, True],
            [False, False, False, False, False],
        ]
    )
    return x, positions, mask


def test_none_mask_matches_all_true_mask() -> None:
    model = _ltae()
    x, positions, _ = _inputs()

    legacy, legacy_att = model(x, positions, True)
    masked, masked_att = model(
        x, positions, return_att=True, time_mask=torch.ones(3, 5, dtype=torch.bool)
    )

    torch.testing.assert_close(masked, legacy)
    torch.testing.assert_close(masked_att, legacy_att)


def test_vector_mask_matches_batched_mask() -> None:
    model = _ltae()
    x, positions, _ = _inputs()
    vector = torch.tensor([True, False, True, True, False])

    vector_output = model(x, positions, time_mask=vector)
    batch_output = model(x, positions, time_mask=vector.expand(3, -1))

    torch.testing.assert_close(vector_output, batch_output)


def test_masked_extreme_and_nonfinite_tokens_do_not_change_output() -> None:
    model = _ltae()
    x, positions, mask = _inputs()
    changed = x.clone()
    changed[~mask] = 1e20
    changed[0, 2] = torch.tensor([float("nan"), float("inf"), -1e20, 1e20])

    expected = model(x, positions, time_mask=mask)
    actual = model(changed, positions, time_mask=mask)

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


def test_masked_attention_weights_are_exactly_zero() -> None:
    model = _ltae()
    x, positions, mask = _inputs()

    _, attention = model(x, positions, return_att=True, time_mask=mask)

    expanded_mask = mask[:, None, None, :].expand_as(attention)
    assert torch.count_nonzero(attention[~expanded_mask]) == 0


def test_all_invalid_sample_embedding_and_attention_are_zero() -> None:
    model = _ltae()
    x, positions, mask = _inputs()

    output, attention = model(x, positions, return_att=True, time_mask=mask)

    torch.testing.assert_close(output[2], torch.zeros_like(output[2]), atol=0, rtol=0)
    torch.testing.assert_close(
        attention[2], torch.zeros_like(attention[2]), atol=0, rtol=0
    )


def test_legacy_positional_return_att_call_remains_supported() -> None:
    model = _ltae()
    x, positions, _ = _inputs()

    output = model(x, positions)
    output_with_attention, attention = model(x, positions, True)

    assert output.shape == (3, 5)
    torch.testing.assert_close(output_with_attention, output)
    assert attention.shape == (3, 2, 1, 5)


@pytest.mark.parametrize(
    "mask,match",
    [
        (torch.ones(4), "time_mask"),
        (torch.ones(2, 5), "time_mask"),
        (torch.ones(3, 5, 1), "time_mask"),
        (torch.tensor([1, 1, 2, 1, 1]), "0/1"),
        (torch.tensor([1.0, 1.0, float("nan"), 1.0, 1.0]), "0/1"),
    ],
)
def test_invalid_masks_raise_value_error(mask: torch.Tensor, match: str) -> None:
    model = _ltae()
    x, positions, _ = _inputs()

    with pytest.raises(ValueError, match=match):
        model(x, positions, time_mask=mask)


def test_masked_input_gradients_are_exactly_zero() -> None:
    model = _ltae()
    x, positions, mask = _inputs()
    x.requires_grad_()

    output = model(x, positions, time_mask=mask)
    output.square().sum().backward()

    assert x.grad is not None
    assert torch.count_nonzero(x.grad[~mask]) == 0
    assert torch.isfinite(x.grad[mask]).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_forward_preserves_dtype(dtype: torch.dtype) -> None:
    model = _ltae(dtype)
    x, positions, mask = _inputs(dtype)

    output, attention = model(x, positions, return_att=True, time_mask=mask)

    assert output.dtype == dtype
    assert attention.dtype == dtype
    assert torch.isfinite(output).all()
    assert torch.isfinite(attention).all()


def test_multihead_attention_mask_matches_ltae_mask_semantics() -> None:
    torch.manual_seed(303)
    attention = MultiHeadAttention(n_head=2, d_k=3, d_in=6).eval()
    x = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, False, True, False], [False, False, False, False]])

    embedding, weights = attention(x, time_mask=mask)

    assert embedding.shape == (2, 6)
    assert weights.shape == (2, 2, 1, 4)
    assert torch.count_nonzero(weights[0, :, :, ~mask[0]]) == 0
    assert torch.count_nonzero(weights[1]) == 0
    assert torch.count_nonzero(embedding[1]) == 0


def test_masked_path_does_not_change_parameters_or_module_structure() -> None:
    model = _ltae()
    baseline = copy.deepcopy(model.state_dict())
    x, positions, mask = _inputs()

    model(x, positions, time_mask=mask)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, baseline[name])
