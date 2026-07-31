from __future__ import annotations

import torch
from torch import nn

from models.ltae import ComponentAwareSharedLTAE


def _encoder() -> ComponentAwareSharedLTAE:
    torch.manual_seed(1201)
    return ComponentAwareSharedLTAE(
        in_channels=4,
        n_head=2,
        d_k=3,
        n_neurons=(6, 5),
        dropout=0.0,
        d_model=6,
        T=100,
        max_temporal_shift=2,
        max_position=12,
    ).eval()


def _inputs():
    torch.manual_seed(1202)
    components = tuple(torch.randn(2, 5, 4) for _ in range(3))
    positions = torch.tensor([[0, 1, 3, 6, 9], [1, 2, 4, 7, 10]])
    mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, False, True]]
    )
    return components, positions, mask


def test_component_stems_and_output_norms_are_independent_without_batch_norm() -> None:
    encoder = _encoder()

    assert set(encoder.stems) == {"trend", "dynamics", "residual"}
    assert set(encoder.output_norms) == {"trend", "dynamics", "residual"}
    stem_linears = [encoder.stems[name][0] for name in encoder.component_names]
    stem_norms = [encoder.stems[name][1] for name in encoder.component_names]
    output_norms = [encoder.output_norms[name] for name in encoder.component_names]
    assert all(isinstance(module, nn.Linear) and module.bias is None for module in stem_linears)
    assert all(isinstance(module, nn.LayerNorm) for module in (*stem_norms, *output_norms))
    assert len({id(module.weight) for module in stem_linears}) == 3
    assert len({id(module.weight) for module in stem_norms}) == 3
    assert len({id(module.weight) for module in output_norms}) == 3
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in encoder.modules())


def test_all_components_share_one_attention_body_call_and_keep_output_shape() -> None:
    encoder = _encoder()
    components, positions, mask = _inputs()
    calls = []
    handle = encoder.attention_heads.register_forward_hook(
        lambda _module, args, output: calls.append(args[0].shape)
    )

    outputs = encoder(*components, positions, time_mask=mask)
    handle.remove()

    assert calls == [(6, 5, 6)]
    assert all(output.shape == (2, 5) for output in outputs)


def test_padding_is_invariant_and_all_invalid_output_is_zero() -> None:
    encoder = _encoder()
    components, positions, mask = _inputs()
    expected = encoder(*components, positions, time_mask=mask)
    padded_components = tuple(
        torch.cat([component, torch.randn(2, 3, 4) * 1e6], dim=1)
        for component in components
    )
    padded_positions = torch.cat([positions, torch.zeros(2, 3, dtype=torch.long)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(2, 3, dtype=torch.bool)], dim=1)

    actual = encoder(*padded_components, padded_positions, time_mask=padded_mask)
    for expected_component, actual_component in zip(expected, actual):
        torch.testing.assert_close(actual_component, expected_component)

    all_invalid = torch.zeros_like(mask)
    invalid_outputs = encoder(*components, positions, time_mask=all_invalid)
    assert all(torch.count_nonzero(output) == 0 for output in invalid_outputs)


def test_each_stem_and_shared_attention_receive_gradients() -> None:
    encoder = _encoder().train()
    components, positions, mask = _inputs()
    outputs = encoder(*components, positions, time_mask=mask)
    sum(output.square().sum() for output in outputs).backward()

    for name in encoder.component_names:
        assert encoder.stems[name][0].weight.grad is not None
        assert encoder.stems[name][0].weight.grad.abs().sum() > 0
    assert encoder.attention_heads.key.weight.grad is not None
    assert encoder.attention_heads.key.weight.grad.abs().sum() > 0


def test_changing_trend_stem_does_not_mutate_other_stem_parameters() -> None:
    encoder = _encoder()
    before = {
        name: encoder.stems[name][0].weight.detach().clone()
        for name in ("dynamics", "residual")
    }
    with torch.no_grad():
        encoder.stems["trend"][0].weight.add_(1.0)

    for name, value in before.items():
        torch.testing.assert_close(encoder.stems[name][0].weight, value)
