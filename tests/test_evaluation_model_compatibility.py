from types import SimpleNamespace

import torch

from evaluation import forward_evaluation_logits


class _TensorOutputModel(torch.nn.Module):
    def forward(self, pixels, valid_pixels, positions, extra):
        return pixels.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)


class _StructuredOutputModel(torch.nn.Module):
    def forward(self, pixels, valid_pixels, positions, extra):
        logits = pixels.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return SimpleNamespace(logits=logits)


def _inputs():
    return (
        torch.ones(2, 3, 4, 5),
        torch.ones(2, 3, 5),
        torch.arange(3).expand(2, -1),
        None,
    )


def test_evaluation_accepts_original_pseltae_tensor_output_contract():
    logits = forward_evaluation_logits(_TensorOutputModel(), *_inputs())
    assert logits.shape == (2, 1)
    assert torch.equal(logits, torch.ones_like(logits))


def test_evaluation_accepts_structured_logits_output_without_geometry_keyword():
    logits = forward_evaluation_logits(_StructuredOutputModel(), *_inputs())
    assert logits.shape == (2, 1)
    assert torch.equal(logits, torch.ones_like(logits))
