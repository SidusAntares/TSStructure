"""Differentiable attention-derived selection for the independent M1 source run."""

from contextlib import contextmanager

import torch
from torch import nn


def soft_top50(scores, valid_time, tau=.5, rho=.5):
    if scores.ndim != 3 or valid_time.shape != (scores.size(0), scores.size(2)):
        raise ValueError('Expected scores [B,H,T] and valid_time [B,T]')
    if tau <= 0 or not 0 < rho < 1:
        raise ValueError('tau must be positive and rho must be in (0,1)')
    valid = valid_time.bool()[:, None].expand_as(scores)
    count = valid.sum(-1, keepdim=True)
    if (count == 0).any() or not torch.isfinite(scores[valid]).all():
        raise ValueError('Every sample needs finite scores at one or more valid dates')
    ordered = scores.masked_fill(~valid, float('inf')).sort(-1).values
    rank = (count-1)*(1-rho)
    low, high = rank.floor().long(), rank.ceil().long()
    # Stop only the quantile threshold gradient. Scores and sigmoid stay live.
    threshold = (ordered.gather(-1, low)*(high.to(rank.dtype)-rank)
                 + ordered.gather(-1, high)*(rank-low.to(rank.dtype)))
    # When both ranks coincide, the interpolation above has zero coefficients.
    threshold = torch.where(low == high, ordered.gather(-1, low), threshold).detach()
    safe_scores = torch.where(valid, scores, torch.zeros_like(scores))
    selection = torch.sigmoid((safe_scores-threshold)/tau)*valid.to(scores.dtype)
    order = scores.detach().masked_fill(~valid, float('-inf')).argsort(dim=-1, descending=True, stable=True)
    selected_count = (count*rho).ceil().long()
    target_rank = torch.arange(scores.size(-1), device=scores.device)[None, None] < selected_count
    hard_target = torch.zeros_like(scores).scatter_(-1, order, target_rank.to(scores.dtype))
    mean = selection.sum(-1, keepdim=True)/count
    ratio = ((mean-rho)**2).mean()
    sharpness = (((selection-hard_target)**2)*valid).sum()/valid.sum()
    spread = (((selection-mean)**2)*valid).sum(-1)/count[..., 0]
    return dict(selection=selection, hard_target=hard_target, valid_time=valid_time.bool(),
                ratio_loss=ratio, sharpness_loss=sharpness,
                gate_std_per_head=spread.clamp_min(0).sqrt(),
                selection_fraction=mean[..., 0], effective_count=((selection >= .5)&valid).sum(-1),
                near_zero=((selection < .1)&valid).sum()/valid.sum(),
                near_one=((selection > .9)&valid).sum()/valid.sum())


def keep_head_embeddings(attention, values, selection):
    if attention.shape != selection.shape or values.shape[:-1] != attention.shape:
        raise ValueError('Incompatible attention, selection and Value shapes')
    weighted = attention*selection
    mass = weighted.sum(-1, keepdim=True)
    normalized = weighted/mass.clamp_min(torch.finfo(weighted.dtype).tiny)
    pooled = (normalized[..., None]*values).sum(-2).flatten(1)
    return pooled, normalized, mass[..., 0]


@contextmanager
def frozen_batchnorm_only(*roots):
    """Keep Dropout active and gradients enabled, freezing only running BN stats."""
    states = [(module, module.training) for root in roots for module in root.modules()
              if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    try:
        for module, _ in states:
            module.eval()
        yield
    finally:
        for module, state in states:
            module.train(state)


def schedule(epoch, epochs, tau_start=.5, tau_end=.05, warmup_epochs=5):
    fraction = epoch/max(epochs-1, 1)
    return tau_start*(tau_end/tau_start)**fraction, min(1., (epoch+1)/max(warmup_epochs, 1))
