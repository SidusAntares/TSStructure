"""Head-local temporal evidence for the closed-set TimeMatch experiments.

The interval is selected from the *undropped* LTAE attention distribution.
Only actually observed dates inside the interval are used; no interpolation
or Fourier reconstruction is performed.
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F


def select_attention_intervals(
    attention, positions, top_fraction=0.30, coverage=0.75,
    min_observations=3, min_outside=2,
):
    """Return (inside[B,H,T], valid[B,H]) for shortest calendar intervals.

    The interval must contain `coverage` of the attention mass among the
    highest `top_fraction` observations. Observation-count checks take place
    *after* selection: they never artificially widen a weak candidate.
    """
    if not (0 < top_fraction <= 1 and 0 < coverage <= 1):
        raise ValueError("top_fraction and coverage must be in (0, 1]")
    if min_observations < 1 or min_outside < 1:
        raise ValueError("min_observations and min_outside must be positive")
    if attention.ndim != 3 or positions.ndim != 2:
        raise ValueError("expected attention [B,H,T] and positions [B,T]")
    batch, heads, steps = attention.shape
    if positions.shape != (batch, steps):
        raise ValueError("attention and positions have inconsistent shapes")
    if steps < min_observations + min_outside:
        return (torch.zeros_like(attention, dtype=torch.bool),
                torch.zeros(batch, heads, device=attention.device, dtype=torch.bool))

    with torch.no_grad():
        weights = attention.detach()
        count = max(1, int(torch.ceil(torch.tensor(top_fraction * steps)).item()))
        top_indices = weights.topk(count, dim=-1).indices
        selected = torch.zeros_like(weights)
        selected.scatter_(-1, top_indices, weights.gather(-1, top_indices))
        prefix = selected.cumsum(dim=-1)
        before = torch.cat([torch.zeros_like(prefix[..., :1]), prefix[..., :-1]], dim=-1)
        interval_mass = prefix.unsqueeze(-2) - before.unsqueeze(-1)
        required_mass = coverage * selected.sum(dim=-1)[..., None, None]

        dates = positions.to(torch.long)
        widths = dates[:, None, None, :] - dates[:, None, :, None]
        eligible = interval_mass >= (required_mass - 1e-7)
        indices = torch.arange(steps, device=attention.device)
        eligible = eligible & (indices[None, :] >= indices[:, None])[None, None]
        max_width = torch.iinfo(torch.long).max
        candidates = widths.expand(batch, heads, steps, steps).masked_fill(
            ~eligible, max_width
        )
        best_flat = candidates.flatten(-2).argmin(dim=-1)
        start, end = best_flat // steps, best_flat % steps
        observed_count = end - start + 1
        valid = ((observed_count >= min_observations)
                 & (steps - observed_count >= min_outside)
                 & (candidates.flatten(-2).amin(dim=-1) != max_width))
        timeline = indices.view(1, 1, steps)
        inside = ((timeline >= start.unsqueeze(-1))
                  & (timeline <= end.unsqueeze(-1))
                  & valid.unsqueeze(-1))
        return inside, valid


def build_counterfactual_head_embeddings(attention, values, inside):
    """Produce full and one-head-at-a-time ablated embeddings [B,H+1,C].

    Values in a head's selected region are replaced by the *same sample's*
    unweighted mean of the observations outside that region. Attention is
    unchanged and is never renormalized after removing a region. All other
    heads stay exactly as they were.
    """
    if attention.shape != values.shape[:-1] or inside.shape != attention.shape:
        raise ValueError("incompatible attention, values and interval shapes")
    batch, heads, _ = attention.shape
    outside = (~inside).to(values.dtype).unsqueeze(-1)
    reference = (values * outside).sum(dim=-2) / outside.sum(dim=-2).clamp_min(1)
    original = (attention.unsqueeze(-1) * values).sum(dim=-2)
    removed = (attention.unsqueeze(-1) * inside.unsqueeze(-1)
               * (values - reference.unsqueeze(-2))).sum(dim=-2)
    identity = torch.eye(heads, device=values.device, dtype=values.dtype)
    alternatives = (original[:, None] -
                    identity[None, :, :, None] * removed[:, None])
    return torch.cat([original.flatten(1).unsqueeze(1),
                      alternatives.flatten(2)], dim=1)


def relative_logit_contributions(logits, labels):
    """Class-vs-best-other logit margin lost after each head intervention.

    `logits` is [B,H+1,C] (full sample, then H counterfactuals).
    Output is [B,H], where positive means the head supports `labels`.
    """
    if logits.ndim != 3 or labels.shape != (logits.size(0),):
        raise ValueError("expected logits [B,H+1,C] and labels [B]")
    if logits.size(-1) < 2:
        raise ValueError("at least two classes are required")
    target = labels[:, None, None].expand(-1, logits.size(1), 1)
    class_logits = logits.gather(-1, target).squeeze(-1)
    competitors = logits.scatter(-1, target, float("-inf")).amax(dim=-1)
    margins = class_logits - competitors
    return margins[:, :1] - margins[:, 1:]


@contextmanager
def frozen_evidence_normalization(*modules):
    """Use running BN statistics without disabling gradient calculation.

    The ordinary supervised/teacher-student branch remains in train mode;
    only the counterfactual comparison uses deterministic inference-time BN.
    """
    states = [(module, module.training) for root in modules for module in root.modules()]
    try:
        for root in modules:
            root.eval()
        yield
    finally:
        for module, training in states:
            module.training = training


class SourceEvidenceTracker:
    """Online class-head source statistics; target labels never update it."""

    def __init__(self, num_classes, num_heads, device, min_count=16,
                 min_positive_rate=0.65, min_source_contribution=0.05,
                 rank_gap=0.20):
        shape = (num_classes, num_heads)
        self.sums = torch.zeros(shape, device=device)
        self.counts = torch.zeros(shape, device=device)
        self.positive_counts = torch.zeros(shape, device=device)
        self.min_count = min_count
        self.min_positive_rate = min_positive_rate
        self.min_source_contribution = min_source_contribution
        self.rank_gap = rank_gap

    @torch.no_grad()
    def update(self, labels, contributions, valid):
        mask = valid.to(contributions.dtype)
        self.sums.index_add_(0, labels, contributions.detach() * mask)
        self.counts.index_add_(0, labels, mask)
        self.positive_counts.index_add_(
            0, labels, (contributions.detach() > 0).to(mask.dtype) * mask
        )

    def source_roles(self):
        mean = self.sums / self.counts.clamp_min(1)
        positive_rate = self.positive_counts / self.counts.clamp_min(1)
        active = ((self.counts >= self.min_count)
                  & (positive_rate >= self.min_positive_rate)
                  & (mean >= self.min_source_contribution))
        return mean, active

    def losses(self, contributions, valid, pseudo_labels, confidence,
               positive_margin=0.10, ranking_margin=0.10):
        mean, active = self.source_roles()
        available = active[pseudo_labels] & valid
        sample_weight = confidence.detach().unsqueeze(-1)
        weighted_mask = available.to(contributions.dtype) * sample_weight
        normalizer = weighted_mask.sum().clamp_min(1)
        positive = (F.relu(positive_margin - contributions)
                    * weighted_mask).sum() / normalizer

        source_mean = mean[pseudo_labels]
        ordered = (source_mean.unsqueeze(-1) - source_mean.unsqueeze(-2)
                   >= self.rank_gap)
        pairs = (ordered & available.unsqueeze(-1) & available.unsqueeze(-2))
        weighted_pairs = pairs.to(contributions.dtype) * sample_weight.unsqueeze(-1)
        target_difference = (contributions.unsqueeze(-1)
                             - contributions.unsqueeze(-2))
        ranking = (F.relu(ranking_margin - target_difference)
                   * weighted_pairs).sum() / weighted_pairs.sum().clamp_min(1)
        return positive, ranking, int(available.sum().item())
