"""Detached RBF PELT candidate discovery and shared Full/Keep classification.

Segmentation copies only observed PSE features to CPU float64, standardizes each
feature over the observed sequence (population std, constant channels use 1),
and uses exp(-gamma * squared Euclidean distance). Gamma is inverse median
positive pairwise squared distance (1 for constant data). Penalty is penalty*N
for N observed input dates, as the INSHAPE normalized penalty convention.
No imputation, fixed segment count, segmentation gradient, or external package.
"""

import numpy as np
import torch

from models.segment_selector import DETACHED_DROP_PROTOCOL, DROP_PROTOCOLS, DROP_PROXY_PROTOCOLS
from utils.soft_top50 import frozen_batchnorm_only, keep_head_embeddings


def _rbf_cost(features):
    x = np.asarray(features, dtype=np.float64)
    scale = x.std(0)
    x = (x-x.mean(0)) / np.where(scale > 1e-12, scale, 1.)
    distances = np.maximum(((x[:, None]-x[None])**2).sum(-1), 0.)
    upper = distances[np.triu_indices(len(x), 1)]
    positive = upper[upper > 1e-12]
    gamma = 1. / np.median(positive) if len(positive) else 1.
    kernel = np.exp(-gamma*distances)
    integral = np.pad(kernel.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def cost(start, end):
        total = integral[end, end] - integral[start, end] - integral[end, start] + integral[start, start]
        return max(0., end-start-total/(end-start))
    return cost


def _pelt_boundaries(x, penalty, min_size):
    n = len(x)
    if n < 2*min_size:
        return [0, n]
    cost = _rbf_cost(x)
    objective = np.full(n+1, np.inf)
    previous = np.zeros(n+1, dtype=int)
    objective[0] = -penalty
    active = []
    expires = {}
    for end in range(min_size, n+1):
        # Delay pruning by min_size: replacement via a newly found boundary
        # must itself produce a feasible next segment. K=0 for kernel SSE.
        active = [start for start in active if expires.get(start, n+1) > end]
        start = end-min_size
        if np.isfinite(objective[start]):
            active.append(start)
        values = [objective[start] + cost(start, end) + penalty for start in active]
        winner = int(np.argmin(values))
        objective[end] = values[winner]
        previous[end] = active[winner]
        for start in active:
            if objective[start] + cost(start, end) > objective[end] + 1e-10:
                expires[start] = min(expires.get(start, n+1), end+min_size)
    boundaries = [n]
    while boundaries[-1] > 0:
        boundaries.append(int(previous[boundaries[-1]]))
    return boundaries[::-1]


def segment_pelt(features, valid=None, penalty=.1, min_size=3, penalty_mode='per_observation'):
    """Return [B,T] labels, -1 for invalid dates, contiguous observed segments.

    Exact PELT with all integer candidate boundaries (jump=1); short sequences
    below min_size remain one segment. Min_size counts observed dates. The legacy
    default scales penalty by N; opt-in absolute mode uses a fixed cost penalty.
    """
    if features.ndim != 3 or not np.isfinite(penalty) or penalty < 0 or min_size < 1 or int(min_size) != min_size:
        raise ValueError('Expected features [B,T,C], nonnegative penalty, positive integer min_size')
    if penalty_mode not in ('per_observation', 'absolute'):
        raise ValueError('penalty_mode must be per_observation or absolute')
    valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device) if valid is None else valid.bool()
    if valid.shape != features.shape[:2] or not valid.any(-1).all():
        raise ValueError('Each sequence requires at least one observed date')
    labels = torch.full(features.shape[:2], -1, dtype=torch.long, device=features.device)
    for batch in range(features.size(0)):
        x = features[batch, valid[batch]].detach().to(device='cpu', dtype=torch.float64).numpy()
        if not np.isfinite(x).all():
            raise ValueError('Observed features must be finite')
        actual_penalty = penalty*len(x) if penalty_mode == 'per_observation' else penalty
        boundaries = _pelt_boundaries(x, actual_penalty, int(min_size))
        sequence = np.empty(len(x), dtype=np.int64)
        for k, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            sequence[start:end] = k
        labels[batch, valid[batch]] = torch.as_tensor(sequence, device=features.device)
    return labels


def _pool_gate_gradient(encoder, features, positions, valid, gate):
    """Estimate gate gradients using frozen, dropout-free temporal features."""
    with torch.no_grad(), frozen_batchnorm_only(encoder):
        encoded = encoder.encode_dates(features, positions, valid)
        _, _, attention, values = encoder.attention_heads(
            encoded, return_parts=True, valid_time=valid, apply_dropout=False)
    # A small floor bounds the gradient when retained dates have little
    # attention mass. The proxy value itself is canceled before classification.
    soft_gate = .05 + .95 * gate
    pooled, _, _ = keep_head_embeddings(attention, values, soft_gate[:, None].expand_as(attention))
    return pooled


def _classify_retained_observations(model, selector, features, positions, segments, deterministic):
    encoder = model.temporal_encoder
    selector_features = features.detach() if selector.protocol == DETACHED_DROP_PROTOCOL else features
    info = selector(selector_features, None, segments, variant='inshape', deterministic=deterministic,
                    head_count=encoder.attention_heads.n_head)
    gate = info['point_mask'][:, 0]
    retained = gate.detach().bool()
    pool_gradient = None
    if selector.protocol in DROP_PROXY_PROTOCOLS:
        selected = [x[mask] for x, mask in zip(features, retained)]
        if torch.is_grad_enabled() and gate.requires_grad:
            pool_gradient = _pool_gate_gradient(encoder, features, positions, segments >= 0, gate)
    else:
        # Historical protocol passes the gradient through retained gates only.
        selected = [x[mask] * g[mask, None] for x, mask, g in zip(features, retained, gate)]
    selected_dates = [dates[mask] for dates, mask in zip(positions, retained)]
    packed = torch.nn.utils.rnn.pad_sequence(selected, batch_first=True)
    packed_dates = torch.nn.utils.rnn.pad_sequence(selected_dates, batch_first=True)
    valid = torch.arange(packed.size(1), device=packed.device)[None] < retained.sum(-1, keepdim=True)
    temporal, attention, _, _ = encoder(packed, packed_dates, return_attention_scores=True,
                                        valid_time=valid, pool_gradient=pool_gradient)
    # Export attention on the original observation axis for visualization.
    aligned = attention.new_zeros((*features.shape[:2], attention.size(1)))
    aligned[retained] = attention.transpose(1, 2)[valid]
    info.update(features=features, attention=aligned.transpose(1, 2))
    return model.decoder(temporal), info


def structure_forward(model, selector, pixels, mask, positions, extra, variant='attention',
                      deterministic=False, full_loss=True, penalty=.1, min_size=3,
                      penalty_mode='per_observation', unified=False):
    """One PSE pass, with an opt-in unified Keep-only training path.

    unified=True uses original attention dropout and one normal backend BN/dropout
    pass. Full is evaluated separately. The legacy two-path behavior remains
    available for inspecting archived experiments.

    Target full_loss=False avoids the unused Full MLP/classifier entirely and
    updates their BN once using Keep. Gates multiply raw attention continuously
    in autograd while remaining exactly binary in the forward pass.
    """
    if unified and full_loss:
        raise ValueError('Unified training uses only Keep; evaluate Full separately in eval mode')
    valid = mask.bool().any(-1)
    if not valid.any(-1).all():
        raise ValueError('Each sample requires an observed date')
    if bool(valid.all()):
        features = model.spatial_encoder(pixels, mask, extra)
        attention_mask = None
    else:
        selected_extra = None if extra is None else extra[:, None].expand(-1, pixels.size(1), -1)[valid]
        selected = model.spatial_encoder(pixels[valid][:, None], mask[valid][:, None], selected_extra)
        features = selected.new_zeros(*pixels.shape[:2], selected.size(-1))
        features[valid] = selected[:, 0]
        attention_mask = valid
    encoder = model.temporal_encoder
    if getattr(selector, 'protocol', None) in DROP_PROTOCOLS:
        if variant != 'inshape':
            raise ValueError('Dropping observations before LTAE requires the pure-content variant')
        segments = segment_pelt(features, valid, penalty=penalty, min_size=min_size, penalty_mode=penalty_mode)
        keep, info = _classify_retained_observations(model, selector, features, positions, segments, deterministic)
        full = (model.decoder(encoder(features, positions, return_attention_scores=True,
                                      valid_time=attention_mask)[0]) if full_loss else None)
        return full, keep, info
    full = None
    if full_loss:
        temporal, attention, values, scores = encoder(features, positions, return_attention_scores=True, valid_time=attention_mask)
        full = model.decoder(temporal)
    else:
        x = features
        if encoder.inconv is not None:
            if attention_mask is None:
                x = encoder.inconv(x)
            else:
                projected = encoder.inconv(x[valid])
                x = features.new_zeros(*features.shape[:2], projected.size(-1))
                x[valid] = projected
        safe_positions = torch.where(valid, positions, torch.zeros_like(positions))
        x = x + encoder.positional_enc(safe_positions + encoder.max_temporal_shift)
        _, _, attention, values, scores = encoder.attention_heads(x, return_parts=True,
            return_scores=True, valid_time=attention_mask, apply_dropout=not unified)
    segments = segment_pelt(features, valid, penalty=penalty, min_size=min_size, penalty_mode=penalty_mode)
    info = selector(features, attention, segments, variant=variant, deterministic=deterministic)
    pooled, normalized, mass = keep_head_embeddings(attention, values, info['point_mask'])
    if unified:
        # Use the original attention-dropout and aggregation operations after
        # mask normalization. Backend BN/dropout runs exactly once in its
        # normal mode, just as for the unmodified LTAE.
        dropped = encoder.attention_heads.dropout(normalized.unsqueeze(-2))
        pooled = (dropped @ values).transpose(1, 2).contiguous().view(pixels.size(0), -1)
    if full_loss:
        with frozen_batchnorm_only(encoder.mlp, model.decoder):
            keep = model.decoder(encoder.dropout(encoder.mlp(pooled)))
    else:
        keep = model.decoder(encoder.dropout(encoder.mlp(pooled)))
    info.update(features=features, attention=attention, values=values, scores=scores,
                keep_attention=normalized, retained_mass=mass)
    return full, keep, info
