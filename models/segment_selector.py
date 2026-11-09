"""Shared segment content encoder and head-conditioned binary ReinMax selector.

ReinMax backward adapted from microsoft/ReinMax, reinmax/reinmax.py (tau=1).
https://github.com/microsoft/ReinMax

MIT License. Copyright (c) Microsoft Corporation.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn

DETACHED_DROP_PROTOCOL = 'inshape_drop_proxy_detached_v1'
CONTEXT_DROP_PROTOCOL = 'inshape_drop_proxy_context_v1'
DETACHED_DROP_PROTOCOLS = (DETACHED_DROP_PROTOCOL, CONTEXT_DROP_PROTOCOL)
DROP_PROXY_PROTOCOLS = ('inshape_drop_proxy_v1', *DETACHED_DROP_PROTOCOLS)
DROP_PROTOCOLS = ('inshape_drop_v1', *DROP_PROXY_PROTOCOLS)


def selector_only_loss(loss, parameters):
    """Preserve scalar value and exact first-order Selector gradients only.

    The classification graph stays intact, including its path through content
    to the PSE. Sparse gradients are extracted without accumulating backbone
    .grad, then attached exclusively to Selector parameters.
    """
    if not torch.is_grad_enabled() or not loss.requires_grad:
        return loss.detach()
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    restricted = loss.detach()
    for parameter, gradient in zip(parameters, gradients):
        if gradient is not None:
            restricted = restricted + ((parameter-parameter.detach())*gradient.detach()).sum()
    return restricted


class _ReinMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, tau=1.):
        probability = logits.softmax(-1)
        draw = torch.multinomial(probability.reshape(-1, 2), 1).reshape(*logits.shape[:-1], 1)
        hard = torch.zeros_like(probability).scatter_(-1, draw, 1.)
        ctx.save_for_backward(hard, probability, (logits/tau).softmax(-1))
        return hard, probability

    @staticmethod
    def backward(ctx, grad_hard, grad_probability):
        hard, probability, tempered = ctx.saved_tensors
        if grad_hard is None:
            grad_hard = torch.zeros_like(probability)
        if grad_probability is None:
            grad_probability = torch.zeros_like(probability)
        shifted = .5 * (tempered + hard)
        first = 2 * grad_hard * shifted
        first = first - shifted * first.sum(-1, keepdim=True)
        zeroth = (-.5 * grad_hard + grad_probability) * probability
        zeroth = zeroth - probability * zeroth.sum(-1, keepdim=True)
        gradient = first + zeroth
        gradient = gradient - gradient.mean(-1, keepdim=True)
        return gradient if len(ctx.needs_input_grad) == 1 else (gradient, None)


class SegmentSelector(nn.Module):
    """Segment content encoder with archived and restored INSHAPE protocols.

    inshape_coverage_v1 reads content only, samples one gate per segment shared
    across LTAE heads, and applies adaptive sampled observation-coverage cost.
    inshape_attention_coverage_v1 adds detached mass/enrichment from all 16
    heads to that same shared scorer, with otherwise identical gate and loss.
    The conditioning description below applies only to archived legacy runs.

    `heads` is the number of Transformer heads, independent of LTAE head count.
    Segments use consecutive labels from zero; invalid time steps have label -1.
    As in upstream INSHAPE ROITransformerSelector, a learned CLS token pools
    each masked segment without positional embeddings. This implementation
    adapts input channels to PSE features and adds shared head conditioning.
    The three conditioning values are mass, T/n * mass, and n/T. Attention
    mass and enrichment are zero in the content-only `inshape` variant. Both
    variants receive the same length fraction, so length metadata is controlled.
    Attention is detached only
    for this conditioning; segment content retains its gradient to the PSE.
    """
    def __init__(self, feature_dim=128, width=64, heads=4, dropout=0.0, layers=1, protocol='legacy'):
        super().__init__()
        if protocol not in ('legacy', 'inshape_coverage_v1', 'inshape_attention_coverage_v1', *DROP_PROTOCOLS):
            raise ValueError('Unknown Selector protocol')
        self.protocol = protocol
        self.feature_dim = feature_dim
        self.width, self.heads, self.layers = width, heads, layers
        self.projection = nn.Linear(feature_dim, width)
        self.cls = nn.Parameter(torch.randn(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=(4 if protocol != 'legacy' else 2)*width,
                                          dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        if protocol == 'legacy':
            self.scorer = nn.Sequential(nn.Linear(width+3, width), nn.ReLU(), nn.Linear(width, 1))
        else:
            self.scorer = nn.Sequential(nn.Linear(width, 2))
            with torch.no_grad():
                self.scorer[0].bias.copy_(torch.tensor([-.69314718056, .69314718056]))
            if protocol == 'inshape_attention_coverage_v1':
                # Equivalent to concatenating 32 attention conditions to CLS
                # before the same two-logit linear scorer. Zero extra weights
                # preserve baseline initial scores and the global RNG state.
                with torch.random.fork_rng(devices=[]):
                    self.attention_scorer = nn.Linear(32, 2, bias=False)
                nn.init.zeros_(self.attention_scorer.weight)
        if protocol == CONTEXT_DROP_PROTOCOL:
            # Preserve the baseline initialization and start with a zero residual.
            with torch.random.fork_rng(devices=[]):
                self.segment_context = nn.MultiheadAttention(width, num_heads=1, batch_first=True)
            nn.init.zeros_(self.segment_context.out_proj.weight)
            nn.init.zeros_(self.segment_context.out_proj.bias)

    def forward(self, features, attention, segments, variant='attention', deterministic=False,
                head_count=None):
        expected_variant = ('attention_shared' if self.protocol == 'inshape_attention_coverage_v1' else 'inshape')
        if self.protocol != 'legacy' and variant != expected_variant:
            raise ValueError('Selector protocol and variant do not match')
        if variant not in ('inshape', 'attention', 'attention_shared'):
            raise ValueError('variant must be inshape or attention')
        if features.ndim != 3 or segments.shape != features.shape[:2]:
            raise ValueError('Expected features [B,T,C] and segments [B,T]')
        if attention is None:
            if self.protocol not in DROP_PROTOCOLS or head_count is None:
                raise ValueError('Only the drop protocol accepts content without attention')
        else:
            if attention.ndim != 3 or attention.shape[0] != features.shape[0] or attention.shape[2] != features.shape[1]:
                raise ValueError('Expected attention [B,H,T]')
            head_count = attention.size(1)
        valid = segments >= 0
        if not valid.any(-1).all():
            raise ValueError('Each sequence requires at least one segment')
        batch, time, _ = features.shape
        count = segments.max(-1).values + 1
        maximum = int(count.max())
        candidates = torch.arange(maximum, device=features.device)
        membership = (segments[:, None, :] == candidates[None, :, None]) & valid[:, None]
        lengths = membership.sum(-1)
        valid_segments = lengths > 0
        if not torch.equal(valid_segments, candidates[None] < count[:, None]):
            raise ValueError('Segment labels must be contiguous from zero')
        # Encode each variable-length segment independently, excluding padded tokens.
        groups = [features[b, membership[b, k]] for b in range(batch) for k in range(int(count[b]))]
        padded = nn.utils.rnn.pad_sequence(groups, batch_first=True)
        sizes = torch.tensor([len(group) for group in groups], device=features.device)
        pad = torch.arange(padded.size(1), device=features.device)[None] >= sizes[:, None]
        tokens = torch.cat((self.cls.expand(padded.size(0), -1, -1), self.projection(padded)), 1)
        pad = torch.cat((torch.zeros_like(pad[:, :1]), pad), 1)
        content = self.encoder(tokens, src_key_padding_mask=pad)[:, 0]
        embeddings = features.new_zeros(batch, maximum, content.size(-1))
        embeddings[valid_segments] = content
        if self.protocol == CONTEXT_DROP_PROTOCOL:
            context, _ = self.segment_context(embeddings, embeddings, embeddings,
                                              key_padding_mask=~valid_segments, need_weights=False)
            embeddings = embeddings + context.masked_fill(~valid_segments[..., None], 0.)
        if self.protocol in ('legacy', 'inshape_attention_coverage_v1'):
            mass = torch.einsum('bht,bkt->bhk', attention.detach(), membership.to(attention.dtype))
            fraction = lengths.to(features.dtype) / valid.sum(-1, keepdim=True)
            enrichment = mass / fraction[:, None].clamp_min(1e-12)
        if self.protocol == 'legacy':
            if variant == 'inshape':
                mass, enrichment = torch.zeros_like(mass), torch.zeros_like(enrichment)
            conditions = torch.stack((mass, enrichment, fraction[:, None].expand(-1, head_count, -1)), -1)
            inputs = torch.cat((embeddings[:, None].expand(-1, head_count, -1, -1), conditions), -1)
            logit = self.scorer(inputs).squeeze(-1)
            logits = torch.stack((torch.zeros_like(logit), logit), -1)
        else:
            shared_logits = self.scorer(embeddings)
            if self.protocol == 'inshape_attention_coverage_v1':
                if head_count != 16:
                    raise ValueError('Shared attention conditioning requires 16 LTAE heads')
                attention_inputs = torch.cat((mass.transpose(1, 2), enrichment.transpose(1, 2)), -1)
                shared_logits = shared_logits + self.attention_scorer(attention_inputs)
            logits = shared_logits[:, None]
        if deterministic:
            probability = logits.softmax(-1)[..., 1]
            gate = (probability >= .5).to(features.dtype)
        else:
            hard, soft = _ReinMax.apply(logits, 1.3 if self.protocol != 'legacy' else 1.)
            gate, probability = hard[..., 1], soft[..., 1]
        if self.protocol != 'legacy':
            gate = gate.expand(-1, head_count, -1)
            probability = probability.expand(-1, head_count, -1)
        probability = probability * valid_segments[:, None]
        gate = gate * valid_segments[:, None]
        unrepaired_points = torch.einsum('bhk,bkt->bht', gate, membership.to(gate.dtype))
        raw_coverage = (unrepaired_points.sum(-1)/valid.sum(-1)[:, None]).mean()
        sparse = (probability.sum(-1).mean() if self.protocol == 'legacy'
                  else (2*raw_coverage.detach()-1)*raw_coverage)
        empty = gate.detach().sum(-1) == 0
        best = probability.detach().masked_fill(~valid_segments[:, None], -1).argmax(-1)
        repair = torch.zeros_like(gate).scatter_(-1, best[..., None], 1.)
        # Constant forward repair, retaining the original ReinMax gradient.
        gate = gate + repair * empty[..., None]
        point_mask = torch.einsum('bhk,bkt->bht', gate, membership.to(gate.dtype))
        return dict(gate=gate, probabilities=probability, point_mask=point_mask,
                    sparse_loss=selector_only_loss(sparse, self.parameters()), segments=segments,
                    raw_coverage=raw_coverage.detach(), sparse_coefficient=(2*raw_coverage.detach()-1),
                    valid_segments=valid_segments, valid_time=valid,
                    fallback_rate=empty.float().mean(), repaired=empty,
                    selected_count=gate.detach().sum(-1), candidate_count=count,
                    coverage=point_mask.detach().sum(-1)/valid.sum(-1)[:, None])
