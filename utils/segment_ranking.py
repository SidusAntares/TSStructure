"""Read-only candidate ranking checks; masks here never enter training losses."""
import torch

from utils.soft_top50 import keep_head_embeddings


def _focal(logits, label, gamma):
    log_probability = logits.log_softmax(-1)[:, int(label)]
    return -(1-log_probability.exp()).pow(gamma)*log_probability


def _spearman(left, right):
    def ranks(values):
        order = values.argsort()
        result = torch.empty_like(values)
        for value in values.unique():
            positions = (values[order] == value).nonzero().flatten()
            result[order[positions]] = positions.to(values.dtype).mean()
        return result-result.mean()
    left, right = ranks(left.double()), ranks(right.double())
    denominator = left.norm()*right.norm()
    return float((left*right).sum()/denominator) if float(denominator) > 0 else None


@torch.no_grad()
def batch_ranking_diagnostic(model, info, labels, gamma, random_repeats=10):
    """Compare one/half retained candidates against ten random subsets.

    The same subset is used across LTAE heads; scores are averaged over heads
    (identical scores for the content-only INSHAPE variant). Single-candidate
    sequences are excluded because neither subset ranking nor deletion is
    identifiable. Removal contributions are conditional on all other segments.
    """
    if model.training:
        raise ValueError('Ranking diagnostic requires eval mode')
    if random_repeats < 1:
        raise ValueError('random_repeats must be positive')
    rows, deletions = [], []
    for b, candidate_count in enumerate(info['candidate_count'].tolist()):
        count = int(candidate_count)
        if count < 2:
            continue
        scores = info['probabilities'][b, :, :count].mean(0)
        order = scores.argsort(descending=True, stable=True)
        # One batched backend call per sample: full, every deletion, then
        # ranked and matched-count random subsets for each diagnostic budget.
        masks = [torch.ones_like(scores)]
        for k in range(count):
            gate = torch.ones_like(scores)
            gate[k] = 0
            masks.append(gate)
        budgets = []
        for name, kept in [('one', 1), ('half', min(count-1, (count+1)//2))]:
            start = len(masks)
            ranked = torch.zeros_like(scores)
            ranked[order[:kept]] = 1
            masks.append(ranked)
            for _ in range(random_repeats):
                random_gate = torch.zeros_like(scores)
                random_gate[torch.randperm(count, device=scores.device)[:kept]] = 1
                masks.append(random_gate)
            budgets.append((name, kept, start))
        masks = torch.stack(masks)
        segments = info['segments'][b]
        observed = segments >= 0
        point_masks = masks[:, segments.clamp_min(0)]*observed
        attention = info['attention'][b:b+1].expand(len(masks), -1, -1)
        values = info['values'][b:b+1].expand(len(masks), -1, -1, -1)
        pooled, _, _ = keep_head_embeddings(attention, values,
            point_masks[:, None].expand_as(attention))
        logits = model.decoder(model.temporal_encoder.dropout(model.temporal_encoder.mlp(pooled)))
        losses = _focal(logits, labels[b], gamma)
        correct = (logits.argmax(-1) == labels[b]).float()
        coverage = point_masks.sum(-1)/observed.sum()
        delta = losses[1:count+1]-losses[0]
        correlation = _spearman(scores, delta)
        for k in range(count):
            deletions.append(dict(probe_index=b, segment=k, candidate_count=count,
                probability=float(scores[k]), deletion_loss_delta=float(delta[k]),
                full_loss=float(losses[0]), removed_loss=float(losses[k+1])))
        for name, kept, start in budgets:
            random_losses = losses[start+1:start+1+random_repeats]
            rows.append(dict(probe_index=b, budget=name, candidate_count=count, kept_segments=kept,
                selector_loss=float(losses[start]), random_loss_mean=float(random_losses.mean()),
                random_loss_std=float(random_losses.std(unbiased=False)),
                loss_gain_vs_random=float(random_losses.mean()-losses[start]),
                selector_accuracy=float(correct[start]),
                random_accuracy_mean=float(correct[start+1:start+1+random_repeats].mean()),
                selector_coverage=float(coverage[start]),
                random_coverage_mean=float(coverage[start+1:start+1+random_repeats].mean()),
                deletion_spearman=correlation, random_repeats=random_repeats))
    return rows, deletions


def ranking_summary(rows):
    result = dict(ranking_identifiable_samples=len({row['probe_index'] for row in rows}),
                  ranking_control='common subset across heads; matched candidate count, not duration')
    for budget in ('one', 'half'):
        subset = [row for row in rows if row['budget'] == budget]
        for field in ('selector_loss', 'random_loss_mean', 'loss_gain_vs_random',
                      'selector_accuracy', 'random_accuracy_mean',
                      'selector_coverage', 'random_coverage_mean', 'deletion_spearman'):
            values = [row[field] for row in subset if row[field] is not None]
            result[f'ranking_{budget}_{field}'] = sum(values)/len(values) if values else None
    return result
