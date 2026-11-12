"""Calendar-time reference boundaries and source-only class feedback."""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.segment_structure import segment_pelt


SETTINGS = dict(update_epoch=10, reference_per_class=128, projection_dim=8,
                region_days=60, min_support=.20, min_support_count=5, top_k=2,
                dtw_limit=1., boundary_slack=7, split_budget=2, merge_budget=1,
                merge_min_probability=.60, merge_max_points=24, merge_max_span=90,
                merge_max_gap=24, merge_coverage_cost=.10, merge_tolerance=.01)


def calendar_cuts(dates, segments):
    breaks = np.flatnonzero(np.diff(segments))
    return (dates[breaks] + dates[breaks + 1]) / 2


def calendar_labels(dates, cuts):
    ids = np.searchsorted(cuts, dates, side='right')
    return np.r_[0, np.cumsum(np.diff(ids) != 0)]


def calendar_batch(dates, valid, cuts):
    rows = np.full(dates.shape, -1, dtype=np.int64)
    for row, (day, keep, boundaries) in enumerate(zip(dates.cpu().numpy(), valid.cpu().numpy(), cuts)):
        rows[row, keep] = calendar_labels(day[keep], boundaries)
    return torch.as_tensor(rows, device=dates.device)


def ranges_for(dates, cuts):
    starts = np.r_[0, np.searchsorted(dates, cuts, side='right'), len(dates)]
    return list(zip(starts[:-1], starts[1:]))


def cut_gap(dates, index):
    return float((dates[index - 1] + dates[index]) / 2)


def dtw(a, b):
    """RMS feature distance along a length-normalized Sakoe–Chiba band."""
    n, m = len(a), len(b)
    costs = ((a[:, None] - b[None]) ** 2).mean(-1)
    dp = np.full((n + 1, m + 1), np.inf)
    steps = np.zeros((n + 1, m + 1), dtype=int)
    dp[0, 0] = 0.
    for i in range(1, n + 1):
        center = 1 + (i - 1) * (m - 1) / max(1, n - 1)
        radius = max(2, int(np.ceil(.25 * m)), int(np.ceil(m / n)))
        for j in range(max(1, int(center) - radius), min(m, int(center) + radius + 1) + 1):
            parent = min(((i - 1, j - 1), (i - 1, j), (i, j - 1)), key=dp.__getitem__)
            dp[i, j] = dp[parent] + costs[i - 1, j - 1]
            steps[i, j] = steps[parent] + 1
    return float(np.sqrt(dp[n, m] / max(1, steps[n, m])))


def region(left, right):
    return int((left + right) / 2 // SETTINGS['region_days'])


def match_segment(shape, left, right, prototypes):
    matches = []
    for p in prototypes:
        if abs(region(left, right) - p['region']) > 1:
            continue
        if not all(bounds[0] - 21 <= day <= bounds[2] + 21
                   for day, bounds in zip((left, right), p['bounds'])):
            continue
        if max(len(shape), len(p['shape'])) > 2 * min(len(shape), len(p['shape'])):
            continue
        distance = dtw(shape, p['shape'])
        if distance <= SETTINGS['dtw_limit']:
            matches.append((distance, p['id']))
    return min(matches)[1] if matches else None


def cluster_selected(records, eligible_dates, previous):
    """Assign to bounded nearby medoids and recount parcel-level support each round."""
    clusters = {p['id']: dict(p, members=[]) for p in previous}
    next_id = max(clusters, default=-1) + 1
    for record in records:
        left, right = record['bounds']
        area = region(left, right)
        match = match_segment(record['shape'], left, right, clusters.values())
        if match is None:
            if sum(p['region'] == area for p in clusters.values()) >= 2 * SETTINGS['top_k']:
                continue
            match = next_id
            next_id += 1
            clusters[match] = dict(id=match, region=area, shape=record['shape'],
                                   bounds=[[left]*3, [right]*3], members=[])
        clusters[match]['members'].append(record)
    ranked = defaultdict(list)
    for cluster in clusters.values():
        members = cluster.pop('members')
        if not members:
            continue
        bounds = np.quantile([m['bounds'] for m in members], [.10, .50, .90], axis=0).T
        left, right = bounds[:, 1]
        eligible = {parcel for parcel, dates in eligible_dates.items()
                    if np.count_nonzero((dates >= left) & (dates <= right)) >= 3}
        supporting = sorted(eligible.intersection(m['parcel'] for m in members))
        frequency = len(supporting) / max(1, len(eligible))
        if len(supporting) < SETTINGS['min_support_count'] or frequency < SETTINGS['min_support']:
            continue
        # Approximate the medoid within a bounded pool of actual cluster members.
        panel = [members[i] for i in np.linspace(0, len(members)-1, min(12, len(members)), dtype=int)]
        costs = np.zeros(len(panel))
        for i, a in enumerate(panel):
            for j, b in enumerate(panel[:i]):
                costs[i] += (distance := dtw(a['shape'], b['shape']))
                costs[j] += distance
        representative = panel[int(costs.argmin())]
        cluster.update(shape=representative['shape'], medoid_parcel=representative['parcel'],
                       medoid_bounds=representative['bounds'], bounds=bounds.tolist(),
                       frequency=frequency, eligible=len(eligible), supporting=supporting)
        ranked[cluster['region']].append(cluster)
    return [p for group in ranked.values()
            for p in sorted(group, key=lambda p: (-p['frequency'], p['id']))[:SETTINGS['top_k']]]


def split_cuts(dates, shape, cuts, prototypes, min_size):
    candidates = []
    for p in prototypes:
        (ll, lm, lh), (rl, rm, rh) = p['bounds']
        existing = match_segment_for_prototype(dates, shape, cuts, p)
        if existing:
            continue
        for start, stop in ranges_for(dates, cuts):
            if stop - start < 2 * min_size or dates[start] > lh or dates[stop-1] < rl:
                continue
            slack = SETTINGS['boundary_slack']
            left = [i for i in range(start, stop) if ll - slack <= dates[i] <= lh + slack
                    and (i == start or dates[i] > dates[i-1])]
            right = [i for i in range(start + 1, stop + 1) if rl - slack <= dates[i-1] <= rh + slack
                     and (i == stop or dates[i] > dates[i-1])]
            left = sorted(left, key=lambda i: abs(dates[i] - lm))[:3]
            right = sorted(right, key=lambda i: abs(dates[i-1] - rm))[:3]
            for a in left:
                for b in right:
                    if b - a < min_size or (a > start and a-start < min_size) or (b < stop and stop-b < min_size):
                        continue
                    if not .5 * len(p['shape']) <= b-a <= 2 * len(p['shape']):
                        continue
                    additions = [cut_gap(dates, i) for i in (a, b) if start < i < stop]
                    if not additions or any(np.any(np.abs(cuts - c) <= SETTINGS['boundary_slack']) for c in additions):
                        continue
                    distance = dtw(shape[a:b], p['shape'])
                    if distance <= SETTINGS['dtw_limit']:
                        candidates.append((-p['frequency'], distance, additions))
    inserted = []
    for _, _, additions in sorted(candidates):
        if len(inserted) + len(additions) > SETTINGS['split_budget']:
            continue
        if any(abs(a-b) <= SETTINGS['boundary_slack'] for a in additions for b in inserted):
            continue
        updated = np.unique(np.r_[cuts, inserted, additions])
        if all(b-a >= min_size for a, b in ranges_for(dates, updated)):
            inserted.extend(additions)
    return inserted


def match_segment_for_prototype(dates, shape, cuts, prototype):
    slack = SETTINGS['boundary_slack']
    for a, b in ranges_for(dates, cuts):
        left, right = float(dates[a]), float(dates[b-1])
        if all(lo - slack <= day <= hi + slack
               for day, (lo, _, hi) in zip((left, right), prototype['bounds'])):
            if match_segment(shape[a:b], left, right, [prototype]) is not None:
                return True
    return False


@torch.no_grad()
def merge_costs(model, features, dates, y, masks, gamma):
    sequences = torch.nn.utils.rnn.pad_sequence([features[mask] for mask in masks], batch_first=True)
    positions = torch.nn.utils.rnn.pad_sequence([dates[mask] for mask in masks], batch_first=True)
    count = masks.sum(1)
    valid = torch.arange(positions.size(1), device=dates.device)[None] < count[:, None]
    embedded = model.temporal_encoder(sequences, positions, return_attention_scores=True, valid_time=valid)[0]
    logp = model.decoder(embedded).log_softmax(-1)[:, y]
    focal = -(1-logp.exp()).pow(gamma) * logp
    return (focal + SETTINGS['merge_coverage_cost'] * count / len(dates)).view(-1, 3).cpu().numpy()


@torch.no_grad()
def refine_parcel(model, features, positions, y, cuts, gates, probabilities, bank, min_size, gamma):
    dates = positions.cpu().numpy()
    shape = project(features.cpu().numpy(), bank)
    added = split_cuts(dates, shape, cuts, bank['classes'].get(y, []), min_size)
    removed = merge_cuts(model, features, positions, y, cuts, gates, probabilities, added, gamma)
    return np.sort(np.r_[cuts[~np.isin(cuts, removed)], added]), added, removed


def merge_cuts(model, features, positions, y, cuts, gates, probabilities, protected, gamma):
    dates = positions.cpu().numpy()
    ranges = ranges_for(dates, cuts)
    labels = calendar_labels(dates, cuts)
    points = torch.as_tensor(labels, device=features.device)
    kept = torch.as_tensor(gates[labels], device=features.device)
    pairs, masks = [], []
    for k, ((a, split), (_, b)) in enumerate(zip(ranges, ranges[1:])):
        if not gates[k:k+2].all() or (probabilities[k:k+2] < SETTINGS['merge_min_probability']).any():
            continue
        if (b-a > SETTINGS['merge_max_points'] or dates[b-1]-dates[a] > SETTINGS['merge_max_span']
                or dates[split]-dates[split-1] > SETTINGS['merge_max_gap']
                or any(dates[a] < cut < dates[b-1] for cut in protected)):
            continue
        pairs.append(k)
        masks.extend((kept & (points != k+1), kept & (points != k), kept))
    removed = []
    if masks:
        costs = merge_costs(model, features, positions, y, torch.stack(masks), gamma)
        regret = costs[:, 2] - costs.min(1)
        for j in np.argsort(regret, kind='stable')[:SETTINGS['merge_budget']]:
            if regret[j] <= SETTINGS['merge_tolerance']:
                removed.append(float(cuts[pairs[j]]))
    return removed


def project(x, bank):
    return ((x - bank['mean']) @ bank['basis'].T) / bank['scale']


class FixedViews(Dataset):
    def __init__(self, dataset, indices, seed):
        self.dataset, self.indices, self.seed = dataset, list(indices), seed
        self.parcels = dataset.get_parcel_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        previous = random.getstate()
        try:
            random.seed(self.seed + int(self.parcels[index]))
            return self.dataset[index]
        finally:
            random.setstate(previous)


def selector_state(model, selector, features, labels):
    return selector(features, None, labels, variant='inshape', deterministic=True,
                    head_count=model.temporal_encoder.attention_heads.n_head)


class SourceRefinement:
    def __init__(self, dataset, config):
        self.dataset, self.config = dataset, config
        self.folder = Path(config.output_dir) / 'refinement'
        self.folder.mkdir()
        self.initial, self.cuts, self.edits = {}, {}, {}
        self.round_history, self.protected = [], {}
        self.bank = None
        labels = dataset.get_labels()
        rng = np.random.default_rng(config.seed)
        self.panel = [i for y in np.unique(labels) for i in
                      rng.permutation(np.flatnonzero(labels == y))[:SETTINGS['reference_per_class']]]

    def views(self, model, indices):
        cfg = self.config
        loader = DataLoader(FixedViews(self.dataset, indices, cfg.seed), batch_size=48,
                            num_workers=cfg.num_workers, generator=torch.Generator().manual_seed(cfg.seed))
        for sample in loader:
            x, mask, extra = (sample[key].to(cfg.device) for key in ('pixels', 'valid_pixels', 'extra'))
            yield (sample['parcel_index'].tolist(), sample['label'].tolist(),
                   sample['positions'].to(cfg.device), model.spatial_encoder(x, mask, extra))

    @torch.no_grad()
    def initialize(self, model):
        model.spatial_encoder.eval()
        start = time.monotonic()
        print(f'SEGMENT_REFERENCE_START|parcels={len(self.dataset)}', flush=True)
        for parcels, _, dates, features in self.views(model, range(len(self.dataset))):
            seg = segment_pelt(features.cpu(), penalty=self.config.pelt_penalty,
                               min_size=self.config.pelt_min_size, penalty_mode='absolute').numpy()
            for parcel, day, labels in zip(parcels, dates.cpu().numpy(), seg):
                self.cuts[parcel] = calendar_cuts(day, labels)
        self.initial = self.cuts.copy()
        print(f'SEGMENT_REFERENCE|parcels={len(self.cuts)}|seconds={time.monotonic()-start:.1f}', flush=True)

    def labels(self, parcels, dates, valid):
        return calendar_batch(dates, valid, [self.cuts[p] for p in parcels.tolist()])

    def selections(self, model, selector, indices):
        for ids, ys, dates, features in self.views(model, indices):
            cuts = [self.cuts[p] for p in ids]
            labels = calendar_batch(dates, torch.ones_like(dates, dtype=torch.bool), cuts)
            state = selector_state(model, selector, features, labels)
            gates = state['gate'][:, 0].bool().cpu().numpy()
            probs = state['probabilities'][:, 0].cpu().numpy()
            for i, (parcel, y, boundaries) in enumerate(zip(ids, ys, cuts)):
                n = len(boundaries) + 1
                yield parcel, y, dates[i], features[i], boundaries, gates[i, :n], probs[i, :n]

    @torch.no_grad()
    def fit_bank(self, model, selector):
        panel = [(p, y, dates.cpu().numpy(), features.cpu().numpy(), cuts, gates, probs)
                 for p, y, dates, features, cuts, gates, probs in self.selections(model, selector, self.panel)]
        if self.bank is None:
            values = np.concatenate([row[3][::4] for row in panel])
            mean = values.mean(0)
            _, singular, basis = np.linalg.svd(values - mean, full_matrices=False)
            width = SETTINGS['projection_dim']
            self.bank = dict(mean=mean, basis=basis[:width],
                             scale=np.maximum(singular[:width] / np.sqrt(len(values)), .01), classes={})
        records, eligible = defaultdict(list), defaultdict(dict)
        for parcel, y, dates, features, cuts, gates, probs in panel:
            eligible[y][parcel] = dates
            projected = project(features, self.bank)
            for k, (a, b) in enumerate(ranges_for(dates, cuts)):
                if gates[k] and probs[k] >= .5:
                    records[y].append(dict(parcel=parcel, shape=projected[a:b],
                                           bounds=[float(dates[a]), float(dates[b-1])]))
        previous = self.bank['classes']
        self.bank['classes'] = {y: cluster_selected(records[y], dates, previous.get(y, []))
                                for y, dates in eligible.items()}

    @torch.no_grad()
    def update(self, model, selector):
        start = time.monotonic()
        model.eval()
        selector.eval()
        print(f'SEGMENT_REFINE_START|parcels={len(self.dataset)}|reference_panel={len(self.panel)}', flush=True)
        self.fit_bank(model, selector)
        # This frozen snapshot replays the same refinement on held-out parcels for plots.
        torch.save(dict(state_dict=model.state_dict(), selector_state_dict=selector.state_dict(),
                        bank=serializable(self.bank), settings=SETTINGS), self.folder / 'update_epoch10.pt')
        counts = dict(split=0, merge=0, changed=0)
        for parcel, y, dates, features, cuts, gates, probs in self.selections(model, selector, range(len(self.dataset))):
            updated, added, removed = refine_parcel(model, features, dates, y, cuts, gates, probs, self.bank,
                self.config.pelt_min_size, self.config.focal_loss_gamma)
            self.cuts[parcel] = updated
            if added or removed:
                self.edits[parcel] = dict(split=added, merge=removed)
                counts['split'] += len(added)
                counts['merge'] += len(removed)
                counts['changed'] += 1
        counts.update(seconds=time.monotonic()-start, parcels=len(self.cuts),
                      prototypes=sum(map(len, self.bank['classes'].values())),
                      initial_candidate_mean=float(np.mean([len(c)+1 for c in self.initial.values()])),
                      refined_candidate_mean=float(np.mean([len(c)+1 for c in self.cuts.values()])))
        (self.folder / 'update_summary.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
        print('SEGMENT_REFINE|' + json.dumps(counts), flush=True)

    @torch.no_grad()
    def refine_round_step(self, model, selector, epoch):
        """Apply a scheduled edit after the model has trained on the preceding cuts."""
        offset = epoch - SETTINGS['update_epoch']
        adapt = self.config.refinement_adapt_epochs
        if offset < 0 or offset % adapt or epoch >= self.config.epochs:
            return
        round_index = offset // (2 * adapt) + 1
        phase = 'merge' if offset % (2 * adapt) else 'split'
        model.eval()
        selector.eval()
        start = time.monotonic()
        print(f'SEGMENT_REFINE_PHASE_START|round={round_index}|phase={phase}|epoch={epoch}', flush=True)
        changes = defaultdict(lambda: dict(parcels=0, cuts=0))
        if phase == 'split':
            self.fit_bank(model, selector)
            self.protected = {}
            for parcels, labels, dates, features in self.views(model, range(len(self.dataset))):
                projected = project(features.cpu().numpy(), self.bank)
                for parcel, y, day, shape in zip(parcels, labels, dates.cpu().numpy(), projected):
                    cuts = self.cuts[parcel]
                    added = split_cuts(day, shape, cuts, self.bank['classes'].get(y, []),
                                       self.config.pelt_min_size)
                    if added:
                        self.cuts[parcel] = np.sort(np.r_[cuts, added])
                        self.protected[parcel] = added
                        changes[y]['parcels'] += 1
                        changes[y]['cuts'] += len(added)
        else:
            # Selections are computed on the current post-Split cuts with the updated Selector.
            for parcel, y, dates, features, cuts, gates, probs in self.selections(
                    model, selector, range(len(self.dataset))):
                removed = merge_cuts(model, features, dates, y, cuts, gates, probs,
                                     self.protected.get(parcel, ()), self.config.focal_loss_gamma)
                if removed:
                    self.cuts[parcel] = cuts[~np.isin(cuts, removed)]
                    changes[y]['parcels'] += 1
                    changes[y]['cuts'] += len(removed)
        record = dict(round=round_index, phase=phase, epoch=epoch,
                      parcels_changed=sum(item['parcels'] for item in changes.values()),
                      cuts_changed=sum(item['cuts'] for item in changes.values()),
                      mean_segments=float(np.mean([len(cuts)+1 for cuts in self.cuts.values()])),
                      prototype_counts={y: len(p) for y, p in self.bank['classes'].items()},
                      by_class=dict(changes), seconds=round(time.monotonic()-start, 2))
        self.round_history.append(record)
        (self.folder / 'round_history.json').write_text(json.dumps(self.round_history, indent=2), encoding='utf-8')
        print('SEGMENT_REFINE_PHASE|' + json.dumps(record), flush=True)

    @torch.no_grad()
    def finish(self, model, selector):
        model.eval()
        selector.eval()
        self.fit_bank(model, selector)
        (self.folder / 'prototypes.json').write_text(
            json.dumps(dict(settings=SETTINGS, **serializable(self.bank))), encoding='utf-8')
        with (self.folder / 'parcels.jsonl').open('w', encoding='utf-8') as out:
            for parcel, y, dates, features, cuts, gates, probs in self.selections(model, selector, range(len(self.dataset))):
                dates = dates.cpu().numpy()
                shape = project(features.cpu().numpy(), self.bank)
                segments = [dict(id=k, first_day=int(dates[a]), last_day=int(dates[b-1]),
                                 probability=float(probs[k]), selected=bool(gates[k]),
                                 cluster=match_segment(shape[a:b], float(dates[a]), float(dates[b-1]),
                                                       self.bank['classes'].get(y, [])))
                            for k, (a, b) in enumerate(ranges_for(dates, cuts))]
                initial = self.initial[parcel]
                edits = self.edits.get(parcel, {})
                if self.round_history:
                    edits = dict(split=np.setdiff1d(cuts, initial).tolist(),
                                 merge=np.setdiff1d(initial, cuts).tolist())
                out.write(json.dumps(dict(parcel=parcel, label=y, initial=initial.tolist(),
                                          final=cuts.tolist(), edits=edits,
                                          segments=segments)) + '\n')


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serializable(v) for v in value]
    return value


def restore_bank(bank):
    for key in ('mean', 'basis', 'scale'):
        bank[key] = np.asarray(bank[key], dtype=np.float32)
    bank['classes'] = {int(key): value for key, value in bank['classes'].items()}
    for prototypes in bank['classes'].values():
        for p in prototypes:
            p['shape'] = np.asarray(p['shape'], dtype=np.float32)
    return bank
