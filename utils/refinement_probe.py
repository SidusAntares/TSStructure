"""Measure whether fixed class reference windows are retained on Source validation parcels."""

from collections import Counter, defaultdict
import hashlib
import json

import numpy as np
import torch

from utils.source_segment_refinement import (
    SETTINGS, calendar_batch, calendar_cuts, calendar_labels, dtw, merge_cuts, project,
    prototype_windows, ranges_for, selector_state, split_cuts,
)
from utils.segment_structure import segment_pelt
from utils.guided_segment_refinement import guided_merge, match_shapelets, split_matches


def reference_window(dates, shape, prototype, min_size):
    """Search the full sequence independently of the current segmentation."""
    n = len(prototype['shape'])
    candidates = [(dtw(shape[a:b], prototype['shape']), a, b)
                  for a, b in prototype_windows(dates, 0, len(dates), prototype)
                  if max(min_size, .5*n) <= b-a <= 2*n]
    if candidates:
        distance, a, b = min(candidates)
        if distance <= SETTINGS['dtw_limit']:
            return a, b
    return None


def complete_window(window, selected):
    a, b = window
    for left, right in selected:
        overlap = max(0, min(b, right)-max(a, left))
        union = b-a + right-left - overlap
        if overlap >= .9*(b-a) and overlap >= .5*union:
            return True
    return False


def matched_pair_counts(groups):
    return sum(n*(n-1)//2 for n in groups.values())


class RefinementProbe:
    def __init__(self, dataset, config, folder):
        self.dataset, self.config, self.folder = dataset, config, folder
        labels = dataset.get_labels()
        rng = np.random.default_rng(config.seed + 127)
        self.indices = [i for y in np.unique(labels)
                        for i in rng.permutation(np.flatnonzero(labels == y))[:24]]
        self.batches, self.cuts, self.targets, self.protected = [], {}, {}, {}
        self.guided = getattr(config, 'shapelet_guided_merge', False)
        self.settings = dict(SETTINGS, **getattr(config, 'refinement_options', {}))
        self.matches = {}
        self.history, self.previous = [], {}
        warmup = getattr(config, 'refinement_epoch_plan', (10,))[0]
        self.phase, self.round_index, self.edit_epoch = 'initial', 0, warmup

    @torch.no_grad()
    def initialize(self, batches, bank):
        manifest = []
        for ids, ys, dates, features in batches:
            self.batches.append((ids, ys, dates, features))
            labels = segment_pelt(features.cpu(), penalty=self.config.pelt_penalty,
                                  min_size=self.config.pelt_min_size, penalty_mode='absolute').numpy()
            projected = project(features.cpu().numpy(), bank)
            for parcel, y, day, shape, seg in zip(ids, ys, dates.cpu().numpy(), projected, labels):
                self.cuts[parcel] = calendar_cuts(day, seg)
                targets = {}
                prototypes = bank['classes'].get(y, [])
                if self.guided:
                    matches = match_shapelets(day, shape, prototypes, self.config.pelt_min_size, self.settings)
                    self.matches[parcel] = matches
                    targets = {m.prototype: (m.start, m.stop) for m in matches}
                else:
                    for p in prototypes:
                        window = reference_window(day, shape, p, self.config.pelt_min_size)
                        if window is not None:
                            targets[p['id']] = window
                self.targets[parcel] = targets
                manifest.append((parcel, y, sorted(targets.items())))
        self.reference_sha256 = hashlib.sha256(json.dumps(sorted(manifest)).encode()).hexdigest()

    def selections(self, model, selector):
        for ids, ys, dates, features in self.batches:
            cuts = [self.cuts[p] for p in ids]
            labels = calendar_batch(dates, torch.ones_like(dates, dtype=torch.bool), cuts)
            state = selector_state(model, selector, features, labels)
            gates = state['gate'][:, 0].bool().cpu().numpy()
            probs = state['probabilities'][:, 0].cpu().numpy()
            for i, (parcel, y, cut) in enumerate(zip(ids, ys, cuts)):
                n = len(cut) + 1
                yield parcel, y, dates[i], features[i], cut, gates[i, :n], probs[i, :n]

    @torch.no_grad()
    def edit(self, model, selector, bank, phase, round_index, epoch):
        if phase == 'split':
            self.protected.clear()
            for ids, ys, dates, features in self.batches:
                projected = project(features.cpu().numpy(), bank)
                for parcel, y, day, shape in zip(ids, ys, dates.cpu().numpy(), projected):
                    cuts = self.cuts[parcel]
                    prototypes = bank['classes'].get(y, [])
                    if self.guided:
                        if round_index > 1:
                            self.matches[parcel] = match_shapelets(
                                day, shape, prototypes, self.config.pelt_min_size, self.settings)
                        added = split_matches(day, cuts, self.matches[parcel],
                                              self.config.pelt_min_size, self.settings)
                    else:
                        added = split_cuts(day, shape, cuts, prototypes, self.config.pelt_min_size)
                    self.cuts[parcel] = np.sort(np.r_[cuts, added])
                    self.protected[parcel] = added
        else:
            for parcel, y, dates, features, cuts, gates, probs in self.selections(model, selector):
                if self.guided:
                    removed, _ = guided_merge(model, features, dates, y, cuts, gates, self.protected[parcel],
                                              self.matches[parcel], self.config.focal_loss_gamma, self.settings)
                else:
                    removed = merge_cuts(model, features, dates, y, cuts, gates, probs,
                                         self.protected[parcel], self.config.focal_loss_gamma)
                self.cuts[parcel] = cuts[~np.isin(cuts, removed)]
        self.phase, self.round_index, self.edit_epoch = phase, round_index, epoch
        self.previous.clear()
        self.record(model, selector, epoch)

    @torch.no_grad()
    def record(self, model, selector, epoch):
        model.eval()
        selector.eval()
        targets, complete = defaultdict(Counter), defaultdict(Counter)
        parcels, parcels_with_targets, feasible = Counter(), Counter(), Counter()
        kept, stability = [], []
        for parcel, y, dates, _, cuts, gates, _ in self.selections(model, selector):
            day = dates.cpu().numpy()
            candidates = ranges_for(day, cuts)
            selected = [span for span, gate in zip(candidates, gates) if gate]
            points = gates[calendar_labels(day, cuts)]
            kept.append(float(points.mean()))
            if parcel in self.previous:
                previous = self.previous[parcel]
                union = np.count_nonzero(points | previous)
                stability.append(np.count_nonzero(points & previous)/union if union else 1.)
            self.previous[parcel] = points
            parcels[y] += 1
            parcels_with_targets[y] += int(bool(self.targets[parcel]))
            for prototype, window in self.targets[parcel].items():
                targets[y][prototype] += 1
                feasible[y] += int(complete_window(window, candidates))
                complete[y][prototype] += int(complete_window(window, selected))
        classes = {}
        for y in sorted(parcels):
            classes[y] = dict(parcels=parcels[y], parcels_with_targets=parcels_with_targets[y],
                              targets=sum(targets[y].values()), feasible=feasible[y],
                              complete=sum(complete[y].values()), target_pairs=matched_pair_counts(targets[y]),
                              complete_pairs=matched_pair_counts(complete[y]))
        totals = {key: sum(row[key] for row in classes.values())
                  for key in ('parcels', 'parcels_with_targets', 'targets', 'feasible', 'complete',
                              'target_pairs', 'complete_pairs')}
        record = dict(epoch=epoch, round=self.round_index, phase=self.phase,
                      epochs_since_edit=epoch-self.edit_epoch, reference_sha256=self.reference_sha256,
                      **totals, by_class=classes,
                      mean_kept_observation_fraction=float(np.mean(kept)) if kept else None,
                      previous_selection_iou=float(np.mean(stability)) if stability else None,
                      candidate_complete_fraction=totals['feasible']/totals['targets'] if totals['targets'] else None,
                      complete_fraction=totals['complete']/totals['targets'] if totals['targets'] else None,
                      complete_pair_fraction=totals['complete_pairs']/totals['target_pairs']
                      if totals['target_pairs'] else None)
        self.history.append(record)
        (self.folder/'validation_probe.json').write_text(json.dumps(self.history, indent=2), encoding='utf-8')
        print('REFINEMENT_PROBE|' + json.dumps({k: v for k, v in record.items() if k != 'by_class'}), flush=True)
