"""Frozen epoch5 INSHAPE checkpoint: separate content ranking from length preference."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.segment_ranking import _focal, _spearman
from utils.soft_top50 import keep_head_embeddings

TASKS = dict(AT1_DK1='austria/33UVP/2017', FR1_FR2='france/30TXT/2017',
             FR2_DK1='france/31TCJ/2017', DK1_AT1='denmark/32VNH/2017')


@torch.no_grad()
def batch_length_controls(model, info, labels, gamma):
    """Enumerate same-count subsets; exact coverage matches exclude Selector itself."""
    if model.training:
        raise ValueError('Frozen length controls require eval mode')
    rows, deletions = [], []
    for b, count in enumerate(info['candidate_count'].tolist()):
        count = int(count)
        if count < 2:
            continue
        if count > 10:
            raise ValueError('Expected at most 10 candidates from 30 observations/min_size=3')
        segments = info['segments'][b]
        observed = segments >= 0
        lengths = torch.bincount(segments[observed], minlength=count)
        scores = info['probabilities'][b, :, :count].mean(0)
        ranked = scores.argsort(descending=True, stable=True).tolist()
        longest = lengths.argsort(descending=True, stable=True).tolist()
        masks = [torch.ones_like(scores)]
        for k in range(count):
            mask = torch.ones_like(scores)
            mask[k] = 0
            masks.append(mask)
        budgets = []
        for name, kept in [('one', 1), ('half', min(count-1, (count+1)//2))]:
            combinations = list(itertools.combinations(range(count), kept))
            start = len(masks)
            selected = tuple(sorted(ranked[:kept]))
            by_length = tuple(sorted(longest[:kept]))
            for subset in combinations:
                mask = torch.zeros_like(scores)
                mask[list(subset)] = 1
                masks.append(mask)
            budgets.append((name, kept, combinations, start,
                            combinations.index(selected), combinations.index(by_length)))
        masks = torch.stack(masks)
        points = masks[:, segments.clamp_min(0)]*observed
        attention = info['attention'][b:b+1].expand(len(masks), -1, -1)
        values = info['values'][b:b+1].expand(len(masks), -1, -1, -1)
        pooled, _, _ = keep_head_embeddings(attention, values, points[:, None].expand_as(attention))
        logits = model.decoder(model.temporal_encoder.dropout(model.temporal_encoder.mlp(pooled)))
        losses, correct = _focal(logits, labels[b], gamma), (logits.argmax(-1) == labels[b]).float()
        observations = points.sum(-1).long()
        delta = losses[1:count+1]-losses[0]
        for k in range(count):
            deletions.append(dict(probe_index=b, segment=k, observation_count=int(lengths[k]),
                probability=float(scores[k]), deletion_loss_delta=float(delta[k])))
        for name, kept, combinations, start, selected, by_length in budgets:
            own, long = start+selected, start+by_length
            alternatives = [start+j for j in range(len(combinations)) if j != selected]
            matched = [j for j in alternatives if observations[j] == observations[own]]
            rows.append(dict(probe_index=b, budget=name, candidate_count=count, kept_segments=kept,
                selector_observations=int(observations[own]), longest_observations=int(observations[long]),
                matched_observations=int(observations[own]) if matched else None,
                selector_is_longest=selected == by_length, exact_match_alternatives=len(matched),
                selector_loss=float(losses[own]), longest_loss=float(losses[long]),
                random_same_count_loss=float(losses[alternatives].mean()),
                matched_random_loss=float(losses[matched].mean()) if matched else None,
                gain_vs_longest=float(losses[long]-losses[own]),
                gain_vs_matched_random=float(losses[matched].mean()-losses[own]) if matched else None,
                selector_accuracy=float(correct[own]), longest_accuracy=float(correct[long]),
                matched_random_accuracy=float(correct[matched].mean()) if matched else None,
                probability_length_spearman=_spearman(scores, lengths.float()),
                probability_contribution_spearman=_spearman(scores, delta)))
    return rows, deletions


def summarize(rows):
    result = {}
    fields = ('selector_loss', 'longest_loss', 'random_same_count_loss', 'matched_random_loss',
        'gain_vs_longest', 'gain_vs_matched_random', 'selector_accuracy', 'longest_accuracy',
        'matched_random_accuracy', 'selector_is_longest', 'probability_length_spearman',
        'probability_contribution_spearman')
    for budget in ('one', 'half'):
        subset = [row for row in rows if row['budget'] == budget]
        result[budget] = dict(samples=len(subset), exact_matched_samples=sum(row['exact_match_alternatives'] > 0 for row in subset))
        for field in fields:
            values = [row[field] for row in subset if row[field] is not None]
            result[budget][field] = sum(values)/len(values) if values else None
        # Comparisons to coverage-matched controls must use the same eligible cohort.
        eligible = [row for row in subset if row['exact_match_alternatives'] > 0]
        for field in ('selector_loss', 'selector_accuracy'):
            result[budget]['matched_cohort_'+field] = (sum(row[field] for row in eligible)/len(eligible) if eligible else None)
    return result


@torch.no_grad()
def analyze(args):
    from dataset import PixelSetData
    from torchvision.transforms import Compose
    from transforms import RandomSamplePixels, RandomSampleTimeSteps, Normalize, ToTensor
    from utils.segment_smoke import FixedProbeTransform, preserve_rng_and_modes, _forward
    from utils.segment_training import new_models, write_history
    root = Path(args.experiment_root)
    leaf = root/'inshape'/args.task
    output = root/'diagnostics/length_control'/args.task
    if output.exists():
        raise FileExistsError(f'Diagnostic already exists: {output}')
    checkpoint = leaf/'fold_0/smoke_checkpoint.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    packet = torch.load(checkpoint, map_location=args.device, weights_only=False)
    if (packet.get('role') != 'source_structure_smoke' or packet.get('status') != 'completed'
        or packet.get('completed_smoke_epochs') != 5 or packet.get('variant') != 'inshape'
        or packet.get('classification_path') != 'keep_only_unified'):
        raise ValueError('Requires the completed epoch5 Keep-only INSHAPE smoke checkpoint')
    config = SimpleNamespace(**json.loads((leaf/'resolved_config.json').read_text(encoding='utf-8')))
    probe = json.loads((leaf/'probe_parcel_ids.json').read_text(encoding='utf-8'))
    splits = json.loads((leaf/'split_indices.json').read_text(encoding='utf-8'))
    ids = set(probe['parcel_ids'])
    if (config.source != TASKS[args.task] or not ids
        or probe.get('split') != 'SourceTrain only' or not ids.issubset(splits[config.source]['train'])
        or len(ids) != len(probe['parcel_ids'])
        or packet.get('original_source_checkpoint_sha256') != config.original_source_checkpoint_sha256):
        raise ValueError('Checkpoint/probe cohort does not match archived SourceTrain protocol')
    config.device, config.num_classes, config.variant = args.device, len(config.classes), 'inshape'
    model, selector = new_models(config)
    model.load_state_dict(packet['state_dict'], strict=True)
    selector.load_state_dict(packet['selector_state_dict'], strict=True)
    model.requires_grad_(False).eval()
    selector.requires_grad_(False).eval()
    transform = FixedProbeTransform(Compose([RandomSamplePixels(config.num_pixels),
        RandomSampleTimeSteps(config.seq_length), Normalize(), ToTensor()]), probe['seed'])
    data = PixelSetData(args.data_root, config.source, config.classes, transform,
        indices=ids, closed_set=True, combine_spring_and_winter=False)
    if len(data) != len(ids):
        raise ValueError('Archived probe parcels are not all present')
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, num_workers=0)
    rows, deletions, offset = [], [], 0
    with preserve_rng_and_modes(model, selector, probe['seed'], args.device):
        for sample in loader:
            _, _, info = _forward(model, selector, sample, config, deterministic=True)
            current, removed = batch_length_controls(model, info, sample['label'].to(args.device), config.focal_loss_gamma)
            for record in current+removed:
                index = record['probe_index']
                record.update(parcel_index=int(sample['parcel_index'][index]), class_id=int(sample['label'][index]))
                record['probe_index'] += offset
            rows.extend(current)
            deletions.extend(removed)
            offset += len(sample['label'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Checkpoint changed during read-only diagnostic')
    result = dict(task=args.task, checkpoint=str(checkpoint), checkpoint_sha256=digest,
        completed_smoke_epochs=5, frozen=True, no_training=True, probe_split='archived SourceTrain cohort',
        probe_samples=offset, identifiable_samples=len({row['probe_index'] for row in rows}),
        control='uniform enumeration of same-count subsets; exact observation count; identical subset excluded',
        metrics=summarize(rows))
    output.mkdir(parents=True, exist_ok=False)
    write_history(output/'budgets.csv', rows)
    write_history(output/'deletions.csv', deletions)
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('LENGTH_CONTROL_SUMMARY|'+json.dumps(result, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=tuple(TASKS))
    parser.add_argument('--experiment_root', default='outputs/timematch_segments_smoke_seed1')
    parser.add_argument('--data_root', default='/data/user/dataset/timematch_data')
    parser.add_argument('--device', default='cuda')
    analyze(parser.parse_args())


if __name__ == '__main__':
    main()
