"""Paired checkpoint diagnostic for TimeMatch's random observation removal."""

import argparse
import csv
from itertools import combinations
from pathlib import Path
import random

import numpy as np

from analysis.segment_cut_transfer import (
    candidate_edges, forward_options, transfer_segments, validate_final_drop_jobs,
)
from analysis.segment_structure_visualization import TASKS, _batch, _sample, load_job


def boundary_agreement(full, sparse, start, stop, tolerance):
    """One-to-one tolerance matching within the shared observed date span."""
    a = full[(full > start) & (full < stop)]
    b = sparse[(sparse > start) & (sparse < stop)]
    i = j = matched = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    nearest = None
    if len(a) and len(b):
        distances = np.abs(a[:, None] - b[None, :])
        nearest = float(np.concatenate((distances.min(0), distances.min(1))).mean())
    return dict(full_cuts_in_span=len(a), sparse_cuts_in_span=len(b), matched_cuts=matched,
                boundary_f1=2*matched/(len(a)+len(b)) if len(a)+len(b) else 1.,
                boundary_nearest_days=nearest)


def interval_intersection(a, b):
    return np.maximum(0., np.minimum(a[:, None, 1], b[None, :, 1])
                      - np.maximum(a[:, None, 0], b[None, :, 0]))


def interval_iou(a, b, span=None):
    if span is not None:
        a, b = (np.clip(intervals, *span) for intervals in (a, b))
    intersection = interval_intersection(a, b).sum()
    union = np.diff(a, axis=1).sum() + np.diff(b, axis=1).sum() - intersection
    return float(intersection/union) if union > 0 else None


def cosine_distance(a, b):
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(1 - np.clip(np.dot(a, b)/norm, -1, 1)) if norm > 0 else None


def retained_segment_distance(full, sparse):
    """Greedily match retained segments by interval IoU, then compare PSE means."""
    a, b = full['intervals'], sparse['intervals']
    intersection = interval_intersection(a, b)
    union = np.diff(a, axis=1) + np.diff(b, axis=1).T - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    distances = []
    matched = 0
    while iou.size and iou.max() > 0:
        i, j = np.unravel_index(iou.argmax(), iou.shape)
        value = cosine_distance(full['segment_means'][i], sparse['segment_means'][j])
        if value is not None:
            distances.append(value)
        matched += 1
        iou[i, :] = iou[:, j] = 0
    return matched, float(np.mean(distances)) if distances else None


def describe_selection(dates, features, logits, info, row, label):
    segments = info['segments'][row].cpu().numpy()
    count = int(info['candidate_count'][row])
    keep = info['gate'][row, 0, :count].cpu().numpy() >= .5
    retained = keep[segments]
    edges = candidate_edges(dates, segments)
    scores = logits[row].cpu().numpy()
    rivals = np.delete(scores, label)
    return dict(cuts=edges[1:-1], intervals=np.column_stack((edges[:-1], edges[1:]))[keep],
                segment_means=[features[segments == i].mean(0) for i in np.flatnonzero(keep)],
                kept_mean=features[retained].mean(0), prediction=int(scores.argmax()),
                true_margin=float(scores[label] - rivals.max()) if len(rivals) else None,
                retained=retained, kept_segments=int(keep.sum()), candidates=count)


def compare_selection(full, sparse, dates, indices):
    span = (dates[0], dates[-1])
    matched, segment_distance = retained_segment_distance(full, sparse)
    shared = full['retained'][indices]
    sampled = sparse['retained']
    shared_union = (shared | sampled).sum()
    return dict(kept_iou_common_span=interval_iou(full['intervals'], sparse['intervals'], span),
                kept_iou_full_span=interval_iou(full['intervals'], sparse['intervals']),
                kept_iou_sampled_dates=float((shared & sampled).sum()/shared_union) if shared_union else None,
                kept_mean_cosine_distance=cosine_distance(full['kept_mean'], sparse['kept_mean']),
                matched_kept_segments=matched, matched_segment_cosine_distance=segment_distance,
                prediction_flip=int(full['prediction'] != sparse['prediction']),
                margin_delta=(sparse['true_margin'] - full['true_margin'] if full['true_margin'] is not None else None))


def parcel_trials(model, selector, job, dataset, index, shift, args):
    import torch
    from utils.segment_structure import _classify_retained_observations, segment_pelt, structure_forward

    sample = _sample(dataset, int(index), args.seed)
    pixels, mask, positions, extra = _batch(sample, args.device)
    valid = mask[0].bool().any(-1)
    count = int(valid.sum())
    if count <= args.dates:
        return []
    label, parcel = int(sample['label']), int(sample['parcel_index'])
    options = forward_options(job)
    _, logits, info = structure_forward(model, selector, pixels.float(), mask.float(), positions.long()+shift,
                                       extra.float(), **options)
    full_features = info['features'][:, valid]
    full_positions = positions[:, valid].long()
    dates = full_positions[0].cpu().numpy()
    features = full_features[0].cpu().numpy()
    info['segments'] = info['segments'][:, valid]
    full = describe_selection(dates, features, logits, info, 0, label)
    rng = random.Random(f'{args.seed}:{parcel}')
    draws = np.array([sorted(rng.sample(range(count), args.dates)) for _ in range(args.repeats)])
    indices = torch.as_tensor(draws, device=full_features.device)
    thinned_features, thinned_positions = full_features[0, indices], full_positions[0, indices]
    new_segments = segment_pelt(thinned_features, penalty=options['penalty'], min_size=options['min_size'],
                                penalty_mode=options['penalty_mode'])
    old_segments = torch.as_tensor(np.stack([transfer_segments(dates[draw], full['cuts']) for draw in draws]),
                                   device=full_features.device)
    paired_features = torch.cat((thinned_features, thinned_features))
    paired_dates = torch.cat((thinned_positions, thinned_positions)) + shift
    paired_segments = torch.cat((new_segments, old_segments))
    thin_logits, thin_info = _classify_retained_observations(
        model, selector, paired_features, paired_dates, paired_segments, deterministic=True)

    rows, draw_intervals = [], []
    for trial, draw in enumerate(draws):
        day, feature = dates[draw], features[draw]
        recomputed = describe_selection(day, feature, thin_logits, thin_info, trial, label)
        fixed = describe_selection(day, feature, thin_logits, thin_info, trial+len(draws), label)
        row = dict(parcel=parcel, label=label, trial=trial+1, full_dates=count, sampled_dates=len(draw),
                   sampled_positions=day.tolist(), full_boundaries=full['cuts'].tolist(),
                   recomputed_boundaries=recomputed['cuts'].tolist(), fixed_boundaries=fixed['cuts'].tolist(),
                   full_kept_intervals=full['intervals'].tolist(), recomputed_kept_intervals=recomputed['intervals'].tolist(),
                   fixed_kept_intervals=fixed['intervals'].tolist())
        row.update(boundary_agreement(full['cuts'], recomputed['cuts'], day[0], day[-1], args.cut_tolerance))
        for name, trace in (('full', full), ('recomputed', recomputed), ('fixed', fixed)):
            for key in ('prediction', 'true_margin', 'candidates', 'kept_segments'):
                row[f'{name}_{key}'] = trace[key]
            row[f'{name}_correct'] = int(trace['prediction'] == label)
        for name, trace in (('recomputed', recomputed), ('fixed', fixed)):
            row.update({f'{name}_{key}': value for key, value in compare_selection(full, trace, day, draw).items()})
        row['recomputed_vs_fixed_iou'] = interval_iou(recomputed['intervals'], fixed['intervals'])
        row['recomputed_vs_fixed_flip'] = int(recomputed['prediction'] != fixed['prediction'])
        rows.append(row)
        draw_intervals.append(recomputed['intervals'])
    other_draw_ious = [[] for _ in rows]
    for i, j in combinations(range(len(rows)), 2):
        start = max(dates[draws[i, 0]], dates[draws[j, 0]])
        stop = min(dates[draws[i, -1]], dates[draws[j, -1]])
        overlap = interval_iou(draw_intervals[i], draw_intervals[j], (start, stop)) if start < stop else None
        if overlap is not None:
            other_draw_ious[i].append(overlap)
            other_draw_ious[j].append(overlap)
    for row, values in zip(rows, other_draw_ious):
        row['recomputed_other_draw_kept_iou'] = float(np.mean(values)) if values else None
    return rows


def write_csv(path, rows):
    if rows:
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(rows, label_name):
    keys = ['full_dates', 'sampled_dates', 'boundary_f1', 'boundary_nearest_days', 'full_cuts_in_span', 'sparse_cuts_in_span',
            'full_correct', 'full_candidates', 'full_kept_segments',
            'recomputed_vs_fixed_iou', 'recomputed_vs_fixed_flip', 'recomputed_other_draw_kept_iou']
    for branch in ('recomputed', 'fixed'):
        keys.extend(f'{branch}_{key}' for key in ('kept_iou_common_span', 'kept_iou_full_span',
            'kept_iou_sampled_dates', 'kept_mean_cosine_distance', 'matched_kept_segments',
            'matched_segment_cosine_distance', 'prediction_flip', 'margin_delta', 'correct',
            'candidates', 'kept_segments'))
    result = dict(class_name=label_name, parcels=len({row['parcel'] for row in rows}), trials=len(rows))
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    for branch in ('recomputed', 'fixed'):
        values = [row[f'{branch}_kept_iou_common_span'] for row in rows
                  if row[f'{branch}_kept_iou_common_span'] is not None]
        result[f'{branch}_kept_iou_p10'] = float(np.percentile(values, 10)) if values else None
    return result


def analyze_job(job, args):
    model, selector, packet, _, source, target = load_job(job, args)
    output = Path(args.output_dir)
    classes = job['config']['classes']
    summaries = []
    for domain, dataset, shift in (('source', source, packet['source_to_target_shift']), ('target', target, 0)):
        if domain not in args.domains:
            continue
        if len(dataset.date_positions) <= args.dates:
            print(f'SAMPLING_SKIP route={job["route"]} task={job["task"]} domain={domain} '
                  f'reason=at_most_{args.dates}_observed_dates', flush=True)
            continue
        labels = dataset.get_labels()
        rows = []
        for label, name in enumerate(classes):
            pool = np.random.default_rng(args.seed+label).permutation(np.flatnonzero(labels == label))
            class_rows = []
            eligible = 0
            for index in pool:
                trials = parcel_trials(model, selector, job, dataset, index, shift, args)
                if trials:
                    class_rows.extend(trials)
                    eligible += 1
                    if eligible == args.parcels_per_class:
                        break
            if class_rows:
                rows.extend(class_rows)
                summaries.append(dict(route=job['route'], task=job['task'], domain=domain,
                                      **summarize(class_rows, name)))
            print(f'SAMPLING_CLASS route={job["route"]} task={job["task"]} domain={domain} '
                  f'class={name!r} parcels={eligible} trials={len(class_rows)}', flush=True)
        if not rows:
            print(f'SAMPLING_SKIP route={job["route"]} task={job["task"]} domain={domain} '
                  f'reason=no_parcels_with_more_than_{args.dates}_dates', flush=True)
            continue
        write_csv(output / f'{job["route"]}_{job["task"]}_{domain}_trials.csv', rows)
        result = summarize(rows, 'ALL')
        summaries.append(dict(route=job['route'], task=job['task'], domain=domain, **result))
        print(f'SAMPLING_SUMMARY route={job["route"]} task={job["task"]} domain={domain} {result}', flush=True)
    return summaries


def main(argv=None):
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment_root', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--joint_task', choices=TASKS, default='AT1_DK1')
    parser.add_argument('--domains', nargs='+', choices=('source', 'target'), default=['source', 'target'])
    parser.add_argument('--dates', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--parcels_per_class', type=int, default=12)
    parser.add_argument('--cut_tolerance', type=float, default=7.)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.dates < 2 or min(args.repeats, args.parcels_per_class) < 1 or not np.isfinite(args.cut_tolerance) or args.cut_tolerance < 0:
        parser.error('Require at least two dates, positive counts and finite nonnegative cut tolerance')
    root, output = Path(args.experiment_root).resolve(), Path(args.output_dir).resolve()
    if output == root or root in output.parents:
        parser.error('Diagnostic output must be outside the training directory')
    jobs = validate_final_drop_jobs(root, args.joint_task)
    output.mkdir(parents=True, exist_ok=True)
    print(f'SAMPLING_PROTOCOL dates={args.dates} repeats={args.repeats} parcels_per_class={args.parcels_per_class} '
          f'cut_tolerance_days={args.cut_tolerance} seed={args.seed} selection=deterministic fixed_pixels=true', flush=True)
    summaries = []
    with torch.no_grad():
        for job in jobs:
            print(f'JOB_START route={job["route"]} task={job["task"]} '
                  f'sha256={job["tested_checkpoint"]["sha256"]}', flush=True)
            summaries.extend(analyze_job(job, args))
    if not summaries:
        raise ValueError('No eligible parcels in the selected test splits')
    write_csv(output / 'summary.csv', summaries)


if __name__ == '__main__':
    main()
