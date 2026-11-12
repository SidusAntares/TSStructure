"""Compare source refinement runs on the same parcels and initial PELT cuts."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np

from analysis.segment_structure_visualization import TASKS, _batch, _sample, fit_projection, ifft_display


ROUTES = ('one_round', 'two_rounds')
LABELS = ('Initial INSHAPE PELT', 'One round / 20 epochs', 'Two rounds / 30 epochs')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_source(root, task, route):
    leaf = root / route / task / 'source'
    config = read_json(leaf / 'resolved_config.json')
    rounds = ROUTES.index(route) + 1
    epochs = 10 + 10 * rounds
    if config.get('refinement_rounds') != rounds or config.get('epochs') != epochs:
        raise ValueError(f'Unexpected refinement configuration in {leaf}')
    folder = leaf / 'refinement'
    with (folder / 'parcels.jsonl').open(encoding='utf-8') as stream:
        rows = {}
        for line in stream:
            row = json.loads(line)
            if row['parcel'] in rows:
                raise ValueError(f'Duplicate parcel {row["parcel"]} in {leaf}')
            rows[row['parcel']] = row
    history = read_json(folder / 'round_history.json')
    phases = [(r['epoch'], r['phase']) for r in history]
    if phases != [(epoch, phase) for i in range(rounds)
                  for epoch, phase in ((10+10*i, 'split'), (15+10*i, 'merge'))]:
        raise ValueError(f'Incomplete refinement schedule in {leaf}: {phases}')
    bank = read_json(folder / 'prototypes.json')['classes']
    test = read_json(leaf / 'fold_0' / f'test_{config["source"].replace("/", "_")}.json')
    if test['completed_epochs'] != epochs:
        raise ValueError(f'Unexpected evaluated epoch in {leaf}')
    return dict(leaf=leaf, config=config, rows=rows, history=history, bank=bank, test=test)


def match_cuts(left, right, tolerance=14):
    """Match as many ordered boundaries as possible, breaking ties by date error."""
    scores = [[(0, 0.) for _ in range(len(right)+1)] for _ in range(len(left)+1)]
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            options = [scores[i-1][j], scores[i][j-1]]
            error = abs(a-b)
            if error <= tolerance:
                count, negative_error = scores[i-1][j-1]
                options.append((count+1, negative_error-error))
            scores[i][j] = max(options)
    count, negative_error = scores[-1][-1]
    return count, -negative_error, len(left)+len(right)


def cut_metrics(counts):
    matches, error, candidates = counts
    return dict(paired_cut_f1=float(2*matches/candidates) if candidates else None,
                mean_matched_error_days=float(error/matches) if matches else None)


def parcel_metrics(rows):
    totals = Counter()
    for row in rows:
        selected = [segment for segment in row['segments'] if segment['selected']]
        hits = sum(segment['cluster'] is not None for segment in selected)
        totals.update(parcels=1, candidates=len(row['segments']), selected=len(selected), hits=hits,
                      parcels_with_hits=int(hits > 0), edited=int(row['initial'] != row['final']))
    n = totals['parcels']
    return dict(parcels=n, changed_parcels=totals['edited'],
                mean_candidates=totals['candidates']/n, mean_selected=totals['selected']/n,
                selected_segment_match_fraction=totals['hits']/max(1, totals['selected']),
                parcel_match_fraction=totals['parcels_with_hits']/n)


def summarize_task(one, two, seed, samples_per_class):
    classes = one['config']['classes']
    if (classes != two['config']['classes'] or one['config']['source'] != two['config']['source']
            or one['config']['original_source_checkpoint_sha256']
            != two['config']['original_source_checkpoint_sha256']):
        raise ValueError('Paired runs require the same Source and class order')
    left, right = one['rows'], two['rows']
    if not left or left.keys() != right.keys():
        raise ValueError('Paired runs require identical training parcel IDs')
    by_class = defaultdict(list)
    for parcel in sorted(left):
        if left[parcel]['label'] != right[parcel]['label'] or left[parcel]['initial'] != right[parcel]['initial']:
            raise ValueError(f'Initial reference differs on parcel {parcel}')
        by_class[left[parcel]['label']].append(parcel)
    aggregate = np.zeros((3, 3))
    class_stats, plot_ids = {}, {}
    for label, ids in by_class.items():
        rng = np.random.default_rng(seed+label)
        cohort = rng.choice(ids, size=min(128, len(ids)), replace=False).tolist()
        plot_ids[label] = cohort[:samples_per_class]
        pairs = list(combinations(cohort, 2))
        selected = rng.choice(len(pairs), size=min(500, len(pairs)), replace=False)
        totals = np.zeros_like(aggregate)
        for index in selected:
            a, b = pairs[index]
            for i, (rows, key) in enumerate(((left, 'initial'), (left, 'final'), (right, 'final'))):
                totals[i] += match_cuts(rows[a][key], rows[b][key])
        aggregate += totals
        class_stats[classes[label]] = dict(
            pairs=len(selected), initial=cut_metrics(totals[0]),
            one_round=dict(**cut_metrics(totals[1]), **parcel_metrics(left[p] for p in ids),
                           prototypes=len(one['bank'].get(str(label), []))),
            two_rounds=dict(**cut_metrics(totals[2]), **parcel_metrics(right[p] for p in ids),
                            prototypes=len(two['bank'].get(str(label), []))))
    result = dict(initial=cut_metrics(aggregate[0]), classes=class_stats)
    for i, (route, run) in enumerate(zip(ROUTES, (one, two)), 1):
        test = run['test']
        result[route] = dict(**cut_metrics(aggregate[i]), **parcel_metrics(run['rows'].values()),
                             source_epochs=test['completed_epochs'],
                             prototypes=sum(map(len, run['bank'].values())),
                             rounds=run['history'],
                             heldout_native_pelt={kind: {key: test[kind][key]
                                                        for key in ('accuracy', 'macro_f1')}
                                                  for kind in ('full', 'keep')})
    return result, plot_ids


def load_plot_features(run, parcel_ids, data_root, device):
    from types import SimpleNamespace
    import torch
    from torchvision.transforms import Compose
    from dataset import PixelSetData
    from transforms import RandomSamplePixels, Normalize, ToTensor
    from utils.segment_training import new_models

    cfg = SimpleNamespace(**run['config'])
    cfg.device = device
    transform = Compose([RandomSamplePixels(cfg.num_pixels), Normalize(), ToTensor()])
    dataset = PixelSetData(str(data_root), cfg.source, cfg.classes, transform, indices=set(parcel_ids),
                           closed_set=cfg.closed_set, combine_spring_and_winter=cfg.combine_spring_and_winter)
    model, _ = new_models(cfg)
    packet = torch.load(run['leaf'] / 'fold_0' / 'model.pt', map_location=device, weights_only=False)
    if packet.get('role') != 'source_structure' or packet.get('refinement_rounds') != 1:
        raise ValueError('Expected the one-round Source checkpoint for display features')
    model.load_state_dict(packet['state_dict'])
    model.eval()
    views = {}
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = _sample(dataset, i, cfg.seed)
            pixels, mask, dates, extra = _batch(sample, device)
            values = model.spatial_encoder(pixels, mask, extra)[0].cpu().numpy()
            views[int(sample['parcel_index'])] = dates[0].cpu().numpy(), values
    if set(views) != set(parcel_ids):
        raise ValueError('Plot data did not contain the paired training parcels')
    mean, component = fit_projection([values for _, values in views.values()])
    return {parcel: (dates, (values-mean) @ component) for parcel, (dates, values) in views.items()}


def plot_class(task, name, ids, one, two, curves, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, len(ids), squeeze=False, sharex=True, sharey=True,
                                figsize=(4.8*len(ids), 7), layout='constrained')
    for column, parcel in enumerate(ids):
        dates, values = curves[parcel]
        grid, smooth = ifft_display(dates, values)
        first, second = one['rows'][parcel], two['rows'][parcel]
        panels = ((first['initial'], None), (first['final'], first['segments']),
                  (second['final'], second['segments']))
        for row, (cuts, segments) in enumerate(panels):
            ax = axes[row, column]
            ax.plot(grid, smooth, color='#263b56', lw=1.25)
            ax.scatter(dates, values, color='#263b56', alpha=.45, s=9)
            for cut in cuts:
                ax.axvline(cut, color='#277c77' if cut not in first['initial'] else '#6c7886',
                           linestyle='--', lw=.85)
            if segments is not None:
                edges = [dates[0], *cuts, dates[-1]]
                for i, segment in enumerate(segments):
                    ax.axvspan(edges[i], edges[i+1], ymin=.025, ymax=.08, color='#277c77',
                               alpha=.85 if segment['selected'] else .15)
            kept = '' if segments is None else f' / {sum(s["selected"] for s in segments)} kept'
            ax.set_title(f'{LABELS[row]} · {len(cuts)+1} segments{kept}', fontsize=9)
            ax.grid(axis='y', alpha=.18)
        axes[0, column].text(.02, .96, f'Parcel {parcel}', transform=axes[0, column].transAxes,
                             va='top', fontsize=9)
    figure.suptitle(f'{task} · {name}\nSame training parcels · green dashed lines: added cuts · dark bars: kept')
    figure.supxlabel('Day relative to Source start')
    figure.supylabel('Shared PSE PC1')
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment_root', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--data_root', type=Path)
    parser.add_argument('--plot_task', choices=TASKS, default='AT1_DK1')
    parser.add_argument('--samples_per_class', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.samples_per_class < 1:
        parser.error('--samples_per_class must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, plot_inputs = {}, None
    for task in TASKS:
        one, two = (load_source(args.experiment_root, task, route) for route in ROUTES)
        results[task], plot_ids = summarize_task(one, two, args.seed, args.samples_per_class)
        for route in ROUTES:
            record = results[task][route]
            print('SOURCE_REFINEMENT_COMPARE|' + json.dumps(dict(task=task, route=route,
                  paired_cut_f1=record['paired_cut_f1'], heldout_native_pelt=record['heldout_native_pelt'],
                  selected_segment_match_fraction=record['selected_segment_match_fraction'])), flush=True)
        if args.data_root is not None and task == args.plot_task:
            ids = [parcel for group in plot_ids.values() for parcel in group]
            plot_inputs = (
                dict(leaf=one['leaf'], config=one['config'], rows={p: one['rows'][p] for p in ids}),
                dict(rows={p: two['rows'][p] for p in ids}),
                plot_ids,
            )
    report = dict(pairing=dict(seed=args.seed, max_parcels_per_class=128, max_pairs_per_class=500,
                               tolerance_days=14), tasks=results)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if plot_inputs is not None:
        one, two, plot_ids = plot_inputs
        curves = load_plot_features(one, [p for ids in plot_ids.values() for p in ids],
                                    args.data_root, args.device)
        folder = args.output_dir / args.plot_task
        folder.mkdir(exist_ok=True)
        for label, ids in plot_ids.items():
            name = one['config']['classes'][label]
            filename = f'{label:02d}_{re.sub(r"[^A-Za-z0-9_-]", "_", name)}.png'
            plot_class(args.plot_task, name, ids, one, two, curves, folder / filename)
        (folder / 'paired_parcels.json').write_text(json.dumps(
            {one['config']['classes'][y]: ids for y, ids in plot_ids.items()}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
