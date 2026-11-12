"""Compare source refinement runs on the same parcels and initial PELT cuts."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np

from analysis.segment_structure_visualization import TASKS, _batch, _sample, fit_projection, ifft_display


ROUTES = ('one_round', 'two_rounds', 'three_rounds')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def expected_phases(rounds, adapt=5):
    return [(10+adapt*i, 'merge' if i % 2 else 'split') for i in range(2*rounds)]


def complete_history(history, rounds, adapt=5):
    return (isinstance(history, list)
            and all(isinstance(record, dict) for record in history)
            and [(r.get('epoch'), r.get('phase')) for r in history] == expected_phases(rounds, adapt))


class SourceScheduleError(ValueError):
    """The saved checkpoint belongs to a different Source schedule."""


def verify_source_checkpoint(leaf, rounds, adapt=5):
    import torch

    packet = torch.load(leaf / 'fold_0' / 'model.pt', map_location='cpu', weights_only=True)
    expected = ('source_structure', rounds, 10 + 2*adapt*rounds, adapt)
    actual = (packet.get('role'), packet.get('refinement_rounds'), packet.get('completed_epochs'),
              packet.get('refinement_adapt_epochs', 5))
    if actual != expected:
        raise SourceScheduleError(
            f'{leaf}: checkpoint does not confirm the requested {rounds} refinement round(s). '
            f'Found role={actual[0]!r}, refinement_rounds={actual[1]!r}, completed_epochs={actual[2]!r}, '
            f'refinement_adapt_epochs={actual[3]!r}. '
            'Resume the existing experiment with REPAIR_SOURCE=1 to replace this Source result.')
    return packet


def load_round_history(leaf, rounds, source_log, adapt=5):
    history_path = leaf / 'refinement' / 'round_history.json'
    if history_path.is_file():
        try:
            history = read_json(history_path)
        except json.JSONDecodeError:
            history = None
        if complete_history(history, rounds, adapt):
            return history, 'round_history.json'

    checkpoint = leaf / 'fold_0' / 'model.pt'
    if checkpoint.is_file():
        packet = verify_source_checkpoint(leaf, rounds, adapt)
        history = packet.get('refinement_round_history')
        if complete_history(history, rounds, adapt):
            return history, 'checkpoint'

    if source_log is not None and source_log.is_file():
        prefix = 'SEGMENT_REFINE_PHASE|'
        history = []
        with source_log.open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                if line.startswith(prefix):
                    try:
                        history.append(json.loads(line[len(prefix):]))
                    except json.JSONDecodeError:
                        continue
        if complete_history(history, rounds, adapt):
            return history, 'source_log'

    if checkpoint.is_file():
        # Older staged checkpoints certify the schedule but did not embed counts.
        return None, 'unavailable'
    raise FileNotFoundError(
        f'{leaf}: no complete round history, final checkpoint or Source phase log to verify this run')


def load_source(root, task, route, source_log_dir=None, adapt=5):
    leaf = root / route / task / 'source'
    config = read_json(leaf / 'resolved_config.json')
    rounds = ROUTES.index(route) + 1
    epochs = 10 + 2*adapt*rounds
    if (config.get('refinement_rounds') != rounds or config.get('epochs') != epochs
            or config.get('refinement_adapt_epochs', 5) != adapt):
        raise ValueError(f'Unexpected refinement configuration in {leaf}')
    folder = leaf / 'refinement'
    source_log = None if source_log_dir is None else source_log_dir / f'{task}_{route}_source.log'
    history, history_source = load_round_history(leaf, rounds, source_log, adapt)
    probe = None
    if adapt == 3:
        probe = read_json(folder / 'validation_probe.json')
        expected = [(10, 0, 'initial', 0)]
        for i, (epoch, phase) in enumerate(expected_phases(rounds, adapt)):
            expected.extend((epoch+age, i//2+1, phase, age) for age in range(adapt+1))
        observed = [(r['epoch'], r['round'], r['phase'], r['epochs_since_edit']) for r in probe]
        if observed != expected or len({r['reference_sha256'] for r in probe}) != 1:
            raise ValueError(f'Incomplete or inconsistent validation probe in {leaf}')
    with (folder / 'parcels.jsonl').open(encoding='utf-8') as stream:
        rows = {}
        for line in stream:
            row = json.loads(line)
            if row['parcel'] in rows:
                raise ValueError(f'Duplicate parcel {row["parcel"]} in {leaf}')
            rows[row['parcel']] = row
    bank = read_json(folder / 'prototypes.json')['classes']
    test = read_json(leaf / 'fold_0' / f'test_{config["source"].replace("/", "_")}.json')
    if test['completed_epochs'] != epochs:
        raise ValueError(f'Unexpected evaluated epoch in {leaf}')
    return dict(leaf=leaf, config=config, rows=rows, history=history, probe=probe,
                history_source=history_source, bank=bank, test=test)


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


def summarize_task(one, two, seed, samples_per_class, three=None):
    runs = (one, two) if three is None else (one, two, three)
    routes = ROUTES[:len(runs)]
    classes = one['config']['classes']
    left = one['rows']
    for run in runs[1:]:
        if any(one['config'][key] != run['config'][key]
               for key in ('classes', 'source', 'original_source_checkpoint_sha256')):
            raise ValueError('Paired runs require the same Source and class order')
        if not left or left.keys() != run['rows'].keys():
            raise ValueError('Paired runs require identical training parcel IDs')
    probes = [run['probe'] for run in runs if run.get('probe') is not None]
    if probes and (len(probes) != len(runs) or any(
            rows[0]['reference_sha256'] != probes[0][0]['reference_sha256'] for rows in probes)):
        raise ValueError('Paired runs require identical fixed validation reference windows')
    by_class = defaultdict(list)
    for parcel in sorted(left):
        if any(left[parcel][key] != run['rows'][parcel][key]
               for run in runs[1:] for key in ('label', 'initial')):
            raise ValueError(f'Initial reference differs on parcel {parcel}')
        by_class[left[parcel]['label']].append(parcel)
    aggregate = np.zeros((len(runs)+1, 3))
    cut_sources = [(left, 'initial'), *((run['rows'], 'final') for run in runs)]
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
            for i, (rows, key) in enumerate(cut_sources):
                totals[i] += match_cuts(rows[a][key], rows[b][key])
        aggregate += totals
        class_stats[classes[label]] = dict(pairs=len(selected), initial=cut_metrics(totals[0]))
        for i, (route, run) in enumerate(zip(routes, runs), 1):
            class_stats[classes[label]][route] = dict(
                **cut_metrics(totals[i]), **parcel_metrics(run['rows'][p] for p in ids),
                prototypes=len(run['bank'].get(str(label), [])))
    result = dict(initial=cut_metrics(aggregate[0]), classes=class_stats)
    for i, (route, run) in enumerate(zip(routes, runs), 1):
        test = run['test']
        result[route] = dict(**cut_metrics(aggregate[i]), **parcel_metrics(run['rows'].values()),
                             source_epochs=test['completed_epochs'],
                             prototypes=sum(map(len, run['bank'].values())),
                             rounds=run['history'],
                             round_history_source=run.get('history_source', 'round_history.json'),
                             heldout_native_pelt={kind: {key: test[kind][key]
                                                        for key in ('accuracy', 'macro_f1')}
                                                  for kind in ('full', 'keep')})
        if run.get('probe'):
            result[route]['validation_probe'] = run['probe']
            result[route]['validation_structure'] = run['probe'][-1]
    if probes:
        result['initial']['validation_structure'] = probes[0][0]
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


def plot_class(task, name, ids, one, two, curves, path, three=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = (one, two) if three is None else (one, two, three)
    labels = ['Initial INSHAPE PELT'] + [
        f'{i} round(s) / {run["test"]["completed_epochs"]} epochs' for i, run in enumerate(runs, 1)]
    rows = len(labels)
    figure, axes = plt.subplots(rows, len(ids), squeeze=False, sharex=True, sharey=True,
                                figsize=(4.8*len(ids), 2.3*rows), constrained_layout=True)
    for column, parcel in enumerate(ids):
        dates, values = curves[parcel]
        grid, smooth = ifft_display(dates, values)
        first = one['rows'][parcel]
        parcels = [run['rows'][parcel] for run in runs]
        panels = [(first['initial'], None)] + [(r['final'], r['segments']) for r in parcels]
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
            ax.set_title(f'{labels[row]} · {len(cuts)+1} segments{kept}', fontsize=9)
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
    parser.add_argument('--source_log_dir', type=Path,
                        help='Defaults to logs/<experiment name>; recovers missing phase histories')
    parser.add_argument('--refinement_adapt_epochs', type=int, choices=(3, 5), default=5)
    parser.add_argument('--data_root', type=Path)
    parser.add_argument('--plot_task', choices=TASKS, default='AT1_DK1')
    parser.add_argument('--samples_per_class', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.samples_per_class < 1:
        parser.error('--samples_per_class must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.source_log_dir or Path('logs') / args.experiment_root.name
    adapt = args.refinement_adapt_epochs
    routes = ROUTES if adapt == 3 else ROUTES[:2]
    results, plot_inputs = {}, None
    for task in TASKS:
        runs = [load_source(args.experiment_root, task, route, log_dir, adapt) for route in routes]
        results[task], plot_ids = summarize_task(*runs[:2], args.seed, args.samples_per_class,
                                                 three=runs[2] if len(runs) == 3 else None)
        for route in routes:
            record = results[task][route]
            print('SOURCE_REFINEMENT_COMPARE|' + json.dumps(dict(task=task, route=route,
                  paired_cut_f1=record['paired_cut_f1'], heldout_native_pelt=record['heldout_native_pelt'],
                  selected_segment_match_fraction=record['selected_segment_match_fraction'],
                  validation_structure=record.get('validation_structure'),
                  round_history_source=record['round_history_source'])), flush=True)
        if args.data_root is not None and task == args.plot_task:
            ids = [parcel for group in plot_ids.values() for parcel in group]
            plot_inputs = [dict(leaf=run['leaf'], config=run['config'], test=run['test'],
                                rows={p: run['rows'][p] for p in ids}) for run in runs], plot_ids
    report = dict(pairing=dict(seed=args.seed, max_parcels_per_class=128, max_pairs_per_class=500,
                               tolerance_days=14), tasks=results)
    if adapt == 3:
        report['validation_probe_protocol'] = dict(
            reference_bank='fixed epoch-10 training shapelets', label_guided=True,
            max_validation_parcels_per_class=24, min_reference_observation_recall=.9,
            min_observation_iou=.5, complete_pair='both parcels retain the same prototype in one segment each')
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if plot_inputs is not None:
        runs, plot_ids = plot_inputs
        one, two = runs[:2]
        curves = load_plot_features(one, [p for ids in plot_ids.values() for p in ids],
                                    args.data_root, args.device)
        folder = args.output_dir / args.plot_task
        folder.mkdir(exist_ok=True)
        for label, ids in plot_ids.items():
            name = one['config']['classes'][label]
            filename = f'{label:02d}_{re.sub(r"[^A-Za-z0-9_-]", "_", name)}.png'
            plot_class(args.plot_task, name, ids, one, two, curves, folder / filename,
                       three=runs[2] if len(runs) == 3 else None)
        (folder / 'paired_parcels.json').write_text(json.dumps(
            {one['config']['classes'][y]: ids for y, ids in plot_ids.items()}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
