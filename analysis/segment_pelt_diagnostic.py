"""Freeze historical Source PSE, cache SourceTrain features, and scan PELT.

No classifier or Selector is trained. Complete candidate partitions always
cover the observed sequence; learned selected coverage is not measured here.
Cached features can be transferred for CPU-only scans without the dataset.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from utils.pelt_diagnostic import sweep_sequence, summarize_records


TASKS = ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def read_cache(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if not all(manifest.get(key) is True for key in ('source_train_only', 'frozen_model', 'no_training')):
        raise ValueError('Expected a verified frozen SourceTrain feature cache')
    result = {}
    for task, entry in manifest['tasks'].items():
        path = root/entry['features_file']
        metadata = json.loads((root/entry['metadata_file']).read_text(encoding='utf-8'))
        if metadata.get('source_train_only') is not True:
            raise ValueError(f'SourceTrain metadata missing for {task}')
        with np.load(path, allow_pickle=False) as arrays:
            copied = {key: arrays[key].copy() for key in arrays.files}
        features, dates, ids = (copied[key] for key in ('features', 'dates', 'parcel_ids'))
        if (features.ndim != 3 or dates.shape != features.shape[:2]
                or len(ids) != len(features) or not np.isfinite(features).all()):
            raise ValueError(f'Invalid cached feature/date shapes for {task}')
        result[task] = dict(arrays=copied, metadata=metadata,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if not result:
        raise ValueError('Empty SourceTrain feature cache')
    return result


def scan_cache(feature_root, output, args):
    data = read_cache(feature_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, examples, sequence_count = [], {}, 0
    for task, packet in data.items():
        arrays = packet['arrays']
        for sample, features in enumerate(arrays['features']):
            dates = arrays['dates'][sample]
            full = np.arange(len(features))
            views = [('full', -1, full)]
            for repeat, indices in enumerate(arrays['subset_indices'][sample]):
                indices = np.asarray(indices, dtype=np.int64)
                if (len(np.unique(indices)) != len(indices) or (indices < 0).any()
                        or (indices >= len(features)).any() or (np.diff(indices) < 0).any()):
                    raise ValueError('Invalid cached random timestep indices')
                views.append(('sample30', repeat, indices))
            for view, repeat, indices in views:
                scanned = sweep_sequence(features[indices], dates[indices],
                    penalties=args.penalties, normalizations=args.normalizations,
                    gamma_multipliers=args.gamma_multipliers,
                    penalty_modes=args.penalty_modes, min_size=args.min_size)
                for record in scanned:
                    record.update(task=task, view=view, repeat=repeat,
                                  parcel_id=int(arrays['parcel_ids'][sample]))
                rows.extend(scanned)
                sequence_count += 1
                if sample == 0:
                    examples.setdefault(task, {})[view] = dict(features=features[indices],
                        dates=dates[indices], records=scanned, parcel_id=int(arrays['parcel_ids'][sample]))
            if (sample+1) % 16 == 0 or sample+1 == len(arrays['features']):
                print(f'PELT_SCAN|task={task}|parcels={sample+1}/{len(arrays["features"])}|conditions={len(scanned)}', flush=True)
    summary = summarize_records(rows)
    write_csv(output/'per_sequence.csv', rows)
    write_csv(output/'parameter_summary.csv', summary)
    for task, views in examples.items():
        serializable = {name: dict(dates=view['dates'].tolist(), parcel_id=view['parcel_id'],
                                  segmentations=view['records']) for name, view in views.items()}
        (output/f'example_boundaries_{task}.json').write_text(json.dumps(serializable, indent=2), encoding='utf-8')
    manifest = dict(no_training=True, source_train_only=True, frozen_model=True,
        numpy_version=np.__version__,
        feature_root=str(Path(feature_root).resolve()), sequence_count=sequence_count,
        parameter_combinations=len(args.penalties)*len(args.normalizations)*len(args.gamma_multipliers)*len(args.penalty_modes),
        normalization=args.normalizations, gamma_multipliers=args.gamma_multipliers,
        penalty_coefficients=args.penalties, penalty_modes=args.penalty_modes, min_size=args.min_size,
        baseline='feature normalization, median-positive RBF bandwidth, actual penalty=0.1*T, min_size=3',
        candidate_partition_coverage=1., learned_selected_coverage='not applicable: no Selector training',
        no_automatic_parameter_selection=True,
        tasks={task: dict(cache_sha256=packet['sha256'], metadata=packet['metadata']) for task, packet in data.items()})
    root = Path(__file__).resolve().parents[1]
    manifest['diagnostic_code_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('analysis/segment_pelt_diagnostic.py', 'utils/pelt_diagnostic.py', 'utils/pelt_feature_cache.py')}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if not args.no_plots:
        plot_summary(summary, examples, output)
    print(f'PELT_SCAN_COMPLETE|sequences={sequence_count}|output={output}', flush=True)
    return manifest


def plot_summary(summary, examples, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks = list(examples)
    for view in ('full', 'sample30'):
        selected = [row for row in summary if row['view'] == view]
        if not selected:
            continue
        vertical = sorted({(row['normalization'], row['gamma_multiplier']) for row in selected})
        horizontal = sorted({(row['penalty_mode'], row['penalty_coefficient']) for row in selected})
        for field, title in (('single_segment_rate', 'Single-segment rate'), ('mean_K', 'Mean candidate count')):
            fig, axes = plt.subplots(2, 2, figsize=(15, 11), squeeze=False)
            vmax = 1 if field == 'single_segment_rate' else max(row[field] for row in selected)
            for task, ax in zip(tasks, axes.flat):
                matrix = np.full((len(vertical), len(horizontal)), np.nan)
                for row in selected:
                    if row['task'] == task:
                        i = vertical.index((row['normalization'], row['gamma_multiplier']))
                        j = horizontal.index((row['penalty_mode'], row['penalty_coefficient']))
                        matrix[i, j] = row[field]
                image = ax.imshow(matrix, vmin=0, vmax=vmax, aspect='auto', cmap='viridis')
                ax.set_title(task)
                ax.set_yticks(range(len(vertical)), [f'{norm} / gamma x{gamma:g}' for norm, gamma in vertical], fontsize=8)
                ax.set_xticks(range(len(horizontal)), [f'{mode}\n{value:g}' for mode, value in horizontal], rotation=55, ha='right', fontsize=8)
                for i in range(len(vertical)):
                    for j in range(len(horizontal)):
                        ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=7,
                                color='white' if matrix[i,j] < vmax*.5 else 'black')
            for ax in axes.flat[len(tasks):]:
                ax.set_visible(False)
            fig.suptitle(title+' | '+view+' | frozen SourceTrain features; no training')
            fig.colorbar(image, ax=list(axes.flat), shrink=.6)
            fig.subplots_adjust(left=.12, right=.88, bottom=.15, top=.93, hspace=.65, wspace=.5)
            fig.savefig(output/f'{field}_{view}.png', dpi=140)
            plt.close(fig)
    for task, views in examples.items():
        view = views.get('sample30', views['full'])
        features, dates = view['features'], view['dates']
        centered = features-features.mean(0)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        curve = centered@right[0]
        choices = [record for record in view['records']
                   if record['normalization'] == 'feature' and record['gamma_multiplier'] == 1]
        choices = sorted(choices, key=lambda r: (r['penalty_mode'], r['penalty_coefficient']))[:10]
        if not choices:
            continue
        fig, axes = plt.subplots(2, 5, figsize=(18, 7), squeeze=False)
        for record, ax in zip(choices, axes.flat):
            ax.plot(dates, curve, 'o-', color='#245681', markersize=3, lw=1)
            bounds = record['boundaries']
            edges = np.r_[dates[0], (dates[:-1]+dates[1:])/2, dates[-1]]
            for segment, (start, end) in enumerate(zip(bounds, bounds[1:])):
                ax.axvspan(edges[start], edges[end], color=plt.get_cmap('tab20')(segment % 20), alpha=.18)
                ax.text((edges[start]+edges[end])/2, .96, f'R{segment+1}', transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=7)
                if start:
                    ax.axvline(edges[start], color='black', ls='--', lw=.8)
            ax.set_title(f'{record["penalty_mode"]}: c={record["penalty_coefficient"]:g}\nactual={record["actual_penalty"]:.3g}, K={record["segment_count"]}', fontsize=9)
            ax.grid(alpha=.15)
        for ax in axes.flat[len(choices):]: ax.set_visible(False)
        fig.suptitle(f'{task} / parcel {view["parcel_id"]} | same feature RBF; vary penalty only\nPCA is display-only; PELT uses every cached feature dimension')
        fig.supxlabel('Original TimeMatch observation day')
        fig.tight_layout(rect=(0, .04, 1, .91))
        fig.savefig(output/f'example_boundaries_{task}.png', dpi=140)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'extract', 'scan'), default='all')
    parser.add_argument('--data_root')
    parser.add_argument('--source_root', default='outputs/timematch_historical_rerun_seed1/source')
    parser.add_argument('--feature_root', help='Existing frozen SourceTrain cache for CPU-only scan')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--samples_per_task', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seq_length', type=int, default=30)
    parser.add_argument('--min_size', type=int, default=3)
    parser.add_argument('--penalties', nargs='+', type=float, default=[.003, .01, .03, .1, .3])
    parser.add_argument('--normalizations', nargs='+', choices=('feature', 'global', 'none'), default=['feature', 'global', 'none'])
    parser.add_argument('--gamma_multipliers', nargs='+', type=float, default=[.25, 1., 4.])
    parser.add_argument('--penalty_modes', nargs='+', choices=('absolute', 'per_observation'), default=['absolute', 'per_observation'])
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args(argv)
    if args.stage == 'scan' and not args.feature_root:
        parser.error('--stage scan requires --feature_root')
    if args.stage != 'scan' and not args.data_root:
        parser.error('Feature extraction requires --data_root')
    if min(args.samples_per_task, args.batch_size, args.repeats, args.seq_length, args.min_size) < 1:
        parser.error('Sample counts and sequence lengths must be positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    features = Path(args.feature_root) if args.stage == 'scan' else output/'features'
    if args.stage != 'scan':
        from utils.pelt_feature_cache import extract_features
        extract_features(args.data_root, args.source_root, features, tasks=args.tasks,
            samples_per_task=args.samples_per_task, seq_length=args.seq_length, repeats=args.repeats,
            seed=args.seed, device=args.device, batch_size=args.batch_size)
    if args.stage != 'extract':
        scan_cache(features, output/'scan', args)


if __name__ == '__main__':
    main()
