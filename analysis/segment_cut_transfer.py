"""Visualize final Student selections or diagnose same-class source cut transfer."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from analysis.segment_structure_visualization import (
    TASKS, _batch, _date, _sample, fit_projection, ifft_display, load_job, validate_jobs,
)


def candidate_edges(dates, segments):
    """Candidate boundaries are midpoints between the bordering observations."""
    dates = np.asarray(dates, dtype=float)
    cuts = np.flatnonzero(np.diff(segments))
    return np.r_[dates[0], (dates[cuts] + dates[cuts + 1]) / 2, dates[-1]]


def transfer_segments(target_dates, source_cuts):
    """Apply cuts on the target date axis and collapse intervals with no observations."""
    intervals = np.searchsorted(source_cuts, target_dates, side='right')
    return np.r_[0, np.cumsum(np.diff(intervals) != 0)].astype(np.int64)


def forward_options(job):
    config = job['config']
    return dict(variant=job['variant'], deterministic=True, full_loss=False, unified=True,
                penalty=config['pelt_penalty'], min_size=config['pelt_min_size'],
                penalty_mode=config['pelt_penalty_mode'])


def source_and_target_pools(source, target, label, seed):
    rng = np.random.default_rng(seed + label)
    return [rng.permutation(np.flatnonzero(dataset.get_labels() == label))
            for dataset in (source, target)]


def collect_observations(model, selector, job, dataset, index, shift, args):
    from utils.segment_structure import structure_forward

    sample = _sample(dataset, int(index), args.seed)
    pixels, mask, dates, extra = _batch(sample, args.device)
    _, _, info = structure_forward(model, selector, pixels.float(), mask.float(),
                                   dates.long()+shift, extra.float(), **forward_options(job))
    valid = info['valid_time'][0]
    return dict(parcel=int(sample['parcel_index']), dates=dates[0, valid].cpu().numpy(),
                segments=info['segments'][0, valid].cpu().numpy(),
                gate=info['gate'][0, 0].cpu().numpy().astype(bool),
                features=info['features'][0, valid].cpu().numpy())


def retained_overlap(a, b):
    """Jaccard overlap of kept day intervals within the common date span."""
    start = max(a['dates'][0], b['dates'][0])
    stop = min(a['dates'][-1], b['dates'][-1])
    if stop <= start:
        return None
    grid = np.arange(start, stop) + .5
    keep = []
    for trace in (a, b):
        edges = candidate_edges(trace['dates'], trace['segments'])
        keep.append(trace['gate'][np.searchsorted(edges[1:-1], grid, side='right')])
    union = (keep[0] | keep[1]).sum()
    return float((keep[0] & keep[1]).sum() / union) if union else None


def plot_class(sources, target, path, title, keep_fraction):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def trace(ax, packet, name, color, row=0):
        dates, curve, labels = (packet[key] for key in ('dates', 'curve', 'segments'))
        edges = candidate_edges(dates, labels)
        grid, smooth = ifft_display(dates, curve, keep_fraction)
        ax.plot(grid, smooth, color=color, label=name, lw=1.6)
        ax.scatter(dates, curve, color=color, s=9, alpha=.45)
        for i, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
            kept = packet['gate'][i]
            ax.axvspan(start, end, ymin=.025+row*.07, ymax=.08+row*.07,
                       color=color, alpha=.78 if kept else .17)
        for cut in edges[1:-1]:
            ax.axvline(cut, color=color, ls='--', lw=.9, alpha=.55)

    fig, axes = plt.subplots(1+len(sources), 1, figsize=(12, 2.5+1.85*len(sources)),
                             sharex=True, sharey=True, constrained_layout=True)
    trace(axes[0], sources[0], f"Source {sources[0]['parcel']}", '#1565c0')
    trace(axes[0], target, f"Target {target['parcel']}", '#d86018', row=1)
    axes[0].set_title('Cross-domain comparison', loc='left')
    axes[0].legend(loc='upper right', ncol=2)
    overlaps = [value for a, b in combinations(sources, 2)
                if (value := retained_overlap(a, b)) is not None]
    summary = f'mean kept-day IoU {np.mean(overlaps):.2f}' if overlaps else 'kept-day IoU N/A'
    for i, (ax, packet) in enumerate(zip(axes[1:], sources)):
        trace(ax, packet, 'Source', '#1565c0')
        candidate_count = int(packet['segments'][-1])+1
        kept_count = int(packet['gate'][:candidate_count].sum())
        label = f"Source {packet['parcel']} | {kept_count}/{candidate_count} segments kept"
        ax.set_title(f'{label} | within-class {summary}' if i == 0 else label, loc='left', fontsize=10)
    for ax in axes:
        ax.grid(axis='y', alpha=.18)
        ax.margins(y=.19)
    fig.suptitle(title + '\niFFT curves with observed points | dashed: cuts | dark/light bars: kept/rejected')
    fig.supylabel('Shared Source-train PSE PC1')
    fig.supxlabel('Original calendar day relative to Source start')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_images(job, args):
    model, selector, checkpoint, train, source, target = load_job(job, args, sample_pixels=False)
    rng = np.random.default_rng(args.seed)
    train_ids = rng.permutation(len(train))[:args.pca_parcels]
    if not len(train_ids):
        raise ValueError('Source training split is empty; cannot fit the display projection')
    features = []
    for index in train_ids:
        pixels, mask, _, extra = _batch(_sample(train, int(index), args.seed), args.device)
        features.append(model.spatial_encoder(pixels.float(), mask.float(), extra.float())[0].cpu().numpy())
    mean, component = fit_projection(features)
    target_origin = (_date(target.metadata['start_date']) - _date(source.metadata['start_date'])).days
    output = Path(args.output_dir) / job['route'] / job['task']
    for label, name in enumerate(job['config']['classes']):
        source_pool, target_pool = source_and_target_pools(source, target, label, args.seed)
        if not len(source_pool) or not len(target_pool):
            print(f'PLOT_SKIP class={name!r} reason=missing_shared_test_class', flush=True)
            continue
        traces = [collect_observations(model, selector, job, source, i,
                   checkpoint['source_to_target_shift'], args) for i in source_pool[:args.source_samples]]
        target_trace = collect_observations(model, selector, job, target, target_pool[0], 0, args)
        target_trace['dates'] = target_trace['dates'] + target_origin
        for packet in (*traces, target_trace):
            packet['curve'] = (packet['features'] - mean) @ component
        safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', name)
        path = output / f'{label:02d}_{safe_name}.png'
        plot_class(traces, target_trace, path, f'{job["route"]} / {job["task"]} / {name} / Student epoch20',
                   args.keep_fraction)
        print(f'PLOT_SAVED {path}', flush=True)


def target_cut_diagnostic(job, args):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from dataset import create_evaluation_loaders
    from utils.segment_structure import _classify_retained_observations, structure_forward
    from utils.segment_training import sample_to_device, seed_all

    model, selector, checkpoint, _, source, target = load_job(job, args, sample_pixels=False)
    config = SimpleNamespace(**job['config'])
    config.data_root, config.device, config.num_workers = args.data_root, args.device, args.num_workers
    print('DIAGNOSTIC_PROTOCOL target_label_oracle=true source_donors_per_class=1 '
          f'source_to_target_shift={checkpoint["source_to_target_shift"]}', flush=True)
    donors = {}
    for label, name in enumerate(config.classes):
        source_pool, target_pool = source_and_target_pools(source, target, label, args.seed)
        if not len(source_pool) or not len(target_pool):
            continue
        donor = collect_observations(model, selector, job, source, source_pool[0], 0, args)
        cuts = candidate_edges(donor['dates'], donor['segments'])[1:-1]
        donors[label] = cuts + checkpoint['source_to_target_shift']
        print(f'DONOR class={name!r} source_parcel={donor["parcel"]} target_axis_cuts={donors[label].tolist()}', flush=True)

    _, loader = create_evaluation_loaders(config.target, job['evaluation_splits'], config, config.sample_pixels_val)
    seed_all(config.seed+1000)
    truth, native_predictions, transferred_predictions = [], [], []
    full_truth, full_native = [], []
    native_counts, transferred_counts = [], []
    changed = 0
    for i, sample in enumerate(loader, 1):
        pixels, mask, dates, extra = sample_to_device(sample, args.device)
        _, native, info = structure_forward(model, selector, pixels, mask, dates, extra, **forward_options(job))
        labels = sample['label'].tolist()
        predictions = native.argmax(-1).cpu().tolist()
        full_truth.extend(labels)
        full_native.extend(predictions)
        matched = [j for j, label in enumerate(labels) if label in donors]
        if matched:
            original = info['segments'][matched]
            valid = original >= 0
            replacement = torch.full_like(original, -1)
            for row, j in enumerate(matched):
                day = dates[j, valid[row]].cpu().numpy()
                labels_from_source = transfer_segments(day, donors[labels[j]])
                replacement[row, valid[row]] = torch.as_tensor(labels_from_source, device=dates.device)
            transferred, transfer_info = _classify_retained_observations(
                model, selector, info['features'][matched], dates[matched], replacement, deterministic=True)
            truth.extend(labels[j] for j in matched)
            native_predictions.extend(predictions[j] for j in matched)
            transferred_predictions.extend(transferred.argmax(-1).cpu().tolist())
            native_counts.extend(info['candidate_count'][matched].cpu().tolist())
            transferred_counts.extend(transfer_info['candidate_count'].cpu().tolist())
            changed += int((original != replacement).any(-1).sum())
        if i % 100 == 0 or i == len(loader):
            print(f'DIAGNOSTIC_PROGRESS batches={i}/{len(loader)}', flush=True)

    if not truth:
        raise ValueError('No target test samples have a same-class source donor')
    label_ids = list(range(len(config.classes)))
    def metrics(y, predicted, labels):
        return (accuracy_score(y, predicted), f1_score(y, predicted, labels=labels,
                                                      average='macro', zero_division=0))
    all_acc, all_f1 = metrics(full_truth, full_native, label_ids)
    saved = job['test_report']['keep']
    print(f'NATIVE_REPLAY saved_acc={saved["accuracy"]:.6f} saved_macro_f1={saved["macro_f1"]:.6f} '
          f'delta_acc={all_acc-saved["accuracy"]:+.6f} delta_f1={all_f1-saved["macro_f1"]:+.6f}', flush=True)
    matched_labels = sorted(set(truth))
    base_acc, base_f1 = metrics(truth, native_predictions, matched_labels)
    cut_acc, cut_f1 = metrics(truth, transferred_predictions, matched_labels)
    print(f'NATIVE_ALL n={len(full_truth)} acc={all_acc:.6f} macro_f1={all_f1:.6f}', flush=True)
    print(f'PAIRED_RESULT n={len(truth)}/{len(full_truth)} changed_cuts={changed} '
          f'native_acc={base_acc:.6f} source_cut_acc={cut_acc:.6f} delta_acc={cut_acc-base_acc:+.6f} '
          f'native_macro_f1={base_f1:.6f} source_cut_macro_f1={cut_f1:.6f} delta_f1={cut_f1-base_f1:+.6f} '
          f'mean_candidates_native={np.mean(native_counts):.3f} mean_candidates_source_cut={np.mean(transferred_counts):.3f}', flush=True)
    y, base, cut = map(np.asarray, (truth, native_predictions, transferred_predictions))
    for label in matched_labels:
        rows = y == label
        print(f'CLASS_RESULT class={config.classes[label]!r} n={rows.sum()} '
              f'native_acc={np.mean(base[rows] == label):.6f} source_cut_acc={np.mean(cut[rows] == label):.6f}', flush=True)


def validate_final_drop_jobs(experiment_root, joint_task):
    from models.segment_selector import DROP_PROTOCOLS

    jobs = validate_jobs(experiment_root, variants=('finetune',), comparison=True)
    jobs += validate_jobs(experiment_root, tasks=(joint_task,), variants=('joint',), comparison=True)
    for job in jobs:
        if job['config'].get('selector_protocol') not in DROP_PROTOCOLS:
            raise ValueError(f'Expected an INSHAPE hard-delete checkpoint: {job["checkpoint"]}')
        target = job['config']['target']
        path = job['checkpoint'].parent / f'test_{target.replace("/", "_")}.json'
        if not path.is_file():
            raise ValueError(f'Final target test report is required: {path}')
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('completed_epochs') != 20 or report.get('domain') != target or 'keep' not in report:
            raise ValueError(f'Expected completed epoch20 target Keep report: {path}')
        job['test_report'] = report
    return jobs


def main(argv=None):
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('plots', 'diagnostic'))
    parser.add_argument('--experiment_root', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', help='PNG output directory for plots mode')
    parser.add_argument('--joint_task', choices=TASKS, default='AT1_DK1')
    parser.add_argument('--source_samples', type=int, default=3)
    parser.add_argument('--pca_parcels', type=int, default=256)
    parser.add_argument('--keep_fraction', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.source_samples < 2 or args.pca_parcels < 1 or not 0 < args.keep_fraction <= 1:
        parser.error('Require at least two source samples, positive PCA count and 0 < keep_fraction <= 1')
    if args.mode == 'plots':
        if args.output_dir is None:
            parser.error('plots mode requires --output_dir')
        experiment, output = Path(args.experiment_root).resolve(), Path(args.output_dir).resolve()
        if output == experiment or experiment in output.parents:
            parser.error('Plot directory must be outside the experiment directory')
    jobs = validate_final_drop_jobs(args.experiment_root, args.joint_task)
    run_job = make_images if args.mode == 'plots' else target_cut_diagnostic
    with torch.no_grad():
        for job in jobs:
            print(f'JOB_START mode={args.mode} route={job["route"]} task={job["task"]} '
                  f'checkpoint={job["checkpoint"]} sha256={job["tested_checkpoint"]["sha256"]}', flush=True)
            run_job(job, args)
            print(f'JOB_DONE route={job["route"]} task={job["task"]}', flush=True)


if __name__ == '__main__':
    main()
