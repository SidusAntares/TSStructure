"""Read-only, held-out same-class segment comparison after every UDA job finishes.

The Fourier operation is display-only. Observations, selectors and predictions
always use the original full observation sequence; no missing dates are imputed.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np

TASKS = ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1')
VARIANTS = ('inshape', 'attention')


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


def checkpoint_digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(piece)
    return result.hexdigest()


def validate_jobs(experiment_root, tasks=TASKS, variants=VARIANTS, comparison=False):
    """All-or-nothing preflight; this function creates no output directories."""
    import torch
    jobs = []
    for variant in variants:
        for task in tasks:
            leaf = Path(experiment_root) / variant / task / 'uda'
            checkpoint = leaf / 'fold_0' / 'model.pt'
            if not checkpoint.is_file():
                raise ValueError(f'All jobs must finish before visualization: missing {checkpoint}')
            packet = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if packet.get('role') != 'student' or packet.get('completed_epochs') != 20:
                raise ValueError(f'Expected final Student completed_epochs=20: {checkpoint}')
            model_variant = 'inshape' if comparison else variant
            if packet.get('variant') != model_variant:
                raise ValueError(f'Checkpoint variant mismatch: {checkpoint}')
            for key in ('state_dict', 'selector_state_dict', 'source_to_target_shift'):
                if key not in packet:
                    raise ValueError(f'Missing checkpoint {key}: {checkpoint}')
            config_path = leaf / 'resolved_config.json'
            if not config_path.exists():
                config_path = leaf / 'config.json'
            config = json.loads(config_path.read_text(encoding='utf-8'))
            tested = None
            if comparison:
                tested_path = leaf / 'fold_0' / 'tested_checkpoint.json'
                if not tested_path.is_file():
                    raise ValueError(f'Missing tested checkpoint record: {tested_path}')
                tested = json.loads(tested_path.read_text(encoding='utf-8'))
                if (tested.get('sha256') != checkpoint_digest(checkpoint)
                        or tested.get('role') != 'student' or tested.get('completed_epochs') != 20):
                    raise ValueError(f'Final model differs from tested checkpoint: {checkpoint}')
                expected_epochs = 100 if variant == 'joint' else 20
                expected_init = 'scratch' if variant == 'joint' else 'checkpoint'
                if (config.get('source_training_epochs') != expected_epochs
                        or config.get('source_init') != expected_init
                        or config.get('variant') != model_variant):
                    raise ValueError(f'Comparison route/config mismatch: {leaf}')
            splits = json.loads((leaf / 'split_indices.json').read_text(encoding='utf-8'))
            evaluation_path = leaf / 'evaluation_split_indices.json'
            evaluation_splits = (json.loads(evaluation_path.read_text(encoding='utf-8'))
                                 if evaluation_path.exists() else splits)
            for domain in (config['source'], config['target']):
                parts = splits[domain]
                if any(set(parts[a]) & set(parts[b]) for a, b in
                       (('train', 'val'), ('train', 'test'), ('val', 'test'))):
                    raise ValueError(f'Overlapping held-out splits for {domain}')
                if not set(evaluation_splits[domain]['test']).issubset(parts['test']):
                    raise ValueError(f'Evaluation test parcels must be held out of UDA training: {domain}')
            jobs.append(dict(task=task, variant=model_variant, route=variant, checkpoint=checkpoint,
                             tested_checkpoint=tested,
                             config=config, splits=splits, evaluation_splits=evaluation_splits,
                             evaluation_split_provenance=(str(evaluation_path) if evaluation_path.exists()
                                                          else str(leaf / 'split_indices.json'))))
            del packet
    return jobs


def validate_comparison_jobs(experiment_root):
    return validate_jobs(experiment_root, variants=('finetune', 'joint'), comparison=True)


def ifft_display(dates, values, keep_fraction=.2):
    """Interpolate to a uniform day grid then low-pass rFFT/irFFT for plotting."""
    dates, values = np.asarray(dates, dtype=float), np.asarray(values, dtype=float)
    if dates.ndim != 1 or values.shape != dates.shape or not len(dates):
        raise ValueError('Expected matching nonempty one-dimensional observations')
    if not 0 < keep_fraction <= 1 or not np.isfinite(values).all() or not np.isfinite(dates).all():
        raise ValueError('Expected finite values and 0 < keep_fraction <= 1')
    order = np.argsort(dates, kind='stable')
    ordered, inverse = np.unique(dates[order], return_inverse=True)
    averaged = np.bincount(inverse, weights=values[order]) / np.bincount(inverse)
    grid = np.linspace(ordered[0], ordered[-1], max(2, int(np.ceil(np.ptp(ordered))) + 1))
    interpolated = np.interp(grid, ordered, averaged)
    spectrum = np.fft.rfft(interpolated)
    spectrum[max(1, int(np.ceil(len(spectrum) * keep_fraction))):] = 0
    return grid, np.fft.irfft(spectrum, n=len(grid))


def fit_projection(features):
    matrix = np.concatenate(features).astype(np.float64)
    mean = matrix.mean(0)
    _, _, right = np.linalg.svd(matrix - mean, full_matrices=False)
    component = right[0]
    if component[np.argmax(np.abs(component))] < 0:
        component = -component
    return mean, component


def plot_pair(source, target, output, title, keep_fraction=.2):
    """Sixteen heads share both domain projection and common x/y scales."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(18, 13), sharex=True, sharey=True)
    all_values = np.concatenate([np.asarray(p['curve']) for p in (source, target)])
    span = max(float(np.ptp(all_values)), 1e-5)
    limits = (float(all_values.min() - .12 * span), float(all_values.max() + .08 * span))
    for head, ax in enumerate(axes.flat):
        for index, (name, packet, color) in enumerate((('Source', source, '#1565c0'), ('Target', target, '#d86018'))):
            dates, curve, segments = (np.asarray(packet[k]) for k in ('dates', 'curve', 'segments'))
            grid, smooth = ifft_display(dates, curve, keep_fraction)
            ax.plot(grid, smooth, color=color, lw=1.3, label=name)
            ax.scatter(dates, curve, color=color, s=8, alpha=.55)
            edges = np.r_[dates[0], (dates[:-1] + dates[1:]) / 2, dates[-1]]
            for segment in np.unique(segments[segments >= 0]):
                positions = np.flatnonzero(segments == segment)
                if packet['gate'][head][int(segment)] >= .5:
                    ax.axvspan(edges[positions[0]], edges[positions[-1]+1],
                               ymin=.015 + index*.045, ymax=.05 + index*.045,
                               color=color, alpha=.65)
                    ax.axvspan(edges[positions[0]], edges[positions[-1]+1],
                               color=color, alpha=.055)
                # Label every candidate, including rejected ones, on two domain rows.
                center = (edges[positions[0]] + edges[positions[-1]+1]) / 2
                chosen = packet['gate'][head][int(segment)] >= .5
                ax.text(center, .98-index*.085, f'S{int(segment)+1}{"+" if chosen else "-"}',
                        transform=ax.get_xaxis_transform(), ha='center', va='top',
                        fontsize=6, color=color)
            for cut in np.flatnonzero(np.diff(segments)) + 1:
                ax.axvline(edges[cut], color=color, ls='--', lw=.5, alpha=.25)
        ax.set_title(f'Head {head+1:02d}', fontsize=10)
        ax.set_ylim(*limits)
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(title + '\nShared Source-train PCA; lines: display-only iFFT; dashed: candidate boundaries; S+/shading/strips: kept; S-: rejected')
    fig.supxlabel('Original calendar day relative to Source start date (no model shift)')
    fig.supylabel('Shared PSE principal component')
    fig.tight_layout(rect=(.02, .025, 1, .95))
    fig.savefig(output.with_suffix('.png'), dpi=140)
    write_json(output.with_suffix('.json'), dict(title=title, source=source, target=target,
               keep_fraction=keep_fraction, display_only=True, y_limits=list(limits), heads=16))
    plt.close(fig)
    return 16


def _sample(dataset, index, seed):
    parcel = int(dataset.get_parcel_indices()[index])
    state, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed + parcel)
        np.random.seed((seed + parcel) % 2**32)
        return dataset[index]
    finally:
        random.setstate(state)
        np.random.set_state(np_state)


def _batch(sample, device):
    import torch
    return [torch.as_tensor(sample[key], device=device).unsqueeze(0)
            for key in ('pixels', 'valid_pixels', 'positions', 'extra')]


def _date(value):
    return dt.datetime.strptime(str(value).replace('-', '')[:8], '%Y%m%d').date()


def dataset_indices(job, domain, partition):
    """PCA uses raw UDA train; figures use the strict all-stage test holdout."""
    partitions = job.get('evaluation_splits', job['splits']) if partition == 'test' else job['splits']
    return set(partitions[domain][partition])


def load_job(job, args):
    import torch
    from dataset import PixelSetData
    from models.stclassifier import PseLTae
    from models.segment_selector import SegmentSelector
    from transforms import Normalize, RandomSamplePixels
    config = job['config']
    packet = torch.load(job['checkpoint'], map_location=args.device, weights_only=False)
    model = PseLTae(input_dim=config.get('input_dim', 10), num_classes=len(config['classes']),
                   with_extra=config.get('with_extra', False)).to(args.device).eval()
    model.load_state_dict(packet['state_dict'], strict=True)
    selector = SegmentSelector(width=config.get('selector_width', 64),
                               heads=config.get('selector_heads', 4),
                               dropout=config.get('selector_dropout', 0.),
                               protocol=config.get('selector_protocol', 'legacy')).to(args.device).eval()
    selector.load_state_dict(packet['selector_state_dict'], strict=True)
    def transform(sample):
        return Normalize()(RandomSamplePixels(config.get('num_pixels', 64))(sample))
    def dataset(domain, part):
        return PixelSetData(args.data_root, domain, config['classes'], transform=transform,
                            indices=dataset_indices(job, domain, part), closed_set=True,
                            with_extra=config.get('with_extra', False), combine_spring_and_winter=False)
    return model, selector, packet, dataset(config['source'], 'train'), dataset(config['source'], 'test'), dataset(config['target'], 'test')


def analyze_job(job, args):
    import torch
    from utils.segment_structure import structure_forward
    model, selector, checkpoint, train, source, target = load_job(job, args)
    config = job['config']
    route = job.get('route', job['variant'])
    output = Path(args.output_dir) / route / job['task']
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    train_indices = rng.permutation(len(train))[:args.pca_parcels]
    if not len(train_indices):
        raise ValueError('Source train split is empty; cannot fit a shared projection')
    features = []
    with torch.no_grad():
        for index in train_indices:
            pixels, mask, _, extra = _batch(_sample(train, int(index), args.seed), args.device)
            features.append(model.spatial_encoder(pixels.float(), mask.float(), extra.float())[0].cpu().numpy())
        mean, component = fit_projection(features)
        del features
        np.save(output / 'pca_components.npy', np.stack([mean, component]))
        origin = _date(source.metadata['start_date'])
        def collect(dataset, index, shift):
            sample = _sample(dataset, int(index), args.seed)
            pixels, mask, dates, extra = _batch(sample, args.device)
            _, keep, info = structure_forward(model, selector, pixels.float(), mask.float(),
                dates.long()+shift, extra.float(), variant=job['variant'], deterministic=True,
                full_loss=False, penalty=config.get('pelt_penalty', .1), min_size=config.get('pelt_min_size', 3),
                penalty_mode=config.get('pelt_penalty_mode', 'per_observation'),
                unified=config.get('structure_protocol', {}).get('classification_path') == 'keep_only_unified')
            result = {key: info[key][0].detach().cpu().tolist()
                      for key in ('segments', 'gate', 'probabilities', 'point_mask', 'attention')}
            result.update(parcel_id=int(sample['parcel_index']), label=int(sample['label']),
                dates=(np.asarray(sample['positions']) + (_date(dataset.metadata['start_date'])-origin).days).tolist(),
                original_positions=np.asarray(sample['positions']).tolist(), applied_model_shift=int(shift),
                curve=((info['features'][0].cpu().numpy()-mean) @ component).tolist(),
                keep_logits=keep[0].cpu().tolist(), prediction=int(keep[0].argmax()))
            return result
        coverage = {}
        for label, name in enumerate(config['classes']):
            # Reset per class, so the same held-out parcels pair across variants.
            class_rng = np.random.default_rng(args.seed + label)
            pools = [class_rng.permutation(np.flatnonzero(ds.get_labels() == label)) for ds in (source, target)]
            count = min(args.pairs_per_class, *(len(pool) for pool in pools))
            record = dict(source_test_count=len(pools[0]), target_test_count=len(pools[1]),
                          status='available' if count else 'N/A', pair_count=count, pairs=[])
            if not count:
                record['reason'] = 'class_absent_from_source_or_target_heldout_test'
            for pair in range(count):
                s = collect(source, pools[0][pair], checkpoint['source_to_target_shift'])
                t = collect(target, pools[1][pair], 0)
                filename = f'{label:02d}_{re.sub(r"[^A-Za-z0-9_-]+", "_", name)}_pair{pair+1:02d}'
                plot_pair(s, t, output / filename, f'{job["task"]} / {route} / UDA epoch20 / {name} / pair {pair+1}', args.keep_fraction)
                record['pairs'].append(dict(source_parcel_id=s['parcel_id'], target_parcel_id=t['parcel_id'],
                                            source_prediction=s['prediction'], target_prediction=t['prediction'],
                                            figure=filename+'.png', numeric=filename+'.json'))
            record['source_sampled_keep_accuracy'] = (sum(p['source_prediction'] == label for p in record['pairs']) / count if count else None)
            record['target_sampled_keep_accuracy'] = (sum(p['target_prediction'] == label for p in record['pairs']) / count if count else None)
            coverage[name] = record
            print(f'SEGMENT_CLASS|variant={job["variant"]}|task={job["task"]}|class={name}|status={record["status"]}|pairs={count}', flush=True)
    manifest = dict(task=job['task'], variant=job['variant'], route=route, seed=args.seed,
                    tested_checkpoint=job.get('tested_checkpoint'),
                    checkpoint=str(job['checkpoint']), checkpoint_sha256=checkpoint_digest(job['checkpoint']),
                    role='student', completed_epochs=20, source=config['source'], target=config['target'],
                    source_train_pca_parcel_ids=train.get_parcel_indices()[train_indices].tolist(),
                    projection='shared Source-train PSE PCA, reused for both domains and all heads',
                    sampling='same-class Source/Target held-out test only; without replacement',
                    evaluation_split_provenance=job.get('evaluation_split_provenance', 'split_indices.json'),
                    source_heldout_test_count=len(source), target_heldout_test_count=len(target),
                    source_raw_uda_test_count=len(job.get('splits', {}).get(config['source'], {}).get('test', [])),
                    display_only_ifft=True, no_imputation=True, keep_fraction=args.keep_fraction,
                    pairs_per_class=args.pairs_per_class, num_workers=args.num_workers,
                    class_coverage=coverage)
    write_json(output / 'manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment_root', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--layout', choices=('variants', 'compare'), default='variants',
                        help='compare reads finetune/joint routes, both using the INSHAPE selector')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pairs_per_class', type=int, default=3)
    parser.add_argument('--pca_parcels', type=int, default=256)
    parser.add_argument('--keep_fraction', type=float, default=.2)
    args = parser.parse_args(argv)
    experiment, output = Path(args.experiment_root).resolve(), Path(args.output_dir).resolve()
    if output == experiment or experiment in output.parents:
        raise ValueError('Visualization output must be separate from the training directory')
    if args.pairs_per_class < 1 or args.pca_parcels < 1 or not 0 < args.keep_fraction <= 1:
        raise ValueError('Positive sample counts and 0 < keep_fraction <= 1 required')
    # Even requested subsets cannot bypass the complete eight-job barrier.
    complete_jobs = (validate_comparison_jobs(experiment) if args.layout == 'compare'
                     else validate_jobs(experiment))
    jobs = [job for job in complete_jobs if job['task'] in args.tasks and job['variant'] in args.variants]
    print(f'SEGMENT_VISUALIZATION_READY|completed_jobs={len(complete_jobs)}|selected_jobs={len(jobs)}', flush=True)
    manifests = []
    for number, job in enumerate(jobs, 1):
        print(f'SEGMENT_JOB|job={number}/{len(jobs)}|route={job["route"]}|variant={job["variant"]}|task={job["task"]}', flush=True)
        manifests.append(analyze_job(job, args))
    write_json(output / 'manifest.json', dict(completed_jobs=len(complete_jobs), jobs=manifests,
                                            no_imputation=True, display_only_ifft=True))


if __name__ == '__main__':
    main()
