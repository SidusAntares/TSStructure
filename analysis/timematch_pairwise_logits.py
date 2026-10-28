"""Read-only class/head and pairwise evidence audit with complete sample logits."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import random

import numpy as np
import torch

from analysis.timematch_diagnostic import _batch, _dataset, _sample, sha256, write_json
from utils.head_evidence import build_counterfactual_head_embeddings, select_attention_top_fraction
from utils.timematch_diagnostic import load_final_student


TASKS = {
    'AT1_DK1': ('austria/33UVP/2017', 'denmark/32VNH/2017'),
    'FR1_FR2': ('france/30TXT/2017', 'france/31TCJ/2017'),
    'FR2_DK1': ('france/31TCJ/2017', 'denmark/32VNH/2017'),
    'DK1_AT1': ('denmark/32VNH/2017', 'austria/33UVP/2017'),
}


def derive_contributions(original, masked, labels):
    """Keep every logit; pairwise c-vs-k differs from c-vs-best-other evidence."""
    original, masked, labels = np.asarray(original), np.asarray(masked), np.asarray(labels)
    if original.ndim != 2 or masked.ndim != 3 or masked.shape[::2] != original.shape:
        raise ValueError('Expected original[N,C] and masked[N,H,C]')
    if original.shape[1] < 2 or labels.shape != (len(original),):
        raise ValueError('Expected at least two classifier outputs and one true label per sample')
    own = original[np.arange(len(labels)), labels]
    masked_own = np.take_along_axis(masked, labels[:, None, None], axis=2)[..., 0]
    original_pairs = own[:, None]-original
    masked_pairs = masked_own[..., None]-masked
    competitor = original.copy()
    competitor[np.arange(len(labels)), labels] = -np.inf
    masked_competitor = masked.copy()
    np.put_along_axis(masked_competitor, labels[:, None, None], -np.inf, axis=2)
    return dict(
        delta_logits=original[:, None]-masked,
        pairwise_contributions=original_pairs[:, None]-masked_pairs,
        relative_margin_contributions=(own-competitor.max(1))[:, None]
                                      -(masked_own-masked_competitor.max(2)),
        original_pair_margins=original_pairs, masked_pair_margins=masked_pairs,
        original_predictions=original.argmax(1), masked_predictions=masked.argmax(2),
    )


def resolve_student_shift(packet, config, override=None, events=None):
    """Student Source shift is fixed at epoch 0; later Teacher shifts are different."""
    if override is not None:
        return int(override), 'explicit_override'
    if 'student_shifts' in packet:
        if packet['student_shifts']['target'] != 0:
            raise ValueError('Original TimeMatch Student target shift must be zero')
        return int(packet['student_shifts']['source']), 'checkpoint_student_shifts'
    if not config.get('shift_source', True) or not config.get('estimate_shift', True):
        return 0, 'training_config_shift_disabled'
    if events is None:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            path = str(config['tensorboard_log_dir']) + '_fold0'
            accumulator = EventAccumulator(path, size_guidance={'scalars': 0})
            accumulator.Reload()
            events = [(event.step, event.value) for event in accumulator.Scalars('train/temporal_shift')]
        except Exception as error:
            raise ValueError('Cannot recover Student source shift: supply the original TensorBoard '
                             'events or --student-source-shift; never assume zero') from error
    initial = {float(value) for step, value in events if step == 0}
    if len(initial) != 1 or not next(iter(initial)).is_integer():
        raise ValueError('Expected exactly one integer epoch0 train/temporal_shift value')
    return -int(initial.pop()), 'tensorboard_train_temporal_shift_epoch0_negated'


def reconstruct_uda_splits(config, datasets):
    """Reproduce existing seed-based UDA splits without altering global RNG state."""
    rng = random.Random(config['seed'])
    splits = {}
    for domain in (config['source'], config['target']):
        indices = datasets[domain].get_parcel_indices().tolist()
        rng.shuffle(indices)
        n_test, n_val = int(config['test_ratio']*len(indices)), int(config['val_ratio']*len(indices))
        n_train = len(indices)-n_test-n_val
        splits[domain] = dict(train=set(indices[:n_train]),
                              val=set(indices[n_train:n_train+n_val]), test=set(indices[n_train+n_val:]))
    return splits


@torch.no_grad()
def collect_logits(model, dataset, indices, seed, shift, device):
    if model.training or model.temporal_encoder.attention_heads.n_head != 16:
        raise ValueError('Diagnostic requires an eval-mode 16-head PSE-LTAE')
    pieces = {key: [] for key in ('original_logits', 'masked_logits', 'labels', 'parcel_ids',
                                  'positions', 'attention', 'structure_mask')}
    for start in range(0, len(indices), 8):
        samples = [_sample(dataset, index, seed) for index in indices[start:start+8]]
        pixels, valid, positions, extra = _batch(samples, device)
        positions = positions.long()
        spatial = model.spatial_encoder(pixels.float(), valid.float(), extra.float())
        temporal, attention, values = model.temporal_encoder(spatial, positions+shift, return_head_parts=True)
        original = model.decoder(temporal)
        selected, _ = select_attention_top_fraction(attention, positions, .3)
        embeddings = build_counterfactual_head_embeddings(attention, values, selected,
                                                          intervention='gate', gate=.5)
        logits = model.decoder(model.temporal_encoder.mlp(embeddings.flatten(0, 1))).reshape(len(samples), 17, -1)
        if not torch.allclose(original, logits[:, 0], atol=1e-5, rtol=1e-4):
            raise ValueError('Original logits do not match unmodified head pooling')
        pieces['original_logits'].append(original.cpu().numpy())
        pieces['masked_logits'].append(logits[:, 1:].cpu().numpy())
        pieces['labels'].append(np.array([int(sample['label']) for sample in samples]))
        pieces['parcel_ids'].append(np.array([int(sample['parcel_index']) for sample in samples]))
        pieces['positions'].append(positions.cpu().numpy())
        pieces['attention'].append(attention.cpu().numpy())
        pieces['structure_mask'].append(selected.cpu().numpy())
    if indices:
        arrays = {key: np.concatenate(value) for key, value in pieces.items()}
    else:
        classes = model.decoder[-1].out_features
        arrays = dict(original_logits=np.empty((0, classes)), masked_logits=np.empty((0, 16, classes)),
                      labels=np.empty(0, dtype=int), parcel_ids=np.empty(0, dtype=int),
                      positions=np.empty((0, 0)), attention=np.empty((0, 16, 0)),
                      structure_mask=np.empty((0, 16, 0), dtype=bool))
    if not np.isfinite(arrays['original_logits']).all() or not np.isfinite(arrays['masked_logits']).all():
        raise ValueError('Nonfinite model logits')
    arrays['applied_shift'] = np.array(shift)
    arrays.update(derive_contributions(arrays['original_logits'], arrays['masked_logits'], arrays['labels']))
    return arrays


def export_logits(output, arrays, class_names, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'sample_logits.npz', **arrays, class_names=class_names)
    fields = ['task', 'epoch', 'domain', 'parcel_id', 'true_class', 'case', 'prediction', 'correct']
    fields += [f'logit::{name}' for name in class_names]
    with gzip.open(output / 'sample_logits.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, label in enumerate(arrays['labels']):
            for head in range(17):
                logits = arrays['original_logits'][index] if head == 0 else arrays['masked_logits'][index, head-1]
                prediction = int(logits.argmax())
                row = dict(task=metadata['task'], epoch=metadata['epoch'], domain=metadata['domain'],
                           parcel_id=int(arrays['parcel_ids'][index]), true_class=class_names[label],
                           case='original' if head == 0 else f'head_{head:02d}',
                           prediction=class_names[prediction], correct=prediction == label)
                row.update({f'logit::{name}': float(value) for name, value in zip(class_names, logits)})
                writer.writerow(row)
    write_json(output / 'logit_schema.json', dict(
        **metadata, class_names=class_names, sample_count=len(arrays['labels']),
        csv_rows=len(arrays['labels'])*17, original_shape=list(arrays['original_logits'].shape),
        masked_shape=list(arrays['masked_logits'].shape), gate=.5, top_fraction=.3,
        delta_logits='original minus masked, all output classes retained',
        sampling='fixed UDA-test parcels, seed+parcel pixel sampling; at most 100 per class',
    ))


def summarize_contributions(arrays, classes, min_mean=.05, min_rate=.65, min_count=16):
    counts = np.bincount(arrays['labels'], minlength=classes)
    class_mean = np.full((classes, 16), np.nan)
    class_rate = np.full_like(class_mean, np.nan)
    pair_mean = np.full((classes, classes, 16), np.nan)
    pair_rate = np.full_like(pair_mean, np.nan)
    flip_rate = np.full_like(pair_mean, np.nan)
    accuracy = np.full(classes, np.nan)
    for label in range(classes):
        mask = arrays['labels'] == label
        if not mask.any():
            continue
        scores = arrays['relative_margin_contributions'][mask]
        class_mean[label], class_rate[label] = scores.mean(0), (scores > 0).mean(0)
        pairs = arrays['pairwise_contributions'][mask]
        pair_mean[label], pair_rate[label] = pairs.mean(0).T, (pairs > 0).mean(0).T
        wins = arrays['original_pair_margins'][mask] > 0
        flips = ((arrays['masked_pair_margins'][mask] <= 0) & wins[:, None]).sum(0).T
        flip_rate[label] = np.divide(flips, wins.sum(0)[:, None],
                                     out=np.full((classes, 16), np.nan), where=wins.sum(0)[:, None] > 0)
        accuracy[label] = (arrays['original_logits'][mask].argmax(1) == label).mean()
    reliable = (pair_mean >= min_mean) & (pair_rate >= min_rate) & (counts[:, None, None] >= min_count)
    evaluable = (counts[:, None] > 0) & (counts[None, :] > 0) & ~np.eye(classes, dtype=bool)
    reliable &= evaluable[..., None]
    for label in range(classes):
        pair_mean[label, label] = pair_rate[label, label] = flip_rate[label, label] = np.nan
    return dict(counts=counts, class_mean=class_mean, class_positive_rate=class_rate,
                class_reliable=(class_mean >= min_mean) & (class_rate >= min_rate) & (counts[:, None] >= min_count),
                accuracy=accuracy, pair_mean=pair_mean, pair_positive_rate=pair_rate,
                pair_flip_rate=flip_rate, pair_reliable=reliable,
                directed_head_counts=np.where(evaluable, reliable.sum(2), np.nan),
                mutual_head_counts=np.where(evaluable, (reliable & reliable.transpose(1, 0, 2)).sum(2), np.nan))


def write_statistics(output, summary, names):
    output = Path(output)
    np.savez_compressed(output / 'aggregate_evidence.npz', **summary, class_names=names)
    def number(value):
        return float(value) if np.isfinite(value) else 'N/A'
    with (output / 'class_head_statistics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['class', 'head', 'mean', 'positive_rate', 'samples', 'accuracy', 'reliable'])
        for c, name in enumerate(names):
            for h in range(16):
                writer.writerow([name, h+1, number(summary['class_mean'][c, h]),
                                 number(summary['class_positive_rate'][c, h]), int(summary['counts'][c]),
                                 number(summary['accuracy'][c]), bool(summary['class_reliable'][c, h])
                                 if summary['counts'][c] else 'N/A'])
    with gzip.open(output / 'pair_head_statistics.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['class', 'competitor', 'head', 'mean', 'positive_rate',
                         'flip_rate_given_original_pair_win', 'samples', 'competitor_samples', 'reliable', 'status'])
        for c, name in enumerate(names):
            for k, competitor in enumerate(names):
                if c == k:
                    continue
                valid = summary['counts'][c] > 0 and summary['counts'][k] > 0
                for h in range(16):
                    writer.writerow([name, competitor, h+1,
                                     number(summary['pair_mean'][c, k, h]) if valid else 'N/A',
                                     number(summary['pair_positive_rate'][c, k, h]) if valid else 'N/A',
                                     number(summary['pair_flip_rate'][c, k, h]) if valid else 'N/A',
                                     int(summary['counts'][c]), int(summary['counts'][k]),
                                     bool(summary['pair_reliable'][c, k, h]) if valid else 'N/A',
                                     'available' if valid else 'N/A: class or competitor absent'])
    coverage = []
    for c, name in enumerate(names):
        observed = [k for k in range(len(names)) if k != c and summary['counts'][k] > 0]
        present = bool(summary['counts'][c])
        covered = [names[k] for k in observed if summary['directed_head_counts'][c, k] > 0]
        mutual = [names[k] for k in observed if summary['mutual_head_counts'][c, k] > 0]
        all_observed = len(covered) == len(observed) if present and observed else None
        coverage.append(dict(class_name=name, sample_count=int(summary['counts'][c]),
                             supported_competitors=covered, mutual_same_head_competitors=mutual,
                             observed_competitors=[names[k] for k in observed],
                             all_observed_competitors_supported=all_observed,
                             all_model_competitors_supported=all_observed
                             if len(observed) == len(names)-1 else None))
    write_json(output / 'class_coverage.json', coverage)


def plot_heatmaps(output, summary, names, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    specifications = (
        ('class_head_mean.png', summary['class_mean'], [f'H{h+1}' for h in range(16)],
         'Class-vs-best-other mean margin contribution', True),
        ('pair_head_count.png', summary['directed_head_counts'], names,
         'Directed class-vs-competitor reliable head count', False),
        ('pair_mutual_head_count.png', summary['mutual_head_counts'], names,
         'Same head supporting both directions of a class pair', False),
    )
    for filename, matrix, columns, description, signed in specifications:
        fig, axis = plt.subplots(figsize=(max(10, len(columns)*.55), max(6, len(names)*.42)))
        palette = plt.get_cmap('coolwarm' if signed else 'Blues').copy()
        palette.set_bad('#e3e3e3')
        maximum = max(float(np.abs(matrix[np.isfinite(matrix)]).max()), 1e-6) if np.isfinite(matrix).any() else 1.
        image = axis.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap=palette,
                            vmin=-maximum if signed else 0, vmax=maximum if signed else 16)
        axis.set_xticks(np.arange(len(columns)))
        axis.set_xticklabels(columns, rotation=90, fontsize=8)
        axis.set_yticks(np.arange(len(names)))
        axis.set_yticklabels([f'{name}\n(n={int(count)})' for name, count in zip(names, summary['counts'])], fontsize=8)
        axis.set_xlabel('Head' if signed else 'Competitor class')
        axis.set_ylabel('True class')
        for c in range(len(names)):
            for k in range(len(columns)):
                value = matrix[c, k]
                text = f'{value:.3g}' if signed else f'{value:.0f}'
                axis.text(k, c, text if np.isfinite(value) else 'N/A', ha='center', va='center', fontsize=6)
        axis.set_title(f'{title}\n{description}')
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)


def load_task(source_root, uda_root, task):
    source, target = TASKS[task]
    source_tile, target_tile = source.split('/')[1], target.split('/')[1]
    source_dir = Path(source_root) / f'closedset_source_{source_tile}_seed1'
    uda_dir = Path(uda_root) / f'closedset_timematch_{source_tile}_to_{target_tile}_seed1'
    source_config = json.loads((source_dir / 'train_config.json').read_text(encoding='utf-8'))
    config = json.loads((uda_dir / 'train_config.json').read_text(encoding='utf-8'))
    source_protocol = json.loads((source_dir / 'closed_set_protocol.json').read_text(encoding='utf-8'))
    protocol = json.loads((uda_dir / 'closed_set_protocol.json').read_text(encoding='utf-8'))
    for cfg, expected in ((source_config, dict(source=source, target=source, epochs=100, method=None)),
                          (config, dict(source=source, target=target, epochs=20, method='timematch', output_student=True))):
        expected.update(seed=1, model='pseltae', closed_set=True, combine_spring_and_winter=False, num_folds=1)
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise ValueError(f'{task}: checkpoint config mismatch: {key}={cfg.get(key)!r}, expected {value!r}')
    if config.get('evidence_mode', 'off') != 'off':
        raise ValueError('This audit requires original TimeMatch, not A/B evidence training')
    if source_protocol['classes'] != protocol['classes'] or len(protocol['classes']) < 2:
        raise ValueError('Source and UDA classifier label order differs')
    if Path(config['weights']).resolve() != source_dir.resolve():
        raise ValueError('UDA was initialized from a different Source directory')
    for key in ('input_dim', 'with_extra'):
        if source_config[key] != config[key]:
            raise ValueError(f'Source and UDA model configuration differs: {key}')
    source_checkpoint = source_dir / 'fold_0/model.pt'
    final_checkpoint = uda_dir / 'fold_0/model.pt'
    final = load_final_student(final_checkpoint)
    initial = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
    config.update(classes=protocol['classes'], num_classes=len(protocol['classes']))
    return config, initial, final, source_checkpoint, final_checkpoint


@torch.no_grad()
def run_task(args):
    from dataset import PixelSetData
    from models.stclassifier import PseLTae
    config, initial, final, initial_path, final_path = load_task(args.source_root, args.uda_root, args.task)
    config['data_root'] = args.data_root
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    import os
    os.environ['MPLCONFIGDIR'] = str((output / 'matplotlib_cache').resolve())
    shift, shift_provenance = resolve_student_shift(final, config, args.student_source_shift)
    domains = {domain: PixelSetData(args.data_root, domain, config['classes'],
                                    closed_set=True, combine_spring_and_winter=False)
               for domain in (config['source'], config['target'])}
    splits = reconstruct_uda_splits(config, domains)
    datasets = {name: _dataset(config, splits, config[name], 'test') for name in ('source', 'target')}
    selections, fixed_samples = {}, {}
    for domain_index, (name, dataset) in enumerate(datasets.items()):
        rng = np.random.default_rng(config['seed']+domain_index)
        labels = dataset.get_labels()
        selections[name], fixed_samples[name] = [], {}
        for label, class_name in enumerate(config['classes']):
            indices = rng.permutation(np.flatnonzero(labels == label))[:100].tolist()
            selections[name].extend(indices)
            fixed_samples[name][class_name] = [int(dataset.samples[index][1]) for index in indices]
    write_json(output / 'fixed_samples.json', fixed_samples)
    device = torch.device(args.device)
    model = PseLTae(input_dim=config['input_dim'], num_classes=config['num_classes'],
                   with_extra=config['with_extra']).to(device).eval()
    cases = []
    thresholds = dict(min_mean=args.min_mean, min_rate=args.min_rate, min_count=args.min_count)
    for epoch, packet, checkpoint in ((0, initial, initial_path), (20, final, final_path)):
        model.load_state_dict(packet['state_dict'])
        for name, dataset in datasets.items():
            applied_shift = shift if epoch == 20 and name == 'source' else 0
            case_output = output / f'epoch_{epoch:02d}' / name
            metadata = dict(task=args.task, epoch=epoch, domain=name, dataset=config[name],
                            checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                            checkpoint_role='source_supervised' if epoch == 0 else 'student',
                            applied_shift=applied_shift,
                            shift_provenance=(shift_provenance if name == 'source' else 'student_target_native_coordinates_no_shift')
                            if epoch == 20 else 'native_source_model_coordinates_no_shift',
                            seed=1, num_pixels=config['num_pixels'], data_root=args.data_root,
                            torch_version=str(torch.__version__), numpy_version=np.__version__, thresholds=thresholds,
                            contribution='original minus masked relative logit margin',
                            interpretation='Descriptive intervention evidence, not standalone head accuracy or statistical significance',
                            split='current UDA test; source pretraining disjointness not asserted')
            arrays = collect_logits(model, dataset, selections[name], 1, applied_shift, device)
            export_logits(case_output, arrays, config['classes'], metadata)
            summary = summarize_contributions(arrays, config['num_classes'], **thresholds)
            write_statistics(case_output, summary, config['classes'])
            plot_heatmaps(case_output, summary, config['classes'],
                          f'{args.task} | epoch {epoch} | {name} | Top30, g=0.5 | shift={applied_shift}')
            cases.append(dict(epoch=epoch, domain=name, samples=len(arrays['labels']),
                              csv_rows=len(arrays['labels'])*17,
                              logit_shape=list(arrays['masked_logits'].shape),
                              absent_classes=[class_name for class_name, count in zip(config['classes'], summary['counts']) if count == 0]))
            print(f'CASE_COMPLETE|task={args.task}|epoch={epoch}|domain={name}|samples={len(arrays["labels"])}', flush=True)
    write_json(output / 'manifest.json', dict(task=args.task, cases=cases, class_names=config['classes'],
                                             source_checkpoint_sha256=sha256(initial_path),
                                             final_checkpoint_sha256=sha256(final_path),
                                             final_completed_epochs=20, role='student', thresholds=thresholds))
    print(f'TASK_COMPLETE|task={args.task}|cases=4|complete_logits=true', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', default='outputs/timematch_historical_rerun_seed1/source')
    parser.add_argument('--uda-root', default='outputs/timematch_historical_rerun_seed1/uda')
    parser.add_argument('--data-root', default='/data/user/dataset/timematch_data')
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--student-source-shift', type=int)
    parser.add_argument('--min-mean', type=float, default=.05)
    parser.add_argument('--min-rate', type=float, default=.65)
    parser.add_argument('--min-count', type=int, default=16)
    args = parser.parse_args()
    if args.min_mean < 0 or not 0 <= args.min_rate <= 1 or args.min_count < 1:
        parser.error('Invalid descriptive support thresholds')
    run_task(args)


if __name__ == '__main__':
    main()
