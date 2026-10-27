"""Standalone, read-only original TimeMatch Student diagnostic worker.

Run via ``python -m analysis.timematch_diagnostic``. Training launches this
worker with its own stdout/stderr log at completed epochs 0, 10 and 20.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from utils.head_evidence import relative_logit_contributions
from utils.timematch_diagnostic import (
    CLASSES, INTERVENTIONS, STAGES, STRUCTURES, counterfactual_embeddings,
    fourier_display, load_final_student, select_structure,
)


def write_json(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(content, indent=2), encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(root, source, seed):
    """Validate the existing closed-set source; never silently retrain it."""
    root = Path(root)
    config = json.loads((root / 'train_config.json').read_text(encoding='utf-8'))
    protocol = json.loads((root / 'closed_set_protocol.json').read_text(encoding='utf-8'))
    required = {'source': source, 'target': source, 'model': 'pseltae',
                'seed': seed, 'epochs': 100, 'closed_set': True,
                'combine_spring_and_winter': False, 'input_dim': 10,
                'with_extra': False}
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f'Source config mismatch: {key}={config.get(key)!r}, expected {expected!r}')
    if config.get('method') is not None:
        raise ValueError('Expected a source-supervised checkpoint, not an adapted model')
    if protocol.get('source') != source or not protocol.get('closed_set'):
        raise ValueError('Source closed-set protocol mismatch')
    missing = set(CLASSES) - set(protocol['classes'])
    if missing:
        raise ValueError(f'Requested diagnostic classes absent from source classifier: {sorted(missing)}')
    checkpoint = root / 'fold_0/model.pt'
    packet = torch.load(checkpoint, map_location='cpu', weights_only=False)
    # Strict loading catches model/class-count/config incompatibilities.
    from models.stclassifier import PseLTae
    PseLTae(num_classes=len(protocol['classes']), with_extra=False).load_state_dict(packet['state_dict'])
    print(f'SOURCE_VALID|path={checkpoint}|sha256={sha256(checkpoint)}', flush=True)


def _dataset(config, splits, domain, partition):
    from dataset import PixelSetData
    from transforms import Normalize, RandomSamplePixels

    # ToTensor in old project versions uses np.long; explicit as_tensor below
    # supports both old server NumPy and the newer local book environment.
    def transform(sample):
        return Normalize()(RandomSamplePixels(config['num_pixels'])(sample))

    return PixelSetData(
        config['data_root'], domain, config['classes'], transform,
        indices=splits[domain][partition], closed_set=True,
        combine_spring_and_winter=False,
    )


def _sample(dataset, index, seed):
    # Re-select exactly the same pixels at every epoch, even if parcel counts
    # are below 100 or batch processing order changes.
    parcel = int(dataset.samples[index][1])
    random.seed(int(seed) + parcel)
    np.random.seed((int(seed) + parcel) % (2**32))
    return dataset[index]


def _batch(samples, device):
    return [torch.stack([torch.as_tensor(sample[key]) for sample in samples]).to(device)
            for key in ('pixels', 'valid_pixels', 'positions', 'extra')]


def _fixed_pca(model, source_train, seed, device, path):
    """Fit once on this UDA run's Source Train; reuse across domains/epochs."""
    if path.exists():
        packet = np.load(path)
        return packet['mean'], packet['component']
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(source_train))[:100]
    features = []
    for index in indices:
        pixels, valid, _, extra = _batch([_sample(source_train, int(index), seed)], device)
        features.append(model.spatial_encoder(pixels, valid, extra)[0].cpu().numpy())
    matrix = np.concatenate(features).astype(np.float64)
    mean = matrix.mean(0)
    _, _, right = np.linalg.svd(matrix - mean, full_matrices=False)
    component = right[0]
    if component[np.argmax(np.abs(component))] < 0:
        component = -component
    np.savez_compressed(path, mean=mean, component=component,
                        source_train_parcels=source_train.get_parcel_indices()[indices])
    return mean, component


def _selection(dataset, class_names, seed):
    labels = dataset.get_labels()
    rng = np.random.default_rng(seed)
    selected = {}
    for name in CLASSES:
        if name not in class_names:
            raise ValueError(f'Diagnostic class {name} is not a source classifier output')
        pool = np.flatnonzero(labels == class_names.index(name))
        if not len(pool):
            raise ValueError(f'No held-out {name} samples in {dataset.dataset_name}')
        selected[name] = rng.permutation(pool)[:100].tolist()
    return selected


@torch.no_grad()
def analyze_domain(model, dataset, selected, config, shift, mean, component, device, output):
    """Per-sample interventions first, then average true-class relative margins."""
    output.mkdir(parents=True, exist_ok=True)
    means = np.empty((3, 3, 3, 16))
    errors = np.empty_like(means)
    counts = []
    representatives = []
    for class_index, name in enumerate(CLASSES):
        all_attention, all_masks, all_scores, all_logits, parcels = [], [], [], [], []
        indices = selected[name]
        representative = None
        for start in range(0, len(indices), 8):
            samples = [_sample(dataset, index, config['seed']) for index in indices[start:start+8]]
            pixels, valid, positions, extra = _batch(samples, device)
            positions = positions.long()
            spatial = model.spatial_encoder(pixels.float(), valid.float(), extra.float())
            temporal, attention, values = model.temporal_encoder(
                spatial, positions + int(shift), return_head_parts=True,
            )
            logits = model.decoder(temporal)
            attention_np, dates = attention.cpu().numpy(), positions.cpu().numpy()
            masks_np = np.array([
                [[select_structure(head, date, method) for head in parcel]
                 for parcel, date in zip(attention_np, dates)]
                for method in STRUCTURES
            ])
            scores = np.empty((3, 3, len(samples), 16))
            labels = torch.full((len(samples),), config['classes'].index(name),
                                dtype=torch.long, device=device)
            for structure_index in range(3):
                mask = torch.as_tensor(masks_np[structure_index], device=device)
                for intervention_index, intervention in enumerate(INTERVENTIONS):
                    embeddings = counterfactual_embeddings(attention, values, mask, intervention)
                    features = model.temporal_encoder.mlp(embeddings.flatten(0, 1))
                    altered_logits = model.decoder(features).reshape(len(samples), 17, -1)
                    if not torch.allclose(altered_logits[:, 0], logits, atol=1e-5, rtol=1e-4):
                        raise ValueError('Unmodified head embedding does not reproduce original logits')
                    scores[structure_index, intervention_index] = relative_logit_contributions(
                        altered_logits, labels,
                    ).cpu().numpy()
            if representative is None:
                signal = (spatial[0].cpu().numpy() - mean) @ component
                grid, curve, observed = fourier_display(dates[0], signal)
                representative = dict(name=name, parcel=int(samples[0]['parcel_index']),
                                      dates=dates[0], attention=attention_np[0],
                                      masks=masks_np[:, 0], grid=grid, curve=curve,
                                      observed=observed, shift=int(shift))
            all_attention.append(attention_np)
            all_masks.append(masks_np)
            all_scores.append(scores)
            all_logits.append(logits.cpu().numpy())
            parcels.extend(int(sample['parcel_index']) for sample in samples)
        attention_np = np.concatenate(all_attention)
        masks_np = np.concatenate(all_masks, axis=1)
        scores = np.concatenate(all_scores, axis=2)
        means[:, :, class_index] = scores.mean(axis=2)
        errors[:, :, class_index] = (scores.std(axis=2, ddof=1)/np.sqrt(len(indices))
                                   if len(indices) > 1 else 0.)
        counts.append(len(indices))
        np.savez_compressed(output / f'{name}.npz',
                            parcel_ids=parcels, positions=dates[0],
                            attention=attention_np, structures=masks_np,
                            contributions=scores, original_logits=np.concatenate(all_logits),
                            structure_methods=STRUCTURES, interventions=INTERVENTIONS,
                            class_names=config['classes'], applied_shift=int(shift),
                            grid=representative['grid'], curve=representative['curve'],
                            observed=representative['observed'])
        representatives.append(representative)
        print(f'CLASS_COMPLETE|domain={dataset.dataset_name}|class={name}|n={len(indices)}', flush=True)
    return dict(representatives=representatives, means=means, errors=errors, counts=counts)


def plot_stage(task, stage, domains, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    colors = ('tab:blue', 'tab:orange', 'tab:green')
    structure_files, contribution_files = [], []
    rows = []
    for domain, results in domains.items():
        for structure_index, method in enumerate(STRUCTURES):
            folder = output / domain / method
            folder.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(4, 4, figsize=(16, 13), sharex=True, sharey=True)
            for head, axis in enumerate(axes.flat):
                counts = []
                for sample, color in zip(results['representatives'], colors):
                    mask = sample['masks'][structure_index, head]
                    axis.plot(sample['grid'], sample['curve'], color=color, lw=1.2,
                              label=f"{sample['name']} parcel={sample['parcel']}")
                    axis.scatter(sample['dates'][mask], sample['observed'][mask],
                                 color=color, edgecolor='black', linewidth=.3, s=22, zorder=3)
                    counts.append(int(mask.sum()))
                axis.set_title(f'Head {head+1} | selected={counts}', fontsize=9)
                axis.grid(alpha=.2)
            axes[0, 0].legend(fontsize=6)
            shift = results['representatives'][0]['shift']
            fig.suptitle(f'{task} | epoch {stage} | {domain} | {method} | Student shift={shift}')
            fig.supxlabel('Original observation day')
            fig.supylabel('Fixed epoch-0 Source-Train PSE PC1 (Fourier display only)')
            fig.tight_layout(rect=(.02, .02, 1, .97))
            path = folder / 'structures.png'
            fig.savefig(path, dpi=120)
            plt.close(fig)
            structure_files.append(str(path.relative_to(output)))
            for intervention_index, intervention in enumerate(INTERVENTIONS):
                fig, axis = plt.subplots(figsize=(15, 5))
                for class_index, (name, color) in enumerate(zip(CLASSES, colors)):
                    values = results['means'][structure_index, intervention_index, class_index]
                    errors = results['errors'][structure_index, intervention_index, class_index]
                    axis.bar(np.arange(16)+(class_index-1)*.25, values, .25,
                             yerr=errors, capsize=2, color=color,
                             label=f"{name} (n={results['counts'][class_index]})")
                    for head, (value, error) in enumerate(zip(values, errors)):
                        rows.append(dict(task=task, epoch=stage, domain=domain, structure=method,
                                         intervention=intervention, class_name=name, head=head+1,
                                         mean=float(value), standard_error=float(error),
                                         count=results['counts'][class_index]))
                axis.axhline(0, color='black', lw=.8)
                axis.set_xticks(np.arange(16))
                axis.set_xticklabels([f'H{head+1}' for head in range(16)])
                axis.set_ylabel('Original minus intervened class-vs-best-other logit margin')
                axis.set_title(f'{task} | epoch {stage} | {domain} | {method} | {intervention}')
                axis.legend(fontsize=9)
                axis.grid(axis='y', alpha=.2)
                fig.tight_layout()
                path = folder / f'contributions_{intervention}.png'
                fig.savefig(path, dpi=120)
                plt.close(fig)
                contribution_files.append(str(path.relative_to(output)))
    with (output / 'contributions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return structure_files, contribution_files


@torch.no_grad()
def render(snapshot, output, task):
    from models.stclassifier import PseLTae
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str((output / 'matplotlib_cache').resolve()))
    packet = torch.load(snapshot, map_location='cpu', weights_only=False)
    config = packet['config']
    stage = packet['completed_epochs']
    if packet['role'] != 'student' or stage not in STAGES:
        raise ValueError('Expected a staged original TimeMatch Student snapshot')
    device = torch.device(config['device'])
    model = PseLTae(input_dim=config['input_dim'], num_classes=config['num_classes'],
                   with_extra=config['with_extra']).to(device).eval()
    model.load_state_dict(packet['state_dict'])
    source_train = _dataset(config, packet['splits'], config['source'], 'train')
    pca_path = output / 'source_train_epoch00_pca.npz'
    if stage != 0 and not pca_path.exists():
        raise ValueError('Missing fixed epoch-0 Source Train PCA')
    mean, component = _fixed_pca(model, source_train, config['seed'], device, pca_path)
    domains, chosen = {}, {}
    stage_dir = output / f'epoch_{stage:02d}'
    selection_path = output / 'fixed_samples.json'
    previous = json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.exists() else None
    if stage != 0 and previous is None:
        raise ValueError('Missing fixed diagnostic samples from epoch 0')
    for domain_index, domain in enumerate(('source', 'target')):
        dataset = _dataset(config, packet['splits'], config[domain], 'test')
        selected = _selection(dataset, config['classes'], config['seed'] + domain_index)
        chosen[domain] = {
            name: [int(dataset.samples[index][1]) for index in selected[name]] for name in CLASSES
        }
        if previous is not None and previous[domain] != chosen[domain]:
            raise ValueError('Diagnostic sample identities changed between stages')
        domains[domain] = analyze_domain(
            model, dataset, selected, config, packet['student_shifts'][domain],
            mean, component, device, stage_dir / 'data' / domain,
        )
    write_json(selection_path, chosen)
    structure_files, contribution_files = plot_stage(task, stage, domains, stage_dir)
    if len(structure_files) != 6 or len(contribution_files) != 18:
        raise ValueError('Wrong diagnostic figure count')
    write_json(stage_dir / 'manifest.json', dict(
        task=task, completed_epochs=stage, role='student',
        snapshot=str(snapshot), snapshot_sha256=sha256(snapshot),
        student_shifts=packet['student_shifts'], fixed_sample_split='current UDA test split',
        source_pretraining_split_disjointness='not asserted; original source splitting is unchanged',
        sample_counts={domain: result['counts'] for domain, result in domains.items()},
        structure_figures=structure_files, contribution_figures=contribution_files,
        attention='pre-dropout, real observations only',
        selection='Top30 intersect shortest 75%-mass/count interval, or Top30',
        pca='fixed epoch-0 Source Train; Fourier 13 modes, reg=1e-3, period=365, display only',
    ))
    print(f'STAGE_VERIFIED|task={task}|epoch={stage}|structure_figures=6|contribution_figures=18', flush=True)


def verify_task(root, task, training_dir):
    root, training_dir = Path(root), Path(training_dir)
    for stage in STAGES:
        stage_dir = root / f'epoch_{stage:02d}'
        manifest = json.loads((stage_dir / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['task'] != task or manifest['completed_epochs'] != stage:
            raise ValueError('Diagnostic stage metadata mismatch')
        for key, count in (('structure_figures', 6), ('contribution_figures', 18)):
            files = manifest[key]
            if len(set(files)) != count or any(not (stage_dir / file).is_file() for file in files):
                raise ValueError(f'Incomplete diagnostic figures: {task} epoch {stage}')
        if manifest['snapshot_sha256'] != sha256(root / 'checkpoints' / f'checkpoint_epoch{stage:02d}.pt'):
            raise ValueError('Diagnostic checkpoint changed after analysis')
    final = load_final_student(root / 'checkpoints/checkpoint_epoch20.pt')
    tested = json.loads((training_dir / 'tested_checkpoint.json').read_text(encoding='utf-8'))
    if tested.get('completed_epochs') != 20 or tested.get('sha256') != sha256(root / 'checkpoints/checkpoint_epoch20.pt'):
        raise ValueError('Final test did not use the diagnostic 20-epoch Student')
    exported = torch.load(training_dir / 'model.pt', map_location='cpu', weights_only=False)
    if exported.get('completed_epochs') != 20 or exported.get('role') != 'student':
        raise ValueError('Exported final model is not the 20-epoch Student')
    if any(not torch.equal(value, exported['state_dict'][key]) for key, value in final['state_dict'].items()):
        raise ValueError('Final exported and tested Student parameters differ')
    print(f'TASK_VERIFIED|task={task}|tested_epoch=20|structure_figures=18|contribution_figures=54', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    source = subparsers.add_parser('validate-source')
    source.add_argument('--root', required=True)
    source.add_argument('--source', required=True)
    source.add_argument('--seed', type=int, required=True)
    worker = subparsers.add_parser('render')
    worker.add_argument('--snapshot', required=True)
    worker.add_argument('--output', required=True)
    worker.add_argument('--task', required=True)
    verify = subparsers.add_parser('verify-task')
    verify.add_argument('--root', required=True)
    verify.add_argument('--task', required=True)
    verify.add_argument('--training-dir', required=True)
    args = parser.parse_args()
    if args.command == 'validate-source':
        validate_source(args.root, args.source, args.seed)
    elif args.command == 'render':
        render(args.snapshot, args.output, args.task)
    else:
        verify_task(args.root, args.task, args.training_dir)


if __name__ == '__main__':
    main()
