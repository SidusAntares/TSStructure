"""Read-only, frozen Original Source PSE features for PELT diagnostics.

Heavy inference dependencies are imported only when extraction is requested.
No training entry point, target dataset, or temporal augmentation is used.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace


TASKS = {
    'AT1_DK1': ('austria/33UVP/2017', 'denmark/32VNH/2017'),
    'FR1_FR2': ('france/30TXT/2017', 'france/31TCJ/2017'),
    'FR2_DK1': ('france/31TCJ/2017', 'denmark/32VNH/2017'),
    'DK1_AT1': ('denmark/32VNH/2017', 'austria/33UVP/2017'),
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_original_source(leaf, source):
    """Reuse the stdlib-only archive validator and validate its class protocol."""
    from train_segment_timematch import validate_source
    leaf = Path(leaf)
    config = SimpleNamespace(weights=str(leaf), source=source)
    validate_source(config)
    archived = json.loads((leaf / 'train_config.json').read_text(encoding='utf-8'))
    protocol = json.loads((leaf / 'closed_set_protocol.json').read_text(encoding='utf-8'))
    expected = dict(source=source, target=source, closed_set=True,
                    combine_spring_and_winter=False, seed=1, val_ratio=.1, test_ratio=.2)
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f'Original Source protocol mismatch: {key}={protocol.get(key)!r}, expected {value!r}')
    classes = protocol.get('classes')
    if (not isinstance(classes, list) or not classes or
            any(not isinstance(c, str) for c in classes) or
            len(classes) != len(set(classes)) or 'unknown' in classes):
        raise ValueError('Original Source protocol must define unique retained known classes')
    if protocol.get('class_to_idx') != {name: i for i, name in enumerate(classes)}:
        raise ValueError('Original Source class_to_idx does not match classes')
    if 'classes' in archived and archived['classes'] != classes:
        raise ValueError('Original Source config/protocol classes disagree')
    if 'num_classes' in archived and archived['num_classes'] != len(classes):
        raise ValueError('Original Source num_classes does not match protocol')
    return archived, protocol, config.source_checkpoint_sha256


def original_source_train_ids(eligible_ids, seed=1, val_ratio=.1, test_ratio=.2):
    """Reproduce source==target's second split (the first dict entry is overwritten).

    Each loop iteration starts from a fresh copy of the metadata order, while
    Python's random generator continues from the preceding shuffle.
    """
    original = [int(i) for i in eligible_ids]
    if len(set(original)) != len(original):
        raise ValueError('Eligible parcel IDs must be unique')
    rng = random.Random(seed)
    for _ in range(2):
        indices = original.copy()
        rng.shuffle(indices)
    n_train = len(indices) - int(test_ratio * len(indices)) - int(val_ratio * len(indices))
    return indices[:n_train]


def balanced_source_train_sample(eligible_ids, labels, train_ids, count, seed=1):
    """Return metadata IDs in seeded, class-balanced round-robin order."""
    if count <= 0:
        raise ValueError('samples_per_task must be positive')
    if len(eligible_ids) != len(labels):
        raise ValueError('Labels and eligible parcel IDs must have the same length')
    train = set(map(int, train_ids))
    groups = {}
    for parcel_id, label in zip(eligible_ids, labels):
        if int(parcel_id) in train:
            groups.setdefault(int(label), []).append(int(parcel_id))
    rng = random.Random(seed)
    for label in sorted(groups):
        rng.shuffle(groups[label])
    selected = []
    offset = 0
    while len(selected) < count:
        added = False
        for label in sorted(groups):
            if offset < len(groups[label]):
                selected.append(groups[label][offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        offset += 1
    if not selected:
        raise ValueError('No eligible SourceTrain parcels')
    return selected


def _build_frozen_model(num_classes, input_dim=10):
    """Build exactly PseLTae's modules without importing competing classifiers."""
    import torch
    from models.pse import PixelSetEncoder
    from models.ltae import LTAE
    from models.decoder import get_decoder
    model = torch.nn.Module()
    model.spatial_encoder = PixelSetEncoder(input_dim, mlp1=[10, 32, 64],
        pooling='mean_std', mlp2=[128, 128], with_extra=False, extra_size=4)
    model.temporal_encoder = LTAE(in_channels=128, n_head=16, d_k=8,
        d_model=256, n_neurons=[256, 128], dropout=.2, T=1000,
        max_temporal_shift=100)
    model.decoder = get_decoder([128, 64, 32], num_classes)
    model.eval().requires_grad_(False)
    return model


def _load_dataset(data_root, source, classes):
    from dataset import PixelSetData
    return PixelSetData(str(data_root), source, classes, with_extra=False,
        closed_set=True, combine_spring_and_winter=False)


def _parcel_sample(dataset, index, parcel_id, seed):
    import numpy as np
    from transforms import RandomSamplePixels, Normalize
    # Pixel sampling uses Python random today; preserve both generators so future
    # transforms and dataset readers cannot perturb the caller's random state.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed + int(parcel_id))
        np.random.seed((seed + int(parcel_id)) % (2 ** 32))
        sample = dataset[index]
        return Normalize()(RandomSamplePixels(64)(sample))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _extract_task(dataset, model, selected, seed, device, batch_size, seq_length, repeats):
    import numpy as np
    import torch
    lookup = {int(parcel_id): i for i, parcel_id in enumerate(dataset.get_parcel_indices())}
    chunks, dates, labels, subsets = [], [], [], []
    buffers_before = {name: value.detach().clone() for name, value in model.named_buffers()}
    model.eval().requires_grad_(False)
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch_ids = selected[start:start + batch_size]
            samples = [_parcel_sample(dataset, lookup[p], p, seed) for p in batch_ids]
            pixels = torch.as_tensor(np.stack([s['pixels'] for s in samples]), device=device).contiguous()
            mask = torch.as_tensor(np.stack([s['valid_pixels'] for s in samples]), device=device).contiguous()
            features = model.spatial_encoder(pixels, mask, None)
            if features.ndim != 3 or features.shape[-1] != 128:
                raise ValueError('Expected PSE content features [N,T,128]')
            chunks.append(features.cpu().numpy().astype(np.float32))
            for parcel_id, sample in zip(batch_ids, samples):
                positions = np.asarray(sample['positions'], dtype=np.int64)
                if len(positions) != features.shape[1] or not len(positions) or (np.diff(positions) < 0).any():
                    raise ValueError('Dates must match the full ordered observation sequence')
                dates.append(positions)
                labels.append(int(sample['label']))
                rng = random.Random(seed + int(parcel_id))
                width = min(seq_length, len(positions))
                subsets.append([sorted(rng.sample(range(len(positions)), width)) for _ in range(repeats)])
    for name, value in model.named_buffers():
        if not torch.equal(value, buffers_before[name]):
            raise RuntimeError(f'Frozen model buffer changed during extraction: {name}')
    result = dict(features=np.concatenate(chunks), dates=np.stack(dates).astype(np.int64),
        labels=np.asarray(labels, dtype=np.int64), parcel_ids=np.asarray(selected, dtype=np.int64),
        subset_indices=np.asarray(subsets, dtype=np.int64))
    if not np.isfinite(result['features']).all():
        raise ValueError('PSE features contain nonfinite values')
    return result


def extract_features(data_root, source_root, output_dir, tasks=None, samples_per_task=64,
                     seq_length=30, repeats=2, seed=1, device='cuda', batch_size=16):
    """Create a new atomic cache leaf and return its JSON-compatible manifest.

    ``source_root`` is the historical ``.../seed1/source`` directory. All task
    filenames in the manifest are relative to ``output_dir``. Cache arrays have
    no object dtype and can be opened using ``np.load(..., allow_pickle=False)``.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite feature cache: {output}')
    task_names = list(TASKS if tasks is None else tasks)
    if not task_names or len(set(task_names)) != len(task_names) or any(t not in TASKS for t in task_names):
        raise ValueError('tasks must be a nonempty unique subset of the four historical tasks')
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
           for x in (samples_per_task, seq_length, repeats, batch_size)):
        raise ValueError('Sample count, sequence length, repeats and batch size must be positive integers')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    archives = {}
    for task in task_names:
        source, _ = TASKS[task]
        leaf = Path(source_root) / f'closedset_source_{source.split("/")[1]}_seed1'
        archives[task] = (leaf, *validate_original_source(leaf, source))
    import numpy as np
    import torch
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(version=1, seed=seed, source_only=True, source_train_only=True,
        frozen_model=True, no_training=True, numpy_version=np.__version__,
        torch_version=str(torch.__version__), tasks={})
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as staging_name:
        staging = Path(staging_name) / 'cache'
        staging.mkdir()
        for task in task_names:
            source, target = TASKS[task]
            leaf, config, protocol, before_hash = archives[task]
            checkpoint = leaf / 'fold_0/model.pt'
            dataset = _load_dataset(data_root, source, protocol['classes'])
            eligible = dataset.get_parcel_indices().tolist()
            if ('eligible_source_samples' in protocol and
                    protocol['eligible_source_samples'] != len(eligible)):
                raise ValueError('Source metadata eligible count differs from Original Source protocol')
            train_ids = original_source_train_ids(eligible, seed=config['seed'],
                val_ratio=config['val_ratio'], test_ratio=config['test_ratio'])
            selected = balanced_source_train_sample(eligible, dataset.get_labels(), train_ids,
                samples_per_task, seed)
            # Module initialization consumes torch RNG, so isolate it as well.
            with torch.random.fork_rng(devices=[]):
                model = _build_frozen_model(len(protocol['classes']), config['input_dim'])
            packet = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if not isinstance(packet, dict) or not isinstance(packet.get('state_dict'), dict):
                raise ValueError('Original Source checkpoint must contain a state_dict dictionary')
            model.load_state_dict(packet['state_dict'], strict=True)
            model.to(device).eval().requires_grad_(False)
            arrays = _extract_task(dataset, model, selected, seed, device, batch_size, seq_length, repeats)
            after_hash = _sha256(checkpoint)
            if before_hash != after_hash:
                raise RuntimeError('Original Source checkpoint changed during read-only extraction')
            np.savez_compressed(staging / f'{task}.npz', **arrays)
            metadata = dict(task=task, source=source, target=target, classes=protocol['classes'],
                source_config=config, closed_set_protocol=protocol, seed=seed,
                source_checkpoint=str(checkpoint.resolve()), source_checkpoint_sha256_before=before_hash,
                source_checkpoint_sha256_after=after_hash, source_train_ids=train_ids,
                source_train_ids_sha256=hashlib.sha256(json.dumps(train_ids).encode()).hexdigest(),
                sample_ids=selected, selection='class-balanced round robin from Original SourceTrain only',
                split='second shuffle of original eligible metadata order; historical source==target overwrite',
                samples=len(selected), full_T=int(arrays['features'].shape[1]), feature_dim=128,
                seq_length=seq_length, repeats=repeats, pixel_seed='seed + parcel_id',
                transforms=['RandomSamplePixels(64)', 'Normalize()'], temporal_shift=False,
                frozen=True, frozen_model=True, source_train_only=True, no_training=True,
                model_buffers_unchanged=True, target_dataset_loaded=False)
            (staging / f'{task}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            manifest['tasks'][task] = dict(features_file=f'{task}.npz', metadata_file=f'{task}.json',
                source=source, target=target, samples=len(selected))
            del model, dataset, packet
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite feature cache: {output}')
        os.rename(staging, output)
    return manifest
