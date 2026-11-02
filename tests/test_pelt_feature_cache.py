"""Source-only cache tests use synthetic parcels and a frozen dummy checkpoint."""
import json
import random
import subprocess
import sys

import numpy as np
import pytest

from utils import pelt_feature_cache as cache


def source_archive(tmp_path):
    source = cache.TASKS['AT1_DK1'][0]
    leaf = tmp_path / 'source' / 'closedset_source_33UVP_seed1'
    (leaf / 'fold_0').mkdir(parents=True)
    config = dict(source=source, target=source, seed=1, num_workers=8,
        epochs=100, model='pseltae', closed_set=True, num_folds=1, num_blocks=100,
        val_ratio=.1, test_ratio=.2, sample_pixels_val=True, batch_size=128,
        seq_length=30, num_pixels=64, input_dim=10, combine_spring_and_winter=False,
        with_extra=False, with_shift_aug=False, lr=.001, weight_decay=.0001,
        focal_loss_gamma=1., shift_aug_p=1., max_shift_aug=60, mode=None,
        m1=False, method=None, classes=['a', 'b'])
    protocol = dict(source=source, target=source, seed=1, val_ratio=.1, test_ratio=.2,
        closed_set=True, combine_spring_and_winter=False,
        classes=['a', 'b'], class_to_idx={'a': 0, 'b': 1})
    (leaf / 'train_config.json').write_text(json.dumps(config), encoding='utf-8')
    (leaf / 'closed_set_protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    (leaf / 'fold_0/model.pt').write_bytes(b'dummy source checkpoint')
    return leaf, config, protocol


def test_import_is_lazy_and_does_not_load_training_stack():
    script = ('import sys; import utils.pelt_feature_cache; '
        'assert not any(x in sys.modules for x in '
        '["torch", "dataset", "train", "timematch", "models.competings", '
        '"utils.imputation_training", "torch.utils.tensorboard"])')
    subprocess.run([sys.executable, '-c', script], check=True)


def test_original_source_validation_and_protocol_rejection(tmp_path):
    leaf, config, protocol = source_archive(tmp_path)
    archived, checked, digest = cache.validate_original_source(leaf, config['source'])
    assert archived == config and checked == protocol
    assert digest == cache._sha256(leaf / 'fold_0/model.pt')
    config['epochs'] = 20
    (leaf / 'train_config.json').write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError, match='epochs'):
        cache.validate_original_source(leaf, config['source'])
    config['epochs'] = 100
    (leaf / 'train_config.json').write_text(json.dumps(config), encoding='utf-8')
    protocol['target'] = cache.TASKS['AT1_DK1'][1]
    (leaf / 'closed_set_protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    with pytest.raises(ValueError, match='target'):
        cache.validate_original_source(leaf, config['source'])


def test_second_shuffle_uses_fresh_metadata_order_and_continuing_rng():
    ids = [11, 20, 21, 29, 38, 44, 48, 67, 72, 99]
    rng = random.Random(1)
    first = ids.copy()
    rng.shuffle(first)
    second = ids.copy()
    rng.shuffle(second)
    actual = cache.original_source_train_ids(ids)
    assert actual == second[:7]
    assert actual != first[:7]
    assert ids == [11, 20, 21, 29, 38, 44, 48, 67, 72, 99]


def test_balanced_sampling_excludes_holdouts_and_caps_available_samples():
    ids = list(range(100, 120))
    labels = [i % 2 for i in range(20)]
    train = cache.original_source_train_ids(ids)
    sampled = cache.balanced_source_train_sample(ids, labels, train, 8, seed=17)
    assert sampled == cache.balanced_source_train_sample(ids, labels, train, 8, seed=17)
    assert len(set(sampled)) == 8 and set(sampled) <= set(train)
    lookup = dict(zip(ids, labels))
    assert [lookup[p] for p in sampled] == [0, 1] * 4
    all_available = cache.balanced_source_train_sample(ids, labels, train, 64)
    assert set(all_available) == set(train)


@pytest.mark.parametrize('time_steps', [5, 35])
def test_extraction_preserves_checkpoint_bn_rng_and_full_dates(tmp_path, monkeypatch, time_steps):
    torch = pytest.importorskip('torch')
    leaf, config, _ = source_archive(tmp_path)
    model = cache._build_frozen_model(2)
    checkpoint = leaf / 'fold_0/model.pt'
    torch.save({'state_dict': model.state_dict()}, checkpoint)
    before = checkpoint.read_bytes()
    before_buffers = {k: v.clone() for k, v in model.named_buffers()}

    def forbidden_temporal_forward(*args, **kwargs):
        raise AssertionError('Only PSE content features may be evaluated')

    monkeypatch.setattr(model.temporal_encoder, 'forward', forbidden_temporal_forward)
    monkeypatch.setattr(model.decoder, 'forward', forbidden_temporal_forward)
    monkeypatch.setattr(cache, '_build_frozen_model', lambda *args: model)

    class SourceParcels:
        def get_parcel_indices(self):
            return np.arange(100, 120, dtype=np.int64)

        def get_labels(self):
            return np.arange(20) % 2

        def __getitem__(self, index):
            # Reads change global RNG states deliberately; the extractor must restore them.
            random.random()
            np.random.random()
            return dict(pixels=np.random.default_rng(index).integers(0, 65535,
                (time_steps, 10, 80), dtype=np.uint16), valid_pixels=np.ones((time_steps, 80)),
                positions=np.arange(time_steps) * 3 + np.arange(time_steps) // 2, label=index % 2,
                parcel_index=index + 100)

    calls = []

    def source_only(data_root, source, classes):
        calls.append(source)
        assert source == config['source']
        assert classes == ['a', 'b']
        return SourceParcels()

    monkeypatch.setattr(cache, '_load_dataset', source_only)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    args = dict(data_root=tmp_path / 'data', source_root=leaf.parent,
        tasks=['AT1_DK1'], samples_per_task=8, seq_length=30, repeats=2,
        seed=1, device='cpu', batch_size=3)
    first = cache.extract_features(output_dir=tmp_path / 'first', **args)
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0] and after_numpy[2:] == numpy_state[2:]
    np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    second = cache.extract_features(output_dir=tmp_path / 'second', **args)
    assert first == second and calls == [config['source'], config['source']]
    assert checkpoint.read_bytes() == before
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert all(p.grad is None for p in model.parameters())
    for name, value in model.named_buffers():
        assert torch.equal(value, before_buffers[name])
    for module in model.modules():
        assert not module.training
    entry = first['tasks']['AT1_DK1']
    with np.load(tmp_path / 'first' / entry['features_file'], allow_pickle=False) as a, \
            np.load(tmp_path / 'second' / entry['features_file'], allow_pickle=False) as b:
        assert set(a.files) == {'features', 'dates', 'labels', 'parcel_ids', 'subset_indices'}
        assert a['features'].shape == (8, time_steps, 128) and a['features'].dtype == np.float32
        assert a['subset_indices'].shape == (8, 2, min(30, time_steps))
        assert (np.diff(a['subset_indices'], axis=-1) > 0).all()
        assert a['subset_indices'].min() >= 0 and a['subset_indices'].max() < time_steps
        np.testing.assert_array_equal(a['dates'], np.tile(np.arange(time_steps) * 3 + np.arange(time_steps) // 2, (8, 1)))
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        metadata = json.loads((tmp_path / 'first' / entry['metadata_file']).read_text())
        assert set(a['parcel_ids']) <= set(metadata['source_train_ids'])
        assert metadata['source_checkpoint_sha256_before'] == metadata['source_checkpoint_sha256_after']
    with pytest.raises(FileExistsError):
        cache.extract_features(output_dir=tmp_path / 'first', **args)


def test_failed_validation_does_not_create_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        cache.extract_features(tmp_path, tmp_path / 'missing-source', tmp_path / 'cache', tasks=['AT1_DK1'])
    assert not (tmp_path / 'cache').exists()


def test_failed_checkpoint_load_leaves_no_partial_cache(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    leaf, _, _ = source_archive(tmp_path)
    torch.save({'state_dict': {}}, leaf / 'fold_0/model.pt')
    checkpoint_bytes = (leaf / 'fold_0/model.pt').read_bytes()

    class MetadataOnly:
        def get_parcel_indices(self):
            return np.arange(10)

        def get_labels(self):
            return np.arange(10) % 2

    monkeypatch.setattr(cache, '_load_dataset', lambda *args: MetadataOnly())
    with pytest.raises(RuntimeError, match='state_dict'):
        cache.extract_features(tmp_path, leaf.parent, tmp_path / 'cache',
            tasks=['AT1_DK1'], device='cpu')
    assert not (tmp_path / 'cache').exists()
    assert not list(tmp_path.glob('.cache-*'))
    assert (leaf / 'fold_0/model.pt').read_bytes() == checkpoint_bytes
