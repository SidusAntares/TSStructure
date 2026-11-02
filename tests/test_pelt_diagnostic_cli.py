import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from analysis import segment_pelt_diagnostic as diagnostic


def cache(root):
    root.mkdir()
    rng = np.random.default_rng(4)
    features = rng.normal(0, .03, (2, 12, 4))
    features[:, 6:] += 3
    path = root/'AT1_DK1.npz'
    np.savez_compressed(path, features=features, dates=np.arange(12)[None].repeat(2, 0)*20,
                       labels=[0, 1], parcel_ids=[10, 11],
                       subset_indices=np.array([[[0, 2, 4, 6, 8, 10]], [[1, 3, 5, 7, 9, 11]]]))
    (root/'AT1_DK1.json').write_text(json.dumps(dict(source='source', classes=['a', 'b'], source_train_only=True)))
    (root/'manifest.json').write_text(json.dumps(dict(source_train_only=True, frozen_model=True,
        no_training=True, tasks={'AT1_DK1': dict(features_file=path.name, metadata_file='AT1_DK1.json')})))
    return path


def test_scan_only_needs_cached_features_and_never_changes_cache(tmp_path):
    source = tmp_path/'features'
    path = cache(source)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path/'scan'
    args = ['--stage', 'scan', '--feature_root', str(source), '--output_dir', str(output),
            '--penalties', '.01', '.1', '--normalizations', 'feature',
            '--gamma_multipliers', '1', '--no_plots']
    diagnostic.main(args)
    assert (output/'scan/parameter_summary.csv').is_file()
    assert (output/'scan/per_sequence.csv').is_file()
    manifest = json.loads((output/'scan/manifest.json').read_text())
    assert manifest['no_training'] and manifest['source_train_only']
    assert manifest['sequence_count'] == 4
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with pytest.raises(FileExistsError):
        diagnostic.main(args)


def test_diagnostic_plot_draws_candidate_boundaries(tmp_path):
    root = tmp_path/'features'
    cache(root)
    diagnostic.main(['--stage', 'scan', '--feature_root', str(root), '--output_dir', str(tmp_path/'plot'),
        '--penalties', '.01', '.03', '.1', '.3', '--normalizations', 'feature', '--gamma_multipliers', '1'])
    assert (tmp_path/'plot/scan/example_boundaries_AT1_DK1.png').is_file()
    assert (tmp_path/'plot/scan/single_segment_rate_full.png').is_file()


def test_scan_rejects_unverified_source_cache(tmp_path):
    root = tmp_path/'features'
    cache(root)
    metadata = json.loads((root/'manifest.json').read_text())
    metadata['source_train_only'] = False
    (root/'manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='SourceTrain'):
        diagnostic.main(['--stage', 'scan', '--feature_root', str(root), '--output_dir', str(tmp_path/'bad')])
