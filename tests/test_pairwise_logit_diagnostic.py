import csv
import gzip
import importlib.util

import numpy as np
import pytest
import torch


def test_pairwise_module_exists():
    assert importlib.util.find_spec('analysis.timematch_pairwise_logits') is not None


def test_pairwise_and_best_other_contributions_are_not_interchangeable():
    from analysis.timematch_pairwise_logits import derive_contributions
    original = np.array([[3., 2., 1.]])
    masked = np.array([[[2., 2.5, 1.], [3., 2., 2.]]])
    results = derive_contributions(original, masked, np.array([0]))
    assert np.allclose(results['relative_margin_contributions'], [[1.5, 0.]])
    assert np.allclose(results['pairwise_contributions'][0, :, 1:], [[1.5, 1.], [0., 1.]])
    assert np.allclose(original[:, None]-results['delta_logits'], masked)


def test_student_shift_uses_first_epoch_not_last_teacher_shift():
    from analysis.timematch_pairwise_logits import resolve_student_shift
    config = dict(shift_source=True, estimate_shift=True)
    shift, provenance = resolve_student_shift({}, config, events=[(0, 8), (19, -3)])
    assert shift == -8 and 'epoch0' in provenance
    assert resolve_student_shift({'student_shifts': {'source': -7, 'target': 0}}, config)[0] == -7
    assert resolve_student_shift({}, config, override=0)[0] == 0
    with pytest.raises(ValueError):
        resolve_student_shift({}, config, events=[(19, -3)])


def test_summary_marks_absent_classes_and_requires_both_directions_for_mutual_heads():
    from analysis.timematch_pairwise_logits import summarize_contributions
    # One direction is positive; the reverse is negative. This is not a mutual head.
    arrays = dict(labels=np.array([0]*20+[1]*20), original_logits=np.tile([2., 1., 0.], (40, 1)),
                  relative_margin_contributions=np.ones((40, 16)),
                  pairwise_contributions=np.concatenate([np.ones((20, 16, 3)), -np.ones((20, 16, 3))]),
                  original_pair_margins=np.ones((40, 3)), masked_pair_margins=np.ones((40, 16, 3)))
    summary = summarize_contributions(arrays, 3)
    assert summary['directed_head_counts'][0, 1] == 16
    assert summary['directed_head_counts'][1, 0] == 0
    assert summary['mutual_head_counts'][0, 1] == 0
    assert np.isnan(summary['class_mean'][2]).all()
    assert np.isnan(summary['directed_head_counts'][0, 2])


def test_complete_logits_export_includes_every_class_and_each_intervention(tmp_path):
    from analysis.timematch_pairwise_logits import derive_contributions, export_logits
    original = np.arange(6, dtype=float).reshape(2, 3)
    masked = np.repeat(original[:, None], 16, axis=1)-.5
    arrays = dict(original_logits=original, masked_logits=masked, labels=np.array([0, 2]),
                  parcel_ids=np.array([10, 20]), positions=np.array([1, 20]), applied_shift=np.array(-8))
    arrays.update(derive_contributions(original, masked, arrays['labels']))
    export_logits(tmp_path, arrays, ['a', 'b', 'c'], dict(task='AT1_DK1', epoch=20, domain='target'))
    packet = np.load(tmp_path / 'sample_logits.npz')
    assert packet['masked_logits'].shape == (2, 16, 3)
    with gzip.open(tmp_path / 'sample_logits.csv.gz', 'rt', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 34
    assert all(all(f'logit::{name}' in row for name in ('a', 'b', 'c')) for row in rows)
    assert rows[0]['case'] == 'original' and rows[16]['case'] == 'head_16'
    assert float(rows[0]['logit::c']) == original[0, 2]
    assert float(rows[16]['logit::b']) == masked[0, 15, 1]
    assert rows[17]['parcel_id'] == '20' and rows[17]['case'] == 'original'


def test_eval_collection_preserves_model_and_matches_direct_logits():
    from analysis.timematch_pairwise_logits import collect_logits
    from models.stclassifier import PseLTae
    model = PseLTae(num_classes=3, with_extra=False).eval()

    class Dataset:
        samples = [(None, i, i, None) for i in range(3)]

        def __getitem__(self, index):
            return dict(pixels=np.full((20, 10, 4), .1*(index+1), dtype=np.float32),
                        valid_pixels=np.ones((20, 4), dtype=np.float32), positions=np.arange(20)*15,
                        extra=np.zeros(4, dtype=np.float32), label=index, parcel_index=index)

    before = {key: value.clone() for key, value in model.state_dict().items()}
    arrays = collect_logits(model, Dataset(), [0, 1, 2], 1, -8, torch.device('cpu'))
    assert arrays['original_logits'].shape == (3, 3)
    assert arrays['masked_logits'].shape == (3, 16, 3)
    assert arrays['structure_mask'].sum(-1).tolist() == [[6]*16]*3
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert model.training is False


def test_split_reconstruction_matches_original_uda_function():
    import ast
    import random
    from pathlib import Path
    from analysis.timematch_pairwise_logits import reconstruct_uda_splits
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'train.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'create_train_val_test_folds')
    scope = dict(random=random, np=np)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<original-splits>', 'exec'), scope)
    indices = dict(src=np.arange(100)*2, dst=np.arange(120)*3)
    config = dict(seed=1, source='src', target='dst', val_ratio=.1, test_ratio=.2)
    class Dataset:
        def __init__(self, values): self.values = values
        def get_parcel_indices(self): return self.values
    state = random.getstate()
    try:
        random.seed(1)
        expected = scope['create_train_val_test_folds'](['src', 'dst'], 1, indices, .1, .2)[0]
    finally:
        random.setstate(state)
    actual = reconstruct_uda_splits(config, {key: Dataset(value) for key, value in indices.items()})
    assert actual == expected and random.getstate() == state


def test_read_only_endpoint_pipeline_exports_four_complete_logit_cases(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import dataset
    import analysis.timematch_pairwise_logits as audit
    from models.stclassifier import PseLTae
    source_name, target_name = audit.TASKS['AT1_DK1']
    source_root, uda_root = tmp_path / 'source', tmp_path / 'uda'
    source_dir = source_root / 'closedset_source_33UVP_seed1'
    uda_dir = uda_root / 'closedset_timematch_33UVP_to_32VNH_seed1'
    config = dict(source=source_name, target=target_name, seed=1, model='pseltae', closed_set=True,
                  combine_spring_and_winter=False, num_folds=1, input_dim=10, with_extra=False,
                  num_pixels=4, val_ratio=.1, test_ratio=.5, epochs=20, method='timematch',
                  output_student=True, evidence_mode='off', weights=str(source_dir),
                  tensorboard_log_dir=str(tmp_path / 'tensorboard'), shift_source=True, estimate_shift=True)
    classes = ['a', 'b', 'c']
    model = PseLTae(num_classes=3, with_extra=False)
    for root, cfg in ((source_dir, dict(config, target=source_name, epochs=100, method=None)), (uda_dir, config)):
        (root / 'fold_0').mkdir(parents=True)
        (root / 'train_config.json').write_text(json.dumps(cfg))
        (root / 'closed_set_protocol.json').write_text(json.dumps(dict(classes=classes)))
    initial = source_dir / 'fold_0/model.pt'
    final = uda_dir / 'fold_0/model.pt'
    torch.save(dict(state_dict=model.state_dict()), initial)
    torch.save(dict(state_dict=model.state_dict(), role='student', completed_epochs=20,
                    student_shifts=dict(source=-8, target=0)), final)
    hashes = (audit.sha256(initial), audit.sha256(final))

    class Dataset:
        def __init__(self, data_root, domain, class_names, transform=None, indices=None, **kwargs):
            self.samples = [(None, i, i % 3, None) for i in range(36) if indices is None or i in indices]
            self.transform = transform
        def __len__(self): return len(self.samples)
        def get_parcel_indices(self): return np.array([sample[1] for sample in self.samples])
        def get_labels(self): return np.array([sample[2] for sample in self.samples])
        def __getitem__(self, index):
            _, parcel, label, _ = self.samples[index]
            sample = dict(pixels=np.full((12, 10, 4), 1000.+parcel, dtype=np.float32),
                          valid_pixels=np.ones((12, 4), dtype=np.float32), positions=np.arange(12)*25,
                          extra=np.zeros(4, dtype=np.float32), label=label, parcel_index=parcel)
            return self.transform(sample) if self.transform else sample

    monkeypatch.setattr(dataset, 'PixelSetData', Dataset)
    output = tmp_path / 'audit'
    audit.run_task(SimpleNamespace(source_root=str(source_root), uda_root=str(uda_root),
                                  task='AT1_DK1', data_root='synthetic', output=str(output),
                                  student_source_shift=None, device='cpu', min_mean=.05, min_rate=.65, min_count=16))
    assert hashes == (audit.sha256(initial), audit.sha256(final))
    manifest = json.loads((output / 'manifest.json').read_text())
    assert len(manifest['cases']) == 4
    for domain in ('source', 'target'):
        zero = np.load(output / f'epoch_00/{domain}/sample_logits.npz')
        twenty = np.load(output / f'epoch_20/{domain}/sample_logits.npz')
        assert np.array_equal(zero['parcel_ids'], twenty['parcel_ids'])
        assert twenty['masked_logits'].shape == (18, 16, 3)
        assert zero['applied_shift'] == 0
        assert twenty['applied_shift'] == (-8 if domain == 'source' else 0)
    assert len(list(output.glob('epoch_*/*/*.png'))) == 12
    config['evidence_mode'] = 'positive'
    (uda_dir / 'train_config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='original TimeMatch'):
        audit.load_task(source_root, uda_root, 'AT1_DK1')
    config['evidence_mode'] = 'off'
    (uda_dir / 'train_config.json').write_text(json.dumps(config))
    torch.save(dict(state_dict=model.state_dict(), epoch=9, best_f1=1.), final)
    with pytest.raises(ValueError, match='20-epoch'):
        audit.load_task(source_root, uda_root, 'AT1_DK1')


def test_empty_samples_stay_na_and_shift_conflicts_fail():
    from analysis.timematch_pairwise_logits import derive_contributions, summarize_contributions, resolve_student_shift
    original, masked, labels = np.empty((0, 3)), np.empty((0, 16, 3)), np.empty(0, dtype=int)
    arrays = dict(original_logits=original, masked_logits=masked, labels=labels)
    arrays.update(derive_contributions(original, masked, labels))
    summary = summarize_contributions(arrays, 3)
    assert np.isnan(summary['class_mean']).all() and np.isnan(summary['directed_head_counts']).all()
    with pytest.raises(ValueError):
        resolve_student_shift({}, dict(shift_source=True, estimate_shift=True), events=[(0, 8), (0, 7)])
