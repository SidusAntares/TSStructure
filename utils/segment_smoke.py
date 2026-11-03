"""Bounded SourceTrain-only structure finetuning health checks; never runs UDA."""
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from utils.focal_loss import FocalLoss
from utils.segment_structure import structure_forward
from utils.segment_training import sample_to_device, write_history
from utils.soft_top50 import keep_head_embeddings
from utils.segment_ranking import batch_ranking_diagnostic, ranking_summary


@contextmanager
def preserve_rng_and_modes(model, selector, seed, device):
    modes = [(module, module.training) for root in (model, selector) for module in root.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    location = torch.device(device)
    devices = [(location.index if location.index is not None else torch.cuda.current_device())] if location.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            model.eval()
            selector.eval()
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modes:
            module.training = training


def statistics(info):
    count = info['candidate_count'].detach().float()
    valid = info['valid_segments'][:, None].expand_as(info['probabilities'])
    probability = info['probabilities'].detach()
    gate = info['gate'].detach()
    labels = info['segments']
    lengths = gate.new_zeros(labels.size(0), gate.size(-1))
    lengths.scatter_add_(1, labels.clamp_min(0), (labels >= 0).to(lengths.dtype))
    actual_lengths = lengths[info['valid_segments']]
    mean = probability.sum(-1, keepdim=True)/count[:, None, None]
    spread = (((probability-mean)**2)*valid).sum(-1)/count[:, None]
    return dict(candidate_count=float(count.mean()), candidate_count_min=float(count.min()),
        candidate_count_max=float(count.max()), single_segment_rate=float((count == 1).float().mean()),
        selected_count=float(info['selected_count'].float().mean()),
        coverage=float(info['coverage'].float().mean()), fallback_rate=float(info['fallback_rate']),
        all_selected_rate=float((gate.sum(-1) == count[:, None]).float().mean()),
        mean_gate_probability=float(probability[valid].mean()),
        raw_coverage=float(info.get('raw_coverage', info['coverage'].mean())),
        sparse_coefficient=float(info.get('sparse_coefficient', probability.new_tensor(1.))),
        mean_probability_std=float(spread.sqrt().mean()),
        expected_empty_rate=float(torch.where(valid, 1-probability, torch.ones_like(probability)).prod(-1).mean()),
        sparse=float(info['sparse_loss'].detach()),
        median_segment_observations=float(actual_lengths.median()))


def _forward(model, selector, sample, config, deterministic=False):
    pixels, mask, dates, extra = sample_to_device(sample, config.device)
    return structure_forward(model, selector, pixels, mask, dates, extra,
        variant=config.variant, deterministic=deterministic, full_loss=False, unified=True,
        penalty=config.pelt_penalty, min_size=config.pelt_min_size,
        penalty_mode=config.pelt_penalty_mode)


def matched_random_point_mask(info):
    """Random segments with exactly the learned count per sample/head (not duration)."""
    gate = info['gate'].detach()
    rank = torch.rand_like(gate).masked_fill(~info['valid_segments'][:, None], float('inf')).argsort(-1)
    chosen = torch.arange(gate.size(-1), device=gate.device)[None, None] < gate.sum(-1, keepdim=True)
    random_gate = torch.zeros_like(gate).scatter(-1, rank, chosen.to(gate.dtype))
    labels = info['segments']
    points = random_gate.gather(-1, labels.clamp_min(0)[:, None].expand(-1, gate.size(1), -1))
    return points*(labels >= 0)[:, None], random_gate


@torch.no_grad()
def probe_statistics(model, selector, loader, config, ranking=False, checkpoint_step=0):
    sums, samples, minimum, maximum = {}, 0, float('inf'), 0
    ranking_rows, deletion_rows = [], []
    criterion = FocalLoss(gamma=config.focal_loss_gamma)
    with preserve_rng_and_modes(model, selector, config.seed+12000, config.device):
        for sample in loader:
            _, keep, info = _forward(model, selector, sample, config, deterministic=True)
            full = model(*sample_to_device(sample, config.device))
            random_mask, _ = matched_random_point_mask(info)
            pooled, _, _ = keep_head_embeddings(info['attention'], info['values'], random_mask)
            random_logits = model.decoder(model.temporal_encoder.dropout(model.temporal_encoder.mlp(pooled)))
            n = len(full)
            labels = sample['label'].to(config.device)
            if ranking:
                ranked, deleted = batch_ranking_diagnostic(model, info, labels, config.focal_loss_gamma)
                classes = labels.tolist()
                parcels = sample['parcel_index'].tolist() if 'parcel_index' in sample else None
                for record in ranked+deleted:
                    local_index = record['probe_index']
                    record['probe_index'] += samples
                    record['class_id'] = classes[local_index]
                    if parcels is not None:
                        record['parcel_index'] = parcels[local_index]
                ranking_rows.extend(ranked)
                deletion_rows.extend(deleted)
            row = dict(statistics(info), full_loss=float(criterion(full, labels)),
                keep_loss=float(criterion(keep, labels)),
                full_accuracy=float((full.argmax(-1) == labels).float().mean()),
                keep_accuracy=float((keep.argmax(-1) == labels).float().mean()),
                random_keep_loss=float(criterion(random_logits, labels)),
                random_keep_accuracy=float((random_logits.argmax(-1) == labels).float().mean()))
            minimum = min(minimum, row['candidate_count_min'])
            maximum = max(maximum, row['candidate_count_max'])
            for key, value in row.items():
                sums[key] = sums.get(key, 0.)+value*n
            samples += n
    if not samples:
        raise ValueError('Empty fixed SourceTrain probe cohort')
    result = {key: value/samples for key, value in sums.items()}
    result.update(candidate_count_min=minimum, candidate_count_max=maximum)
    if ranking:
        result.update(ranking_summary(ranking_rows))
        result['ranking_unidentifiable_samples'] = samples-result['ranking_identifiable_samples']
        directory = Path(config.fold_dir)/'ranking'
        directory.mkdir(exist_ok=True)
        for name, rows in [('budgets', ranking_rows), ('deletions', deletion_rows)]:
            if rows:
                write_history(directory/f'step_{checkpoint_step:06d}_{name}.csv', rows)
    return result


def inspect_health(rows, config):
    if len(rows) < config.warmup_steps+config.health_window:
        return None
    window = rows[-config.health_window:]
    if sum(row['single_segment_rate'] for row in window)/len(window) >= .95:
        return 'failed_candidate_collapse'
    if sum(row['fallback_rate'] for row in window)/len(window) >= .95:
        return 'failed_gate_collapse'
    return None


def _norm(parameters):
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.
    if not all(torch.isfinite(gradient).all() for gradient in grads):
        raise FloatingPointError('Nonfinite smoke gradients')
    return float(torch.stack([gradient.norm() for gradient in grads]).norm())


def keep_gradient_check(loss, model, selector):
    parameters = [selector.scorer[0].weight, model.temporal_encoder.attention_heads.query,
                  model.temporal_encoder.attention_heads.key.weight,
                  model.spatial_encoder.mlp1[0].linear.weight]
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    result = {}
    for name, gradient in zip(('keep_selector_grad_norm', 'keep_query_grad_norm',
                               'keep_key_grad_norm', 'keep_pse_grad_norm'), gradients):
        if gradient is not None and not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite isolated Keep gradient')
        result[name] = float(gradient.norm()) if gradient is not None else 0.
    return result


def selector_loss_gradients(keep_loss, sparse_loss, selector, keep_weight, sparse_weight):
    """Compare both losses on all trainable selector parameters without .grad writes."""
    parameters = [p for p in selector.parameters() if p.requires_grad]
    vectors = []
    for loss in (keep_loss, sparse_loss):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vector = torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                            for p, g in zip(parameters, gradients)])
        if not torch.isfinite(vector).all():
            raise FloatingPointError('Nonfinite loss-specific selector gradient')
        vectors.append(vector)
    keep, sparse = vectors
    keep_norm, sparse_norm = float(keep.norm()), float(sparse.norm())
    weighted_keep, weighted_sparse = keep*keep_weight, sparse*sparse_weight
    wk, ws = float(weighted_keep.norm()), float(weighted_sparse.norm())
    cosine = float(torch.dot(keep, sparse)/keep.norm()/sparse.norm()) if keep_norm and sparse_norm else None
    return dict(keep_all_selector_grad_norm=keep_norm, sparse_selector_grad_norm=sparse_norm,
        weighted_keep_selector_grad_norm=wk, weighted_sparse_selector_grad_norm=ws,
        combined_selector_loss_grad_norm=float((weighted_keep+weighted_sparse).norm()),
        selector_gradient_cosine=cosine,
        sparse_to_keep_gradient_ratio=ws/wk if wk else None)


def train_smoke_steps(model, selector, loader, probe_loader, config):
    if not len(loader):
        raise ValueError('No complete source training batches')
    horizon = config.scheduler_epochs*len(loader)
    smoke_epochs = getattr(config, 'smoke_epochs', None)
    if smoke_epochs not in (None, 5):
        raise ValueError('Epoch smoke budget must be exactly 5')
    total_steps = smoke_epochs*len(loader) if smoke_epochs else config.steps
    probe_interval = len(loader) if smoke_epochs else config.probe_every
    if not isinstance(total_steps, int) or total_steps < 1 or total_steps >= horizon:
        raise ValueError('Smoke must stop before the full 20-epoch schedule; reduce --steps')
    optimizer = torch.optim.Adam([{'params': model.parameters()},
        {'params': selector.parameters(), 'lr': config.selector_lr}],
        lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, horizon)
    criterion = FocalLoss(gamma=config.focal_loss_gamma)
    rows, probes, gradient_rows = [], [], []
    status = 'completed'
    initial = dict(step=0, source_epoch=0.,
        **probe_statistics(model, selector, probe_loader, config, ranking=True, checkpoint_step=0))
    probes.append(initial)
    print('SMOKE_PROBE|' + json.dumps(initial), flush=True)
    iterator = iter(loader)
    for step in range(1, total_steps+1):
        try:
            sample = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            sample = next(iterator)
        model.train()
        selector.train()
        _, keep, info = _forward(model, selector, sample, config)
        labels = sample['label'].to(config.device)
        keep_loss = criterion(keep, labels)
        loss = keep_loss+config.sparse_weight*info['sparse_loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite source smoke loss')
        checks = {}
        if step == 1 or step % config.gradient_every == 0 or step == total_steps:
            checks = keep_gradient_check(keep_loss, model, selector)
            checks.update(selector_loss_gradients(keep_loss, info['sparse_loss'], selector,
                                                  1., config.sparse_weight))
            sparse_pse = torch.autograd.grad(info['sparse_loss'],
                model.spatial_encoder.mlp1[0].linear.weight, retain_graph=True, allow_unused=True)[0]
            if sparse_pse is not None:
                raise RuntimeError('Sparse loss must not have a PSE gradient path')
            checks['sparse_pse_grad_norm'] = 0.
            gradient_rows.append(dict(step=step, **checks))
            print('SMOKE_GRADIENT|' + json.dumps(dict(step=step, sparse_weight=config.sparse_weight,
                                                      **checks)), flush=True)
        optimizer.zero_grad()
        loss.backward()
        model_grad, selector_grad = _norm(model.parameters()), _norm(selector.parameters())
        optimizer.step()
        scheduler.step()
        row = dict(step=step, source_epoch=step/len(loader), variant=config.variant, sparse_weight=config.sparse_weight,
            actual_pelt_penalty=config.pelt_penalty,
            total_loss=float(loss.detach()), keep_loss=float(keep_loss.detach()),
            learning_rate=optimizer.param_groups[0]['lr'], model_grad_norm=model_grad,
            selector_grad_norm=selector_grad, **statistics(info), **checks)
        rows.append(row)
        write_history(Path(config.fold_dir)/'steps.csv', rows)
        if step == 1 or step % 10 == 0 or step == total_steps:
            print('SMOKE_STEP|' + json.dumps(row), flush=True)
        failed = inspect_health(rows, config)
        if step % probe_interval == 0 or step == total_steps or failed:
            record = dict(step=step, source_epoch=step/len(loader),
                **probe_statistics(model, selector, probe_loader, config, ranking=True, checkpoint_step=step))
            probes.append(record)
            print('SMOKE_PROBE|' + json.dumps(record), flush=True)
        if failed:
            status = failed
            break
    write_history(Path(config.fold_dir)/'probes.csv', probes)
    write_history(Path(config.fold_dir)/'keep_gradients.csv', gradient_rows)
    last = rows[-config.health_window:]
    tail = {key: sum(row[key] for row in last)/len(last)
            for key in ('candidate_count', 'single_segment_rate', 'fallback_rate', 'coverage', 'mean_gate_probability')}
    warnings = []
    if tail['fallback_rate'] > .5:
        warnings.append('More than half of head/sample gates require empty-selection repair')
    if tail['single_segment_rate'] > .25:
        warnings.append('More than one quarter of samples still have only one candidate segment')
    if not any(row['keep_selector_grad_norm'] > 0 for row in gradient_rows):
        warnings.append('No nonzero isolated Keep gradient observed on selector scorer')
    if probes[-1]['all_selected_rate'] >= .95:
        warnings.append('At least 95% of deterministic probe head/sample gates retain all segments')
    result = dict(status=status, completed_steps=len(rows), requested_steps=total_steps,
        requested_smoke_epochs=smoke_epochs, completed_source_epochs=len(rows)/len(loader),
        steps_per_source_epoch=len(loader),
        scheduler_total_steps=horizon, last_learning_rate=optimizer.param_groups[0]['lr'],
        variant=config.variant, pelt_penalty_mode=config.pelt_penalty_mode,
        actual_pelt_penalty=config.pelt_penalty, tail_training_statistics=tail,
        sparse_weight=config.sparse_weight, classification_path='keep_only_unified',
        selector_protocol=getattr(config, 'selector_protocol', 'legacy'),
        initial_probe=probes[0], final_probe=probes[-1], warnings=warnings,
        probe_split='fixed SourceTrain diagnostic cohort, not validation or test',
        no_uda=True, no_target_labels=True, full_training_not_started=True,
        original_source_checkpoint_sha256=config.original_source_checkpoint_sha256)
    with (Path(config.fold_dir)/'health_summary.json').open('w') as stream:
        json.dump(result, stream, indent=2)
    torch.save(dict(state_dict=model.state_dict(), selector_state_dict=selector.state_dict(),
        role='source_structure_smoke', completed_steps=len(rows), requested_steps=total_steps,
        completed_smoke_epochs=len(rows)/len(loader),
        status=status, variant=config.variant, sparse_weight=config.sparse_weight,
        classification_path='keep_only_unified',
        selector_protocol=getattr(config, 'selector_protocol', 'legacy'),
        original_source_checkpoint_sha256=config.original_source_checkpoint_sha256),
        Path(config.fold_dir)/'smoke_checkpoint.pt')
    print('SMOKE_COMPLETE|' + json.dumps(result), flush=True)
    return result


class FixedProbeTransform:
    def __init__(self, transform, seed):
        self.transform, self.seed = transform, seed

    def __call__(self, sample):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            seed = self.seed+int(sample['parcel_index'])
            random.seed(seed)
            np.random.seed(seed % (2**32))
            return self.transform(sample)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def run_smoke(config):
    from dataset import PixelSetData, create_train_loader
    from torchvision.transforms import Compose
    from transforms import RandomSamplePixels, RandomSampleTimeSteps, Normalize, ToTensor
    from utils.segment_training import prepare, new_models, write_resolved
    from utils.pelt_feature_cache import balanced_source_train_sample
    config, splits = prepare(config, source_only=True)
    write_resolved(config)
    model, selector = new_models(config)
    model.load_state_dict(torch.load(Path(config.weights)/'fold_0/model.pt',
                                    map_location=config.device, weights_only=False)['state_dict'])
    transform = Compose([RandomSamplePixels(config.num_pixels), RandomSampleTimeSteps(config.seq_length),
                         Normalize(), ToTensor()])
    kwargs = dict(indices=splits[config.source]['train'], closed_set=True, combine_spring_and_winter=False)
    dataset = PixelSetData(config.data_root, config.source, config.classes, transform, **kwargs)
    loader = create_train_loader(dataset, config.batch_size, config.num_workers)
    parcel_ids = dataset.get_parcel_indices().tolist()
    selected = balanced_source_train_sample(parcel_ids, dataset.get_labels(), parcel_ids,
                                            config.probe_samples, config.seed+12000)
    probe_dataset = PixelSetData(config.data_root, config.source, config.classes,
        FixedProbeTransform(transform, config.seed+12000), indices=set(selected),
        closed_set=True, combine_spring_and_winter=False)
    probe_loader = torch.utils.data.DataLoader(probe_dataset, batch_size=min(32, config.batch_size),
                                               shuffle=False, num_workers=0)
    with (Path(config.output_dir)/'probe_parcel_ids.json').open('w') as stream:
        json.dump(dict(split='SourceTrain only', parcel_ids=selected, seed=config.seed+12000), stream, indent=2)
    result = train_smoke_steps(model, selector, loader, probe_loader, config)
    if result['status'] != 'completed':
        raise RuntimeError('Source structure smoke stopped early: '+result['status'])
    return result
