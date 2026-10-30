"""Isolated imputer training and optional hooks into the unchanged TimeMatch losses."""
import csv
import hashlib
import json
import random
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.timeperceiver_imputer import TimePerceiverImputer
from utils.focal_loss import FocalLoss
from utils.imputation import (probe_attention, reconstruction_loss,
                              select_gap_queries, select_recovery_mask)


@contextmanager
def evaluation_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def sample_to_device(sample, device):
    return tuple(sample[key].to(device) if key in sample else None
                 for key in ('pixels', 'valid_pixels', 'positions', 'extra'))


def classify_features(model, features, dates, valid=None):
    """Shared LTAE, masking padded tokens and interpolating its frozen date table."""
    encoder = model.temporal_encoder
    if valid is None:
        valid = torch.ones_like(dates, dtype=torch.bool)
    if bool(valid.all()) and bool((dates == dates.round()).all()):
        return model.decoder(encoder(features, dates.long()))
    if encoder.inconv is not None:
        if bool(valid.all()):
            projected = encoder.inconv(features)
        else:
            flat = encoder.inconv(features[valid])
            projected = flat.new_zeros(*dates.shape, flat.shape[-1])
            projected[valid] = flat
    else:
        projected = features
    positions = torch.where(valid, dates, dates.new_zeros(())) + encoder.max_temporal_shift
    if (positions[valid] < 0).any() or (positions[valid] > encoder.positional_enc.num_embeddings-1).any():
        raise ValueError('Augmented dates outside original LTAE position table')
    low, high = positions.floor().long(), positions.ceil().long()
    weight = (positions-low).unsqueeze(-1)
    encoded = projected + (1-weight)*encoder.positional_enc(low) + weight*encoder.positional_enc(high)
    pooled, _ = encoder.attention_heads(encoded, valid_time=valid)
    return model.decoder(encoder.dropout(encoder.mlp(pooled)))


def augmented_classification(model, imputer, features, dates, attention, shift=0,
                             detach_prediction=False, generator=None):
    queries = select_gap_queries(dates, attention, generator=generator)
    qvalid = queries['valid']
    prediction = imputer(features, dates, queries['dates'], query_valid=qvalid,
                         anchors=queries['anchors'])['prediction']
    if detach_prediction:
        prediction = prediction.detach()
    dates_float = dates.to(features.dtype)
    merged = torch.cat((features, prediction), dim=1)
    positions = torch.cat((dates_float, queries['dates']), dim=1)
    valid = torch.cat((torch.ones_like(dates, dtype=torch.bool), qvalid), dim=1)
    order = positions.masked_fill(~valid, float('inf')).argsort(dim=1, stable=True)
    positions = positions.gather(1, order)
    valid = valid.gather(1, order)
    merged = merged.gather(1, order[..., None].expand(-1, -1, merged.shape[-1]))
    logits = classify_features(model, merged, positions+shift, valid)
    active_gaps = queries['gap_lengths'][qvalid]
    info = dict(query_count=int(qvalid.sum()),
                gap_mean=float(active_gaps.mean()) if active_gaps.numel() else 0.,
                gap_max=float(active_gaps.max()) if active_gaps.numel() else 0.,
                real_norm=float(features.detach().norm(dim=-1).mean()),
                predicted_norm=float(prediction.detach()[qvalid].norm(dim=-1).mean()) if qvalid.any() else 0.)
    return logits, info


def _compact(features, dates, selected):
    width = int(selected.sum(-1).max())
    x = features.new_zeros(len(features), width, features.shape[-1])
    t = dates.new_zeros(len(features), width)
    valid = torch.zeros_like(t, dtype=torch.bool)
    for b in range(len(features)):
        count = int(selected[b].sum())
        x[b, :count], t[b, :count] = features[b, selected[b]], dates[b, selected[b]]
        valid[b, :count] = True
    return x, t, valid


def feature_reconstruction(model, imputer, features, dates, shift=0, nll_weight=.01,
                           generator=None):
    features = features.detach()
    initial_attention = probe_attention(model, features, dates, shift=shift)
    hidden = select_recovery_mask(dates, initial_attention, generator=generator)
    rows = hidden.any(-1)
    if not rows.any():
        zero = sum(parameter.sum()*0 for parameter in imputer.parameters())
        return dict(loss=zero, count=0, rmse=0., linear_rmse=0., uncertainty=0.)
    features, dates, hidden = features[rows], dates[rows], hidden[rows]
    visible_x, visible_t, visible_valid = _compact(features, dates, ~hidden)
    targets, query_dates, query_valid = _compact(features, dates, hidden)
    # Recompute anchors only after removing hidden observations entirely.
    attention = probe_attention(model, visible_x, visible_t, shift=shift, valid=visible_valid)
    anchors = torch.zeros_like(visible_valid)
    for b in range(len(features)):
        indices = torch.where(visible_valid[b])[0]
        k = max(1, int(np.ceil(.25*len(indices))))
        top = attention[b, :, indices].topk(k, dim=-1).indices.flatten().unique()
        anchors[b, indices[top]] = True
    result = imputer(visible_x, visible_t, query_dates, valid=visible_valid,
                     query_valid=query_valid, anchors=anchors)
    pred, target = result['prediction'][query_valid], targets[query_valid]
    loss = reconstruction_loss(pred, target, result['log_variance'][query_valid], nll_weight)
    return dict(loss=loss, count=int(query_valid.sum()),
                rmse=float((pred.detach()-target).square().mean().sqrt()),
                linear_rmse=float((result['linear'][query_valid]-target).square().mean().sqrt()),
                uncertainty=float(result['log_variance'][query_valid].detach().exp().mean()))


def _write_history(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_imputer_epochs(model, imputer, loader, config, writer=None):
    model.eval().requires_grad_(False)
    imputer.train()
    optimizer = torch.optim.AdamW(imputer.parameters(), lr=config.imputer_lr,
                                  weight_decay=config.imputer_weight_decay)
    if not len(loader):
        raise ValueError('No complete source training batches')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.imputer_epochs*len(loader))
    criterion, history = FocalLoss(gamma=config.focal_loss_gamma), []
    for epoch in range(config.imputer_epochs):
        totals, count = {}, 0
        for batch in loader:
            pixels, mask, dates, extra = sample_to_device(batch, config.device)
            with torch.no_grad():
                features = model.spatial_encoder(pixels, mask, extra)
            rec = feature_reconstruction(model, imputer, features, dates, nll_weight=config.nll_weight)
            classification = features.new_zeros(())
            query_count = 0
            if epoch >= config.pretrain_recovery_epochs:
                attention = probe_attention(model, features, dates)
                logits, info = augmented_classification(model, imputer, features, dates, attention)
                classification = criterion(logits, batch['label'].to(config.device))
                query_count = info['query_count']
            loss = config.reconstruction_weight*rec['loss'] + config.source_aug_weight*classification
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite imputer pretraining loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            row = dict(loss=float(loss.detach()), recovery=float(rec['loss'].detach()),
                       classification=float(classification.detach()), rmse=rec['rmse'],
                       linear_rmse=rec['linear_rmse'], uncertainty=rec['uncertainty'],
                       recovery_points=rec['count'], query_count=query_count)
            for key, value in row.items():
                totals[key] = totals.get(key, 0.) + value
            count += 1
        record = dict(epoch=epoch+1, **{key: value/count for key, value in totals.items()})
        history.append(record)
        _write_history(Path(config.fold_dir)/'pretraining.csv', history)
        print('IMPUTER_PRETRAIN|' + json.dumps(record), flush=True)
        if writer:
            for key, value in record.items():
                writer.add_scalar('imputer_pretrain/'+key, value, epoch+1)
    torch.save(dict(imputer_state_dict=imputer.state_dict(), completed_epochs=config.imputer_epochs,
                    role='source_imputer', source=config.source,
                    source_checkpoint_sha256=config.source_checkpoint_sha256),
               Path(config.fold_dir)/'imputer_epoch20.pt')
    return history


class ImputationRuntime:
    """Explicit opt-in classification/recovery hooks; Teacher stays original."""
    def __init__(self, config, imputer):
        self.config, self.imputer = config, imputer
        self.target_probe_shift = 0
        self.statistics = {}
        self.epoch_rows, self.history = [], []

    def attach(self, student, teacher):
        self.student, self.teacher = student, teacher
        self.imputer.to(next(student.parameters()).device).train()

    def classification(self, pixels, mask, positions, extra, domain, applied_shift=0):
        if domain == 'target' and self.config.variant == 'B':
            return self.student(pixels, mask, positions, extra)
        dates = positions-applied_shift
        features = self.student.spatial_encoder(pixels, mask, extra)
        with torch.no_grad(), evaluation_mode(self.teacher):
            probe_features = self.teacher.spatial_encoder(pixels, mask, extra)
            attention = probe_attention(self.teacher, probe_features, dates,
                                         shift=self.target_probe_shift if domain == 'target' else applied_shift)
        logits, info = augmented_classification(
            self.student, self.imputer, features, dates, attention,
            shift=applied_shift, detach_prediction=domain == 'target')
        self.statistics.update({domain+'/'+key: value for key, value in info.items()})
        return logits

    def recovery(self, sample, domain, shift, device):
        pixels, mask, dates, extra = sample_to_device(sample, device)
        # Recovery sees current features but never updates PSE or BN statistics.
        with torch.no_grad(), evaluation_mode(self.student):
            features = self.student.spatial_encoder(pixels, mask, extra)
        rec = feature_reconstruction(self.student, self.imputer, features, dates,
                                     shift=shift, nll_weight=self.config.nll_weight)
        self.statistics.update({domain+'/recovery_'+key: float(value.detach()) if torch.is_tensor(value) else value
                                for key, value in rec.items()})
        return rec['loss']

    def record_step(self, source_loss, target_loss, pseudo_confidence, pseudo_mask):
        self.epoch_rows.append(dict(self.statistics,
            source_classification=float(source_loss.detach()), target_classification=float(target_loss),
            teacher_confidence=float(pseudo_confidence.mean()),
            pseudo_coverage=float(pseudo_mask.float().mean())))

    def end_epoch(self, epoch, shift):
        keys = sorted(set().union(*(row.keys() for row in self.epoch_rows)))
        record = dict(epoch=epoch, variant=self.config.variant, target_to_source_shift=shift,
                      **{key: sum(row.get(key, 0.) for row in self.epoch_rows)/len(self.epoch_rows) for key in keys})
        self.history.append(record)
        # Stable field set even when a complete epoch has no accepted target samples.
        fields = list(dict.fromkeys(key for row in self.history for key in row))
        with (Path(self.config.fold_dir)/'imputation_training.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.history)
        print('IMPUTATION_UDA|' + json.dumps(record), flush=True)
        self.epoch_rows.clear()


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare(config, source_only=False):
    from train import prepare_data_protocol, create_train_val_test_folds
    config = deepcopy(config)
    if source_only:
        config.target = config.source
    _seed(config.seed)
    config.fold_dir = str(Path(config.output_dir)/'fold_0')
    indices, protocol = prepare_data_protocol(config)
    source_protocol = json.loads((Path(config.weights)/'closed_set_protocol.json').read_text())
    if config.classes != source_protocol['classes']:
        raise ValueError('Historical Source classifier classes differ from the current data protocol')
    # Source-only uses [source,source], exactly as the historical Source script.
    splits = create_train_val_test_folds([config.source, config.target], config.num_folds,
                                         indices, config.val_ratio, config.test_ratio)[0]
    checkpoint = Path(config.weights)/'fold_0/model.pt'
    config.source_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with (Path(config.output_dir)/'resolved_config.json').open('w') as stream:
        json.dump(vars(config), stream, indent=2)
    with (Path(config.output_dir)/'split_indices.json').open('w') as stream:
        json.dump({name: {part: sorted(values) for part, values in split.items()}
                   for name, split in splits.items()}, stream)
    return config, splits


def _source_model(config):
    from models.stclassifier import PseLTae
    model = PseLTae(input_dim=config.input_dim, num_classes=config.num_classes,
                    with_extra=config.with_extra).to(config.device)
    packet = torch.load(Path(config.weights)/'fold_0/model.pt', map_location=config.device, weights_only=False)
    model.load_state_dict(packet['state_dict'])
    return model


def _new_imputer(config):
    return TimePerceiverImputer(**getattr(config, 'imputer_options', {})).to(config.device)


def pretrain_imputer(config):
    from dataset import PixelSetData, create_train_loader, create_evaluation_loaders
    from torchvision.transforms import Compose
    from transforms import RandomSamplePixels, RandomSampleTimeSteps, Normalize, ToTensor
    from torch.utils.tensorboard import SummaryWriter
    config, splits = _prepare(config, source_only=True)
    model, imputer = _source_model(config), _new_imputer(config)
    transform = Compose([RandomSamplePixels(config.num_pixels), RandomSampleTimeSteps(config.seq_length),
                         Normalize(), ToTensor()])
    dataset = PixelSetData(config.data_root, config.source, config.classes, transform,
                           indices=splits[config.source]['train'], closed_set=True,
                           combine_spring_and_winter=False)
    loader = create_train_loader(dataset, config.batch_size, config.num_workers)
    writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
    try:
        train_imputer_epochs(model, imputer, loader, config, writer)
    finally:
        writer.close()
    path = Path(config.fold_dir)/'imputer_epoch20.pt'
    packet = torch.load(path, map_location=config.device, weights_only=False)
    imputer.load_state_dict(packet['imputer_state_dict'])
    _, test_loader = create_evaluation_loaders(config.source, splits, config, config.sample_pixels_val)
    evaluate_endpoints(model, imputer, test_loader, config, config.source, shift=0)


def run_imputation_uda(config):
    from dataset import create_evaluation_loaders
    from timematch import train_timematch
    from torch.utils.tensorboard import SummaryWriter
    config, splits = _prepare(config)
    model, imputer = _source_model(config), _new_imputer(config)
    initial = torch.load(config.imputer_checkpoint, map_location=config.device, weights_only=False)
    if (initial.get('role') != 'source_imputer' or initial.get('completed_epochs') != 20
            or initial.get('source') != config.source
            or initial.get('source_checkpoint_sha256') != config.source_checkpoint_sha256):
        raise ValueError('Expected epoch20 imputer trained on this exact original Source checkpoint')
    imputer.load_state_dict(initial['imputer_state_dict'])
    config.imputer_checkpoint_sha256 = hashlib.sha256(Path(config.imputer_checkpoint).read_bytes()).hexdigest()
    config._imputation_runtime = ImputationRuntime(config, imputer)
    path = Path(config.fold_dir)/'model.pt'
    writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
    try:
        train_timematch(model, config, writer, None, torch.device(config.device), str(path), 0, splits)
    finally:
        writer.close()
    # Test the serialized joint endpoint rather than in-memory or validation weights.
    final = torch.load(path, map_location=config.device, weights_only=False)
    if final.get('role') != 'student' or final.get('completed_epochs') != 20:
        raise ValueError('Final testing requires the completed epoch20 Student')
    model.load_state_dict(final['state_dict'])
    imputer.load_state_dict(final['imputer_state_dict'])
    with (Path(config.fold_dir)/'tested_checkpoint.json').open('w') as stream:
        json.dump(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       completed_epochs=20, role='student',
                       source_checkpoint_sha256=config.source_checkpoint_sha256,
                       imputer_initial_sha256=config.imputer_checkpoint_sha256, variant=config.variant), stream, indent=2)
    for domain, shift in ((config.source, final['source_to_target_shift']), (config.target, 0)):
        _, test_loader = create_evaluation_loaders(domain, splits, config, config.sample_pixels_val)
        evaluate_endpoints(model, imputer, test_loader, config, domain, shift=shift)


@torch.no_grad()
def evaluate_endpoints(model, imputer, loader, config, domain, shift=0):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
    # Stable gap subsampling and pixel sampling make A/B endpoint reports comparable.
    _seed(config.seed+1000)
    model.eval()
    imputer.eval()
    original, augmented, labels, diagnostics = [], [], [], []
    reconstruction = []
    for batch in loader:
        pixels, mask, dates, extra = sample_to_device(batch, config.device)
        features = model.spatial_encoder(pixels, mask, extra)
        attention = probe_attention(model, features, dates, shift=shift)
        original.append(classify_features(model, features, dates+shift).cpu())
        logits, info = augmented_classification(model, imputer, features, dates, attention, shift=shift)
        augmented.append(logits.cpu())
        labels.extend(batch['label'].tolist())
        diagnostics.append(info)
        rec = feature_reconstruction(model, imputer, features, dates, shift=shift, nll_weight=config.nll_weight)
        reconstruction.append({key: float(value) for key, value in rec.items() if key != 'loss'})
    if not labels:
        raise ValueError('Empty final test loader')
    truth = np.asarray(labels)
    arrays = dict(labels=truth, original=torch.cat(original).numpy(), augmented=torch.cat(augmented).numpy())
    prefix = domain.replace('/', '_')
    report = dict(domain=domain, shift=shift, selection='fixed final epoch; no target validation selection',
                  diagnostics=diagnostics, reconstruction=reconstruction)
    for name in ('original', 'augmented'):
        logits = arrays[name]
        predicted = logits.argmax(-1)
        pairs = {}
        for left, right in [('spring_barley', 'meadow'), ('spring_oat', 'spring_barley')]:
            if left not in config.classes or right not in config.classes:
                pairs[left+'__'+right] = 'N/A: class absent from source-defined classifier'
                continue
            a, b = config.classes.index(left), config.classes.index(right)
            pairs[left+'__'+right] = {
                left: float((logits[truth == a, a]-logits[truth == a, b]).mean()) if (truth == a).any() else None,
                right: float((logits[truth == b, b]-logits[truth == b, a]).mean()) if (truth == b).any() else None}
        report[name] = dict(accuracy=float(accuracy_score(truth, predicted)),
                            macro_f1=float(f1_score(truth, predicted, average='macro', zero_division=0)),
                            classification_report=classification_report(truth, predicted,
                                labels=list(range(len(config.classes))), target_names=config.classes,
                                output_dict=True, zero_division=0),
                            confusion_matrix=confusion_matrix(truth, predicted,
                                labels=list(range(len(config.classes)))).tolist(), pairwise_margins=pairs)
    with (Path(config.fold_dir)/('test_'+prefix+'.json')).open('w') as stream:
        json.dump(report, stream, indent=2)
    np.savez_compressed(Path(config.fold_dir)/('logits_'+prefix+'.npz'), **arrays)
    print('IMPUTATION_TEST|' + json.dumps(dict(domain=domain,
        original_f1=report['original']['macro_f1'], augmented_f1=report['augmented']['macro_f1'])), flush=True)
    return report
