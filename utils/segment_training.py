"""Source-only structure finetuning and opt-in structure-guided TimeMatch UDA.

The content-only variant adapts INSHAPE's segment selection to the same PSE-LTAE
predictor as the attention variant. It is not a reproduction of InceptionTime.
"""
import csv
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from models.stclassifier import PseLTae
from models.segment_selector import SegmentSelector
from utils.segment_structure import structure_forward
from utils.focal_loss import FocalLoss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_to_device(sample, device):
    return tuple(sample[key].to(device) if key in sample else None
                 for key in ('pixels', 'valid_pixels', 'positions', 'extra'))


def structure_statistics(info):
    return {key: float(info[key].detach().float().mean())
            for key in ('selected_count', 'candidate_count', 'coverage', 'fallback_rate')}


def write_history(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_source_structure_epochs(model, selector, loader, config, writer=None):
    if not len(loader):
        raise ValueError('No complete source batches for structure finetuning')
    optimizer = torch.optim.Adam([{'params': model.parameters()},
        {'params': selector.parameters(), 'lr': config.selector_lr}],
        lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs*len(loader))
    criterion, history = FocalLoss(gamma=config.focal_loss_gamma), []
    for epoch in range(config.epochs):
        model.train()
        selector.train()
        rows = []
        for sample in loader:
            pixels, mask, dates, extra = sample_to_device(sample, config.device)
            _, keep, info = structure_forward(model, selector, pixels, mask, dates, extra,
                variant=config.variant, full_loss=False, unified=True, penalty=config.pelt_penalty,
                min_size=config.pelt_min_size, penalty_mode=config.pelt_penalty_mode)
            labels = sample['label'].to(config.device)
            keep_loss = criterion(keep, labels)
            loss = keep_loss+config.sparse_weight*info['sparse_loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite source structure loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            rows.append(dict(total=float(loss.detach()), keep=float(keep_loss.detach()),
                sparse=float(info['sparse_loss'].detach()),
                **structure_statistics(info)))
        record = dict(epoch=epoch+1, variant=config.variant,
                      **{key: sum(row[key] for row in rows)/len(rows) for key in rows[0]})
        history.append(record)
        write_history(Path(config.fold_dir)/'structure_finetuning.csv', history)
        print('STRUCTURE_FINETUNE|' + json.dumps(record), flush=True)
        if writer:
            for key, value in record.items():
                if isinstance(value, (float, int)):
                    writer.add_scalar('structure_finetune/'+key, value, epoch+1)
    torch.save(dict(state_dict=model.state_dict(), selector_state_dict=selector.state_dict(),
                    role='source_structure', completed_epochs=config.epochs, variant=config.variant,
                    classification_path='keep_only_unified',
                    selector_protocol=getattr(config, 'selector_protocol', 'legacy'),
                    source=config.source, original_source_checkpoint_sha256=config.original_source_checkpoint_sha256),
               Path(config.fold_dir)/'model.pt')
    return history


class StructureRuntime:
    def __init__(self, config, selector):
        self.config, self.selector = config, selector
        self.history, self.rows, self.statistics = [], [], {}
        self.reset_step()

    def attach(self, student, teacher):
        self.student, self.teacher = student, teacher
        self.selector.to(next(student.parameters()).device).train()

    def reset_step(self):
        self.full_source = self.keep_source = self.source_info = self.target_info = None
        self.statistics = {}

    def classification(self, pixels, mask, positions, extra, domain='source', applied_shift=0):
        _, keep, info = structure_forward(self.student, self.selector, pixels, mask, positions, extra,
            variant=self.config.variant, full_loss=False, unified=True,
            penalty=self.config.pelt_penalty, min_size=self.config.pelt_min_size,
            penalty_mode=self.config.pelt_penalty_mode)
        self.statistics.update({domain+'/'+key: value for key, value in structure_statistics(info).items()})
        if domain == 'source':
            self.keep_source, self.source_info = keep, info
        else:
            self.target_info = info
        return keep

    def source_loss(self, criterion, labels):
        keep = criterion(self.keep_source, labels)
        self.statistics.update(source_keep=float(keep.detach()))
        return keep

    def sparse_loss(self):
        loss = self.source_info['sparse_loss']
        if self.target_info is not None:
            loss = loss+self.target_info['sparse_loss']
        self.statistics['sparse'] = float(loss.detach())
        return self.config.sparse_weight*loss

    def record_step(self, source_loss, target_loss, confidence, pseudo_mask):
        self.rows.append(dict(self.statistics, source_classification=float(source_loss.detach()),
            target_classification=float(target_loss), teacher_confidence=float(confidence.mean()),
            pseudo_coverage=float(pseudo_mask.float().mean())))

    def end_epoch(self, epoch, shift):
        keys = sorted(set().union(*(row.keys() for row in self.rows)))
        record = dict(epoch=epoch, variant=self.config.variant, target_to_source_shift=shift,
            **{key: sum(row.get(key, 0.) for row in self.rows)/len(self.rows) for key in keys})
        self.history.append(record)
        write_history(Path(self.config.fold_dir)/'structure_uda.csv', self.history)
        print('STRUCTURE_UDA|' + json.dumps(record), flush=True)
        self.rows.clear()


def prepare(config, source_only=False):
    from train import prepare_data_protocol, create_train_val_test_folds
    config = deepcopy(config)
    config.original_weights = config.weights
    if source_only:
        config.target = config.source
    seed_all(config.seed)
    config.fold_dir = str(Path(config.output_dir)/'fold_0')
    indices, _ = prepare_data_protocol(config)
    original_protocol = json.loads((Path(config.original_weights)/'closed_set_protocol.json').read_text())
    if config.classes != original_protocol['classes']:
        raise ValueError('Original Source classifier class order differs from current closed-set protocol')
    splits = create_train_val_test_folds([config.source, config.target], config.num_folds,
        indices, config.val_ratio, config.test_ratio)[0]
    original = Path(config.original_weights)/'fold_0/model.pt'
    config.original_source_checkpoint_sha256 = hashlib.sha256(original.read_bytes()).hexdigest()
    with (Path(config.output_dir)/'split_indices.json').open('w') as stream:
        json.dump({domain: {part: sorted(ids) for part, ids in split.items()}
                   for domain, split in splits.items()}, stream)
    return config, splits


def write_resolved(config):
    with (Path(config.output_dir)/'resolved_config.json').open('w') as stream:
        json.dump(vars(config), stream, indent=2)


def new_models(config):
    model = PseLTae(input_dim=config.input_dim, num_classes=config.num_classes,
                   with_extra=config.with_extra).to(config.device)
    selector = SegmentSelector(width=config.selector_width, heads=config.selector_heads,
                               dropout=config.selector_dropout,
                               protocol=getattr(config, 'selector_protocol', 'legacy')).to(config.device)
    return model, selector


def finetune_structure(config):
    from dataset import PixelSetData, create_train_loader, create_evaluation_loaders
    from torchvision.transforms import Compose
    from transforms import RandomSamplePixels, RandomSampleTimeSteps, Normalize, ToTensor
    from torch.utils.tensorboard import SummaryWriter
    config, splits = prepare(config, source_only=True)
    write_resolved(config)
    model, selector = new_models(config)
    checkpoint = Path(config.weights)/'fold_0/model.pt'
    model.load_state_dict(torch.load(checkpoint, map_location=config.device, weights_only=False)['state_dict'])
    transform = Compose([RandomSamplePixels(config.num_pixels), RandomSampleTimeSteps(config.seq_length),
                         Normalize(), ToTensor()])
    dataset = PixelSetData(config.data_root, config.source, config.classes, transform,
        indices=splits[config.source]['train'], closed_set=True, combine_spring_and_winter=False)
    loader = create_train_loader(dataset, config.batch_size, config.num_workers)
    writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
    try:
        train_source_structure_epochs(model, selector, loader, config, writer)
    finally:
        writer.close()
    packet = torch.load(Path(config.fold_dir)/'model.pt', map_location=config.device, weights_only=False)
    if packet['completed_epochs'] != 20 or packet['role'] != 'source_structure':
        raise ValueError('Testing requires the completed epoch20 source structure checkpoint')
    model.load_state_dict(packet['state_dict'])
    selector.load_state_dict(packet['selector_state_dict'])
    _, test_loader = create_evaluation_loaders(config.source, splits, config, config.sample_pixels_val)
    evaluate_structure(model, selector, test_loader, config, config.source, 0)


def run_structure_uda(config):
    from timematch import train_timematch
    from dataset import create_evaluation_loaders
    from torch.utils.tensorboard import SummaryWriter
    config, splits = prepare(config)
    model, selector = new_models(config)
    initial_path = Path(config.structure_checkpoint)
    initial = torch.load(initial_path, map_location=config.device, weights_only=False)
    if (initial.get('role') != 'source_structure' or initial.get('completed_epochs') != 20
        or initial.get('classification_path') != 'keep_only_unified'
        or initial.get('selector_protocol', 'legacy') != getattr(config, 'selector_protocol', 'legacy')
        or initial.get('variant') != config.variant or initial.get('source') != config.source
        or initial.get('original_source_checkpoint_sha256') != config.original_source_checkpoint_sha256):
        raise ValueError('UDA requires this variant\'s epoch20 structure checkpoint from the exact original Source')
    if initial_path.name != 'model.pt' or initial_path.parent.name != 'fold_0':
        raise ValueError('Expected structure checkpoint in <finetune>/fold_0/model.pt')
    model.load_state_dict(initial['state_dict'])
    selector.load_state_dict(initial['selector_state_dict'])
    config.weights = str(initial_path.parent.parent)
    config.structure_checkpoint_sha256 = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    # Historical Source training shuffles [source,source], while UDA shuffles
    # [source,target]. Preserve both archived protocols, but evaluate Source only
    # on parcels held out across both stages (Target's test set is unchanged).
    fine_splits = json.loads((initial_path.parent.parent/'split_indices.json').read_text())
    evaluation_splits = deepcopy(splits)
    evaluation_splits[config.source]['test'] &= set(fine_splits[config.source]['test'])
    with (Path(config.output_dir)/'evaluation_split_indices.json').open('w') as stream:
        json.dump({domain: {part: sorted(ids) for part, ids in split.items()}
                   for domain, split in evaluation_splits.items()}, stream)
    config.source_evaluation_protocol = 'intersection of historical Source/fine and UDA held-out test parcels'
    write_resolved(config)
    config._structure_runtime = StructureRuntime(config, selector)
    path = Path(config.fold_dir)/'model.pt'
    writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
    try:
        train_timematch(model, config, writer, None, torch.device(config.device), str(path), 0, splits)
    finally:
        writer.close()
    final = torch.load(path, map_location=config.device, weights_only=False)
    if final.get('role') != 'student' or final.get('completed_epochs') != 20:
        raise ValueError('Final testing requires the completed epoch20 Student')
    model.load_state_dict(final['state_dict'])
    selector.load_state_dict(final['selector_state_dict'])
    with (Path(config.fold_dir)/'tested_checkpoint.json').open('w') as stream:
        json.dump(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            role='student', completed_epochs=20, variant=config.variant,
            structure_initial_sha256=config.structure_checkpoint_sha256,
            original_source_checkpoint_sha256=config.original_source_checkpoint_sha256), stream, indent=2)
    for domain, shift in ((config.source, final['source_to_target_shift']), (config.target, 0)):
        _, test_loader = create_evaluation_loaders(domain, evaluation_splits, config, config.sample_pixels_val)
        evaluate_structure(model, selector, test_loader, config, domain, shift)


@torch.no_grad()
def evaluate_structure(model, selector, loader, config, domain, shift):
    from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
    seed_all(config.seed+1000)
    model.eval()
    selector.eval()
    print(f'STRUCTURE_TEST_START|variant={config.variant}|domain={domain}|checkpoint_epoch=20', flush=True)
    full_rows, keep_rows, labels, statistics = [], [], [], []
    for index, sample in enumerate(loader):
        pixels, mask, dates, extra = sample_to_device(sample, config.device)
        full, keep, info = structure_forward(model, selector, pixels, mask, dates+shift, extra,
            variant=config.variant, deterministic=True, full_loss=True,
            penalty=config.pelt_penalty, min_size=config.pelt_min_size,
            penalty_mode=config.pelt_penalty_mode)
        full_rows.append(full.cpu())
        keep_rows.append(keep.cpu())
        labels.extend(sample['label'].tolist())
        statistics.append(structure_statistics(info))
        if index % 10 == 0:
            print(f'STRUCTURE_TEST_PROGRESS|domain={domain}|batches={index+1}/{len(loader)}', flush=True)
    if not labels:
        raise ValueError('Empty structure final test loader')
    truth = np.asarray(labels)
    arrays = dict(labels=truth, full=torch.cat(full_rows).numpy(), keep=torch.cat(keep_rows).numpy())
    report = dict(domain=domain, variant=config.variant, shift=shift, completed_epochs=20,
        structure_statistics={key: sum(row[key] for row in statistics)/len(statistics) for key in statistics[0]})
    for name in ('full', 'keep'):
        predictions = arrays[name].argmax(-1)
        text = classification_report(truth, predictions, labels=list(range(config.num_classes)),
                                     target_names=config.classes, zero_division=0)
        report[name] = dict(accuracy=float(accuracy_score(truth, predictions)),
            macro_f1=float(f1_score(truth, predictions, average='macro', zero_division=0)),
            classification_report=classification_report(truth, predictions, labels=list(range(config.num_classes)),
                target_names=config.classes, output_dict=True, zero_division=0),
            confusion_matrix=confusion_matrix(truth, predictions, labels=list(range(config.num_classes))).tolist())
        print(f"STRUCTURE_TEST|variant={config.variant}|domain={domain}|input={name}|accuracy={report[name]['accuracy']:.6f}|macro_f1={report[name]['macro_f1']:.6f}", flush=True)
        print(text, flush=True)
    prefix = domain.replace('/', '_')
    with (Path(config.fold_dir)/f'test_{prefix}.json').open('w') as stream:
        json.dump(report, stream, indent=2)
    np.savez_compressed(Path(config.fold_dir)/f'logits_{prefix}.npz', **arrays)
    return report
