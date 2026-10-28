"""Losses, selection monitoring and source-only paired evaluation for M1."""

import numpy as np
import torch
import torch.nn.functional as F


def m1_loss(full, keep, info, labels, criterion, lambda_keep=1., lambda_budget=.1, sharpness=1.):
    losses = dict(full=criterion(full, labels), keep=F.cross_entropy(keep, labels),
                  ratio=info['ratio_loss'], sharpness=info['sharpness_loss'])
    # The sharpness term safeguards the selector; it is not a class/pair/importance loss.
    losses['budget'] = losses['ratio'] + sharpness*losses['sharpness']
    losses['total'] = losses['full'] + lambda_keep*losses['keep'] + lambda_budget*losses['budget']
    return losses['total'], losses


def selection_metrics(info):
    return dict(gate_fraction=float(info['selection_fraction'].detach().mean()),
                gate_std_mean=float(info['gate_std_per_head'].detach().mean()),
                gate_std_min=float(info['gate_std_per_head'].detach().min()),
                near_zero=float(info['near_zero'].detach()), near_one=float(info['near_one'].detach()),
                effective_count=float(info['effective_count'].float().mean()),
                retained_attention_mass=float(info['retained_mass'].detach().mean()))


@torch.no_grad()
def paired_source_evaluation(model, loader, config, device, tau):
    """Full and Keep see the same source-test batches/pixels; no target data."""
    import sklearn.metrics as metrics
    from utils.focal_loss import FocalLoss
    model.eval()
    labels, predictions = [], {'full': [], 'keep': []}
    sums, total = {'full': 0., 'keep': 0.}, 0
    criterion = FocalLoss(gamma=config.focal_loss_gamma)
    for sample in loader:
        pixels, mask, positions, extra = [sample[key].to(device) for key in ('pixels', 'valid_pixels', 'positions', 'extra')]
        targets = sample['label'].to(device)
        full, keep, _ = model.forward_m1(pixels, mask, positions, extra, tau=tau, rho=config.m1_rho)
        n = len(targets)
        sums['full'] += float(criterion(full, targets))*n
        sums['keep'] += float(F.cross_entropy(keep, targets))*n
        total += n
        labels.extend(targets.cpu().tolist())
        predictions['full'].extend(full.argmax(1).cpu().tolist())
        predictions['keep'].extend(keep.argmax(1).cpu().tolist())
    if not total:
        raise ValueError('M1 requires nonempty source test data')
    result = {}
    class_ids = list(range(len(config.classes)))
    for name, predicted in predictions.items():
        result[name] = dict(accuracy=metrics.accuracy_score(labels, predicted), loss=sums[name]/total,
                            macro_f1=metrics.f1_score(labels, predicted, average='macro', zero_division=0),
                            weighted_f1=metrics.f1_score(labels, predicted, average='weighted', zero_division=0),
                            kappa=metrics.cohen_kappa_score(labels, predicted, labels=class_ids),
                            classification_report=metrics.classification_report(labels, predicted, labels=class_ids,
                                                                                target_names=config.classes, zero_division=0),
                            confusion_matrix=metrics.confusion_matrix(labels, predicted, labels=class_ids))
    return result
