import os
import time
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from tqdm import tqdm
import sklearn.metrics
from utils.train_utils import (
    AverageMeter,
    format_log_block,
    progress_bar_disabled,
    to_cuda,
)


def format_validation_summary(
    metrics,
    best_before,
    best_after,
    checkpoint_saved,
    validation_seconds,
):
    return format_log_block(
        "[VALIDATION]",
        [
            f"loss: {metrics['loss']:.6f}",
            f"accuracy: {metrics['accuracy']:.6f}",
            f"macro_f1: {metrics['macro_f1']:.6f}",
            f"kappa: {metrics['kappa']:.6f}",
            f"best_macro_f1_before: {best_before:.6f}",
            f"best_macro_f1_after: {best_after:.6f}",
            f"checkpoint_saved: {str(checkpoint_saved).lower()}",
            f"validation_time: {validation_seconds:.2f} s",
        ],
        border="-",
    )


def validation(best_f1, best_model_path, config, criterion, device, epoch, model, val_loader, writer, temporal_shift=None):
    validation_started = time.perf_counter()
    val_metrics = evaluation(
        model,
        val_loader,
        device,
        config.classes,
        criterion,
        mode='val',
        temporal_shift=temporal_shift,
        progress_bar=getattr(config, "progress_bar", "auto"),
    )
    val_loss, val_acc, val_f1, val_kappa = val_metrics['loss'], val_metrics['accuracy'], val_metrics['macro_f1'], val_metrics['kappa']
    writer.add_scalar('val/loss', val_loss, global_step=epoch)
    writer.add_scalar('val/accuracy', val_acc, global_step=epoch)
    writer.add_scalar('val/f1', val_f1, global_step=epoch)
    writer.add_scalar('val/kappa', val_kappa, global_step=epoch)
    best_before = best_f1
    checkpoint_saved = False
    if val_f1 > best_f1:
        best_f1 = val_f1
        if best_model_path is not None:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_f1': best_f1}, best_model_path)
            checkpoint_saved = True
    print(format_validation_summary(
        val_metrics,
        best_before,
        best_f1,
        checkpoint_saved,
        time.perf_counter() - validation_started,
    ))
    return best_f1


@torch.no_grad()
def evaluation(
    model,
    data_loader,
    device,
    class_names,
    criterion=None,
    mode='val',
    temporal_shift=None,
    progress_bar='auto',
):
    y_true, y_pred = [], []

    loss_meter = AverageMeter()

    model.eval()
    for sample in tqdm(
        data_loader,
        desc='Validating' if mode == 'val' else 'Testing',
        disable=progress_bar_disabled(progress_bar),
    ):
        target = sample['label']
        y_true.extend(target.tolist())
        target = target.cuda(device=device, non_blocking=True)

        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        if temporal_shift is not None:
            logits = model.forward(pixels, valid_pixels, positions + temporal_shift, extra)
        else:
            logits = model.forward(pixels, valid_pixels, positions, extra)

        predictions = logits.argmax(dim=1)

        if criterion is not None:
            loss = criterion(logits, target)
            loss_meter.update(loss.item(), n=pixels.size(0))
        y_pred.extend(predictions.tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    metrics = {
        'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred),
        'loss': loss_meter.avg,
        'macro_f1': sklearn.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': sklearn.metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'kappa': sklearn.metrics.cohen_kappa_score(y_true, y_pred, labels=list(range(len(class_names)))),
        'classification_report': sklearn.metrics.classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0),
        'confusion_matrix': sklearn.metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))),
   }

    return metrics
