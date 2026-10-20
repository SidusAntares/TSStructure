import os
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from tqdm import tqdm
import sklearn.metrics
from utils.train_utils import AverageMeter, progress_bar_disabled, to_cuda


def classification_metrics(y_true, y_pred, class_names, mode='val'):
    """Compute original TimeMatch metrics plus fixed-universe diagnostics."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.arange(len(class_names))
    precision, recall, per_class_f1, support = (
        sklearn.metrics.precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0,
        )
    )
    return {
        'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred),
        'macro_f1': sklearn.metrics.f1_score(
            y_true, y_pred, average='macro', zero_division=0,
        ),
        'weighted_f1': sklearn.metrics.f1_score(
            y_true, y_pred, average='weighted', zero_division=0,
        ),
        'kappa': sklearn.metrics.cohen_kappa_score(
            y_true, y_pred, labels=labels.tolist(),
        ),
        'classification_report': sklearn.metrics.classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            zero_division=0,
        ),
        'confusion_matrix': sklearn.metrics.confusion_matrix(
            y_true, y_pred, labels=labels,
        ),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': per_class_f1.tolist(),
        'support': support.astype(np.int64).tolist(),
    }


def validation(best_f1, best_model_path, config, criterion, device, epoch, model, val_loader, writer, temporal_shift=None):
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
    print(f"Validation result: loss={val_loss:.4f}, acc={val_acc:.2f}, f1={val_f1:.4f}")
    if 'shape_head_macro_f1' in val_metrics:
        writer.add_scalar(
            'val/shape_head_macro_f1', val_metrics['shape_head_macro_f1'],
            global_step=epoch,
        )
        writer.add_scalar(
            'val/shape_head_fixed_class_macro_f1',
            val_metrics['shape_head_fixed_class_macro_f1'], global_step=epoch,
        )
        print(
            'SHAPE_AUX_VALIDATION|'
            f'epoch={epoch}|accuracy={val_metrics["shape_head_accuracy"]:.6f}|'
            f'macro_f1={val_metrics["shape_head_macro_f1"]:.6f}|'
            f'fixed_class_macro_f1={val_metrics["shape_head_fixed_class_macro_f1"]:.6f}'
        )
    if val_f1 > best_f1:
        print(f'Validation F1 improved from {best_f1:.4f} to {val_f1:.4f}!')
        best_f1 = val_f1
        if best_model_path is not None:
            print(f'Saving best model to {best_model_path}')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_f1': best_f1}, best_model_path)
    else:
        print(f'Validation F1 did not improve from {best_f1:.4f}.')
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
    y_true, y_pred, shape_y_pred = [], [], []
    collect_shape_head = (
        getattr(model, 'shape_representation', None) == 'direct_shape_token'
    )

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
            if hasattr(model, "forward_with_temporal_shift"):
                if collect_shape_head:
                    output = model.forward_with_temporal_shift(
                        pixels, valid_pixels, positions, extra,
                        temporal_shift=temporal_shift, return_dict=True,
                    )
                    logits = output['logits']
                else:
                    logits = model.forward_with_temporal_shift(
                        pixels, valid_pixels, positions, extra,
                        temporal_shift=temporal_shift,
                    )
            else:
                logits = model.forward(
                    pixels,
                    valid_pixels,
                    positions + temporal_shift,
                    extra,
                )
        else:
            if collect_shape_head:
                output = model.forward(
                    pixels, valid_pixels, positions, extra, return_dict=True,
                )
                logits = output['logits']
            else:
                logits = model.forward(pixels, valid_pixels, positions, extra)

        if collect_shape_head:
            shape_y_pred.extend(output['shape_logits'].argmax(dim=1).tolist())

        predictions = logits.argmax(dim=1)

        if criterion is not None:
            loss = criterion(logits, target)
            loss_meter.update(loss.item(), n=pixels.size(0))
        y_pred.extend(predictions.tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    metrics = classification_metrics(y_true, y_pred, class_names, mode=mode)
    metrics['loss'] = loss_meter.avg
    if collect_shape_head:
        shape_metrics = classification_metrics(
            y_true, np.asarray(shape_y_pred), class_names, mode=mode,
        )
        metrics.update({
            'shape_head_accuracy': shape_metrics['accuracy'],
            'shape_head_macro_f1': shape_metrics['macro_f1'],
            'shape_head_fixed_class_macro_f1': float(np.mean(
                shape_metrics['per_class_f1']
            )),
        })

    return metrics
