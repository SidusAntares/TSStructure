"""Read-only Student snapshots and head interventions for the TimeMatch audit."""

import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from utils.head_evidence import build_counterfactual_head_embeddings


STAGES = (0, 10, 20)
STRUCTURES = ('mass75', 'count75', 'top30')
INTERVENTIONS = ('mean', 'gate01', 'gate05')
CLASSES = ('spring_barley', 'spring_wheat', 'spring_oat')


def select_structure(weights, dates, method, top_fraction=.30):
    """Shortest calendar interval, intersected with Top-30% observed points.

    Equal attention ties use observation order; equal interval widths use the
    earliest start. No interpolated points participate in selection/ablation.
    """
    weights, dates = np.asarray(weights), np.asarray(dates)
    if (weights.ndim != 1 or weights.shape != dates.shape or not len(weights)
            or not np.isfinite(weights).all() or not np.isfinite(dates).all()
            or (weights < 0).any() or (np.diff(dates) < 0).any()):
        raise ValueError('Expected finite nonnegative weights on sorted observed dates')
    if method not in STRUCTURES or not 0 < top_fraction <= 1:
        raise ValueError('Unknown structure method or invalid top fraction')
    count = max(1, math.ceil(len(weights) * top_fraction))
    top = np.sort(np.argsort(-weights, kind='stable')[:count])
    mask = np.zeros(len(weights), dtype=bool)
    if method == 'top30':
        mask[top] = True
        return mask
    scores = weights[top] if method == 'mass75' else np.ones(count)
    required = .75 * scores.sum() if method == 'mass75' else math.ceil(.75 * count)
    best = None
    for left in range(count):
        total = 0.
        for right in range(left, count):
            total += scores[right]
            if total + 1e-12 >= required:
                candidate = (dates[top[right]] - dates[top[left]], left, right)
                if best is None or candidate < best:
                    best = candidate
                break
    if best is None:
        raise ValueError('No interval covers the requested attention')
    _, left, right = best
    mask[top[left:right+1]] = True
    return mask


def counterfactual_embeddings(attention, values, mask, method):
    """Original plus H one-head interventions; attention is never renormalized."""
    if method == 'mean':
        return build_counterfactual_head_embeddings(attention, values, mask)
    if method not in INTERVENTIONS:
        raise ValueError('Unknown intervention')
    g = .1 if method == 'gate01' else .5
    original = (attention[..., None] * values).sum(-2)
    structure = (attention[..., None] * values * mask[..., None]).sum(-2)
    heads = attention.shape[1]
    identity = torch.eye(heads, device=values.device, dtype=values.dtype)
    altered = original[:, None] - (1-g)*identity[None, :, :, None]*structure[:, None]
    return torch.cat([original.flatten(1)[:, None], altered.flatten(2)], 1)


def fourier_display(dates, signal):
    """FreDN's 13-mode irregular Fourier ridge fit + dense iFFT for display only.

    This operates on Source-Train PC1-projected PSE features, never attention
    weights or values supplied to the classifier. Observed markers are sampled
    from the same fitted Fourier series as the display curve.
    """
    dates = np.asarray(dates, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    modes = np.arange(-6, 7)
    design = np.exp(2j*np.pi*dates[:, None]*modes/365.)
    coefficients = np.linalg.solve(
        design.conj().T @ design + 1e-3*np.eye(len(modes)),
        design.conj().T @ signal,
    )
    spectrum = np.zeros(365, dtype=np.complex128)
    spectrum[modes % 365] = coefficients*365
    grid = np.arange(365, dtype=np.float64)
    return grid, np.fft.ifft(spectrum).real, (design @ coefficients).real


def load_final_student(path):
    packet = torch.load(path, map_location='cpu', weights_only=False)
    if packet.get('completed_epochs') != 20 or packet.get('role') != 'student':
        raise ValueError('Final testing requires the completed 20-epoch Student, never validation-best')
    return packet


def save_student_snapshot(model, config, splits, directory, completed_epochs,
                          source_shift, teacher_shift):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'checkpoint_epoch{completed_epochs:02d}.pt'
    torch.save({
        'state_dict': {key: value.detach().cpu().clone()
                       for key, value in model.state_dict().items()},
        'completed_epochs': completed_epochs, 'role': 'student',
        'config': dict(vars(config)), 'splits': splits,
        'student_shifts': {'source': int(source_shift), 'target': 0},
        'teacher_target_to_source_shift': int(teacher_shift),
    }, path)
    return path


class StudentDiagnostic:
    """Synchronous child process: no extra training step or parent RNG draw.

    Each GPU pauses only its own task at a diagnostic stage. Exceptions and
    diagnostic stdout/stderr are kept in that task's diagnostic log.
    """

    def __init__(self, config, splits):
        if (config.epochs != 20 or not config.output_student
                or config.model != 'pseltae' or config.num_folds != 1
                or getattr(config, 'evidence_mode', 'off') != 'off'):
            raise ValueError('Diagnostic run requires original PSE-LTAE Student, 20 epochs, one fold, evidence off')
        self.config, self.splits = config, splits
        self.root = Path(config.diagnostic_dir)
        self.log = Path(config.diagnostic_log)
        self.log.parent.mkdir(parents=True, exist_ok=True)

    def capture(self, stage, student, source_shift, teacher_shift):
        if stage not in STAGES:
            return
        with self.log.open('a', encoding='utf-8', buffering=1) as log:
            try:
                snapshot = save_student_snapshot(
                    student, self.config, self.splits, self.root / 'checkpoints',
                    stage, source_shift, teacher_shift,
                )
                print(f'DIAGNOSTIC_START|task={self.config.diagnostic_task}|epoch={stage}',
                      file=log, flush=True)
                subprocess.run([
                    sys.executable, '-u', '-m', 'analysis.timematch_diagnostic', 'render',
                    '--snapshot', str(snapshot.resolve()), '--output', str(self.root.resolve()),
                    '--task', self.config.diagnostic_task,
                ], stdout=log, stderr=log, check=True)
                print(f'DIAGNOSTIC_COMPLETE|epoch={stage}', file=log, flush=True)
            except Exception:
                import traceback
                traceback.print_exc(file=log)
                raise RuntimeError(f'Diagnostic failed; see {self.log}') from None
