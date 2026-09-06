"""Training-only class-conditional structural alignment for TimeMatch."""

from contextlib import contextmanager
from dataclasses import dataclass
import random
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.fredn.nufft import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
)
from models.fredn.structural_probe import (
    detect_structural_landmarks,
    fit_source_class_projections,
    robust_signal_scale,
    topology_signature,
)


@contextmanager
def preserve_rng_state(seed=None):
    """Run isolated random work without advancing the caller's RNG streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _robust_normalize(curves: torch.Tensor, detach_statistics: bool) -> torch.Tensor:
    statistics = curves.detach() if detach_statistics else curves
    center = torch.quantile(statistics, 0.5, dim=1, keepdim=True)
    lower = torch.quantile(statistics, 0.25, dim=1, keepdim=True)
    upper = torch.quantile(statistics, 0.75, dim=1, keepdim=True)
    eps = torch.finfo(curves.dtype).eps
    return (curves - center) / (upper - lower).clamp_min(eps)


def _stable_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_centered = left - left.mean(dim=1, keepdim=True)
    right_centered = right - right.mean(dim=1, keepdim=True)
    numerator = (left_centered * right_centered).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(left_centered, dim=1)
        * torch.linalg.vector_norm(right_centered, dim=1)
    )
    eps = torch.finfo(left.dtype).eps
    correlation = (numerator / denominator.clamp_min(eps)).clamp(-1.0, 1.0)
    return torch.where(torch.isfinite(correlation), correlation, torch.zeros_like(correlation))


class SourceShapeReferenceBank(nn.Module):
    """Frozen source-only PCA projections, prototypes, and canonical events."""

    def __init__(
        self,
        class_projections: torch.Tensor,
        grid: torch.Tensor,
        prototypes: Mapping[int, torch.Tensor],
        event_masks: Mapping[int, torch.Tensor],
        peak_masks: Mapping[int, torch.Tensor],
        valley_masks: Mapping[int, torch.Tensor],
        sample_counts: torch.Tensor,
        *,
        period_days: float,
        reg: float,
        prominence_rel: float,
        min_distance_days: float,
    ):
        super().__init__()
        modes = tuple(sorted(int(mode) for mode in prototypes))
        if not modes or any(mode not in (9, 13) for mode in modes):
            raise ValueError("shape modes must be selected from 9 and 13")
        self.modes = modes
        self.period_days = float(period_days)
        self.reg = float(reg)
        self.prominence_rel = float(prominence_rel)
        self.min_distance_days = float(min_distance_days)
        self.register_buffer("class_projections", class_projections.detach())
        self.register_buffer("grid", grid.detach())
        self.register_buffer("sample_counts", sample_counts.detach().long())
        for mode in modes:
            self.register_buffer(f"prototype_{mode}", prototypes[mode].detach())
            self.register_buffer(f"event_mask_{mode}", event_masks[mode].detach().bool())
            self.register_buffer(f"peak_mask_{mode}", peak_masks[mode].detach().bool())
            self.register_buffer(f"valley_mask_{mode}", valley_masks[mode].detach().bool())
        self.requires_grad_(False)

    @property
    def prototypes(self) -> Dict[int, torch.Tensor]:
        return {mode: getattr(self, f"prototype_{mode}") for mode in self.modes}

    @classmethod
    def from_source_features(
        cls,
        spatial_features: torch.Tensor,
        positions: torch.Tensor,
        labels: torch.Tensor,
        *,
        modes: Sequence[int] = (13,),
        grid_points: int = 64,
        period_days: float = 365.0,
        reg: float = 1e-3,
        prominence_rel: float = 0.15,
        min_distance_days: float = 14.0,
    ):
        if spatial_features.ndim != 3 or positions.shape != spatial_features.shape[:2]:
            raise ValueError("source features/positions must be [N,L,D] and [N,L]")
        if labels.ndim != 1 or labels.shape[0] != spatial_features.shape[0]:
            raise ValueError("source labels must be [N]")
        modes = tuple(sorted(set(int(mode) for mode in modes)))
        if not modes or any(mode not in (9, 13) for mode in modes):
            raise ValueError("shape modes must be selected from 9 and 13")
        if grid_points < 3:
            raise ValueError("shape_grid_points must be at least 3")

        class_ids = torch.unique(labels, sorted=True)
        expected = torch.arange(
            int(class_ids[-1].item()) + 1, device=labels.device, dtype=labels.dtype
        )
        if not torch.equal(class_ids, expected):
            raise ValueError("source reference requires contiguous represented classes")
        projection_map = fit_source_class_projections(spatial_features, labels)
        projections = torch.stack([projection_map[int(c.item())] for c in class_ids])
        support = positions.to(spatial_features.dtype).reshape(-1)
        start = torch.quantile(support, 0.05)
        end = torch.quantile(support, 0.95)
        if not bool(torch.isfinite(start) & torch.isfinite(end)) or end <= start:
            raise ValueError("source timestamps do not define a finite temporal support")
        grid = torch.linspace(start, end, grid_points, device=positions.device,
                              dtype=spatial_features.dtype)
        dense_positions = grid.unsqueeze(0).expand(spatial_features.shape[0], -1)
        sample_counts = torch.stack([(labels == value).sum() for value in class_ids])

        prototypes = {}
        event_masks = {}
        peak_masks = {}
        valley_masks = {}
        for mode in modes:
            analyzer = BatchedDirectFourierAnalyzer(mode, period_days, reg).to(
                spatial_features.device
            )
            synthesizer = BatchedDirectFourierSynthesizer(mode, period_days).to(
                spatial_features.device
            )
            coefficients, _ = analyzer(spatial_features, positions)
            reconstruction = synthesizer(coefficients, dense_positions)
            class_prototypes = []
            class_events = []
            class_peaks = []
            class_valleys = []
            for class_id in class_ids:
                class_index = int(class_id.item())
                selected = reconstruction[labels == class_id]
                curves = torch.einsum(
                    "nld,d->nl", selected, projections[class_index]
                )
                normalized = _robust_normalize(curves, detach_statistics=False)
                prototype = torch.quantile(normalized, 0.5, dim=0)
                class_prototypes.append(prototype)

                curve_np = prototype.detach().cpu().numpy()
                grid_np = grid.detach().cpu().numpy()
                threshold = prominence_rel * robust_signal_scale(curve_np)
                landmarks = detect_structural_landmarks(
                    grid_np,
                    curve_np,
                    min_distance_days=min_distance_days,
                    prominence_threshold=threshold,
                )
                event_mask = torch.zeros(grid_points, dtype=torch.bool, device=grid.device)
                peak_mask = torch.zeros_like(event_mask)
                valley_mask = torch.zeros_like(event_mask)
                for landmark in landmarks:
                    index = int(np.argmin(np.abs(grid_np - landmark.time)))
                    event_mask[index] = True
                    if landmark.kind == "peak":
                        peak_mask[index] = True
                    else:
                        valley_mask[index] = True
                class_events.append(event_mask)
                class_peaks.append(peak_mask)
                class_valleys.append(valley_mask)
            prototypes[mode] = torch.stack(class_prototypes)
            event_masks[mode] = torch.stack(class_events)
            peak_masks[mode] = torch.stack(class_peaks)
            valley_masks[mode] = torch.stack(class_valleys)

        return cls(
            projections,
            grid,
            prototypes,
            event_masks,
            peak_masks,
            valley_masks,
            sample_counts,
            period_days=period_days,
            reg=reg,
            prominence_rel=prominence_rel,
            min_distance_days=min_distance_days,
        )

    def export_payload(self):
        return {
            "modes": self.modes,
            "period_days": self.period_days,
            "reg": self.reg,
            "prominence_rel": self.prominence_rel,
            "min_distance_days": self.min_distance_days,
            "class_projections": self.class_projections.detach().cpu(),
            "grid": self.grid.detach().cpu(),
            "sample_counts": self.sample_counts.detach().cpu(),
            "prototypes": {k: v.detach().cpu() for k, v in self.prototypes.items()},
            "event_masks": {
                mode: getattr(self, f"event_mask_{mode}").detach().cpu()
                for mode in self.modes
            },
            "peak_masks": {
                mode: getattr(self, f"peak_mask_{mode}").detach().cpu()
                for mode in self.modes
            },
            "valley_masks": {
                mode: getattr(self, f"valley_mask_{mode}").detach().cpu()
                for mode in self.modes
            },
        }

    @classmethod
    def from_payload(cls, payload):
        return cls(
            payload["class_projections"],
            payload["grid"],
            payload["prototypes"],
            payload["event_masks"],
            payload["peak_masks"],
            payload["valley_masks"],
            payload["sample_counts"],
            period_days=payload["period_days"],
            reg=payload["reg"],
            prominence_rel=payload["prominence_rel"],
            min_distance_days=payload["min_distance_days"],
        )

    def manifest(self, source, checkpoint_path, classes, reference_per_class):
        per_class = {}
        for class_index, class_name in enumerate(classes):
            mode_events = {}
            for mode in self.modes:
                peaks = getattr(self, f"peak_mask_{mode}")[class_index]
                valleys = getattr(self, f"valley_mask_{mode}")[class_index]
                mask = peaks | valleys
                mode_events[str(mode)] = {
                    "peak_count": int(peaks.sum().item()),
                    "valley_count": int(valleys.sum().item()),
                    "event_times": self.grid[mask].detach().cpu().tolist(),
                }
            per_class[str(class_name)] = {
                "sample_count": int(self.sample_counts[class_index].item()),
                "events": mode_events,
            }
        return {
            "source": source,
            "checkpoint_path": checkpoint_path,
            "classes": list(classes),
            "reference_per_class": int(reference_per_class),
            "projection_dimension": int(self.class_projections.shape[1]),
            "modes": list(self.modes),
            "period": self.period_days,
            "regularization": self.reg,
            "support_start": float(self.grid[0].item()),
            "support_end": float(self.grid[-1].item()),
            "grid_points": int(self.grid.numel()),
            "prominence_rel": self.prominence_rel,
            "min_distance_days": self.min_distance_days,
            "per_class": per_class,
        }


@dataclass
class ShapeAlignmentResult:
    loss: torch.Tensor
    morph_loss: torch.Tensor
    event_loss: torch.Tensor
    mode_losses: Dict[int, torch.Tensor]
    morph_corr_mean: torch.Tensor
    event_amp_abs_gap: torch.Tensor
    no_event_reference_count: torch.Tensor


class ShapeAlignment(nn.Module):
    """Compare accepted target PSE shapes to frozen pseudo-class references."""

    def __init__(
        self,
        reference: SourceShapeReferenceBank,
        morph_weight: float = 1.0,
        event_weight: float = 0.5,
    ):
        super().__init__()
        self.reference = reference
        self.morph_weight = float(morph_weight)
        self.event_weight = float(event_weight)
        self.analyzers = nn.ModuleDict(
            {
                str(mode): BatchedDirectFourierAnalyzer(
                    mode, reference.period_days, reference.reg
                )
                for mode in reference.modes
            }
        )
        self.synthesizers = nn.ModuleDict(
            {
                str(mode): BatchedDirectFourierSynthesizer(
                    mode, reference.period_days
                )
                for mode in reference.modes
            }
        )
        self.reference.requires_grad_(False)

    def _zero_result(self, spatial_features):
        zero = spatial_features.sum() * 0.0
        return ShapeAlignmentResult(
            loss=zero,
            morph_loss=zero,
            event_loss=zero,
            mode_losses={mode: zero for mode in self.reference.modes},
            morph_corr_mean=zero.detach(),
            event_amp_abs_gap=zero.detach(),
            no_event_reference_count=torch.zeros(
                (), device=spatial_features.device, dtype=torch.long
            ),
        )

    def forward(self, spatial_features, positions, pseudo_classes):
        if spatial_features.shape[0] == 0:
            return self._zero_result(spatial_features)
        if positions.shape != spatial_features.shape[:2]:
            raise ValueError("target positions must match [B,L]")
        if pseudo_classes.shape != (spatial_features.shape[0],):
            raise ValueError("pseudo classes must have shape [B]")
        if pseudo_classes.min() < 0 or pseudo_classes.max() >= self.reference.class_projections.shape[0]:
            raise ValueError("pseudo class is missing from the source reference")

        grid = self.reference.grid.to(
            device=spatial_features.device, dtype=spatial_features.dtype
        ).unsqueeze(0).expand(spatial_features.shape[0], -1)
        projections = self.reference.class_projections.index_select(0, pseudo_classes)
        mode_losses = {}
        morph_losses = []
        event_losses = []
        correlations = []
        event_gaps = []
        no_event_counts = []
        for mode in self.reference.modes:
            coefficients, _ = self.analyzers[str(mode)](spatial_features, positions)
            reconstruction = self.synthesizers[str(mode)](coefficients, grid)
            curves = torch.einsum("bld,bd->bl", reconstruction, projections)
            normalized = _robust_normalize(curves, detach_statistics=True)
            prototype = self.reference.prototypes[mode].index_select(0, pseudo_classes)
            correlation = _stable_correlation(normalized, prototype)
            morph_loss = (1.0 - correlation).mean()

            event_mask = getattr(self.reference, f"event_mask_{mode}").index_select(
                0, pseudo_classes
            )
            pointwise_gap = torch.abs(normalized - prototype)
            event_count = event_mask.sum(dim=1)
            per_sample_event = (
                F.smooth_l1_loss(normalized, prototype, reduction="none")
                * event_mask
            ).sum(dim=1) / event_count.clamp_min(1)
            per_sample_gap = (pointwise_gap * event_mask).sum(dim=1) / event_count.clamp_min(1)
            event_loss = per_sample_event.mean()
            event_gap = per_sample_gap.mean()
            mode_loss = self.morph_weight * morph_loss + self.event_weight * event_loss
            mode_losses[mode] = mode_loss
            morph_losses.append(morph_loss)
            event_losses.append(event_loss)
            correlations.append(correlation.mean())
            event_gaps.append(event_gap)
            no_event_counts.append((event_count == 0).sum())

        return ShapeAlignmentResult(
            loss=torch.stack(list(mode_losses.values())).mean(),
            morph_loss=torch.stack(morph_losses).mean(),
            event_loss=torch.stack(event_losses).mean(),
            mode_losses=mode_losses,
            morph_corr_mean=torch.stack(correlations).mean().detach(),
            event_amp_abs_gap=torch.stack(event_gaps).mean().detach(),
            no_event_reference_count=torch.stack(no_event_counts).max().detach(),
        )

    @torch.no_grad()
    def diagnostics(self, spatial_features, positions, pseudo_classes):
        result = self(spatial_features, positions, pseudo_classes)
        metrics = {
            "morphology_correlation": result.morph_corr_mean,
            "event_amplitude_gap": result.event_amp_abs_gap,
        }
        peak_matches = []
        valley_matches = []
        signature_matches = []
        if spatial_features.shape[0] == 0:
            zero = result.loss.detach()
            metrics.update(
                peak_count_match_rate=zero,
                valley_count_match_rate=zero,
                landmark_signature_match_rate=zero,
            )
            return metrics

        grid_np = self.reference.grid.detach().cpu().numpy()
        for mode in self.reference.modes:
            grid = self.reference.grid.to(spatial_features).unsqueeze(0).expand(
                spatial_features.shape[0], -1
            )
            coeffs, _ = self.analyzers[str(mode)](spatial_features, positions)
            reconstructed = self.synthesizers[str(mode)](coeffs, grid)
            projections = self.reference.class_projections.index_select(0, pseudo_classes)
            curves = _robust_normalize(
                torch.einsum("bld,bd->bl", reconstructed, projections), True
            )
            for index, class_id in enumerate(pseudo_classes.detach().cpu().tolist()):
                curve = curves[index].detach().cpu().numpy()
                threshold = self.reference.prominence_rel * robust_signal_scale(curve)
                landmarks = detect_structural_landmarks(
                    grid_np,
                    curve,
                    self.reference.min_distance_days,
                    threshold,
                )
                signature = topology_signature(landmarks)
                target_peaks = sum(item.kind == "peak" for item in landmarks)
                target_valleys = sum(item.kind == "valley" for item in landmarks)
                source_peak = int(getattr(self.reference, f"peak_mask_{mode}")[class_id].sum())
                source_valley = int(getattr(self.reference, f"valley_mask_{mode}")[class_id].sum())
                peak_matches.append(float(target_peaks == source_peak))
                valley_matches.append(float(target_valleys == source_valley))
                source_kinds = []
                for point in range(self.reference.grid.numel()):
                    if getattr(self.reference, f"peak_mask_{mode}")[class_id, point]:
                        source_kinds.append("P")
                    elif getattr(self.reference, f"valley_mask_{mode}")[class_id, point]:
                        source_kinds.append("V")
                signature_matches.append(float(signature == (len(source_kinds), "-".join(source_kinds))))
        device = spatial_features.device
        dtype = spatial_features.dtype
        metrics.update(
            peak_count_match_rate=torch.tensor(peak_matches, device=device, dtype=dtype).mean(),
            valley_count_match_rate=torch.tensor(valley_matches, device=device, dtype=dtype).mean(),
            landmark_signature_match_rate=torch.tensor(signature_matches, device=device, dtype=dtype).mean(),
        )
        return {name: value.detach() for name, value in metrics.items()}
