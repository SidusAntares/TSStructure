"""Numerical utilities for the offline FreDN structural decomposition probe.

The functions in this module never train a model and never consume target data
when fitting projections or thresholds.  Target labels are only used by the
calling audit script for oracle grouping and evaluation.
"""

from collections import Counter
from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch


Signature = Tuple[int, str]


@dataclass(frozen=True)
class Landmark:
    kind: str
    time: float
    amplitude: float
    prominence: float

    def __post_init__(self):
        if self.kind not in ("peak", "valley"):
            raise ValueError("landmark kind must be 'peak' or 'valley'")


class TopologyMismatchError(ValueError):
    """Raised when landmark correspondence is not topology-compatible."""


def parse_fredn_checkpoint_specs(
    specifications: Sequence[str],
    legacy9: str = None,
    legacy17: str = None,
) -> Dict[int, str]:
    """Parse repeatable ``MODE=PATH`` values with legacy 9/17 compatibility."""
    parsed: Dict[int, str] = {}

    def add(mode, path):
        if path is None:
            return
        if mode <= 0 or mode % 2 == 0:
            raise ValueError("FreDN mode count must be a positive odd integer")
        if mode in parsed and parsed[mode] != path:
            raise ValueError(f"conflicting checkpoint paths for mode {mode}")
        parsed[mode] = path

    for specification in specifications:
        if "=" not in specification:
            raise ValueError("checkpoint specification must be MODE=PATH")
        mode_text, path = specification.split("=", 1)
        if not path:
            raise ValueError("checkpoint path must not be empty")
        try:
            mode = int(mode_text)
        except ValueError as error:
            raise ValueError("checkpoint mode must be an integer") from error
        add(mode, path)
    add(9, legacy9)
    add(17, legacy17)
    if not parsed:
        raise ValueError("at least one FreDN checkpoint is required")
    return dict(sorted(parsed.items()))


def extract_fourier_conditions(
    model,
    spatial_features: torch.Tensor,
    positions: torch.Tensor,
    dense_positions: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Produce no-mask reconstruction and learned-mask trend from one analysis."""
    coefficients, _ = model.fourier_analyzer(spatial_features, positions)
    fourier_recon = model.fourier_synthesizer(coefficients, dense_positions)
    trend_coefficients, _, _ = model.frequency_disentangler(coefficients)
    fredn_trend = model.fourier_synthesizer(
        trend_coefficients, dense_positions
    )
    return {
        "fourier_recon": fourier_recon,
        "fredn_trend": fredn_trend,
    }


def fit_source_class_projections(
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """Fit one deterministic first PCA loading per class from source only.

    ``source_features`` is ``[N,L,D]``.  All sample/time observations of a
    class are centered and stacked before calling ``torch.linalg.svd``.
    """
    if source_features.ndim != 3:
        raise ValueError("source_features must be [N,L,D]")
    if source_labels.ndim != 1 or source_labels.shape[0] != source_features.shape[0]:
        raise ValueError("source_labels must be [N]")
    if source_features.is_complex():
        raise ValueError("source_features must be real-valued")

    projections: Dict[int, torch.Tensor] = {}
    for class_id_tensor in torch.unique(source_labels, sorted=True):
        class_id = int(class_id_tensor.detach().cpu())
        observations = source_features[source_labels == class_id_tensor].reshape(
            -1, source_features.shape[-1]
        )
        centered = observations - observations.mean(dim=0, keepdim=True)
        if not torch.isfinite(centered).all():
            raise ValueError("source features contain non-finite values")
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        loading = vh[0]
        dominant = torch.argmax(torch.abs(loading))
        sign = torch.where(
            loading[dominant] < 0,
            loading.new_tensor(-1.0),
            loading.new_tensor(1.0),
        )
        projections[class_id] = (loading * sign).detach()
    return projections


def project_curves(
    features: torch.Tensor,
    projections: Mapping[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Apply every source-fitted class loading without consuming labels."""
    if features.ndim != 3:
        raise ValueError("features must be [N,L,D]")
    if not projections:
        raise ValueError("at least one source projection is required")
    projected = {}
    for class_id, class_loading in projections.items():
        loading = class_loading.to(
            device=features.device, dtype=features.dtype
        )
        if loading.ndim != 1 or loading.shape[0] != features.shape[-1]:
            raise ValueError("each source projection must have shape [D]")
        projected[int(class_id)] = features @ loading
    return projected


def robust_signal_scale(curves: np.ndarray) -> float:
    """Median per-curve p95-p05 range used to set source-only prominence."""
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    ranges = np.nanpercentile(values, 95, axis=1) - np.nanpercentile(
        values, 5, axis=1
    )
    scale = float(np.nanmedian(ranges))
    return max(scale, np.finfo(np.float64).eps)


def _candidate_prominence(values: np.ndarray, index: int, peak: bool) -> float:
    work = values if peak else -values
    height = work[index]
    left = work[: index + 1]
    right = work[index:]
    return float(max(0.0, min(height - np.min(left), height - np.min(right))))


def _select_by_distance(
    candidates: Sequence[Tuple[int, float]],
    times: np.ndarray,
    min_distance_days: float,
) -> List[Tuple[int, float]]:
    selected: List[Tuple[int, float]] = []
    for candidate in sorted(candidates, key=lambda item: (-item[1], item[0])):
        index = candidate[0]
        if all(abs(times[index] - times[kept[0]]) >= min_distance_days for kept in selected):
            selected.append(candidate)
    return sorted(selected)


def detect_structural_landmarks(
    times: np.ndarray,
    curve: np.ndarray,
    min_distance_days: float = 14.0,
    prominence_threshold: float = 0.0,
) -> List[Landmark]:
    """Detect robust interior extrema using source-derived absolute prominence."""
    times = np.asarray(times, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    if times.ndim != 1 or curve.ndim != 1 or times.shape != curve.shape:
        raise ValueError("times and curve must be matching one-dimensional arrays")
    if times.size < 3 or not np.all(np.diff(times) > 0):
        raise ValueError("times must be strictly increasing with at least 3 points")
    if min_distance_days < 0 or prominence_threshold < 0:
        raise ValueError("landmark thresholds must be non-negative")

    peaks: List[Tuple[int, float]] = []
    valleys: List[Tuple[int, float]] = []
    for index in range(1, curve.size - 1):
        if curve[index] > curve[index - 1] and curve[index] >= curve[index + 1]:
            prominence = _candidate_prominence(curve, index, peak=True)
            if prominence >= prominence_threshold:
                peaks.append((index, prominence))
        if curve[index] < curve[index - 1] and curve[index] <= curve[index + 1]:
            prominence = _candidate_prominence(curve, index, peak=False)
            if prominence >= prominence_threshold:
                valleys.append((index, prominence))

    landmarks = [
        Landmark("peak", times[index], curve[index], prominence)
        for index, prominence in _select_by_distance(
            peaks, times, min_distance_days
        )
    ]
    landmarks.extend(
        Landmark("valley", times[index], curve[index], prominence)
        for index, prominence in _select_by_distance(
            valleys, times, min_distance_days
        )
    )
    return sorted(landmarks, key=lambda item: item.time)


def topology_signature(landmarks: Sequence[Landmark]) -> Signature:
    kinds = "-".join("P" if item.kind == "peak" else "V" for item in landmarks)
    return len(landmarks), kinds


def signature_histogram(signatures: Sequence[Signature]) -> Dict[Signature, float]:
    if not signatures:
        return {}
    counts = Counter(signatures)
    total = float(sum(counts.values()))
    return {signature: count / total for signature, count in counts.items()}


def modal_signature(signatures: Sequence[Signature]) -> Signature:
    if not signatures:
        return 0, ""
    counts = Counter(signatures)
    return sorted(counts, key=lambda item: (-counts[item], item))[0]


def signature_entropy(signatures: Sequence[Signature]) -> float:
    probabilities = signature_histogram(signatures).values()
    return float(-sum(value * math.log(value) for value in probabilities if value > 0))


def summarize_topology(landmark_sets: Sequence[Sequence[Landmark]]) -> Dict[str, object]:
    signatures = [topology_signature(items) for items in landmark_sets]
    peak_counts = np.asarray(
        [sum(item.kind == "peak" for item in items) for items in landmark_sets],
        dtype=np.float64,
    )
    valley_counts = np.asarray(
        [sum(item.kind == "valley" for item in items) for items in landmark_sets],
        dtype=np.float64,
    )
    landmark_counts = peak_counts + valley_counts
    mode = modal_signature(signatures)
    mode_rate = signatures.count(mode) / len(signatures) if signatures else float("nan")
    return {
        "num_samples": len(landmark_sets),
        "peak_count_mean": float(np.mean(peak_counts)) if len(peak_counts) else float("nan"),
        "peak_count_std": float(np.std(peak_counts)) if len(peak_counts) else float("nan"),
        "valley_count_mean": float(np.mean(valley_counts)) if len(valley_counts) else float("nan"),
        "valley_count_std": float(np.std(valley_counts)) if len(valley_counts) else float("nan"),
        "landmark_count_mean": float(np.mean(landmark_counts)) if len(landmark_counts) else float("nan"),
        "landmark_count_std": float(np.std(landmark_counts)) if len(landmark_counts) else float("nan"),
        "modal_signature": f"{mode[0]}:{mode[1]}",
        "modal_signature_rate": float(mode_rate),
        "signature_entropy": signature_entropy(signatures),
    }


def compute_topology_comparison(
    source_by_class: Mapping[int, Sequence[Signature]],
    target_by_class: Mapping[int, Sequence[Signature]],
) -> Dict[str, float]:
    """Compute histogram-based same-class match and different-class collision."""
    common = sorted(set(source_by_class) & set(target_by_class))
    same_matches = []
    modal_rates = []
    for class_id in common:
        source_hist = signature_histogram(source_by_class[class_id])
        target_hist = signature_histogram(target_by_class[class_id])
        same_matches.append(
            sum(source_hist.get(key, 0.0) * target_hist.get(key, 0.0) for key in source_hist)
        )
        source_mode = modal_signature(source_by_class[class_id])
        target_values = target_by_class[class_id]
        modal_rates.append(
            target_values.count(source_mode) / len(target_values) if target_values else 0.0
        )

    collisions = []
    for source_class in common:
        source_hist = signature_histogram(source_by_class[source_class])
        for target_class in common:
            if target_class == source_class:
                continue
            target_hist = signature_histogram(target_by_class[target_class])
            collisions.append(
                sum(source_hist.get(key, 0.0) * target_hist.get(key, 0.0) for key in source_hist)
            )

    same = float(np.mean(same_matches)) if same_matches else float("nan")
    collision = float(np.mean(collisions)) if collisions else float("nan")
    return {
        "same_class_topology_match": same,
        "target_matches_source_modal_signature_rate": (
            float(np.mean(modal_rates)) if modal_rates else float("nan")
        ),
        "different_class_topology_collision": collision,
        "topology_discrimination_margin": same - collision,
    }


def build_landmark_warp(
    sample_landmarks: Sequence[Landmark],
    prototype_landmarks: Sequence[Landmark],
    support_start: float,
    support_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return old-time and canonical-time knots for a strict linear warp."""
    if topology_signature(sample_landmarks) != topology_signature(prototype_landmarks):
        raise TopologyMismatchError("sample topology does not match source modal topology")
    sample_knots = np.asarray(
        [support_start, *[item.time for item in sample_landmarks], support_end],
        dtype=np.float64,
    )
    canonical_knots = np.asarray(
        [support_start, *[item.time for item in prototype_landmarks], support_end],
        dtype=np.float64,
    )
    if not np.all(np.diff(sample_knots) > 0) or not np.all(
        np.diff(canonical_knots) > 0
    ):
        raise ValueError("warp knots must be strictly monotonic and inside support")
    return sample_knots, canonical_knots


def align_curve_to_landmark_template(
    grid: np.ndarray,
    curve: np.ndarray,
    sample_landmarks: Sequence[Landmark],
    prototype_landmarks: Sequence[Landmark],
) -> np.ndarray:
    """Resample a curve after mapping sample landmarks to source template times."""
    grid = np.asarray(grid, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    if grid.ndim != 1 or curve.shape != grid.shape:
        raise ValueError("grid and curve must be matching one-dimensional arrays")
    sample_knots, canonical_knots = build_landmark_warp(
        sample_landmarks,
        prototype_landmarks,
        float(grid[0]),
        float(grid[-1]),
    )
    original_times = np.interp(grid, canonical_knots, sample_knots)
    return np.interp(original_times, grid, curve)


def build_landmark_prototype(
    landmark_sets: Sequence[Sequence[Landmark]],
) -> Tuple[Signature, List[Landmark], List[int]]:
    signatures = [topology_signature(items) for items in landmark_sets]
    mode = modal_signature(signatures)
    accepted = [index for index, signature in enumerate(signatures) if signature == mode]
    if not accepted or mode[0] == 0:
        return mode, [], accepted
    prototype = []
    for order in range(mode[0]):
        entries = [landmark_sets[index][order] for index in accepted]
        prototype.append(
            Landmark(
                kind=entries[0].kind,
                time=float(np.median([item.time for item in entries])),
                amplitude=float(np.median([item.amplitude for item in entries])),
                prominence=float(np.median([item.prominence for item in entries])),
            )
        )
    return mode, prototype, accepted


def pointwise_intra_class_variance(curves: np.ndarray) -> float:
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("curves must be [N,T]")
    return float(np.nanmean(np.nanvar(values, axis=0)))


def normalized_l2(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = np.linalg.norm(first)
    return float(np.linalg.norm(first - second) / max(denominator, np.finfo(float).eps))


def curve_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def median_absolute_deviation(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan")
    median = np.median(array)
    return float(np.median(np.abs(array - median)))


def segment_shape_descriptors(
    grid: np.ndarray,
    curve: np.ndarray,
    landmarks: Sequence[Landmark],
) -> List[Dict[str, float]]:
    """Describe each segment between adjacent extrema without changing phase."""
    grid = np.asarray(grid, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    if grid.ndim != 1 or curve.shape != grid.shape:
        raise ValueError("grid and curve must be matching one-dimensional arrays")
    descriptors = []
    for order, (left, right) in enumerate(zip(landmarks[:-1], landmarks[1:])):
        duration = right.time - left.time
        if duration <= 0:
            raise ValueError("landmarks must be strictly ordered")
        inside = (grid >= left.time) & (grid <= right.time)
        segment_times = grid[inside]
        segment_values = curve[inside]
        if segment_times.size < 2:
            continue
        descriptors.append(
            {
                "segment_order": order,
                "left_kind": left.kind,
                "right_kind": right.kind,
                "duration": float(duration),
                "slope": float(
                    (segment_values[-1] - segment_values[0])
                    / (segment_times[-1] - segment_times[0])
                ),
                "area": float(np.trapz(segment_values, segment_times)),
            }
        )
    return descriptors


def contrastive_feasibility(
    source_prototypes: Mapping[int, np.ndarray],
    target_curves: Mapping[int, Sequence[np.ndarray]],
) -> Dict[str, float]:
    margins = []
    positive_distances = []
    for class_id, curves in target_curves.items():
        if class_id not in source_prototypes:
            continue
        negatives = [key for key in source_prototypes if key != class_id]
        if not negatives:
            continue
        for curve in curves:
            positive = normalized_l2(source_prototypes[class_id], curve)
            negative = min(
                normalized_l2(source_prototypes[other], curve) for other in negatives
            )
            positive_distances.append(positive)
            margins.append(negative - positive)
    values = np.asarray(margins, dtype=np.float64)
    return {
        "num_samples": int(values.size),
        "mean_d_pos": float(np.mean(positive_distances)) if values.size else float("nan"),
        "mean_margin": float(np.mean(values)) if values.size else float("nan"),
        "median_margin": float(np.median(values)) if values.size else float("nan"),
        "positive_margin_rate": float(np.mean(values > 0)) if values.size else float("nan"),
    }


def contrastive_feasibility_by_class(
    source_prototypes: Mapping[int, np.ndarray],
    target_curves: Mapping[int, Sequence[np.ndarray]],
    target_total_by_class: Mapping[int, int],
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Return per-class and macro/micro oracle contrastive diagnostics."""
    rows = []
    all_margins = []
    all_positive_distances = []
    all_negative_distances = []
    for class_id in sorted(target_total_by_class):
        curves = list(target_curves.get(class_id, ()))
        negatives = [key for key in source_prototypes if key != class_id]
        positive_distances = []
        negative_distances = []
        margins = []
        if class_id in source_prototypes and negatives:
            for curve in curves:
                positive = normalized_l2(source_prototypes[class_id], curve)
                negative = min(
                    normalized_l2(source_prototypes[other], curve)
                    for other in negatives
                )
                positive_distances.append(positive)
                negative_distances.append(negative)
                margins.append(negative - positive)
        total = int(target_total_by_class[class_id])
        passed = len(margins)
        rows.append(
            {
                "class_id": class_id,
                "target_total_n": total,
                "gate_pass_n": passed,
                "gate_coverage": passed / total if total else float("nan"),
                "d_pos_mean": float(np.mean(positive_distances)) if passed else float("nan"),
                "d_pos_median": float(np.median(positive_distances)) if passed else float("nan"),
                "d_neg_mean": float(np.mean(negative_distances)) if passed else float("nan"),
                "d_neg_median": float(np.median(negative_distances)) if passed else float("nan"),
                "margin_mean": float(np.mean(margins)) if passed else float("nan"),
                "margin_median": float(np.median(margins)) if passed else float("nan"),
                "positive_margin_rate": (
                    float(np.mean(np.asarray(margins) > 0)) if passed else float("nan")
                ),
            }
        )
        all_positive_distances.extend(positive_distances)
        all_negative_distances.extend(negative_distances)
        all_margins.extend(margins)
    eligible = [row for row in rows if row["gate_pass_n"] > 0]
    total_passed = sum(row["gate_pass_n"] for row in eligible)
    aggregate = {
        "micro_positive_margin_rate": (
            float(np.mean(np.asarray(all_margins) > 0))
            if all_margins
            else float("nan")
        ),
        "macro_positive_margin_rate": (
            float(np.mean([row["positive_margin_rate"] for row in eligible]))
            if eligible
            else float("nan")
        ),
        "contrastive_margin_mean": (
            float(np.mean(all_margins)) if all_margins else float("nan")
        ),
        "contrastive_margin_median": (
            float(np.median(all_margins)) if all_margins else float("nan")
        ),
        "d_pos_mean": (
            float(np.mean(all_positive_distances))
            if all_positive_distances
            else float("nan")
        ),
        "d_neg_mean": (
            float(np.mean(all_negative_distances))
            if all_negative_distances
            else float("nan")
        ),
        "eligible_class_count": len(eligible),
        "dominant_class_fraction": (
            max(row["gate_pass_n"] for row in eligible) / total_passed
            if total_passed
            else float("nan")
        ),
    }
    return aggregate, rows


def pareto_modes(
    rows: Sequence[Mapping[str, float]],
    x_key: str,
    y_key: str,
) -> List[Dict[str, float]]:
    """Return modes not strictly dominated on two higher-is-better axes."""
    finite_rows = [
        candidate
        for candidate in rows
        if np.isfinite(candidate[x_key]) and np.isfinite(candidate[y_key])
    ]
    result = []
    for candidate in finite_rows:
        dominated = any(
            other[x_key] >= candidate[x_key]
            and other[y_key] >= candidate[y_key]
            and (
                other[x_key] > candidate[x_key]
                or other[y_key] > candidate[y_key]
            )
            for other in finite_rows
            if other is not candidate
        )
        if not dominated:
            result.append(dict(candidate))
    return sorted(result, key=lambda row: row["mode"])
