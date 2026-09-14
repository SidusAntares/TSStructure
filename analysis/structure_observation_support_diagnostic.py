"""Source-only observation-support diagnostics for fixed configuration-05 structures."""

from __future__ import annotations

import math
import csv
import json
import shutil
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analysis.recon_event_diagnostic import circular_day_distance
from analysis.recon_structure_segments import match_coarse_segments_one_to_one


FAILURE_REASONS = (
    "NO_COARSE_SEGMENT",
    "NO_SAME_DIRECTION_SEGMENT",
    "CENTER_DISTANCE_FAIL",
    "DURATION_RATIO_FAIL",
    "ASSIGNMENT_CONFLICT",
)


@dataclass(frozen=True)
class ObservationSupport:
    num_observations: int
    observation_density: float
    max_observation_gap_days: float
    nearest_start_observation_days: float
    nearest_end_observation_days: float
    quarter_coverage: float
    support_coverage_radius: float
    crosses_year_boundary: bool
    window_start_unwrapped: float
    window_end_unwrapped: float


@dataclass(frozen=True)
class MatchAudit:
    matched: bool
    matched_sample_segment_id: str
    failure_reason: str
    detector_state: str
    nearest_same_direction_center_distance: float
    nearest_same_direction_duration_ratio: float


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_median(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _window(start_day: float, end_day: float, period_days: float = 365.0):
    start = float(start_day)
    end = float(end_day)
    crosses = end <= start
    return start, end + period_days if crosses else end, crosses


def _unwrapped_positions(positions, start, crosses, period_days=365.0):
    values = np.asarray(positions, dtype=np.float64).reshape(-1)
    if crosses:
        values = np.where(values < start, values + float(period_days), values)
    return values


def observation_support_metrics(
    positions,
    start_day: float,
    end_day: float,
    radius_days: float = 15.0,
    quarter_bins: int = 4,
    valid_time_mask=None,
    period_days: float = 365.0,
) -> ObservationSupport:
    """Measure real acquisition support inside one circular structure window."""
    raw = np.asarray(positions, dtype=np.float64).reshape(-1)
    if valid_time_mask is not None:
        valid = np.asarray(valid_time_mask, dtype=bool).reshape(-1)
        if valid.shape != raw.shape:
            raise ValueError("valid_time_mask must match positions")
        raw = raw[valid]
    raw = raw[np.isfinite(raw)]
    start, end, crosses = _window(start_day, end_day, period_days)
    duration = end - start
    if duration <= 0:
        raise ValueError("structure window duration must be positive")
    unwrapped = _unwrapped_positions(raw, start, crosses, period_days)
    inside = np.sort(unwrapped[(unwrapped >= start) & (unwrapped <= end)])
    if inside.size:
        gap_points = np.concatenate(([start], inside, [end]))
        maximum_gap = float(np.max(np.diff(gap_points)))
    else:
        maximum_gap = float(duration)

    if raw.size:
        copies = np.concatenate((raw - period_days, raw, raw + period_days))
        nearest_start = float(np.min(np.abs(copies - start)))
        nearest_end = float(np.min(np.abs(copies - end)))
    else:
        nearest_start = nearest_end = float("nan")

    edges = np.linspace(start, end, int(quarter_bins) + 1)
    occupied = np.histogram(inside, bins=edges)[0] > 0 if inside.size else np.zeros(quarter_bins, bool)
    quarter_coverage = float(np.mean(occupied))

    first_day = int(math.ceil(start))
    last_day = int(math.floor(end))
    days = np.arange(first_day, last_day + 1, dtype=np.float64)
    if days.size and inside.size:
        supported = np.min(np.abs(days[:, None] - inside[None, :]), axis=1) <= float(radius_days)
        radius_coverage = float(np.mean(supported))
    else:
        radius_coverage = 0.0
    return ObservationSupport(
        num_observations=int(inside.size),
        observation_density=float(inside.size / duration),
        max_observation_gap_days=maximum_gap,
        nearest_start_observation_days=nearest_start,
        nearest_end_observation_days=nearest_end,
        quarter_coverage=quarter_coverage,
        support_coverage_radius=radius_coverage,
        crosses_year_boundary=crosses,
        window_start_unwrapped=start,
        window_end_unwrapped=end,
    )


def detector_state(member_extrema_count: int, coarse_segments: Sequence) -> str:
    if int(member_extrema_count) == 0:
        return "NO_MEMBER_EXTREMA"
    if int(member_extrema_count) < 2:
        return "INSUFFICIENT_MEMBER_EXTREMA"
    if not coarse_segments:
        return "COARSE_SIMPLIFICATION_NO_SURVIVING_SEGMENT"
    if any(getattr(item, "accepted", False) for item in coarse_segments):
        return "COARSE_SEGMENT_EXISTS"
    return "COARSE_GATE_REJECTED"


def audit_sample_matches(
    references: Sequence,
    sample_segments: Sequence,
    center_radius_days: float,
    max_duration_ratio: float,
    state: str,
) -> dict[str, MatchAudit]:
    """Match all references once, then explain every unmatched reference."""
    matches = match_coarse_segments_one_to_one(
        references, sample_segments, center_radius_days, max_duration_ratio
    )
    matched = {prototype.coarse_segment_id: candidate for prototype, candidate in matches}
    accepted = [item for item in sample_segments if getattr(item, "accepted", True)]
    result = {}
    for reference in references:
        reference_id = str(reference.coarse_segment_id)
        if reference_id in matched:
            candidate = matched[reference_id]
            result[reference_id] = MatchAudit(
                True, str(candidate.coarse_segment_id), "", state,
                float(circular_day_distance(reference.center_day, candidate.center_day)),
                float(candidate.duration_days / max(reference.duration_days, 1e-12)),
            )
            continue
        same_direction = [item for item in accepted if item.direction == reference.direction]
        if not accepted:
            reason = "NO_COARSE_SEGMENT"
            nearest = None
        elif not same_direction:
            reason = "NO_SAME_DIRECTION_SEGMENT"
            nearest = None
        else:
            ranked = sorted(
                same_direction,
                key=lambda item: (
                    circular_day_distance(reference.center_day, item.center_day),
                    abs(1.0 - item.duration_days / max(reference.duration_days, 1e-12)),
                    str(item.coarse_segment_id),
                ),
            )
            nearest = ranked[0]
            in_center = [
                item for item in same_direction
                if circular_day_distance(reference.center_day, item.center_day)
                <= float(center_radius_days)
            ]
            if not in_center:
                reason = "CENTER_DISTANCE_FAIL"
            else:
                valid_duration = [
                    item for item in in_center
                    if 1.0 / float(max_duration_ratio)
                    <= item.duration_days / max(reference.duration_days, 1e-12)
                    <= float(max_duration_ratio)
                ]
                reason = "ASSIGNMENT_CONFLICT" if valid_duration else "DURATION_RATIO_FAIL"
                if not valid_duration:
                    nearest = sorted(
                        in_center,
                        key=lambda item: (
                            circular_day_distance(reference.center_day, item.center_day),
                            abs(1.0 - item.duration_days / max(reference.duration_days, 1e-12)),
                        ),
                    )[0]
        result[reference_id] = MatchAudit(
            False,
            "",
            reason,
            state,
            (
                float(circular_day_distance(reference.center_day, nearest.center_day))
                if nearest is not None else float("nan")
            ),
            (
                float(nearest.duration_days / max(reference.duration_days, 1e-12))
                if nearest is not None else float("nan")
            ),
        )
    return result


def join_reliable_references(
    configuration05_rows: Sequence[Mapping],
    validity06a_rows: Sequence[Mapping],
    min_bootstrap_occurrence: float = 0.8,
) -> list[dict]:
    def key05(row):
        return str(row["source_domain"]), int(row["class_id"]), str(row["coarse_segment_id"])

    def key06(row):
        return str(row["source_domain"]), int(row["class_id"]), str(row["reference_structure_id"])

    five, six = {}, {}
    for row in configuration05_rows:
        key = key05(row)
        if key in five:
            raise ValueError(f"duplicate configuration-05 reference: {key}")
        five[key] = dict(row)
    for row in validity06a_rows:
        key = key06(row)
        if key in six:
            raise ValueError(f"duplicate 06A reference: {key}")
        six[key] = dict(row)
    if set(five) != set(six):
        missing = sorted(set(five).symmetric_difference(six))
        raise ValueError(f"missing exact 05/06A reference join: {missing}")
    joined = []
    for key in sorted(five):
        occurrence = float(six[key]["bootstrap_occurrence_rate"])
        if occurrence < float(min_bootstrap_occurrence):
            continue
        row = dict(five[key])
        row.update({
            "bootstrap_occurrence_rate": occurrence,
            "individual_occurrence_rate": float(six[key]["individual_occurrence_rate"]),
            "reference_confidence": six[key].get("reference_confidence", ""),
        })
        joined.append(row)
    return joined


def cliffs_delta(left, right) -> float:
    left = np.asarray(list(left), dtype=np.float64)
    right = np.asarray(list(right), dtype=np.float64)
    left, right = left[np.isfinite(left)], right[np.isfinite(right)]
    if not left.size or not right.size:
        return float("nan")
    comparison = left[:, None] - right[None, :]
    return float((np.count_nonzero(comparison > 0) - np.count_nonzero(comparison < 0)) / comparison.size)


def summarize_structures(rows: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["source_domain"], int(row["class_id"]), row["reference_structure_id"])].append(row)
    output = []
    for key in sorted(grouped):
        group = grouped[key]
        plus = [row for row in group if _as_bool(row["matched"])]
        minus = [row for row in group if not _as_bool(row["matched"])]
        first = group[0]
        item = {
            "task": key[0], "source_domain": key[1], "class_id": key[2],
            "class_name": first["class_name"], "reference_structure_id": key[3],
            "direction": first["direction"], "start_day": first["reference_start_day"],
            "end_day": first["reference_end_day"],
            "duration_days": first["reference_duration_days"],
            "crosses_year_boundary": _as_bool(first["crosses_year_boundary"]),
            "bootstrap_occurrence_rate": float(first["bootstrap_occurrence_rate"]),
            "individual_occurrence_rate": float(first["individual_occurrence_rate"]),
            "num_samples": len(group), "num_matched": len(plus), "num_unmatched": len(minus),
            "matched_rate": len(plus) / len(group) if group else float("nan"),
        }
        metrics = (
            ("num_obs", "num_observations"),
            ("max_gap", "max_observation_gap_days"),
            ("support_coverage", "support_coverage_radius"),
            ("nearest_start", "nearest_start_observation_days"),
            ("nearest_end", "nearest_end_observation_days"),
        )
        for label, field in metrics:
            plus_values = [float(row[field]) for row in plus]
            minus_values = [float(row[field]) for row in minus]
            plus_median = _finite_median(plus_values)
            minus_median = _finite_median(minus_values)
            item[f"Gplus_median_{label}"] = plus_median
            item[f"Gminus_median_{label}"] = minus_median
            item[f"delta_{label}"] = minus_median - plus_median
            if field in {"num_observations", "max_observation_gap_days", "support_coverage_radius"}:
                item[f"cliffs_delta_{label}"] = cliffs_delta(minus_values, plus_values)
        reasons = Counter(row["failure_reason"] for row in minus)
        item["dominant_failure_reason"] = (
            sorted(reasons.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
            if reasons else ""
        )
        output.append(item)
    return output


def summarize_failure_reasons(rows: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], int(row["class_id"]), row["class_name"], row["reference_structure_id"])].append(row)

    def make(scope, key, group):
        unmatched = [row for row in group if not _as_bool(row["matched"])]
        counts = Counter(row["failure_reason"] for row in unmatched)
        item = {
            "scope": scope, "task": key[0],
            "class_id": key[1], "class_name": key[2],
            "reference_structure_id": key[3], "num_unmatched": len(unmatched),
        }
        for reason in FAILURE_REASONS:
            item[reason] = counts[reason]
            item[f"fraction_{reason}"] = counts[reason] / len(unmatched) if unmatched else 0.0
        return item

    output = [make("STRUCTURE", key, group) for key, group in sorted(grouped.items())]
    if rows:
        output.append(make("TOTAL", (rows[0]["task"], "", "", ""), list(rows)))
    return output


def summarize_boundaries(rows: Sequence[Mapping]) -> list[dict]:
    output = []
    for boundary in (True, False):
        group = [row for row in rows if _as_bool(row["crosses_year_boundary"]) == boundary]
        if not group:
            continue
        failures = Counter(row["failure_reason"] for row in group if not _as_bool(row["matched"]))
        structure_groups = defaultdict(list)
        for row in group:
            structure_groups[(row["class_id"], row["reference_structure_id"])].append(row)
        structure_rates = [
            float(np.mean([_as_bool(row["matched"]) for row in values]))
            for values in structure_groups.values()
        ]
        item = {
            "cross_boundary": boundary,
            "num_structures": len({(row["class_id"], row["reference_structure_id"]) for row in group}),
            "num_samples": len(group),
            "matched_rate": float(np.mean([_as_bool(row["matched"]) for row in group])),
            "mean_structure_matched_rate": float(np.mean(structure_rates)),
            "median_structure_matched_rate": float(np.median(structure_rates)),
            "median_num_obs": _finite_median(row["num_observations"] for row in group),
            "median_max_gap": _finite_median(row["max_observation_gap_days"] for row in group),
            "median_support_coverage": _finite_median(row["support_coverage_radius"] for row in group),
        }
        for reason in FAILURE_REASONS:
            item[reason] = failures[reason]
        output.append(item)
    return output


def _window_indices(positions, start_day, end_day, period_days=365.0):
    start, end, crosses = _window(start_day, end_day, period_days)
    values = _unwrapped_positions(positions, start, crosses, period_days)
    return np.flatnonzero((values >= start) & (values <= end)), values, start, end


def random_window_mask(positions, start_day, end_day, fraction, seed, period_days=365.0):
    indices, _, _, _ = _window_indices(positions, start_day, end_day, period_days)
    mask = np.zeros(len(positions), dtype=bool)
    if not len(indices):
        return mask
    count = int(math.ceil(len(indices) * float(fraction)))
    rng = np.random.default_rng(int(seed))
    mask[rng.choice(indices, size=count, replace=False)] = True
    return mask


def contiguous_gap_mask(positions, start_day, end_day, gap_days, seed, period_days=365.0):
    indices, values, start, end = _window_indices(positions, start_day, end_day, period_days)
    gap = float(gap_days)
    if gap > end - start:
        return None
    rng = np.random.default_rng(int(seed))
    gap_start = float(rng.uniform(start, end - gap)) if gap < end - start else start
    mask = np.zeros(len(positions), dtype=bool)
    mask[indices[(values[indices] >= gap_start) & (values[indices] <= gap_start + gap)]] = True
    return mask


def delete_acquisitions(sample: Mapping, removal_mask) -> dict:
    removal = np.asarray(removal_mask, dtype=bool).reshape(-1)
    keep = ~removal
    result = dict(sample)
    for key in ("pixels", "valid_pixels", "positions"):
        value = sample[key]
        if len(value) != len(removal):
            raise ValueError(f"{key} time dimension does not match removal mask")
        if hasattr(value, "__class__") and value.__class__.__module__.startswith("torch"):
            import torch
            result[key] = value[torch.as_tensor(keep, device=value.device)]
        else:
            result[key] = np.asarray(value)[keep]
    return result


def summarize_masking(rows: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["source_domain"], int(row["class_id"]), row["class_name"], row["reference_structure_id"], row["mask_type"], float(row["mask_level"]), _as_bool(row["crosses_year_boundary"]))].append(row)

    def make(scope, key, group):
        valid = [row for row in group if _as_bool(row["valid_mask_run"])]
        recovered = sum(_as_bool(row["masked_matched"]) for row in valid)
        failures = Counter(row["failure_reason"] for row in valid if not _as_bool(row["masked_matched"]))
        return {
            "scope": scope, "task": key[0], "source_domain": key[1],
            "class_id": key[2], "class_name": key[3],
            "reference_structure_id": key[4], "mask_type": key[5], "mask_level": key[6],
            "crosses_year_boundary": key[7],
            "num_samples": len({row["sample_id"] for row in group}),
            "num_runs": len(group), "num_valid_runs": len(valid),
            "recovery_rate": recovered / len(valid) if valid else float("nan"),
            "loss_rate": 1.0 - recovered / len(valid) if valid else float("nan"),
            "median_num_obs_after": _finite_median(row["masked_num_obs"] for row in valid),
            "median_max_gap_after": _finite_median(row["masked_max_gap"] for row in valid),
            "median_support_coverage_after": _finite_median(row["masked_support_coverage"] for row in valid),
            "dominant_failure_reason": (
                sorted(failures.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
                if failures else ""
            ),
        }

    output = [make("STRUCTURE", key, group) for key, group in sorted(grouped.items())]
    total_groups = defaultdict(list)
    for key, group in grouped.items():
        total_groups[(key[0], key[1], "", "", "", key[5], key[6], key[7])].extend(group)
    output.extend(make("TOTAL", key, group) for key, group in sorted(total_groups.items()))
    return output


def select_masking_references(
    references: Sequence[Mapping], max_high_support_controls: int = 2
) -> list[dict]:
    """Select all low-support stable references plus deterministic high-support controls."""
    low = [
        dict(row) for row in references
        if float(row["bootstrap_occurrence_rate"]) >= 0.8
        and float(row["individual_occurrence_rate"]) < 0.5
    ]
    high = [
        dict(row) for row in references
        if float(row["bootstrap_occurrence_rate"]) >= 0.8
        and float(row["individual_occurrence_rate"]) >= 0.9
    ]
    target_duration = _finite_median(float(row["duration_days"]) for row in low)
    chosen = []
    for boundary in (True, False):
        candidates = [row for row in high if _as_bool(row["crosses_year_boundary"]) == boundary]
        if not candidates:
            continue
        candidates.sort(key=lambda row: (
            abs(float(row["duration_days"]) - target_duration)
            if np.isfinite(target_duration) else float(row["duration_days"]),
            int(row["class_id"]), str(row["coarse_segment_id"]),
        ))
        chosen.append(candidates[0])
        if len(chosen) >= int(max_high_support_controls):
            break
    if len(chosen) < int(max_high_support_controls):
        remaining = [row for row in high if row not in chosen]
        remaining.sort(key=lambda row: (
            abs(float(row["duration_days"]) - target_duration)
            if np.isfinite(target_duration) else float(row["duration_days"]),
            int(row["class_id"]), str(row["coarse_segment_id"]),
        ))
        chosen.extend(remaining[: int(max_high_support_controls) - len(chosen)])
    output = []
    for row in low:
        row["masking_selection"] = "STABLE_BUT_LOW_SAMPLE_SUPPORT"
        output.append(row)
    for row in chosen:
        row["masking_selection"] = "HIGH_SUPPORT_CONTROL"
        output.append(row)
    return output


def write_csv(path, rows: Sequence[Mapping], fieldnames=None):
    rows = [dict(row) for row in rows]
    fields = list(fieldnames or (rows[0].keys() if rows else ()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_support_audit(
    output_path,
    grid,
    reference_curve,
    sample_items,
    structure_rows,
    reference,
    margin_days=30,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(grid, dtype=np.float64)
    rows = list(structure_rows)
    plus = [row for row in rows if _as_bool(row["matched"])]
    minus = [row for row in rows if not _as_bool(row["matched"])]
    worst = sorted(minus, key=lambda row: (-float(row["max_observation_gap_days"]), str(row["sample_id"])))[:1]
    best = sorted(minus, key=lambda row: (-float(row["support_coverage_radius"]), float(row["max_observation_gap_days"]), str(row["sample_id"])))[:1]
    selected = sorted(plus, key=lambda row: str(row["sample_id"]))[:2] + worst + [row for row in best if row not in worst]
    by_id = {str(item["sample_id"]): item for item in sample_items}

    start, end, crosses = _window(reference.start_day, reference.end_day)
    display_grid = grid.copy()
    if crosses:
        display_grid = np.where(display_grid < start, display_grid + 365.0, display_grid)
    order = np.argsort(display_grid)
    lo, hi = start - float(margin_days), end + float(margin_days)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(display_grid[order], np.asarray(reference_curve)[order], lw=2.5, color="#1F4E79", label="reference")
    colors = {True: "#2A9D8F", False: "#D95F02"}
    for row in selected:
        item = by_id.get(str(row["sample_id"]))
        if item is None:
            continue
        matched = _as_bool(row["matched"])
        curve = np.asarray(item["curve"])
        axes[0].plot(display_grid[order], curve[order], color=colors[matched], alpha=0.75, label=f"{'G+' if matched else 'G-'} {row['sample_id']}")
        positions = _unwrapped_positions(item["positions"], start, crosses)
        y = np.interp(positions, display_grid[order], curve[order])
        axes[0].scatter(positions, y, s=14, color=colors[matched], zorder=4)
    axes[0].axvspan(start, end, color="#777777", alpha=0.10)
    axes[0].set_xlim(lo, hi)
    axes[0].set_title("Reference and representative acquisitions")
    axes[0].legend(fontsize=7, frameon=False)

    for index, (label, field) in enumerate((
        ("num obs", "num_observations"),
        ("max gap", "max_observation_gap_days"),
        ("coverage", "support_coverage_radius"),
    )):
        for x, group in enumerate((plus, minus)):
            values = [float(row[field]) for row in group]
            if values:
                jitter = np.linspace(-0.08, 0.08, len(values))
                axes[1].scatter(np.full(len(values), index * 3 + x) + jitter, values, s=12, alpha=0.55)
                axes[1].plot([index * 3 + x - 0.2, index * 3 + x + 0.2], [np.median(values)] * 2, color="black")
    axes[1].set_xticks([0.5, 3.5, 6.5], ["num obs", "max gap", "coverage"])
    axes[1].set_title("G+ versus G- support")

    failures = Counter(row["failure_reason"] for row in minus)
    axes[2].barh(list(failures), list(failures.values()), color="#D95F02")
    axes[2].set_title("G- failure reasons")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_masking_diagnostic(output_path, masking_rows, example_curves=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in masking_rows if _as_bool(row["valid_mask_run"])]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for axis, mask_type, title in (
        (axes[0], "random", "Random deletion"),
        (axes[1], "gap", "Contiguous gap"),
    ):
        levels = sorted({float(row["mask_level"]) for row in rows if row["mask_type"] == mask_type})
        rates = []
        for level in levels:
            group = [row for row in rows if row["mask_type"] == mask_type and float(row["mask_level"]) == level]
            rates.append(np.mean([_as_bool(row["masked_matched"]) for row in group]))
        axis.plot(levels, rates, marker="o", color="#C44E52")
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(title)
        axis.set_xlabel("fraction" if mask_type == "random" else "gap days")
        axis.set_ylabel("recovery rate")
    for item in example_curves or ():
        axes[2].plot(item["grid"], item["curve"], label=item["label"], alpha=0.85)
    axes[2].set_title("Baseline and masked Recon13")
    if example_curves:
        axes[2].legend(fontsize=7, frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


@contextmanager
def staged_output(final_dir, required_files=()):
    """Publish a complete task directory while preserving the previous result on failure."""
    final = Path(final_dir)
    staging = final.parent / f".tmp_{final.name}"
    backup = final.parent / f".old_{final.name}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        yield staging
        missing = [name for name in required_files if not (staging / name).is_file()]
        if missing:
            raise RuntimeError(f"incomplete staged output: {missing}")
        if backup.exists():
            shutil.rmtree(backup)
        if final.exists():
            final.rename(backup)
        staging.rename(final)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not final.exists():
            backup.rename(final)
        raise
