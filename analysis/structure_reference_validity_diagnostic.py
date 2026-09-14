"""Source-only validity diagnostics for fixed-PC1 Recon13 structure references."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analysis.recon_event_diagnostic import circular_day_distance, circular_mad_days


STRUCTURE_STABILITY_FIELDS = (
    "source_domain", "class_id", "class_name", "reference_structure_id",
    "direction", "start_day", "end_day", "center_day", "duration_days",
    "individual_occurrence_rate", "bootstrap_occurrence_rate",
    "bootstrap_stability",
    "bootstrap_center_mad_days", "bootstrap_duration_median",
    "bootstrap_duration_iqr", "bootstrap_change_median",
    "bootstrap_change_iqr", "bootstrap_monotonicity_median",
    "bootstrap_monotonicity_iqr", "present_in_medoid",
    "medoid_center_distance_days", "medoid_duration_ratio",
    "reference_confidence", "num_bootstrap_runs", "num_bootstrap_matches",
)


@dataclass(frozen=True)
class StructureMatch:
    reference: object
    candidate: object
    center_distance_days: float
    duration_ratio: float


@dataclass(frozen=True)
class MedoidSelection:
    sample_index: int
    curve: np.ndarray
    candidate_indices: np.ndarray
    total_distance: float


def bootstrap_subsample_indices(
    sample_count: int,
    repeats: int = 100,
    fraction: float = 0.70,
    seed: int = 1,
) -> np.ndarray:
    """Draw deterministic fixed-size subsets without replacement."""
    if sample_count <= 0 or repeats <= 0:
        raise ValueError("sample_count and repeats must be positive")
    if not 0 < float(fraction) <= 1:
        raise ValueError("fraction must be in (0, 1]")
    subset_size = max(1, int(np.floor(sample_count * float(fraction))))
    rng = np.random.default_rng(int(seed))
    return np.stack([
        np.sort(rng.choice(sample_count, size=subset_size, replace=False))
        for _ in range(int(repeats))
    ])


def match_structure_sets(
    references: Sequence,
    candidates: Sequence,
    center_radius_days: float = 30,
    max_duration_ratio: float = 2.0,
) -> tuple[StructureMatch, ...]:
    """Deterministic one-to-one matching inside one source class."""
    possible = []
    for reference in references:
        if not getattr(reference, "accepted", True):
            continue
        for candidate in candidates:
            if not getattr(candidate, "accepted", True):
                continue
            if reference.direction != candidate.direction:
                continue
            distance = circular_day_distance(reference.center_day, candidate.center_day)
            ratio = candidate.duration_days / max(reference.duration_days, 1e-12)
            if distance > float(center_radius_days):
                continue
            if not 1.0 / float(max_duration_ratio) <= ratio <= float(max_duration_ratio):
                continue
            possible.append((
                float(distance), abs(1.0 - float(ratio)),
                str(reference.coarse_segment_id), str(candidate.coarse_segment_id),
                reference, candidate, float(ratio),
            ))
    used_reference, used_candidate, matches = set(), set(), []
    for distance, _, reference_id, candidate_id, reference, candidate, ratio in sorted(possible):
        if reference_id in used_reference or candidate_id in used_candidate:
            continue
        used_reference.add(reference_id)
        used_candidate.add(candidate_id)
        matches.append(StructureMatch(reference, candidate, distance, ratio))
    return tuple(matches)


def _median_iqr(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    q25, q75 = np.quantile(values, (0.25, 0.75))
    return float(np.median(values)), float(q75 - q25)


def reference_confidence(
    bootstrap_occurrence: float,
    individual_occurrence: float,
    high_threshold: float = 0.80,
    low_threshold: float = 0.50,
    individual_threshold: float = 0.50,
) -> str:
    if not np.isfinite(bootstrap_occurrence) or not np.isfinite(individual_occurrence):
        return "INCONSISTENT_DIAGNOSTIC"
    if bootstrap_occurrence >= high_threshold:
        return (
            "ROBUST_REFERENCE"
            if individual_occurrence >= individual_threshold
            else "STABLE_BUT_LOW_SAMPLE_SUPPORT"
        )
    if bootstrap_occurrence < low_threshold:
        return "UNSTABLE_REFERENCE"
    return "INTERMEDIATE_REFERENCE"


def bootstrap_stability(
    occurrence: float,
    high_threshold: float = 0.80,
    low_threshold: float = 0.50,
) -> str:
    if not np.isfinite(occurrence):
        return "INCONSISTENT_STABILITY"
    if occurrence >= high_threshold:
        return "HIGH_STABILITY"
    if occurrence >= low_threshold:
        return "MEDIUM_STABILITY"
    return "LOW_STABILITY"


def audit_reference_structures(
    source_domain: str,
    class_id: int,
    class_name: str,
    reference_structures: Sequence,
    bootstrap_structure_sets: Sequence[Sequence],
    medoid_structures: Sequence,
    center_radius_days: float = 30,
    max_duration_ratio: float = 2.0,
    high_threshold: float = 0.80,
    low_threshold: float = 0.50,
    individual_threshold: float = 0.50,
) -> list[dict]:
    """Summarize each fixed reference against bootstrap and medoid realizations."""
    by_reference = {
        str(item.coarse_segment_id): [] for item in reference_structures
        if getattr(item, "accepted", True)
    }
    for structures in bootstrap_structure_sets:
        for match in match_structure_sets(
            reference_structures, structures, center_radius_days, max_duration_ratio
        ):
            by_reference[str(match.reference.coarse_segment_id)].append(match.candidate)
    medoid_matches = {
        str(item.reference.coarse_segment_id): item
        for item in match_structure_sets(
            reference_structures, medoid_structures,
            center_radius_days, max_duration_ratio,
        )
    }
    rows = []
    repeats = len(bootstrap_structure_sets)
    for reference in reference_structures:
        if not getattr(reference, "accepted", True):
            continue
        structure_id = str(reference.coarse_segment_id)
        matched = by_reference[structure_id]
        occurrence = len(matched) / repeats if repeats else float("nan")
        duration_median, duration_iqr = _median_iqr(
            [item.duration_days for item in matched]
        )
        change_median, change_iqr = _median_iqr(
            [item.absolute_change for item in matched]
        )
        monotonicity_median, monotonicity_iqr = _median_iqr(
            [item.monotonicity_ratio for item in matched]
        )
        medoid_match = medoid_matches.get(structure_id)
        individual = float(reference.source_occurrence_rate)
        rows.append({
            "source_domain": source_domain,
            "class_id": int(class_id),
            "class_name": class_name,
            "reference_structure_id": structure_id,
            "direction": reference.direction,
            "start_day": float(reference.start_day),
            "end_day": float(reference.end_day),
            "center_day": float(reference.center_day),
            "duration_days": float(reference.duration_days),
            "individual_occurrence_rate": individual,
            "bootstrap_occurrence_rate": float(occurrence),
            "bootstrap_stability": bootstrap_stability(
                occurrence, high_threshold, low_threshold
            ),
            "bootstrap_center_mad_days": circular_mad_days(
                [item.center_day for item in matched], reference_day=reference.center_day
            ),
            "bootstrap_duration_median": duration_median,
            "bootstrap_duration_iqr": duration_iqr,
            "bootstrap_change_median": change_median,
            "bootstrap_change_iqr": change_iqr,
            "bootstrap_monotonicity_median": monotonicity_median,
            "bootstrap_monotonicity_iqr": monotonicity_iqr,
            "present_in_medoid": medoid_match is not None,
            "medoid_center_distance_days": (
                float(medoid_match.center_distance_days)
                if medoid_match is not None else float("nan")
            ),
            "medoid_duration_ratio": (
                float(medoid_match.duration_ratio)
                if medoid_match is not None else float("nan")
            ),
            "reference_confidence": reference_confidence(
                occurrence, individual, high_threshold, low_threshold,
                individual_threshold,
            ),
            "num_bootstrap_runs": repeats,
            "num_bootstrap_matches": len(matched),
        })
    return rows


def select_multivariate_medoid(
    curves,
    max_samples: int = 128,
    seed: int = 1,
) -> MedoidSelection:
    """Select an actual sample using mean-time multivariate Euclidean distance."""
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim != 3 or len(values) == 0:
        raise ValueError("curves must be non-empty [N,K,D]")
    limit = min(len(values), int(max_samples))
    if limit <= 0:
        raise ValueError("max_samples must be positive")
    if limit == len(values):
        selected = np.arange(len(values))
    else:
        rng = np.random.default_rng(int(seed))
        selected = np.sort(rng.choice(len(values), size=limit, replace=False))
    candidates = values[selected]
    totals = np.zeros(limit, dtype=np.float64)
    for left in range(limit):
        distances = np.linalg.norm(candidates[left][None] - candidates, axis=-1).mean(axis=-1)
        totals[left] = distances.sum()
    winner = int(np.argmin(totals))
    sample_index = int(selected[winner])
    return MedoidSelection(sample_index, values[sample_index], selected, float(totals[winner]))


def build_class_summary(
    structure_rows: Sequence[Mapping],
    medoid_structure_counts: Mapping[tuple[str, int], int],
) -> list[dict]:
    groups = {}
    for row in structure_rows:
        key = (str(row["source_domain"]), int(row["class_id"]))
        groups.setdefault(key, []).append(row)
    results = []
    for key in sorted(groups):
        rows = groups[key]
        labels = [row["reference_confidence"] for row in rows]
        results.append({
            "source_domain": key[0], "class_id": key[1],
            "class_name": rows[0]["class_name"],
            "num_reference_structures": len(rows),
            "num_robust_reference": labels.count("ROBUST_REFERENCE"),
            "num_stable_low_sample_support": labels.count("STABLE_BUT_LOW_SAMPLE_SUPPORT"),
            "num_intermediate_reference": labels.count("INTERMEDIATE_REFERENCE"),
            "num_unstable_reference": labels.count("UNSTABLE_REFERENCE"),
            "mean_individual_occurrence": float(np.mean([
                row["individual_occurrence_rate"] for row in rows
            ])),
            "mean_bootstrap_occurrence": float(np.mean([
                row["bootstrap_occurrence_rate"] for row in rows
            ])),
            "medoid_num_structures": int(medoid_structure_counts.get(key, 0)),
            "num_reference_present_in_medoid": sum(bool(row["present_in_medoid"]) for row in rows),
        })
    return results


def build_total_summary(
    structure_rows: Sequence[Mapping], class_rows: Sequence[Mapping]
) -> dict:
    total = len(structure_rows)
    count = lambda label: sum(row["reference_confidence"] == label for row in structure_rows)
    ratio = lambda predicate: (
        sum(bool(predicate(row)) for row in structure_rows) / total if total else float("nan")
    )
    return {
        "num_source_domains": len({row["source_domain"] for row in class_rows}),
        "num_classes": len(class_rows),
        "num_reference_structures": total,
        "num_robust_reference": count("ROBUST_REFERENCE"),
        "num_stable_low_sample_support": count("STABLE_BUT_LOW_SAMPLE_SUPPORT"),
        "num_unstable_reference": count("UNSTABLE_REFERENCE"),
        "fraction_reference_bootstrap_ge_08": ratio(
            lambda row: row["bootstrap_occurrence_rate"] >= 0.8
        ),
        "fraction_reference_bootstrap_lt_05": ratio(
            lambda row: row["bootstrap_occurrence_rate"] < 0.5
        ),
        "fraction_reference_present_in_medoid": ratio(
            lambda row: row["present_in_medoid"]
        ),
    }


def _write_rows(path: Path, rows: Sequence[Mapping], fieldnames=None):
    rows = [dict(row) for row in rows]
    fields = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_source_outputs(
    output_dir,
    structure_rows,
    class_rows,
    bootstrap_rows,
    manifest,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostics").mkdir(exist_ok=True)
    _write_rows(output / "structure_stability.csv", structure_rows, STRUCTURE_STABILITY_FIELDS)
    _write_rows(output / "class_summary.csv", class_rows)
    _write_rows(output / "bootstrap_summary.csv", bootstrap_rows)
    (output / "manifest.json").write_text(
        json.dumps(dict(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def plot_bootstrap_stability(
    output_path,
    grid,
    reference_curve,
    bootstrap_curves,
    reference_structures,
    structure_rows,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(grid)
    bootstrap = np.asarray(bootstrap_curves)
    figure, axes = plt.subplots(2, 1, figsize=(11, 8))
    low, median, high = np.quantile(bootstrap, (0.1, 0.5, 0.9), axis=0)
    axes[0].fill_between(grid, low, high, color="#9ECAE1", alpha=0.45, label="bootstrap 10–90%")
    axes[0].plot(grid, median, color="#4C78A8", ls="--", label="bootstrap median")
    axes[0].plot(grid, reference_curve, color="#1F4E79", lw=2.3, label="reference median")
    by_id = {row["reference_structure_id"]: row for row in structure_rows}
    for structure in reference_structures:
        if not getattr(structure, "accepted", True):
            continue
        row = by_id.get(str(structure.coarse_segment_id), {})
        axes[0].axvspan(structure.start_day, structure.end_day, alpha=0.08, color="#1F4E79")
        axes[0].text(
            structure.center_day,
            np.interp(structure.center_day, grid, reference_curve),
            f"{structure.coarse_segment_id}\nboot={row.get('bootstrap_occurrence_rate', float('nan')):.2f}\nind={row.get('individual_occurrence_rate', float('nan')):.2f}",
            fontsize=7, ha="center",
        )
    axes[0].legend(frameon=False)
    axes[0].set_ylabel("Fixed source-class PC1")
    axes[0].grid(alpha=0.2)
    for row in structure_rows:
        axes[1].scatter(
            row["individual_occurrence_rate"], row["bootstrap_occurrence_rate"],
            s=42, label=row["reference_structure_id"],
        )
        axes[1].annotate(
            row["reference_structure_id"],
            (row["individual_occurrence_rate"], row["bootstrap_occurrence_rate"]),
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )
    axes[1].axvline(0.5, color="#777777", ls=":")
    axes[1].axhline(0.5, color="#777777", ls=":")
    axes[1].axhline(0.8, color="#C44E52", ls="--")
    axes[1].set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel="Individual occurrence", ylabel="Bootstrap occurrence")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_median_vs_medoid(
    output_path,
    grid,
    median_curve,
    medoid_curve,
    median_structures,
    medoid_structures,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(grid)
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(grid, median_curve, color="#1F4E79", lw=2.3, label="median prototype")
    axis.plot(grid, medoid_curve, color="#B85C00", lw=2.0, label="medoid real sample")
    for structures, color, offset in (
        (median_structures, "#1F4E79", 0), (medoid_structures, "#B85C00", 10)
    ):
        for item in structures:
            if not getattr(item, "accepted", True):
                continue
            axis.annotate(
                str(item.coarse_segment_id),
                (item.center_day, np.interp(item.center_day, grid, median_curve if offset == 0 else medoid_curve)),
                xytext=(0, offset), textcoords="offset points", ha="center", color=color,
            )
    axis.set(xlabel="Calendar day", ylabel="Fixed source-class PC1")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
