"""Source-only, time-free structural identity followed by timing measurement."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from analysis.structure_observation_support_diagnostic import staged_output, write_csv, write_json

EPS = 1e-8
SOURCES = ("AT1", "DK1", "FR1", "FR2")
LABELS = ("LIKELY_SAME_STRUCTURE", "STRUCTURALLY_PLAUSIBLE", "AMBIGUOUS", "NO_STRUCTURAL_COUNTERPART")
TIMING_LABELS = LABELS[:2]
TIMING_FIELDS = ("center_displacement_days", "start_displacement_days", "end_displacement_days",
                 "reference_duration_days", "sample_duration_days", "duration_ratio",
                 "duration_difference_days", "stretch_difference_days")


@dataclass(frozen=True)
class Descriptor:
    direction: str
    curve_change: float
    domain_change: float
    monotonicity: float
    fine_count: int


def descriptor(segment):
    result = Descriptor(str(segment["direction"]), float(segment["curve_normalized_change"]),
                        float(segment["domain_normalized_change"]), float(segment["monotonicity_ratio"]),
                        int(segment["num_fine_segments_covered"]))
    values = [result.curve_change, result.domain_change, result.monotonicity, result.fine_count]
    if result.direction not in ("RISE", "FALL") or not np.isfinite(values).all() or min(values) < 0:
        raise ValueError("invalid structural descriptor")
    return result


def differences(reference, sample):
    return np.array([
        abs(np.log((sample.curve_change + EPS) / (reference.curve_change + EPS))),
        abs(np.log((sample.domain_change + EPS) / (reference.domain_change + EPS))),
        abs(sample.monotonicity - reference.monotonicity),
        abs(np.log((1 + sample.fine_count) / (1 + reference.fine_count))),
    ])


def structural_cost(reference, sample, calibration):
    if reference.direction != sample.direction:
        return float("inf")
    return float(np.mean(differences(reference, sample) / np.asarray(calibration["scales"])))


def calibrate(records, min_pairs=50, scale_quantile=.90, cost_quantile=.95, source_domains=SOURCES):
    if set(source_domains) != set(SOURCES) or {row["source_domain"] for row in records} != set(SOURCES):
        raise ValueError("calibration requires G+ contributions from all four sources")
    if min_pairs < 1 or not (0 < scale_quantile <= 1 and 0 < cost_quantile <= 1):
        raise ValueError("invalid calibration settings")
    if len(records) < min_pairs:
        raise ValueError("insufficient all-source pooled calibration pairs")
    output = {}
    for source in SOURCES:
        for direction in ("RISE", "FALL"):
            pools = [("source_direction", [r for r in records if r["source_domain"] == source and r["direction"] == direction]),
                     ("source_pooled", [r for r in records if r["source_domain"] == source]),
                     ("all_source_pooled", records)]
            level, pool = next((level, pool) for level, pool in pools if len(pool) >= min_pairs)
            ds = np.asarray([r["differences"] for r in pool], dtype=float)
            if ds.shape != (len(pool), 4) or not np.isfinite(ds).all() or np.any(ds < 0):
                raise ValueError("invalid G+ differences")
            scales = np.quantile(ds, scale_quantile, axis=0) + EPS
            threshold = float(np.quantile((ds / scales).mean(axis=1), cost_quantile))
            output[source + ":" + direction] = dict(source_domain=source, direction=direction,
                calibration_level=level, num_Gplus_pairs=len(pool), scales=scales.tolist(), threshold=threshold,
                contributing_sources=sorted({r["source_domain"] for r in pool}))
    return output


@dataclass(frozen=True)
class Alignment:
    pairs: tuple
    rotation: int
    total_cost: float
    gap_count: int


def circular_alignment(costs, thresholds):
    """Lexicographic partial alignment; indices encode order, never time proximity.

    Gaps count unmatched sequence elements. Equal-count paths consequently have
    equal gap counts. Pair-index order resolves remaining within-rotation ties.
    """
    costs = np.asarray(costs, dtype=float)
    n, m = costs.shape
    thresholds = np.asarray(thresholds)
    if thresholds.shape != (n,) or np.isnan(costs).any():
        raise ValueError("invalid alignment costs/thresholds")
    if not n or not m:
        return Alignment((), 0, 0., n + m)
    best = None
    for rotation in range(m):
        order = [(j + rotation) % m for j in range(m)]
        dp = [[(0., ()) for _ in range(m + 1)] for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                options = [dp[i-1][j], dp[i][j-1]]
                c = costs[i-1, order[j-1]]
                if np.isfinite(c) and c <= thresholds[i-1]:
                    prior, pairs = dp[i-1][j-1]
                    options.append((prior + c, pairs + ((i-1, order[j-1]),)))
                dp[i][j] = min(options, key=lambda entry: (-len(entry[1]), entry[0], entry[1]))
        cost, pairs = dp[n][m]
        result = Alignment(pairs, rotation, float(cost), n + m - 2 * len(pairs))
        key = (-len(pairs), cost, result.gap_count, rotation, pairs)
        if best is None or key < best[0]:
            best = key, result
    return best[1]


def classify_alignment(costs, thresholds, alignment):
    costs = np.asarray(costs)
    n, m = costs.shape
    pairs = dict(alignment.pairs)
    result = []
    for i in range(n):
        eligible = np.flatnonzero(np.isfinite(costs[i]) & (costs[i] <= thresholds[i]))
        ordered = sorted(float(value) for value in costs[i] if np.isfinite(value))
        j = pairs.get(i)
        mutual, neighbors = False, 0
        if j is not None:
            row_best, col_best = costs[i].min(), costs[:, j].min()
            mutual = bool(costs[i, j] == row_best == col_best
                          and np.count_nonzero(costs[i] == row_best) == 1
                          and np.count_nonzero(costs[:, j] == col_best) == 1)
            if n > 1 and m > 1:
                neighbors = sum(pairs.get((i + step) % n) == (j + step) % m for step in (-1, 1))
        if not len(eligible):
            label = "NO_STRUCTURAL_COUNTERPART"
        elif j is None:
            label = "AMBIGUOUS"
        elif mutual or neighbors >= 1:
            label = "LIKELY_SAME_STRUCTURE"
        elif len(eligible) > 1:
            label = "AMBIGUOUS"
        else:
            label = "STRUCTURALLY_PLAUSIBLE"
        result.append(dict(identity_status=label, candidate_index=j, assignment_conflict=bool(len(eligible) and j is None),
            mutual_best=mutual, neighbor_support=int(neighbors), num_identity_candidates=len(eligible),
            best_structural_cost=ordered[0] if ordered else float("nan"),
            second_best_structural_cost=ordered[1] if len(ordered) > 1 else float("nan"),
            cost_margin=ordered[1]-ordered[0] if len(ordered) > 1 else float("nan"),
            structural_cost=float(costs[i, j]) if j is not None else float("nan")))
    return result


def measure_timing(reference, sample):
    a, x = float(reference["start_day"]), float(sample["start_day"])
    b, y = float(reference["end_day"]), float(sample["end_day"])
    if b <= a:
        b += 365.
    if y <= x:
        y += 365.
    offset = min((-365., 0., 365.), key=lambda k: (abs((x+y)/2 + k - (a+b)/2), abs(k), k))
    x, y = x + offset, y + offset
    return dict(center_displacement_days=(x+y-a-b)/2, start_displacement_days=x-a,
        end_displacement_days=y-b, reference_duration_days=b-a, sample_duration_days=y-x,
        duration_ratio=(y-x)/(b-a), duration_difference_days=(y-x)-(b-a), stretch_difference_days=(y-b)-(x-a))


def identify(references, candidates, calibrations):
    refs, samples = [descriptor(r) for r in references], [descriptor(s) for s in candidates]
    costs = np.array([[structural_cost(r, s, calibrations[r.direction]) for s in samples] for r in refs]).reshape(len(refs), len(samples))
    thresholds = np.array([calibrations[r.direction]["threshold"] for r in refs])
    alignment = circular_alignment(costs, thresholds)
    rows = classify_alignment(costs, thresholds, alignment)
    for i, row in enumerate(rows):
        j = row["candidate_index"]
        row.update(reference_structure_id=references[i]["coarse_segment_id"], reference_direction=refs[i].direction,
            identity_sample_segment_id=candidates[j]["coarse_segment_id"] if j is not None else "",
            identity_threshold=float(thresholds[i]), calibration_level=calibrations[refs[i].direction]["calibration_level"],
            reference_cross_boundary=bool(references[i]["crosses_year_boundary"]),
            sample_cross_boundary=bool(candidates[j]["crosses_year_boundary"]) if j is not None else None)
        row.update({field: float("nan") for field in TIMING_FIELDS})
        if row["identity_status"] in TIMING_LABELS:
            row.update(measure_timing(references[i], candidates[j]))
    return rows, alignment


def grouped_candidates(segments, min_monotonicity):
    """Auxiliary 1:3 evidence on the complete circular chain, never primary pairs."""
    groups = []
    if len(segments) < 3:
        return groups
    for i in range(len(segments)):
        triple = [segments[(i+j) % len(segments)] for j in range(3)]
        first, middle, last = triple
        if first["direction"] != last["direction"] or first["direction"] == middle["direction"]:
            continue
        signed = float(last["end_value"]) - float(first["start_value"])
        if (first["direction"] == "RISE" and signed <= 0) or (first["direction"] == "FALL" and signed >= 0):
            continue
        path = sum(float(s["total_path_variation"]) for s in triple)
        monotonicity = abs(signed) / (path + EPS)
        if monotonicity < min_monotonicity:
            continue
        # Normalizers are shared across a sample's segments. Signed sums also
        # avoid recovering a scale from a nearly zero endpoint difference.
        def net(name):
            return abs(sum((1 if s["direction"] == "RISE" else -1) * float(s[name]) for s in triple))
        groups.append(dict(coarse_segment_id="+".join(s["coarse_segment_id"] for s in triple),
            grouped_candidate_ids=[s["coarse_segment_id"] for s in triple], direction=first["direction"],
            curve_normalized_change=net("curve_normalized_change"), domain_normalized_change=net("domain_normalized_change"),
            monotonicity_ratio=monotonicity, num_fine_segments_covered=sum(int(s["num_fine_segments_covered"]) for s in triple)))
    return groups


def audit_group(reference, groups, calibration):
    options = [(structural_cost(descriptor(reference), descriptor(group), calibration), index, group)
               for index, group in enumerate(groups)]
    valid = [entry for entry in options if np.isfinite(entry[0]) and entry[0] <= calibration["threshold"]]
    if not valid:
        return dict(has_grouped_candidate=False, grouped_status="NO_PLAUSIBLE_GROUPED_COUNTERPART",
                    grouped_candidate_ids=[], grouped_structural_cost=float("nan"), grouped_monotonicity=float("nan"),
                    group_start_segment="", group_end_segment="")
    cost, _, group = min(valid, key=lambda entry: entry[:2])
    ids = group["grouped_candidate_ids"]
    return dict(has_grouped_candidate=True, grouped_status="HAS_PLAUSIBLE_GROUPED_COUNTERPART",
                grouped_candidate_ids=ids, grouped_structural_cost=cost, grouped_monotonicity=group["monotonicity_ratio"],
                group_start_segment=ids[0], group_end_segment=ids[-1])


def index_baseline(rows, expected_keys):
    expected_keys = list(expected_keys)
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("duplicate expected reference/sample key")
    result = {}
    for row in rows:
        key = (row["source_domain"], int(row["class_id"]), str(row["reference_structure_id"]), int(row["sample_id"]))
        if key in result:
            raise ValueError(f"duplicate 06B baseline key: {key}")
        matched = str(row["matched"]).strip().lower()
        plus = matched in ("true", "1")
        if (matched not in ("true", "false", "1", "0")
                or plus != bool(row["matched_sample_segment_id"])
                or plus == bool(row["failure_reason"])):
            raise ValueError(f"inconsistent 06B historical state: {key}")
        result[key] = row
    if set(result) != set(expected_keys):
        raise ValueError(f"06B baseline key mismatch: missing={len(set(expected_keys)-set(result))}, extra={len(set(result)-set(expected_keys))}")
    return result


def _quantile(rows, field, q):
    values = np.asarray([row[field] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else float("nan")


def _rate(count, total):
    return count / total if total else float("nan")


def summarize_structures(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source_domain"], row["class_id"], row["reference_structure_id"])].append(row)
    result = []
    for _, group in sorted(groups.items()):
        first = group[0]
        minus = [r for r in group if r["baseline_06B_status"] == "G-"]
        plus = [r for r in group if r["baseline_06B_status"] == "G+"]
        rescued = [r for r in minus if r["identity_status"] in TIMING_LABELS]
        measured = [dict(r, abs_center=abs(r["center_displacement_days"]), abs_stretch=abs(r["stretch_difference_days"])) for r in rescued]
        consistent = sum(r["identity_sample_segment_id"] == r["baseline_matched_segment_id"] for r in plus)
        row = {k: first[k] for k in ("task", "source_domain", "class_id", "class_name", "reference_structure_id", "bootstrap_occurrence", "baseline_individual_occurrence")}
        row.update(num_samples=len(group), num_Gplus=len(plus), num_Gminus=len(minus),
            num_Gminus_likely_same=sum(r["identity_status"] == LABELS[0] for r in minus),
            num_Gminus_plausible=sum(r["identity_status"] == LABELS[1] for r in minus),
            num_Gminus_ambiguous=sum(r["identity_status"] == LABELS[2] for r in minus),
            num_Gminus_no_counterpart=sum(r["identity_status"] == LABELS[3] for r in minus),
            Gminus_identity_rescue_rate=_rate(len(rescued), len(minus)),
            gplus_identity_consistency_count=consistent, gplus_identity_consistency_rate=_rate(consistent, len(plus)),
            median_center_displacement=_quantile(measured, "center_displacement_days", .5),
            p90_abs_center_displacement=_quantile(measured, "abs_center", .9),
            max_abs_center_displacement=_quantile(measured, "abs_center", 1),
            median_duration_ratio=_quantile(measured, "duration_ratio", .5),
            duration_ratio_iqr=_quantile(measured, "duration_ratio", .75)-_quantile(measured, "duration_ratio", .25),
            median_abs_stretch_difference=_quantile(measured, "abs_stretch", .5),
            num_grouped_candidates=sum(bool(r["has_grouped_candidate"]) for r in minus))
        result.append(row)
    return result


def summarize_transitions(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["baseline_06B_status"] != "G-":
            continue
        key = (row["class_id"], row["reference_structure_id"], row["baseline_failure_reason"])
        groups[("STRUCTURE",) + key].append(row)
        groups[("TOTAL", "", "", row["baseline_failure_reason"])].append(row)
    return [dict(scope=key[0], class_id=key[1], reference_structure_id=key[2], baseline_failure_reason=key[3],
                 identity_status=label, count=sum(r["identity_status"] == label for r in group),
                 fraction=sum(r["identity_status"] == label for r in group)/len(group))
            for key, group in sorted(groups.items()) for label in LABELS]


def summarize_boundaries(rows):
    result = []
    for boundary in (True, False):
        minus = [r for r in rows if r["baseline_06B_status"] == "G-" and r["reference_cross_boundary"] == boundary]
        rescued = [dict(r, abs_center=abs(r["center_displacement_days"])) for r in minus if r["identity_status"] in TIMING_LABELS]
        result.append(dict(cross_boundary=boundary, num_Gminus=len(minus), identity_rescue_rate=_rate(len(rescued), len(minus)),
            median_abs_center_displacement=_quantile(rescued, "abs_center", .5),
            p90_abs_center_displacement=_quantile(rescued, "abs_center", .9), median_duration_ratio=_quantile(rescued, "duration_ratio", .5),
            grouped_candidate_rate=_rate(sum(bool(r["has_grouped_candidate"]) for r in minus), len(minus))))
    return result


def _shade(axis, segment, color):
    a, b = float(segment["start_day"]), float(segment["end_day"])
    for left, right in ((a, b),) if b > a else ((a, 365), (0, b)):
        axis.axvspan(left, right, alpha=.13, color=color)


def plot_identity(path, prototype, curve, references, candidates, alignment, row, calibration, grouped=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    ref = next(r for r in references if r["coarse_segment_id"] == row["reference_structure_id"])
    ids = row["grouped_candidate_ids"] if grouped else [row["identity_sample_segment_id"]]
    chosen = [s for s in candidates if s["coarse_segment_id"] in ids]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [2, 1.3, 1]})
    axes[0].plot(np.arange(len(prototype)), prototype, label="reference prototype", color="#245580", lw=2)
    axes[0].plot(np.arange(len(curve)), curve, label=f"source sample {row['sample_id']}", color="#be6900")
    _shade(axes[0], ref, "#245580")
    for item in chosen:
        _shade(axes[0], item, "#be6900")
    axes[0].set(xlabel="Source calendar day", ylabel="Fixed source-class PC1", xlim=(0, 365))
    axes[0].legend()
    axes[0].grid(alpha=.2)
    for objects, height, prefix in ((references, 1, "reference"), (candidates, 0, "sample")):
        for i, item in enumerate(objects):
            x = i / max(len(objects)-1, 1)
            axes[1].text(x, height, item["coarse_segment_id"] + (" ↑" if item["direction"] == "RISE" else " ↓"), ha="center", fontsize=8)
    for i, j in alignment.pairs:
        axes[1].add_artist(ConnectionPatch((i/max(len(references)-1, 1), .9), (j/max(len(candidates)-1, 1), .15),
            "data", "data", axesA=axes[1], axesB=axes[1], color="gray", alpha=.6))
    axes[1].set(xlim=(-.08, 1.08), ylim=(-.2, 1.2), title="Circular sequence order / time-free primary assignment")
    axes[1].axis("off")
    if grouped:
        text = ("grouped diagnostic only — not primary correspondence\n"
                f"segments: {' + '.join(ids)}\ncost={row['grouped_structural_cost']:.4f}; "
                f"threshold={row['identity_threshold']:.4f}; monotonicity={row['grouped_monotonicity']:.4f}")
    else:
        ds = differences(descriptor(ref), descriptor(chosen[0]))
        text = (f"cost={row['structural_cost']:.4f}; threshold={row['identity_threshold']:.4f}; "
                f"unique mutual best={row['mutual_best']}; neighbor support={row['neighbor_support']}\n"
                f"curve difference={ds[0]:.4f}; domain difference={ds[1]:.4f}; "
                f"monotonicity difference={ds[2]:.4f}; fine-count difference={ds[3]:.4f}\n"
                f"scales={np.round(calibration['scales'], 4).tolist()}\n"
                f"center displacement={row['center_displacement_days']:+.1f} days; duration ratio={row['duration_ratio']:.3f}")
    axes[2].text(.01, .95, text, transform=axes[2].transAxes, va="top", fontsize=10)
    axes[2].axis("off")
    fig.suptitle(f"{row['task']} | {row['class_name']} | {row['reference_structure_id']}\n"
                 f"06B={row['baseline_failure_reason']} | 06C={row['identity_status']}")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_phase_distribution(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r["baseline_06B_status"] == "G-" and r["identity_status"] in TIMING_LABELS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if not selected:
        for axis in axes:
            axis.text(.5, .5, "No identity-rescued G-", ha="center", transform=axis.transAxes)
    for reason in sorted({r["baseline_failure_reason"] for r in selected}):
        group = [r for r in selected if r["baseline_failure_reason"] == reason]
        axes[0].hist([r["center_displacement_days"] for r in group], bins=15, alpha=.5, label=reason)
        axes[1].hist([r["duration_ratio"] for r in group], bins=15, alpha=.5)
        axes[2].scatter([r["start_displacement_days"] for r in group], [r["end_displacement_days"] for r in group], alpha=.4, s=10)
    axes[0].set(xlabel="Center displacement (days)", ylabel="Source samples")
    axes[1].set(xlabel="Sample / reference duration")
    axes[2].set(xlabel="Start displacement (days)", ylabel="End displacement (days)")
    if selected:
        axes[0].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=.2)
    first = rows[0]
    fig.suptitle(f"{first['task']} | {first['class_name']} | {first['reference_structure_id']} | identity then timing")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
