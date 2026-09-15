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
COMPONENT_NAMES = ("curve_change", "domain_change", "monotonicity", "fine_count")


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


def cost_components(reference, sample, calibration):
    raw = differences(reference, sample)
    active = np.asarray(calibration.get("active", [True] * len(raw)), dtype=bool)
    scales = np.asarray(calibration["scales"], dtype=float)
    if raw.shape != active.shape or scales.shape != raw.shape:
        raise ValueError("invalid calibration component shape")
    contribution = np.full(raw.shape, np.nan, dtype=float)
    if np.any(active & (~np.isfinite(scales) | (scales <= 0))):
        raise ValueError("active calibration scales must be finite and positive")
    contribution[active] = raw[active] / scales[active]
    return raw, contribution


def structural_cost(reference, sample, calibration):
    if reference.direction != sample.direction:
        return float("inf")
    _, contribution = cost_components(reference, sample, calibration)
    if not np.isfinite(contribution).any():
        raise ValueError("calibration has no active components")
    return float(np.nanmean(contribution))


def _fit_calibration_pool(pool, min_pairs, scale_quantile, cost_quantile,
                          min_positive_pairs, min_active_components):
    if len(pool) < min_pairs:
        return None
    ds = np.asarray([r["differences"] for r in pool], dtype=float)
    if ds.shape != (len(pool), 4) or not np.isfinite(ds).all() or np.any(ds < 0):
        raise ValueError("invalid G+ differences")
    positive_pairs = np.sum(ds > 1e-12, axis=0)
    active = positive_pairs >= min_positive_pairs
    if int(active.sum()) < min_active_components:
        return None
    scales = np.full(4, np.nan, dtype=float)
    for index in np.flatnonzero(active):
        scales[index] = np.quantile(ds[ds[:, index] > 1e-12, index], scale_quantile)
    if np.any(~np.isfinite(scales[active]) | (scales[active] <= 0)):
        return None
    costs = np.mean(ds[:, active] / scales[active], axis=1)
    threshold = float(np.quantile(costs, cost_quantile))
    if not np.isfinite(threshold) or threshold >= 1e4:
        return None
    return dict(scales=scales.tolist(), active=active.tolist(),
                positive_pairs=positive_pairs.astype(int).tolist(),
                num_active_components=int(active.sum()), threshold=threshold)


def calibrate(records, min_pairs=50, scale_quantile=.90, cost_quantile=.95,
              source_domains=SOURCES, min_positive_pairs=20, min_active_components=2):
    if set(source_domains) != set(SOURCES) or {row["source_domain"] for row in records} != set(SOURCES):
        raise ValueError("calibration requires G+ contributions from all four sources")
    if (min_pairs < 1 or min_positive_pairs < 1 or not 1 <= min_active_components <= 4
            or not (0 < scale_quantile <= 1 and 0 < cost_quantile <= 1)):
        raise ValueError("invalid calibration settings")
    if len(records) < min_pairs:
        raise ValueError("insufficient all-source pooled calibration pairs")
    output = {}
    for source in SOURCES:
        for direction in ("RISE", "FALL"):
            pools = [("source_direction", [r for r in records if r["source_domain"] == source and r["direction"] == direction]),
                     ("source_pooled", [r for r in records if r["source_domain"] == source]),
                     ("all_source_pooled", records)]
            selected = None
            for level, pool in pools:
                fitted = _fit_calibration_pool(pool, min_pairs, scale_quantile, cost_quantile,
                                               min_positive_pairs, min_active_components)
                if fitted is not None:
                    selected = level, pool, fitted
                    break
            if selected is None:
                raise ValueError(f"insufficient active components for all-source calibration: {source}:{direction}")
            level, pool, fitted = selected
            output[source + ":" + direction] = dict(source_domain=source, direction=direction,
                calibration_level=level, num_Gplus_pairs=len(pool),
                contributing_sources=sorted({r["source_domain"] for r in pool}), **fitted)
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
        ref_unique, sample_unique, mutual, neighbors = False, False, False, 0
        if j is not None:
            row_best, col_best = costs[i].min(), costs[:, j].min()
            ref_unique = bool(costs[i, j] == row_best and np.count_nonzero(costs[i] == row_best) == 1)
            sample_unique = bool(costs[i, j] == col_best and np.count_nonzero(costs[:, j] == col_best) == 1)
            mutual = ref_unique and sample_unique
            if n > 1 and m > 1:
                neighbors = sum(pairs.get((i + step) % n) == (j + step) % m for step in (-1, 1))
        if not len(eligible):
            label = "NO_STRUCTURAL_COUNTERPART"
        elif j is None:
            label = "AMBIGUOUS"
        elif mutual:
            label = "LIKELY_SAME_STRUCTURE"
        elif len(eligible) > 1:
            label = "AMBIGUOUS"
        else:
            label = "STRUCTURALLY_PLAUSIBLE"
        result.append(dict(identity_status=label, candidate_index=j, assignment_conflict=bool(len(eligible) and j is None),
            reference_unique_best=ref_unique, sample_unique_best=sample_unique,
            unique_mutual_best=mutual, mutual_best=mutual,
            neighbor_support=int(neighbors), num_identity_candidates=len(eligible),
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
    primary = [item for item in candidates if bool(item.get("accepted", False))]
    secondary = [item for item in candidates if not bool(item.get("accepted", False))]
    refs, samples = [descriptor(r) for r in references], [descriptor(s) for s in primary]
    costs = np.array([[structural_cost(r, s, calibrations[r.direction]) for s in samples]
                      for r in refs]).reshape(len(refs), len(samples))
    thresholds = np.array([calibrations[r.direction]["threshold"] for r in refs])
    alignment = circular_alignment(costs, thresholds)
    rows = classify_alignment(costs, thresholds, alignment)
    for i, row in enumerate(rows):
        j = row["candidate_index"]
        chosen = primary[j] if j is not None else None
        row.update(candidate_source="PRIMARY_ACCEPTED" if chosen is not None else "NONE",
                   num_secondary_candidates=0)
        if row["identity_status"] in ("NO_STRUCTURAL_COUNTERPART", "AMBIGUOUS") and secondary:
            secondary_costs = np.asarray([structural_cost(refs[i], descriptor(item), calibrations[refs[i].direction])
                                          for item in secondary])
            eligible = np.flatnonzero(np.isfinite(secondary_costs) & (secondary_costs <= thresholds[i]))
            row["num_secondary_candidates"] = int(len(eligible))
            if len(eligible) == 1:
                chosen = secondary[int(eligible[0])]
                row.update(identity_status="STRUCTURALLY_PLAUSIBLE", candidate_source="SECONDARY_REJECTED",
                           structural_cost=float(secondary_costs[eligible[0]]),
                           best_structural_cost=float(np.min(secondary_costs)),
                           second_best_structural_cost=float("nan"), cost_margin=float("nan"),
                           reference_unique_best=False, sample_unique_best=False,
                           unique_mutual_best=False, mutual_best=False, assignment_conflict=False)
        row.update(reference_structure_id=references[i]["coarse_segment_id"], reference_direction=refs[i].direction,
            identity_sample_segment_id=chosen["coarse_segment_id"] if chosen is not None else "",
            primary_sample_segment_id=primary[j]["coarse_segment_id"] if j is not None else "",
            sample_segment_accepted=bool(chosen["accepted"]) if chosen is not None else None,
            identity_threshold=float(thresholds[i]), calibration_level=calibrations[refs[i].direction]["calibration_level"],
            reference_cross_boundary=bool(references[i]["crosses_year_boundary"]),
            sample_cross_boundary=bool(chosen["crosses_year_boundary"]) if chosen is not None else None)
        if chosen is not None:
            raw, contribution = cost_components(refs[i], descriptor(chosen), calibrations[refs[i].direction])
        else:
            raw, contribution = np.full(4, np.nan), np.full(4, np.nan)
        for name, value, contribution_value in zip(COMPONENT_NAMES, raw, contribution):
            row[f"{name}_difference"] = float(value)
            row[f"{name}_cost_contribution"] = float(contribution_value)
        active = calibrations[refs[i].direction]["active"]
        row.update(active_component_count=int(sum(active)),
                   curve_component_active=bool(active[0]), curve_component_contribution=float(contribution[0]),
                   domain_component_active=bool(active[1]), domain_component_contribution=float(contribution[1]),
                   monotonicity_component_active=bool(active[2]), monotonicity_component_contribution=float(contribution[2]),
                   fine_count_component_active=bool(active[3]), fine_count_component_contribution=float(contribution[3]))
        row.update({field: float("nan") for field in TIMING_FIELDS})
        if row["identity_status"] in TIMING_LABELS and chosen is not None:
            row.update(measure_timing(references[i], chosen))
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
        high = [r for r in minus if r["candidate_source"] == "PRIMARY_ACCEPTED"
                and r["identity_status"] == "LIKELY_SAME_STRUCTURE"]
        loose = [r for r in minus if r["identity_status"] in TIMING_LABELS]
        measured = [dict(r, abs_center=abs(r["center_displacement_days"]), abs_stretch=abs(r["stretch_difference_days"]))
                    for r in minus if r["candidate_source"] == "PRIMARY_ACCEPTED"
                    and r["identity_status"] in TIMING_LABELS]
        exact = sum(bool(r.get("gplus_exact_segment_recovered")) for r in plus)
        any_identity = sum(r["identity_status"] in TIMING_LABELS for r in plus)
        exact_likely = sum(bool(r.get("gplus_exact_segment_recovered"))
                           and r["identity_status"] == "LIKELY_SAME_STRUCTURE" for r in plus)
        row = {k: first[k] for k in ("task", "source_domain", "class_id", "class_name", "reference_structure_id", "bootstrap_occurrence", "baseline_individual_occurrence")}
        row.update(num_samples=len(group), num_Gplus=len(plus), num_Gminus=len(minus),
            num_Gminus_likely_same=sum(r["identity_status"] == LABELS[0] for r in minus),
            num_Gminus_plausible=sum(r["identity_status"] == LABELS[1] for r in minus),
            num_Gminus_ambiguous=sum(r["identity_status"] == LABELS[2] for r in minus),
            num_Gminus_no_counterpart=sum(r["identity_status"] == LABELS[3] for r in minus),
            Gminus_high_confidence_rescue_rate=_rate(len(high), len(minus)),
            Gminus_plausible_or_better_rate=_rate(len(loose), len(minus)),
            Gminus_primary_likely=len(high),
            Gminus_primary_plausible=sum(r["candidate_source"] == "PRIMARY_ACCEPTED"
                                         and r["identity_status"] == "STRUCTURALLY_PLAUSIBLE" for r in minus),
            Gminus_secondary_plausible=sum(r["candidate_source"] == "SECONDARY_REJECTED"
                                           and r["identity_status"] == "STRUCTURALLY_PLAUSIBLE" for r in minus),
            gplus_exact_segment_count=exact, gplus_exact_segment_rate=_rate(exact, len(plus)),
            gplus_any_identity_rate=_rate(any_identity, len(plus)),
            gplus_exact_likely_rate=_rate(exact_likely, len(plus)),
            wrong_segment_likely_rate=_rate(sum(r["identity_status"] == "LIKELY_SAME_STRUCTURE"
                                                and not r.get("gplus_exact_segment_recovered", False) for r in plus), len(plus)),
            median_center_displacement=_quantile(measured, "center_displacement_days", .5),
            p90_abs_center_displacement=_quantile(measured, "abs_center", .9),
            max_abs_center_displacement=_quantile(measured, "abs_center", 1),
            median_duration_ratio=_quantile(measured, "duration_ratio", .5),
            duration_ratio_iqr=_quantile(measured, "duration_ratio", .75)-_quantile(measured, "duration_ratio", .25),
            median_abs_stretch_difference=_quantile(measured, "abs_stretch", .5),
            num_grouped_candidates=sum(bool(r["has_grouped_candidate"]) for r in minus))
        result.append(row)
    return result


def summarize_gplus(rows):
    groups = defaultdict(list)
    plus = [row for row in rows if row["baseline_06B_status"] == "G+"]
    for row in plus:
        groups[("STRUCTURE", row["source_domain"], row["class_id"], row["reference_structure_id"])].append(row)
        groups[("TOTAL", "", "", "")].append(row)
    result = []
    for key, group in sorted(groups.items()):
        total = len(group)
        any_identity = sum(r["identity_status"] in TIMING_LABELS for r in group)
        exact = sum(bool(r.get("gplus_exact_segment_recovered")) for r in group)
        exact_likely = sum(bool(r.get("gplus_exact_segment_recovered"))
                           and r["identity_status"] == "LIKELY_SAME_STRUCTURE" for r in group)
        wrong_likely = sum(not bool(r.get("gplus_exact_segment_recovered"))
                           and r["identity_status"] == "LIKELY_SAME_STRUCTURE" for r in group)
        result.append(dict(scope=key[0], source_domain=key[1], class_id=key[2], reference_structure_id=key[3],
                           num_Gplus=total, num_any_identity=any_identity,
                           gplus_any_identity_rate=_rate(any_identity, total),
                           num_exact_segment=exact, gplus_exact_segment_rate=_rate(exact, total),
                           num_exact_likely=exact_likely, gplus_exact_likely_rate=_rate(exact_likely, total),
                           num_wrong_segment_likely=wrong_likely,
                           wrong_segment_likely_rate=_rate(wrong_likely, total)))
    return result


def summarize_transitions(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["baseline_06B_status"] != "G-":
            continue
        key = (row["class_id"], row["reference_structure_id"], row["baseline_failure_reason"], row["candidate_source"])
        groups[("STRUCTURE",) + key].append(row)
        groups[("TOTAL", "", "", row["baseline_failure_reason"], row["candidate_source"])].append(row)
    return [dict(scope=key[0], class_id=key[1], reference_structure_id=key[2], baseline_failure_reason=key[3],
                 candidate_source=key[4],
                 identity_status=label, count=sum(r["identity_status"] == label for r in group),
                 fraction=sum(r["identity_status"] == label for r in group)/len(group))
            for key, group in sorted(groups.items()) for label in LABELS]


def summarize_boundaries(rows):
    result = []
    for boundary in (True, False):
        minus = [r for r in rows if r["baseline_06B_status"] == "G-" and r["reference_cross_boundary"] == boundary]
        rescued = [dict(r, abs_center=abs(r["center_displacement_days"])) for r in minus
                   if r["candidate_source"] == "PRIMARY_ACCEPTED" and r["identity_status"] == "LIKELY_SAME_STRUCTURE"]
        loose = [r for r in minus if r["identity_status"] in TIMING_LABELS]
        result.append(dict(cross_boundary=boundary, num_Gminus=len(minus),
            high_confidence_rescue_rate=_rate(len(rescued), len(minus)),
            plausible_or_better_rate=_rate(len(loose), len(minus)),
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
    primary = [item for item in candidates if bool(item.get("accepted", False))]
    ordered_primary = primary[alignment.rotation:] + primary[:alignment.rotation]
    for objects, height, prefix in ((references, 1, "reference"), (ordered_primary, 0, "sample")):
        for i, item in enumerate(objects):
            x = i / max(len(objects)-1, 1)
            axes[1].text(x, height, f"{item['coarse_segment_id']}\n{item['direction']}\naccepted={item.get('accepted', False)}", ha="center", fontsize=7)
    for i, j in alignment.pairs:
        adjusted = (j - alignment.rotation) % max(len(primary), 1)
        axes[1].add_artist(ConnectionPatch((i/max(len(references)-1, 1), .9), (adjusted/max(len(primary)-1, 1), .15),
            "data", "data", axesA=axes[1], axesB=axes[1], color="gray", alpha=.6))
    axes[1].set(xlim=(-.08, 1.08), ylim=(-.2, 1.2),
                title="Sequence order only — x-axis is NOT calendar time")
    axes[1].axis("off")
    if grouped:
        text = ("grouped diagnostic only — not primary correspondence\n"
                f"segments: {' + '.join(ids)}\ncost={row['grouped_structural_cost']:.4f}; "
                f"threshold={row['identity_threshold']:.4f}; monotonicity={row['grouped_monotonicity']:.4f}")
    else:
        ds, contributions = cost_components(descriptor(ref), descriptor(chosen[0]), calibration)
        component_lines = []
        for name, raw, scale, active, contribution in zip(
                COMPONENT_NAMES, ds, calibration["scales"], calibration["active"], contributions):
            component_lines.append(f"{name}: raw={raw:.4f}, scale={scale:.4f}, contribution={contribution:.4f}"
                                   if active else f"{name}: DISABLED (raw={raw:.4f})")
        text = (f"cost={row['structural_cost']:.4f}; threshold={row['identity_threshold']:.4f}; "
                f"source={row['candidate_source']}; accepted={row['sample_segment_accepted']}\n"
                f"unique mutual best={row['unique_mutual_best']}; neighbor support={row['neighbor_support']}\n"
                + "\n".join(component_lines) + "\n"
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
    selected = [r for r in rows if r["baseline_06B_status"] == "G-"
                and r["candidate_source"] == "PRIMARY_ACCEPTED"
                and r["identity_status"] == "LIKELY_SAME_STRUCTURE"]
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


def select_gplus_misassignments(rows, limit=20):
    """Deterministically prioritize wrong-likely cases, class coverage, then margin extremes."""
    pool = [row for row in rows if row["baseline_06B_status"] == "G+"
            and not row.get("gplus_exact_segment_recovered", False)]
    def stable_key(row):
        return int(row["class_id"]), str(row["reference_structure_id"]), int(row["sample_id"])
    def margin(row):
        value = float(row.get("cost_margin", float("nan")))
        return value if np.isfinite(value) else -float("inf")
    wrong_likely = sorted((row for row in pool if row["identity_status"] == "LIKELY_SAME_STRUCTURE"),
                          key=stable_key)
    selected = wrong_likely[:limit]
    remaining = [row for row in pool if row not in selected]
    for class_id in sorted({int(row["class_id"]) for row in pool}):
        item = next((row for row in sorted(remaining, key=stable_key) if int(row["class_id"]) == class_id), None)
        if item is not None and len(selected) < limit:
            selected.append(item)
    remaining = [row for row in remaining if row not in selected]
    extremes = []
    ordered = sorted(remaining, key=lambda row: (margin(row), stable_key(row)))
    while ordered:
        extremes.append(ordered.pop(0))
        if ordered:
            extremes.append(ordered.pop())
    selected.extend(row for row in extremes if len(selected) < limit)
    return selected[:limit]


def plot_gplus_misassignment(path, prototype, curve, reference, baseline_segment, chosen_segment, row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [2.5, 1]})
    axes[0].plot(np.arange(len(prototype)), prototype, color="#245580", lw=2, label="reference prototype")
    axes[0].plot(np.arange(len(curve)), curve, color="#be6900", lw=1.5, label=f"source sample {row['sample_id']}")
    _shade(axes[0], reference, "#245580")
    _shade(axes[0], baseline_segment, "#2ca02c")
    if chosen_segment is not None:
        _shade(axes[0], chosen_segment, "#d62728")
    axes[0].set(xlim=(0, 365), xlabel="Source calendar day", ylabel="Fixed source-class PC1")
    axes[0].legend()
    axes[0].grid(alpha=.2)
    chosen_id = chosen_segment["coarse_segment_id"] if chosen_segment is not None else "NONE"
    axes[1].text(.02, .92,
        f"reference: {reference['coarse_segment_id']} ({reference['direction']})\n"
        f"06B baseline matched: {baseline_segment['coarse_segment_id']} (green)\n"
        f"06C chosen: {chosen_id} (red)\n"
        f"status={row['identity_status']}; candidate_source={row['candidate_source']}; "
        f"accepted={row['sample_segment_accepted']}; margin={row['cost_margin']}",
        transform=axes[1].transAxes, va="top")
    axes[1].axis("off")
    fig.suptitle(f"G+ exact-segment mismatch | {row['task']} | {row['class_name']}")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=140)
    plt.close(fig)
