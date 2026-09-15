"""Pure, source-only scoring utilities for the 07A local-waveform audit."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import shutil
import uuid

import numpy as np

EPS = 1e-12
TASKS = ("AT1_DK1", "DK1_FR1", "FR1_FR2", "FR2_AT1")
REQUIRED_TASK_FILES = (
    "gplus_local_ranking.csv", "gplus_structure_summary.csv",
    "metric_comparison.csv", "transition_summary.csv", "template_summary.csv",
    "bootstrap_summary.csv", "cue_pair_audit.csv", "cue_summary.csv", "manifest.json",
)
REQUIRED_ROOT_FILES = (
    "metric_comparison.csv", "transition_summary.csv", "bootstrap_summary.csv",
    "cue_summary.csv", "manifest.json",
)
CUE_PAIR_FIELDS = (
    "task", "source_domain", "class_id", "class_name", "reference_structure_id", "sample_id",
    "positive_segment_id", "hard_negative_segment_id", "reference_event_sequence",
    "positive_event_sequence", "negative_event_sequence", "reference_event_tokens",
    "positive_event_tokens", "negative_event_tokens", "positive_type_exact", "negative_type_exact",
    "positive_location_error", "negative_location_error", "positive_prominence_error",
    "negative_prominence_error", "positive_pc1_slope_ned", "negative_pc1_slope_ned",
    "positive_multi_slope_ned", "negative_multi_slope_ned",
    "reference_curve_change", "positive_curve_change", "negative_curve_change",
    "reference_domain_change", "positive_domain_change", "negative_domain_change",
    "reference_monotonicity", "positive_monotonicity", "negative_monotonicity",
    "reference_fine_count", "positive_fine_count", "negative_fine_count",
    "cue_status", "cue_help_source",
)
CUE_SUMMARY_FIELDS = (
    "task", "class_id", "class_name", "reference_structure_id", "scope", "num_multi_ned_wrong",
    "cue_helpful_count", "cue_helpful_rate", "type_sequence_helpful_count",
    "type_sequence_helpful_rate", "pareto_helpful_count", "pareto_helpful_rate",
    "ambiguous_count", "ambiguous_rate", "positive_type_exact_count", "positive_type_exact_rate",
    "negative_type_exact_count", "negative_type_exact_rate", "positive_exact_negative_not_count",
    "positive_exact_negative_not_rate",
)


@dataclass(frozen=True)
class ExtremaToken:
    event_type: str
    relative_location: float
    normalized_prominence: float


def _value(item, name):
    return item[name] if isinstance(item, dict) else getattr(item, name)


def extract_extrema_tokens(events, segment, period_days=365.0):
    """Extract the complete, ordered extrema chain inside one coarse window."""
    start = float(_value(segment, "start_day"))
    end = float(_value(segment, "end_day"))
    if end <= start:
        end += float(period_days)
    duration = end - start
    if duration <= 0:
        raise ValueError("invalid coarse segment duration")
    selected = []
    for item in events:
        day = float(_value(item, "day")) % float(period_days)
        if day < start - EPS:
            day += float(period_days)
        if start - EPS <= day <= end + EPS:
            kind = str(_value(item, "kind")).lower()
            event_type = "P" if kind == "peak" else "V" if kind == "valley" else kind.upper()
            try:
                prominence = float(_value(item, "relative_prominence"))
            except (AttributeError, KeyError):
                prominence = float(_value(item, "normalized_prominence"))
            selected.append((day, str(_value(item, "event_id")), ExtremaToken(
                event_type=event_type,
                relative_location=float(np.clip((day - start) / duration, 0.0, 1.0)),
                normalized_prominence=prominence,
            )))
    return tuple(item[2] for item in sorted(selected, key=lambda value: (value[0], value[1])))


def event_sequence(tokens):
    return "-".join(item.event_type for item in tokens)


def serialize_event_tokens(tokens):
    return ";".join(
        f"{item.event_type}@{item.relative_location:.8f}:{item.normalized_prominence:.8f}"
        for item in tokens
    )


def _aligned_cue_errors(reference, candidate):
    exact = (
        len(reference) == len(candidate)
        and all(left.event_type == right.event_type for left, right in zip(reference, candidate))
    )
    if not exact:
        return False, float("nan"), float("nan")
    if not reference:
        return True, 0.0, 0.0
    location = float(np.mean([
        abs(left.relative_location - right.relative_location)
        for left, right in zip(reference, candidate)
    ]))
    prominence = float(np.mean([
        abs(left.normalized_prominence - right.normalized_prominence)
        for left, right in zip(reference, candidate)
    ]))
    return True, location, prominence


def audit_structural_cues(reference, positive, negative):
    positive_exact, positive_location, positive_prominence = _aligned_cue_errors(reference, positive)
    negative_exact, negative_location, negative_prominence = _aligned_cue_errors(reference, negative)
    status, source = "WAVEFORM_AND_CUE_AMBIGUOUS", "AMBIGUOUS"
    if positive_exact and not negative_exact:
        status, source = "CUE_HELPFUL", "TYPE_SEQUENCE"
    elif positive_exact and negative_exact:
        no_worse = positive_location <= negative_location and positive_prominence <= negative_prominence
        strictly_better = positive_location < negative_location or positive_prominence < negative_prominence
        if no_worse and strictly_better:
            status, source = "CUE_HELPFUL", "LOCATION_PROMINENCE_PARETO"
    return {
        "reference_event_sequence": event_sequence(reference),
        "positive_event_sequence": event_sequence(positive),
        "negative_event_sequence": event_sequence(negative),
        "positive_type_exact": positive_exact,
        "negative_type_exact": negative_exact,
        "positive_location_error": positive_location,
        "negative_location_error": negative_location,
        "positive_prominence_error": positive_prominence,
        "negative_prominence_error": negative_prominence,
        "cue_status": status,
        "cue_help_source": source,
    }


def relative_queries(segment, points=32, period_days=365.0):
    if points < 2:
        raise ValueError("local waveform requires at least two points")
    start, end = float(segment["start_day"]), float(segment["end_day"])
    if end <= start:
        end += period_days
    if end <= start:
        raise ValueError("invalid segment interval")
    return np.linspace(start, end, points, endpoint=True)


def sample_periodic_curve(curve, query_days, period_days=365.0):
    values = np.asarray(curve, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) != int(period_days):
        raise ValueError("periodic curve must be [period_days,D]")
    query = np.mod(np.asarray(query_days, dtype=float), period_days)
    base = np.arange(len(values) + 1, dtype=float)
    closed = np.concatenate([values, values[:1]], axis=0)
    return np.stack([np.interp(query, base, closed[:, index]) for index in range(values.shape[1])], axis=-1)


def evaluate_mode13(coefficients, query_days, period_days=365.0):
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 2 or coefficients.shape[0] != 13:
        raise ValueError("Mode13 coefficients must be [13,D]")
    modes = np.arange(-6, 7, dtype=float)
    points = 2 * np.pi * np.asarray(query_days, dtype=float) / period_days
    basis = np.exp(1j * points[:, None] * modes[None])
    return (basis @ coefficients).real.astype(np.float64)


def endpoint_relative(values):
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError("waveform must be [K] or [K,D]")
    return values - values[0]


def endpoint_relative_ned(reference, sample, eps=EPS):
    q, w = endpoint_relative(reference), endpoint_relative(sample)
    if q.shape != w.shape:
        raise ValueError("waveform shapes must match")
    return float(np.sum((q - w) ** 2) / (np.sum(q ** 2) + np.sum(w ** 2) + eps))


def slope_ned(reference, sample, eps=EPS):
    q, w = np.diff(np.asarray(reference, dtype=float), axis=0), np.diff(np.asarray(sample, dtype=float), axis=0)
    if q.shape != w.shape:
        raise ValueError("waveform shapes must match")
    return float(np.sum((q - w) ** 2) / (np.sum(q ** 2) + np.sum(w ** 2) + eps))


def endpoint_relative_cosine(reference, sample, eps=EPS):
    q, w = endpoint_relative(reference), endpoint_relative(sample)
    if q.shape != w.shape:
        raise ValueError("waveform shapes must match")
    q_energy, w_energy = float(np.linalg.norm(q)), float(np.linalg.norm(w))
    degenerate = q_energy < eps or w_energy < eps
    if degenerate:
        score = 0.0 if q_energy < eps and w_energy < eps else 1.0
    else:
        score = 1.0 - float(np.sum(q * w) / (q_energy * w_energy + eps))
    return score, q_energy, w_energy, degenerate


def candidate_pool(segments, direction, baseline_segment_id):
    pool = [row for row in segments if bool(row.get("accepted", False)) and row["direction"] == direction]
    if baseline_segment_id not in {row["coarse_segment_id"] for row in pool}:
        raise ValueError(f"baseline segment is not an accepted same-direction candidate: {baseline_segment_id}")
    return pool


def rank_scores(scores, positive_id):
    if positive_id not in scores or not scores:
        raise ValueError("positive candidate missing from scores")
    for key, value in scores.items():
        if not np.isfinite(value):
            raise ValueError(f"non-finite candidate score: {key}")
    ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]))
    minimum = ordered[0][1]
    tied = [key for key, value in ordered if np.isclose(value, minimum, rtol=1e-9, atol=1e-12)]
    best_id = min(tied)
    positive_rank = next(index + 1 for index, (key, _) in enumerate(ordered) if key == positive_id)
    positive = float(scores[positive_id])
    negatives = [float(value) for key, value in scores.items() if key != positive_id]
    best_negative = min(negatives) if negatives else float("nan")
    margin = ((best_negative - positive) / max(best_negative, positive, EPS)) if negatives else float("nan")
    wins = sum(positive < value and not np.isclose(positive, value, rtol=1e-9, atol=1e-12) for value in negatives)
    return dict(best_id=best_id, exact=best_id == positive_id, unique_best=len(tied) == 1,
                positive_rank=positive_rank, top2=positive_rank <= 2, mrr=1.0 / positive_rank,
                positive_distance=positive, best_negative_distance=best_negative, margin=float(margin),
                pairwise_wins=int(wins), pairwise_total=len(negatives))


def compare_candidate_waveforms(reference, candidates, pc1_axis, positive_id, handcrafted_scores):
    reference = np.asarray(reference, dtype=float)
    axis = np.asarray(pc1_axis, dtype=float)
    if reference.ndim != 2 or axis.shape != (reference.shape[1],):
        raise ValueError("reference/PC1 dimensions do not match")
    pc1_scores, multi_scores, cosine_scores = {}, {}, {}
    for candidate_id, waveform in candidates.items():
        waveform = np.asarray(waveform, dtype=float)
        multi_scores[candidate_id] = endpoint_relative_ned(reference, waveform)
        pc1_scores[candidate_id] = endpoint_relative_ned(reference @ axis, waveform @ axis)
        cosine_scores[candidate_id] = endpoint_relative_cosine(reference, waveform)[0]
    return {"handcrafted": rank_scores(handcrafted_scores, positive_id),
            "pc1_ned": rank_scores(pc1_scores, positive_id),
            "multi_ned": rank_scores(multi_scores, positive_id),
            "multi_cosine": rank_scores(cosine_scores, positive_id)}


def template_pc1_consistency_error(multivariate_template, pc1_axis, expected_pc1, projection_offset=0.0):
    actual = np.asarray(multivariate_template) @ np.asarray(pc1_axis) - float(projection_offset)
    expected = np.asarray(expected_pc1)
    return float(np.linalg.norm(actual - expected) / (np.linalg.norm(expected) + EPS))


def reliable_references(rows, threshold=.8):
    return [row for row in rows if float(row["bootstrap_occurrence_rate"]) >= threshold]


def gplus_rows(rows):
    return [row for row in rows if row.get("matched") is True or str(row.get("matched", "")).lower() in ("true", "1")]


def _aggregate(rows, prefix, output_prefix=None):
    output_prefix = output_prefix or prefix
    exact_key = f"{prefix}_exact"
    rank_key = f"{prefix}_positive_rank"
    wins_key = f"{prefix}_pairwise_wins"
    total_key = f"{prefix}_pairwise_total"
    if prefix == "handcrafted_local":
        rank_key, wins_key, total_key = "handcrafted_positive_rank", "handcrafted_pairwise_wins", "handcrafted_pairwise_total"
    if not rows:
        return {f"{output_prefix}_top1": float("nan"), f"{output_prefix}_top2": float("nan"),
                f"{output_prefix}_mrr": float("nan"), f"{output_prefix}_pairwise_win": float("nan")}
    wins = sum(row[wins_key] for row in rows)
    total = sum(row[total_key] for row in rows)
    return {f"{output_prefix}_top1": float(np.mean([row[exact_key] for row in rows])),
            f"{output_prefix}_top2": float(np.mean([row[rank_key] <= 2 for row in rows])),
            f"{output_prefix}_mrr": float(np.mean([1 / row[rank_key] for row in rows])),
            f"{output_prefix}_pairwise_win": wins / total if total else float("nan")}


def metric_summary(rows, task):
    output = []
    for scope, selected in (("ALL_GPLUS", list(rows)),
                            ("MULTI_CANDIDATE_ONLY", [row for row in rows if row["multi_candidate"]]),
                            ("THREE_PLUS_CANDIDATES", [row for row in rows if row.get("num_same_direction_candidates", 0) >= 3])):
        record = dict(task=task, scope=scope, num_pairs=len(selected))
        for prefix, output_prefix in (("handcrafted_local", "handcrafted"), ("pc1_ned", "pc1"), ("multi_ned", "multi")):
            record.update(_aggregate(selected, prefix, output_prefix))
        record["multi_cosine_top1"] = float(np.mean([row["multi_cosine_exact"] for row in selected])) if selected else float("nan")
        record["multi_minus_handcrafted"] = record["multi_top1"] - record["handcrafted_top1"]
        record["multi_minus_pc1"] = record["multi_top1"] - record["pc1_top1"]
        record["pc1_minus_handcrafted"] = record["pc1_top1"] - record["handcrafted_top1"]
        output.append(record)
    return output


def structure_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["class_id"], row["reference_structure_id"])].append(row)
    for task in sorted({row["task"] for row in rows}):
        groups[(task, "TOTAL", "TOTAL")] = [row for row in rows if row["task"] == task]
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        for scope, selected in (("ALL_GPLUS", group), ("MULTI_CANDIDATE_ONLY", [r for r in group if r["multi_candidate"]])):
            record = dict(task=key[0], class_id=key[1],
                          class_name="TOTAL" if key[1] == "TOTAL" else group[0].get("class_name", ""),
                          reference_structure_id=key[2], scope=scope,
                          num_Gplus=len(selected), num_multi_candidate=sum(r["multi_candidate"] for r in selected))
            for prefix, output_prefix in (("handcrafted_local", "handcrafted"), ("pc1_ned", "pc1"), ("multi_ned", "multi")):
                record.update(_aggregate(selected, prefix, output_prefix))
            record["multi_unique_best_rate"] = float(np.mean([r["multi_ned_unique_best"] for r in selected])) if selected else float("nan")
            margins = [r["multi_ned_margin"] for r in selected if np.isfinite(r["multi_ned_margin"])]
            record["median_multi_margin"] = float(np.median(margins)) if margins else float("nan")
            output.append(record)
    return output


def cue_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["class_id"], row["class_name"], row["reference_structure_id"])].append(row)
    for task in sorted({row["task"] for row in rows}):
        groups[(task, "TOTAL", "TOTAL", "TOTAL")] = [row for row in rows if row["task"] == task]
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        total = len(group)
        helpful = sum(row["cue_status"] == "CUE_HELPFUL" for row in group)
        type_helpful = sum(row["cue_help_source"] == "TYPE_SEQUENCE" for row in group)
        pareto_helpful = sum(row["cue_help_source"] == "LOCATION_PROMINENCE_PARETO" for row in group)
        ambiguous = sum(row["cue_status"] == "WAVEFORM_AND_CUE_AMBIGUOUS" for row in group)
        positive_exact = sum(bool(row["positive_type_exact"]) for row in group)
        negative_exact = sum(bool(row["negative_type_exact"]) for row in group)
        positive_only = sum(bool(row["positive_type_exact"]) and not bool(row["negative_type_exact"]) for row in group)
        output.append({
            "task": key[0], "class_id": key[1], "class_name": key[2],
            "reference_structure_id": key[3], "scope": "MULTI_NED_WRONG",
            "num_multi_ned_wrong": total,
            "cue_helpful_count": helpful,
            "cue_helpful_rate": helpful / total if total else float("nan"),
            "type_sequence_helpful_count": type_helpful,
            "type_sequence_helpful_rate": type_helpful / total if total else float("nan"),
            "pareto_helpful_count": pareto_helpful,
            "pareto_helpful_rate": pareto_helpful / total if total else float("nan"),
            "ambiguous_count": ambiguous,
            "ambiguous_rate": ambiguous / total if total else float("nan"),
            "positive_type_exact_count": positive_exact,
            "positive_type_exact_rate": positive_exact / total if total else float("nan"),
            "negative_type_exact_count": negative_exact,
            "negative_type_exact_rate": negative_exact / total if total else float("nan"),
            "positive_exact_negative_not_count": positive_only,
            "positive_exact_negative_not_rate": positive_only / total if total else float("nan"),
        })
    return output


def transition_summary(rows, task):
    predicates = (
        ("06C_WRONG_TO_MULTI_CORRECT", lambda r: not r["06c_final_exact"] and r["multi_ned_exact"]),
        ("06C_CORRECT_TO_MULTI_WRONG", lambda r: r["06c_final_exact"] and not r["multi_ned_exact"]),
        ("HANDCRAFTED_WRONG_TO_MULTI_CORRECT", lambda r: not r["handcrafted_local_exact"] and r["multi_ned_exact"]),
        ("HANDCRAFTED_CORRECT_TO_MULTI_WRONG", lambda r: r["handcrafted_local_exact"] and not r["multi_ned_exact"]),
        ("BOTH_06C_AND_MULTI_CORRECT", lambda r: r["06c_final_exact"] and r["multi_ned_exact"]),
        ("BOTH_06C_AND_MULTI_WRONG", lambda r: not r["06c_final_exact"] and not r["multi_ned_exact"]),
        ("WAVEFORM_HARD_CUE_HELPFUL", lambda r: not r["multi_ned_exact"] and r.get("cue_status") == "CUE_HELPFUL"),
        ("WAVEFORM_AND_CUE_AMBIGUOUS", lambda r: not r["multi_ned_exact"] and r.get("cue_status") == "WAVEFORM_AND_CUE_AMBIGUOUS"),
    )
    return [dict(task=task, transition=name, count=sum(predicate(row) for row in rows))
            for name, predicate in predicates if any(predicate(row) for row in rows)]


def cluster_bootstrap(rows, repeats=1000, seed=1):
    rows = [row for row in rows if row["multi_candidate"]]
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("task", ""), row["sample_id"])].append(row)
    keys = sorted(groups)
    if not keys:
        return dict(num_clusters=0, repeats=repeats,
                    multi_minus_handcrafted_mean=float("nan"), multi_minus_handcrafted_ci_low=float("nan"),
                    multi_minus_handcrafted_ci_high=float("nan"), multi_minus_pc1_mean=float("nan"),
                    multi_minus_pc1_ci_low=float("nan"), multi_minus_pc1_ci_high=float("nan"))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = rng.integers(0, len(keys), size=len(keys))
        batch = [row for index in sampled for row in groups[keys[int(index)]]]
        values.append((np.mean([r["multi_ned_exact"] for r in batch]) - np.mean([r["handcrafted_local_exact"] for r in batch]),
                       np.mean([r["multi_ned_exact"] for r in batch]) - np.mean([r["pc1_ned_exact"] for r in batch])))
    values = np.asarray(values)
    return dict(num_clusters=len(keys), repeats=repeats,
                multi_minus_handcrafted_mean=float(values[:, 0].mean()),
                multi_minus_handcrafted_ci_low=float(np.quantile(values[:, 0], .025)),
                multi_minus_handcrafted_ci_high=float(np.quantile(values[:, 0], .975)),
                multi_minus_pc1_mean=float(values[:, 1].mean()),
                multi_minus_pc1_ci_low=float(np.quantile(values[:, 1], .025)),
                multi_minus_pc1_ci_high=float(np.quantile(values[:, 1], .975)))


def publish_revision(staging, final):
    staging, final = Path(staging), Path(final)
    for name in REQUIRED_ROOT_FILES:
        path = staging / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"incomplete 07A revision: {name}")
    for task in TASKS:
        for name in REQUIRED_TASK_FILES:
            path = staging / task / name
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"incomplete 07A revision: {task}/{name}")
    final.parent.mkdir(parents=True, exist_ok=True)
    backup = final.parent / f".old_{final.name}_{uuid.uuid4().hex}"
    moved = False
    try:
        if final.exists():
            final.replace(backup); moved = True
        staging.replace(final)
    except Exception:
        if moved and backup.exists() and not final.exists():
            backup.replace(final)
        raise
    if backup.exists():
        shutil.rmtree(backup)
