"""Audit TimeMatch metadata time coordinates without changing dataset code."""

import argparse
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np

DOMAIN_PATHS = {
    "AT1": "austria/33UVP/2017",
    "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
}


def _parse_date(value):
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    digits = "".join(character for character in str(value) if character.isdigit())
    if len(digits) != 8:
        raise ValueError("dates must use YYYYMMDD or YYYY-MM-DD")
    return dt.date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))


def _number(values, reducer):
    array = np.asarray(values, dtype=np.float64)
    return float(reducer(array)) if array.size else float("nan")


def audit_domain_metadata(domain, metadata):
    """Return calendar-direction and sampling statistics for one domain."""
    domain_start = metadata["start_date"]
    domain_dates = metadata["dates"]
    parcels = metadata.get("parcels", [])
    sample_records = parcels if parcels else [{}]

    starts = []
    all_positions = []
    spans = []
    lengths = []
    less = equal = greater = 0
    less_examples = []
    has_sample_start = False

    for parcel in sample_records:
        has_sample_start = has_sample_start or "start_date" in parcel
        start_value = parcel.get("start_date", domain_start)
        date_values = parcel.get("dates", domain_dates)
        start = _parse_date(start_value)
        dates = [_parse_date(value) for value in date_values]
        signed = np.asarray([(date - start).days for date in dates], dtype=np.int64)
        positions = np.abs(signed)

        starts.append(start)
        all_positions.extend(positions.tolist())
        lengths.append(len(positions))
        spans.append(float(positions.max() - positions.min()) if len(positions) else 0.0)
        less += int((signed < 0).sum())
        equal += int((signed == 0).sum())
        greater += int((signed > 0).sum())
        for date, delta in zip(dates, signed.tolist()):
            if delta < 0 and len(less_examples) < 5:
                less_examples.append(
                    {
                        "sample_index": len(starts) - 1,
                        "date": date.strftime("%Y%m%d"),
                        "start_date": start.strftime("%Y%m%d"),
                    }
                )

    unique_starts = sorted(set(starts))
    return {
        "domain": domain,
        "start_date": _parse_date(domain_start).strftime("%Y%m%d"),
        "start_date_fixed": len(unique_starts) == 1,
        "sample_specific_start_date": has_sample_start,
        "sample_start_dates": [value.strftime("%Y%m%d") for value in unique_starts],
        "sample_count": len(parcels),
        "observation_count": len(all_positions),
        "position_min": _number(all_positions, np.min),
        "position_max": _number(all_positions, np.max),
        "position_mean": _number(all_positions, np.mean),
        "position_median": _number(all_positions, np.median),
        "sample_span_mean": _number(spans, np.mean),
        "sample_span_median": _number(spans, np.median),
        "sample_span_p95": _number(spans, lambda values: np.quantile(values, 0.95)),
        "sample_span_max": _number(spans, np.max),
        "sequence_length_mean": _number(lengths, np.mean),
        "sequence_length_median": _number(lengths, np.median),
        "sequence_length_min": int(min(lengths)) if lengths else 0,
        "sequence_length_max": int(max(lengths)) if lengths else 0,
        "date_lt_start_date_count": less,
        "date_eq_start_date_count": equal,
        "date_gt_start_date_count": greater,
        "date_lt_start_date_examples": less_examples,
        "position_unit": "days",
    }


def summarize_temporal_positions(data_root, domains):
    """Compatibility entry point for auditing relative dataset paths."""
    reports = []
    for domain in domains:
        metadata_path = Path(data_root) / domain / "meta" / "metadata.pkl"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"metadata unavailable: {metadata_path}")
        with metadata_path.open("rb") as handle:
            metadata = pickle.load(handle)
        reports.append(audit_domain_metadata(domain, metadata))
    starts = {report["start_date"] for report in reports}
    return {
        "origins_consistent": len(starts) <= 1,
        "domains": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--domains", nargs="+", default=list(DOMAIN_PATHS))
    args = parser.parse_args()

    reports = []
    for domain in args.domains:
        if domain not in DOMAIN_PATHS:
            raise SystemExit(f"unknown domain alias: {domain}")
        metadata_path = Path(args.data_root) / DOMAIN_PATHS[domain] / "meta" / "metadata.pkl"
        if not metadata_path.is_file():
            raise SystemExit(f"metadata unavailable: {metadata_path}")
        with metadata_path.open("rb") as handle:
            metadata = pickle.load(handle)
        reports.append(audit_domain_metadata(domain, metadata))

    print(json.dumps(reports, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
