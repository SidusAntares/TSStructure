"""Utilities for extracting raw parcel-level scalar series from TimeMatch pixels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np


SENTINEL2_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
BAND_TO_INDEX = {name: index for index, name in enumerate(SENTINEL2_BANDS)}

DOMAIN_ALIASES = {
    "AT1": "austria/33UVP/2017",
    "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
}


@dataclass(frozen=True)
class RawParcelSeries:
    dataset_name: str
    dataset_index: int
    parcel_index: int
    class_id: int
    class_name: str
    positions: np.ndarray
    signals: Dict[str, np.ndarray]


def resolve_dataset_name(value: str) -> str:
    return DOMAIN_ALIASES.get(value.upper(), value)


def parse_signal_names(value: str) -> Sequence[str]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("at least one signal is required")
    expanded = []
    for token in tokens:
        upper = token.upper()
        if upper in {"ALL", "ALL_BANDS"}:
            expanded.extend(SENTINEL2_BANDS)
            if upper == "ALL":
                expanded.append("NDVI")
        elif upper == "NDVI":
            expanded.append("NDVI")
        elif upper in BAND_TO_INDEX:
            expanded.append(upper)
        else:
            raise ValueError(
                f"unknown signal {token!r}; use {SENTINEL2_BANDS}, NDVI, ALL_BANDS or ALL"
            )
    # Stable deduplication.
    return tuple(dict.fromkeys(expanded))


def _spatial_reduce(values: np.ndarray, reduction: str, pixel_index: int) -> np.ndarray:
    if reduction == "mean":
        return np.mean(values, axis=-1)
    if reduction == "median":
        return np.median(values, axis=-1)
    if reduction == "pixel":
        if pixel_index < 0 or pixel_index >= values.shape[-1]:
            raise IndexError(
                f"pixel_index={pixel_index} outside parcel with {values.shape[-1]} pixels"
            )
        return values[..., pixel_index]
    raise ValueError("spatial reduction must be mean, median or pixel")


def extract_signals(
    pixels: np.ndarray,
    *,
    signal_names: Iterable[str],
    spatial_reduction: str,
    pixel_index: int,
) -> Dict[str, np.ndarray]:
    """Convert raw [T,10,S] pixel sets to scalar observed time series.

    This is a same-date spatial reduction only.  It never changes, inserts or
    removes acquisition times and therefore is not temporal interpolation.
    """
    x = np.asarray(pixels)
    if x.ndim != 3 or x.shape[1] != len(SENTINEL2_BANDS):
        raise ValueError(
            f"expected raw TimeMatch pixels [T,10,S], got {x.shape}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("raw parcel contains NaN/Inf; no imputation is performed")
    result: Dict[str, np.ndarray] = {}
    for name in signal_names:
        if name in BAND_TO_INDEX:
            band_values = x[:, BAND_TO_INDEX[name], :].astype(np.float64)
            result[name] = _spatial_reduce(band_values, spatial_reduction, pixel_index)
        elif name == "NDVI":
            red = x[:, BAND_TO_INDEX["B4"], :].astype(np.float64)
            nir = x[:, BAND_TO_INDEX["B8"], :].astype(np.float64)
            denominator = nir + red
            ndvi = np.divide(
                nir - red,
                denominator,
                out=np.zeros_like(denominator, dtype=np.float64),
                where=np.abs(denominator) > 1e-12,
            )
            result[name] = _spatial_reduce(ndvi, spatial_reduction, pixel_index)
        else:
            raise KeyError(name)
    return result
