#!/usr/bin/env python3
"""Visualize 15 training-free decompositions on raw TimeMatch parcel series.

No temporal interpolation, resampling-to-calendar-grid, or imputation is
performed.  The raw Zarr parcel is loaded as [T, 10, S], reduced across pixels
at each acquisition date, and each selected scalar signal is passed directly to
all requested decomposition methods.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.raw_decomposition import (  # noqa: E402
    ALL_METHODS,
    DecompositionConfig,
    OptionalDependencyError,
    decompose,
)
from analysis.raw_decomposition.plotting import plot_result  # noqa: E402
from analysis.raw_decomposition.timematch_raw import (  # noqa: E402
    DOMAIN_ALIASES,
    RawParcelSeries,
    extract_signals,
    parse_signal_names,
    resolve_dataset_name,
)
from dataset import PixelSetData  # noqa: E402
from utils import label_utils  # noqa: E402


def _parse_csv(value: str) -> Sequence[str]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_methods(value: str) -> Sequence[str]:
    if value.strip().lower() == "all":
        return ALL_METHODS
    methods = _parse_csv(value)
    unknown = [method for method in methods if method not in ALL_METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown methods {unknown}; available: {', '.join(ALL_METHODS)}"
        )
    return tuple(dict.fromkeys(methods))


def _parse_class_names(value: str) -> Sequence[str]:
    return _parse_csv(value)


def _safe_slug(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _select_indices(
    labels: np.ndarray,
    classes: Sequence[str],
    *,
    requested_classes: Sequence[str],
    samples_per_class: int,
    seed: int,
) -> List[int]:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    requested = list(requested_classes) if requested_classes else list(classes)
    unknown = sorted(set(requested) - set(classes))
    if unknown:
        raise ValueError(f"classes not available in this dataset: {unknown}")
    rng = np.random.default_rng(int(seed))
    selected: List[int] = []
    for class_name in requested:
        class_id = classes.index(class_name)
        candidates = np.flatnonzero(labels == class_id)
        if candidates.size == 0:
            continue
        order = rng.permutation(candidates)
        selected.extend(int(index) for index in order[:samples_per_class])
    return selected


def _load_raw_parcel(
    dataset: PixelSetData,
    dataset_name: str,
    dataset_index: int,
    classes: Sequence[str],
    signal_names: Sequence[str],
    *,
    spatial_reduction: str,
    pixel_index: int,
) -> RawParcelSeries:
    sample = dataset[int(dataset_index)]
    pixels = np.asarray(sample["pixels"])
    positions = np.asarray(sample["positions"], dtype=np.float64)
    if pixels.shape[0] != positions.size:
        raise ValueError("raw pixels and acquisition positions have different T")
    if not np.all(np.diff(positions) > 0):
        raise ValueError("acquisition positions must be strictly increasing")
    class_id = int(sample["label"])
    signals = extract_signals(
        pixels,
        signal_names=signal_names,
        spatial_reduction=spatial_reduction,
        pixel_index=pixel_index,
    )
    return RawParcelSeries(
        dataset_name=dataset_name,
        dataset_index=int(dataset_index),
        parcel_index=int(sample["parcel_index"]),
        class_id=class_id,
        class_name=classes[class_id],
        positions=positions,
        signals=signals,
    )


def _config_from_args(args) -> DecompositionConfig:
    return DecompositionConfig(
        autoformer_kernel=args.autoformer_kernel,
        fedformer_kernel=args.fedformer_kernel,
        dlinear_kernel=args.dlinear_kernel,
        micn_conv_kernels=tuple(args.micn_conv_kernels),
        timemixer_kernel=args.timemixer_kernel,
        timemixer_downsample_window=args.timemixer_downsample_window,
        timemixer_downsample_layers=args.timemixer_downsample_layers,
        timemixer_dft_top_k=args.timemixer_dft_top_k,
        xpatch_alpha=args.xpatch_alpha,
        stl_period=args.stl_period,
        stl_robust=args.stl_robust,
        emd_max_imf=args.emd_max_imf,
        ceemdan_max_imf=args.ceemdan_max_imf,
        ceemdan_trials=args.ceemdan_trials,
        random_seed=args.seed,
        vmd_alpha=args.vmd_alpha,
        vmd_tau=args.vmd_tau,
        vmd_k=args.vmd_k,
        vmd_dc=args.vmd_dc,
        vmd_init=args.vmd_init,
        vmd_tol=args.vmd_tol,
        wavelet_name=args.wavelet,
        wavelet_level=args.wavelet_level,
        ssa_window=args.ssa_window,
        ssa_components=args.ssa_components,
        fourier_low_fraction=args.fourier_low_fraction,
        fourier_mid_fraction=args.fourier_mid_fraction,
        lomb_top_k=args.lomb_top_k,
        lomb_grid_size=args.lomb_grid_size,
        lomb_min_separation_bins=args.lomb_min_separation_bins,
    )


def _run_one_signal(
    parcel: RawParcelSeries,
    signal_name: str,
    values: np.ndarray,
    methods: Sequence[str],
    config: DecompositionConfig,
    output_dir: Path,
    *,
    dpi: int,
    allow_missing_optional: bool,
) -> List[dict]:
    """Run decompositions and save one PNG per method.

    Output layout:
        <output_dir>/<signal_name>/<method>/<class_name>.png
    """
    signal_dir = output_dir / signal_name
    signal_dir.mkdir(parents=True, exist_ok=True)
    statuses: List[dict] = []
    for method in methods:
        method_dir = signal_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = decompose(method, values, parcel.positions, config)
        except OptionalDependencyError as error:
            status = {
                "method": method,
                "status": "missing_optional_dependency",
                "error": str(error),
            }
            statuses.append(status)
            if not allow_missing_optional:
                raise
            continue
        except Exception as error:
            status = {
                "method": method,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            statuses.append(status)
            raise

        title = (
            f"{parcel.dataset_name} | class={parcel.class_name} | parcel={parcel.parcel_index} | "
            f"signal={signal_name} | method={method}"
        )
        file_path = method_dir / f"{_safe_slug(parcel.class_name)}.png"
        plot_result(
            file_path,
            raw_values=values,
            raw_positions=parcel.positions,
            result=result,
            title=title,
            dpi=dpi,
        )
        statuses.append(
            {
                "method": method,
                "status": "completed",
                "output": str(file_path),
                "components": [component.name for component in result.components],
            }
        )
    return statuses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "TimeMatch dataset path (e.g. austria/33UVP/2017) or alias "
            f"{', '.join(DOMAIN_ALIASES)}"
        ),
    )
    parser.add_argument("--output-dir", default="outputs/raw_decomposition_visualization")
    parser.add_argument("--signals", default="ALL_BANDS", help="comma list of B2..B12, NDVI, ALL_BANDS, or ALL")
    parser.add_argument("--methods", type=_parse_methods, default=ALL_METHODS)
    parser.add_argument("--class-names", type=_parse_class_names, default=())
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--sample-indices", default="", help="optional comma-separated dataset indices; bypasses class sampling")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--spatial-reduction", choices=("mean", "median", "pixel"), default="mean")
    parser.add_argument("--pixel-index", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--allow-missing-optional",
        action="store_true",
        help="skip optional methods whose packages are not installed instead of failing the experiment",
    )

    parser.add_argument("--autoformer-kernel", type=int, default=25)
    parser.add_argument("--fedformer-kernel", type=int, default=25)
    parser.add_argument("--dlinear-kernel", type=int, default=25)
    parser.add_argument("--micn-conv-kernels", type=int, nargs="+", default=[12, 16])
    parser.add_argument("--timemixer-kernel", type=int, default=25)
    parser.add_argument("--timemixer-downsample-window", type=int, default=2)
    parser.add_argument("--timemixer-downsample-layers", type=int, default=2)
    parser.add_argument("--timemixer-dft-top-k", type=int, default=5)
    parser.add_argument("--xpatch-alpha", type=float, default=0.30)
    parser.add_argument("--stl-period", type=int, default=12)
    parser.add_argument("--stl-robust", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--emd-max-imf", type=int, default=5)
    parser.add_argument("--ceemdan-max-imf", type=int, default=5)
    parser.add_argument("--ceemdan-trials", type=int, default=50)
    parser.add_argument("--vmd-alpha", type=float, default=2000.0)
    parser.add_argument("--vmd-tau", type=float, default=0.0)
    parser.add_argument("--vmd-k", type=int, default=5)
    parser.add_argument("--vmd-dc", type=int, default=0)
    parser.add_argument("--vmd-init", type=int, default=1)
    parser.add_argument("--vmd-tol", type=float, default=1e-7)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--wavelet-level", type=int, default=3)
    parser.add_argument("--ssa-window", type=int, default=0)
    parser.add_argument("--ssa-components", type=int, default=5)
    parser.add_argument("--fourier-low-fraction", type=float, default=0.15)
    parser.add_argument("--fourier-mid-fraction", type=float, default=0.40)
    parser.add_argument("--lomb-top-k", type=int, default=3)
    parser.add_argument("--lomb-grid-size", type=int, default=1024)
    parser.add_argument("--lomb-min-separation-bins", type=int, default=8)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset_name = resolve_dataset_name(args.dataset)
    country = dataset_name.split("/")[-3]
    signal_names = parse_signal_names(args.signals)
    methods = tuple(args.methods) if not isinstance(args.methods, str) else _parse_methods(args.methods)
    config = _config_from_args(args)

    # label_utils resolves mapping files relative to repository root.
    if Path.cwd().resolve() != REPOSITORY_ROOT:
        import os
        os.chdir(REPOSITORY_ROOT)

    classes = label_utils.get_classes(country)
    dataset = PixelSetData(
        args.data_root,
        dataset_name,
        classes,
        transform=None,
        closed_set=True,
    )
    labels = dataset.get_labels()

    if args.sample_indices.strip():
        selected = [int(item) for item in _parse_csv(args.sample_indices)]
    else:
        selected = _select_indices(
            labels,
            classes,
            requested_classes=args.class_names,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
    if not selected:
        raise RuntimeError("no parcels were selected")

    root = Path(args.output_dir).resolve() / _safe_slug(dataset_name)
    root.mkdir(parents=True, exist_ok=True)


    all_statuses = []
    for dataset_index in selected:
        parcel = _load_raw_parcel(
            dataset,
            dataset_name,
            dataset_index,
            classes,
            signal_names,
            spatial_reduction=args.spatial_reduction,
            pixel_index=args.pixel_index,
        )
        for signal_name, values in parcel.signals.items():
            statuses = _run_one_signal(
                parcel,
                signal_name,
                values,
                methods,
                config,
                root,
                dpi=args.dpi,
                allow_missing_optional=args.allow_missing_optional,
            )
            for status in statuses:
                all_statuses.append(
                    {
                        "dataset_index": parcel.dataset_index,
                        "parcel_index": parcel.parcel_index,
                        "class_name": parcel.class_name,
                        "signal": signal_name,
                        **status,
                    }
                )
            completed = sum(status["status"] == "completed" for status in statuses)
            print(
                f"DECOMP_VIS|dataset={dataset_name}|parcel={parcel.parcel_index}|"
                f"class={parcel.class_name}|signal={signal_name}|completed={completed}/{len(methods)}"
            )

    incomplete = [row for row in all_statuses if row["status"] != "completed"]
    if incomplete and not args.allow_missing_optional:
        return 2

    duplicate_keys = []
    seen = set()
    for row in all_statuses:
        if row.get("status") != "completed":
            continue
        key = (row["signal"], row["method"], row["class_name"])
        if key in seen:
            duplicate_keys.append(key)
        else:
            seen.add(key)
    if duplicate_keys:
        unique = sorted(set(duplicate_keys))
        print(
            "WARNING: multiple selected parcels share the same class name. "
            "With output layout signal/method/class.png, later samples overwrite earlier ones. "
            f"Overwritten keys: {unique}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
