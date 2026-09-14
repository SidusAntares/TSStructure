#!/usr/bin/env python3
"""Audit the source-only validity of configuration-05 structure references."""

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.recon_structure_segments import build_coarse_structure, detect_structure_segments
from analysis.shift_visualization import (
    replay_train_indices,
    resolve_visualization_config_dir,
)
from analysis.structure_reference_validity_diagnostic import (
    audit_reference_structures,
    bootstrap_subsample_indices,
    build_class_summary,
    build_total_summary,
    plot_bootstrap_stability,
    plot_median_vs_medoid,
    select_multivariate_medoid,
    write_source_outputs,
)
from models.fourier_reconstruction import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
)
from scripts.visualize_shift_configs_4tasks import (
    fit_class_projections,
    load_classes,
    load_raw_spatial_encoder,
    project_dataset,
    reconstruct_windows_batched,
)


DOMAINS = {
    "AT1": "austria/33UVP/2017",
    "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def _checkpoint_paths(value):
    supplied = Path(value)
    checkpoint = supplied if supplied.name == "model.pt" else supplied / "fold_0" / "model.pt"
    config = checkpoint.parent.parent / "train_config.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"source config not found: {config}")
    return checkpoint, config


def build_source_split(data_root, source_path, classes, seed, config):
    """Replay the first fold-0 shuffle used for the source side of configuration 05."""
    from dataset import PixelSetData
    from torchvision.transforms import transforms
    from transforms import Normalize, ToTensor

    base = PixelSetData(data_root, source_path, classes, closed_set=True)
    eligible = {source_path: base.get_parcel_indices().tolist()}
    split = replay_train_indices(
        (source_path,), eligible, seed,
        float(config.get("val_ratio", 0.1)), float(config.get("test_ratio", 0.2)),
    )[source_path]["train"]
    return PixelSetData(
        data_root=data_root,
        dataset_name=source_path,
        classes=classes,
        indices=split,
        transform=transforms.Compose([Normalize(), ToTensor()]),
        with_extra=bool(config.get("with_extra", False)),
        closed_set=True,
        combine_spring_and_winter=False,
    )


def _detector_settings(manifest):
    capture = manifest["structure_capture"]
    segment = capture["segment"]
    coarse = capture["coarse"]
    fine = {
        "min_distance_days": segment["min_distance_days"],
        "member_min_width_days": segment["member_min_width_days"],
        "member_min_relative_prominence": segment["member_min_relative_prominence"],
        "member_min_domain_prominence": segment["member_min_domain_prominence"],
        "min_duration_days": segment["min_duration_days"],
        "max_duration_days": segment["max_duration_days"],
        "min_domain_change": segment["min_domain_change"],
        "min_curve_change": segment["min_curve_change"],
    }
    coarse_settings = {
        "max_reversal_ratio": coarse["max_reversal_ratio"],
        "max_reversal_domain_change": coarse["max_reversal_domain_change"],
        "max_reversal_duration_days": coarse["max_reversal_duration_days"],
        "max_merge_depth": coarse["max_merge_depth"],
        "min_duration_days": coarse["min_duration_days"],
        "max_duration_days": coarse["max_duration_days"],
        "min_curve_change": coarse["min_curve_change"],
        "min_domain_change": coarse["min_domain_change"],
        "min_monotonicity": coarse["min_monotonicity"],
    }
    return fine, coarse_settings


def _detect_coarse(
    curve, baseline, fine_settings, coarse_settings,
    event_prefix, fine_prefix, coarse_prefix,
):
    _, chain, fine = detect_structure_segments(
        curve, np.arange(365.0), baseline,
        event_prefix=event_prefix, segment_prefix=fine_prefix, **fine_settings,
    )
    return build_coarse_structure(
        chain, fine, curve, baseline.iqr, segment_prefix=coarse_prefix,
        **coarse_settings,
    ).segments


def _load_reference_rows(view_root, reference_task):
    folder = resolve_visualization_config_dir(
        view_root, "05_reconshift13_structure_segments", reference_task
    )
    manifest_path = folder / "manifest.json"
    rows_path = folder / "source_coarse_segments.csv"
    if not manifest_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError(
            f"configuration-05 inputs missing: {manifest_path}, {rows_path}"
        )
    return folder, _read_json(manifest_path), _read_csv(rows_path)


def _attach_configuration05_reference(computed, rows, class_id):
    saved = {
        row["coarse_segment_id"]: row for row in rows
        if int(row["class_id"]) == int(class_id)
    }
    attached = []
    for item in computed:
        row = saved.get(item.coarse_segment_id)
        if row is None:
            raise RuntimeError(
                f"configuration-05 reference id mismatch for class {class_id}: {item.coarse_segment_id}"
            )
        for field in ("center_day", "duration_days"):
            if not np.isclose(float(row[field]), float(getattr(item, field)), atol=1e-5):
                raise RuntimeError(
                    f"configuration-05 {field} mismatch for {item.coarse_segment_id}"
                )
        attached.append(replace(
            item,
            accepted=_bool(row["accepted"]),
            rejection_reason=row.get("rejection_reason", ""),
            source_occurrence_rate=float(row["source_occurrence_rate"]),
            source_center_timing_mad_days=float(row["source_center_timing_mad_days"]),
            source_duration_ratio_median=float(row["source_duration_ratio_median"]),
        ))
    if set(saved) != {item.coarse_segment_id for item in computed}:
        raise RuntimeError(f"configuration-05 reference count mismatch for class {class_id}")
    return tuple(attached)


def _safe_name(value):
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def run_source(args):
    alias = args.source_domain.upper()
    if alias not in DOMAINS:
        raise ValueError(f"unknown source domain: {alias}")
    source_path = DOMAINS[alias]
    checkpoint, config_path = _checkpoint_paths(args.source_checkpoint)
    config = _read_json(config_path)
    classes = load_classes(checkpoint, source_path, args.data_root)
    source_set = build_source_split(args.data_root, source_path, classes, args.seed, config)
    device = torch.device(args.device)
    print(
        f"PROGRESS|source={alias}|stage=fit_fixed_raw_pse_pc1|samples={len(source_set)}",
        flush=True,
    )
    spatial = load_raw_spatial_encoder(checkpoint, config, classes, device)
    pca_grid = np.linspace(0.0, 365.0, args.grid_size)
    projections, latent_dim = fit_class_projections(
        spatial, source_set, len(classes), pca_grid, args.batch_size, device,
        bool(config.get("with_extra", False)),
    )
    analyzer = BatchedDirectFourierAnalyzer(
        num_modes=13, period_days=365.0, reg=0.001
    ).to(device)
    synthesizer = BatchedDirectFourierSynthesizer(
        num_modes=13, period_days=365.0
    ).to(device)
    print(f"PROGRESS|source={alias}|stage=extract_mode13_coefficients", flush=True)
    _, coefficients, _, baselines = project_dataset(
        spatial, source_set, projections, pca_grid, args.batch_size, device,
        bool(config.get("with_extra", False)), analyzer,
        collect_samplewise=True,
    )
    view_folder, view_manifest, saved_rows = _load_reference_rows(
        args.structure_view_root, args.reference_task
    )
    fine_settings, coarse_settings = _detector_settings(view_manifest)
    output = Path(args.output_root) / alias
    structure_rows, bootstrap_rows, medoid_counts = [], [], {}
    days = np.arange(365.0)

    for class_id, class_name in enumerate(classes):
        values = coefficients.get(class_id)
        if values is None or len(values) == 0 or class_id not in projections:
            continue
        print(
            f"PROGRESS|source={alias}|class={class_name}|stage=reconstruct_mode13|samples={len(values)}",
            flush=True,
        )
        query = np.tile(days, (len(values), 1))
        mode13 = reconstruct_windows_batched(
            values, query, synthesizer, device, args.reconstruction_batch_size
        )
        projection = projections[class_id]
        reference_multi = np.median(mode13, axis=0)
        reference_curve = projection.transform(reference_multi[None])[0]
        computed_reference = _detect_coarse(
            reference_curve, baselines[class_id], fine_settings, coarse_settings,
            "S", "SSEG", "SC",
        )
        reference = _attach_configuration05_reference(
            computed_reference, saved_rows, class_id
        )
        subsets = bootstrap_subsample_indices(
            len(mode13), args.prototype_bootstrap_repeats,
            args.prototype_bootstrap_fraction,
            args.prototype_bootstrap_seed + class_id,
        )
        bootstrap_curves, bootstrap_structures = [], []
        for repeat, indices in enumerate(subsets):
            if repeat % 10 == 0:
                print(
                    f"PROGRESS|source={alias}|class={class_name}|stage=bootstrap|repeat={repeat}/{len(subsets)}",
                    flush=True,
                )
            prototype_multi = np.median(mode13[indices], axis=0)
            curve = projection.transform(prototype_multi[None])[0]
            structures = _detect_coarse(
                curve, baselines[class_id], fine_settings, coarse_settings,
                f"B{repeat}E", f"B{repeat}F", f"B{repeat}C",
            )
            bootstrap_curves.append(curve)
            bootstrap_structures.append(structures)
            bootstrap_rows.append({
                "source_domain": alias, "class_id": class_id,
                "class_name": class_name, "repeat": repeat,
                "sample_count": len(indices),
                "num_coarse_candidates": len(structures),
                "num_coarse_accepted": sum(item.accepted for item in structures),
            })
        medoid = select_multivariate_medoid(
            mode13, args.medoid_max_samples,
            args.prototype_bootstrap_seed + 10000 + class_id,
        )
        medoid_curve = projection.transform(medoid.curve[None])[0]
        medoid_structures = _detect_coarse(
            medoid_curve, baselines[class_id], fine_settings, coarse_settings,
            "ME", "MF", "MC",
        )
        medoid_counts[(alias, class_id)] = sum(item.accepted for item in medoid_structures)
        rows = audit_reference_structures(
            alias, class_id, class_name, reference, bootstrap_structures,
            medoid_structures, args.prototype_match_center_radius_days,
            args.prototype_match_max_duration_ratio,
            args.prototype_high_stability_threshold,
            args.prototype_low_stability_threshold,
            args.individual_support_threshold,
        )
        structure_rows.extend(rows)
        name = f"{class_id:02d}_{_safe_name(class_name)}"
        plot_bootstrap_stability(
            output / "diagnostics" / f"{name}_bootstrap_stability.png",
            days, reference_curve, np.asarray(bootstrap_curves), reference,
            rows,
        )
        plot_median_vs_medoid(
            output / "diagnostics" / f"{name}_median_vs_medoid.png",
            days, reference_curve, medoid_curve, reference, medoid_structures,
        )
        del mode13

    class_rows = build_class_summary(structure_rows, medoid_counts)
    represented = {(row["source_domain"], row["class_id"]) for row in class_rows}
    for class_id, class_name in enumerate(classes):
        if (alias, class_id) not in represented:
            class_rows.append({
                "source_domain": alias, "class_id": class_id,
                "class_name": class_name, "num_reference_structures": 0,
                "num_robust_reference": 0,
                "num_stable_low_sample_support": 0,
                "num_intermediate_reference": 0,
                "num_unstable_reference": 0,
                "mean_individual_occurrence": float("nan"),
                "mean_bootstrap_occurrence": float("nan"),
                "medoid_num_structures": medoid_counts.get((alias, class_id), 0),
                "num_reference_present_in_medoid": 0,
            })
    class_rows.sort(key=lambda row: int(row["class_id"]))
    manifest = {
        "audit_type": "source_structure_reference_validity_06A",
        "source_domain": alias,
        "source_dataset": source_path,
        "source_checkpoint": str(checkpoint),
        "source_split": "fold0 first source shuffle",
        "source_sample_count": len(source_set),
        "class_count": len(classes),
        "latent_dim": latent_dim,
        "mode": 13,
        "pc1_fit": "full-source Raw-PSE interpolation; one fixed axis per class",
        "reference_prototype": "pointwise multivariate Mode13 median",
        "bootstrap": {
            "repeats": args.prototype_bootstrap_repeats,
            "fraction": args.prototype_bootstrap_fraction,
            "with_replacement": False,
            "seed": args.prototype_bootstrap_seed,
            "pc1_refit": False,
        },
        "medoid": {
            "distance": "mean over calendar days of multivariate Mode13 Euclidean distance",
            "max_samples": args.medoid_max_samples,
            "uses_pc1": False,
        },
        "matching": {
            "direction_equal": True,
            "center_radius_days": args.prototype_match_center_radius_days,
            "max_duration_ratio": args.prototype_match_max_duration_ratio,
            "one_to_one": True,
        },
        "diagnostic_thresholds": {
            "high": args.prototype_high_stability_threshold,
            "low": args.prototype_low_stability_threshold,
            "individual": args.individual_support_threshold,
        },
        "configuration05_folder": str(view_folder),
        "configuration05_modified": False,
        "model_training": False,
    }
    write_source_outputs(output, structure_rows, class_rows, bootstrap_rows, manifest)
    print(f"[FINISHED] 06A source={alias} output={output}", flush=True)


def summarize_root(root):
    root = Path(root)
    structures, classes = [], []
    for alias in DOMAINS:
        structures.extend(_read_csv(root / alias / "structure_stability.csv"))
        classes.extend(_read_csv(root / alias / "class_summary.csv"))
    for row in structures:
        row["bootstrap_occurrence_rate"] = float(row["bootstrap_occurrence_rate"])
        row["present_in_medoid"] = _bool(row["present_in_medoid"])
    total = build_total_summary(structures, classes)
    with (root / "summary_total.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(total))
        writer.writeheader()
        writer.writerow(total)
    print("STRUCTURE_REFERENCE_VALIDITY_TOTAL|" + "|".join(
        f"{key}={value}" for key, value in total.items()
    ), flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source-domain", choices=tuple(DOMAINS))
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--reference-task")
    parser.add_argument("--structure-view-root", type=Path, default=Path("outputs/shift_visualizations_seed1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/structure_reference_validity"))
    parser.add_argument("--summarize-root", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reconstruction-batch-size", type=int, default=128)
    parser.add_argument("--prototype-bootstrap-repeats", type=int, default=100)
    parser.add_argument("--prototype-bootstrap-fraction", type=float, default=0.70)
    parser.add_argument("--prototype-bootstrap-seed", type=int, default=1)
    parser.add_argument("--prototype-match-center-radius-days", type=float, default=30)
    parser.add_argument("--prototype-match-max-duration-ratio", type=float, default=2.0)
    parser.add_argument("--prototype-high-stability-threshold", type=float, default=0.80)
    parser.add_argument("--prototype-low-stability-threshold", type=float, default=0.50)
    parser.add_argument("--individual-support-threshold", type=float, default=0.50)
    parser.add_argument("--medoid-max-samples", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.summarize_root is not None:
        summarize_root(args.summarize_root)
        return
    for field in ("data_root", "source_domain", "source_checkpoint", "reference_task"):
        if getattr(args, field) is None:
            raise SystemExit(f"ERROR: --{field.replace('_', '-')} is required")
    if not args.data_root.is_dir():
        raise SystemExit(f"ERROR: data root not found: {args.data_root}")
    run_source(args)


if __name__ == "__main__":
    main()
