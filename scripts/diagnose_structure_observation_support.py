#!/usr/bin/env python3
"""Run the source-only 06B observation-support and acquisition-masking audit."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.recon_structure_segments import build_coarse_structure, detect_structure_segments
from analysis.shift_visualization import resolve_visualization_config_dir
from analysis.structure_observation_support_diagnostic import (
    FAILURE_REASONS,
    audit_sample_matches,
    contiguous_gap_mask,
    delete_acquisitions,
    detector_state,
    join_reliable_references,
    observation_support_metrics,
    plot_masking_diagnostic,
    plot_support_audit,
    random_window_mask,
    select_masking_references,
    staged_output,
    summarize_boundaries,
    summarize_failure_reasons,
    summarize_masking,
    summarize_structures,
    write_csv,
    write_json,
)
from models.fourier_reconstruction import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
)
from scripts.diagnose_structure_reference_validity import (
    _checkpoint_paths,
    _detector_settings,
    _read_csv,
    _read_json,
    build_source_split,
)
from scripts.visualize_shift_configs_4tasks import (
    fit_class_projections,
    load_classes,
    load_raw_spatial_encoder,
    project_dataset,
    reconstruct_class_prototype,
    reconstruct_projected_coefficients,
)


TASKS = {
    "AT1_DK1": ("AT1", "austria/33UVP/2017"),
    "DK1_FR1": ("DK1", "denmark/32VNH/2017"),
    "FR1_FR2": ("FR1", "france/30TXT/2017"),
    "FR2_AT1": ("FR2", "france/31TCJ/2017"),
}

SAMPLE_FIELDS = (
    "task", "source_domain", "class_id", "class_name",
    "reference_structure_id", "direction", "reference_start_day",
    "reference_end_day", "reference_center_day", "reference_duration_days",
    "crosses_year_boundary", "bootstrap_occurrence_rate",
    "individual_occurrence_rate", "sample_id", "matched",
    "matched_sample_segment_id", "num_observations", "observation_density",
    "max_observation_gap_days", "nearest_start_observation_days",
    "nearest_end_observation_days", "quarter_coverage",
    "support_coverage_radius", "failure_reason", "detector_state",
    "nearest_same_direction_center_distance",
    "nearest_same_direction_duration_ratio",
)

MASK_FIELDS = (
    "task", "source_domain", "class_id", "class_name",
    "reference_structure_id", "crosses_year_boundary", "sample_id",
    "mask_type", "mask_level", "repeat", "seed", "baseline_num_obs",
    "masked_num_obs", "baseline_max_gap", "masked_max_gap",
    "baseline_support_coverage", "masked_support_coverage",
    "baseline_matched", "masked_matched", "failure_reason", "valid_mask_run",
)

STRUCTURE_SUMMARY_FIELDS = (
    "task", "source_domain", "class_id", "class_name",
    "reference_structure_id", "direction", "start_day", "end_day",
    "duration_days", "crosses_year_boundary", "bootstrap_occurrence_rate",
    "individual_occurrence_rate", "num_samples", "num_matched",
    "num_unmatched", "matched_rate", "Gplus_median_num_obs",
    "Gminus_median_num_obs", "delta_num_obs", "cliffs_delta_num_obs",
    "Gplus_median_max_gap", "Gminus_median_max_gap", "delta_max_gap",
    "cliffs_delta_max_gap", "Gplus_median_support_coverage",
    "Gminus_median_support_coverage", "delta_support_coverage",
    "cliffs_delta_support_coverage", "Gplus_median_nearest_start",
    "Gminus_median_nearest_start", "delta_nearest_start",
    "Gplus_median_nearest_end", "Gminus_median_nearest_end",
    "delta_nearest_end", "dominant_failure_reason",
)

FAILURE_SUMMARY_FIELDS = (
    "scope", "task", "class_id", "class_name", "reference_structure_id",
    "num_unmatched",
) + tuple(
    field for reason in FAILURE_REASONS
    for field in (reason, f"fraction_{reason}")
)

BOUNDARY_SUMMARY_FIELDS = (
    "cross_boundary", "num_structures", "num_samples", "matched_rate",
    "mean_structure_matched_rate", "median_structure_matched_rate",
    "median_num_obs", "median_max_gap", "median_support_coverage",
) + FAILURE_REASONS

MASK_SUMMARY_FIELDS = (
    "scope", "task", "source_domain", "class_id", "class_name",
    "reference_structure_id", "mask_type", "mask_level",
    "crosses_year_boundary", "num_samples", "num_runs", "num_valid_runs",
    "recovery_rate", "loss_rate", "median_num_obs_after",
    "median_max_gap_after", "median_support_coverage_after",
    "dominant_failure_reason",
)

REQUIRED_OUTPUTS = (
    "sample_structure_support.csv", "structure_summary.csv",
    "failure_reason_summary.csv", "boundary_summary.csv",
    "masked_ablation.csv", "masked_ablation_summary.csv", "manifest.json",
)


def _bool(value):
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}


def _parse_numbers(value, cast=float):
    return tuple(cast(part.strip()) for part in str(value).split(",") if part.strip())


def _safe_name(value):
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in str(value))


def _reference_namespace(row):
    return SimpleNamespace(
        coarse_segment_id=str(row["coarse_segment_id"]),
        direction=str(row["direction"]),
        start_day=float(row["start_day"]),
        end_day=float(row["end_day"]),
        center_day=float(row["center_day"]),
        duration_days=float(row["duration_days"]),
        accepted=True,
    )


def _load_inputs(args, source_domain):
    folder = resolve_visualization_config_dir(
        args.structure_view_root, "05_reconshift13_structure_segments", args.task
    )
    paths = {
        "manifest": folder / "manifest.json",
        "coarse": folder / "source_coarse_segments.csv",
        "classes": folder / "class_summary.csv",
        "validity": args.validity_root / source_domain / "structure_stability.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("06B input missing: " + ", ".join(missing))
    manifest = _read_json(paths["manifest"])
    coarse = _read_csv(paths["coarse"])
    for row in coarse:
        row["source_domain"] = source_domain
    accepted = [row for row in coarse if _bool(row["accepted"])]
    validity = _read_csv(paths["validity"])
    reliable = join_reliable_references(
        accepted, validity, args.min_bootstrap_occurrence
    )
    return folder, manifest, coarse, _read_csv(paths["classes"]), reliable


def _detect(curve, baseline, fine_settings, coarse_settings, prefix):
    _, chain, fine = detect_structure_segments(
        curve,
        np.arange(365.0),
        baseline,
        event_prefix=f"{prefix}E",
        segment_prefix=f"{prefix}F",
        **fine_settings,
    )
    coarse = build_coarse_structure(
        chain,
        fine,
        curve,
        baseline.iqr,
        segment_prefix=f"{prefix}C",
        **coarse_settings,
    )
    return chain, coarse.segments


def _validate_computed_references(computed, saved_rows, class_id):
    saved = {
        str(row["coarse_segment_id"]): row
        for row in saved_rows
        if int(row["class_id"]) == int(class_id)
    }
    current = {str(item.coarse_segment_id): item for item in computed}
    if set(saved) != set(current):
        raise RuntimeError(
            f"configuration-05 reference id mismatch for class {class_id}: "
            f"saved={sorted(saved)}, computed={sorted(current)}"
        )
    for structure_id, item in current.items():
        row = saved[structure_id]
        for name in ("start_day", "end_day", "center_day", "duration_days"):
            if not np.isclose(float(row[name]), float(getattr(item, name)), atol=1e-5):
                raise RuntimeError(
                    f"configuration-05 {name} mismatch for class {class_id} {structure_id}"
                )
    return tuple(item for item in computed if item.accepted)


def _sample_row(task, source_domain, class_name, reference_row, sample_id, support, audit):
    return {
        "task": task,
        "source_domain": source_domain,
        "class_id": int(reference_row["class_id"]),
        "class_name": class_name,
        "reference_structure_id": reference_row["coarse_segment_id"],
        "direction": reference_row["direction"],
        "reference_start_day": float(reference_row["start_day"]),
        "reference_end_day": float(reference_row["end_day"]),
        "reference_center_day": float(reference_row["center_day"]),
        "reference_duration_days": float(reference_row["duration_days"]),
        "crosses_year_boundary": support.crosses_year_boundary,
        "bootstrap_occurrence_rate": float(reference_row["bootstrap_occurrence_rate"]),
        "individual_occurrence_rate": float(reference_row["individual_occurrence_rate"]),
        "sample_id": int(sample_id),
        "matched": audit.matched,
        "matched_sample_segment_id": audit.matched_sample_segment_id,
        "num_observations": support.num_observations,
        "observation_density": support.observation_density,
        "max_observation_gap_days": support.max_observation_gap_days,
        "nearest_start_observation_days": support.nearest_start_observation_days,
        "nearest_end_observation_days": support.nearest_end_observation_days,
        "quarter_coverage": support.quarter_coverage,
        "support_coverage_radius": support.support_coverage_radius,
        "failure_reason": audit.failure_reason,
        "detector_state": audit.detector_state,
        "nearest_same_direction_center_distance": audit.nearest_same_direction_center_distance,
        "nearest_same_direction_duration_ratio": audit.nearest_same_direction_duration_ratio,
    }


@torch.inference_mode()
def _masked_curves(
    variants,
    spatial_encoder,
    analyzer,
    synthesizer,
    projection,
    device,
    with_extra,
):
    """Rerun PSE for every masked input, batching variants with equal time length."""
    grouped = defaultdict(list)
    for index, variant in enumerate(variants):
        grouped[len(variant["positions"])].append((index, variant))
    output = [None] * len(variants)
    daily = torch.arange(365.0, device=device)
    for group in grouped.values():
        pixels = torch.stack([item["pixels"] for _, item in group]).to(device)
        masks = torch.stack([item["valid_pixels"] for _, item in group]).to(device)
        positions = torch.stack([item["positions"] for _, item in group]).to(device)
        extra = (
            torch.stack([item["extra"] for _, item in group]).to(device)
            if with_extra else None
        )
        features = spatial_encoder(pixels, masks, extra)
        coefficients, _ = analyzer(features, positions, collect_diagnostics=False)
        query = daily.to(dtype=features.dtype).unsqueeze(0).expand(len(group), -1)
        reconstructed = synthesizer(coefficients, query).cpu().numpy()
        projected = projection.transform(reconstructed)
        for (output_index, _), curve in zip(group, projected):
            output[output_index] = curve
    return output


def _mask_seed(base, class_id, structure_order, sample_order, level_order, repeat):
    return int(base + class_id * 10_000_000 + structure_order * 100_000 + sample_order * 1_000 + level_order * 10 + repeat)


def _run_masking(
    args,
    task,
    source_domain,
    source_set,
    observational_rows,
    reliable_rows,
    matching_references,
    baselines,
    projections,
    fine_settings,
    coarse_settings,
    matching_radius,
    matching_duration_ratio,
    spatial_encoder,
    analyzer,
    synthesizer,
    with_extra,
):
    selected = select_masking_references(reliable_rows)
    rows, examples = [], defaultdict(list)
    dataset_index = {
        int(parcel_index): index
        for index, (_path, parcel_index, _label, _extra) in enumerate(source_set.samples)
    }
    baseline_by_key = {
        (int(row["class_id"]), str(row["reference_structure_id"]), int(row["sample_id"])): row
        for row in observational_rows
    }
    class_names = {int(row["class_id"]): row["class_name"] for row in reliable_rows}
    random_fractions = _parse_numbers(args.mask_random_fractions)
    gap_days = _parse_numbers(args.mask_gap_days)

    for structure_order, reference_row in enumerate(selected):
        class_id = int(reference_row["class_id"])
        structure_id = str(reference_row["coarse_segment_id"])
        reference = next(
            item for item in matching_references[class_id]
            if str(item.coarse_segment_id) == structure_id
        )
        plus = sorted(
            (
                row for row in observational_rows
                if int(row["class_id"]) == class_id
                and str(row["reference_structure_id"]) == structure_id
                and _bool(row["matched"])
            ),
            key=lambda row: int(row["sample_id"]),
        )
        if len(plus) > args.mask_max_samples_per_structure:
            rng = np.random.default_rng(args.mask_seed + class_id + structure_order * 1000)
            selected_indices = np.sort(rng.choice(
                len(plus), args.mask_max_samples_per_structure, replace=False
            ))
            plus = [plus[index] for index in selected_indices]

        for sample_order, baseline in enumerate(plus):
            sample_id = int(baseline["sample_id"])
            sample = source_set[dataset_index[sample_id]]
            positions = sample["positions"].cpu().numpy()
            variant_specs, variants = [], []
            levels = [("random", value) for value in random_fractions] + [("gap", value) for value in gap_days]
            for level_order, (mask_type, level) in enumerate(levels):
                repeats = args.mask_random_repeats if mask_type == "random" else args.mask_gap_repeats
                for repeat in range(repeats):
                    seed = _mask_seed(args.mask_seed, class_id, structure_order, sample_order, level_order, repeat)
                    if mask_type == "random":
                        removal = random_window_mask(
                            positions, reference.start_day, reference.end_day, level, seed
                        )
                    else:
                        removal = contiguous_gap_mask(
                            positions, reference.start_day, reference.end_day, level, seed
                        )
                    invalid_reason = ""
                    if removal is None:
                        invalid_reason = "GAP_EXCEEDS_WINDOW"
                    elif mask_type == "random" and not removal.any():
                        invalid_reason = "NO_WINDOW_OBSERVATION"
                    elif len(positions) - int(removal.sum()) < args.minimum_timepoints:
                        invalid_reason = "INVALID_MASK"
                    if invalid_reason:
                        rows.append({
                            "task": task, "source_domain": source_domain,
                            "class_id": class_id, "class_name": class_names[class_id],
                            "reference_structure_id": structure_id,
                            "crosses_year_boundary": _bool(reference_row["crosses_year_boundary"]),
                            "sample_id": sample_id, "mask_type": mask_type,
                            "mask_level": level, "repeat": repeat, "seed": seed,
                            "baseline_num_obs": baseline["num_observations"],
                            "masked_num_obs": float("nan"),
                            "baseline_max_gap": baseline["max_observation_gap_days"],
                            "masked_max_gap": float("nan"),
                            "baseline_support_coverage": baseline["support_coverage_radius"],
                            "masked_support_coverage": float("nan"),
                            "baseline_matched": True, "masked_matched": False,
                            "failure_reason": invalid_reason, "valid_mask_run": False,
                        })
                        continue
                    variant_specs.append((mask_type, level, repeat, seed, removal))
                    variants.append(delete_acquisitions(sample, removal))

            curves = _masked_curves(
                variants, spatial_encoder, analyzer, synthesizer,
                projections[class_id], device=next(spatial_encoder.parameters()).device,
                with_extra=with_extra,
            ) if variants else []
            for curve, (mask_type, level, repeat, seed, removal), variant in zip(curves, variant_specs, variants):
                chain, coarse = _detect(
                    curve, baselines[class_id], fine_settings, coarse_settings,
                    f"M{structure_order}_{sample_order}_{mask_type}_{level}_{repeat}_",
                )
                state = detector_state(len(chain), coarse)
                audit = audit_sample_matches(
                    matching_references[class_id], coarse,
                    matching_radius, matching_duration_ratio, state,
                )[structure_id]
                masked_positions = variant["positions"].cpu().numpy()
                support = observation_support_metrics(
                    masked_positions, reference.start_day, reference.end_day,
                    args.observation_support_radius_days,
                )
                rows.append({
                    "task": task, "source_domain": source_domain,
                    "class_id": class_id, "class_name": class_names[class_id],
                    "reference_structure_id": structure_id,
                    "crosses_year_boundary": support.crosses_year_boundary,
                    "sample_id": sample_id, "mask_type": mask_type,
                    "mask_level": level, "repeat": repeat, "seed": seed,
                    "baseline_num_obs": baseline["num_observations"],
                    "masked_num_obs": support.num_observations,
                    "baseline_max_gap": baseline["max_observation_gap_days"],
                    "masked_max_gap": support.max_observation_gap_days,
                    "baseline_support_coverage": baseline["support_coverage_radius"],
                    "masked_support_coverage": support.support_coverage_radius,
                    "baseline_matched": True, "masked_matched": audit.matched,
                    "failure_reason": audit.failure_reason,
                    "valid_mask_run": True,
                })
                if sample_order == 0 and mask_type == "random" and repeat == 0:
                    examples[(class_id, structure_id)].append({
                        "grid": np.arange(365.0), "curve": curve,
                        "label": f"random {level:g} ({'kept' if audit.matched else 'lost'})",
                    })
    return rows, selected, examples


def run_task(args):
    if args.task not in TASKS:
        raise ValueError(f"unknown task: {args.task}")
    source_domain, source_path = TASKS[args.task]
    checkpoint, config_path = _checkpoint_paths(args.source_checkpoint)
    config = _read_json(config_path)
    classes = load_classes(checkpoint, source_path, args.data_root)
    source_set = build_source_split(
        args.data_root, source_path, classes, args.seed, config
    )
    folder, manifest05, saved_coarse, class_rows05, reliable = _load_inputs(
        args, source_domain
    )
    audited_class_ids = {int(row["class_id"]) for row in class_rows05}
    reliable = [row for row in reliable if int(row["class_id"]) in audited_class_ids]
    device = torch.device(args.device)
    spatial_encoder = load_raw_spatial_encoder(checkpoint, config, classes, device)
    grid = np.linspace(0.0, 365.0, args.grid_size)
    projections, latent_dim = fit_class_projections(
        spatial_encoder, source_set, len(classes), grid, args.batch_size,
        device, bool(config.get("with_extra", False)),
    )
    analyzer = BatchedDirectFourierAnalyzer(13, period_days=365.0, reg=0.001).to(device)
    synthesizer = BatchedDirectFourierSynthesizer(13, period_days=365.0).to(device)
    _, coefficients, records, baselines = project_dataset(
        spatial_encoder, source_set, projections, grid, args.batch_size, device,
        bool(config.get("with_extra", False)), analyzer, collect_samplewise=True,
    )
    fine_settings, coarse_settings = _detector_settings(manifest05)
    stability = manifest05["structure_capture"]["coarse"]["source_stability"]
    matching_radius = float(stability["occurrence_radius_days"])
    matching_duration_ratio = float(stability["max_duration_ratio"])
    reliable_by_class = defaultdict(list)
    for row in reliable:
        reliable_by_class[int(row["class_id"])].append(row)

    observational_rows, sample_items, matching_references = [], defaultdict(list), {}
    reference_curves = {}
    for class_id in sorted(reliable_by_class):
        if class_id not in coefficients or class_id not in projections:
            raise RuntimeError(f"reliable reference class has no source samples: {class_id}")
        prototype = reconstruct_class_prototype(
            coefficients[class_id], synthesizer, device,
            sample_batch_size=args.reconstruction_batch_size,
        )
        reference_curve = projections[class_id].transform(prototype[None])[0]
        chain, computed = _detect(
            reference_curve, baselines[class_id], fine_settings, coarse_settings,
            "S",
        )
        del chain
        matching_references[class_id] = _validate_computed_references(
            computed, saved_coarse, class_id
        )
        reference_curves[class_id] = reference_curve
        curves = reconstruct_projected_coefficients(
            coefficients[class_id], projections[class_id], np.arange(365.0)
        )
        class_records = records[class_id]
        if len(curves) != len(class_records):
            raise RuntimeError(f"sample order mismatch for class {class_id}")
        by_reference_id = {
            str(row["coarse_segment_id"]): row for row in reliable_by_class[class_id]
        }
        for sample_index, (curve, record) in enumerate(zip(curves, class_records)):
            chain, coarse = _detect(
                curve, baselines[class_id], fine_settings, coarse_settings,
                f"I{sample_index}_",
            )
            state = detector_state(len(chain), coarse)
            audits = audit_sample_matches(
                matching_references[class_id], coarse,
                matching_radius, matching_duration_ratio, state,
            )
            sample_id = int(record["sample_id"])
            sample_items[class_id].append({
                "sample_id": sample_id,
                "curve": curve,
                "positions": np.asarray(record["positions"], dtype=np.float64),
            })
            for structure_id, reference_row in by_reference_id.items():
                reference = next(
                    item for item in matching_references[class_id]
                    if str(item.coarse_segment_id) == structure_id
                )
                support = observation_support_metrics(
                    record["positions"], reference.start_day, reference.end_day,
                    args.observation_support_radius_days,
                )
                observational_rows.append(_sample_row(
                    args.task, source_domain, classes[class_id], reference_row,
                    sample_id, support, audits[structure_id],
                ))

    structure_summary = summarize_structures(observational_rows)
    failure_summary = summarize_failure_reasons(observational_rows)
    boundary_summary = summarize_boundaries(observational_rows)
    masking_rows, masking_selection, masking_examples = [], [], {}
    if args.run_observation_masking_ablation:
        masking_rows, masking_selection, masking_examples = _run_masking(
            args, args.task, source_domain, source_set, observational_rows,
            reliable, matching_references, baselines, projections,
            fine_settings, coarse_settings, matching_radius,
            matching_duration_ratio, spatial_encoder, analyzer, synthesizer,
            bool(config.get("with_extra", False)),
        )
    masking_summary = summarize_masking(masking_rows)

    output = args.output_root / args.task
    support_plots = [
        f"diagnostics/{int(row['class_id']):02d}_{_safe_name(classes[int(row['class_id'])])}_{row['coarse_segment_id']}_support_audit.png"
        for row in reliable if float(row["individual_occurrence_rate"]) < 0.6
    ]
    masking_plots = [
        f"masking_diagnostics/{int(row['class_id']):02d}_{_safe_name(classes[int(row['class_id'])])}_{row['coarse_segment_id']}_masking.png"
        for row in masking_selection
    ]
    with staged_output(
        output, REQUIRED_OUTPUTS + tuple(support_plots) + tuple(masking_plots)
    ) as staging:
        write_csv(staging / "sample_structure_support.csv", observational_rows, SAMPLE_FIELDS)
        write_csv(staging / "structure_summary.csv", structure_summary, STRUCTURE_SUMMARY_FIELDS)
        write_csv(staging / "failure_reason_summary.csv", failure_summary, FAILURE_SUMMARY_FIELDS)
        write_csv(staging / "boundary_summary.csv", boundary_summary, BOUNDARY_SUMMARY_FIELDS)
        write_csv(staging / "masked_ablation.csv", masking_rows, MASK_FIELDS)
        write_csv(staging / "masked_ablation_summary.csv", masking_summary, MASK_SUMMARY_FIELDS)
        diagnostics = staging / "diagnostics"
        masking_diagnostics = staging / "masking_diagnostics"
        diagnostics.mkdir()
        masking_diagnostics.mkdir()
        for row in reliable:
            if float(row["individual_occurrence_rate"]) >= 0.6:
                continue
            class_id = int(row["class_id"])
            structure_id = str(row["coarse_segment_id"])
            reference = next(
                item for item in matching_references[class_id]
                if str(item.coarse_segment_id) == structure_id
            )
            group = [
                item for item in observational_rows
                if int(item["class_id"]) == class_id
                and str(item["reference_structure_id"]) == structure_id
            ]
            plot_support_audit(
                diagnostics / f"{class_id:02d}_{_safe_name(classes[class_id])}_{structure_id}_support_audit.png",
                np.arange(365.0), reference_curves[class_id],
                sample_items[class_id], group, reference,
            )
        for row in masking_selection:
            class_id = int(row["class_id"])
            structure_id = str(row["coarse_segment_id"])
            group = [
                item for item in masking_rows
                if int(item["class_id"]) == class_id
                and str(item["reference_structure_id"]) == structure_id
            ]
            examples = [{
                "grid": np.arange(365.0),
                "curve": reference_curves[class_id],
                "label": "reference",
            }]
            if group:
                baseline_sample_id = min(int(item["sample_id"]) for item in group)
                baseline_item = next(
                    item for item in sample_items[class_id]
                    if int(item["sample_id"]) == baseline_sample_id
                )
                examples.append({
                    "grid": np.arange(365.0),
                    "curve": baseline_item["curve"],
                    "label": "baseline G+ sample",
                })
            examples.extend(masking_examples.get((class_id, structure_id), ()))
            plot_masking_diagnostic(
                masking_diagnostics / f"{class_id:02d}_{_safe_name(classes[class_id])}_{structure_id}_masking.png",
                group, examples,
            )
        write_json(staging / "manifest.json", {
            "experiment": {"name": "structure_observation_support_06B"},
            "task": args.task,
            "source_domain": source_domain,
            "source_dataset": source_path,
            "source_checkpoint": str(checkpoint),
            "source_sample_count": len(source_set),
            "reference": {
                "source": "05_reconshift13_structure_segments",
                "folder": str(folder),
                "validity_source": "structure_reference_validity_06A",
                "min_bootstrap_occurrence": args.min_bootstrap_occurrence,
                "num_reliable_structures": len(reliable),
            },
            "observation_support": {
                "radius_days": args.observation_support_radius_days,
                "quarter_bins": 4,
                "timestamp_source": "PixelSetData positions",
                "valid_pixels_is_time_mask": False,
            },
            "matching": {
                "inherited_from_05": True,
                "one_joint_greedy_assignment_per_sample_class": True,
                "center_radius_days": matching_radius,
                "max_duration_ratio": matching_duration_ratio,
            },
            "masking": {
                "enabled": args.run_observation_masking_ablation,
                "selected_structures": masking_selection,
                "random_fractions": _parse_numbers(args.mask_random_fractions),
                "random_repeats": args.mask_random_repeats,
                "contiguous_gap_days": _parse_numbers(args.mask_gap_days),
                "gap_repeats": args.mask_gap_repeats,
                "max_samples_per_structure": args.mask_max_samples_per_structure,
                "seed": args.mask_seed,
                "deletion_rounding": "ceil",
                "deletes_pixels_valid_pixels_positions": True,
                "reruns_pse": True,
            },
            "model": {"frozen": True, "pc1_refit": False, "mode": 13},
            "latent_dim": latent_dim,
            "target_used": False,
            "training": False,
            "structure_completion": False,
        })
    print(f"[FINISHED] 06B {args.task}: {output}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose-structure-observation-support", action="store_true")
    parser.add_argument("--run-observation-masking-ablation", action="store_true")
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--structure-view-root", type=Path, default=Path("outputs/shift_visualizations_seed1"))
    parser.add_argument("--validity-root", type=Path, default=Path("outputs/structure_reference_validity"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/shift_visualizations_seed1/06B_structure_observation_support"))
    parser.add_argument("--min-bootstrap-occurrence", type=float, default=0.8)
    parser.add_argument("--observation-support-radius-days", type=float, default=15)
    parser.add_argument("--mask-random-fractions", default="0.25,0.50,0.75")
    parser.add_argument("--mask-random-repeats", type=int, default=5)
    parser.add_argument("--mask-gap-days", default="30,60,90")
    parser.add_argument("--mask-gap-repeats", type=int, default=5)
    parser.add_argument("--mask-max-samples-per-structure", type=int, default=128)
    parser.add_argument("--mask-seed", type=int, default=1)
    parser.add_argument("--minimum-timepoints", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reconstruction-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.diagnose_structure_observation_support:
        raise SystemExit("ERROR: --diagnose-structure-observation-support is required")
    for name in ("task", "data_root", "source_checkpoint"):
        if getattr(args, name) is None:
            raise SystemExit(f"ERROR: --{name.replace('_', '-')} is required")
    if not args.data_root.is_dir():
        raise SystemExit(f"ERROR: data root not found: {args.data_root}")
    run_task(args)


if __name__ == "__main__":
    main()
