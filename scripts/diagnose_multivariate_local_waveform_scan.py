#!/usr/bin/env python3
"""07A: source-only local waveform ranking on fixed Mode13 structure proposals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.multivariate_local_waveform_scan_diagnostic import (
    REQUIRED_TASK_FILES, candidate_pool, cluster_bootstrap,
    compare_candidate_waveforms, endpoint_relative, endpoint_relative_ned,
    evaluate_mode13, metric_summary,
    publish_revision, relative_queries, sample_periodic_curve, structure_summary,
    template_pc1_consistency_error, transition_summary,
)
from analysis.structure_identity_phase_diagnostic import descriptor, structural_cost
from analysis.structure_observation_support_diagnostic import staged_output, write_csv, write_json

TASKS = {"AT1_DK1": ("AT1", "austria/33UVP/2017"), "DK1_FR1": ("DK1", "denmark/32VNH/2017"),
         "FR1_FR2": ("FR1", "france/30TXT/2017"), "FR2_AT1": ("FR2", "france/31TCJ/2017")}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(value):
    return value is True or str(value).strip().lower() in ("true", "1")


def _unique_index(rows, fields, name):
    output = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        if key in output:
            raise ValueError(f"duplicate {name} key: {key}")
        output[key] = row
    return output


def flatten_ranking(prefix, ranking):
    if prefix == "handcrafted_local":
        names = dict(best_id="handcrafted_local_best_id", exact="handcrafted_local_exact",
                     unique_best="handcrafted_local_unique_best", positive_rank="handcrafted_positive_rank",
                     positive_distance="handcrafted_positive_distance", best_negative_distance="handcrafted_best_negative_distance",
                     margin="handcrafted_margin", pairwise_wins="handcrafted_pairwise_wins",
                     pairwise_total="handcrafted_pairwise_total")
    else:
        names = {key: f"{prefix}_{key}" for key in ranking}
    return {names.get(key, f"{prefix}_{key}"): value for key, value in ranking.items()}


def _plot_case(path, prototype_pc1, sample_pc1, reference, baseline, chosen, wrong,
               reference_pc1, waveforms_pc1, reference_multi, waveforms_multi, row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    axes[0].plot(np.arange(365), prototype_pc1, color="#245580", lw=2, label="reference prototype")
    axes[0].plot(np.arange(365), sample_pc1, color="#be6900", lw=1.3, label=f"sample {row['sample_id']}")
    colors = ((reference, "#245580", "reference"), (baseline, "#2ca02c", "06B correct"),
              (chosen, "#d62728", "07A Multi"), (wrong, "#9467bd", "06C chosen"))
    for item, color, label in colors:
        if item is None:
            continue
        a, b = float(item["start_day"]), float(item["end_day"])
        spans = ((a, b),) if b > a else ((a, 365), (0, b))
        for left, right in spans:
            axes[0].axvspan(left, right, color=color, alpha=.12, label=label if left == spans[0][0] else None)
    axes[0].set(xlim=(0, 365), ylabel="Fixed source-class PC1",
                title="Calendar view — 07A identity score does NOT use calendar position")
    axes[0].legend(ncol=3, fontsize=8); axes[0].grid(alpha=.2)
    u = np.linspace(0, 1, len(reference_pc1))
    axes[1].plot(u, reference_pc1, lw=2, label="reference")
    for sid, label, color in ((row["baseline_segment_id"], "06B correct", "#2ca02c"),
                              (row["multi_ned_best_id"], "07A Multi", "#d62728"),
                              (row["06c_final_segment_id"], "06C chosen", "#9467bd")):
        if sid in waveforms_pc1:
            axes[1].plot(u, waveforms_pc1[sid], color=color, label=label)
    axes[1].set(xlabel="Relative phase u", ylabel="PC1 waveform"); axes[1].legend(); axes[1].grid(alpha=.2)
    q = endpoint_relative(reference_multi)
    correct = endpoint_relative(waveforms_multi[row["baseline_segment_id"]])
    wrong_ids = [sid for sid in waveforms_multi if sid != row["baseline_segment_id"]]
    axes[2].plot(u, np.sqrt(np.mean((q - correct) ** 2, axis=1)), label="correct candidate mismatch")
    if wrong_ids:
        best_wrong = min(wrong_ids, key=lambda sid: (endpoint_relative_ned(reference_multi, waveforms_multi[sid]), sid))
        axes[2].plot(u, np.sqrt(np.mean((q - endpoint_relative(waveforms_multi[best_wrong])) ** 2, axis=1)), label="best wrong mismatch")
    axes[2].set(xlabel="Relative phase u", ylabel="Multivariate RMS mismatch",
                title=f"positive rank={row['multi_ned_positive_rank']}; margin={row['multi_ned_margin']:.4f}")
    axes[2].legend(); axes[2].grid(alpha=.2)
    fig.suptitle(f"{row['task']} | {row['class_name']} | {row['reference_structure_id']}")
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(path, dpi=140); plt.close(fig)


def _select_examples(rows, predicate, limit=20):
    pool = sorted((row for row in rows if predicate(row)),
                  key=lambda row: (int(row["class_id"]), str(row["reference_structure_id"]), int(row["sample_id"])))
    selected = []
    for key in sorted({(row["class_id"], row["reference_structure_id"]) for row in pool}):
        item = next((row for row in pool if (row["class_id"], row["reference_structure_id"]) == key), None)
        if item is not None and len(selected) < limit:
            selected.append(item)
    selected.extend(row for row in pool if row not in selected and len(selected) < limit)
    return selected[:limit]


def run_task(args):
    import torch
    from models.fourier_reconstruction import BatchedDirectFourierAnalyzer, BatchedDirectFourierSynthesizer
    from scripts.diagnose_structure_observation_support import _detect, _load_inputs, _validate_computed_references
    from scripts.diagnose_structure_reference_validity import _checkpoint_paths, _detector_settings, build_source_split
    from scripts.visualize_shift_configs_4tasks import (
        fit_class_projections, load_classes, load_raw_spatial_encoder, project_dataset,
        reconstruct_class_prototype, reconstruct_projected_coefficients,
    )
    if args.task not in TASKS:
        raise ValueError("--task is required")
    source, source_path = TASKS[args.task]
    checkpoint, config_path = _checkpoint_paths(args.source_checkpoint)
    config = read_json(config_path)
    classes = load_classes(checkpoint, source_path, args.data_root)
    dataset = build_source_split(args.data_root, source_path, classes, args.seed, config)
    folder05, manifest05, saved05, class_scope, reliable = _load_inputs(args, source)
    valid_classes = {int(row["class_id"]) for row in class_scope}
    reliable = [row for row in reliable if int(row["class_id"]) in valid_classes and float(row["bootstrap_occurrence_rate"]) >= .8]
    if not reliable:
        raise ValueError("no bootstrap>=0.8 references")
    baseline_path = args.observation_root / args.task / "sample_structure_support.csv"
    baseline_rows = [row for row in read_csv(baseline_path) if truth(row["matched"])]
    identity_packet = read_json(args.identity_root / "calibration.json")
    identity_rows = read_csv(args.identity_root / args.task / "sample_structure_identity.csv")
    calibrations = {direction: identity_packet["calibrations"][f"{source}:{direction}"] for direction in ("RISE", "FALL")}
    baseline_index = _unique_index(baseline_rows, ("source_domain", "class_id", "reference_structure_id", "sample_id"), "06B G+")
    identity_index = _unique_index(identity_rows, ("source_domain", "class_id", "reference_structure_id", "sample_id"), "06C")
    device = torch.device(args.device)
    encoder = load_raw_spatial_encoder(checkpoint, config, classes, device)
    grid = np.linspace(0., 365., args.grid_size)
    extra = bool(config.get("with_extra", False))
    with torch.inference_mode():
        projections, _ = fit_class_projections(encoder, dataset, len(classes), grid, args.batch_size, device, extra)
        analyzer = BatchedDirectFourierAnalyzer(13, period_days=365., reg=.001).to(device)
        synthesizer = BatchedDirectFourierSynthesizer(13, period_days=365.).to(device)
        _, coefficients, records, baselines = project_dataset(encoder, dataset, projections, grid, args.batch_size,
            device, extra, analyzer, collect_samplewise=True)
        fine_settings, coarse_settings = _detector_settings(manifest05)
        references_by_class = defaultdict(list)
        for row in reliable:
            references_by_class[int(row["class_id"])].append(row)
        all_rows, templates, contexts = [], [], {}
        for class_id, saved_refs in sorted(references_by_class.items()):
            prototype_multi = reconstruct_class_prototype(coefficients[class_id], synthesizer, device,
                                                           sample_batch_size=args.reconstruction_batch_size)
            projection = projections[class_id]
            prototype_pc1 = projection.transform(prototype_multi[None])[0]
            _, computed = _detect(prototype_pc1, baselines[class_id], fine_settings, coarse_settings, "S")
            _validate_computed_references(computed, saved05, class_id)
            geometry = {row.coarse_segment_id: asdict(row) for row in computed}
            sample_pc1_curves = reconstruct_projected_coefficients(coefficients[class_id], projection, np.arange(365.))
            record_by_id = {int(record["sample_id"]): (index, record) for index, record in enumerate(records[class_id])}
            for saved in saved_refs:
                reference_id = str(saved["coarse_segment_id"])
                reference = geometry[reference_id]
                queries = relative_queries(reference, args.local_waveform_points)
                reference_multi = sample_periodic_curve(prototype_multi, queries)
                reference_pc1 = reference_multi @ projection.axis - float(projection.center @ projection.axis)
                expected_pc1 = sample_periodic_curve(prototype_pc1, queries)[:, 0]
                consistency = template_pc1_consistency_error(
                    reference_multi, projection.axis, expected_pc1,
                    projection_offset=float(projection.center @ projection.axis))
                if not np.isfinite(consistency) or consistency > 1e-8:
                    raise RuntimeError(
                        f"template PC1 consistency failure for {args.task} class={class_id} "
                        f"reference={reference_id}: relative_error={consistency:.6g}"
                    )
                templates.append(dict(task=args.task, source_domain=source, class_id=class_id, class_name=classes[class_id],
                    reference_structure_id=reference_id, reference_direction=reference["direction"],
                    template_pc1_consistency_error=consistency, reference_energy=float(np.linalg.norm(endpoint_relative(reference_multi)))))
                matching = [row for key, row in baseline_index.items()
                            if key[0] == source and int(key[1]) == class_id and key[2] == reference_id]
                for baseline_row in matching:
                    sample_id = int(baseline_row["sample_id"])
                    if sample_id not in record_by_id:
                        raise ValueError(f"06B G+ sample missing from source replay: {sample_id}")
                    sample_index, record = record_by_id[sample_id]
                    _, detected = _detect(sample_pc1_curves[sample_index], baselines[class_id], fine_settings, coarse_settings, f"I{sample_index}_")
                    segments = [asdict(item) for item in detected]
                    baseline_id = str(baseline_row["matched_sample_segment_id"])
                    pool = candidate_pool(segments, reference["direction"], baseline_id)
                    waveforms = {item["coarse_segment_id"]: evaluate_mode13(record["coefficients"], relative_queries(item, args.local_waveform_points)) for item in pool}
                    pc1_waveforms = {sid: value @ projection.axis - float(projection.center @ projection.axis) for sid, value in waveforms.items()}
                    handcrafted = {item["coarse_segment_id"]: structural_cost(descriptor(saved), descriptor(item), calibrations[reference["direction"]]) for item in pool}
                    ranked = compare_candidate_waveforms(reference_multi, waveforms, projection.axis, baseline_id, handcrafted)
                    key = (source, str(class_id), reference_id, str(sample_id))
                    if key not in identity_index:
                        raise ValueError(f"06C comparison row missing: {key}")
                    identity = identity_index[key]
                    row = dict(task=args.task, source_domain=source, class_id=class_id, class_name=classes[class_id],
                        reference_structure_id=reference_id, reference_direction=reference["direction"], sample_id=sample_id,
                        baseline_segment_id=baseline_id, num_same_direction_candidates=len(pool), multi_candidate=len(pool) >= 2,
                        three_plus_candidates=len(pool) >= 3, **{"06c_final_segment_id": identity["identity_sample_segment_id"],
                        "06c_final_exact": identity["identity_sample_segment_id"] == baseline_id},
                        reference_energy=float(np.linalg.norm(endpoint_relative(reference_multi))),
                        baseline_candidate_energy=float(np.linalg.norm(endpoint_relative(waveforms[baseline_id]))),
                        degenerate_waveform=bool(any(np.linalg.norm(endpoint_relative(value)) < 1e-12 for value in waveforms.values())))
                    for method, result in ranked.items():
                        row.update(flatten_ranking(method, result))
                    all_rows.append(row)
                    contexts[(class_id, reference_id, sample_id)] = (prototype_pc1, sample_pc1_curves[sample_index], reference,
                        {item["coarse_segment_id"]: item for item in pool}, reference_pc1, pc1_waveforms, reference_multi, waveforms)
    if set(baseline_index) != {(row["source_domain"], str(row["class_id"]), row["reference_structure_id"], str(row["sample_id"])) for row in all_rows}:
        raise ValueError("baseline G+ join failure: reliable-reference rows do not match exactly")
    output = args.output_root / args.task
    with staged_output(output, REQUIRED_TASK_FILES) as staging:
        write_csv(staging / "gplus_local_ranking.csv", all_rows)
        write_csv(staging / "gplus_structure_summary.csv", structure_summary(all_rows))
        write_csv(staging / "metric_comparison.csv", metric_summary(all_rows, args.task))
        write_csv(staging / "transition_summary.csv", transition_summary(all_rows, args.task))
        write_csv(staging / "template_summary.csv", templates)
        write_csv(staging / "bootstrap_summary.csv", [dict(task=args.task, **cluster_bootstrap(all_rows, args.bootstrap_repeats, args.seed))])
        diagnostic_specs = (
            ("diagnostics/06c_wrong_07a_correct", lambda r: not r["06c_final_exact"] and r["multi_ned_exact"]),
            ("diagnostics/both_wrong", lambda r: not r["06c_final_exact"] and not r["multi_ned_exact"]),
            ("diagnostics/06c_correct_07a_wrong", lambda r: r["06c_final_exact"] and not r["multi_ned_exact"]),
        )
        plots = []
        for folder, predicate in diagnostic_specs:
            (staging / folder).mkdir(parents=True)
            for index, row in enumerate(_select_examples(all_rows, predicate, args.max_examples)):
                context = contexts[(row["class_id"], row["reference_structure_id"], row["sample_id"])]
                prototype_pc1, sample_pc1, reference, segments, ref_pc1, pc1s, ref_multi, multis = context
                path = f"{folder}/{index:02d}_class{row['class_id']}_{row['reference_structure_id']}_sample{row['sample_id']}.png"
                _plot_case(staging / path, prototype_pc1, sample_pc1, reference,
                           segments[row["baseline_segment_id"]], segments.get(row["multi_ned_best_id"]),
                           segments.get(row["06c_final_segment_id"]), ref_pc1, pc1s, ref_multi, multis, row)
                plots.append(path)
        write_json(staging / "manifest.json", dict(experiment="07A_multivariate_local_waveform_scan", task=args.task,
            source_domain=source, target_used=False, training=False, parameter_updates=False, local_waveform_points=args.local_waveform_points,
            mode=13, candidate_pool="accepted=true and same direction", evaluation_positive="06B G+ baseline matched segment",
            pc1="fixed configuration-05 source-class Raw-PSE PC1; never refit", multivariate_score_uses_pc1=False,
            template_pc1_consistency_tolerance=1e-8,
            distance="endpoint-relative normalized Euclidean", per_window_zscore=False, bootstrap_cluster="sample_id",
            num_references=len(templates), num_Gplus=len(all_rows), num_multi_candidate=sum(r["multi_candidate"] for r in all_rows),
            average_candidate_count=float(np.mean([r["num_same_direction_candidates"] for r in all_rows])),
            baseline_join_failures=0, degenerate_waveform_count=sum(r["degenerate_waveform"] for r in all_rows), plots=plots,
            inputs={"checkpoint": file_hash(checkpoint), "05_manifest": file_hash(folder05 / "manifest.json"),
                    "06B": file_hash(baseline_path), "06C": file_hash(args.identity_root / args.task / "sample_structure_identity.csv")}))
    print(f"[FINISHED] 07A {args.task}: G+={len(all_rows)}, multi={sum(r['multi_candidate'] for r in all_rows)}", flush=True)


def run_aggregate(args):
    rows = []
    for task in TASKS:
        rows.extend(read_csv(args.output_root / task / "gplus_local_ranking.csv"))
    typed = []
    bool_fields = ("multi_candidate", "handcrafted_local_exact", "pc1_ned_exact", "multi_ned_exact", "multi_cosine_exact", "06c_final_exact")
    int_fields = ("sample_id", "num_same_direction_candidates", "handcrafted_positive_rank", "pc1_ned_positive_rank", "multi_ned_positive_rank",
                  "handcrafted_pairwise_wins", "handcrafted_pairwise_total", "pc1_ned_pairwise_wins", "pc1_ned_pairwise_total",
                  "multi_ned_pairwise_wins", "multi_ned_pairwise_total")
    for row in rows:
        typed.append(dict(row, **{field: truth(row[field]) for field in bool_fields},
                          **{field: int(row[field]) for field in int_fields}))
    write_csv(args.output_root / "metric_comparison.csv", metric_summary(typed, "TOTAL"))
    write_csv(args.output_root / "transition_summary.csv", transition_summary(typed, "TOTAL"))
    write_csv(args.output_root / "bootstrap_summary.csv", [dict(task="TOTAL", **cluster_bootstrap(typed, args.bootstrap_repeats, args.seed))])
    write_json(args.output_root / "manifest.json", dict(experiment="07A_multivariate_local_waveform_scan", completed=True,
        tasks=list(TASKS), target_used=False, training=False))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("task", "aggregate", "publish"), default="task")
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--structure-view-root", type=Path, default=Path("outputs/shift_visualizations_seed1"))
    parser.add_argument("--validity-root", type=Path, default=Path("outputs/06A_structure_reference_validity"))
    parser.add_argument("--observation-root", type=Path, default=Path("outputs/shift_visualizations_seed1/06B_structure_observation_support"))
    parser.add_argument("--identity-root", type=Path, default=Path("outputs/shift_visualizations_seed1/06C_structure_identity_phase"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/shift_visualizations_seed1/07A_multivariate_local_waveform_scan"))
    parser.add_argument("--revision-root", type=Path)
    parser.add_argument("--local-waveform-points", type=int, default=32)
    parser.add_argument("--min-bootstrap-occurrence", type=float, default=.8)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reconstruction-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.local_waveform_points != 32 or not np.isclose(args.min_bootstrap_occurrence, .8):
        raise ValueError("07A fixes local waveform points=32 and bootstrap gate=0.8")
    if args.stage == "task":
        run_task(args)
    elif args.stage == "aggregate":
        run_aggregate(args)
    else:
        if args.revision_root is None:
            raise ValueError("--revision-root required")
        publish_revision(args.revision_root, args.output_root)
        print(f"[PUBLISHED] 07A: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
