#!/usr/bin/env python3
"""06C: prepare four source datasets, calibrate together, then audit identity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import uuid
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_identity_phase_diagnostic import (
    SOURCES, TIMING_FIELDS, TIMING_LABELS, audit_group, calibrate, descriptor, differences,
    grouped_candidates, identify, index_baseline, plot_gplus_misassignment, plot_identity, plot_phase_distribution,
    select_gplus_misassignments,
    staged_output, summarize_boundaries, summarize_gplus, summarize_structures, summarize_transitions, write_csv, write_json,
)

TASKS = {"AT1_DK1": ("AT1", "austria/33UVP/2017"), "DK1_FR1": ("DK1", "denmark/32VNH/2017"),
         "FR1_FR2": ("FR1", "france/30TXT/2017"), "FR2_AT1": ("FR2", "france/31TCJ/2017")}
REQUIRED_OUTPUTS = ("identity_calibration.csv", "sample_structure_identity.csv", "structure_identity_summary.csv",
                    "gplus_consistency_summary.csv", "failure_transition_summary.csv", "boundary_summary.csv", "manifest.json")
SAMPLE_FIELDS = ("task", "source_domain", "class_id", "class_name", "reference_structure_id", "reference_direction",
    "sample_id", "baseline_06B_status", "baseline_failure_reason", "baseline_matched_segment_id", "identity_status",
    "identity_sample_segment_id", "primary_sample_segment_id", "candidate_source", "sample_segment_accepted",
    "structural_cost", "identity_threshold", "calibration_level", "reference_unique_best", "sample_unique_best",
    "unique_mutual_best", "neighbor_support", "assignment_conflict", "num_identity_candidates", "num_secondary_candidates",
    "best_structural_cost", "second_best_structural_cost", "cost_margin",
    "curve_change_difference", "domain_change_difference", "monotonicity_difference", "fine_count_difference",
    "curve_change_cost_contribution", "domain_change_cost_contribution", "monotonicity_cost_contribution", "fine_count_cost_contribution",
    "active_component_count", "curve_component_active", "curve_component_contribution",
    "domain_component_active", "domain_component_contribution", "monotonicity_component_active",
    "monotonicity_component_contribution", "fine_count_component_active", "fine_count_component_contribution",
    "gplus_exact_segment_recovered") + TIMING_FIELDS + ("reference_cross_boundary", "sample_cross_boundary", "has_grouped_candidate",
    "grouped_status", "grouped_candidate_ids", "grouped_structural_cost", "grouped_monotonicity", "group_start_segment",
    "group_end_segment", "baseline_individual_occurrence", "bootstrap_occurrence")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def is_true(value):
    return value is True or str(value).lower() in ("true", "1")


def validate_06b_packet(task, source, manifest, rows):
    """Reject stale/misrouted 06B inputs before historical labels are trusted."""
    experiment = manifest.get("experiment", {})
    model = manifest.get("model", {})
    if (manifest.get("task") != task or manifest.get("source_domain") != source
            or experiment.get("name") != "structure_observation_support_06B"
            or model.get("mode") != 13 or model.get("frozen") is not True
            or model.get("pc1_refit") is not False
            or manifest.get("target_used") is not False or manifest.get("training") is not False):
        raise ValueError(f"06B manifest ownership/configuration mismatch for {task}")
    for row in rows:
        if row.get("task") != task or row.get("source_domain") != source:
            raise ValueError(f"06B row ownership mismatch for {task}")


def validate_args(args):
    if not np.isclose(float(args.min_bootstrap_occurrence), .8, atol=0., rtol=0.):
        raise ValueError("06C fixes --min-bootstrap-occurrence at 0.8")
    if args.max_examples_per_structure < 0:
        raise ValueError("max examples must be nonnegative")
    if args.stage in ("prepare", "audit") and args.task is None:
        raise ValueError("--task required for prepare/audit")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_revision_root(root):
    root = Path(root)
    calibration_path = root / "calibration.json"
    if not calibration_path.is_file() or not calibration_path.stat().st_size:
        raise RuntimeError("incomplete 06C revision: missing global calibration")
    packet = read_json(calibration_path)
    if set(packet.get("source_domains", ())) != set(SOURCES):
        raise RuntimeError("incomplete 06C revision: invalid global calibration")
    for task in TASKS:
        folder = root / task
        for name in REQUIRED_OUTPUTS:
            path = folder / name
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"incomplete 06C revision: {task}/{name}")
        manifest = read_json(folder / "manifest.json")
        if manifest.get("completed") is not True or manifest.get("task") != task:
            raise RuntimeError(f"incomplete 06C revision manifest: {task}")
        for relative in manifest.get("plots", ()):
            if not (folder / relative).is_file():
                raise RuntimeError(f"incomplete 06C revision plot: {task}/{relative}")


def publish_revision(revision_root, final_root):
    """Validate the entire four-task revision before replacing the prior root."""
    revision_root, final_root = Path(revision_root), Path(final_root)
    validate_revision_root(revision_root)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    backup = final_root.parent / f".old_{final_root.name}_{uuid.uuid4().hex}"
    moved_old = False
    try:
        if final_root.exists():
            final_root.replace(backup)
            moved_old = True
        revision_root.replace(final_root)
    except Exception:
        if moved_old and backup.exists() and not final_root.exists():
            backup.replace(final_root)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def id_order(value):
    """SC2 precedes SC10, preserving the existing detector ID order."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value)))


def collect_gplus(source, payload):
    result = []
    refs = {r["coarse_segment_id"]: r for r in payload["references"]}
    for sample in payload["samples"]:
        segments = {s["coarse_segment_id"]: s for s in sample["segments"]}
        for row in sample["baseline"]:
            if not is_true(row["matched"]):
                continue
            sid = row["matched_sample_segment_id"]
            if sid not in segments:
                raise ValueError(f"G+ historical segment not found: {source}, {payload['class_id']}, {sample['sample_id']}, {sid}")
            ref = refs[row["reference_structure_id"]]
            rd, sd = descriptor(ref), descriptor(segments[sid])
            if rd.direction != sd.direction:
                raise ValueError("G+ historical direction mismatch")
            result.append(dict(source_domain=source, direction=rd.direction,
                class_id=payload["class_id"], reference_structure_id=row["reference_structure_id"],
                sample_id=sample["sample_id"], differences=differences(rd, sd).tolist()))
    return result


def audit_prepared_class(task, payload, calibration, min_monotonicity):
    refs = payload["references"]
    source = TASKS[task][0]
    rows, alignments = [], {}
    for sample in payload["samples"]:
        identified, alignment = identify(refs, sample["segments"], calibration)
        alignments[sample["sample_id"]] = alignment
        history = {r["reference_structure_id"]: r for r in sample["baseline"]}
        groups = None
        for ref, row in zip(refs, identified):
            baseline = history[ref["coarse_segment_id"]]
            plus = is_true(baseline["matched"])
            row.pop("candidate_index")
            row.update(task=task, source_domain=source, class_id=payload["class_id"], class_name=payload["class_name"],
                sample_id=sample["sample_id"], baseline_06B_status="G+" if plus else "G-",
                baseline_failure_reason=baseline["failure_reason"], baseline_matched_segment_id=baseline["matched_sample_segment_id"],
                bootstrap_occurrence=ref["bootstrap_occurrence"], baseline_individual_occurrence=ref["baseline_individual_occurrence"])
            row["gplus_exact_segment_recovered"] = bool(
                plus and row["candidate_source"] == "PRIMARY_ACCEPTED"
                and row["identity_sample_segment_id"] == row["baseline_matched_segment_id"])
            row.update(audit_group(ref, [], calibration[ref["direction"]]))
            row["grouped_status"] = "NOT_AUDITED"
            if not plus and row["identity_status"] in ("AMBIGUOUS", "NO_STRUCTURAL_COUNTERPART"):
                if groups is None:
                    groups = grouped_candidates(sample["segments"], min_monotonicity)
                row.update(audit_group(ref, groups, calibration[ref["direction"]]))
            rows.append(row)
    return rows, alignments


def run_prepare(args):
    # Imports with model/data dependencies belong only to feature preparation.
    import torch
    from models.fourier_reconstruction import BatchedDirectFourierAnalyzer, BatchedDirectFourierSynthesizer
    from scripts.diagnose_structure_observation_support import _load_inputs, _detect, _validate_computed_references
    from scripts.diagnose_structure_reference_validity import _checkpoint_paths, build_source_split, _detector_settings
    from scripts.visualize_shift_configs_4tasks import (
        fit_class_projections, load_classes, load_raw_spatial_encoder, project_dataset,
        reconstruct_class_prototype, reconstruct_projected_coefficients,
    )
    validate_args(args)
    source, source_path = TASKS[args.task]
    if args.data_root is None or not args.data_root.is_dir():
        raise FileNotFoundError(f"source data root missing: {args.data_root}")
    if not args.source_checkpoint:
        raise ValueError("--source-checkpoint is required for prepare")
    checkpoint, config_path = _checkpoint_paths(args.source_checkpoint)
    config = read_json(config_path)
    folder, manifest05, saved_coarse, class_scope, reliable = _load_inputs(args, source)
    baseline_folder = args.observation_root / args.task
    baseline_path = baseline_folder / "sample_structure_support.csv"
    baseline_manifest = read_json(baseline_folder / "manifest.json")
    baseline_rows = read_csv(baseline_path)
    validate_06b_packet(args.task, source, baseline_manifest, baseline_rows)
    scope = {int(row["class_id"]) for row in class_scope}
    reliable = [r for r in reliable if int(r["class_id"]) in scope]
    if not reliable:
        raise ValueError("no reliable 05/06A references")
    classes = load_classes(checkpoint, source_path, args.data_root)
    source_set = build_source_split(args.data_root, source_path, classes, args.seed, config)
    encoder = load_raw_spatial_encoder(checkpoint, config, classes, torch.device(args.device))
    grid = np.linspace(0., 365., args.grid_size)
    device = torch.device(args.device)
    extra = bool(config.get("with_extra", False))
    with torch.no_grad():
        # Replay exactly the 05 full-source Raw-PSE PC1 protocol, then freeze it.
        projections, _ = fit_class_projections(encoder, source_set, len(classes), grid, args.batch_size, device, extra)
        analyzer = BatchedDirectFourierAnalyzer(13, period_days=365., reg=.001).to(device)
        synthesizer = BatchedDirectFourierSynthesizer(13, period_days=365.).to(device)
        _, coeffs, records, baselines = project_dataset(encoder, source_set, projections, grid, args.batch_size,
            device, extra, analyzer, collect_samplewise=True)
        fine_settings, coarse_settings = _detector_settings(manifest05)
        by_class = defaultdict(list)
        for ref in reliable:
            by_class[int(ref["class_id"])].append(ref)
        expected = [(source, cid, ref["coarse_segment_id"], int(record["sample_id"]))
                    for cid, refs in by_class.items() for ref in refs for record in records.get(cid, [])]
        history = index_baseline(baseline_rows, expected)
        gplus = []
        final_work = args.work_root / args.task
        required = ["manifest.json"] + [f"class_{cid}.{ext}" for cid in by_class for ext in ("json", "npz")]
        with staged_output(final_work, required) as staging:
            for cid, saved in sorted(by_class.items()):
                if cid not in coeffs or cid not in projections:
                    raise ValueError(f"reference class has no source features: {cid}")
                prototype = reconstruct_class_prototype(coeffs[cid], synthesizer, device, sample_batch_size=args.reconstruction_batch_size)
                proto = projections[cid].transform(prototype[None])[0]
                _, computed = _detect(proto, baselines[cid], fine_settings, coarse_settings, "S")
                _validate_computed_references(computed, saved_coarse, cid)
                by_id = {r.coarse_segment_id: r for r in computed}
                refs = []
                for record in sorted(saved, key=lambda r: id_order(r["coarse_segment_id"])):
                    ref = asdict(by_id[record["coarse_segment_id"]])
                    # Descriptor values are authoritative configuration-05 values;
                    # replayed curves serve figures and verify geometry only.
                    for name in ("curve_normalized_change", "domain_normalized_change", "monotonicity_ratio"):
                        if not np.isclose(float(record[name]), float(ref[name]), rtol=1e-4, atol=1e-6):
                            raise ValueError(f"05 descriptor mismatch: {cid}/{ref['coarse_segment_id']}/{name}")
                        ref[name] = float(record[name])
                    if int(record["num_fine_segments_covered"]) != int(ref["num_fine_segments_covered"]):
                        raise ValueError(f"05 descriptor mismatch: {cid}/{ref['coarse_segment_id']}/num_fine_segments_covered")
                    ref["num_fine_segments_covered"] = int(record["num_fine_segments_covered"])
                    ref.update(bootstrap_occurrence=float(record["bootstrap_occurrence_rate"]),
                               # 06A is only the bootstrap-stability gate in
                               # 06C. Preserve 05's original occurrence
                               # measurement as the descriptive baseline.
                               baseline_individual_occurrence=float(record["source_occurrence_rate"]))
                    refs.append(ref)
                curves = reconstruct_projected_coefficients(coeffs[cid], projections[cid], np.arange(365.))
                if len(curves) != len(records[cid]):
                    raise ValueError("source sample/curve order mismatch")
                payload = dict(class_id=cid, class_name=classes[cid], references=refs, samples=[])
                for index, (curve, record) in enumerate(zip(curves, records[cid])):
                    sid = int(record["sample_id"])
                    _, coarse = _detect(curve, baselines[cid], fine_settings, coarse_settings, f"I{index}_")
                    payload["samples"].append(dict(sample_id=sid, segments=[asdict(s) for s in coarse],
                        baseline=[history[(source, cid, r["coarse_segment_id"], sid)] for r in refs]))
                gplus.extend(collect_gplus(source, payload))
                write_json(staging / f"class_{cid}.json", payload)
                np.savez_compressed(staging / f"class_{cid}.npz", prototype=proto, curves=curves)
                print(f"PREPARED|task={args.task}|class={cid}|samples={len(curves)}|references={len(refs)}", flush=True)
            write_json(staging / "manifest.json", dict(task=args.task, source_domain=source, class_ids=sorted(by_class),
                gplus=gplus, completed=True, coarse_min_monotonicity=coarse_settings["min_monotonicity"],
                fine_settings=fine_settings, coarse_settings=coarse_settings, seed=args.seed, grid_size=args.grid_size,
                checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint), baseline_06B=baseline_manifest,
                inputs={str(p): file_hash(p) for p in (folder / "manifest.json", folder / "source_coarse_segments.csv",
                    args.validity_root / source / "structure_stability.csv", baseline_path)},
                pc1_protocol="replay configuration-05 full-source Raw-PSE interpolation PC1; fixed for all Mode13 structures"))
    print(f"[PREPARED] {args.task}: G+ pairs={len(gplus)}", flush=True)


def run_calibration(args):
    validate_args(args)
    manifests, hashes, records = [], {}, []
    for task, (source, _) in TASKS.items():
        path = args.work_root / task / "manifest.json"
        packet = read_json(path)
        if packet.get("completed") is not True or packet["source_domain"] != source or packet["task"] != task:
            raise ValueError(f"incomplete/wrong preparation: {task}")
        if any(row["source_domain"] != source for row in packet["gplus"]):
            raise ValueError(f"incorrect G+ source ownership: {task}")
        manifests.append(packet)
        hashes[task] = file_hash(path)
        records.extend(packet["gplus"])
    result = calibrate(records, args.identity_calibration_min_pairs,
                       args.identity_component_scale_quantile, args.identity_cost_quantile,
                       min_positive_pairs=args.identity_component_min_positive_pairs,
                       min_active_components=args.identity_calibration_min_active_components)
    write_json(args.work_root / "calibration.json", dict(calibrations=result, prepared_manifest_hashes=hashes,
        source_domains=list(SOURCES), min_pairs=args.identity_calibration_min_pairs,
        component_scale_quantile=args.identity_component_scale_quantile, cost_quantile=args.identity_cost_quantile,
        component_min_positive_pairs=args.identity_component_min_positive_pairs,
        min_active_components=args.identity_calibration_min_active_components,
        revision="robust-active-components-primary-secondary-v2"))
    for key, row in result.items():
        print(f"CALIBRATION|{key}|level={row['calibration_level']}|n={row['num_Gplus_pairs']}|threshold={row['threshold']:.6g}", flush=True)


def calibration_csv(calibrations):
    rows = []
    for row in calibrations.values():
        result = dict(source_domain=row["source_domain"], direction=row["direction"], calibration_level=row["calibration_level"],
                      num_Gplus_pairs=row["num_Gplus_pairs"], num_active_components=row["num_active_components"],
                      identity_cost_threshold=row["threshold"])
        for name, scale, active, count in zip(("curve_change", "domain_change", "monotonicity", "fine_count"),
                                              row["scales"], row["active"], row["positive_pairs"]):
            result.update({f"{name}_active": active, f"{name}_positive_pairs": count, f"{name}_scale": scale})
        rows.append(result)
    return rows


def _plot_class(staging, payload, rows, alignments, curves, prototype, calibration, max_examples):
    required = []
    by_sample = {s["sample_id"]: (i, s) for i, s in enumerate(payload["samples"])}
    for ref in payload["references"]:
        group = [r for r in rows if r["reference_structure_id"] == ref["coarse_segment_id"]]
        prefix = f"{payload['class_id']:02d}_{payload['class_name']}_{ref['coarse_segment_id']}"
        path = f"diagnostics/{prefix}_phase_distribution.png"
        plot_phase_distribution(staging / path, group)
        required.append(path)
        rescued = [r for r in group if r["baseline_failure_reason"] in ("CENTER_DISTANCE_FAIL", "DURATION_RATIO_FAIL")
                   and r["identity_status"] == "LIKELY_SAME_STRUCTURE"]
        # Include examples from both failure types before filling the quota.
        selected = []
        for reason in ("CENTER_DISTANCE_FAIL", "DURATION_RATIO_FAIL"):
            chosen = sorted([r for r in rescued if r["baseline_failure_reason"] == reason], key=lambda r: r["sample_id"])
            if chosen:
                selected.append(chosen[0])
        selected.extend(r for r in sorted(rescued, key=lambda r: r["sample_id"]) if r not in selected)
        grouped = sorted([r for r in group if r["has_grouped_candidate"]], key=lambda r: r["sample_id"])[:max_examples]
        for is_group, examples in ((False, selected[:max_examples]), (True, grouped)):
            for row in examples:
                idx, sample = by_sample[row["sample_id"]]
                path = (f"fragmentation_diagnostics/{prefix}_{row['sample_id']}.png" if is_group else
                        f"diagnostics/{prefix}_{row['sample_id']}_identity.png")
                plot_identity(staging / path, prototype, curves[idx], payload["references"], sample["segments"],
                    alignments[row["sample_id"]], row, calibration[ref["direction"]], grouped=is_group)
                required.append(path)
    contexts = {int(row["sample_id"]): (payload, by_sample[int(row["sample_id"])][1],
                                        curves[by_sample[int(row["sample_id"])][0]], prototype)
                for row in rows}
    return required, contexts


def run_audit(args):
    validate_args(args)
    source = TASKS[args.task][0]
    work = args.work_root / args.task
    prepared = read_json(work / "manifest.json")
    calibration_packet = read_json(args.work_root / "calibration.json")
    if set(calibration_packet["source_domains"]) != set(SOURCES):
        raise ValueError("all four source calibration inputs are required")
    for task in TASKS:
        if calibration_packet["prepared_manifest_hashes"][task] != file_hash(args.work_root / task / "manifest.json"):
            raise ValueError(f"prepared packet changed after calibration: {task}")
    calibration = {direction: calibration_packet["calibrations"][source + ":" + direction] for direction in ("RISE", "FALL")}
    all_rows, required_plots, visual_contexts = [], [], {}
    with staged_output(args.output_root / args.task, REQUIRED_OUTPUTS) as staging:
        (staging / "diagnostics").mkdir()
        (staging / "fragmentation_diagnostics").mkdir()
        (staging / "gplus_misassignment_diagnostics").mkdir()
        for cid in prepared["class_ids"]:
            payload = read_json(work / f"class_{cid}.json")
            rows, alignments = audit_prepared_class(args.task, payload, calibration, prepared["coarse_min_monotonicity"])
            all_rows.extend(rows)
            with np.load(work / f"class_{cid}.npz", allow_pickle=False) as arrays:
                plots, contexts = _plot_class(staging, payload, rows, alignments, arrays["curves"], arrays["prototype"], calibration, args.max_examples_per_structure)
                required_plots.extend(plots)
                visual_contexts.update({(cid, sid): context for sid, context in contexts.items()})
            print(f"AUDITED|task={args.task}|class={cid}|rows={len(rows)}", flush=True)
        if not all_rows:
            raise ValueError("empty identity audit")
        for index, row in enumerate(select_gplus_misassignments(all_rows, 20)):
            payload, sample, curve, prototype = visual_contexts[(int(row["class_id"]), int(row["sample_id"]))]
            refs = {item["coarse_segment_id"]: item for item in payload["references"]}
            segments = {item["coarse_segment_id"]: item for item in sample["segments"]}
            baseline = segments.get(row["baseline_matched_segment_id"])
            if baseline is None:
                raise ValueError("G+ baseline segment missing during misassignment visualization")
            chosen = segments.get(row["identity_sample_segment_id"])
            name = f"gplus_misassignment_diagnostics/{index:02d}_class{row['class_id']}_{row['reference_structure_id']}_sample{row['sample_id']}.png"
            plot_gplus_misassignment(staging / name, prototype, curve, refs[row["reference_structure_id"]], baseline, chosen, row)
            required_plots.append(name)
        summary = summarize_structures(all_rows)
        plus_total = sum(r["num_Gplus"] for r in summary)
        exact_total = sum(r["gplus_exact_segment_count"] for r in summary)
        consistency = exact_total / plus_total if plus_total else None
        csv_rows = [dict(r, grouped_candidate_ids=json.dumps(r["grouped_candidate_ids"])) for r in all_rows]
        write_csv(staging / "sample_structure_identity.csv", csv_rows, SAMPLE_FIELDS)
        write_csv(staging / "structure_identity_summary.csv", summary)
        write_csv(staging / "gplus_consistency_summary.csv", summarize_gplus(all_rows))
        write_csv(staging / "identity_calibration.csv", calibration_csv(calibration))
        write_csv(staging / "failure_transition_summary.csv", summarize_transitions(all_rows),
                  ("scope", "class_id", "reference_structure_id", "baseline_failure_reason", "candidate_source", "identity_status", "count", "fraction"))
        write_csv(staging / "boundary_summary.csv", summarize_boundaries(all_rows))
        write_json(staging / "manifest.json", dict(task=args.task, source_domain=source, completed=True,
            experiment=dict(name="structure_identity_phase_06C", revision="robust_calibration_v2"),
            preparation={k: v for k, v in prepared.items() if k != "gplus"}, calibration=calibration_packet,
            gplus_identity_consistency_rate=consistency, num_rows=len(all_rows), plots=required_plots,
            target_used=False, training=False, timing_used_in_identity=False, duration_used_in_identity=False,
            revision="robust-active-components-primary-secondary-v2",
            calibration_policy=dict(zero_scale_policy="disable_component",
                scale_source="positive_gplus_differences_only",
                component_scale_quantile=args.identity_component_scale_quantile,
                min_positive_pairs=args.identity_component_min_positive_pairs,
                min_active_components=args.identity_calibration_min_active_components),
            candidate_pool=dict(primary="accepted_coarse_segments", secondary="rejected_coarse_segments",
                                secondary_can_be_likely=False),
            likely_requires_unique_mutual_best=True, neighbor_support_high_confidence=False,
            sample_sequence="primary accepted coarse segments; rejected segments are secondary fallback only",
            reference_sequence="05 accepted, 06A bootstrap>=0.8, numeric structure ID order",
            descriptor=["direction", "A_curve", "A_domain", "monotonicity", "num_fine_segments_covered"],
            identity_policy="primary accepted unique mutual-best -> likely; primary single unsupported or unique rejected fallback -> plausible; neighbor support diagnostic only",
            gap_definition="unmatched sequence elements", mutual_best="unique exact minimum on both sides; ties are not mutual-best",
            rescue_definition="high-confidence = primary accepted likely; loose = likely or plausible; grouped excluded",
            timing_scope="formal summaries use primary accepted identities only; secondary timing remains row-level",
            grouped_scope="G- ambiguous/no counterpart only; consecutive 3 on full sample circle; diagnostic only"))
        for name in REQUIRED_OUTPUTS + tuple(required_plots):
            if not (staging / name).is_file() or (staging / name).stat().st_size == 0:
                raise RuntimeError(f"incomplete 06C output: {name}")
        if len(read_csv(staging / "sample_structure_identity.csv")) != len(all_rows):
            raise RuntimeError("06C CSV row-count mismatch")
    print(f"[FINISHED] 06C {args.task}: rows={len(all_rows)}, G+ consistency={consistency}", flush=True)


def run_publish(args):
    if args.revision_root is None:
        raise ValueError("--revision-root is required for publish")
    publish_revision(args.revision_root, args.output_root)
    print(f"[PUBLISHED] complete 06C revision: {args.output_root}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "calibrate", "audit", "publish"), required=True)
    p.add_argument("--task", choices=tuple(TASKS))
    p.add_argument("--data-root", type=Path)
    p.add_argument("--source-checkpoint")
    p.add_argument("--structure-view-root", type=Path, default=Path("outputs/shift_visualizations_seed1"))
    p.add_argument("--validity-root", type=Path, default=Path("outputs/06A_structure_reference_validity"))
    p.add_argument("--observation-root", type=Path, default=Path("outputs/shift_visualizations_seed1/06B_structure_observation_support"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/shift_visualizations_seed1/06C_structure_identity_phase"))
    p.add_argument("--work-root", type=Path, default=Path("outputs/shift_visualizations_seed1/.06C_work"))
    p.add_argument("--revision-root", type=Path)
    p.add_argument("--min-bootstrap-occurrence", type=float, default=.8)
    p.add_argument("--identity-component-scale-quantile", type=float, default=.90)
    p.add_argument("--identity-cost-quantile", type=float, default=.95)
    p.add_argument("--identity-calibration-min-pairs", type=int, default=50)
    p.add_argument("--identity-component-min-positive-pairs", type=int, default=20)
    p.add_argument("--identity-calibration-min-active-components", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--grid-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--reconstruction-batch-size", type=int, default=128)
    p.add_argument("--max-examples-per-structure", type=int, default=3)
    p.add_argument("--device", default="cuda")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    {"prepare": run_prepare, "calibrate": run_calibration, "audit": run_audit, "publish": run_publish}[args.stage](args)


if __name__ == "__main__":
    main()
