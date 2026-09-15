#!/usr/bin/env python
"""Train and evaluate the source-only 07B structure identity experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.multivariate_local_waveform_scan_diagnostic import (  # noqa: E402
    endpoint_relative_ned,
    evaluate_mode13,
    sample_periodic_curve,
)
from analysis.recon_structure_segments import (  # noqa: E402
    build_coarse_structure,
    detect_structure_segments,
)
from analysis.structure_identity_encoder_experiment import (  # noqa: E402
    apply_rejection_thresholds,
    assign_groups_to_split,
    calibrate_rejection_thresholds,
    cosine_candidate_scores,
    evaluate_with_frozen_thresholds,
    fit_normalization_statistics,
    retrieval_rows,
    seed_everything,
    staged_source_output,
    stratified_sample_split,
    structure_identity_loss,
    summarize_retrieval,
    validate_train_reference,
)
from analysis.structure_observation_support_diagnostic import write_csv, write_json  # noqa: E402
from models.structure_identity import StructureIdentityEncoder  # noqa: E402


SOURCES = {
    "AT1": ("AT1_DK1", "austria/33UVP/2017"),
    "DK1": ("DK1_FR1", "denmark/32VNH/2017"),
    "FR1": ("FR1_FR2", "france/30TXT/2017"),
    "FR2": ("FR2_AT1", "france/31TCJ/2017"),
}
VARIANTS = ("Waveform", "Event", "Fusion")
SEED_FILES = (
    "config.json", "split_manifest.json", "normalization_stats.json",
    "train_history.csv", "validation_metrics.csv", "threshold_calibration.csv",
    "test_metrics.csv", "test_pair_predictions.csv", "reference_summary.csv",
    "reference_embeddings.npz", "best_model.pt", "manifest.json",
)
SOURCE_FILES = (
    "three_seed_summary.csv", "ablation_summary.csv", "transition_summary.csv",
    "reliability_summary.csv",
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def truth(value):
    return value is True or str(value).lower() in {"true", "1", "yes"}


def _detect_full(curve, baseline, fine_settings, coarse_settings, prefix):
    events, chain, fine = detect_structure_segments(
        curve, np.arange(365.0), baseline,
        event_prefix=f"{prefix}E", segment_prefix=f"{prefix}F", **fine_settings,
    )
    coarse = build_coarse_structure(
        chain, fine, curve, baseline.iqr,
        segment_prefix=f"{prefix}C", **coarse_settings,
    )
    return events, chain, fine, coarse.segments


def _inside_window(day, start, end, period=365.0):
    day = float(day) % period
    start = float(start) % period
    stop = float(end) % period
    if stop <= start:
        stop += period
    if day < start:
        day += period
    return start - 1e-8 <= day <= stop + 1e-8, day, start, stop


def _event_arrays(events, segment):
    selected = []
    for event in events:
        inside, day, start, stop = _inside_window(event.day, segment["start_day"], segment["end_day"])
        if inside:
            selected.append((day, event))
    selected.sort(key=lambda pair: (pair[0], pair[1].event_id))
    if not selected:
        return np.empty((0,), np.int64), np.empty((0, 4), np.float32)
    start, stop = selected[0][0], selected[0][0]
    _, _, window_start, window_stop = _inside_window(segment["start_day"], segment["start_day"], segment["end_day"])
    duration = window_stop - window_start
    start_value = float(segment.get("start_value", selected[0][1].value))
    types, numeric = [], []
    for day, event in selected:
        types.append(1 if event.kind == "peak" else 2)
        numeric.append([
            (day - window_start) / duration,
            float(event.relative_prominence),
            float(event.domain_relative_prominence),
            float(event.value) - start_value,
        ])
    return np.asarray(types, np.int64), np.asarray(numeric, np.float32)


def _fine_arrays(fine_segments, segment):
    ids = set(segment.get("fine_segment_ids", ()))
    selected = [edge for edge in fine_segments if edge.segment_id in ids]
    selected.sort(key=lambda edge: (edge.unwrapped_start_day, edge.segment_id))
    if not selected:
        return np.empty((0,), np.int64), np.empty((0, 6), np.float32)
    start = float(segment["start_day"])
    end = float(segment["end_day"])
    if end <= start:
        end += 365.0
    duration = end - start
    types, numeric = [], []
    for edge in selected:
        edge_start = float(edge.unwrapped_start_day)
        while edge_start < start:
            edge_start += 365.0
        edge_end = edge_start + float(edge.duration_days)
        types.append(1 if edge.direction == "RISE" else 2)
        numeric.append([
            (edge_start - start) / duration,
            (edge_end - start) / duration,
            float(edge.duration_days) / duration,
            float(edge.signed_change),
            float(edge.curve_normalized_change),
            float(edge.domain_normalized_change),
        ])
    return np.asarray(types, np.int64), np.asarray(numeric, np.float32)


def _instance_from_waveform(waveform, segment, events, fine):
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 2 or waveform.shape[0] != 32:
        raise ValueError("local waveform must be [32,D]")
    amplitude = waveform - waveform[:1]
    energy = max(float(np.linalg.norm(amplitude)), 1e-8)
    event_types, event_numeric = _event_arrays(events, segment)
    fine_types, fine_numeric = _fine_arrays(fine, segment)
    return dict(shape=amplitude / energy, amplitude=amplitude,
                event_types=event_types, events=event_numeric,
                fine_types=fine_types, fine=fine_numeric,
                raw_waveform=waveform.astype(np.float64))


def _instance(coefficients, segment, events, fine):
    query = np.linspace(float(segment["start_day"]),
                        float(segment["end_day"]) + (365.0 if float(segment["end_day"]) <= float(segment["start_day"]) else 0.0),
                        32)
    waveform = evaluate_mode13(coefficients, query).astype(np.float32)
    return _instance_from_waveform(waveform, segment, events, fine)


def _normalize_instance(instance, stats):
    output = dict(instance)
    output["amplitude"] = ((np.asarray(instance["amplitude"]) - stats["amplitude_mean"])
                           / stats["amplitude_std"]).astype(np.float32)
    output["events"] = ((np.asarray(instance["events"]) - stats["event_mean"])
                        / stats["event_std"]).astype(np.float32)
    output["fine"] = ((np.asarray(instance["fine"]) - stats["fine_mean"])
                      / stats["fine_std"]).astype(np.float32)
    return output


def _tensorize(instances, device):
    batch = len(instances)
    max_events = max(1, max(len(item["events"]) for item in instances))
    max_fine = max(1, max(len(item["fine"]) for item in instances))
    events = np.zeros((batch, max_events, 4), np.float32)
    event_types = np.zeros((batch, max_events), np.int64)
    fine = np.zeros((batch, max_fine, 6), np.float32)
    fine_types = np.zeros((batch, max_fine), np.int64)
    for index, item in enumerate(instances):
        events[index, :len(item["events"])] = item["events"]
        event_types[index, :len(item["event_types"])] = item["event_types"]
        fine[index, :len(item["fine"])] = item["fine"]
        fine_types[index, :len(item["fine_types"])] = item["fine_types"]
    event_types = torch.as_tensor(event_types, device=device)
    fine_types = torch.as_tensor(fine_types, device=device)
    return dict(
        shape_waveform=torch.as_tensor(np.stack([i["shape"] for i in instances]), device=device),
        amplitude_waveform=torch.as_tensor(np.stack([i["amplitude"] for i in instances]), device=device),
        event_types=event_types,
        event_numeric=torch.as_tensor(events, device=device),
        event_mask=event_types.ne(0),
        fine_types=fine_types,
        fine_numeric=torch.as_tensor(fine, device=device),
        fine_mask=fine_types.ne(0),
    )


def _encode(model, instances, device):
    return model(**_tensorize(instances, device))


def _group_batches(groups, batch_size, rng):
    indices = np.arange(len(groups))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [groups[int(index)] for index in indices[start:start + batch_size]]


def _batch_scores(model, groups, device):
    references, flat_candidates, counts, positives = [], [], [], []
    for group in groups:
        references.append(group["reference"])
        flat_candidates.extend(group["candidates"])
        counts.append(len(group["candidates"]))
        positives.append(int(group["positive_index"]))
    ref_embedding = _encode(model, references, device)
    candidate_embedding = _encode(model, flat_candidates, device)
    maximum = max(counts)
    padded = candidate_embedding.new_zeros((len(groups), maximum, candidate_embedding.shape[-1]))
    mask = torch.zeros((len(groups), maximum), dtype=torch.bool, device=device)
    offset = 0
    for index, count in enumerate(counts):
        padded[index, :count] = candidate_embedding[offset:offset + count]
        mask[index, :count] = True
        offset += count
    scores = cosine_candidate_scores(ref_embedding, padded)
    positive_index = torch.as_tensor(positives, device=device)
    positive_embedding = padded[torch.arange(len(groups), device=device), positive_index]
    return scores, mask, positive_index, ref_embedding, positive_embedding


@torch.inference_mode()
def _evaluate_model(model, groups, device, batch_size):
    model.eval(); arrays = []
    for start in range(0, len(groups), batch_size):
        batch = groups[start:start + batch_size]
        scores, mask, _, _, _ = _batch_scores(model, batch, device)
        arrays.extend(scores[index, mask[index]].cpu().numpy() for index in range(len(batch)))
    return retrieval_rows(groups, arrays)


def _fixed_baseline(groups):
    arrays = []
    for group in groups:
        reference = group["reference"]["raw_waveform"]
        arrays.append(np.asarray([-endpoint_relative_ned(reference, item["raw_waveform"])
                                  for item in group["candidates"]]))
    return retrieval_rows(groups, arrays)


def _train_variant(variant, groups, waveform_dim, seed, args, device):
    seed_everything(seed)
    model = StructureIdentityEncoder(waveform_dim, variant=variant, dropout=.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_key, stale, history = None, (-math.inf, -math.inf), 0, []
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for epoch in range(args.max_epochs):
        model.train(); losses = []
        for batch in _group_batches(groups["train"], args.batch_size, rng):
            scores, mask, positive, references, positives = _batch_scores(model, batch, device)
            packet = structure_identity_loss(scores, mask, positive, references, positives,
                                             temperature=.07, lambda_pos=.1)
            optimizer.zero_grad(set_to_none=True)
            packet["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); losses.append(float(packet["loss"].detach().cpu()))
        validation = _evaluate_model(model, groups["validation"], device, args.batch_size)
        summary = summarize_retrieval(validation)
        top1 = float(summary["top1"])
        margin = float(summary["positive_margin"])
        key = (top1 if np.isfinite(top1) else -math.inf,
               margin if np.isfinite(margin) else -math.inf)
        history.append(dict(variant=variant, epoch=epoch + 1,
                            train_loss=float(np.mean(losses)) if losses else np.nan,
                            validation_top1=key[0], validation_positive_margin=key[1]))
        if best_state is None or key > best_key:
            best_key, stale = key, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"no valid model checkpoint for {variant}")
    model.load_state_dict(best_state)
    return model, best_state, history, time.perf_counter() - started


def _prepare_source(args):
    from models.fourier_reconstruction import BatchedDirectFourierAnalyzer, BatchedDirectFourierSynthesizer
    from scripts.diagnose_structure_observation_support import _load_inputs, _validate_computed_references
    from scripts.diagnose_structure_reference_validity import _checkpoint_paths, _detector_settings, build_source_split
    from scripts.visualize_shift_configs_4tasks import (
        fit_class_projections, load_classes, load_raw_spatial_encoder, project_dataset,
        reconstruct_class_prototype, reconstruct_projected_coefficients,
    )

    task, source_path = SOURCES[args.source]
    checkpoint, config_path = _checkpoint_paths(args.source_checkpoint)
    config = read_json(config_path)
    classes = load_classes(checkpoint, source_path, args.data_root)
    dataset = build_source_split(args.data_root, source_path, classes, 1, config)
    proxy = argparse.Namespace(task=task, structure_view_root=args.structure_view_root,
                               validity_root=args.validity_root,
                               min_bootstrap_occurrence=.8)
    folder05, manifest05, saved05, class_rows, reliable = _load_inputs(proxy, args.source)
    reliable = [row for row in reliable if float(row["bootstrap_occurrence_rate"]) >= .8
                and int(row["class_id"]) in {int(item["class_id"]) for item in class_rows}]
    support_path = args.observation_root / task / "sample_structure_support.csv"
    gplus = [row for row in read_csv(support_path) if truth(row["matched"])]
    waveform_path = args.waveform_root / task / "gplus_local_ranking.csv"
    if not waveform_path.is_file():
        raise FileNotFoundError(f"07A input missing: {waveform_path}")
    waveform_rows = read_csv(waveform_path)
    waveform_keys = {
        (str(row["source_domain"]), int(row["class_id"]),
         str(row["reference_structure_id"]), int(row["sample_id"]))
        for row in waveform_rows
    }
    waveform_index = {
        (str(row["source_domain"]), int(row["class_id"]),
         str(row["reference_structure_id"]), int(row["sample_id"])): row
        for row in waveform_rows
    }
    if len(waveform_index) != len(waveform_rows):
        raise RuntimeError("duplicate 07A candidate-group identity")
    reliable_keys = {(int(row["class_id"]), str(row["coarse_segment_id"])) for row in reliable}
    gplus = [row for row in gplus
             if (int(row["class_id"]), str(row["reference_structure_id"])) in reliable_keys]
    if not gplus:
        raise RuntimeError("no reliable 06B G+ rows")
    missing_07a = [
        (args.source, int(row["class_id"]), str(row["reference_structure_id"]), int(row["sample_id"]))
        for row in gplus
        if (args.source, int(row["class_id"]), str(row["reference_structure_id"]), int(row["sample_id"]))
        not in waveform_keys
    ]
    if missing_07a:
        raise RuntimeError(f"07A/06B G+ identity mismatch: {missing_07a[:5]}")
    split = stratified_sample_split(gplus, seed=1)
    device = torch.device(args.device)
    spatial = load_raw_spatial_encoder(checkpoint, config, classes, device)
    grid = np.linspace(0., 365., args.grid_size)
    with torch.inference_mode():
        projections, latent_dim = fit_class_projections(spatial, dataset, len(classes), grid,
                                                        args.feature_batch_size, device,
                                                        bool(config.get("with_extra", False)))
        analyzer = BatchedDirectFourierAnalyzer(13, period_days=365., reg=.001).to(device)
        synthesizer = BatchedDirectFourierSynthesizer(13, period_days=365.).to(device)
        _, coefficients, records, baselines = project_dataset(
            spatial, dataset, projections, grid, args.feature_batch_size, device,
            bool(config.get("with_extra", False)), analyzer, collect_samplewise=True)
    fine_settings, coarse_settings = _detector_settings(manifest05)
    records_by_class = {class_id: {int(record["sample_id"]): record for record in values}
                        for class_id, values in records.items()}
    coefficient_by_class = {}
    for class_id, values in records.items():
        coefficient_by_class[class_id] = {
            int(record["sample_id"]): record["coefficients"] for record in values
        }
    references, invalid, sample_cache, groups = {}, [], {}, []
    reliable_by_class = defaultdict(list)
    for row in reliable:
        reliable_by_class[int(row["class_id"])].append(row)
    train_ids = set(split["train"])
    with torch.inference_mode():
        for class_id, rows in sorted(reliable_by_class.items()):
            ids = [sample_id for sample_id in sorted(coefficient_by_class.get(class_id, {}))
                   if sample_id in train_ids]
            if not ids:
                invalid.extend(dict(class_id=class_id, reference_structure_id=row["coarse_segment_id"],
                                    valid_reference=False, invalid_reason="no_train_samples") for row in rows)
                continue
            train_coeff = np.stack([coefficient_by_class[class_id][sample_id] for sample_id in ids])
            prototype = reconstruct_class_prototype(train_coeff, synthesizer, device,
                                                     sample_batch_size=args.reconstruction_batch_size)
            prototype_pc1 = projections[class_id].transform(prototype[None])[0]
            events, chain, fine, coarse = _detect_full(
                prototype_pc1, baselines[class_id], fine_settings, coarse_settings, "TR"
            )
            del events, coarse
            for row in rows:
                reference_id = str(row["coarse_segment_id"])
                geometry = dict(row)
                geometry["fine_segment_ids"] = tuple(
                    edge.segment_id for edge in fine
                    if _inside_window(edge.center_day, geometry["start_day"], geometry["end_day"])[0]
                )
                query = np.linspace(float(geometry["start_day"]),
                                    float(geometry["end_day"]) +
                                    (365 if float(geometry["end_day"]) <= float(geometry["start_day"]) else 0), 32)
                instance = _instance_from_waveform(
                    sample_periodic_curve(prototype, query), geometry, chain, fine
                )
                state = validate_train_reference(instance)
                if not state["valid_reference"]:
                    invalid.append(dict(class_id=class_id, reference_structure_id=reference_id, **state))
                else:
                    references[(class_id, reference_id)] = instance
    pc1_curves = {class_id: reconstruct_projected_coefficients(
        coefficients[class_id], projections[class_id], np.arange(365.))
        for class_id in references_by_class if class_id in coefficients}
    record_indices = {class_id: {int(record["sample_id"]): index for index, record in enumerate(records[class_id])}
                      for class_id in records}
    gplus_index = defaultdict(list)
    for row in gplus:
        gplus_index[(int(row["class_id"]), int(row["sample_id"]))].append(row)
    for (class_id, sample_id), rows in sorted(gplus_index.items()):
        if class_id not in record_indices or sample_id not in record_indices[class_id]:
            raise RuntimeError(f"06B sample absent from source replay: {sample_id}")
        index = record_indices[class_id][sample_id]
        record = records[class_id][index]
        events, chain, fine, coarse = _detect_full(
            pc1_curves[class_id][index], baselines[class_id], fine_settings, coarse_settings,
            f"I{index}_")
        del events
        coarse_dicts = [asdict(item) for item in coarse]
        for row in rows:
            reference_id = str(row["reference_structure_id"])
            key = (class_id, reference_id)
            if key not in references:
                continue
            direction = next(item["direction"] for item in reliable_by_class[class_id]
                             if str(item["coarse_segment_id"]) == reference_id)
            candidates = [item for item in coarse_dicts if item["accepted"] and item["direction"] == direction]
            positive_id = str(row["matched_sample_segment_id"])
            ids = [str(item["coarse_segment_id"]) for item in candidates]
            audit_key = (args.source, class_id, reference_id, sample_id)
            expected_count = int(waveform_index[audit_key]["num_same_direction_candidates"])
            if len(ids) != expected_count:
                raise RuntimeError(
                    f"07A candidate-pool replay mismatch for {audit_key}: "
                    f"expected={expected_count}, current={len(ids)}"
                )
            if positive_id not in ids:
                raise RuntimeError(f"G+ positive missing from candidate pool: {positive_id}")
            candidate_instances = []
            for segment in candidates:
                cache_key = (class_id, sample_id, segment["coarse_segment_id"])
                if cache_key not in sample_cache:
                    item = _instance(record["coefficients"], segment, chain, fine)
                    sample_cache[cache_key] = item
                candidate_instances.append(sample_cache[cache_key])
            positive_index = ids.index(positive_id)
            group_id = f"{class_id}:{reference_id}:{sample_id}"
            groups.append(dict(group_id=group_id, class_id=class_id, sample_id=sample_id,
                               reference_structure_id=reference_id, direction=direction,
                               candidate_ids=ids, positive_id=positive_id,
                               positive_index=positive_index,
                               reference=references[key], candidates=candidate_instances))
    split_groups = assign_groups_to_split(groups, split)
    train_instances = []
    for group in split_groups["train"]:
        train_instances.append(group["reference"]); train_instances.extend(group["candidates"])
    stats = fit_normalization_statistics(train_instances)
    normalized_references = {
        key: _normalize_instance(value, stats) for key, value in references.items()
    }
    for partition in split_groups.values():
        for group in partition:
            key = (int(group["class_id"]), str(group["reference_structure_id"]))
            group["reference"] = normalized_references[key]
            group["candidates"] = [_normalize_instance(item, stats) for item in group["candidates"]]
    reference_summary = []
    for row in reliable:
        key = (int(row["class_id"]), str(row["coarse_segment_id"]))
        state = validate_train_reference(references[key]) if key in references else {
            "valid_reference": False, "invalid_reason": "train_reference_invalid"
        }
        reference_summary.append(dict(source=args.source, class_id=key[0],
                                      reference_structure_id=key[1],
                                      bootstrap_occurrence_rate=float(row["bootstrap_occurrence_rate"]), **state))
    return dict(groups=split_groups, split=split, stats=stats, references=normalized_references,
                reference_summary=reference_summary, latent_dim=latent_dim,
                classes=classes, task=task, checkpoint=checkpoint,
                input_hashes=dict(structure05=str(folder05 / "manifest.json"),
                                  support06b=str(support_path), waveform07a=str(waveform_path)))


def _metric_row(variant, split_name, rows, thresholds=None):
    record = dict(variant=variant, split=split_name, num_groups=len(rows),
                  num_multi_candidate=sum(int(row["num_candidates"]) >= 2 for row in rows),
                  **summarize_retrieval(rows))
    if thresholds is not None:
        reliability = evaluate_with_frozen_thresholds(rows, thresholds)
        record.update({key: reliability[key] for key in (
            "precision", "coverage", "accepted_correct", "accepted_wrong", "rejected")})
    return record


def _seed_run(prepared, seed, args, seed_dir, device):
    histories, validation_metrics, calibrations, test_metrics, predictions = [], [], [], [], []
    checkpoints, test_by_variant, reference_embeddings = {}, {}, {}
    for variant in VARIANTS:
        if variant not in args.variants:
            continue
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, state, history, runtime = _train_variant(
            variant, prepared["groups"], prepared["latent_dim"], seed, args, device)
        histories.extend(history); checkpoints[variant] = state
        validation = _evaluate_model(model, prepared["groups"]["validation"], device, args.batch_size)
        test = _evaluate_model(model, prepared["groups"]["test"], device, args.batch_size)
        threshold = calibrate_rejection_thresholds(validation, .95)
        calibration_row = dict(variant=variant, **threshold)
        calibrations.append(calibration_row)
        validation_metrics.append(_metric_row(variant, "validation", validation, threshold))
        total_params = sum(parameter.numel() for parameter in model.parameters())
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        test_metrics.append(dict(_metric_row(variant, "test", test, threshold),
                                 training_seconds=runtime, total_params=total_params,
                                 trainable_params=trainable_params,
                                 peak_gpu_memory_bytes=int(peak_memory)))
        evaluated = apply_rejection_thresholds(test, threshold)
        for row in evaluated:
            predictions.append(dict(variant=variant, **row))
        test_by_variant[variant] = evaluated
        if variant == "Fusion":
            model.eval()
            for key, reference in sorted(prepared["references"].items()):
                reference_embeddings[f"{key[0]}:{key[1]}"] = _encode(model, [reference], device)[0].detach().cpu().numpy()
    fixed_validation = _fixed_baseline(prepared["groups"]["validation"])
    fixed_test = _fixed_baseline(prepared["groups"]["test"])
    fixed_threshold = calibrate_rejection_thresholds(fixed_validation, .95)
    calibrations.append(dict(variant="07A_FIXED_MULTI_NED", **fixed_threshold))
    validation_metrics.append(_metric_row("07A_FIXED_MULTI_NED", "validation", fixed_validation, fixed_threshold))
    test_metrics.append(_metric_row("07A_FIXED_MULTI_NED", "test", fixed_test, fixed_threshold))
    fixed_test = apply_rejection_thresholds(fixed_test, fixed_threshold)
    for row in fixed_test:
        predictions.append(dict(variant="07A_FIXED_MULTI_NED", **row))
    seed_dir.mkdir(parents=True)
    write_json(seed_dir / "config.json", {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    })
    write_json(seed_dir / "split_manifest.json", dict(split_seed=1, split_by="class/sample_id", **prepared["split"]))
    write_json(seed_dir / "normalization_stats.json", prepared["stats"])
    write_csv(seed_dir / "train_history.csv", histories)
    write_csv(seed_dir / "validation_metrics.csv", validation_metrics)
    write_csv(seed_dir / "threshold_calibration.csv", calibrations)
    write_csv(seed_dir / "test_metrics.csv", test_metrics)
    write_csv(seed_dir / "test_pair_predictions.csv", predictions)
    write_csv(seed_dir / "reference_summary.csv", prepared["reference_summary"])
    np.savez(seed_dir / "reference_embeddings.npz", **reference_embeddings)
    torch.save(checkpoints, seed_dir / "best_model.pt")
    write_json(seed_dir / "manifest.json", dict(experiment="07B_structure_identity_encoder",
        source=args.source, task=prepared["task"], model_seed=seed, target_data_used=False,
        class_enters_network=False, absolute_time_enters_network=False,
        split_seed=1, variants=list(args.variants), fixed_baseline_same_test_split=True,
        train_only_normalization=True, train_only_reference=True,
        input_hashes=prepared["input_hashes"]))
    missing = [name for name in SEED_FILES if not (seed_dir / name).is_file()]
    if missing:
        raise RuntimeError("incomplete seed output: " + ", ".join(missing))
    return test_metrics, predictions


def _aggregate_source(seed_results, source_dir):
    metrics = [row for result, _ in seed_results for row in result]
    grouped = defaultdict(list)
    for row in metrics:
        grouped[row["variant"]].append(row)
    summary = []
    for variant, rows in sorted(grouped.items()):
        record = dict(variant=variant, num_seeds=len(rows))
        for name in ("top1", "top2", "mrr", "pairwise", "positive_margin", "unique_best", "precision", "coverage"):
            values = np.asarray([float(row.get(name, np.nan)) for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            record[f"{name}_mean"] = float(finite.mean()) if len(finite) else np.nan
            record[f"{name}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0 if len(finite) else np.nan
        summary.append(record)
    write_csv(source_dir / "three_seed_summary.csv", summary)
    write_csv(source_dir / "ablation_summary.csv", summary)
    transitions = []
    reliability = []
    for seed_index, (_, predictions) in enumerate(seed_results):
        by_variant = {variant: {row["group_id"]: row for row in predictions if row["variant"] == variant}
                      for variant in {row["variant"] for row in predictions}}
        for left, right, name in (("07A_FIXED_MULTI_NED", "Fusion", "FIXED_TO_FUSION"),
                                  ("Waveform", "Fusion", "WAVEFORM_TO_FUSION")):
            if left not in by_variant or right not in by_variant:
                continue
            common = sorted(set(by_variant[left]) & set(by_variant[right]))
            transitions.append(dict(model_seed=seed_index + 1, transition=name,
                wrong_to_correct=sum(not by_variant[left][key]["exact"] and by_variant[right][key]["exact"] for key in common),
                correct_to_wrong=sum(by_variant[left][key]["exact"] and not by_variant[right][key]["exact"] for key in common),
                num_groups=len(common)))
        for variant, rows in by_variant.items():
            accepted = [row for row in rows.values() if row["accepted"]]
            reliability.append(dict(model_seed=seed_index + 1, variant=variant,
                accepted_correct=sum(row["exact"] for row in accepted),
                accepted_wrong=sum(not row["exact"] for row in accepted),
                rejected=sum(not row["accepted"] for row in rows.values())))
    write_csv(source_dir / "transition_summary.csv", transitions)
    write_csv(source_dir / "reliability_summary.csv", reliability)


def _plot_transition_case(path, group, predictions, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = {candidate_id: item for candidate_id, item in
                  zip(group["candidate_ids"], group["candidates"])}
    reference = group["reference"]
    positive = candidates[group["positive_id"]]
    u = np.linspace(0.0, 1.0, 32)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    def scalar(item):
        values = np.asarray(item["raw_waveform"], dtype=float)
        return np.sqrt(np.mean((values - values[:1]) ** 2, axis=1))
    axes[0].plot(u, scalar(reference), lw=2.5, label="train-only reference")
    axes[0].plot(u, scalar(positive), lw=2.0, label="06B G+ positive")
    for variant, color in (("07A_FIXED_MULTI_NED", "#555555"),
                           ("Waveform", "#d97706"), ("Fusion", "#0f766e")):
        if variant not in predictions:
            continue
        row = predictions[variant]
        selected = candidates.get(str(row["selected_id"]))
        if selected is not None:
            axes[1].plot(u, scalar(selected), lw=2, color=color,
                         label=f"{variant}: {row['selected_id']} s={row['top1_similarity']:.3f} m={row['margin']:.3f}")
    for level, (label, item, color) in enumerate((
        ("reference", reference, "#245580"), ("positive", positive, "#2ca02c"))):
        locations = np.asarray(item["events"], dtype=float)
        types = np.asarray(item["event_types"], dtype=int)
        if len(locations):
            axes[2].scatter(locations[:, 0], np.full(len(locations), level),
                            marker="^", c=[color], s=50)
            for x, event_type in zip(locations[:, 0], types):
                axes[2].text(x, level + .08, "P" if event_type == 1 else "V", ha="center", fontsize=8)
    axes[2].set_yticks([0, 1], ["reference", "positive"])
    axes[2].set_xlabel("Relative structure time u")
    axes[0].set_ylabel("multivariate RMS")
    axes[1].set_ylabel("selected RMS")
    axes[2].set_ylabel("extrema chain")
    axes[0].legend(); axes[1].legend(fontsize=8)
    for axis in axes: axis.grid(alpha=.2)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160); plt.close(fig)


def _render_diagnostics(prepared, seed_results, source_dir, maximum=20):
    if not seed_results:
        return
    predictions = seed_results[0][1]
    by_variant = defaultdict(dict)
    for row in predictions:
        by_variant[row["variant"]][row["group_id"]] = row
    groups = {group["group_id"]: group for group in prepared["groups"]["test"]}
    specs = (
        ("07a_wrong_07b_correct", "07A_FIXED_MULTI_NED", "Fusion", False, True),
        ("07a_correct_07b_wrong", "07A_FIXED_MULTI_NED", "Fusion", True, False),
        ("waveform_wrong_fusion_correct", "Waveform", "Fusion", False, True),
    )
    for folder, left, right, left_state, right_state in specs:
        output = source_dir / folder
        output.mkdir(parents=True, exist_ok=True)
        common = sorted(set(groups) & set(by_variant.get(left, {})) & set(by_variant.get(right, {})))
        chosen = [key for key in common if bool(by_variant[left][key]["exact"]) == left_state
                  and bool(by_variant[right][key]["exact"]) == right_state][:maximum]
        for index, key in enumerate(chosen):
            packet = {name: rows[key] for name, rows in by_variant.items() if key in rows}
            _plot_transition_case(output / f"{index:02d}_{key.replace(':', '_')}.png",
                                  groups[key], packet, f"07B {prepared['task']} | {folder} | {key}")


def run(args):
    if args.source not in SOURCES:
        raise SystemExit(f"ERROR: unknown source: {args.source}")
    if not args.data_root.is_dir():
        raise SystemExit(f"ERROR: source data not found: {args.data_root}")
    prepared = _prepare_source(args)
    final = args.output_root / args.source
    with staged_source_output(final, SOURCE_FILES) as staging:
        seed_results = []
        for seed in args.model_seeds:
            seed_results.append(_seed_run(prepared, seed, args, staging / f"model_seed{seed}",
                                          torch.device(args.device)))
        _aggregate_source(seed_results, staging)
        _render_diagnostics(prepared, seed_results, staging)
    counts = {name: len({group["sample_id"] for group in prepared["groups"][name]})
              for name in ("train", "validation", "test")}
    print("07B_HEALTH|source={}|train_samples={}|val_samples={}|test_samples={}|references={}|invalid_references={}|multi_test={}|target_data_used=false".format(
        args.source, counts["train"], counts["validation"], counts["test"],
        sum(row["valid_reference"] for row in prepared["reference_summary"]),
        sum(not row["valid_reference"] for row in prepared["reference_summary"]),
        sum(len(group["candidates"]) >= 2 for group in prepared["groups"]["test"])), flush=True)


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--source", choices=tuple(SOURCES), required=True)
    value.add_argument("--source-checkpoint", type=Path, required=True)
    value.add_argument("--data-root", type=Path, default=Path("/data/user/dataset/timematch_data"))
    value.add_argument("--structure-view-root", type=Path, default=Path("outputs/shift_visualizations_seed1"))
    value.add_argument("--validity-root", type=Path, default=Path("outputs/06A_structure_reference_validity"))
    value.add_argument("--observation-root", type=Path, default=Path("outputs/shift_visualizations_seed1/06B_structure_observation_support"))
    value.add_argument("--waveform-root", type=Path, default=Path("outputs/shift_visualizations_seed1/07A_multivariate_local_waveform_scan"))
    value.add_argument("--output-root", type=Path, default=Path("outputs/shift_visualizations_seed1/07B_structure_identity_encoder"))
    value.add_argument("--model-seeds", type=int, nargs="+", default=[1, 2, 3])
    value.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    value.add_argument("--max-epochs", type=int, default=100)
    value.add_argument("--patience", type=int, default=15)
    value.add_argument("--batch-size", type=int, default=128)
    value.add_argument("--feature-batch-size", type=int, default=128)
    value.add_argument("--grid-size", type=int, default=128)
    value.add_argument("--reconstruction-batch-size", type=int, default=256)
    value.add_argument("--device", default="cuda")
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    run(arguments)
