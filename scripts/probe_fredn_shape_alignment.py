#!/usr/bin/env python3
"""Offline mode-9/mode-13 shape and coarse-to-fine alignment probe."""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fredn.structural_probe import (
    align_curve_to_landmark_template,
    build_direct_fourier_views,
    build_source_alignment_template,
    candidate_alignment_prediction,
    classification_metrics,
    coarse_fine_hierarchy,
    detect_structural_landmarks,
    fit_source_class_projections,
    nearest_prototype_prediction,
    project_curves,
    robust_signal_scale,
    shape_distance,
    stratified_bootstrap_macro_f1_delta,
    topology_signature,
)
from models.stclassifier import PseFreDNLTae


DOMAIN_PATHS = {
    "AT1": "austria/33UVP/2017",
    "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
}
DISTANCES = ("absolute", "correlation", "derivative")
PATHS = (
    "mode9_unaligned",
    "mode13_unaligned",
    "mode9_aligned_by_mode9",
    "mode13_aligned_by_mode9",
    "mode13_aligned_by_mode13",
)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hash_indices(indices):
    canonical = ",".join(str(int(value)) for value in sorted(indices))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def replay_protocol_splits(
    source_path,
    target_path,
    source_indices,
    target_indices,
    seed,
    fold_creator=None,
):
    """Replay, without rewriting, the two historical split invocations."""
    if fold_creator is None:
        from train import create_train_val_test_folds

        fold_creator = create_train_val_test_folds
    random.seed(seed)
    source_replay = fold_creator(
        [source_path, source_path],
        1,
        {source_path: list(source_indices)},
    )[0][source_path]
    random.seed(seed)
    target_replay = fold_creator(
        [source_path, target_path],
        1,
        {
            source_path: list(source_indices),
            target_path: list(target_indices),
        },
    )[0][target_path]
    return source_replay, target_replay


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def _load_state(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    state = payload.get("state_dict", payload)
    return {key.removeprefix("module."): value for key, value in state.items()}


def _load_anchor(path, expected_modes, num_classes, args, device):
    path = Path(path)
    config_path = path.parent.parent / "train_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "model": "psefrednltae",
        "fredn_num_modes": expected_modes,
        "fredn_fourier_solver": "dense_direct",
        "fredn_nufft_reg": args.fourier_reg,
        "fredn_period_days": args.period_days,
        "source": DOMAIN_PATHS[args.source],
        "target": DOMAIN_PATHS[args.source],
        "seed": args.seed,
        "num_folds": 1,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if path.parent.name != "fold_0":
        mismatches["fold"] = (path.parent.name, "fold_0")
    if mismatches:
        raise ValueError(f"checkpoint metadata mismatch: {mismatches}")
    model = PseFreDNLTae(
        input_dim=config.get("input_dim", args.input_dim),
        with_extra=config.get("with_extra", args.with_extra),
        num_classes=num_classes,
        fredn_num_modes=expected_modes,
        fredn_nufft_reg=args.fourier_reg,
        fredn_period_days=args.period_days,
        fredn_fourier_solver="dense_direct",
    )
    model.load_state_dict(_load_state(path, device), strict=True)
    model.to(device).eval().requires_grad_(False)
    return model.spatial_encoder, {
        "checkpoint": str(path.resolve()),
        "checkpoint_modes": expected_modes,
        "train_config": str(config_path.resolve()),
    }


def _protocol_classes(data_root, source_path):
    from dataset import PixelSetData
    from utils import label_utils

    candidates = [
        name
        for name in label_utils.get_classes(source_path.split("/")[0])
        if name != "unknown"
    ]
    dataset = PixelSetData(
        data_root,
        source_path,
        candidates,
        closed_set=True,
        combine_spring_and_winter=False,
    )
    labels, counts = np.unique(dataset.get_labels(), return_counts=True)
    classes = [
        candidates[int(label)]
        for label, count in zip(labels, counts)
        if count >= 200
    ]
    if not classes:
        raise ValueError("no source class satisfies the existing closed-set protocol")
    return classes


def _dataset(data_root, domain_path, classes, indices, args):
    from torchvision.transforms import transforms
    from dataset import PixelSetData
    from transforms import Normalize, RandomSamplePixels, ToTensor

    transform = transforms.Compose(
        [RandomSamplePixels(args.num_pixels), Normalize(), ToTensor()]
    )
    return PixelSetData(
        data_root,
        domain_path,
        classes,
        transform=transform,
        indices=indices,
        with_extra=args.with_extra,
        closed_set=True,
        combine_spring_and_winter=False,
    )


def _base_dataset(data_root, domain_path, classes, args):
    from dataset import PixelSetData

    return PixelSetData(
        data_root,
        domain_path,
        classes,
        with_extra=args.with_extra,
        closed_set=True,
        combine_spring_and_winter=False,
    )


def _loader(dataset, args):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def _extract_spatial(spatial_encoder, dataset, args, device):
    _seed_everything(args.seed)
    features, positions, labels, parcel_indices = [], [], [], []
    for sample in _loader(dataset, args):
        pixels = sample["pixels"].to(device)
        valid = sample["valid_pixels"].to(device)
        extra = sample["extra"].to(device) if args.with_extra else None
        features.append(spatial_encoder(pixels, valid, extra).cpu())
        positions.append(sample["positions"].cpu())
        labels.append(sample["label"].cpu())
        parcel_indices.append(sample["parcel_index"].cpu())
    if not features:
        raise ValueError(f"empty dataset split: {dataset.dataset_name}")
    return {
        "features": torch.cat(features),
        "positions": torch.cat(positions),
        "labels": torch.cat(labels).numpy(),
        "parcel_indices": torch.cat(parcel_indices).numpy(),
    }


@torch.inference_mode()
def _reconstruct_and_project(extracted, projections, grid, args, device):
    projected = {9: defaultdict(list), 13: defaultdict(list)}
    features = extracted["features"]
    positions = extracted["positions"]
    dense_template = torch.as_tensor(grid, dtype=features.dtype, device=device)
    for start in range(0, len(features), args.batch_size):
        spatial = features[start : start + args.batch_size].to(device)
        batch_positions = positions[start : start + args.batch_size].to(device)
        dense_positions = dense_template.unsqueeze(0).expand(len(spatial), -1)
        # Both resolutions consume this exact spatial tensor; PSE is not rerun.
        views = build_direct_fourier_views(
            spatial,
            batch_positions,
            dense_positions,
            mode_counts=(9, 13),
            period_days=args.period_days,
            reg=args.fourier_reg,
        )
        for mode, values in views.items():
            for class_id, curves in project_curves(values, projections).items():
                projected[mode][class_id].append(curves.cpu().numpy())
    return {
        mode: {
            class_id: np.concatenate(chunks)
            for class_id, chunks in class_chunks.items()
        }
        for mode, class_chunks in projected.items()
    }


def _source_grid(source_dataset, step):
    """Use a source-only canonical grid; Fourier synthesis needs no extrapolation."""
    start = min(source_dataset.date_positions)
    end = max(source_dataset.date_positions)
    if end <= start:
        raise ValueError("source temporal support is empty")
    grid = np.arange(np.ceil(start), np.floor(end) + step * 0.5, step)
    if len(grid) < 3:
        raise ValueError("common temporal support has fewer than three points")
    return grid


def _groups(labels):
    result = defaultdict(list)
    for index, label in enumerate(labels):
        result[int(label)].append(index)
    return result


def _fit_source_objects(source_views, source_labels, grid, args):
    groups = _groups(source_labels)
    prototypes = {9: {}, 13: {}}
    thresholds = {9: {}, 13: {}}
    templates = {"9_to_9": {}, "9_to_13": {}, "13_to_13": {}}
    for class_id, indices in groups.items():
        curves9 = source_views[9][class_id][indices]
        curves13 = source_views[13][class_id][indices]
        prototypes[9][class_id] = np.median(curves9, axis=0)
        prototypes[13][class_id] = np.median(curves13, axis=0)
        thresholds[9][class_id] = robust_signal_scale(curves9) * args.prominence_rel
        thresholds[13][class_id] = robust_signal_scale(curves13) * args.prominence_rel
        templates["9_to_9"][class_id] = build_source_alignment_template(
            grid, curves9, curves9, thresholds[9][class_id], args.min_distance_days
        )
        templates["9_to_13"][class_id] = build_source_alignment_template(
            grid, curves9, curves13, thresholds[9][class_id], args.min_distance_days
        )
        templates["13_to_13"][class_id] = build_source_alignment_template(
            grid, curves13, curves13, thresholds[13][class_id], args.min_distance_days
        )
    return prototypes, thresholds, templates


def _sample_curves(views, index):
    return {
        mode: {class_id: curves[index] for class_id, curves in by_class.items()}
        for mode, by_class in views.items()
    }


def _predict_all(views, grid, prototypes, thresholds, templates, distance, args):
    count = len(next(iter(views[9].values())))
    predictions = {path: [] for path in PATHS}
    details = []
    for index in range(count):
        curves = _sample_curves(views, index)
        unaligned9 = nearest_prototype_prediction(curves[9], prototypes[9], distance)
        unaligned13 = nearest_prototype_prediction(curves[13], prototypes[13], distance)
        aligned9 = candidate_alignment_prediction(
            grid, curves[9], curves[9], templates["9_to_9"], thresholds[9],
            prototypes[9], distance, args.min_distance_days,
        )
        aligned13_by9 = candidate_alignment_prediction(
            grid, curves[9], curves[13], templates["9_to_13"], thresholds[9],
            prototypes[13], distance, args.min_distance_days,
        )
        aligned13_by13 = candidate_alignment_prediction(
            grid, curves[13], curves[13], templates["13_to_13"], thresholds[13],
            prototypes[13], distance, args.min_distance_days,
        )
        predictions["mode9_unaligned"].append(unaligned9.prediction)
        predictions["mode13_unaligned"].append(unaligned13.prediction)
        predictions["mode9_aligned_by_mode9"].append(aligned9.prediction)
        predictions["mode13_aligned_by_mode9"].append(aligned13_by9.prediction)
        predictions["mode13_aligned_by_mode13"].append(aligned13_by13.prediction)
        details.append(
            {
                "index": index,
                "unaligned9": unaligned9,
                "unaligned13": unaligned13,
                "aligned9": aligned9,
                "aligned13_by9": aligned13_by9,
                "aligned13_by13": aligned13_by13,
            }
        )
    return {key: np.asarray(value) for key, value in predictions.items()}, details


def _per_class_metrics(labels, predictions, details, classes, anchor, split, distance):
    rows = []
    for class_id in sorted(np.unique(labels)):
        selected = np.flatnonzero(labels == class_id)
        alignment_details = [details[index]["aligned13_by9"] for index in selected]
        aligned = sum(item.used_alignment for item in alignment_details)
        fallback = len(selected) - aligned
        row = {
            "anchor": anchor,
            "split": split,
            "distance": distance,
            "class_id": int(class_id),
            "class_name": classes[int(class_id)],
            "target_n": int(len(selected)),
            "aligned_n": int(aligned),
            "fallback_n": int(fallback),
            "coverage": aligned / len(selected) if len(selected) else float("nan"),
            "gate_pass_n": int(aligned),
            "gate_coverage": aligned / len(selected) if len(selected) else float("nan"),
            "eligible_class_count_mean": float(
                np.mean([len(item.eligible_classes) for item in alignment_details])
            ),
        }
        for path, values in predictions.items():
            row[f"{path}_accuracy"] = float(np.mean(values[selected] == labels[selected]))
        aligned_indices = np.asarray(
            [
                index
                for index in selected
                if details[index]["aligned13_by9"].used_alignment
            ],
            dtype=np.int64,
        )
        row["mode13_aligned_by_mode9_aligned_subset_accuracy"] = (
            float(
                np.mean(
                    predictions["mode13_aligned_by_mode9"][aligned_indices]
                    == labels[aligned_indices]
                )
            )
            if len(aligned_indices)
            else float("nan")
        )
        margins = []
        for index in selected:
            distances = dict(details[index]["aligned13_by9"].distances)
            if int(class_id) in distances:
                negatives = [value for key, value in distances.items() if key != class_id]
                if negatives:
                    margins.append(min(negatives) - distances[int(class_id)])
        row["positive_margin_rate"] = (
            float(np.mean(np.asarray(margins) > 0)) if margins else float("nan")
        )
        row["margin_mean"] = float(np.mean(margins)) if margins else float("nan")
        rows.append(row)
    return rows


def _margin_summary(labels, details, per_class_rows):
    margins = []
    for index, true_class in enumerate(labels):
        distances = dict(details[index]["aligned13_by9"].distances)
        true_class = int(true_class)
        negatives = [value for key, value in distances.items() if key != true_class]
        if true_class in distances and negatives:
            margins.append(min(negatives) - distances[true_class])
    eligible_rows = [row for row in per_class_rows if np.isfinite(row["positive_margin_rate"])]
    aligned_counts = [row["aligned_n"] for row in per_class_rows]
    total_aligned = sum(aligned_counts)
    result = {
        "mean_margin": float(np.mean(margins)) if margins else float("nan"),
        "median_margin": float(np.median(margins)) if margins else float("nan"),
        "positive_margin_rate_micro": (
            float(np.mean(np.asarray(margins) > 0)) if margins else float("nan")
        ),
        "positive_margin_rate_macro": (
            float(np.mean([row["positive_margin_rate"] for row in eligible_rows]))
            if eligible_rows else float("nan")
        ),
        "eligible_class_count": len(eligible_rows),
        "dominant_class_fraction": (
            max(aligned_counts) / total_aligned if total_aligned else float("nan")
        ),
    }
    for minimum in (20, 50):
        selected = [row for row in per_class_rows if row["aligned_n"] >= minimum]
        result[f"classes_aligned_n_ge_{minimum}"] = len(selected)
        result[f"macro_accuracy_aligned_n_ge_{minimum}"] = (
            float(np.mean([row["mode13_aligned_by_mode9_accuracy"] for row in selected]))
            if selected else float("nan")
        )
        result[f"macro_positive_margin_rate_aligned_n_ge_{minimum}"] = (
            float(np.mean([row["positive_margin_rate"] for row in selected]))
            if selected else float("nan")
        )
    return result


def _fisher_rows(labels, views, prototypes, predictions, details, anchor, distance):
    rows = []
    for path in ("mode9_unaligned", "mode13_unaligned", "mode13_aligned_by_mode9"):
        intra, inter = [], []
        for index, true_class in enumerate(labels):
            true_class = int(true_class)
            if path == "mode9_unaligned":
                distances = dict(
                    nearest_prototype_prediction(
                        _sample_curves(views, index)[9], prototypes[9], distance
                    ).distances
                )
            elif path == "mode13_unaligned":
                distances = dict(details[index]["unaligned13"].distances)
            else:
                distances = dict(details[index]["aligned13_by9"].distances)
            if true_class not in distances:
                continue
            negatives = [value for key, value in distances.items() if key != true_class]
            if negatives:
                intra.append(distances[true_class])
                inter.append(float(np.mean(negatives)))
        d_intra = float(np.mean(intra)) if intra else float("nan")
        d_inter = float(np.mean(inter)) if inter else float("nan")
        rows.append(
            {
                "anchor": anchor,
                "distance": distance,
                "path": path,
                "d_intra": d_intra,
                "d_inter": d_inter,
                "r_shape": d_inter / max(d_intra, 1e-12),
                "num_samples": len(intra),
            }
        )
    return rows


def _oracle_metrics(labels, views, grid, templates, thresholds, prototypes, distance, args):
    before_distances = []
    after_distances = []
    aligned_count = 0
    for index, true_class in enumerate(labels):
        true_class = int(true_class)
        curves = _sample_curves(views, index)
        landmarks = detect_structural_landmarks(
            grid, curves[9][true_class], args.min_distance_days, thresholds[9][true_class]
        )
        template = templates["9_to_13"][true_class]
        if topology_signature(landmarks) != template.signature or not template.canonical_landmarks:
            continue
        aligned_count += 1
        warped = align_curve_to_landmark_template(
            grid,
            curves[13][true_class],
            landmarks,
            template.canonical_landmarks,
        )
        before_distances.append(
            shape_distance(curves[13][true_class], prototypes[13][true_class], distance)
        )
        after_distances.append(
            shape_distance(warped, template.aligned_prototype, distance)
        )
    before = float(np.mean(before_distances)) if before_distances else float("nan")
    after = float(np.mean(after_distances)) if after_distances else float("nan")
    return {
        "total_n": len(labels),
        "aligned_n": aligned_count,
        "coverage": aligned_count / len(labels) if len(labels) else float("nan"),
        "true_class_distance_before": before,
        "true_class_distance_after": after,
        "true_class_distance_reduction": before - after,
    }


def _hierarchy_rows(views, grid, thresholds, prototypes, anchor, split):
    rows = []
    count = len(next(iter(views[9].values())))
    for index in range(count):
        coarse_prediction = nearest_prototype_prediction(
            {class_id: curves[index] for class_id, curves in views[9].items()},
            prototypes[9],
            "absolute",
        ).prediction
        coarse = detect_structural_landmarks(
            grid, views[9][coarse_prediction][index],
            min_distance_days=0.0,
            prominence_threshold=thresholds[9][coarse_prediction],
        )
        fine = detect_structural_landmarks(
            grid, views[13][coarse_prediction][index],
            min_distance_days=0.0,
            prominence_threshold=thresholds[13][coarse_prediction],
        )
        rows.append({"anchor": anchor, "split": split, **coarse_fine_hierarchy(coarse, fine)})
    return rows


def _plot_classes(output_dir, classes, grid, source_labels, target_labels, source_views, target_views, templates, thresholds, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_groups, target_groups = _groups(source_labels), _groups(target_labels)
    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for class_id in sorted(set(source_groups) & set(target_groups)):
        source_indices, target_indices = source_groups[class_id], target_groups[class_id]
        name = classes[class_id].replace("/", "_").replace(" ", "_")
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for axis, mode in zip(axes, (9, 13)):
            source = source_views[mode][class_id][source_indices]
            target = target_views[mode][class_id][target_indices]
            axis.plot(grid, np.median(source, axis=0), label=f"source m{mode}")
            axis.plot(grid, np.median(target, axis=0), label=f"target m{mode}")
            if mode == 9:
                for item in templates["9_to_9"][class_id].canonical_landmarks:
                    axis.axvline(item.time, color="black", alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(figures / f"{name}_coarse_fine_before.png", dpi=150)
        plt.close(figure)

        aligned_source, aligned_target = [], []
        representative = None
        template = templates["9_to_13"][class_id]
        for dataset_views, indices, output in (
            (source_views, source_indices, aligned_source),
            (target_views, target_indices, aligned_target),
        ):
            for index in indices:
                landmarks = detect_structural_landmarks(
                    grid, dataset_views[9][class_id][index],
                    args.min_distance_days,
                    thresholds[9][class_id],
                )
                if topology_signature(landmarks) == template.signature and template.canonical_landmarks:
                    aligned_curve = align_curve_to_landmark_template(
                        grid, dataset_views[13][class_id][index], landmarks,
                        template.canonical_landmarks,
                    )
                    output.append(aligned_curve)
                    if dataset_views is target_views and representative is None:
                        representative = (index, landmarks, aligned_curve)
        if aligned_source and aligned_target:
            figure, axis = plt.subplots(figsize=(10, 4))
            for values, label, color in (
                (np.stack(aligned_source), "source aligned m13", "tab:blue"),
                (np.stack(aligned_target), "target aligned m13", "tab:orange"),
            ):
                median = np.median(values, axis=0)
                lower, upper = np.quantile(values, [0.25, 0.75], axis=0)
                axis.plot(grid, median, color=color, label=label)
                axis.fill_between(grid, lower, upper, color=color, alpha=0.2)
            for item in template.canonical_landmarks:
                axis.axvline(item.time, color="black", alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(figures / f"{name}_coarse_fine_after.png", dpi=150)
            plt.close(figure)
        if representative is not None:
            index, landmarks, aligned_curve = representative
            figure, axis = plt.subplots(figsize=(10, 4))
            axis.plot(grid, target_views[9][class_id][index], label="target mode9")
            axis.plot(grid, target_views[13][class_id][index], label="target mode13")
            axis.plot(grid, aligned_curve, label="mode13 warped by mode9")
            axis.plot(grid, template.aligned_prototype, label="source aligned mode13 prototype")
            for item in landmarks:
                axis.axvline(item.time, color="black", alpha=0.18)
            axis.legend()
            figure.tight_layout()
            figure.savefig(figures / f"{name}_target_sample_alignment.png", dpi=150)
            plt.close(figure)


def _summary_text(rows, bootstrap_rows):
    target = [row for row in rows if row["split"] == "target_test" and row["distance"] == "absolute"]
    by_anchor = defaultdict(dict)
    for row in target:
        by_anchor[row["anchor"]][row["path"]] = row["macro_f1"]
    lines = [
        "# FreDN coarse-to-fine shape alignment probe",
        "",
        "Target labels were used only for final metrics, per-class analysis, and bootstrap.",
        "",
    ]
    for row in target:
        lines.append(
            f"- {row['anchor']} / {row['path']}: accuracy={row['accuracy']:.4f}, "
            f"macro-F1={row['macro_f1']:.4f}"
        )
    lines.extend(["", "Bootstrap comparisons:"])
    for row in bootstrap_rows:
        qualified = row["ci_lower"] > 0
        lines.append(
            f"- {row['anchor']} {row['comparison']}: mean={row['mean_delta']:.4f}, "
            f"95% CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], "
            f"clear improvement={str(qualified).lower()}"
        )
    lines.extend(["", "Case assessment by anchor (absolute distance):"])
    for anchor, values in sorted(by_anchor.items()):
        source_values = {
            row["path"]: row["macro_f1"]
            for row in rows
            if row["anchor"] == anchor
            and row["split"] == "source_heldout"
            and row["distance"] == "absolute"
        }
        fine_better = source_values.get("mode13_unaligned", -np.inf) > source_values.get("mode9_unaligned", np.inf)
        alignment_better = values.get("mode13_aligned_by_mode9", -np.inf) > values.get("mode13_unaligned", np.inf)
        if fine_better and alignment_better:
            case = "A: mode13 adds source shape discrimination and mode9-guided alignment improves target F1"
        elif fine_better:
            case = "B: mode13 adds source shape discrimination, but alignment does not improve target F1"
        elif alignment_better:
            case = "C: alignment improves target F1 without clear extra mode13 source discrimination"
        else:
            case = "D: no evidence supporting the coarse-to-fine hypothesis"
        lines.append(f"- {anchor}: {case}.")
    signs = []
    for values in by_anchor.values():
        if "mode13_aligned_by_mode9" in values and "mode13_unaligned" in values:
            signs.append(np.sign(values["mode13_aligned_by_mode9"] - values["mode13_unaligned"]))
    if len(signs) == 2 and signs[0] != signs[1]:
        lines.extend(["", "Conclusion: encoder-sensitive; anchor directions disagree."])
    else:
        lines.extend(["", "Conclusion: anchor directions are consistent for the primary comparison."])
    return "\n".join(lines) + "\n"


def run(args):
    print("TARGET_LABEL_USED_FOR_ALIGNMENT=false")
    print("TARGET_LABEL_USED_FOR_PROTOTYPE=false")
    print("TARGET_LABEL_USED_FOR_EVALUATION_ONLY=true")
    source_path, target_path = DOMAIN_PATHS[args.source], DOMAIN_PATHS[args.target]
    if args.source != "AT1" or args.target != "DK1":
        raise ValueError("this frozen probe is defined for AT1 -> DK1")
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = _protocol_classes(args.data_root, source_path)
    source_base = _base_dataset(args.data_root, source_path, classes, args)
    target_base = _base_dataset(args.data_root, target_path, classes, args)
    source_split, target_split = replay_protocol_splits(
        source_path,
        target_path,
        source_base.get_parcel_indices(),
        target_base.get_parcel_indices(),
        args.seed,
    )
    datasets = {
        "source_train": _dataset(args.data_root, source_path, classes, source_split["train"], args),
        "source_heldout": _dataset(args.data_root, source_path, classes, source_split["test"], args),
        "target_test": _dataset(args.data_root, target_path, classes, target_split["test"], args),
    }
    grid = _source_grid(datasets["source_train"], args.grid_step_days)
    device = torch.device(args.device)
    all_summary, all_per_class, all_distance = [], [], []
    all_transfer, all_transfer_class, all_predictions = [], [], []
    all_oracle, all_hierarchy, robustness, bootstrap_rows = [], [], [], []
    anchor_metadata = {}
    for anchor, checkpoint, checkpoint_modes in (
        ("anchor_9", args.mode9_checkpoint, 9),
        ("anchor_13", args.mode13_checkpoint, 13),
    ):
        spatial_encoder, metadata = _load_anchor(
            checkpoint, checkpoint_modes, len(classes), args, device
        )
        anchor_metadata[anchor] = metadata
        extracted = {
            name: _extract_spatial(spatial_encoder, dataset, args, device)
            for name, dataset in datasets.items()
        }
        source_tensor = extracted["source_train"]["features"]
        source_labels_tensor = torch.as_tensor(extracted["source_train"]["labels"])
        projections = fit_source_class_projections(source_tensor, source_labels_tensor)
        views = {
            name: _reconstruct_and_project(values, projections, grid, args, device)
            for name, values in extracted.items()
        }
        prototypes, thresholds, templates = _fit_source_objects(
            views["source_train"], extracted["source_train"]["labels"], grid, args
        )
        for split in ("source_heldout", "target_test"):
            labels = extracted[split]["labels"]
            for distance in DISTANCES:
                predictions, details = _predict_all(
                    views[split], grid, prototypes, thresholds, templates, distance, args
                )
                metrics_by_path = {}
                for path, values in predictions.items():
                    metrics = classification_metrics(labels, values, range(len(classes)))
                    metrics_by_path[path] = metrics
                    all_summary.append(
                        {"anchor": anchor, "split": split, "distance": distance, "path": path, **metrics}
                    )
                current_per_class = _per_class_metrics(
                    labels, predictions, details, classes, anchor, split, distance
                )
                all_per_class.extend(current_per_class)
                all_distance.extend(
                    {"record_type": "fisher", **row, "split": split}
                    for row in _fisher_rows(
                        labels, views[split], prototypes, predictions, details, anchor, distance
                    )
                )
                all_transfer.append(
                    {
                        "anchor": anchor,
                        "split": split,
                        "distance": distance,
                        "delta1_aligned13by9_minus_unaligned13": metrics_by_path["mode13_aligned_by_mode9"]["macro_f1"] - metrics_by_path["mode13_unaligned"]["macro_f1"],
                        "delta2_aligned13by9_minus_aligned9": metrics_by_path["mode13_aligned_by_mode9"]["macro_f1"] - metrics_by_path["mode9_aligned_by_mode9"]["macro_f1"],
                        "delta3_aligned13by9_minus_aligned13by13": metrics_by_path["mode13_aligned_by_mode9"]["macro_f1"] - metrics_by_path["mode13_aligned_by_mode13"]["macro_f1"],
                        "alignment_coverage": float(np.mean([item["aligned13_by9"].used_alignment for item in details])),
                        **_margin_summary(labels, details, current_per_class),
                    }
                )
                all_transfer_class.extend(current_per_class)
                if split == "target_test":
                    for index, item in enumerate(details):
                        aligned = item["aligned13_by9"]
                        signatures = dict(aligned.signatures)
                        all_predictions.append(
                            {
                                "anchor": anchor,
                                "distance": distance,
                                "sample_index": int(extracted[split]["parcel_indices"][index]),
                                "true_label": int(labels[index]),
                                "pred_unaligned13": int(predictions["mode13_unaligned"][index]),
                                "pred_aligned13_by9": int(predictions["mode13_aligned_by_mode9"][index]),
                                "mode9_signature": ";".join(f"{key}:{value[0]}:{value[1]}" for key, value in signatures.items()),
                                "num_eligible_classes": len(aligned.eligible_classes),
                                "eligible_classes": ";".join(map(str, aligned.eligible_classes)),
                                "chosen_class": aligned.prediction,
                                "chosen_distance": aligned.chosen_distance,
                                "used_alignment": aligned.used_alignment,
                                "used_fallback": aligned.used_fallback,
                            }
                        )
                    all_oracle.append(
                        {
                            "anchor": anchor,
                            "distance": distance,
                            **_oracle_metrics(
                                labels,
                                views[split],
                                grid,
                                templates,
                                thresholds,
                                prototypes,
                                distance,
                                args,
                            ),
                        }
                    )
                    for baseline_path, comparison in (
                        ("mode13_unaligned", "aligned13_by9_minus_unaligned13"),
                        ("mode9_aligned_by_mode9", "aligned13_by9_minus_aligned9"),
                    ):
                        bootstrap_rows.append(
                            {
                                "anchor": anchor,
                                "distance": distance,
                                "comparison": comparison,
                                **stratified_bootstrap_macro_f1_delta(
                                    labels,
                                    predictions["mode13_aligned_by_mode9"],
                                    predictions[baseline_path],
                                    args.bootstrap_repeats,
                                    args.seed,
                                    range(len(classes)),
                                ),
                            }
                        )
            all_hierarchy.extend(
                _hierarchy_rows(
                    views[split], grid, thresholds, prototypes, anchor, split
                )
            )
        _plot_classes(
            output_dir / anchor, classes, grid,
            extracted["source_train"]["labels"], extracted["target_test"]["labels"],
            views["source_train"], views["target_test"], templates, thresholds,
            args,
        )
        del spatial_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_index = {(row["anchor"], row["split"], row["distance"], row["path"]): row for row in all_summary}
    for anchor in ("anchor_9", "anchor_13"):
        for distance in DISTANCES:
            for path in ("mode9_unaligned", "mode13_unaligned"):
                source_metrics = summary_index[(anchor, "source_heldout", distance, path)]
                target_metrics = summary_index[(anchor, "target_test", distance, path)]
                all_distance.append(
                    {
                        "record_type": "domain_drop",
                        "anchor": anchor,
                        "distance": distance,
                        "path": path,
                        "source_accuracy": source_metrics["accuracy"],
                        "source_macro_f1": source_metrics["macro_f1"],
                        "target_accuracy": target_metrics["accuracy"],
                        "target_macro_f1": target_metrics["macro_f1"],
                        "delta_domain_macro_f1": target_metrics["macro_f1"] - source_metrics["macro_f1"],
                    }
                )
    for anchor in ("anchor_9", "anchor_13"):
        for distance in DISTANCES:
            values = {
                path: summary_index[(anchor, "target_test", distance, path)]["macro_f1"]
                for path in PATHS
            }
            robustness.append(
                {
                    "anchor": anchor,
                    "metric": f"target_macro_f1_{distance}",
                    "mode9": values["mode9_unaligned"],
                    "mode13_unaligned": values["mode13_unaligned"],
                    "mode13_aligned_by9": values["mode13_aligned_by_mode9"],
                }
            )
    manifest = {
        "probe": "mode9_mode13_shape_alignment",
        "git_commit": args.git_commit,
        "git_branch": args.git_branch,
        "git_dirty": args.git_dirty,
        "source": args.source,
        "target": args.target,
        "classes": classes,
        "seed": args.seed,
        "split_strategy": "replay_existing_protocol",
        "source_split_replay": [args.source, args.source],
        "target_split_replay": [args.source, args.target],
        "source_split_dataset_paths": [source_path, source_path],
        "target_split_dataset_paths": [source_path, target_path],
        "split_counts": {
            "source_train": len(source_split["train"]),
            "source_val": len(source_split["val"]),
            "source_test": len(source_split["test"]),
            "target_test": len(target_split["test"]),
        },
        "split_parcel_index_hashes": {
            "source_train": _hash_indices(source_split["train"]),
            "source_val": _hash_indices(source_split["val"]),
            "source_test": _hash_indices(source_split["test"]),
            "target_test": _hash_indices(target_split["test"]),
        },
        "shared_indices_across_anchors": True,
        "anchors": anchor_metadata,
        "fourier_modes": [9, 13],
        "fourier_reg": args.fourier_reg,
        "period_days": args.period_days,
        "target_label_policy": "evaluation_per_class_and_bootstrap_only",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_csv(output_dir / "shape_classification_summary.csv", all_summary)
    _write_csv(output_dir / "shape_classification_per_class.csv", all_per_class)
    _write_csv(output_dir / "shape_distance_summary.csv", all_distance)
    _write_csv(output_dir / "alignment_transfer_summary.csv", all_transfer + bootstrap_rows)
    _write_csv(output_dir / "alignment_transfer_per_class.csv", all_transfer_class)
    _write_csv(output_dir / "candidate_alignment_predictions.csv", all_predictions)
    _write_csv(output_dir / "oracle_upper_bound.csv", all_oracle)
    _write_csv(output_dir / "coarse_fine_hierarchy.csv", all_hierarchy)
    _write_csv(output_dir / "anchor_robustness.csv", robustness)
    (output_dir / "summary.md").write_text(
        _summary_text(all_summary, bootstrap_rows), encoding="utf-8"
    )
    print(f"PROBE_FINISHED output={output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source", choices=DOMAIN_PATHS, default="AT1")
    parser.add_argument("--target", choices=DOMAIN_PATHS, default="DK1")
    parser.add_argument("--mode9-checkpoint", required=True)
    parser.add_argument("--mode13-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-pixels", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--with-extra", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid-step-days", type=float, default=1.0)
    parser.add_argument("--prominence-rel", type=float, default=0.15)
    parser.add_argument("--min-distance-days", type=float, default=14.0)
    parser.add_argument("--fourier-reg", type=float, default=1e-3)
    parser.add_argument("--period-days", type=float, default=365.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--git-commit", default="unavailable")
    parser.add_argument("--git-branch", default="unavailable")
    parser.add_argument("--git-dirty", choices=("clean", "dirty", "unavailable"), default="unavailable")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
