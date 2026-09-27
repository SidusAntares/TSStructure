#!/usr/bin/env python3
"""Read-only audit of the V2-Clean structure representation chain."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.structure_da.discriminative_structure import normalized_candidate_concentration


CODE_VERSION = "0503bd6b02d7b0de8908c43ae0866b5b1acefae4"
DOMAINS = {
    "AT1": "austria/33UVP/2017",
    "DK1": "denmark/32VNH/2017",
    "FR2": "france/31TCJ/2017",
}
TASKS = {
    "FR2_DK1": ("FR2", "DK1"),
    "AT1_DK1": ("AT1", "DK1"),
}
REPRESENTATIONS = (
    "fourier_flat",
    "shape_token_flat",
    "anchor_ordered",
    "anchor_sorted",
    "shape_strength",
    "shape_response",
    "qshape",
)
REPRESENTATION_FIELDS = (
    "task", "class_protocol", "representation", "feature_dim",
    "source_val_macro_f1", "target_oracle_macro_f1",
    "source_to_target_macro_f1", "source_to_target_knn_macro_f1",
)
REPRESENTATION_PER_CLASS_FIELDS = (
    "task", "class_protocol", "representation", "class",
    "source_support", "target_support", "source_val_f1",
    "target_oracle_f1", "source_to_target_f1",
    "source_to_target_knn_f1",
)
ANCHOR_COVERAGE_FIELDS = (
    "task", "class_protocol", "domain", "class", "support",
    "mean_anchor_coverage", "std_anchor_coverage",
    "mean_anchor_margin", "std_anchor_margin",
)
HARD_CLASSES = (
    "horsebeans", "spring_barley", "spring_peas", "winter_triticale",
)


def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def compose_chain_representations(
    fourier, tokens, similarity, response_to_query, beta=5., eps=1e-12,
):
    """Compose the exact ordered, phase-free, pooled and query representations."""
    if fourier.ndim != 3 or tokens.ndim != 3 or similarity.ndim != 3:
        raise ValueError("fourier, tokens, and similarity must be rank-3 tensors")
    if tokens.shape[:2] != similarity.shape[:2]:
        raise ValueError("tokens and anchor similarity must share candidates")
    weights = torch.softmax(float(beta) * similarity, dim=1)
    strength = (weights * similarity).sum(dim=1)
    candidate_mask = torch.ones(
        similarity.shape[:2], dtype=torch.bool, device=similarity.device,
    )
    concentration = normalized_candidate_concentration(
        weights, candidate_mask=candidate_mask, eps=eps,
    )
    response = torch.cat((strength, concentration), dim=-1)
    return {
        "fourier_flat": fourier.flatten(1),
        "shape_token_flat": tokens.flatten(1),
        "anchor_ordered": similarity.flatten(1),
        "anchor_sorted": similarity.sort(dim=1).values.flatten(1),
        "shape_strength": strength,
        "shape_response": response,
        "qshape": response_to_query(response),
    }


def extract_chain_batch(model, pixels, valid_pixels, positions, extra):
    spatial = model.spatial_encoder(pixels, valid_pixels, extra)
    branch = model.structure_branch
    fourier, grid = branch.exposer(spatial, positions)
    window_groups, scales = branch.window_extractor(fourier)
    tokens = torch.cat(
        [branch.token_generator(windows) for windows in window_groups], dim=1,
    )
    similarity = branch.shapelet_dictionary.compute_similarity(tokens)
    representations = compose_chain_representations(
        fourier, tokens, similarity, branch.response_to_query,
        beta=branch.shapelet_dictionary.beta,
    )
    return representations, similarity, grid, scales


def audit_split_indices(
    source, target, eligible, seed, val_ratio, test_ratio, fold_creator,
):
    """Replay source-only and transfer splits from the same initial seed."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    source_split = fold_creator(
        [source, source], 1, {source: eligible[source]}, val_ratio, test_ratio,
    )[0]
    random.seed(int(seed))
    np.random.seed(int(seed))
    transfer_split = fold_creator(
        [source, target], 1, eligible, val_ratio, test_ratio,
    )[0]
    return {
        "source_train": source_split[source]["train"],
        "source_val": source_split[source]["val"],
        "target_val": transfer_split[target]["val"],
    }


def common_class_names(at1_classes, fr2_classes, dk1_available):
    return sorted(set(at1_classes) & set(fr2_classes) & set(dk1_available))


def remap_to_classes(features, labels, original_classes, selected_classes):
    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    original_classes = list(original_classes)
    selected_classes = list(selected_classes)
    selected_set = set(selected_classes)
    keep = np.asarray(
        [original_classes[int(label)] in selected_set for label in labels],
        dtype=bool,
    )
    new_index = {name: index for index, name in enumerate(selected_classes)}
    remapped = np.asarray(
        [new_index[original_classes[int(label)]] for label in labels[keep]],
        dtype=np.int64,
    )
    return features[keep], remapped


def _per_class_f1(labels, predictions, class_ids):
    return f1_score(
        labels, predictions, labels=np.asarray(class_ids),
        average=None, zero_division=0,
    )


def _macro_f1(labels, predictions, class_ids):
    values = _per_class_f1(labels, predictions, class_ids)
    return float(values.mean()) if values.size else float("nan")


def make_linear_probe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RidgeClassifier(alpha=1.0, class_weight="balanced")),
    ])


def fit_source_probe(
    source_train, source_train_labels, source_val, source_val_labels,
    target_val, target_val_labels, class_ids,
):
    estimator = make_linear_probe()
    estimator.fit(source_train, source_train_labels)
    source_prediction = estimator.predict(source_val)
    target_prediction = estimator.predict(target_val)
    return {
        "estimator": estimator,
        "source_val_predictions": source_prediction,
        "target_predictions": target_prediction,
        "source_val_macro_f1": _macro_f1(
            source_val_labels, source_prediction, class_ids,
        ),
        "source_val_per_class_f1": _per_class_f1(
            source_val_labels, source_prediction, class_ids,
        ),
        "source_to_target_macro_f1": _macro_f1(
            target_val_labels, target_prediction, class_ids,
        ),
        "source_to_target_per_class_f1": _per_class_f1(
            target_val_labels, target_prediction, class_ids,
        ),
    }


def target_oracle_probe(features, labels, class_ids, seed):
    """Cross-validate only inside target validation; support-one classes are NA."""
    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    counts = Counter(labels.tolist())
    available = np.asarray(
        [class_id for class_id in class_ids if counts[int(class_id)] >= 2],
        dtype=np.int64,
    )
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    per_class = np.full(class_ids.shape, np.nan, dtype=np.float64)
    if available.size < 2:
        return {
            "folds": 0, "available_classes": available,
            "predictions": predictions, "macro_f1": float("nan"),
            "per_class_f1": per_class,
        }
    selected = np.isin(labels, available)
    selected_features = features[selected]
    selected_labels = labels[selected]
    folds = min(5, min(counts[int(class_id)] for class_id in available))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    selected_predictions = np.full(selected_labels.shape, -1, dtype=np.int64)
    for train_indices, val_indices in splitter.split(selected_features, selected_labels):
        estimator = make_linear_probe()
        estimator.fit(selected_features[train_indices], selected_labels[train_indices])
        selected_predictions[val_indices] = estimator.predict(selected_features[val_indices])
    predictions[selected] = selected_predictions
    available_f1 = _per_class_f1(selected_labels, selected_predictions, available)
    location = {int(class_id): index for index, class_id in enumerate(class_ids)}
    for class_id, value in zip(available, available_f1):
        per_class[location[int(class_id)]] = value
    return {
        "folds": int(folds), "available_classes": available,
        "predictions": predictions, "macro_f1": float(available_f1.mean()),
        "per_class_f1": per_class,
    }


def cosine_knn_predictions(
    gallery, gallery_labels, queries, k=5, device="cpu", gallery_chunk=2048,
):
    """Exact cosine kNN with chunked gallery transfer and deterministic voting."""
    gallery = torch.as_tensor(gallery, dtype=torch.float32)
    queries = F.normalize(
        torch.as_tensor(queries, dtype=torch.float32), dim=-1,
    ).to(device)
    gallery_labels = np.asarray(gallery_labels, dtype=np.int64)
    k = min(int(k), int(gallery.shape[0]))
    if k < 1:
        raise ValueError("kNN gallery must not be empty")
    best_scores = torch.full(
        (queries.shape[0], k), -torch.inf, dtype=queries.dtype, device=device,
    )
    best_indices = torch.full(
        (queries.shape[0], k), -1, dtype=torch.long, device=device,
    )
    for start in range(0, gallery.shape[0], int(gallery_chunk)):
        block = F.normalize(gallery[start:start + gallery_chunk], dim=-1).to(device)
        similarity = queries @ block.T
        indices = torch.arange(
            start, start + block.shape[0], device=device,
        ).unsqueeze(0).expand(queries.shape[0], -1)
        candidate_scores = torch.cat((best_scores, similarity), dim=1)
        candidate_indices = torch.cat((best_indices, indices), dim=1)
        best_scores, order = candidate_scores.topk(k, dim=1)
        best_indices = candidate_indices.gather(1, order)
    neighbor_labels = gallery_labels[best_indices.cpu().numpy()]
    return np.asarray([
        np.bincount(row).argmax() for row in neighbor_labels
    ], dtype=np.int64)


def anchor_coverage_rows(
    task, class_protocol, domain, similarity, labels, class_names,
):
    similarity = np.asarray(similarity)
    labels = np.asarray(labels, dtype=np.int64)
    if similarity.ndim != 3 or similarity.shape[0] != labels.shape[0]:
        raise ValueError("similarity must be [B,N,M] and align with labels")
    ordered = np.sort(similarity, axis=-1)
    coverage = ordered[..., -1]
    margin = ordered[..., -1] - ordered[..., -2]
    rows = []
    for class_id, class_name in enumerate(class_names):
        selected = labels == class_id
        if not selected.any():
            continue
        class_coverage = coverage[selected].reshape(-1)
        class_margin = margin[selected].reshape(-1)
        rows.append({
            "task": task,
            "class_protocol": class_protocol,
            "domain": domain,
            "class": class_name,
            "support": int(selected.sum()),
            "mean_anchor_coverage": float(class_coverage.mean()),
            "std_anchor_coverage": float(class_coverage.std()),
            "mean_anchor_margin": float(class_margin.mean()),
            "std_anchor_margin": float(class_margin.std()),
        })
    return rows


def _pad_pixel_collate(samples):
    """Deterministically pad full parcels; valid_pixels preserves exact PSE pooling."""
    max_pixels = max(sample["pixels"].shape[-1] for sample in samples)
    pixels, masks = [], []
    for sample in samples:
        pad = max_pixels - sample["pixels"].shape[-1]
        pixels.append(F.pad(sample["pixels"], (0, pad)))
        masks.append(F.pad(sample["valid_pixels"], (0, pad)))
    return {
        "pixels": torch.stack(pixels),
        "valid_pixels": torch.stack(masks),
        "positions": torch.stack([sample["positions"] for sample in samples]),
        "extra": torch.stack([sample["extra"] for sample in samples]),
        "label": torch.stack([sample["label"] for sample in samples]),
    }


def deterministic_loader(dataset, batch_size, num_workers):
    shapes = dataset.get_shapes()
    order = sorted(range(len(shapes)), key=lambda index: (shapes[index][2], index))
    batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=batches, num_workers=int(num_workers),
        collate_fn=_pad_pixel_collate, pin_memory=torch.cuda.is_available(),
    )


def load_source_model(checkpoint, device):
    from train import create_model

    packet = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(packet.get("config"), dict):
        raise ValueError(f"checkpoint config missing: {checkpoint}")
    config = SimpleNamespace(**packet["config"])
    model = create_model(config)
    model.load_state_dict(packet["state_dict"], strict=True)
    return model.to(device).eval(), config


def _dataset(data_root, dataset_name, config, indices=None):
    from dataset import PixelSetData
    from torchvision.transforms import transforms
    from transforms import Normalize, ToTensor

    return PixelSetData(
        data_root, dataset_name, config.classes,
        transform=transforms.Compose([Normalize(), ToTensor()]),
        indices=indices, with_extra=config.with_extra, closed_set=True,
        combine_spring_and_winter=config.combine_spring_and_winter,
    )


def build_audit_datasets(config, source_name, target_name, data_root, seed):
    from train import create_train_val_test_folds

    bare_source = _dataset(data_root, source_name, config)
    bare_target = _dataset(data_root, target_name, config)
    eligible = {
        source_name: bare_source.get_parcel_indices().tolist(),
        target_name: bare_target.get_parcel_indices().tolist(),
    }
    split = audit_split_indices(
        source_name, target_name, eligible, seed,
        config.val_ratio, config.test_ratio, create_train_val_test_folds,
    )
    datasets = {
        "source_train": _dataset(
            data_root, source_name, config, split["source_train"],
        ),
        "source_val": _dataset(
            data_root, source_name, config, split["source_val"],
        ),
        "target_val": _dataset(
            data_root, target_name, config, split["target_val"],
        ),
    }
    return datasets, split


@torch.no_grad()
def extract_dataset(model, dataset, batch_size, num_workers, device):
    collected = {name: [] for name in REPRESENTATIONS}
    similarities, labels = [], []
    for batch in deterministic_loader(dataset, batch_size, num_workers):
        pixels = batch["pixels"].to(device, non_blocking=True)
        valid = batch["valid_pixels"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        extra = batch["extra"].to(device, non_blocking=True)
        values, similarity, _, _ = extract_chain_batch(
            model, pixels, valid, positions, extra,
        )
        for name, value in values.items():
            collected[name].append(value.detach().float().cpu())
        similarities.append(similarity.detach().float().cpu())
        labels.append(batch["label"].long().cpu())
    return {
        "representations": {
            name: torch.cat(values).numpy() for name, values in collected.items()
        },
        "similarity": torch.cat(similarities).numpy(),
        "labels": torch.cat(labels).numpy(),
    }


def available_target_classes(data_root, target_name, candidates, combine=False):
    from dataset import PixelSetData

    data = PixelSetData(
        data_root, target_name, candidates, closed_set=True,
        combine_spring_and_winter=combine,
    )
    present = np.unique(data.get_labels())
    return [candidates[int(index)] for index in present]


def resolve_common_classes(checkpoint_root, data_root):
    packets = {}
    for source in ("AT1", "FR2"):
        checkpoint = (
            Path(checkpoint_root) / f"source_{source}_seed1" / "fold_0" / "model.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"source checkpoint not found: {checkpoint.resolve()}")
        packets[source] = torch.load(checkpoint, map_location="cpu", weights_only=False)
    at1 = list(packets["AT1"]["config"]["classes"])
    fr2 = list(packets["FR2"]["config"]["classes"])
    candidates = sorted(set(at1) & set(fr2))
    combine = bool(packets["AT1"]["config"].get("combine_spring_and_winter", False))
    dk1 = available_target_classes(data_root, DOMAINS["DK1"], candidates, combine)
    return common_class_names(at1, fr2, dk1)


def _protocol_arrays(extracted, original_classes, selected_classes):
    labels = extracted["labels"]
    remapped = {}
    remapped_labels = None
    for name, values in extracted["representations"].items():
        remapped[name], candidate_labels = remap_to_classes(
            values, labels, original_classes, selected_classes,
        )
        if remapped_labels is None:
            remapped_labels = candidate_labels
        elif not np.array_equal(remapped_labels, candidate_labels):
            raise RuntimeError("representation label remapping diverged")
    similarity, similarity_labels = remap_to_classes(
        extracted["similarity"], labels, original_classes, selected_classes,
    )
    if not np.array_equal(remapped_labels, similarity_labels):
        raise RuntimeError("similarity label remapping diverged")
    return {"representations": remapped, "similarity": similarity, "labels": remapped_labels}


def evaluate_protocol(
    task, protocol, class_names, source_train, source_val, target_val,
    seed, knn_device,
):
    class_ids = np.arange(len(class_names), dtype=np.int64)
    metrics_rows, class_rows = [], []
    source_support = np.bincount(source_val["labels"], minlength=len(class_names))
    target_support = np.bincount(target_val["labels"], minlength=len(class_names))
    for representation in REPRESENTATIONS:
        source = fit_source_probe(
            source_train["representations"][representation], source_train["labels"],
            source_val["representations"][representation], source_val["labels"],
            target_val["representations"][representation], target_val["labels"],
            class_ids,
        )
        oracle = target_oracle_probe(
            target_val["representations"][representation], target_val["labels"],
            class_ids, seed,
        )
        knn_prediction = cosine_knn_predictions(
            source_train["representations"][representation], source_train["labels"],
            target_val["representations"][representation], k=5,
            device=knn_device,
        )
        knn_per_class = _per_class_f1(target_val["labels"], knn_prediction, class_ids)
        metrics_rows.append({
            "task": task, "class_protocol": protocol,
            "representation": representation,
            "feature_dim": int(source_train["representations"][representation].shape[1]),
            "source_val_macro_f1": source["source_val_macro_f1"],
            "target_oracle_macro_f1": oracle["macro_f1"],
            "source_to_target_macro_f1": source["source_to_target_macro_f1"],
            "source_to_target_knn_macro_f1": float(knn_per_class.mean()),
        })
        for class_id, class_name in enumerate(class_names):
            class_rows.append({
                "task": task, "class_protocol": protocol,
                "representation": representation, "class": class_name,
                "source_support": int(source_support[class_id]),
                "target_support": int(target_support[class_id]),
                "source_val_f1": source["source_val_per_class_f1"][class_id],
                "target_oracle_f1": oracle["per_class_f1"][class_id],
                "source_to_target_f1": source["source_to_target_per_class_f1"][class_id],
                "source_to_target_knn_f1": knn_per_class[class_id],
            })
    coverage = []
    coverage.extend(anchor_coverage_rows(
        task, protocol, "source", source_val["similarity"],
        source_val["labels"], class_names,
    ))
    coverage.extend(anchor_coverage_rows(
        task, protocol, "target", target_val["similarity"],
        target_val["labels"], class_names,
    ))
    return metrics_rows, class_rows, coverage


def checkpoint_for(checkpoint_root, source):
    return Path(checkpoint_root) / f"source_{source}_seed1" / "fold_0" / "model.pt"


def run_task(args):
    source_alias, target_alias = TASKS[args.task]
    source_name, target_name = DOMAINS[source_alias], DOMAINS[target_alias]
    checkpoint = checkpoint_for(args.checkpoint_root, source_alias)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {checkpoint.resolve()}")
    seed_all(args.seed)
    device = torch.device(args.device)
    model, config = load_source_model(checkpoint, device)
    datasets, split = build_audit_datasets(
        config, source_name, target_name, args.data_root, args.seed,
    )
    extracted = {
        name: extract_dataset(
            model, dataset, args.batch_size, args.num_workers, device,
        ) for name, dataset in datasets.items()
    }
    common = resolve_common_classes(args.checkpoint_root, args.data_root)
    protocols = {
        "task_native": list(config.classes),
        "common_classes": common,
    }
    metric_rows, class_rows, coverage_rows = [], [], []
    for protocol, classes in protocols.items():
        prepared = {
            name: _protocol_arrays(values, config.classes, classes)
            for name, values in extracted.items()
        }
        metrics, per_class, coverage = evaluate_protocol(
            args.task, protocol, classes,
            prepared["source_train"], prepared["source_val"],
            prepared["target_val"], args.seed, args.device,
        )
        metric_rows.extend(metrics)
        class_rows.extend(per_class)
        coverage_rows.extend(coverage)
    task_root = Path(args.output_root) / args.task
    write_csv(task_root / "representation_metrics.csv", metric_rows, REPRESENTATION_FIELDS)
    write_csv(
        task_root / "representation_per_class.csv", class_rows,
        REPRESENTATION_PER_CLASS_FIELDS,
    )
    write_csv(task_root / "anchor_coverage.csv", coverage_rows, ANCHOR_COVERAGE_FIELDS)
    manifest = {
        "commit": args.code_version,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "source": source_name,
        "target": target_name,
        "classes": list(config.classes),
        "common_classes": common,
        "split_counts": {name: len(indices) for name, indices in split.items()},
        "representations": list(REPRESENTATIONS),
        "feature_dimensions": {
            name: int(extracted["source_train"]["representations"][name].shape[1])
            for name in REPRESENTATIONS
        },
        "probe_config": {
            "linear": "StandardScaler+RidgeClassifier(alpha=1.0,class_weight=balanced)",
            "target_oracle_cv_max_folds": 5,
            "knn": "cosine,k=5,L2-normalized",
        },
        "test_split_accessed": False,
        "uda_checkpoint_used": False,
        "anchor_coverage_source_split": "source_val",
        "anchor_coverage_target_split": "target_val",
    }
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(f"REP_CHAIN_TASK_FINISHED|task={args.task}|output={task_root}")


def _print_summary(metrics, per_class, coverage):
    print("REP_CHAIN_SUMMARY")
    for row in metrics:
        print(
            f"task={row['task']}|protocol={row['class_protocol']}|"
            f"representation={row['representation']}|"
            f"source_f1={row['source_val_macro_f1']}|"
            f"target_oracle_f1={row['target_oracle_macro_f1']}|"
            f"source_to_target_f1={row['source_to_target_macro_f1']}|"
            f"knn_f1={row['source_to_target_knn_macro_f1']}"
        )
    coverage_index = {
        (row["task"], row["class_protocol"], row["domain"], row["class"]): row
        for row in coverage
    }
    print("FR2_HARD_CLASS_CHAIN")
    short_names = {
        "fourier_flat": "F", "shape_token_flat": "Z",
        "anchor_ordered": "S_ordered", "anchor_sorted": "S_sorted",
        "shape_response": "response", "qshape": "Qshape",
    }
    for row in per_class:
        if (
            row["task"] != "FR2_DK1"
            or row["class_protocol"] != "task_native"
            or row["class"] not in HARD_CLASSES
            or row["representation"] not in short_names
        ):
            continue
        anchor = coverage_index.get(
            ("FR2_DK1", "task_native", "target", row["class"]), {},
        )
        print(
            f"class={row['class']}|stage={short_names[row['representation']]}|"
            f"target_oracle_f1={row['target_oracle_f1']}|"
            f"source_to_target_f1={row['source_to_target_f1']}|"
            f"knn_f1={row['source_to_target_knn_f1']}|"
            f"anchor_coverage={anchor.get('mean_anchor_coverage', '')}"
        )


def merge_outputs(args):
    output_root = Path(args.output_root)
    metrics, per_class, coverage, manifests = [], [], [], {}
    for task in TASKS:
        task_root = output_root / task
        required = (
            task_root / "representation_metrics.csv",
            task_root / "representation_per_class.csv",
            task_root / "anchor_coverage.csv",
            task_root / "audit_manifest.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("incomplete audit output: " + ", ".join(missing))
        metrics.extend(read_csv(required[0]))
        per_class.extend(read_csv(required[1]))
        coverage.extend(read_csv(required[2]))
        manifests[task] = json.loads(required[3].read_text(encoding="utf-8"))
    write_csv(output_root / "representation_metrics.csv", metrics, REPRESENTATION_FIELDS)
    write_csv(
        output_root / "representation_per_class.csv", per_class,
        REPRESENTATION_PER_CLASS_FIELDS,
    )
    write_csv(output_root / "anchor_coverage.csv", coverage, ANCHOR_COVERAGE_FIELDS)
    (output_root / "audit_manifest.json").write_text(
        json.dumps({
            "commit": args.code_version,
            "seed": args.seed,
            "tasks": manifests,
            "test_split_accessed": False,
            "uda_checkpoint_used": False,
        }, indent=2),
        encoding="utf-8",
    )
    _print_summary(metrics, per_class, coverage)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument(
        "--checkpoint-root",
        default="outputs/structure_proto_v2clean_4tasks_seed1/source",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/structure_representation_chain_audit_seed1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--code-version", default=CODE_VERSION)
    return parser


def main():
    args = build_parser().parse_args()
    if args.merge:
        merge_outputs(args)
    elif args.task:
        run_task(args)
    else:
        raise SystemExit("one of --task or --merge is required")


if __name__ == "__main__":
    main()
