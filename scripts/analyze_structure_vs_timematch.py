#!/usr/bin/env python3
"""Unified read-only diagnostics for original and structure TimeMatch checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EPS = 1e-8


def _normalize(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), EPS)


def _macro_f1(truth, prediction, classes=None):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    classes = np.unique(np.concatenate((truth, prediction))) if classes is None else np.asarray(classes)
    scores = []
    for label in classes:
        tp = np.sum((truth == label) & (prediction == label))
        fp = np.sum((truth != label) & (prediction == label))
        fn = np.sum((truth == label) & (prediction != label))
        scores.append(2 * tp / max(2 * tp + fp + fn, EPS))
    return float(np.mean(scores)) if scores else float("nan")


def _js(first, second, axis=-1):
    first = np.maximum(np.asarray(first, dtype=np.float64), EPS)
    second = np.maximum(np.asarray(second, dtype=np.float64), EPS)
    first /= first.sum(axis=axis, keepdims=True)
    second /= second.sum(axis=axis, keepdims=True)
    middle = (first + second) / 2
    return .5 * np.sum(first * np.log(first / middle), axis=axis) + .5 * np.sum(second * np.log(second / middle), axis=axis)


def representation_metrics(source, source_labels, target, target_labels):
    source, target = _normalize(source), _normalize(target)
    source_labels, target_labels = np.asarray(source_labels), np.asarray(target_labels)
    common = sorted(set(source_labels.tolist()) & set(target_labels.tolist()))
    if not common:
        raise ValueError("no common classes for representation metrics")
    source_centers = {c: _normalize(source[source_labels == c].mean(0)) for c in common}
    target_centers = {c: _normalize(target[target_labels == c].mean(0)) for c in common}
    per_class = {c: {"domain_gap": float(1 - np.dot(source_centers[c], target_centers[c]))} for c in common}

    def inter(centers):
        pairs = [1 - np.dot(centers[a], centers[b]) for i, a in enumerate(common) for b in common[i + 1:]]
        return float(np.mean(pairs)) if pairs else float("nan")

    def intra(values, labels, centers):
        return float(np.mean([1 - np.dot(value, centers[int(label)]) for value, label in zip(values, labels)]))

    result = {
        "domain_gap": float(np.mean([row["domain_gap"] for row in per_class.values()])),
        "source_inter": inter(source_centers), "target_inter": inter(target_centers),
        "source_intra": intra(source, source_labels, source_centers),
        "target_intra": intra(target, target_labels, target_centers),
    }
    return result, per_class


def pseudo_label_metrics(truth, prediction, accepted, num_classes):
    truth, prediction, accepted = np.asarray(truth), np.asarray(prediction), np.asarray(accepted, dtype=bool)
    selected_truth, selected_prediction = truth[accepted], prediction[accepted]
    classes = np.arange(num_classes)
    per_class = {}
    for label in classes:
        gt = truth == label
        chosen = gt & accepted
        per_class[int(label)] = {
            "gt_count": int(gt.sum()), "accepted_count": int(chosen.sum()),
            "coverage": float(chosen.sum() / max(gt.sum(), 1)),
            "precision": float(np.mean(prediction[chosen] == label)) if chosen.any() else "",
        }
    if selected_truth.size:
        true_hist = np.bincount(selected_truth, minlength=num_classes)
        pred_hist = np.bincount(selected_prediction, minlength=num_classes)
        precision = float(np.mean(selected_truth == selected_prediction))
        macro = _macro_f1(selected_truth, selected_prediction, classes)
        js = float(_js(true_hist, pred_hist))
    else:
        precision = macro = js = ""
    return {
        "coverage": float(accepted.mean()), "precision": precision,
        "macro_f1": macro, "distribution_js": js,
    }, per_class


def prototype_geometry(features, labels, prototypes):
    features, prototypes = _normalize(features), _normalize(prototypes)
    labels = np.asarray(labels, dtype=np.int64)
    similarities = features @ prototypes.T
    predictions = similarities.argmax(1)
    correct = similarities[np.arange(len(labels)), labels]
    wrong = similarities.copy()
    wrong[np.arange(len(labels)), labels] = -np.inf
    margins = correct - wrong.max(1)
    per_class = {}
    for label in np.unique(labels):
        chosen = labels == label
        per_class[int(label)] = {
            "margin_mean": float(margins[chosen].mean()),
            "nearest_accuracy": float(np.mean(predictions[chosen] == label)),
        }
    return {
        "margin_mean": float(margins.mean()), "margin_median": float(np.median(margins)),
        "positive_fraction": float(np.mean(margins > 0)),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": _macro_f1(labels, predictions, np.arange(len(prototypes))),
    }, per_class


def anchor_usage(responses):
    responses = np.asarray(responses)
    winners = responses.argmax(1)
    counts = np.bincount(winners, minlength=responses.shape[1]).astype(float)
    probabilities = counts[counts > 0] / max(counts.sum(), 1)
    return {"active_anchor_count": int((counts > 0).sum()),
            "entropy": float(-(probabilities * np.log(probabilities)).sum()),
            "max_fraction": float(counts.max() / max(counts.sum(), 1))}


def query_ratio_summary(master, shape):
    master, shape = np.asarray(master), np.asarray(shape)
    values = np.linalg.norm(shape, axis=-1) / (np.linalg.norm(master, axis=-1) + EPS)
    return {"values": values, "mean": float(values.mean()), "std": float(values.std()),
            "p10": float(np.quantile(values, .1)), "p50": float(np.quantile(values, .5)),
            "p90": float(np.quantile(values, .9))}


def attention_js(first, second):
    if np.asarray(first).shape != np.asarray(second).shape:
        raise ValueError("attention tensors must have identical shape")
    return float(np.mean(_js(first, second, axis=-1)))


def validate_final_student_packet(name, packet):
    if Path(name).name != "checkpoint_last.pt":
        raise ValueError("END must be checkpoint_last.pt, not validation-best model.pt")
    config = packet.get("config", {})
    if config.get("output_student") is not True:
        raise ValueError("END checkpoint does not select the student")
    if packet.get("epoch") != int(config.get("epochs", -1)) - 1:
        raise ValueError("END checkpoint is not the final configured epoch")
    if "teacher_state_dict" not in packet:
        raise ValueError("END checkpoint lacks the training teacher state")


def _finite(row, identity):
    for key, value in row.items():
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            raise FloatingPointError(f"non-finite {identity}:{key}={value}")


def parse_source_log(path):
    text = Path(path).read_text(errors="replace")
    source_part = text.split("UDA ", 1)[0]
    f1 = [float(value) for value in re.findall(r"Validation result:.*?f1=([0-9.]+)", source_part)]
    test = re.findall(r"Test result for .*?:.*?f1=([0-9.]+)", source_part)
    losses = [float(value) for value in re.findall(r"loss_cls_source[=|:]([0-9.eE+-]+)", source_part)]
    def spread(values, count, beginning):
        current = values[:count] if beginning else values[-count:]
        return float(np.std(current)) if current else ""
    drops = [f1[i - 1] - f1[i] for i in range(1, len(f1))]
    return {
        "source_test_f1": float(test[-1]) if test else "",
        "source_best_f1": max(f1) if f1 else "",
        "source_last_f1": f1[-1] if f1 else "",
        "source_best_last_gap": max(f1) - f1[-1] if f1 else "",
        "source_early_f1_std": spread(f1, 10, True), "source_late_f1_std": spread(f1, 20, False),
        "source_max_epoch_drop": max(drops) if drops else "",
        "source_early_loss_std": spread(losses, 10, True), "source_late_loss_std": spread(losses, 20, False),
    }


def fixed_class_indices(labels, classes, limit, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    output = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        selected = candidates if len(candidates) <= limit else rng.choice(candidates, limit, replace=False)
        output.extend(np.sort(selected).tolist())
    return np.asarray(sorted(output), dtype=np.int64)


def _load_model(path, device, final=False, teacher=False):
    import torch
    from train import create_model
    packet = torch.load(path, map_location=device, weights_only=False)
    if final:
        validate_final_student_packet(path, packet)
    config = SimpleNamespace(**packet["config"])
    model = create_model(config)
    state = packet["teacher_state_dict"] if teacher else packet["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model, config, packet


def _datasets(config, source, target, data_root, seed):
    from torchvision.transforms import transforms
    from dataset import PixelSetData
    from train import create_train_val_test_folds
    from transforms import Identity, Normalize, RandomSamplePixels, ToTensor
    bare = {name: PixelSetData(data_root, name, config.classes, closed_set=True,
            combine_spring_and_winter=config.combine_spring_and_winter) for name in (source, target)}
    eligible = {name: dataset.get_parcel_indices().tolist() for name, dataset in bare.items()}
    random.seed(seed); np.random.seed(seed)
    split = create_train_val_test_folds([source, target], 1, eligible, config.val_ratio, config.test_ratio)[0]
    random.seed(seed); np.random.seed(seed)
    source_only_split = create_train_val_test_folds(
        [source, source], 1, {source: eligible[source]}, config.val_ratio, config.test_ratio,
    )[0]
    fixed = transforms.Compose([RandomSamplePixels(config.num_pixels), Normalize(), ToTensor()])
    full = transforms.Compose([Identity(), Normalize(), ToTensor()])
    result = {}
    for name in (source, target):
        for part in ("train", "test"):
            indices = source_only_split[source]["test"] if name == source and part == "test" else split[name][part]
            result[(name, part)] = PixelSetData(
                data_root, name, config.classes, fixed if part == "train" else full,
                indices=indices, closed_set=True,
                combine_spring_and_winter=config.combine_spring_and_winter,
            )
    return result


def _loader(dataset, batch_size, indices=None):
    import torch
    from dataset import GroupByShapesBatchSampler
    data = torch.utils.data.Subset(dataset, indices.tolist()) if indices is not None else dataset
    return torch.utils.data.DataLoader(data, num_workers=0,
        batch_sampler=GroupByShapesBatchSampler(data, batch_size))


def _forward_features(model, sample, device, shift=0, master_only=False):
    import torch
    pixels = sample["pixels"].to(device); mask = sample["valid_pixels"].to(device)
    positions = sample["positions"].to(device); extra = sample["extra"].to(device)
    if torch.is_tensor(shift):
        if shift.numel() != 1: raise ValueError("offline pseudo replay expects one global scalar shift")
        shift = float(shift.detach().cpu().reshape(-1)[0])
    shift_value = float(shift)
    if not shift_value.is_integer(): raise ValueError(f"global shift must be an integer day, got {shift_value}")
    shift = int(shift_value)
    spatial = model.spatial_encoder(pixels, mask, extra)
    if hasattr(model, "structure_branch"):
        structure = model.structure_branch(spatial, positions)
        external = None if master_only else structure["shape_class_token"]
        instance, attention = model.temporal_encoder(spatial, positions + shift, return_att=True, external_query=external)
        logits = model.decoder(instance)
        return logits, spatial, instance, structure, attention
    instance, attention = model.temporal_encoder(spatial, positions + shift, return_att=True)
    return model.decoder(instance), spatial, instance, {}, attention


def _extract(model, loader, device, shift=0, master_only=False):
    import torch
    output = {key: [] for key in ("logits", "pse", "pse_temporal", "positions", "instance", "response", "tokens", "attention", "labels")}
    with torch.no_grad():
        for sample in loader:
            logits, spatial, instance, structure, attention = _forward_features(model, sample, device, shift, master_only)
            output["logits"].append(logits.cpu().numpy()); output["pse"].append(spatial.mean(1).cpu().numpy())
            output["pse_temporal"].append(spatial.cpu().numpy()); output["positions"].append(sample["positions"].numpy())
            output["instance"].append(instance.cpu().numpy()); output["attention"].append(attention.cpu().numpy())
            output["labels"].append(sample["label"].numpy())
            if structure:
                output["response"].append(structure["shapelet_response"].cpu().numpy())
                output["tokens"].append(structure["shape_tokens"].cpu().numpy())
    return {key: np.concatenate(value) if value else None for key, value in output.items()}


def _reset_rng(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _render_pse_curves(root, task, seed, start_source, start_target, end_source, end_target, classes):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    groups = [start_source, start_target, end_source, end_target]
    rng = np.random.default_rng(seed); fit = []
    for data in groups:
        flat = data["pse_temporal"].reshape(-1, data["pse_temporal"].shape[-1])
        if len(flat) > 50000: flat = flat[rng.choice(len(flat), 50000, replace=False)]
        fit.append(flat)
    matrix = np.concatenate(fit); center = matrix.mean(0); _, _, vh = np.linalg.svd(matrix - center, full_matrices=False); axis = vh[0]
    if axis[np.argmax(np.abs(axis))] < 0: axis = -axis
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    bins = np.linspace(0, 365, 25); centers = (bins[:-1] + bins[1:]) / 2
    for panel, data, title in zip(axes.flat, groups, ("START source", "START target", "END source", "END target")):
        scores = np.tensordot(data["pse_temporal"] - center, axis, axes=([-1], [0]))
        for label in sorted(set(data["labels"])):
            selected = data["labels"] == label; pos = data["positions"][selected].reshape(-1); val = scores[selected].reshape(-1)
            ids = np.clip(np.searchsorted(bins, pos, side="right") - 1, 0, 23)
            curve = np.array([val[ids == b].mean() if np.any(ids == b) else np.nan for b in range(24)])
            panel.plot(centers, curve, label=classes[int(label)], linewidth=1.2)
        panel.set_title(title); panel.grid(alpha=.2); panel.set_xlim(0, 365)
    axes[0, 0].legend(fontsize=6, ncol=2); figure.suptitle(f"{task} seed{seed} | Ours PSE PC1 (one shared basis)")
    figure.tight_layout(); figure.savefig(Path(root) / "figures" / f"{task}_seed{seed}_pse_temporal.png", dpi=150); plt.close(figure)


def _evaluate(model, loader, device, shift=0, master_only=False):
    data = _extract(model, loader, device, shift, master_only)
    return _macro_f1(data["labels"], data["logits"].argmax(1)), data


def _support(anchors, tokens, seed):
    anchors = _normalize(anchors); tokens = np.asarray(tokens).reshape(-1, tokens.shape[-1])
    if len(tokens) > 50000:
        tokens = tokens[np.random.default_rng(seed).choice(len(tokens), 50000, replace=False)]
    tokens = _normalize(tokens)
    best = np.full(len(anchors), -np.inf)
    for begin in range(0, len(tokens), 4096):
        best = np.maximum(best, (anchors @ tokens[begin:begin + 4096].T).max(1))
    return float(np.mean(1 - best))


def _anchor_redundancy(anchors):
    anchors = _normalize(anchors); similarity = anchors @ anchors.T
    values = similarity[~np.eye(len(anchors), dtype=bool)]
    return float(values.mean()), float(values.max())


def _append_csv(path, rows):
    if not rows: return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    existing = list(csv.DictReader(path.open(newline=""))) if path.exists() else []
    columns = sorted(set().union(*(row.keys() for row in existing + rows)))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns); writer.writeheader(); writer.writerows(existing + rows)


def run_one(args):
    import torch
    device = torch.device(args.device)
    start, start_cfg, _ = _load_model(args.start_checkpoint, device)
    end, end_cfg, end_packet = _load_model(args.end_checkpoint, device, final=True)
    if (start_cfg.source, start_cfg.target) != (args.source, args.source):
        raise ValueError("START is not the requested source-only checkpoint")
    if (end_cfg.source, end_cfg.target, int(end_cfg.seed)) != (args.source, args.target, args.seed):
        raise ValueError("END task/seed mismatch")
    expected = "pseltae" if args.method == "original_timematch" else "psestructureprotoltae"
    if start_cfg.model != expected or end_cfg.model != expected:
        raise ValueError(f"method requires model={expected}")
    if list(start_cfg.classes) != list(end_cfg.classes):
        raise ValueError("START and END use different class protocols")
    datasets = _datasets(start_cfg, args.source, args.target, args.data_root, args.seed)
    source_train, target_train = datasets[(args.source, "train")], datasets[(args.target, "train")]
    common = sorted(set(source_train.get_labels()) & set(target_train.get_labels()))
    source_idx = fixed_class_indices(source_train.get_labels(), common, args.samples_per_class, args.seed)
    target_idx = fixed_class_indices(target_train.get_labels(), common, args.samples_per_class, args.seed)
    source_loader = _loader(source_train, args.batch_size, source_idx); target_loader = _loader(target_train, args.batch_size, target_idx)
    _reset_rng(args.seed); start_source = _extract(start, source_loader, device)
    _reset_rng(args.seed + 10000); start_target = _extract(start, target_loader, device)
    _reset_rng(args.seed); end_source = _extract(end, source_loader, device)
    _reset_rng(args.seed + 10000); end_target = _extract(end, target_loader, device)
    source_test_loader = _loader(datasets[(args.source, "test")], args.batch_size)
    target_test_loader = _loader(datasets[(args.target, "test")], args.batch_size)
    source_f1_start, _ = _evaluate(start, source_test_loader, device)
    source_f1_end, _ = _evaluate(end, source_test_loader, device)
    target_f1_start, _ = _evaluate(start, target_test_loader, device)
    target_f1_end, _ = _evaluate(end, target_test_loader, device)
    row = {"task": args.task, "seed": args.seed, "method": args.method,
           "status": "available", "source_test_f1_start": source_f1_start,
           "source_test_f1_end": source_f1_end, "target_test_f1_start": target_f1_start,
           "target_test_f1_end": target_f1_end, "source_test_f1": source_f1_start,
           "target_test_f1": target_f1_end, **parse_source_log(args.source_log)}
    class_rows = {c: {"task": args.task, "seed": args.seed, "method": args.method,
                      "class_name": start_cfg.classes[c]} for c in common}
    for layer in ("pse", "instance") + (("response",) if args.method == "structure_shapelet" else ()):
        start_metric, start_per = representation_metrics(start_source[layer], start_source["labels"], start_target[layer], start_target["labels"])
        end_metric, end_per = representation_metrics(end_source[layer], end_source["labels"], end_target[layer], end_target["labels"])
        row.update({f"{layer}_domain_gap_start": start_metric["domain_gap"], f"{layer}_domain_gap_end": end_metric["domain_gap"],
                    f"{layer}_domain_gap_reduction": 1 - end_metric["domain_gap"] / (start_metric["domain_gap"] + EPS),
                    f"{layer}_inter_start": start_metric["target_inter"], f"{layer}_inter_end": end_metric["target_inter"],
                    f"{layer}_separation_retention": end_metric["target_inter"] / (start_metric["target_inter"] + EPS),
                    f"{layer}_source_inter_start": start_metric["source_inter"], f"{layer}_source_inter_end": end_metric["source_inter"],
                    f"{layer}_intra_start": start_metric["target_intra"], f"{layer}_intra_end": end_metric["target_intra"],
                    f"{layer}_source_intra_start": start_metric["source_intra"], f"{layer}_source_intra_end": end_metric["source_intra"]})
        if layer == "response":
            row["response_fisher_source_start"] = start_metric["source_inter"] / (start_metric["source_intra"] + EPS)
            row["response_fisher_source_end"] = end_metric["source_inter"] / (end_metric["source_intra"] + EPS)
            row["response_fisher_target_start"] = start_metric["target_inter"] / (start_metric["target_intra"] + EPS)
            row["response_fisher_target_end"] = end_metric["target_inter"] / (end_metric["target_intra"] + EPS)
            row["response_fisher_source"] = row["response_fisher_source_end"]
            row["response_fisher_target"] = row["response_fisher_target_end"]
        for c in common:
            class_rows[c][f"{layer}_domain_gap_start"] = start_per[c]["domain_gap"]
            class_rows[c][f"{layer}_domain_gap_end"] = end_per[c]["domain_gap"]

    teacher, _, _ = _load_model(args.end_checkpoint, device, teacher=True)
    pseudo_shift = end_packet.get("global_temporal_shift", 0)
    _reset_rng(args.seed + 10000)
    pseudo_data = _extract(teacher, target_loader, device, pseudo_shift)
    probabilities = np.exp(pseudo_data["logits"] - pseudo_data["logits"].max(1, keepdims=True)); probabilities /= probabilities.sum(1, keepdims=True)
    confidence = probabilities.max(1); predictions = probabilities.argmax(1)
    pseudo, pseudo_per = pseudo_label_metrics(pseudo_data["labels"], predictions, confidence > float(end_cfg.pseudo_threshold), len(start_cfg.classes))
    row.update({f"pseudo_{key}": value for key, value in pseudo.items()})
    for c in common:
        class_rows[c].update({f"pseudo_{key}": value for key, value in pseudo_per[c].items()})

    if args.method == "structure_shapelet":
        ratio_by_domain, attention_by_domain = {}, {}
        qmaster = end.temporal_encoder.attention_heads.query.detach().cpu().numpy().reshape(-1)
        for domain, loader, rng_seed in (("source", source_loader, args.seed), ("target", target_loader, args.seed + 10000)):
            ratios, attention_values = [], []; _reset_rng(rng_seed)
            for sample in loader:
                with torch.no_grad():
                    _, _, _, structure, full_att = _forward_features(end, sample, device)
                    _, _, _, _, master_att = _forward_features(end, sample, device, master_only=True)
                    qshape = end.temporal_encoder.attention_heads.external_query_projection(structure["shape_class_token"]).cpu().numpy()
                    ratios.extend(query_ratio_summary(np.repeat(qmaster[None], len(qshape), axis=0), qshape)["values"].tolist())
                    attention_values.append(attention_js(full_att.cpu().numpy(), master_att.cpu().numpy()))
            summary = query_ratio_summary(np.ones((len(ratios), 1)), np.asarray(ratios)[:, None])
            ratio_by_domain[domain] = summary; attention_by_domain[domain] = float(np.mean(attention_values))
            row.update({f"query_ratio_{domain}_mean": summary["mean"], f"query_ratio_{domain}_std": summary["std"],
                        f"query_ratio_{domain}_p10": summary["p10"], f"query_ratio_{domain}_p50": summary["p50"],
                        f"query_ratio_{domain}_p90": summary["p90"], f"attention_js_{domain}": attention_by_domain[domain]})
        row.update({"query_ratio_mean": ratio_by_domain["target"]["mean"], "query_ratio_p50": ratio_by_domain["target"]["p50"],
                    "query_ratio_p90": ratio_by_domain["target"]["p90"], "attention_js": attention_by_domain["target"]})
        full_source, _ = _evaluate(end, source_test_loader, device); ablated_source, _ = _evaluate(end, source_test_loader, device, master_only=True)
        full_target, _ = _evaluate(end, target_test_loader, device); ablated_target, _ = _evaluate(end, target_test_loader, device, master_only=True)
        row["query_ablation_delta_f1_source"] = full_source - ablated_source
        row["query_ablation_delta_f1_target"] = full_target - ablated_target
        anchors_start = start.structure_branch.shapelet_dictionary.anchors.detach().cpu().numpy()
        anchors_end = end.structure_branch.shapelet_dictionary.anchors.detach().cpu().numpy()
        row["shapelet_support_start"] = _support(anchors_start, start_source["tokens"], args.seed)
        row["shapelet_support_end"] = _support(anchors_end, end_source["tokens"], args.seed)
        row["anchor_cos_mean"], row["anchor_cos_max"] = _anchor_redundancy(anchors_end)
        usage = anchor_usage(end_source["response"]); row.update({"active_anchor_count": usage["active_anchor_count"],
            "anchor_usage_entropy": usage["entropy"], "max_anchor_usage_fraction": usage["max_fraction"]})
        for stage, model, source_data, target_data in (("start", start, start_source, start_target), ("end", end, end_source, end_target)):
            prototypes = model.instance_prototype_bank.prototypes.detach().cpu().numpy()
            for domain, data in (("source", source_data), ("target", target_data)):
                geometry, geometry_per = prototype_geometry(data["instance"], data["labels"], prototypes)
                row[f"prototype_margin_{domain}_{stage}"] = geometry["margin_mean"]
                row[f"prototype_positive_fraction_{domain}_{stage}"] = geometry["positive_fraction"]
                row[f"nearest_proto_f1_{domain}_{stage}"] = geometry["macro_f1"]
                for c in common:
                    class_rows[c][f"prototype_margin_{domain}_{stage}"] = geometry_per[c]["margin_mean"]
                    class_rows[c][f"nearest_prototype_accuracy_{domain}_{stage}"] = geometry_per[c]["nearest_accuracy"]
        row["prototype_margin_source"] = row["prototype_margin_source_end"]
        row["prototype_margin_target"] = row["prototype_margin_target_end"]
        row["prototype_positive_fraction_source"] = row["prototype_positive_fraction_source_end"]
        row["prototype_positive_fraction_target"] = row["prototype_positive_fraction_target_end"]
        row["nearest_proto_f1_source"] = row["nearest_proto_f1_source_end"]
        row["nearest_proto_f1_target"] = row["nearest_proto_f1_target_end"]
        Path(args.output_root, "figures").mkdir(parents=True, exist_ok=True)
        _render_pse_curves(args.output_root, args.task, args.seed, start_source, start_target, end_source, end_target, start_cfg.classes)
    identity = f"{args.task}/seed{args.seed}/{args.method}"
    _finite(row, identity)
    for current in class_rows.values(): _finite(current, identity + "/per_class")
    _append_csv(Path(args.output_root) / "summary.csv", [row])
    _append_csv(Path(args.output_root) / "per_class.csv", list(class_rows.values()))
    print(f"DIAGNOSTIC_RUN_FINISHED|task={args.task}|seed={args.seed}|method={args.method}|target_f1={target_f1_end:.6f}")


def finalize(args):
    root = Path(args.output_root); summary_path = root / "summary.csv"
    rows = list(csv.DictReader(summary_path.open())) if summary_path.exists() else []
    root.mkdir(parents=True, exist_ok=True); figures = root / "figures"; figures.mkdir(exist_ok=True)
    tasks = ["AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1"]
    lines = ["# Diagnostic Summary", "", "Offline summary only; no automatic research conclusion.", ""]
    keys = ["target_test_f1_end", "pse_domain_gap_reduction", "pse_separation_retention",
            "instance_domain_gap_reduction", "instance_separation_retention", "pseudo_coverage", "pseudo_precision"]
    for task in tasks:
        lines += [f"## {task.replace('_', ' → ')}", "", "| seed | method | " + " | ".join(keys) + " |",
                  "|---:|---|" + "|".join(["---:"] * len(keys)) + "|"]
        for row in rows:
            if row.get("task") == task:
                lines.append("| " + " | ".join([row.get("seed", ""), row.get("method", "")] + [row.get(k, "") for k in keys]) + " |")
        lines.append("")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("FIGURES_SKIPPED|reason=matplotlib_unavailable"); return
    def numeric(row, key):
        value = row.get(key, "")
        return float(value) if value not in ("", None) else np.nan
    for task in tasks:
        current = [row for row in rows if row.get("task") == task]
        if not current: continue
        labels = [f"s{r['seed']} {r['method']}" for r in current]
        for name, metrics in (("alignment_separation", ["pse_domain_gap_reduction", "pse_separation_retention", "instance_domain_gap_reduction", "instance_separation_retention"]),
                              ("pseudo_quality", ["pseudo_coverage", "pseudo_precision", "pseudo_macro_f1"])):
            fig, ax = plt.subplots(figsize=(10, 4)); x = np.arange(len(labels)); width = .8 / len(metrics)
            for i, metric in enumerate(metrics): ax.bar(x + (i - (len(metrics)-1)/2)*width, [numeric(r, metric) for r in current], width, label=metric)
            ax.set_xticks(x, labels, rotation=25, ha="right"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(figures / f"{task}_{name}.png", dpi=150); plt.close(fig)
        ours = [row for row in current if row.get("method") == "structure_shapelet"]
        if ours:
            metrics = ["query_ablation_delta_f1_target", "query_ratio_mean", "attention_js",
                       "shapelet_support_end", "anchor_usage_entropy", "prototype_margin_target"]
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            for axis, metric in zip(axes.flat, metrics):
                axis.bar([f"seed{row['seed']}" for row in ours], [numeric(row, metric) for row in ours])
                axis.set_title(metric); axis.grid(axis="y", alpha=.2)
            figure_title = f"{task} | structure diagnostics"
            fig.suptitle(figure_title); fig.tight_layout(); fig.savefig(figures / f"{task}_structure_summary.png", dpi=150); plt.close(fig)


def parser():
    root = argparse.ArgumentParser(); sub = root.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("task", "method", "source", "target", "data_root", "start_checkpoint", "end_checkpoint", "source_log", "output_root"):
        run.add_argument("--" + name.replace("_", "-"), required=True)
    run.add_argument("--seed", type=int, required=True); run.add_argument("--samples-per-class", type=int, default=128)
    run.add_argument("--batch-size", type=int, default=32); run.add_argument("--device", default="cuda")
    done = sub.add_parser("finalize"); done.add_argument("--output-root", required=True)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    run_one(args) if args.command == "run" else finalize(args)
