#!/usr/bin/env python3
"""Frozen D1/D2/D3 diagnostics and paired summaries for E/G recheck."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.audit_e_representation_chain import (
    _loader, _seed, official_protocol,
)
from analysis.structure_e_g_validation import parse_epoch_history
from analysis.structure_representation_chain_audit import make_linear_probe
from analysis.summarize_reliable_composition_ab import evaluate_checkpoint_data


TASKS = {
    "FR1_FR2": ("FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
    "FR2_DK1": ("FR2", "france/31TCJ/2017", "DK1", "denmark/32VNH/2017"),
}
VARIANTS = ("E", "G")
BLOCKS = (("strength", 0, 16), ("concentration", 16, 32), ("phase", 32, 96))
FOCUS = {
    "FR1_FR2": ("winter_barley", "winter_wheat"),
    "FR2_DK1": ("spring_barley", "horsebeans", "winter_barley", "winter_wheat"),
}


def _write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(fields or dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block: break
            digest.update(block)
    return digest.hexdigest()


def require_complete_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MISSING_FINAL|path={path}")
    if path.name != "checkpoint_last.pt":
        raise ValueError(f"final checkpoint must be checkpoint_last.pt: {path}")
    packet = torch.load(path, map_location="cpu", weights_only=False)
    raw = packet.get("config", {})
    config = raw if isinstance(raw, dict) else vars(raw)
    if int(config.get("epochs", -1)) != 20 or int(packet.get("epoch", -1)) != 19:
        raise ValueError(
            f"final checkpoint must be epoch=19 of 20: path={path} "
            f"epoch={packet.get('epoch')} epochs={config.get('epochs')}"
        )
    return packet


def _config_dict(packet):
    raw = packet.get("config", {})
    return dict(raw if isinstance(raw, dict) else vars(raw))


def validate_uda_checkpoint(packet, task, variant):
    """Validate the training-relevant E/G configuration stored in a final checkpoint."""
    require = {
        "shape_representation": "phase_moment",
        "shape_injection": "current_query",
        "structure_shift_mode": "none",
        "fourier_num_modes": 13,
        "shape_dim": 128,
        "shape_window_scales": [24],
        "shape_window_stride": 8,
        "shapelet_count": 16,
        "shapelet_beta": 5,
        "shape_resample_length": 16,
        "shapelet_diversity_margin": .5,
        "shapelet_diversity_weight": .01,
        "shape_class_weight": .1,
        "shape_align_weight": 0.,
        "source_minority_mode": "base",
        "with_shift_aug": False,
        "shape_da_mode": "batch_align",
        "shape_alignment_view": "none",
        "shape_equivariance_weight": .05,
        "shape_equivariance_max_shift": 60,
        "detach_target_structure": False,
        "target_structure_detach_epochs": 0 if variant == "E" else 5,
        "epochs": 20,
        "steps_per_epoch": 500,
        "batch_size": 128,
        "lr": .0001,
        "pseudo_threshold": .9,
        "ema_decay": .9999,
        "trade_off": 2.,
        "estimate_shift": True,
        "balance_source": True,
        "use_focal_loss": True,
        "shift_source": True,
        "sample_size": 100,
        "max_temporal_shift": 60,
        "domain_specific_bn": True,
        "shift_estimator": "AM",
        "seed": 1,
        "num_folds": 1,
        "seq_length": 30,
        "num_pixels": 64,
        "closed_set": True,
    }
    _, source, _, target = TASKS[task]
    require.update({"source": source, "target": target})
    config = _config_dict(packet)
    mismatches = {}
    for key, expected in require.items():
        actual = config.get(key)
        if isinstance(expected, float):
            matches = actual is not None and np.isclose(float(actual), expected)
        else:
            matches = actual == expected
        if not matches:
            mismatches[key] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ValueError(
            f"{variant} {task} checkpoint configuration mismatch: {mismatches}"
        )
    return {key: config[key] for key in require}


def strict_load_checkpoint(path, device):
    from train import create_model
    packet = torch.load(path, map_location=device, weights_only=False)
    config = _config_dict(packet)
    if not config:
        raise ValueError(f"checkpoint config missing: {path}")
    namespace = SimpleNamespace(**config)
    model = create_model(namespace)
    model.load_state_dict(packet["state_dict"], strict=True)
    return model.to(device).eval(), namespace, packet


def validate_source(source_checkpoint, task, device="cpu"):
    path = Path(source_checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"MISSING_SOURCE|path={path}")
    model, config, packet = strict_load_checkpoint(path, torch.device(device))
    expected_source = TASKS[task][1]
    if config.source != expected_source or config.target != expected_source:
        raise ValueError(
            f"source checkpoint domain mismatch: expected={expected_source} "
            f"source={config.source} target={config.target}"
        )
    expected = {
        "shape_representation": "phase_moment", "shape_injection": "current_query",
        "fourier_num_modes": 13, "shape_window_scales": [24],
        "shape_window_stride": 8,
        "shapelet_count": 16, "shape_resample_length": 16,
    }
    actual = {key: getattr(config, key, None) for key in expected}
    if actual != expected:
        raise ValueError(f"source checkpoint configuration mismatch: {actual}")
    del model
    return {
        "task": task, "path": str(path), "sha256": sha256_file(path),
        "epoch": int(packet.get("epoch", -1)), "classes": list(config.classes),
        **actual, "strict_load": True,
    }


def decompose_linear_margin(estimator, features, true_labels):
    """Decompose one standardized linear margin using the same fitted probe."""
    features = np.asarray(features, dtype=np.float64)
    true_labels = np.asarray(true_labels, dtype=np.int64)
    scaler = estimator.named_steps.get("scaler", estimator.named_steps.get("scale"))
    classifier = estimator.named_steps.get("classifier")
    standardized = scaler.transform(features)
    scores = estimator.decision_function(features)
    if scores.ndim != 2 or classifier.coef_.shape[0] != len(classifier.classes_):
        raise ValueError("D1 requires a multiclass linear probe")
    class_to_row = {int(value): index for index, value in enumerate(classifier.classes_)}
    true_rows = np.asarray([class_to_row[int(label)] for label in true_labels])
    masked = scores.copy(); masked[np.arange(len(scores)), true_rows] = -np.inf
    competitor_rows = masked.argmax(1)
    delta_weight = classifier.coef_[true_rows] - classifier.coef_[competitor_rows]
    intercept = np.asarray(classifier.intercept_)
    bias = intercept[true_rows] - intercept[competitor_rows]
    result = {
        "prediction": classifier.classes_[scores.argmax(1)].astype(np.int64),
        "competitor": classifier.classes_[competitor_rows].astype(np.int64),
        "margin": scores[np.arange(len(scores)), true_rows]
        - scores[np.arange(len(scores)), competitor_rows],
        "bias_contribution": bias,
    }
    terms = delta_weight * standardized
    for name, start, end in BLOCKS:
        result[f"{name}_contribution"] = (
            terms[:, start:end]
        ).sum(1)
    reconstructed = sum(result[f"{name}_contribution"] for name, _, _ in BLOCKS) + bias
    error = np.abs(reconstructed - result["margin"])
    # The reference BLAS dot product and the three block reductions use different
    # summation orders.  Bound their combined IEEE-754 forward error from the
    # actual absolute term sum instead of applying an arbitrary fixed tolerance.
    calculation_dtypes = (
        np.asarray(scores).dtype,
        np.asarray(standardized).dtype,
        np.asarray(classifier.coef_).dtype,
        np.asarray(intercept).dtype,
    )
    epsilon = max(
        np.finfo(dtype).eps
        for dtype in calculation_dtypes
        if np.issubdtype(dtype, np.floating)
    )
    operations = terms.shape[1] + len(BLOCKS) + 2
    gamma = operations * epsilon / (1. - operations * epsilon)
    magnitude = np.abs(terms).sum(1) + np.abs(bias) + np.abs(result["margin"])
    bound = 4. * gamma * magnitude + np.finfo(np.float64).tiny
    result["margin_reconstruction_error"] = error
    result["margin_reconstruction_bound"] = bound
    failed = error > bound
    if failed.any():
        index = int(np.flatnonzero(failed)[np.argmax(error[failed])])
        raise RuntimeError(
            "D1 contribution blocks do not reconstruct the decision_function "
            f"margin: sample={index} error={error[index]:.12g} "
            f"bound={bound[index]:.12g} margin={result['margin'][index]:.12g} "
            f"absolute_term_sum={np.abs(terms[index]).sum():.12g}"
        )
    return result


def run_independent_diagnostic_stages(stages):
    """Run all requested stages, retaining successes when a peer stage fails."""
    results, statuses = {}, {}
    for name, operation in stages.items():
        try:
            results[name] = operation()
            statuses[name] = {"status": "complete", "error": None}
        except Exception as error:  # continue independent frozen diagnostics
            statuses[name] = {
                "status": "failed", "error": str(error),
                "traceback": traceback.format_exc(),
            }
    return results, statuses


@torch.no_grad()
def neutral_query_intervention(model, spatial, positions, structure):
    """Hold memory/K/V/decoder fixed and remove only the external query correction."""
    query = structure["shape_class_token"]
    normal_instance, normal_attention = model.temporal_encoder(
        spatial, positions, external_query=query, return_att=True,
    )
    neutral_instance, neutral_attention = model.temporal_encoder(
        spatial, positions, external_query=None, return_att=True,
    )
    normal_logits = model.decoder(normal_instance)
    neutral_logits = model.decoder(neutral_instance)
    expected = model._output_from_prepared_structure(spatial, positions, structure)["logits"]
    torch.testing.assert_close(normal_logits, expected, atol=1e-6, rtol=1e-6)
    projection = model.temporal_encoder.attention_heads.external_query_projection
    correction = projection(query)
    return {
        "normal_logits": normal_logits, "neutral_logits": neutral_logits,
        "normal_attention": normal_attention, "neutral_attention": neutral_attention,
        "query_correction_norm": correction.norm(dim=1),
    }


def _margin(logits, labels):
    selected = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.masked_fill(
        torch.nn.functional.one_hot(labels, logits.shape[1]).bool(), -torch.inf,
    ).max(1).values
    return selected - other


@torch.no_grad()
def extract_once(model, dataset, device, batch_size, domain):
    storage = defaultdict(list)
    centers = model.structure_branch.window_extractor.window_centers.detach().cpu()
    centers = centers * model.structure_branch.exposer.period_days / model.structure_branch.window_extractor.grid_points
    beta = float(model.structure_branch.shapelet_dictionary.beta)
    for raw in _loader(dataset, batch_size, 0):
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
        spatial = model.spatial_encoder(batch["pixels"], batch["valid_pixels"], batch["extra"])
        structure = model.prepare_structure(spatial, batch["positions"], 0)
        query = neutral_query_intervention(model, spatial, batch["positions"], structure)
        similarity = structure["shapelet_similarity"]
        weights = torch.softmax(beta * similarity, dim=1)
        anchor_distribution = torch.softmax(beta * similarity, dim=-1)
        windows, _ = model.structure_branch.window_extractor(structure["exposed_curve"])
        curve = torch.cat(windows, 1).norm(dim=-1)
        values = {
            "parcel_id": batch["parcel_index"], "label": batch["label"],
            "response": structure["shapelet_response"], "similarity": similarity,
            "weights": weights, "anchor_distribution": anchor_distribution,
            "window_curve_norm": curve,
            "normal_logits": query["normal_logits"], "neutral_logits": query["neutral_logits"],
            "normal_attention": query["normal_attention"].flatten(1),
            "neutral_attention": query["neutral_attention"].flatten(1),
            "query_correction_norm": query["query_correction_norm"],
        }
        for key, value in values.items(): storage[key].append(value.detach().float().cpu())
    result = {key: torch.cat(parts).numpy() for key, parts in storage.items()}
    result["parcel_id"] = result["parcel_id"].astype(np.int64)
    result["label"] = result["label"].astype(np.int64)
    result["domain"] = domain
    result["window_centers_days"] = centers.numpy()
    return result


def _fixed_f1(labels, prediction, classes):
    return float(f1_score(
        labels, prediction, labels=np.arange(len(classes)), average="macro",
        zero_division=0,
    ))


def analyze_d1(task, variant, arrays, classes, output):
    estimator = make_linear_probe().fit(
        arrays["source_train"]["response"], arrays["source_train"]["label"],
    )
    sample_rows, class_rows, competition_rows, confusion_rows = [], [], [], []
    for split in ("source_val", "target_val"):
        data = arrays[split]
        result = decompose_linear_margin(estimator, data["response"], data["label"])
        for index in range(len(data["label"])):
            sample_rows.append({
                "task": task, "variant": variant, "split": split,
                "domain": data["domain"],
                "parcel_index": int(data["parcel_id"][index]),
                "true_class": classes[data["label"][index]],
                "prediction": classes[result["prediction"][index]],
                "competitor": classes[result["competitor"][index]],
                **{key: float(result[key][index]) for key in (
                    "margin", "strength_contribution", "concentration_contribution",
                    "phase_contribution", "bias_contribution",
                    "margin_reconstruction_error", "margin_reconstruction_bound",
                )},
            })
        cm = confusion_matrix(data["label"], result["prediction"], labels=np.arange(len(classes)))
        for true_id, true_name in enumerate(classes):
            selected = data["label"] == true_id
            class_rows.append({
                "task": task, "variant": variant, "split": split,
                "class_id": true_id, "class_name": true_name,
                "support": int(selected.sum()),
                "f1": f1_score(data["label"], result["prediction"], labels=[true_id], average="macro", zero_division=0),
                **{f"mean_{key}": float(result[key][selected].mean()) if selected.any() else float("nan") for key in (
                    "margin", "strength_contribution", "concentration_contribution", "phase_contribution", "bias_contribution",
                )},
            })
            for pred_id, pred_name in enumerate(classes):
                confusion_rows.append({
                    "task": task, "variant": variant, "split": split,
                    "domain": data["domain"],
                    "true_class": true_name, "predicted_class": pred_name,
                    "count": int(cm[true_id, pred_id]),
                })
        counts = Counter(zip(data["label"], result["competitor"]))
        for (true_id, competitor), count in counts.items():
            competition_rows.append({
                "task": task, "variant": variant, "split": split,
                "true_class": classes[true_id], "competitor": classes[competitor],
                "count": count,
            })
    _write_csv(output / "D1_samples.csv", sample_rows)
    _write_csv(output / "D1_per_class.csv", class_rows)
    _write_csv(output / "D1_competition.csv", competition_rows)
    _write_csv(output / "D1_confusion.csv", confusion_rows)
    return class_rows


def analyze_d2(task, variant, arrays, classes, output):
    source = arrays["source_train"]
    q25, q75 = np.quantile(source["similarity"].reshape(-1, 16), (.25, .75), axis=0)
    weight_q75 = np.quantile(source["weights"].reshape(-1, 16), .75, axis=0)
    probability_q75 = np.quantile(
        source["anchor_distribution"].reshape(-1, 16), .75, axis=0,
    )
    source_class_anchor = np.full((len(classes), 16), -np.inf)
    for class_id in range(len(classes)):
        selected = source["label"] == class_id
        if selected.any():
            source_class_anchor[class_id] = source["similarity"][selected].mean(axis=(0, 1))
    anchor_semantic_class = source_class_anchor.argmax(axis=0)
    rows, semantic_rows, association_rows, candidates = [], [], [], []
    for split in ("source_val", "target_val"):
        data = arrays[split]
        top_anchor = data["similarity"].argmax(axis=-1)
        top_similarity = np.take_along_axis(
            data["similarity"], top_anchor[..., None], axis=-1,
        ).squeeze(-1)
        top_reliable = top_similarity >= q75[top_anchor]
        associated_class = anchor_semantic_class[top_anchor]
        for class_id, class_name in enumerate(classes):
            selected = data["label"] == class_id
            reliable_selected = top_reliable[selected]
            wrong = associated_class[selected] != class_id
            association_rows.append({
                "task": task, "variant": variant, "split": split,
                "domain": data["domain"], "class_id": class_id,
                "class_name": class_name, "sample_support": int(selected.sum()),
                "window_count": int(selected.sum() * top_anchor.shape[1]),
                "trusted_top_anchor_rate": (
                    float(reliable_selected.mean()) if selected.any() else float("nan")
                ),
                "wrong_source_semantic_rate_among_trusted": (
                    float(wrong[reliable_selected].mean())
                    if reliable_selected.any() else float("nan")
                ),
            })
        for anchor in range(16):
            sim = data["similarity"][:, :, anchor]
            weight = data["weights"][:, :, anchor]
            probability = data["anchor_distribution"][:, :, anchor]
            trusted = sim >= q75[anchor]
            low_high = (sim < q25[anchor]) & (weight >= weight_q75[anchor])
            low_relative = (
                (sim < q25[anchor]) & (probability >= probability_q75[anchor])
            )
            rows.append({
                "task": task, "variant": variant, "split": split, "anchor_id": anchor,
                "domain": data["domain"],
                "source_q25": q25[anchor], "source_q75": q75[anchor],
                "source_weight_q75": weight_q75[anchor],
                "source_anchor_probability_q75": probability_q75[anchor],
                "trusted_window_rate": float(trusted.mean()),
                "low_similarity_high_weight_rate": float(low_high.mean()),
                "low_similarity_high_anchor_probability_rate": float(low_relative.mean()),
                "mean_similarity": float(sim.mean()),
                "mean_anchor_probability": float(probability.mean()),
                "mean_pooling_weight": float(weight.mean()),
            })
            for class_id, class_name in enumerate(classes):
                selected = data["label"] == class_id
                semantic_rows.append({
                    "task": task, "variant": variant, "split": split,
                    "domain": data["domain"],
                    "anchor_id": anchor, "class_id": class_id, "class_name": class_name,
                    "support": int(selected.sum()),
                    "trusted_window_rate": float(trusted[selected].mean()) if selected.any() else float("nan"),
                    "low_similarity_high_weight_rate": float(low_high[selected].mean()) if selected.any() else float("nan"),
                    "low_similarity_high_anchor_probability_rate": float(low_relative[selected].mean()) if selected.any() else float("nan"),
                    "phase_magnitude_mean": float(np.linalg.norm(
                        data["response"][selected, 32:].reshape(-1, 4, 16)[:, :, anchor], axis=1,
                    ).mean()) if selected.any() else float("nan"),
                })
            flat = sim.reshape(-1)
            for flat_index in np.argsort(flat)[-2:][::-1]:
                sample, window = np.unravel_index(flat_index, sim.shape)
                candidates.append({
                    "task": task, "variant": variant, "split": split,
                    "domain": data["domain"],
                    "anchor_id": anchor, "parcel_index": int(data["parcel_id"][sample]),
                    "class_name": classes[data["label"][sample]], "window_index": int(window),
                    "window_center_day": float(data["window_centers_days"][window]),
                    "similarity": float(sim[sample, window]),
                    "anchor_probability": float(probability[sample, window]),
                    "pooling_weight": float(weight[sample, window]),
                    "curve": data["window_curve_norm"][sample, window].tolist(),
                })
    _write_csv(output / "D2_anchor_reliability.csv", rows)
    _write_csv(output / "D2_anchor_semantics.csv", semantic_rows)
    _write_csv(output / "D2_class_association.csv", association_rows)
    serializable = [{key: value for key, value in row.items() if key != "curve"} for row in candidates]
    _write_csv(output / "D2_examples.csv", serializable)
    _plot_d2_examples(candidates, output / "D2_representative_windows.png")
    return rows, semantic_rows, association_rows


def _plot_d2_examples(candidates, path):
    import matplotlib.pyplot as plt
    ranked = sorted(candidates, key=lambda row: row["similarity"], reverse=True)[:12]
    if not ranked: return
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), squeeze=False)
    for axis, row in zip(axes.flat, ranked):
        axis.plot(row["curve"])
        axis.set_title(
            f"{row['split']} A{row['anchor_id']} {row['class_name']}\n"
            f"sim={row['similarity']:.3f} day={row['window_center_day']:.1f}", fontsize=8,
        )
        axis.set_ylabel("canonical PSE latent norm")
    for axis in axes.flat[len(ranked):]: axis.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def analyze_d3(task, variant, arrays, classes, output):
    sample_rows, summary_rows, class_rows = [], [], []
    for split in ("source_val", "target_val"):
        data = arrays[split]; labels = data["label"]
        normal = data["normal_logits"]; neutral = data["neutral_logits"]
        normal_pred = normal.argmax(1); neutral_pred = neutral.argmax(1)
        normal_margin = _margin(torch.from_numpy(normal), torch.from_numpy(labels)).numpy()
        neutral_margin = _margin(torch.from_numpy(neutral), torch.from_numpy(labels)).numpy()
        attention_delta = np.abs(data["normal_attention"] - data["neutral_attention"]).mean(1)
        attention_cos = np.sum(data["normal_attention"] * data["neutral_attention"], 1) / np.maximum(
            np.linalg.norm(data["normal_attention"], axis=1) * np.linalg.norm(data["neutral_attention"], axis=1), 1e-12,
        )
        summary_rows.append({
            "task": task, "variant": variant, "split": split,
            "domain": data["domain"],
            "normal_fixed_macro_f1": _fixed_f1(labels, normal_pred, classes),
            "neutral_fixed_macro_f1": _fixed_f1(labels, neutral_pred, classes),
            "prediction_change_rate": float((normal_pred != neutral_pred).mean()),
            "mean_margin_change_normal_minus_neutral": float((normal_margin - neutral_margin).mean()),
            "mean_attention_absolute_difference": float(attention_delta.mean()),
            "mean_attention_cosine": float(attention_cos.mean()),
            "mean_query_correction_norm": float(data["query_correction_norm"].mean()),
        })
        for index in range(len(labels)):
            sample_rows.append({
                "task": task, "variant": variant, "split": split,
                "domain": data["domain"],
                "parcel_index": int(data["parcel_id"][index]), "true_class": classes[labels[index]],
                "normal_prediction": classes[normal_pred[index]], "neutral_prediction": classes[neutral_pred[index]],
                "normal_margin": float(normal_margin[index]), "neutral_margin": float(neutral_margin[index]),
                "attention_absolute_difference": float(attention_delta[index]),
                "attention_cosine": float(attention_cos[index]),
            })
        normal_f1 = f1_score(labels, normal_pred, labels=np.arange(len(classes)), average=None, zero_division=0)
        neutral_f1 = f1_score(labels, neutral_pred, labels=np.arange(len(classes)), average=None, zero_division=0)
        for class_id, name in enumerate(classes):
            selected = labels == class_id
            class_rows.append({
                "task": task, "variant": variant, "split": split,
                "domain": data["domain"],
                "class_id": class_id, "class_name": name, "support": int(selected.sum()),
                "normal_f1": normal_f1[class_id], "neutral_f1": neutral_f1[class_id],
                "f1_delta": normal_f1[class_id] - neutral_f1[class_id],
                "mean_margin_delta": float((normal_margin[selected] - neutral_margin[selected]).mean()) if selected.any() else float("nan"),
                "focus_class": name.lower().replace(" ", "_") in FOCUS[task],
            })
    _write_csv(output / "D3_summary.csv", summary_rows)
    _write_csv(output / "D3_samples.csv", sample_rows)
    _write_csv(output / "D3_per_class.csv", class_rows)
    return summary_rows, class_rows


def run_diagnose(args):
    checkpoint = Path(args.checkpoint)
    packet = require_complete_checkpoint(checkpoint)
    validated_config = validate_uda_checkpoint(packet, args.task, args.variant)
    source = validate_source(args.source_checkpoint, args.task, "cpu")
    model, config, loaded = strict_load_checkpoint(checkpoint, torch.device(args.device))
    if int(loaded.get("epoch", -1)) != 19:
        raise ValueError("diagnostic checkpoint is not final epoch 19")
    if list(config.classes) != source["classes"]:
        raise ValueError("source/UDA class mapping mismatch")
    _, _, _, datasets = official_protocol(config, args.data_root, args.seed)
    domains = {
        "source_train": config.source, "source_val": config.source,
        "target_val": config.target,
    }
    arrays = {
        name: extract_once(
            model, datasets[name], torch.device(args.device), args.batch_size,
            domains[name],
        )
        for name in domains
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    requested = tuple(args.stages)
    operations = {}
    if "D1" in requested:
        operations["D1"] = lambda: analyze_d1(
            args.task, args.variant, arrays, config.classes, output,
        )
    if "D2" in requested:
        operations["D2"] = lambda: analyze_d2(
            args.task, args.variant, arrays, config.classes, output,
        )
    if "D3" in requested:
        operations["D3"] = lambda: analyze_d3(
            args.task, args.variant, arrays, config.classes, output,
        )
    results, stage_status = run_independent_diagnostic_stages(operations)
    previous_manifest = output / "diagnostic_manifest.json"
    if previous_manifest.is_file():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        merged_status = dict(previous.get("stage_status", {}))
        merged_status.update(stage_status)
        stage_status = merged_status
    for stage, status in stage_status.items():
        error_path = output / f"{stage}_error.txt"
        if status["status"] == "failed":
            error_path.write_text(status.get("traceback", status["error"]), encoding="utf-8")
        elif error_path.exists():
            error_path.unlink()
    complete = all(
        stage_status.get(stage, {}).get("status") == "complete"
        for stage in ("D1", "D2", "D3")
    )
    manifest = {
        "task": args.task, "variant": args.variant, "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(packet["epoch"]), "source": source,
        "validated_uda_config": validated_config,
        "classes": list(config.classes), "sample_counts": {name: len(value["label"]) for name, value in arrays.items()},
        "target_labels_used_for": "offline_diagnostics_only", "inference_shift": 0,
        "recorded_global_temporal_shift": float(packet.get("global_temporal_shift", 0.)),
        "diagnostic_splits": ["source_train", "source_val", "target_val"],
        "target_test_used": False,
        "feature_extraction_passes_per_split": 1,
        "stage_status": stage_status,
        "status": "complete" if complete else "partial",
    }
    (output / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if complete and all(stage in results for stage in ("D1", "D2", "D3")):
        d1 = results["D1"]
        d2, semantics, associations = results["D2"]
        d3, d3_classes = results["D3"]
        _write_diagnostic_report(
            output, args.task, args.variant, d1, d2, semantics, associations,
            d3, d3_classes,
        )
    else:
        lines = [
            f"# {args.variant} {args.task} frozen diagnostics", "",
            "Stages execute independently; failed stages do not block peer outputs.", "",
        ]
        for stage in ("D1", "D2", "D3"):
            status = stage_status.get(stage, {}).get("status", "not_run")
            lines.append(f"- {stage}: {status}")
        (output / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    failures = {
        stage: status["error"] for stage, status in stage_status.items()
        if status["status"] == "failed"
    }
    if failures:
        raise RuntimeError(
            "diagnostic stages failed after independent execution: "
            + json.dumps(failures, sort_keys=True)
        )
    print(f"EG_DIAGNOSTIC_COMPLETE|task={args.task}|variant={args.variant}|output={output}")


def _write_diagnostic_report(
    output, task, variant, d1, d2, semantics, associations, d3, d3_classes,
):
    target_d1 = [row for row in d1 if row["split"] == "target_val" and row["class_name"].lower().replace(" ", "_") in FOCUS[task]]
    target_d2 = [row for row in d2 if row["split"] == "target_val"]
    target_d3 = [row for row in d3 if row["split"] == "target_val"]
    lines = [f"# {variant} {task} frozen diagnostics", "", "These are associative frozen-probe/intervention observations, not causal conclusions.", "", "## D1 class competition", ""]
    for row in target_d1:
        lines.append(f"- {row['class_name']}: F1={row['f1']:.4f}, margin={row['mean_margin']:.4f}, strength={row['mean_strength_contribution']:.4f}, concentration={row['mean_concentration_contribution']:.4f}, phase={row['mean_phase_contribution']:.4f}.")
    lines += ["", "## D2 anchor reliability", ""]
    if target_d2:
        lines.append(f"- Mean low-similarity/high-weight rate: {np.mean([row['low_similarity_high_weight_rate'] for row in target_d2]):.6f}.")
    target_associations = [row for row in associations if row["split"] == "target_val"]
    valid_wrong = [
        row["wrong_source_semantic_rate_among_trusted"]
        for row in target_associations
        if np.isfinite(row["wrong_source_semantic_rate_among_trusted"])
    ]
    if valid_wrong:
        lines.append(
            f"- Mean wrong Source-anchor semantic association among trusted Target windows: {np.mean(valid_wrong):.6f}."
        )
    lines += ["", "## D3 query function", ""]
    for row in target_d3:
        lines.append(f"- Normal F1={row['normal_fixed_macro_f1']:.4f}; Neutral F1={row['neutral_fixed_macro_f1']:.4f}; prediction change={row['prediction_change_rate']:.4f}; attention MAD={row['mean_attention_absolute_difference']:.6f}.")
    lines += ["", "Evidence: D1_*.csv, D2_*.csv, D3_*.csv and D2_representative_windows.png.", ""]
    (output / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_summarize_task(args):
    root = Path(args.experiment_root)
    rows, class_rows, history_rows, config_rows = [], [], [], []
    for variant in VARIANTS:
        checkpoint = root / "outputs" / "uda" / f"{variant}_{args.task}_seed1/fold_0/checkpoint_last.pt"
        packet = require_complete_checkpoint(checkpoint)
        validated = validate_uda_checkpoint(packet, args.task, variant)
        result = evaluate_checkpoint_data(checkpoint, args.data_root, args.device, args.batch_size)
        source_sha = json.loads((root / "diagnostics" / f"{variant}_{args.task}" / "diagnostic_manifest.json").read_text(encoding="utf-8"))["source"]["sha256"]
        rows.append({
            "task": args.task, "variant": variant, "source_sha256": source_sha,
            "checkpoint": str(checkpoint), "epoch": result["checkpoint_epoch"],
            "test_macro_f1": result["test_macro_f1"],
            "fixed_class_macro_f1": result["fixed_class_macro_f1"],
        })
        config_rows.extend({
            "task": args.task, "variant": variant, "key": key,
            "value": json.dumps(value, sort_keys=True),
        } for key, value in validated.items())
        for class_id, class_name in enumerate(result["class_names"]):
            class_rows.append({
                "task": args.task, "variant": variant, "class_id": class_id,
                "class_name": class_name, "support": result["support"][class_id],
                "f1": result["per_class_f1"][class_id],
            })
        log = root / "logs" / f"{variant}_{args.task}_seed1.log"
        history_rows.extend(parse_epoch_history(log.read_text(encoding="utf-8", errors="replace"), variant, args.task))
    if rows[0]["source_sha256"] != rows[1]["source_sha256"]:
        raise RuntimeError("paired E/G diagnostics used different source checkpoints")
    output = root / "reports" / args.task; output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "final_metrics.csv", rows)
    _write_csv(output / "per_class_f1.csv", class_rows)
    _write_csv(output / "training_trajectory.csv", history_rows)
    _write_csv(output / "checkpoint_config.csv", config_rows)
    lines = [f"# E/G paired report: {args.task}", "", "E and G share the Source SHA256 shown in final_metrics.csv and differ only in target query detach epochs.", ""]
    for row in rows:
        lines.append(f"- {row['variant']}: final Macro-F1={row['test_macro_f1']:.4f}; fixed-class Macro-F1={row['fixed_class_macro_f1']:.4f}.")
    lines += ["", "## Frozen diagnostics", ""]
    for variant in VARIANTS:
        diagnostic = root / "diagnostics" / f"{variant}_{args.task}"
        d1 = _read_csv(diagnostic / "D1_per_class.csv")
        d2 = _read_csv(diagnostic / "D2_anchor_reliability.csv")
        d3 = _read_csv(diagnostic / "D3_summary.csv")
        target_d1 = [row for row in d1 if row["split"] == "target_val"]
        target_d2 = [row for row in d2 if row["split"] == "target_val"]
        target_d3 = next(row for row in d3 if row["split"] == "target_val")
        lines.append(
            f"- {variant}: mean target probe margin="
            f"{np.mean([float(row['mean_margin']) for row in target_d1]):.4f}; "
            f"low-sim/high-weight rate="
            f"{np.mean([float(row['low_similarity_high_weight_rate']) for row in target_d2]):.6f}; "
            f"Normal-Neutral F1={float(target_d3['normal_fixed_macro_f1']) - float(target_d3['neutral_fixed_macro_f1']):+.4f}; "
            f"query prediction-change={float(target_d3['prediction_change_rate']):.4f}."
        )
    lines += [
        "",
        "These are frozen-probe and inference-intervention observations. Anchor IDs have meaning only within one checkpoint; canonical PSE latent curves are not NDVI. The results establish associations and functional query effects, not a causal explanation of training outcomes.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "summary_manifest.json").write_text(json.dumps({"status": "complete", "task": args.task}, indent=2), encoding="utf-8")


def run_summarize_all(args):
    root = Path(args.experiment_root); rows = []
    for task in TASKS:
        path = root / "reports" / task / "final_metrics.csv"
        if not path.is_file(): continue
        with path.open(newline="", encoding="utf-8") as stream: rows.extend(csv.DictReader(stream))
    _write_csv(root / "reports" / "EG_final_comparison.csv", rows)
    lines = ["# E/G paired recheck", "", "All metrics use checkpoint_last.pt (epoch 19). E/G use the same recheck Source checkpoint per task and differ only in target_structure_detach_epochs.", ""]
    for row in rows:
        lines.append(f"- {row['task']} {row['variant']}: Macro-F1={float(row['test_macro_f1']):.4f}, fixed={float(row['fixed_class_macro_f1']):.4f}.")
    lines += [
        "",
        "D1 reports readable class evidence and competing classes; D2 reports source-calibrated anchor reliability without assigning cross-checkpoint anchor identity; D3 measures the functional effect of removing only the external query correction. These observations do not by themselves identify a causal training mechanism or justify a new loss.",
        "",
    ]
    (root / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("verify-source", "diagnose", "summarize-task", "summarize-all"), required=True)
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--checkpoint")
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--experiment-root", default="experiments/eg_recheck_pair_seed1")
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--stages", nargs="+", choices=("D1", "D2", "D3"),
        default=("D1", "D2", "D3"),
    )
    return parser


def run(args):
    if args.mode == "verify-source":
        print(json.dumps(validate_source(args.source_checkpoint, args.task, args.device), indent=2))
    elif args.mode == "diagnose": run_diagnose(args)
    elif args.mode == "summarize-task": run_summarize_task(args)
    else: run_summarize_all(args)


if __name__ == "__main__": run(build_parser().parse_args())
