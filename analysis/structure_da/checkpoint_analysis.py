"""Read-only post-hoc checkpoint diagnostics for existing Structure DA models."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
import torch

from .log_parser import ParsedRun, parse_task_log


def component_energy_ratios(trend: np.ndarray, dynamics: np.ndarray, residual: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return per-sample mean squared-energy ratios in T/D/R order."""

    arrays = [np.asarray(value, dtype=np.float64) for value in (trend, dynamics, residual)]
    energies = np.stack([np.square(value).sum(axis=-1).mean(axis=-1) for value in arrays], axis=-1)
    return energies / (energies.sum(axis=-1, keepdims=True) + eps)


def component_similarities(trend: np.ndarray, dynamics: np.ndarray, residual: np.ndarray, eps: float = 1e-8) -> dict[str, np.ndarray]:
    """Return sample-level flattened cosine similarities."""

    arrays = [np.asarray(value, dtype=np.float64).reshape(len(value), -1) for value in (trend, dynamics, residual)]
    result = {}
    for name, left, right in (("T_D", arrays[0], arrays[1]), ("T_R", arrays[0], arrays[2]), ("D_R", arrays[1], arrays[2])):
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        result[name] = np.sum(left * right, axis=1) / np.maximum(denominator, eps)
    return result


def combine_domain_component_metrics(
    source: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute temporal diagnostics per domain, then merge sample-level results."""

    source_ratios = component_energy_ratios(
        source["trend"], source["dynamics"], source["residual"]
    )
    target_ratios = component_energy_ratios(
        target["trend"], target["dynamics"], target["residual"]
    )
    source_similarity = component_similarities(
        source["trend"], source["dynamics"], source["residual"]
    )
    target_similarity = component_similarities(
        target["trend"], target["dynamics"], target["residual"]
    )
    similarities = {
        name: np.concatenate([source_similarity[name], target_similarity[name]], axis=0)
        for name in source_similarity
    }
    return np.concatenate([source_ratios, target_ratios], axis=0), similarities


def fit_pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit deterministic two-dimensional PCA on the supplied joint features."""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) < 2 or features.shape[1] < 2:
        raise ValueError("features must have shape [N,D] with N,D >= 2")
    model = PCA(n_components=2, svd_solver="full")
    return model.fit_transform(features), model.explained_variance_ratio_


def quality_scores_long_form(
    run: str,
    domains: np.ndarray,
    classes: np.ndarray,
    branches: Mapping[str, Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    """Create the specified sample-level long-form quality table."""

    rows = []
    for branch_name, metrics in branches.items():
        branch_type = "component" if branch_name.endswith("component") else "structure"
        for metric, values in metrics.items():
            values = np.asarray(values)
            if len(values) != len(domains):
                raise ValueError(f"{branch_name}.{metric} length does not match labels")
            rows.extend({
                "run": run, "domain": str(domain), "class": str(class_name),
                "branch_type": branch_type, "branch_name": branch_name,
                "metric": metric, "value": float(value),
            } for domain, class_name, value in zip(domains, classes, values))
    return pd.DataFrame(rows)


def diversity_diagnostics(
    diversity: Mapping[str, np.ndarray], labels: np.ndarray,
    class_names: Sequence[str], eps: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize source diversity with population std (unbiased=False)."""

    class_rows, summary_rows = [], []
    labels = np.asarray(labels)
    for branch, values in diversity.items():
        values = np.asarray(values, dtype=np.float64)
        means = []
        for class_index, class_name in enumerate(class_names):
            selected = values[labels == class_index]
            if not len(selected):
                continue
            mean = float(selected.mean())
            means.append(mean)
            class_rows.append({"branch_name": branch, "class_name": class_name, "mean": mean, "n_samples": len(selected)})
        mean_of_means = float(np.mean(means))
        std_of_means = float(np.std(means, ddof=0))
        summary_rows.append({
            "branch_name": branch, "mean_of_class_means": mean_of_means,
            "std_of_class_means": std_of_means,
            "cv": std_of_means / (mean_of_means + eps),
            "n_classes": len(means),
        })
    return pd.DataFrame(class_rows), pd.DataFrame(summary_rows)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _model_from_run(run: ParsedRun, checkpoint_path: Path, device: torch.device):
    from methods.structure_da import StructureDAModel

    config = run.config
    model = StructureDAModel(
        num_classes=run.num_classes, input_dim=int(config.get("input_dim", 10)),
        with_extra=bool(config.get("with_extra", False)),
        time_scale=float(config.get("time_scale", 365.0)),
        tau_fast_init=float(config.get("tau_fast_init", 0.05)),
        tau_slow_init=float(config.get("tau_slow_init", 0.2)),
        tau_min=float(config.get("tau_min", 1e-4)),
        delta_tau_min=float(config.get("delta_tau_min", 1e-4)),
        quality_hidden_cap=int(config.get("quality_hidden_cap", 128)),
        quality_eta=float(config.get("quality_eta", 0.1)),
        sda_hidden_dim=int(config.get("sda_hidden_dim", 128)),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" not in checkpoint:
        raise KeyError("checkpoint must contain state_dict")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model


def _deterministic_indices(labels: np.ndarray, samples_per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(np.unique(labels)):
        candidates = np.flatnonzero(labels == label)
        count = min(samples_per_class, len(candidates))
        selected.extend(rng.choice(candidates, count, replace=False).tolist())
    return sorted(selected)


def _sample_pixels(pixels: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically mirror RandomSamplePixels, including its padding mask."""

    pixel_count = pixels.shape[-1]
    if pixel_count > count:
        indices = np.random.default_rng(seed).choice(pixel_count, count, replace=False)
        sampled = pixels[..., indices]
        valid = np.ones(count, dtype=np.float32)
    elif pixel_count < count:
        sampled = np.empty((*pixels.shape[:-1], count), dtype=pixels.dtype)
        sampled[..., :pixel_count] = pixels
        sampled[..., pixel_count:] = pixels[..., :1]
        valid = np.concatenate(
            (np.ones(pixel_count, dtype=np.float32), np.zeros(count - pixel_count, dtype=np.float32))
        )
    else:
        sampled = pixels
        valid = np.ones(count, dtype=np.float32)
    normalized = np.clip(sampled, 0, 65535).astype(np.float32) / 65535.0
    return normalized, np.repeat(valid[None], pixels.shape[0], axis=0)


def normalize_extra_features(extra: np.ndarray) -> np.ndarray:
    """Apply the same approximate geometric maxima as transforms.Normalize."""

    maxima = np.asarray([40000.0, 1e8, 40000.0, 1.0], dtype=np.float32)
    return np.asarray(extra, dtype=np.float32) / maxima


def _quality_branches(component) -> dict[str, object]:
    return {
        "T_temp": component.structural_quality.trend_temporal,
        "D_temp": component.structural_quality.dynamics_temporal,
        "D_channel": component.structural_quality.dynamics_channel,
        "T_component": component.component_quality.trend,
        "D_component": component.component_quality.dynamics,
        "R_component": component.component_quality.residual,
    }


def _collect_domain(model, dataset, domain: str, samples_per_class: int, diagnostic_seed: int, num_pixels: int, device: torch.device) -> dict[str, object]:
    from methods.structure_da import vectorize_channel_statistic

    indices = _deterministic_indices(dataset.get_labels(), samples_per_class, diagnostic_seed)
    result: dict[str, object] = {
        "domain": [], "label": [], "class": [], "dates": tuple(str(value) for value in dataset.metadata["dates"]),
        "embedding": {name: [] for name in ("trend", "dynamics", "residual")},
        "statistic": {name: [] for name in ("T_temp", "D_temp", "D_channel")},
        "gate": {name: [] for name in ("beta_trend_temporal", "beta_dynamics_temporal", "beta_dynamics_channel", "q_trend", "q_dynamics", "q_residual")},
        "quality": {name: {metric: [] for metric in ("transferability", "entropy", "confidence", "domain_logits", "class_logits", "diversity")} for name in ("T_temp", "D_temp", "D_channel", "T_component", "D_component", "R_component")},
        "energy_curve": {name: [] for name in ("trend", "dynamics", "residual")},
        "components": {name: [] for name in ("trend", "dynamics", "residual")},
        "joint": [],
    }
    with torch.no_grad():
        for index in indices:
            sample = dataset[index]
            pixels_np, valid_np = _sample_pixels(
                sample["pixels"], num_pixels, diagnostic_seed + int(index)
            )
            pixels = torch.from_numpy(pixels_np)[None].to(device)
            valid = torch.from_numpy(valid_np)[None].to(device)
            positions = torch.as_tensor(sample["positions"], device=device)
            extra = torch.as_tensor(
                normalize_extra_features(sample["extra"]), dtype=torch.float32, device=device
            )[None]
            output = model.forward_details(pixels, valid, positions, extra, quality_progress=1.0)
            component = output.component
            label = int(sample["label"])
            result["domain"].append(domain); result["label"].append(label); result["class"].append(dataset.classes[label])
            for name, tensor in (("trend", component.trend_embedding), ("dynamics", component.dynamics_embedding), ("residual", component.residual_embedding)):
                result["embedding"][name].append(tensor[0].cpu().numpy())
            stats = {
                "T_temp": component.trend_temporal.statistic,
                "D_temp": component.dynamics_temporal.statistic,
                "D_channel": vectorize_channel_statistic(component.dynamics_channel.statistic),
            }
            for name, tensor in stats.items(): result["statistic"][name].append(tensor[0].cpu().numpy())
            for name in result["gate"]: result["gate"][name].append(float(getattr(component.effective_gates, name)[0].cpu()))
            for branch_name, branch in _quality_branches(component).items():
                scores = branch.scores
                for metric in ("transferability", "entropy", "confidence"):
                    result["quality"][branch_name][metric].append(float(getattr(scores, metric)[0].cpu()))
                result["quality"][branch_name]["domain_logits"].append(scores.domain_logits[0].cpu().numpy())
                result["quality"][branch_name]["class_logits"].append(scores.class_logits[0].cpu().numpy())
                if hasattr(branch, "diversity"):
                    result["quality"][branch_name]["diversity"].append(float(branch.diversity[0].cpu()))
            decomposed = {"trend": component.decomposition.trend, "dynamics": component.decomposition.dynamics, "residual": component.decomposition.residual}
            for name, tensor in decomposed.items():
                array = tensor[0].cpu().numpy()
                result["components"][name].append(array)
                result["energy_curve"][name].append(np.linalg.norm(array, axis=-1) / math.sqrt(array.shape[-1]))
            result["joint"].append(model.build_joint_structure(output).joint[0].cpu().numpy())
    for key in ("domain", "label", "class", "joint"):
        result[key] = np.asarray(result[key])
    for section in ("embedding", "statistic", "gate", "energy_curve", "components"):
        result[section] = {name: np.asarray(values) for name, values in result[section].items()}
    for branch, metrics in result["quality"].items():
        result["quality"][branch] = {name: np.asarray(values) for name, values in metrics.items() if values}
    return result


def _aggregate_numeric(frame: pd.DataFrame, value_columns: Sequence[str], group_columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(list(group_columns), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        for column in value_columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = values.mean(); row[f"{column}_std"] = values.std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def _pca_plot(features: np.ndarray, classes: np.ndarray, domains: np.ndarray, path: Path, color_by: str, title: str) -> None:
    coordinates, variance = fit_pca_2d(features)
    fig, axis = plt.subplots(figsize=(7, 5.5))
    if color_by == "class":
        class_names = sorted(set(classes))
        palette = plt.get_cmap("tab20")
        colors = {name: palette(index % 20) for index, name in enumerate(class_names)}
        markers = {"source": "o", "target": "^"}
        for class_name in class_names:
            for domain in sorted(set(domains)):
                selected = (classes == class_name) & (domains == domain)
                if np.any(selected):
                    axis.scatter(
                        coordinates[selected, 0], coordinates[selected, 1],
                        s=14, alpha=0.55, color=colors[class_name],
                        marker=markers.get(str(domain), "s"), label=f"{class_name} | {domain}",
                    )
    else:
        for domain in sorted(set(domains)):
            selected = domains == domain
            axis.scatter(
                coordinates[selected, 0], coordinates[selected, 1],
                s=12, alpha=0.55, label=domain,
            )
    axis.set_xlabel(f"PC1 ({100 * variance[0]:.1f}%)"); axis.set_ylabel(f"PC2 ({100 * variance[1]:.1f}%)")
    axis.set_title(title); axis.legend(fontsize=7, ncol=2)
    _save(fig, path)


def _gate_plot(frame: pd.DataFrame, columns: Sequence[str], group: str, path: Path, title: str) -> None:
    groups = sorted(frame[group].unique())
    fig, axes = plt.subplots(len(columns), 1, figsize=(max(7, 0.7 * len(groups)), 2.8 * len(columns)), squeeze=False)
    for axis, column in zip(axes[:, 0], columns):
        values = [frame.loc[frame[group] == name, column].to_numpy() for name in groups]
        axis.boxplot(values, tick_labels=groups, showfliers=False)
        axis.set_ylabel(column)
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(title)
    _save(fig, path)


def _component_energy_curves(source: dict[str, object], target: dict[str, object], classes: Sequence[str], root: Path) -> None:
    colors = {"trend": "C0", "dynamics": "C1", "residual": "C2"}
    for class_name in classes:
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=False)
        for axis, component in zip(axes, ("trend", "dynamics", "residual")):
            for domain_data, domain_name in ((source, "source"), (target, "target")):
                selected = domain_data["class"] == class_name
                if not np.any(selected):
                    continue
                dates = pd.to_datetime(list(domain_data["dates"]), format="%Y%m%d")
                curves = domain_data["energy_curve"][component][selected]
                axis.plot(dates, curves.mean(axis=0), label=domain_name, color=colors[component], linestyle="-" if domain_name == "source" else "--")
            axis.set_ylabel(f"{component} energy")
            axis.grid(alpha=0.2)
            axis.legend()
        axes[-1].set_xlabel("Real acquisition date")
        fig.suptitle(f"{class_name}: component energy")
        _save(fig, root / f"{class_name.replace('/', '_')}.png")


def _representation_geometry(features: Mapping[str, np.ndarray], classes: np.ndarray, domains: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, values in features.items():
        class_centroids = []
        within = []
        for class_name in sorted(set(classes)):
            selected = values[classes == class_name]
            if not len(selected): continue
            centroid = selected.mean(axis=0); class_centroids.append(centroid)
            within.extend(np.square(selected - centroid).sum(axis=1))
        between = np.var(np.stack(class_centroids), axis=0, ddof=0).sum() if len(class_centroids) > 1 else 0.0
        rows.append({"representation": name, "metric": "within_class_variance", "value": np.mean(within)})
        rows.append({"representation": name, "metric": "between_class_centroid_variance", "value": between})
        for class_name in sorted(set(classes)):
            source = values[(classes == class_name) & (domains == "source")]
            target = values[(classes == class_name) & (domains == "target")]
            if len(source) and len(target):
                rows.append({"representation": name, "metric": "same_class_source_target_centroid_distance", "class": class_name, "value": np.linalg.norm(source.mean(axis=0) - target.mean(axis=0))})
    return pd.DataFrame(rows)


def run_checkpoint_analysis(task_log: Path | str, checkpoint: Path | str, data_root: Path | str, output_dir: Path | str, samples_per_class: int = 200, device: str = "cuda", diagnostic_seed: int = 0) -> dict[str, object]:
    """Run bounded post-hoc inference only; never train or call backward."""

    from dataset import PixelSetData

    print("TARGET LABELS ARE USED FOR POST-HOC DIAGNOSTICS ONLY, NOT TRAINING OR MODEL SELECTION.")
    task_log, checkpoint = Path(task_log), Path(checkpoint)
    data_root, output_dir = Path(data_root), Path(output_dir)
    run = parse_task_log(task_log)
    torch_device = torch.device(device)
    model = _model_from_run(run, checkpoint, torch_device)
    classes = list(run.classes)
    source_dataset = PixelSetData(str(data_root), run.config["source"], classes, transform=None, closed_set=True, combine_spring_and_winter=False)
    target_dataset = PixelSetData(str(data_root), run.config["target"], classes, transform=None, closed_set=True, combine_spring_and_winter=False)
    num_pixels = int(run.config.get("num_pixels", 64))
    source = _collect_domain(model, source_dataset, "source", samples_per_class, diagnostic_seed, num_pixels, torch_device)
    target = _collect_domain(model, target_dataset, "target", samples_per_class, diagnostic_seed, num_pixels, torch_device)
    domains = np.concatenate([source["domain"], target["domain"]]); labels = np.concatenate([source["label"], target["label"]]); class_labels = np.concatenate([source["class"], target["class"]])
    table_dir = output_dir / "tables"
    checkpoint_table_dir = table_dir / "checkpoint" / run.run_name
    figure_dir = output_dir / "figures" / "checkpoint" / run.run_name
    checkpoint_table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    ratios, similarities = combine_domain_component_metrics(
        source["components"], target["components"]
    )
    ratio_frame = pd.DataFrame({"domain": domains, "class": class_labels, "trend_ratio": ratios[:, 0], "dynamics_ratio": ratios[:, 1], "residual_ratio": ratios[:, 2]})
    energy_table = _aggregate_numeric(ratio_frame, ("trend_ratio", "dynamics_ratio", "residual_ratio"), ("domain", "class"))
    energy_table.to_csv(table_dir / f"checkpoint_{run.run_name}_component_energy.csv", index=False)
    similarity_frame = pd.DataFrame({"domain": domains, "class": class_labels, **similarities})
    similarity_table = _aggregate_numeric(similarity_frame, tuple(similarities), ("domain", "class"))
    similarity_table.to_csv(checkpoint_table_dir / "component_similarity.csv", index=False)
    _component_energy_curves(source, target, classes, figure_dir / "component_energy")

    embeddings = {name: np.concatenate([source["embedding"][name], target["embedding"][name]]) for name in ("trend", "dynamics", "residual")}
    statistics = {name: np.concatenate([source["statistic"][name], target["statistic"][name]]) for name in ("T_temp", "D_temp", "D_channel")}
    for name, values in embeddings.items():
        _pca_plot(values, class_labels, domains, figure_dir / f"embedding_pca_{name}_class.png", "class", f"{name} embedding PCA by class")
        _pca_plot(values, class_labels, domains, figure_dir / f"embedding_pca_{name}_domain.png", "domain", f"{name} embedding PCA by domain")
    for name, values in statistics.items():
        stem = {"T_temp": "T_temp", "D_temp": "D_temp", "D_channel": "D_channel"}[name]
        _pca_plot(values, class_labels, domains, figure_dir / f"structure_pca_{stem}_class.png", "class", f"{name} structure PCA by class")
        _pca_plot(values, class_labels, domains, figure_dir / f"structure_pca_{stem}_domain.png", "domain", f"{name} structure PCA by domain")
    _representation_geometry({**{f"embedding_{k}": v for k, v in embeddings.items()}, **{f"structure_{k}": v for k, v in statistics.items()}}, class_labels, domains).to_csv(checkpoint_table_dir / "representation_geometry.csv", index=False)

    gates = {name: np.concatenate([source["gate"][name], target["gate"][name]]) for name in source["gate"]}
    gate_frame = pd.DataFrame({"domain": domains, "class": class_labels, **gates})
    gate_rows = []
    for (domain_name, class_name), group in gate_frame.groupby(["domain", "class"]):
        for gate in gates:
            values = group[gate].to_numpy()
            gate_rows.append({"domain": domain_name, "class": class_name, "gate": gate, "mean": values.mean(), "std": values.std(ddof=0), "median": np.median(values), "q25": np.quantile(values, .25), "q75": np.quantile(values, .75)})
    pd.DataFrame(gate_rows).to_csv(checkpoint_table_dir / "quality_gates.csv", index=False)
    beta_columns = ("beta_trend_temporal", "beta_dynamics_temporal", "beta_dynamics_channel")
    q_columns = ("q_trend", "q_dynamics", "q_residual")
    _gate_plot(gate_frame, beta_columns, "domain", figure_dir / "beta_by_domain.png", "Structural beta gates by domain")
    _gate_plot(gate_frame, q_columns, "domain", figure_dir / "q_by_domain.png", "Component q gates by domain")
    _gate_plot(gate_frame, beta_columns, "class", figure_dir / "beta_by_class.png", "Structural beta gates by class")
    _gate_plot(gate_frame, q_columns, "class", figure_dir / "q_by_class.png", "Component q gates by class")

    quality_arrays = {}
    for branch in source["quality"]:
        quality_arrays[branch] = {}
        for metric in ("transferability", "entropy", "confidence", "diversity"):
            if metric in source["quality"][branch]:
                quality_arrays[branch][metric] = np.concatenate([source["quality"][branch][metric], target["quality"][branch][metric]])
    quality_frame = quality_scores_long_form(run.run_name, domains, class_labels, quality_arrays)
    quality_frame.to_csv(checkpoint_table_dir / "quality_scores.csv", index=False)

    source_diversity = {branch: source["quality"][branch]["diversity"] for branch in ("T_component", "D_component", "R_component")}
    diversity_classes, diversity_summary = diversity_diagnostics(source_diversity, source["label"], classes)
    diversity_classes.to_csv(checkpoint_table_dir / "diversity_class_means.csv", index=False)
    diversity_summary.to_csv(checkpoint_table_dir / "diversity_cv_summary.csv", index=False)

    domain_rows, class_rows = [], []
    for branch in source["quality"]:
        source_logits = source["quality"][branch]["domain_logits"]; target_logits = target["quality"][branch]["domain_logits"]
        source_pred = source_logits.argmax(axis=1); target_pred = target_logits.argmax(axis=1)
        p_source_source = torch.softmax(torch.as_tensor(source_logits), dim=1)[:, 1].numpy(); p_source_target = torch.softmax(torch.as_tensor(target_logits), dim=1)[:, 1].numpy()
        domain_rows.append({"branch_name": branch, "source_accuracy": np.mean(source_pred == 1), "target_accuracy": np.mean(target_pred == 0), "overall_accuracy": (np.sum(source_pred == 1) + np.sum(target_pred == 0)) / (len(source_pred) + len(target_pred)), "p_source_source_mean": p_source_source.mean(), "p_source_target_mean": p_source_target.mean()})
        source_class_logits = source["quality"][branch]["class_logits"]; predictions = source_class_logits.argmax(axis=1)
        class_rows.append({"branch_name": branch, "accuracy": accuracy_score(source["label"], predictions), "macro_f1": f1_score(source["label"], predictions, average="macro", zero_division=0)})
        fig, axis = plt.subplots(figsize=(6, 4)); axis.hist(p_source_source, bins=25, alpha=.55, label="source"); axis.hist(p_source_target, bins=25, alpha=.55, label="target"); axis.set_title(f"p(source): {branch}"); axis.legend(); _save(fig, figure_dir / f"p_source_distribution_{branch}.png")
    pd.DataFrame(domain_rows).to_csv(checkpoint_table_dir / "quality_domain_diagnostics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(checkpoint_table_dir / "quality_class_diagnostics.csv", index=False)

    with torch.no_grad():
        source_sda = model.adversarial_adapter.discriminator(torch.as_tensor(source["joint"], dtype=torch.float32, device=torch_device)).cpu().numpy()
        target_sda = model.adversarial_adapter.discriminator(torch.as_tensor(target["joint"], dtype=torch.float32, device=torch_device)).cpu().numpy()
    source_score = 1 / (1 + np.exp(-source_sda)); target_score = 1 / (1 + np.exp(-target_sda))
    sda_row = {"source_accuracy": np.mean(source_score >= .5), "target_accuracy": np.mean(target_score < .5), "overall_accuracy": (np.sum(source_score >= .5) + np.sum(target_score < .5)) / (len(source_score) + len(target_score)), "source_score_mean": source_score.mean(), "target_score_mean": target_score.mean()}
    pd.DataFrame([sda_row]).to_csv(checkpoint_table_dir / "sda_domain_diagnostics.csv", index=False)
    fig, axis = plt.subplots(figsize=(6, 4)); axis.hist(source_score, bins=25, alpha=.55, label="source"); axis.hist(target_score, bins=25, alpha=.55, label="target"); axis.set_title("SDA discriminator sigmoid score"); axis.legend(); _save(fig, figure_dir / "sda_score_distribution.png")

    for table, filename, columns in ((energy_table, "component_energy_ratio.png", ("trend_ratio_mean", "dynamics_ratio_mean", "residual_ratio_mean")), (similarity_table, "component_similarity.png", ("T_D_mean", "T_R_mean", "D_R_mean"))):
        fig, axis = plt.subplots(figsize=(9, 4.5)); table.set_index(["domain", "class"])[list(columns)].plot(kind="bar", ax=axis); axis.set_title(filename.replace("_", " ").replace(".png", "")); axis.legend(fontsize=7); _save(fig, figure_dir / filename)
    for filename, table, value in (("diversity_by_class.png", diversity_classes, "mean"),):
        fig, axis = plt.subplots(figsize=(8, 4.5)); table.pivot(index="class_name", columns="branch_name", values=value).plot(kind="bar", ax=axis); axis.set_title("Source diversity by class"); _save(fig, figure_dir / filename)
    fig, axis = plt.subplots(figsize=(7, 4)); [axis.hist(values, bins=25, alpha=.45, label=name) for name, values in source_diversity.items()]; axis.legend(); axis.set_title("Source diversity distributions"); _save(fig, figure_dir / "diversity_distribution.png")

    export_dir = output_dir / "checkpoint_exports"; export_dir.mkdir(parents=True, exist_ok=True)
    quality_export = {
        f"quality_{branch}_{metric}": values
        for branch, metrics in quality_arrays.items()
        for metric, values in metrics.items()
    }
    np.savez_compressed(
        export_dir / f"{run.run_name}.npz", label=labels, domain=domains,
        **{f"embedding_{name}": values for name, values in embeddings.items()},
        **{f"structure_{name}": values for name, values in statistics.items()},
        **{f"gate_{name}": values for name, values in gates.items()},
        **quality_export,
    )
    manifest = {"command": "checkpoint", "timestamp": datetime.now(timezone.utc).isoformat(), "input_experiment_dir": None, "data_root": str(data_root), "task_logs_used": [str(task_log)], "completed": None, "incomplete": None, "failed": None, "diagnostic_sampling_seed": diagnostic_seed, "samples_per_class": samples_per_class, "checkpoint_path": str(checkpoint)}
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"run": run, "manifest": manifest}
