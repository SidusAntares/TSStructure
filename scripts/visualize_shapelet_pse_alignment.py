#!/usr/bin/env python3
"""Offline START/END comparison of source/target PSE temporal features."""

import argparse
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def fixed_class_sample_indices(labels, class_ids, per_class, seed):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in class_ids:
        candidates = np.flatnonzero(labels == class_id)
        count = min(int(per_class), len(candidates))
        if count:
            selected.extend(np.sort(rng.choice(candidates, size=count, replace=False)).tolist())
    return np.asarray(selected, dtype=np.int64)


def _sample_rows(values, limit, rng):
    if len(values) <= limit:
        return values
    return values[rng.choice(len(values), size=limit, replace=False)]


def project_feature_groups(
    groups, seed, max_points_per_domain=50_000, pca_factory=None,
):
    if pca_factory is None:
        from sklearn.decomposition import PCA
        pca_factory = PCA
    required = {"start_source", "start_target", "end_source", "end_target"}
    if set(groups) != required:
        raise ValueError(f"feature groups must be exactly {sorted(required)}")
    rng = np.random.default_rng(seed)
    start_source = groups["start_source"].reshape(-1, groups["start_source"].shape[-1])
    start_target = groups["start_target"].reshape(-1, groups["start_target"].shape[-1])
    fit_values = np.concatenate([
        _sample_rows(start_source, max_points_per_domain, rng),
        _sample_rows(start_target, max_points_per_domain, rng),
    ], axis=0)
    pca = pca_factory(n_components=2)
    pca.fit(fit_values)
    projected = {}
    for name, values in groups.items():
        flat = values.reshape(-1, values.shape[-1])
        projected[name] = pca.transform(flat).reshape(*values.shape[:-1], 2)
    return pca, projected


def temporal_bin_means(positions, scores, num_bins=24):
    positions = np.asarray(positions, dtype=float).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if positions.shape != scores.shape:
        raise ValueError("positions and scores must contain the same number of observations")
    edges = np.linspace(0., 365., int(num_bins) + 1)
    centers = (edges[:-1] + edges[1:]) / 2.
    means = np.full(int(num_bins), np.nan, dtype=float)
    valid = np.isfinite(positions) & np.isfinite(scores) & (positions >= 0.) & (positions <= 365.)
    bin_ids = np.searchsorted(edges, positions[valid], side="right") - 1
    bin_ids = np.clip(bin_ids, 0, int(num_bins) - 1)
    valid_scores = scores[valid]
    for bin_id in range(int(num_bins)):
        in_bin = bin_ids == bin_id
        if np.any(in_bin):
            means[bin_id] = float(valid_scores[in_bin].mean())
    return centers, means


def alignment_gap(source_curve, target_curve):
    source_curve = np.asarray(source_curve, dtype=float)
    target_curve = np.asarray(target_curve, dtype=float)
    valid = np.isfinite(source_curve) & np.isfinite(target_curve)
    if not np.any(valid):
        return float("nan")
    return float(np.abs(source_curve[valid] - target_curve[valid]).mean())


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "class"


def render_task_plots(output_dir, task_name, class_names, bin_centers, curves):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError("matplotlib is required to render PSE alignment figures") from error

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gaps = {}
    for class_id, class_name in class_names.items():
        values = [
            np.asarray(curves[(stage, domain, class_id)], dtype=float)
            for stage in ("start", "end")
            for domain in ("source", "target")
        ]
        finite = np.concatenate([value[np.isfinite(value)] for value in values])
        if finite.size == 0:
            raise ValueError(f"class {class_name} has no finite PC1 observations")
        low, high = float(finite.min()), float(finite.max())
        padding = max((high - low) * .08, 1e-6)
        ylim = (low - padding, high + padding)
        class_gaps = {}
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
        for axis, stage, label in zip(axes, ("start", "end"), ("Stage-2 START", "Stage-2 END")):
            source_curve = curves[(stage, "source", class_id)]
            target_curve = curves[(stage, "target", class_id)]
            gap = alignment_gap(source_curve, target_curve)
            class_gaps[stage] = gap
            axis.plot(bin_centers, source_curve, label="Source", linewidth=2.2)
            axis.plot(bin_centers, target_curve, label="Target", linewidth=2.2)
            axis.set_title(f"{label} | mean PC1 gap={gap:.3f}")
            axis.set_xlim(0., 365.)
            axis.set_ylim(*ylim)
            axis.set_xlabel("Day of year")
            axis.grid(alpha=.25)
        axes[0].set_ylabel("PSE PC1")
        axes[1].legend(loc="best")
        figure.suptitle(f"{task_name} | {class_name}")
        figure.tight_layout()
        figure.savefig(output_dir / f"{_safe_filename(class_name)}.png", dpi=160)
        plt.close(figure)
        gaps[class_id] = class_gaps

    class_ids = list(class_names)
    x = np.arange(len(class_ids))
    width = .38
    figure, axis = plt.subplots(figsize=(max(8., len(class_ids) * .8), 4.8))
    axis.bar(x - width / 2., [gaps[c]["start"] for c in class_ids], width, label="START")
    axis.bar(x + width / 2., [gaps[c]["end"] for c in class_ids], width, label="END")
    axis.set_xticks(x, [class_names[c] for c in class_ids], rotation=35, ha="right")
    axis.set_ylabel("Mean absolute source-target PC1 gap")
    axis.set_title(f"{task_name} | PSE alignment overview")
    axis.grid(axis="y", alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "overview.png", dpi=160)
    plt.close(figure)
    return gaps


def _config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def validate_end_checkpoint_packet(path, packet):
    path = Path(path)
    if path.name != "checkpoint_last.pt":
        raise ValueError("true Stage-2 final student requires fold_0/checkpoint_last.pt")
    config = packet.get("config", {})
    if _config_value(config, "output_student") is not True:
        raise ValueError("true Stage-2 final checkpoint requires output_student=true")
    epochs = _config_value(config, "epochs")
    epoch = packet.get("epoch")
    if epochs is None or epoch != int(epochs) - 1:
        raise ValueError(
            f"true Stage-2 final checkpoint unavailable: epoch={epoch}, epochs={epochs}"
        )


def _load_model(path, device, require_final=False):
    import torch
    from train import create_model

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    packet = torch.load(path, map_location=device, weights_only=False)
    if require_final:
        validate_end_checkpoint_packet(path, packet)
    config = packet.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint config missing or invalid: {path}")
    model_config = SimpleNamespace(**config)
    if model_config.model != "psestructureprotoltae":
        raise ValueError(f"expected psestructureprotoltae checkpoint, got {model_config.model}")
    model = create_model(model_config)
    model.load_state_dict(packet["state_dict"])
    model.to(device).eval()
    return model, model_config


def _validate_checkpoint_roles(args, start_config, end_config):
    if start_config.source != args.source or start_config.target != args.source:
        raise ValueError("START checkpoint must be source-only for the requested source domain")
    if end_config.source != args.source or end_config.target != args.target:
        raise ValueError("END checkpoint source/target does not match the requested transfer")
    if int(start_config.seed) != args.seed or int(end_config.seed) != args.seed:
        raise ValueError("checkpoint seed does not match --seed")
    if list(start_config.classes) != list(end_config.classes):
        raise ValueError("START and END checkpoints use different class protocols")


def _build_train_datasets(args, classes, combine_spring_and_winter):
    from torchvision.transforms import transforms
    from dataset import PixelSetData
    from train import create_train_val_test_folds
    from transforms import Normalize, RandomSamplePixels, ToTensor

    eligible = {}
    for name in (args.source, args.target):
        dataset = PixelSetData(
            args.data_root, name, classes, transform=None, closed_set=True,
            combine_spring_and_winter=combine_spring_and_winter,
        )
        eligible[name] = dataset.get_parcel_indices().tolist()
    random.seed(args.seed)
    splits = create_train_val_test_folds(
        [args.source, args.target], 1, eligible, val_ratio=.1, test_ratio=.2,
    )[0]
    transform = transforms.Compose([RandomSamplePixels(64), Normalize(), ToTensor()])
    train_datasets = {}
    for name in (args.source, args.target):
        train_datasets[name] = PixelSetData(
            args.data_root, name, classes, transform=transform,
            indices=splits[name]["train"], closed_set=True,
            combine_spring_and_winter=combine_spring_and_winter,
        )
    return train_datasets


def _selected_loader(dataset, selected_indices):
    import torch
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, selected_indices.tolist()),
        batch_size=32, shuffle=False, num_workers=0,
    )


def _extract_pse_pair(start_model, end_model, loader, device):
    import torch

    start_features, end_features, positions, labels = [], [], [], []
    with torch.no_grad():
        for sample in loader:
            pixels = sample["pixels"].to(device)
            mask = sample["valid_pixels"].to(device)
            extra = sample["extra"].to(device)
            start = start_model.spatial_encoder(pixels, mask, extra)
            end = end_model.spatial_encoder(pixels, mask, extra)
            start_features.append(start.detach().cpu().numpy())
            end_features.append(end.detach().cpu().numpy())
            positions.append(sample["positions"].numpy())
            labels.append(sample["label"].numpy())
    return {
        "start": np.concatenate(start_features),
        "end": np.concatenate(end_features),
        "positions": np.concatenate(positions),
        "labels": np.concatenate(labels),
    }


def _class_curves(projected, source_data, target_data, class_ids, num_bins):
    curves = {}
    centers = None
    for stage in ("start", "end"):
        for domain, data in (("source", source_data), ("target", target_data)):
            pc1 = projected[f"{stage}_{domain}"][..., 0]
            for class_id in class_ids:
                selected = data["labels"] == class_id
                current_centers, curve = temporal_bin_means(
                    data["positions"][selected], pc1[selected], num_bins=num_bins,
                )
                centers = current_centers if centers is None else centers
                curves[(stage, domain, class_id)] = curve
    return centers, curves


def _domain_alias(domain):
    aliases = {
        "france/30TXT/2017": "FR1",
        "france/31TCJ/2017": "FR2",
        "denmark/32VNH/2017": "DK1",
    }
    return aliases.get(domain, domain.replace("/", "_"))


def run(args):
    import torch

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    device = torch.device(args.device)
    start_model, start_config = _load_model(args.start_checkpoint, device)
    end_model, end_config = _load_model(args.end_checkpoint, device, require_final=True)
    _validate_checkpoint_roles(args, start_config, end_config)

    datasets = _build_train_datasets(
        args, start_config.classes,
        bool(getattr(start_config, "combine_spring_and_winter", False)),
    )
    source_labels = datasets[args.source].get_labels()
    target_labels = datasets[args.target].get_labels()
    common_class_ids = sorted(set(source_labels.tolist()) & set(target_labels.tolist()))
    if not common_class_ids:
        raise RuntimeError("source and target train splits have no common classes")
    source_indices = fixed_class_sample_indices(
        source_labels, common_class_ids, args.samples_per_class, args.seed,
    )
    target_indices = fixed_class_sample_indices(
        target_labels, common_class_ids, args.samples_per_class, args.seed,
    )
    random.seed(args.seed)
    source_data = _extract_pse_pair(
        start_model, end_model,
        _selected_loader(datasets[args.source], source_indices), device,
    )
    random.seed(args.seed)
    target_data = _extract_pse_pair(
        start_model, end_model,
        _selected_loader(datasets[args.target], target_indices), device,
    )

    feature_groups = {
        "start_source": source_data["start"],
        "start_target": target_data["start"],
        "end_source": source_data["end"],
        "end_target": target_data["end"],
    }
    pca, projected = project_feature_groups(feature_groups, seed=args.seed)
    evr = pca.explained_variance_ratio_
    print(f"PCA_PC1_EVR={float(evr[0]):.8f}")
    print(f"PCA_PC2_EVR={float(evr[1]):.8f}")
    print(f"PCA_TOTAL_EVR={float(evr.sum()):.8f}")

    centers, curves = _class_curves(
        projected, source_data, target_data, common_class_ids, args.num_time_bins,
    )
    class_names = {class_id: start_config.classes[class_id] for class_id in common_class_ids}
    task_name = f"{_domain_alias(args.source)}_{_domain_alias(args.target)}"
    gaps = render_task_plots(args.output_dir, task_name, class_names, centers, curves)
    print(
        f"PSE_ALIGNMENT_TASK|task={task_name}|common_classes={len(common_class_ids)}|"
        f"source_samples={len(source_indices)}|target_samples={len(target_indices)}"
    )
    for class_id in common_class_ids:
        print(
            f"PSE_ALIGNMENT_SAMPLES|task={task_name}|class={class_names[class_id]}|"
            f"source={int((source_data['labels'] == class_id).sum())}|"
            f"target={int((target_data['labels'] == class_id).sum())}"
        )
        start_gap = gaps[class_id]["start"]
        end_gap = gaps[class_id]["end"]
        print(
            f"PSE_ALIGNMENT_GAP|task={task_name}|class={class_names[class_id]}|"
            f"start={start_gap:.8f}|end={end_gap:.8f}|delta={end_gap - start_gap:.8f}"
        )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--end-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=64)
    parser.add_argument("--num-time-bins", type=int, default=24)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
