#!/usr/bin/env python3
"""Read-only multi-head attention visualization for original TimeMatch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = {
    "AT1_DK1": ("AT1", "austria/33UVP/2017", "DK1", "denmark/32VNH/2017"),
    "FR1_FR2": ("FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
    "FR2_DK1": ("FR2", "france/31TCJ/2017", "DK1", "denmark/32VNH/2017"),
    "DK1_AT1": ("DK1", "denmark/32VNH/2017", "AT1", "austria/33UVP/2017"),
}
THRESHOLDS = (
    ("top_fraction", .10, "top_10pct"),
    ("top_fraction", .20, "top_20pct"),
    ("top_fraction", .30, "top_30pct"),
    ("absolute", .05, "alpha_gt_0p05"),
    ("absolute", .10, "alpha_gt_0p10"),
    ("absolute", .20, "alpha_gt_0p20"),
)
FOURIER = {
    "num_modes": 13,
    "reg": 1e-3,
    "period_days": 365.,
    "solver": "dense_direct",
    "grid_points": 64,
}


def _config_value(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _scalar_shift(value):
    try:
        import torch
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("global_temporal_shift must be scalar")
            value = value.detach().cpu().reshape(-1)[0].item()
    except ModuleNotFoundError:
        pass
    value = float(value)
    if not value.is_integer():
        raise ValueError(f"global_temporal_shift must be an integer day, got {value}")
    return int(value)


def validate_original_final_checkpoint(path, packet, source, target, seed):
    """Validate role and protocol without accepting a validation-best substitute."""
    if Path(path).name != "checkpoint_last.pt":
        raise ValueError("original TimeMatch visualization requires checkpoint_last.pt")
    config = packet.get("config", {})
    if _config_value(config, "model") != "pseltae":
        raise ValueError("checkpoint is not an original TimeMatch PseLTAE model")
    if (
        _config_value(config, "source") != source
        or _config_value(config, "target") != target
        or int(_config_value(config, "seed", -1)) != int(seed)
    ):
        raise ValueError("checkpoint task or seed does not match the requested experiment")
    if _config_value(config, "output_student") is not True:
        raise ValueError("checkpoint does not contain the configured final Student")
    expected_epoch = int(_config_value(config, "epochs", -1)) - 1
    if int(packet.get("epoch", -1)) != expected_epoch:
        raise ValueError("checkpoint is not the final epoch Student")
    if "state_dict" not in packet or "teacher_state_dict" not in packet:
        raise ValueError("final checkpoint is missing Student or Teacher state")
    if "global_temporal_shift" not in packet:
        raise ValueError("final checkpoint is missing global_temporal_shift")


def _checkpoint_candidates(
    checkpoint_root, source_alias, target_alias, seed,
):
    """Find Final checkpoints without assuming one historical experiment name."""
    root = Path(checkpoint_root)
    if root.is_file():
        return [root]
    task = f"{source_alias}_{target_alias}"
    relative_paths = (
        f"{task}_seed{seed}/fold_0/checkpoint_last.pt",
        f"{task}/fold_0/checkpoint_last.pt",
        f"timematch_{source_alias}_to_{target_alias}_seed{seed}/fold_0/checkpoint_last.pt",
        f"timematch_{source_alias}_to_{target_alias}/fold_0/checkpoint_last.pt",
        f"uda/{task}_seed{seed}/fold_0/checkpoint_last.pt",
        f"timematch_original_4tasks_seed{seed}/uda/{task}_seed{seed}/fold_0/checkpoint_last.pt",
    )
    candidates = [root / relative for relative in relative_paths]
    if root.is_dir():
        source_token = source_alias.lower()
        target_token = target_alias.lower()
        candidates.extend(
            path for path in root.rglob("checkpoint_last.pt")
            if source_token in str(path).lower() and target_token in str(path).lower()
        )
    unique = []
    seen = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen and path.is_file():
            seen.add(key)
            unique.append(path)
    return unique


def _load_checkpoint_packet(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_original_final_checkpoint(
    *, explicit_checkpoint, checkpoint_root, source_alias, target_alias,
    source, target, seed, packet_loader=None,
):
    """Resolve exactly one metadata-compatible final Student checkpoint."""
    loader = packet_loader or _load_checkpoint_packet
    if explicit_checkpoint:
        candidates = [Path(explicit_checkpoint)]
        if not candidates[0].is_file():
            raise FileNotFoundError(
                f"missing explicitly requested Final Student checkpoint: {candidates[0]}"
            )
    else:
        candidates = _checkpoint_candidates(
            checkpoint_root, source_alias, target_alias, seed,
        )
    matches = []
    rejected = []
    for path in candidates:
        try:
            packet = loader(path)
            validate_original_final_checkpoint(path, packet, source, target, seed)
        except Exception as error:
            rejected.append((path, str(error)))
            continue
        matches.append(path)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        rendered = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            "multiple compatible Final Student checkpoints found; pass --checkpoint "
            f"explicitly:\n{rendered}"
        )
    searched = Path(checkpoint_root)
    rejected_text = ""
    if rejected:
        rejected_text = "\nRejected checkpoint_last.pt files:\n" + "\n".join(
            f"  - {path}: {reason}" for path, reason in rejected[:20]
        )
    raise FileNotFoundError(
        "no metadata-compatible original TimeMatch Final Student checkpoint found "
        f"for {source_alias}_{target_alias} seed={seed} under {searched}. "
        "Pass --checkpoint or CHECKPOINT_<TASK> with the exact checkpoint_last.pt path."
        f"{rejected_text}"
    )


def resolve_student_domain_shifts(packet, config):
    """Reproduce the Student branches used during TimeMatch optimization."""
    target_to_source = _scalar_shift(packet.get("global_temporal_shift", 0))
    source_to_target = (
        -target_to_source if bool(_config_value(config, "shift_source", True)) else 0
    )
    return {
        "source": source_to_target,
        "target": 0,
        "target_teacher": target_to_source,
    }


def validate_attention_tensor(attention, heads=16, atol=1e-5):
    values = np.asarray(attention, dtype=np.float64)
    if values.ndim == 4 and values.shape[2] == 1:
        values = values[:, :, 0, :]
    if values.ndim != 3 or values.shape[1] != heads:
        raise ValueError(f"attention must be [B,{heads},1,T] or [B,{heads},T]")
    if not np.isfinite(values).all() or (values < -atol).any():
        raise ValueError("attention contains invalid values")
    if not np.allclose(values.sum(-1), 1., atol=atol, rtol=0.):
        raise ValueError("attention heads must sum to one")
    return values


def select_attention_indices(weights, valid_indices, mode, value):
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.asarray(valid_indices, dtype=np.int64)
    if weights.ndim != 1 or valid.ndim != 1:
        raise ValueError("weights and valid_indices must be one-dimensional")
    if valid.size == 0:
        return valid.copy()
    if valid.min() < 0 or valid.max() >= weights.size:
        raise ValueError("valid observation index is out of range")
    if mode == "top_fraction":
        if not 0 < float(value) <= 1:
            raise ValueError("top fraction must be in (0,1]")
        count = int(math.ceil(valid.size * float(value)))
        order = np.lexsort((valid, -weights[valid]))
        return np.sort(valid[order[:count]])
    if mode == "absolute":
        return valid[weights[valid] > float(value)]
    raise ValueError(f"unknown attention selection mode: {mode}")


def fit_source_pc1(features):
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("Source Train PCA needs at least two feature rows")
    mean = matrix.mean(0)
    centered = matrix - mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    component = right[0]
    pivot = int(np.argmax(np.abs(component)))
    if component[pivot] < 0:
        component = -component
    variance = np.square(singular)
    explained = float(variance[0] / max(variance.sum(), np.finfo(np.float64).eps))
    return mean, component, explained


def project_pc1(features, mean, component):
    return np.tensordot(
        np.asarray(features) - np.asarray(mean), np.asarray(component), axes=([-1], [0]),
    )


def build_sample_pairs(source_by_label, target_by_label, labels, pairs_per_class, seed):
    """Build the three requested comparison groups from deterministic ID pools."""
    if pairs_per_class < 1:
        raise ValueError("pairs_per_class must be positive")
    rng = np.random.default_rng(seed)
    labels = list(labels)
    shuffled_source, shuffled_target = {}, {}
    for label in labels:
        shuffled_source[label] = list(source_by_label[label])
        shuffled_target[label] = list(target_by_label[label])
        rng.shuffle(shuffled_source[label]); rng.shuffle(shuffled_target[label])
        if len(shuffled_source[label]) < 2 * pairs_per_class:
            raise ValueError(f"class {label} lacks distinct Source samples")
        if len(shuffled_target[label]) < pairs_per_class:
            raise ValueError(f"class {label} lacks Target samples")
    rows = []
    for label in labels:
        for pair_index in range(pairs_per_class):
            rows.append({
                "group": "same_domain_same_class", "pair_index": pair_index,
                "first_id": shuffled_source[label][2 * pair_index],
                "second_id": shuffled_source[label][2 * pair_index + 1],
                "first_label": label, "second_label": label,
            })
            rows.append({
                "group": "cross_domain_same_class", "pair_index": pair_index,
                "first_id": shuffled_source[label][pair_index],
                "second_id": shuffled_target[label][pair_index],
                "first_label": label, "second_label": label,
            })
    if len(labels) >= 2:
        for label_index, label in enumerate(labels):
            other = labels[(label_index + 1) % len(labels)]
            for pair_index in range(pairs_per_class):
                rows.append({
                    "group": "same_domain_different_class", "pair_index": pair_index,
                    "first_id": shuffled_source[label][pair_index],
                    "second_id": shuffled_source[other][pair_index],
                    "first_label": label, "second_label": other,
                })
    return rows


def marker_rows(sample, head, mode, value):
    indices = select_attention_indices(
        sample["attention"][head], np.arange(len(sample["positions"])), mode, value,
    )
    output = []
    for curve_type, key in (
        ("pse_pc1", "pse_pc1"), ("fourier", "fourier_observed_pc1"),
    ):
        for index in indices:
            output.append({
                "sample_id": sample["sample_id"], "head": int(head),
                "selection_mode": mode, "selection_value": float(value),
                "curve_type": curve_type, "observation_index": int(index),
                "date": float(sample["positions"][index]),
                "original_date": str(
                    sample.get("original_dates", sample["positions"])[index]
                ),
                "attention_weight": float(sample["attention"][head, index]),
                "curve_value": float(sample[key][index]),
            })
    return output


def selection_summary_row(
    sample_id, weights, positions, head, mode, value, threshold_name,
    original_dates=None,
):
    positions = np.asarray(positions)
    original_dates = positions if original_dates is None else np.asarray(original_dates)
    selected = select_attention_indices(
        weights, np.arange(len(positions)), mode, value,
    )
    return {
        "sample_id": sample_id, "head": int(head),
        "selection_mode": mode, "selection_value": float(value),
        "threshold_name": threshold_name, "selected_count": int(len(selected)),
        "selected_indices": "|".join(map(str, selected.tolist())),
        "selected_dates": "|".join(f"{float(positions[i]):.10g}" for i in selected),
        "selected_original_dates": "|".join(str(original_dates[i]) for i in selected),
        "selected_weights": "|".join(
            f"{float(np.asarray(weights)[i]):.10g}" for i in selected
        ),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def _label_pools(dataset):
    pools = {}
    for index, label in enumerate(dataset.get_labels().tolist()):
        pools.setdefault(int(label), []).append(index)
    return pools


def _select_labels(source, target, class_names, requested, limit, pairs_per_class, seed):
    source_pool, target_pool = _label_pools(source), _label_pools(target)
    eligible = [
        label for label in sorted(set(source_pool) & set(target_pool))
        if len(source_pool[label]) >= 2 * pairs_per_class
        and len(target_pool[label]) >= pairs_per_class
    ]
    if requested:
        missing = [name for name in requested if name not in class_names]
        if missing:
            raise ValueError(f"unknown requested classes: {missing}")
        selected = [class_names.index(name) for name in requested]
        unavailable = [class_names[label] for label in selected if label not in eligible]
        if unavailable:
            raise ValueError(f"requested classes lack sample support: {unavailable}")
        return selected, source_pool, target_pool
    if not eligible:
        raise ValueError("no common Source/Target class has enough samples")
    rng = np.random.default_rng(seed)
    count = min(int(limit), len(eligible))
    selected = sorted(rng.choice(eligible, size=count, replace=False).tolist())
    return selected, source_pool, target_pool


def _fit_pca(model, dataset, device, batch_size, max_parcels, max_observations, seed):
    """Extract Source-only PSE rows without depending on encoder internals."""
    import torch
    from scripts.analyze_structure_vs_timematch import _loader, _reset_rng
    rng = np.random.default_rng(seed)
    count = min(len(dataset), int(max_parcels))
    indices = np.sort(rng.choice(len(dataset), count, replace=False)).astype(np.int64)
    loader = _loader(dataset, batch_size, indices)
    rows = []
    _reset_rng(seed)
    with torch.no_grad():
        for sample in loader:
            pixels = sample["pixels"].to(device)
            mask = sample["valid_pixels"].to(device)
            extra = sample.get("extra")
            if extra is not None:
                extra = extra.to(device)
            spatial = model.spatial_encoder(pixels, mask, extra)
            rows.append(spatial.detach().cpu().numpy().reshape(-1, spatial.shape[-1]))
    if not rows:
        raise RuntimeError("Source Train PCA loader produced no PSE features")
    matrix = np.concatenate(rows, axis=0)
    if len(matrix) > max_observations:
        matrix = matrix[rng.choice(len(matrix), int(max_observations), replace=False)]
    mean, component, explained = fit_source_pc1(matrix)
    return mean, component, explained, indices, len(matrix)


def _batched_sample(sample, device):
    import torch
    output = {}
    for key in ("pixels", "valid_pixels", "positions", "extra"):
        value = sample.get(key)
        if value is not None:
            output[key] = value.unsqueeze(0).to(device)
        else:
            output[key] = None
    return output


def _extract_sample(model, dataset, index, domain, shift, mean, component, device):
    import torch
    from models.fourier_reconstruction import (
        BatchedDirectFourierAnalyzer, BatchedDirectFourierSynthesizer,
    )
    sample = dataset[int(index)]
    batch = _batched_sample(sample, device)
    model.eval()
    with torch.no_grad():
        spatial = model.spatial_encoder(
            batch["pixels"], batch["valid_pixels"], batch["extra"],
        )
        _, attention = model.temporal_encoder(
            spatial, batch["positions"] + int(shift), return_att=True,
        )
        analyzer = BatchedDirectFourierAnalyzer(
            FOURIER["num_modes"], FOURIER["period_days"], FOURIER["reg"],
        ).to(device)
        synthesizer = BatchedDirectFourierSynthesizer(
            FOURIER["num_modes"], FOURIER["period_days"],
        ).to(device)
        coefficients, diagnostics = analyzer(
            spatial, batch["positions"], collect_diagnostics=True,
        )
        if not bool(diagnostics["per_sample_solver_converged"].all()):
            raise RuntimeError("Fourier solve did not converge for selected sample")
        grid = torch.arange(
            FOURIER["grid_points"], device=device, dtype=batch["positions"].dtype,
        ) * (FOURIER["period_days"] / FOURIER["grid_points"])
        grid = grid.unsqueeze(0)
        reconstructed_grid = synthesizer(coefficients, grid)
        reconstructed_observed = synthesizer(coefficients, batch["positions"])
    attention_np = validate_attention_tensor(attention.detach().cpu().numpy())[0]
    spatial_np = spatial[0].detach().cpu().numpy()
    positions = batch["positions"][0].detach().cpu().numpy().astype(np.float64)
    parcel = int(sample["parcel_index"])
    label = int(sample["label"])
    sample_id = f"{domain}:{parcel}"
    return {
        "sample_id": sample_id, "domain": domain, "dataset_index": int(index),
        "parcel_id": parcel, "label": label,
        "positions": positions,
        "original_dates": np.asarray(dataset.dates).astype(str),
        "valid_indices": np.arange(len(positions), dtype=np.int64),
        "ltae_positions": positions + int(shift), "applied_shift": int(shift),
        "attention": attention_np,
        "pse_pc1": project_pc1(spatial_np, mean, component),
        "fourier_grid": grid[0].detach().cpu().numpy(),
        "fourier_grid_pc1": project_pc1(
            reconstructed_grid[0].detach().cpu().numpy(), mean, component,
        ),
        "fourier_observed_pc1": project_pc1(
            reconstructed_observed[0].detach().cpu().numpy(), mean, component,
        ),
    }


def _save_sample(path, sample, class_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_id=sample["sample_id"], domain=sample["domain"],
        dataset_index=sample["dataset_index"], parcel_id=sample["parcel_id"],
        label=sample["label"], class_name=class_name,
        positions=sample["positions"], original_dates=sample["original_dates"],
        valid_indices=sample["valid_indices"], ltae_positions=sample["ltae_positions"],
        applied_shift=sample["applied_shift"], attention=sample["attention"],
        pse_pc1=sample["pse_pc1"], fourier_grid=sample["fourier_grid"],
        fourier_grid_pc1=sample["fourier_grid_pc1"],
        fourier_observed_pc1=sample["fourier_observed_pc1"],
    )


def _plot_heatmap(sample, class_name, path, task):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(12, 5))
    image = axis.imshow(sample["attention"], aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_xlabel("Observation index (raw dates shown sparsely)")
    axis.set_ylabel("LTAE head")
    ticks = np.linspace(0, len(sample["positions"]) - 1, min(8, len(sample["positions"]))).round().astype(int)
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{sample['positions'][i]:.0f}" for i in ticks])
    axis.set_title(
        f"{task} | {sample['domain']} | {class_name} | parcel {sample['parcel_id']} | "
        f"shift={sample['applied_shift']}"
    )
    figure.colorbar(image, ax=axis, label="attention")
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160); plt.close(figure)


def _threshold_title(mode, value):
    return f"Top {100 * value:.0f}%" if mode == "top_fraction" else f"alpha > {value:.2f}"


def _plot_pair_page(first, second, class_names, task, group, mode, value, curve_type):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 4, figsize=(15, 12), sharex=True, sharey=True)
    colors = ("tab:blue", "tab:orange")
    for head, axis in enumerate(axes.flat):
        counts = []
        for sample, color in zip((first, second), colors):
            if curve_type == "pse_pc1":
                x, y = sample["positions"], sample["pse_pc1"]
                marker_y = sample["pse_pc1"]
            else:
                x, y = sample["fourier_grid"], sample["fourier_grid_pc1"]
                marker_y = sample["fourier_observed_pc1"]
            label = (
                f"{sample['domain']} {class_names[sample['label']]} "
                f"parcel={sample['parcel_id']}"
            )
            axis.plot(x, y, color=color, linewidth=1.25, label=label)
            selected = select_attention_indices(
                sample["attention"][head], sample["valid_indices"], mode, value,
            )
            counts.append(len(selected))
            if len(selected):
                axis.scatter(
                    sample["positions"][selected], marker_y[selected],
                    color=color, edgecolor="black", linewidth=.3, s=24, zorder=3,
                )
        axis.set_title(f"Head {head + 1} | n={counts[0]}/{counts[1]}", fontsize=9)
        axis.grid(alpha=.18)
    axes[0, 0].legend(fontsize=6)
    figure.supxlabel("Original observation day")
    figure.supylabel("Shared Source-Train PSE PC1")
    figure.suptitle(
        f"{task} | {group} | {curve_type} | {_threshold_title(mode, value)} | "
        f"shift={first['applied_shift']}/{second['applied_shift']}"
    )
    figure.tight_layout(rect=(.02, .02, 1, .96))
    return figure


def _render_pair(first, second, class_names, task, group, pair_id, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pair_dir = output_dir / group / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    for curve_type in ("pse_pc1", "fourier"):
        with PdfPages(pair_dir / f"{curve_type}_thresholds.pdf") as pdf:
            for mode, value, _ in THRESHOLDS:
                figure = _plot_pair_page(
                    first, second, class_names, task, group, mode, value, curve_type,
                )
                pdf.savefig(figure); plt.close(figure)
        preview = _plot_pair_page(
            first, second, class_names, task, group, "top_fraction", .2, curve_type,
        )
        preview.savefig(pair_dir / f"{curve_type}_top20_preview.png", dpi=150)
        plt.close(preview)


def _sample_record(sample, class_name, npz_path):
    return {
        "sample_id": sample["sample_id"], "domain": sample["domain"],
        "dataset_index": sample["dataset_index"], "parcel_id": sample["parcel_id"],
        "class_id": sample["label"], "class_name": class_name,
        "observation_count": len(sample["positions"]),
        "applied_student_shift": sample["applied_shift"],
        "npz_path": str(npz_path),
    }


def _curve_rows(sample):
    rows = []
    for index, (date, original_date, pse, reconstructed) in enumerate(zip(
        sample["positions"], sample["original_dates"], sample["pse_pc1"],
        sample["fourier_observed_pc1"],
    )):
        rows.append({
            "sample_id": sample["sample_id"], "curve_coordinate": "observation",
            "observation_index": index, "date": float(date),
            "original_date": str(original_date), "pse_pc1": float(pse),
            "fourier_pc1": float(reconstructed),
        })
    rows.extend({
        "sample_id": sample["sample_id"], "curve_coordinate": "fourier_grid",
        "observation_index": "", "date": float(date), "original_date": "",
        "pse_pc1": "", "fourier_pc1": float(value),
    } for date, value in zip(sample["fourier_grid"], sample["fourier_grid_pc1"]))
    return rows


def run_task(args):
    import torch
    from scripts.analyze_structure_vs_timematch import _datasets, _load_model
    if args.task not in TASKS:
        raise ValueError(f"unknown task: {args.task}")
    source_alias, source, target_alias, target = TASKS[args.task]
    checkpoint = resolve_original_final_checkpoint(
        explicit_checkpoint=args.checkpoint,
        checkpoint_root=args.checkpoint_root,
        source_alias=source_alias,
        target_alias=target_alias,
        source=source,
        target=target,
        seed=args.seed,
    )
    print(f"ATTENTION_VIS_CHECKPOINT|task={args.task}|path={checkpoint}")
    device = torch.device(args.device)
    model, config, packet = _load_model(checkpoint, device, final=True, teacher=False)
    validate_original_final_checkpoint(checkpoint, packet, source, target, args.seed)
    if model.temporal_encoder.attention_heads.n_head != 16:
        raise ValueError("original TimeMatch checkpoint does not have 16 LTAE heads")
    datasets = _datasets(config, source, target, args.data_root, args.seed)
    source_train = datasets[(source, "train")]
    source_test = datasets[(source, "test")]
    target_test = datasets[(target, "test")]
    shifts = resolve_student_domain_shifts(packet, config)
    mean, component, explained, pca_indices, fit_rows = _fit_pca(
        model, source_train, device, args.batch_size, args.pca_max_parcels,
        args.pca_max_observations, args.seed,
    )
    selected_labels, source_pool, target_pool = _select_labels(
        source_test, target_test, list(config.classes), args.classes,
        args.num_classes, args.pairs_per_class, args.seed,
    )
    pairs = build_sample_pairs(
        source_pool, target_pool, selected_labels, args.pairs_per_class, args.seed,
    )
    task_root = Path(args.output_root)
    data_root = task_root / "data" / args.task
    figure_root = task_root / "figures" / args.task
    sample_dir = data_root / "samples"
    sample_cache = {}
    sample_rows, pair_rows = [], []
    selection_rows, selection_summary_rows, curve_rows = [], [], []

    def get_sample(domain_name, dataset, index, shift):
        cache_key = (domain_name, int(index))
        if cache_key in sample_cache:
            return sample_cache[cache_key]
        value = _extract_sample(
            model, dataset, index, domain_name, shift, mean, component, device,
        )
        class_name = config.classes[value["label"]]
        filename = _safe_name(value["sample_id"]) + ".npz"
        path = sample_dir / filename
        _save_sample(path, value, class_name)
        _plot_heatmap(
            value, class_name,
            figure_root / "attention_heatmaps" / (Path(filename).stem + ".png"),
            args.task,
        )
        sample_rows.append(_sample_record(value, class_name, path))
        curve_rows.extend(_curve_rows(value))
        for head in range(16):
            for mode, threshold, threshold_name in THRESHOLDS:
                selection_summary_rows.append(selection_summary_row(
                    value["sample_id"], value["attention"][head],
                    value["positions"], head, mode, threshold, threshold_name,
                    value["original_dates"],
                ))
                rows = marker_rows(value, head, mode, threshold)
                for row in rows:
                    row["threshold_name"] = threshold_name
                selection_rows.extend(rows)
        sample_cache[cache_key] = value
        return value

    for pair_number, pair in enumerate(pairs):
        first_domain = source
        second_domain = target if pair["group"] == "cross_domain_same_class" else source
        first_dataset = source_test
        second_dataset = target_test if second_domain == target else source_test
        first = get_sample(first_domain, first_dataset, pair["first_id"], shifts["source"])
        second_shift = shifts["target"] if second_domain == target else shifts["source"]
        second = get_sample(second_domain, second_dataset, pair["second_id"], second_shift)
        pair_id = (
            f"pair_{pair_number:03d}_"
            f"{_safe_name(first['sample_id'])}_vs_{_safe_name(second['sample_id'])}"
        )
        _render_pair(
            first, second, list(config.classes), args.task, pair["group"], pair_id,
            figure_root,
        )
        pair_rows.append({
            "task": args.task, "pair_id": pair_id, "group": pair["group"],
            "pair_index": pair["pair_index"],
            "first_sample_id": first["sample_id"],
            "second_sample_id": second["sample_id"],
            "first_class_id": first["label"],
            "second_class_id": second["label"],
            "first_class_name": config.classes[first["label"]],
            "second_class_name": config.classes[second["label"]],
        })

    data_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_root / "source_train_pca.npz", mean=mean, component=component,
        explained_variance_ratio=explained, source_train_dataset_indices=pca_indices,
        fitted_observation_count=fit_rows,
    )
    _write_csv(data_root / "sample_records.csv", sample_rows)
    _write_csv(data_root / "sample_pairs.csv", pair_rows)
    _write_csv(data_root / "threshold_selections.csv", selection_rows)
    _write_csv(
        data_root / "threshold_selection_summary.csv", selection_summary_rows,
    )
    _write_csv(data_root / "curve_data.csv", curve_rows)
    task_manifest = {
        "task": args.task, "seed": args.seed, "source": source, "target": target,
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": int(packet["epoch"]), "checkpoint_role": "final_student",
        "model": _config_value(config, "model"), "strict_load": True,
        "global_temporal_shift": _scalar_shift(packet["global_temporal_shift"]),
        "student_domain_shifts": shifts,
        "classes": list(config.classes), "selected_class_ids": selected_labels,
        "selected_class_names": [config.classes[label] for label in selected_labels],
        "source_train_pca": {
            "fit_scope": "source_train_only", "parcel_count": len(pca_indices),
            "observation_count": fit_rows, "feature_dim": len(component),
            "explained_variance_ratio": explained,
        },
        "fourier": FOURIER, "attention_heads": 16,
        "thresholds": [
            {"mode": mode, "value": value, "name": name}
            for mode, value, name in THRESHOLDS
        ],
        "pairs_per_class": args.pairs_per_class,
        "pair_count": len(pair_rows), "unique_sample_count": len(sample_rows),
        "target_truth_usage": "offline_sampling_and_display_only",
        "training_or_model_mutation": False,
    }
    (data_root / "manifest.json").write_text(
        json.dumps(task_manifest, indent=2), encoding="utf-8",
    )
    print(
        f"ATTENTION_VIS_TASK_COMPLETE|task={args.task}|pairs={len(pair_rows)}|"
        f"samples={len(sample_rows)}|output={task_root}"
    )


def finalize(args):
    root = Path(args.output_root)
    manifests = []
    for task in args.tasks:
        path = root / "data" / task / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing task manifest: {path}")
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    manifest = {
        "experiment": "original_timematch_ltae_attention_visualization",
        "read_only": True, "tasks": manifests,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = [
        "# Original TimeMatch LTAE Attention Visualization", "",
        "Read-only plots from final Student checkpoints. Attention is shown per head; no head averaging is used.",
        "PSE-PC1 uses one Source-Train-fitted basis per task. Fourier curves are offline reconstructions and never enter LTAE.",
        "Target labels are used only to select and label offline display pairs.", "",
        "## Tasks", "",
    ]
    readme.extend(
        f"- {item['task']}: pairs={item['pair_count']}, samples={item['unique_sample_count']}, "
        f"checkpoint={item['checkpoint']}" for item in manifests
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"ATTENTION_VIS_FINALIZED|tasks={len(manifests)}|output={root}")


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--task", required=True, choices=tuple(TASKS))
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    run.add_argument("--checkpoint-root", default="outputs")
    run.add_argument("--checkpoint")
    run.add_argument("--output-root", default="outputs/timematch_ltae_attention_vis")
    run.add_argument("--device", default="cuda")
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--num-classes", type=int, default=3)
    run.add_argument("--classes", nargs="*")
    run.add_argument("--pairs-per-class", type=int, default=1)
    run.add_argument("--pca-max-parcels", type=int, default=512)
    run.add_argument("--pca-max-observations", type=int, default=50000)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-root", default="outputs/timematch_ltae_attention_vis")
    finalize_parser.add_argument("--tasks", nargs="+", default=tuple(TASKS))
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "run":
        run_task(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
