#!/usr/bin/env python3
"""Render offline Raw-PSE shift comparisons from local artifacts."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.shift_visualization import (
    SourceClassPC1,
    estimate_class_residual_shift,
    interpolate_latent,
    read_best_validation_shift,
    render_class_residual_figure,
    render_task_class_figures,
    replay_train_indices,
    update_class_residual_outputs,
    validate_existing_task_outputs,
    write_task_outputs,
)
from models.fourier_reconstruction import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
)
from models.stclassifier import PseLTae


DOMAINS = {
    "AT1": "austria/33UVP/2017",
    "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
}
TASKS = (("AT1", "DK1"), ("DK1", "FR1"), ("FR1", "FR2"), ("FR2", "AT1"))


def _key_values(values):
    result = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"expected KEY=PATH, got {value!r}")
        key, path = value.split("=", 1)
        result[key.upper()] = Path(path)
    return result


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_source_checkpoint(root, alias, source_path, seed, fold, overrides):
    if alias in overrides:
        candidate = overrides[alias]
        checkpoint = candidate if candidate.name == "model.pt" else candidate / f"fold_{fold}" / "model.pt"
        configs = [checkpoint.parent.parent / "train_config.json"]
    else:
        configs = sorted(Path(root).rglob("train_config.json"))
        matches = []
        for config_path in configs:
            try:
                config = _read_json(config_path)
            except (OSError, ValueError):
                continue
            if (
                config.get("model") == "pseltae"
                and config.get("source") == source_path
                and config.get("target") == source_path
                and int(config.get("seed", -1)) == seed
            ):
                checkpoint = config_path.parent / f"fold_{fold}" / "model.pt"
                if checkpoint.is_file():
                    score = 10 if config_path.parent.name == f"pseltae_{alias}_source_seed{seed}" else 0
                    matches.append((score, config_path, checkpoint))
        if not matches:
            raise FileNotFoundError(f"no Raw PseLTae source checkpoint for {alias} below {root}")
        matches.sort(key=lambda item: (-item[0], len(str(item[2])), str(item[2])))
        if len(matches) > 1 and matches[0][0] == matches[1][0]:
            paths = "\n".join(str(item[2]) for item in matches)
            raise ValueError(f"ambiguous source checkpoint for {alias}; use --source-checkpoint {alias}=PATH:\n{paths}")
        _, config_path, checkpoint = matches[0]
        configs = [config_path]
    checkpoint = Path(checkpoint)
    config_path = Path(configs[0])
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"checkpoint/config missing: {checkpoint}, {config_path}")
    config = _read_json(config_path)
    expected = {"model": "pseltae", "source": source_path, "target": source_path, "seed": seed}
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if checkpoint.parent.name != f"fold_{fold}":
        mismatch["fold"] = (checkpoint.parent.name, f"fold_{fold}")
    if mismatch:
        raise ValueError(f"source checkpoint metadata mismatch: {mismatch}")
    return checkpoint, config


def resolve_task_log(root, source_alias, target_alias, kind, overrides):
    key = f"{source_alias}_{target_alias}"
    if key in overrides:
        path = overrides[key]
        if not path.is_file():
            raise FileNotFoundError(f"configured {kind} log missing: {path}")
        return path
    candidates = []
    pair = key.lower()
    for path in Path(root).rglob("*.log"):
        name = str(path).lower().replace("to_", "").replace("->", "_")
        score = 0
        if pair in name:
            score += 30
        if source_alias.lower() in name and target_alias.lower() in name:
            score += 10
        if kind == "timematch" and "original" in name:
            score += 10
        if kind == "reconshift13" and "reconshift13" in name:
            score += 10
        if score:
            candidates.append((score, len(str(path)), str(path), path))
    if not candidates:
        raise FileNotFoundError(f"no {kind} log for {key} below {root}")
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
        paths = "\n".join(str(item[3]) for item in candidates)
        raise ValueError(f"ambiguous {kind} log for {key}; use --{kind}-log {key}=PATH:\n{paths}")
    return candidates[0][3]


def resolve_task_output(root, source_path, target_path, seed, kind):
    matches = []
    for config_path in Path(root).rglob("train_config.json"):
        try:
            config = _read_json(config_path)
        except (OSError, ValueError):
            continue
        if (
            config.get("model") != "pseltae"
            or config.get("source") != source_path
            or config.get("target") != target_path
            or int(config.get("seed", -1)) != seed
            or config.get("method") != "timematch"
        ):
            continue
        view = config.get("shift_estimation_view", "raw")
        is_recon = view == "fourier_recon"
        if (kind == "reconshift13") != is_recon:
            continue
        matches.append(config_path.parent)
    return matches[0] if len(matches) == 1 else None


def load_classes(checkpoint, source_path, data_root):
    protocol_path = checkpoint.parent.parent / "closed_set_protocol.json"
    if protocol_path.is_file():
        classes = _read_json(protocol_path).get("classes")
        if classes:
            return list(classes)
    from dataset import PixelSetData
    from utils import label_utils

    candidates = [name for name in label_utils.get_classes(source_path.split("/")[0]) if name != "unknown"]
    dataset = PixelSetData(data_root, source_path, candidates, closed_set=True)
    labels, counts = np.unique(dataset.get_labels(), return_counts=True)
    return [candidates[int(label)] for label, count in zip(labels, counts) if count >= 200]


def load_raw_spatial_encoder(checkpoint, config, classes, device):
    model = PseLTae(
        input_dim=int(config.get("input_dim", 10)),
        with_extra=bool(config.get("with_extra", False)),
        num_classes=len(classes),
    )
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    state = payload.get("state_dict", payload)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model.spatial_encoder.to(device).eval().requires_grad_(False)


def build_split_datasets(data_root, source_path, target_path, classes, seed, config):
    from dataset import PixelSetData
    from torchvision.transforms import transforms
    from transforms import Normalize, ToTensor

    base_source = PixelSetData(data_root, source_path, classes, closed_set=True)
    base_target = PixelSetData(data_root, target_path, classes, closed_set=True)
    eligible = {
        source_path: base_source.get_parcel_indices().tolist(),
        target_path: base_target.get_parcel_indices().tolist(),
    }
    splits = replay_train_indices(
        (source_path, target_path),
        eligible,
        seed,
        float(config.get("val_ratio", 0.1)),
        float(config.get("test_ratio", 0.2)),
    )
    transform = transforms.Compose([Normalize(), ToTensor()])
    common = dict(
        data_root=data_root,
        classes=classes,
        transform=transform,
        with_extra=bool(config.get("with_extra", False)),
        closed_set=True,
        combine_spring_and_winter=False,
    )
    source = PixelSetData(dataset_name=source_path, indices=splits[source_path]["train"], **common)
    target = PixelSetData(dataset_name=target_path, indices=splits[target_path]["train"], **common)
    return source, target


def _loader(dataset, batch_size):
    from dataset import GroupByShapesBatchSampler

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_sampler=GroupByShapesBatchSampler(dataset, batch_size),
    )


@torch.inference_mode()
def fit_class_projections(spatial_encoder, dataset, class_count, grid, batch_size, device, with_extra):
    statistics = {
        class_id: {"count": 0, "sum": None, "cross": None}
        for class_id in range(class_count)
    }
    latent_dim = None
    for sample in _loader(dataset, batch_size):
        pixels = sample["pixels"].to(device)
        mask = sample["valid_pixels"].to(device)
        extra = sample["extra"].to(device) if with_extra else None
        features = spatial_encoder(pixels, mask, extra).cpu().numpy()
        positions = sample["positions"].numpy()
        labels = sample["label"].numpy()
        latent_dim = features.shape[-1]
        for feature, position, label in zip(features, positions, labels):
            dense = interpolate_latent(position, feature, grid)
            item = statistics[int(label)]
            item["count"] += dense.shape[0]
            value_sum = dense.sum(axis=0)
            cross = dense.T @ dense
            item["sum"] = value_sum if item["sum"] is None else item["sum"] + value_sum
            item["cross"] = cross if item["cross"] is None else item["cross"] + cross
    projections = {}
    for class_id, item in statistics.items():
        if item["count"] == 0:
            continue
        center = item["sum"] / item["count"]
        covariance = item["cross"] - item["count"] * np.outer(center, center)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0:
            axis = -axis
        projections[class_id] = SourceClassPC1(center, axis)
    return projections, latent_dim


@torch.inference_mode()
def project_dataset(
    spatial_encoder,
    dataset,
    projections,
    grid,
    batch_size,
    device,
    with_extra,
    fourier_analyzer=None,
):
    curves = defaultdict(list)
    coefficients = defaultdict(list)
    for sample in _loader(dataset, batch_size):
        pixels = sample["pixels"].to(device)
        mask = sample["valid_pixels"].to(device)
        extra = sample["extra"].to(device) if with_extra else None
        feature_tensor = spatial_encoder(pixels, mask, extra)
        position_tensor = sample["positions"].to(
            device=device, dtype=feature_tensor.dtype
        )
        coefficient_tensor = None
        if fourier_analyzer is not None:
            coefficient_tensor, _ = fourier_analyzer(
                feature_tensor, position_tensor, collect_diagnostics=False
            )
        features = feature_tensor.cpu().numpy()
        positions = position_tensor.cpu().numpy()
        labels = sample["label"].numpy()
        for feature, position, label in zip(features, positions, labels):
            class_id = int(label)
            if class_id not in projections:
                continue
            dense = interpolate_latent(position, feature, grid)
            curves[class_id].append(projections[class_id].transform(dense[None])[0])
        if coefficient_tensor is not None:
            coefficient_values = coefficient_tensor.cpu().numpy()
            for class_id in np.unique(labels):
                class_id = int(class_id)
                if class_id in projections:
                    coefficients[class_id].append(
                        coefficient_values[labels == class_id]
                    )
    projected = {class_id: np.stack(values) for class_id, values in curves.items()}
    if fourier_analyzer is None:
        return projected
    coefficient_groups = {
        class_id: np.concatenate(values, axis=0)
        for class_id, values in coefficients.items()
    }
    return projected, coefficient_groups


@torch.inference_mode()
def reconstruct_class_prototype(
    coefficients,
    synthesizer,
    device,
    sample_batch_size=256,
    day_chunk_size=16,
):
    """Pointwise median Recon13 prototype on the non-wrapped day 0..364 grid."""
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 3 or coefficients.shape[0] == 0:
        raise ValueError("coefficients must be non-empty [N,F,D]")
    count, _, latent_dim = coefficients.shape
    prototype = np.empty((365, latent_dim), dtype=np.float32)
    for day_start in range(0, 365, day_chunk_size):
        days = np.arange(
            day_start, min(day_start + day_chunk_size, 365), dtype=np.float32
        )
        reconstructed = np.empty((count, len(days), latent_dim), dtype=np.float32)
        for sample_start in range(0, count, sample_batch_size):
            sample_stop = min(sample_start + sample_batch_size, count)
            coefficient_batch = torch.as_tensor(
                coefficients[sample_start:sample_stop], device=device
            )
            positions = torch.as_tensor(days, device=device).unsqueeze(0).expand(
                sample_stop - sample_start, -1
            )
            reconstructed[sample_start:sample_stop] = (
                synthesizer(coefficient_batch, positions).cpu().numpy()
            )
        prototype[day_start : day_start + len(days)] = np.median(
            reconstructed, axis=0
        )
    return prototype


def run_task(args, source_alias, target_alias, checkpoint_overrides, tm_log_overrides, recon_log_overrides):
    source_path, target_path = DOMAINS[source_alias], DOMAINS[target_alias]
    checkpoint, config = resolve_source_checkpoint(
        args.source_checkpoint_root, source_alias, source_path, args.seed, args.fold, checkpoint_overrides
    )
    classes = load_classes(checkpoint, source_path, args.data_root)
    tm_log = resolve_task_log(args.timematch_log_root, source_alias, target_alias, "timematch", tm_log_overrides)
    recon_log = resolve_task_log(args.reconshift_log_root, source_alias, target_alias, "reconshift13", recon_log_overrides)
    tm_output = resolve_task_output(args.timematch_output_root, source_path, target_path, args.seed, "timematch")
    recon_output = resolve_task_output(args.reconshift_output_root, source_path, target_path, args.seed, "reconshift13")
    tm_shift = read_best_validation_shift(tm_log, tm_output)
    recon_shift = read_best_validation_shift(recon_log, recon_output)

    source_dataset, target_dataset = build_split_datasets(
        args.data_root, source_path, target_path, classes, args.seed, config
    )
    device = torch.device(args.device)
    spatial_encoder = load_raw_spatial_encoder(checkpoint, config, classes, device)
    grid = np.linspace(0.0, 365.0, args.grid_size, dtype=np.float64)
    projections, latent_dim = fit_class_projections(
        spatial_encoder, source_dataset, len(classes), grid, args.batch_size, device, bool(config.get("with_extra", False))
    )
    task_name = f"{source_alias}_{target_alias}"
    output_dir = Path(args.output_root) / task_name
    existing_manifest = None
    expected_filenames = None
    if args.add_class_residual_shift:
        existing_manifest, expected_filenames = validate_existing_task_outputs(output_dir)

    fourier_analyzer = None
    fourier_synthesizer = None
    if args.add_class_residual_shift:
        fourier_analyzer = BatchedDirectFourierAnalyzer(
            num_modes=13, period_days=365.0, reg=0.001
        ).to(device)
        fourier_synthesizer = BatchedDirectFourierSynthesizer(
            num_modes=13, period_days=365.0
        ).to(device)
    source_projected = project_dataset(
        spatial_encoder,
        source_dataset,
        projections,
        grid,
        args.batch_size,
        device,
        bool(config.get("with_extra", False)),
        fourier_analyzer,
    )
    source_prototypes = None
    if args.add_class_residual_shift:
        source_curves, source_coefficients = source_projected
        source_prototypes = {
            class_id: reconstruct_class_prototype(
                values, fourier_synthesizer, device, args.batch_size
            )
            for class_id, values in source_coefficients.items()
        }
        del source_coefficients
    target_projected = project_dataset(
        spatial_encoder,
        target_dataset,
        projections,
        grid,
        args.batch_size,
        device,
        bool(config.get("with_extra", False)),
        fourier_analyzer,
    )
    if args.add_class_residual_shift:
        target_curves, target_coefficients = target_projected
        target_prototypes = {
            class_id: reconstruct_class_prototype(
                values, fourier_synthesizer, device, args.batch_size
            )
            for class_id, values in target_coefficients.items()
        }
        del target_coefficients
    else:
        source_curves, target_curves = source_projected, target_projected

    if args.add_class_residual_shift:
        summary_rows = []
        score_curves = {}
        class_metadata = {}
        for class_id, class_name in enumerate(classes):
            matching = [
                name for name in expected_filenames if name.startswith(f"{class_id:02d}_")
            ]
            if not matching:
                continue
            source_group = source_curves.get(class_id)
            target_group = target_curves.get(class_id)
            source_prototype = source_prototypes.get(class_id)
            target_prototype = target_prototypes.get(class_id)
            if any(
                group is None
                for group in (
                    source_group,
                    target_group,
                    source_prototype,
                    target_prototype,
                )
            ):
                raise RuntimeError(
                    f"missing extracted class data for {task_name} class {class_id}"
                )
            result = estimate_class_residual_shift(
                source_prototype,
                target_prototype,
                recon_shift.shift_days,
                args.class_residual_max_days,
            )
            raw_metadata = (
                existing_manifest.get("class_outputs", {})
                .get(str(class_id), {})
                .get("raw")
            )
            if not raw_metadata or "ylim" not in raw_metadata:
                raise ValueError(
                    f"missing baseline Raw ylim for {task_name} class {class_id}"
                )
            metadata = render_class_residual_figure(
                output_dir,
                task_name.replace("_", "→"),
                class_id,
                class_name,
                grid,
                source_group,
                target_group,
                recon_shift.shift_days,
                result.class_residual_shift_days,
                result.score_at_residual_0,
                result.best_score,
                result.score_gain,
                tuple(raw_metadata["ylim"]),
                args.max_spaghetti,
                args.seed,
            )
            class_metadata[str(class_id)] = metadata
            score_curves[class_id] = result.candidates
            summary_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "source_count": len(source_group),
                    "target_count": len(target_group),
                    "global_reconshift_days": result.global_shift_days,
                    "class_residual_shift_days": result.class_residual_shift_days,
                    "final_shift_days": result.final_shift_days,
                    "score_at_residual_0": result.score_at_residual_0,
                    "best_score": result.best_score,
                    "score_gain": result.score_gain,
                    "boundary_hit": result.boundary_hit,
                    "num_valid_channels": result.num_valid_channels,
                    "common_support_days": result.common_support_days,
                }
            )
        generated = {
            path.name
            for path in (output_dir / "04_reconshift13_class_shift20").glob("*.png")
        }
        if generated != expected_filenames:
            raise RuntimeError(
                "configuration 04 class files differ from 01/02/03: "
                f"expected={sorted(expected_filenames)}, generated={sorted(generated)}"
            )
        update_class_residual_outputs(
            output_dir,
            existing_manifest,
            summary_rows,
            score_curves,
            class_metadata,
            args.class_residual_max_days,
        )
        print(f"[FINISHED] {task_name} class residual extension: {output_dir}")
        return

    class_records, summary_rows, skipped = {}, [], []
    for class_id, class_name in enumerate(classes):
        source_group = source_curves.get(class_id)
        target_group = target_curves.get(class_id)
        source_count = 0 if source_group is None else len(source_group)
        target_count = 0 if target_group is None else len(target_group)
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "source_count": source_count,
            "target_count": target_count,
            "timematch_shift_days": tm_shift.shift_days,
            "reconshift_shift_days": recon_shift.shift_days,
        }
        summary_rows.append(row)
        if source_count < args.min_class_samples or target_count < args.min_class_samples:
            skipped.append({**row, "reason": "insufficient_source_or_target_samples"})
            continue
        class_records[str(class_id)] = render_task_class_figures(
            output_dir,
            task_name.replace("_", "→"),
            class_id,
            class_name,
            grid,
            source_group,
            target_group,
            tm_shift.shift_days,
            recon_shift.shift_days,
            args.max_spaghetti,
            args.seed,
        )
    manifest = {
        "audit_type": "offline_raw_pse_shift_visualization",
        "training_performed": False,
        "feature_semantics": "Raw PSE latent trajectory from one source-only PseLTae checkpoint",
        "feature_fixed_across_configs": True,
        "shift_effect": "target time coordinates only; source coordinates and all y values unchanged",
        "task": task_name,
        "source": source_path,
        "target": target_path,
        "seed": args.seed,
        "fold": args.fold,
        "source_checkpoint": str(checkpoint),
        "artifact_inputs": {
            "timematch_output": None if tm_output is None else str(tm_output),
            "timematch_log": str(tm_log),
            "reconshift13_output": None if recon_output is None else str(recon_output),
            "reconshift13_log": str(recon_log),
        },
        "timematch_shift": tm_shift.as_dict(),
        "reconshift13_shift": recon_shift.as_dict(),
        "source_train_count": len(source_dataset),
        "target_train_count": len(target_dataset),
        "classes": classes,
        "skipped_classes": skipped,
        "class_outputs": class_records,
        "pca": "one source-only PC1 per class; target is transform-only; deterministic sign fixing",
        "interpolation": {"method": "linear", "grid": "linspace(0,365,K)", "grid_size": args.grid_size},
        "latent_dim": latent_dim,
        "loader": {"shuffle": False, "num_workers": 0, "full_temporal_sequence": True, "full_pixel_set": True},
        "target_label_use": "offline class grouping and visualization only; never shift selection",
    }
    write_task_outputs(output_dir, manifest, summary_rows)
    print(f"[FINISHED] {task_name}: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--source-checkpoint-root", required=True, type=Path)
    parser.add_argument("--timematch-output-root", required=True, type=Path)
    parser.add_argument("--timematch-log-root", required=True, type=Path)
    parser.add_argument("--reconshift-output-root", required=True, type=Path)
    parser.add_argument("--reconshift-log-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-domain", choices=DOMAINS)
    parser.add_argument("--target-domain", choices=DOMAINS)
    parser.add_argument("--source-checkpoint", action="append", default=[])
    parser.add_argument("--timematch-log", action="append", default=[])
    parser.add_argument("--reconshift13-log", action="append", default=[])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-spaghetti", type=int, default=40)
    parser.add_argument("--min-class-samples", type=int, default=2)
    parser.add_argument("--add-class-residual-shift", action="store_true")
    parser.add_argument("--class-residual-max-days", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fold != 0:
        raise SystemExit("ERROR: this audit reproduces fold 0 only")
    if (args.source_domain is None) != (args.target_domain is None):
        raise SystemExit("ERROR: --source-domain and --target-domain must be provided together")
    for path, label in (
        (args.data_root, "data root"),
        (args.source_checkpoint_root, "source checkpoint root"),
        (args.timematch_output_root, "Original TimeMatch output root"),
        (args.timematch_log_root, "Original TimeMatch log root"),
        (args.reconshift_output_root, "ReconShift13 output root"),
        (args.reconshift_log_root, "ReconShift13 log root"),
    ):
        if not path.is_dir():
            raise SystemExit(f"ERROR: {label} not found: {path}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tasks = ((args.source_domain, args.target_domain),) if args.source_domain else TASKS
    checkpoint_overrides = _key_values(args.source_checkpoint)
    tm_log_overrides = _key_values(args.timematch_log)
    recon_log_overrides = _key_values(args.reconshift13_log)
    for source_alias, target_alias in tasks:
        run_task(args, source_alias, target_alias, checkpoint_overrides, tm_log_overrides, recon_log_overrides)


if __name__ == "__main__":
    main()
