"""Read-only metrics and interventions for the structure-transfer audit."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager

import numpy as np
import torch
from torch.nn import functional as F


COMPONENTS = {
    "FULL": ("raw", "diff", "mean", "std"),
    "MORPH": ("raw", "diff"),
    "STATS": ("mean", "std"),
    "NO_MEAN": ("raw", "diff", "std"),
    "NO_STD": ("raw", "diff", "mean"),
    "MEAN_ONLY": ("mean",),
    "STD_ONLY": ("std",),
}


def deterministic_class_indices(labels, limit, seed):
    """Choose at most ``limit`` examples per class with an isolated RNG."""
    labels = np.asarray(labels)
    rng, selected = np.random.default_rng(int(seed)), []
    for label in np.unique(labels):
        available = np.flatnonzero(labels == label)
        if len(available) > int(limit):
            available = rng.choice(available, int(limit), replace=False)
        selected.extend(np.sort(available).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def remove_direction(values, direction, eps=1e-12):
    """Remove the component parallel to one feature-space direction."""
    direction = torch.as_tensor(direction, device=values.device, dtype=values.dtype)
    unit = direction / direction.norm().clamp_min(eps)
    return values - (values @ unit).unsqueeze(-1) * unit


def cosine_distance(left, right, eps=1e-8):
    left = F.normalize(torch.as_tensor(left).float(), dim=-1, eps=eps)
    right = F.normalize(torch.as_tensor(right).float(), dim=-1, eps=eps)
    return 1.0 - (left * right).sum(-1)


def effective_rank(values, eps=1e-12):
    values = torch.as_tensor(values).float().reshape(-1, values.shape[-1])
    singular = torch.linalg.svdvals(values)
    probability = singular / singular.sum().clamp_min(eps)
    return float(torch.exp(-(probability * probability.clamp_min(eps).log()).sum()))


def candidate_mask(branch, mode, device):
    """Return a mask over the production stride-8 candidate ordering."""
    scales, starts = [], []
    extractor = branch.window_extractor
    for scale in extractor.scales:
        current = list(range(0, extractor.grid_points, extractor.stride))
        scales.extend([int(scale)] * len(current))
        starts.extend(current)
    scales = torch.tensor(scales, device=device)
    starts = torch.tensor(starts, device=device)
    if mode == "FULL":
        return torch.ones_like(scales, dtype=torch.bool)
    if mode.startswith("ONLY_Q"):
        return scales == int(mode.removeprefix("ONLY_Q"))
    if mode.startswith("REMOVE_Q"):
        return scales != int(mode.removeprefix("REMOVE_Q"))
    if mode == "STRIDE16":
        return starts.remainder(16) == 0
    raise ValueError(f"unknown scale/window intervention: {mode}")


@contextmanager
def _scaled_projection(module, alpha):
    if float(alpha) == 1.0:
        yield
        return

    def scale(_module, _inputs, output):
        return output * float(alpha)

    handle = module.register_forward_hook(scale)
    try:
        yield
    finally:
        handle.remove()


def prepare_intervention(model, batch):
    """Compute PSE/Fourier/component encodings once for many interventions."""
    spatial = model.spatial_encoder(
        batch["pixels"], batch["valid_pixels"], batch["extra"],
    )
    branch = model.structure_branch
    positions = batch["positions"]
    exposed, grid = branch.exposer(spatial, positions)
    groups, scales = branch.window_extractor(exposed)
    component_groups = []
    for windows in groups:
        parts = branch.token_generator.components(windows)
        encoded = {
            "raw": branch.token_generator.raw_encoder(parts["normalized"]),
            "diff": branch.token_generator.diff_encoder(parts["difference"]),
            "mean": branch.token_generator.mean_encoder(parts["mean"]),
            "std": branch.token_generator.std_encoder(parts["std"]),
        }
        component_groups.append(encoded)
    return {
        "spatial": spatial, "components_by_scale": component_groups,
        "shape_scales": scales, "exposed_curve": exposed, "exposed_grid": grid,
    }


def _structure_from_prepared(model, prepared, component_mode, scale_mode):
    if component_mode not in COMPONENTS:
        raise ValueError(f"unknown component intervention: {component_mode}")
    branch, keep = model.structure_branch, set(COMPONENTS[component_mode])
    token_groups = []
    combined_components = {key: [] for key in ("raw", "diff", "mean", "std")}
    for encoded in prepared["components_by_scale"]:
        for name, value in encoded.items():
            combined_components[name].append(value)
        fused = torch.cat([
            encoded[name] if name in keep else torch.zeros_like(encoded[name])
            for name in ("raw", "diff", "mean", "std")
        ], dim=-1)
        token_groups.append(branch.token_generator.fusion(fused))
    tokens = torch.cat(token_groups, dim=1)
    mask = candidate_mask(branch, scale_mode, tokens.device)
    details = branch.shapelet_dictionary.compute_response(
        tokens, candidate_mask=mask, return_details=True,
    )
    response = details["response"]
    class_token = branch.response_to_query(response)
    return {
        "shape_tokens": tokens,
        "shapelet_response": response,
        "shape_class_token": class_token,
        "shape_scales": prepared["shape_scales"],
        "exposed_curve": prepared["exposed_curve"],
        "exposed_grid": prepared["exposed_grid"],
        "candidate_weights": details["weights"],
        "candidate_mask": mask,
        "component_features": {
            name: torch.cat(values, dim=1) for name, values in combined_components.items()
        },
    }


def forward_prepared_intervention(
    model, prepared, positions, temporal_shift=0, query_alpha=1.0,
    component_mode="FULL", scale_mode="FULL", return_attention=False,
):
    """Apply lightweight interventions to cached read-only representations."""
    spatial = prepared["spatial"]
    structure = _structure_from_prepared(model, prepared, component_mode, scale_mode)
    projection = model.temporal_encoder.attention_heads.external_query_projection
    with _scaled_projection(projection, query_alpha):
        temporal = model.temporal_encoder(
            spatial, positions + temporal_shift,
            external_query=structure["shape_class_token"],
            return_att=return_attention,
        )
    if return_attention:
        instance, attention = temporal
    else:
        instance, attention = temporal, None
    result = {
        "logits": model.decoder(instance),
        "shape_logits": model.shape_classifier(structure["shapelet_response"]),
        "instance_feature": instance,
        "pse_feature": spatial,
        **structure,
    }
    if attention is not None:
        result["attention"] = attention
    qmaster = model.temporal_encoder.attention_heads.query
    qshape = projection(structure["shape_class_token"]).view(
        len(spatial), model.temporal_encoder.attention_heads.n_head,
        model.temporal_encoder.attention_heads.d_k,
    )
    result.update({"qmaster": qmaster, "qshape": qshape})
    return result


def forward_intervention(
    model, batch, temporal_shift=0, query_alpha=1.0,
    component_mode="FULL", scale_mode="FULL", return_attention=False,
):
    """Run the formal model with diagnostic-only query/component/window changes."""
    prepared = prepare_intervention(model, batch)
    return forward_prepared_intervention(
        model, prepared, batch["positions"], temporal_shift=temporal_shift,
        query_alpha=query_alpha, component_mode=component_mode,
        scale_mode=scale_mode, return_attention=return_attention,
    )


def gradient_conflict_metrics(main_loss, shape_loss, parameter_groups):
    """Compare two losses with autograd.grad without touching parameter .grad."""
    result = {}
    for group, parameters in parameter_groups.items():
        parameters = tuple(value for value in parameters if value.requires_grad)
        main = torch.autograd.grad(
            main_loss, parameters, retain_graph=True, allow_unused=True,
        )
        shape = torch.autograd.grad(
            shape_loss, parameters, retain_graph=True, allow_unused=True,
        )
        main = torch.cat([
            torch.zeros_like(parameter).reshape(-1) if grad is None else grad.reshape(-1)
            for parameter, grad in zip(parameters, main)
        ])
        shape = torch.cat([
            torch.zeros_like(parameter).reshape(-1) if grad is None else grad.reshape(-1)
            for parameter, grad in zip(parameters, shape)
        ])
        main_norm, shape_norm = main.norm(), shape.norm()
        denominator = main_norm * shape_norm
        cosine = (main @ shape) / denominator if denominator > 0 else main.new_tensor(float("nan"))
        result[group] = {
            "gradient_cosine": float(cosine.detach().cpu()),
            "main_grad_norm": float(main_norm.detach().cpu()),
            "shape_grad_norm": float(shape_norm.detach().cpu()),
            "shape_main_norm_ratio": float((shape_norm / main_norm.clamp_min(1e-12)).detach().cpu()),
        }
    return result


def class_centroid_metrics(features, labels):
    features = F.normalize(torch.as_tensor(features).float(), dim=-1)
    labels = torch.as_tensor(labels).long()
    centroids, dispersion = {}, []
    for label in labels.unique(sorted=True):
        current = features[labels == label]
        centroid = F.normalize(current.mean(0), dim=0)
        centroids[int(label)] = centroid
        dispersion.append(cosine_distance(current, centroid).mean())
    pairs = [
        cosine_distance(centroids[left], centroids[right])
        for index, left in enumerate(centroids) for right in list(centroids)[index + 1:]
    ]
    return {
        "centroids": centroids,
        "inter_sep": float(torch.stack(pairs).mean()) if pairs else float("nan"),
        "intra_disp": float(torch.stack(dispersion).mean()) if dispersion else float("nan"),
    }


def domain_gap(source_features, source_labels, target_features, target_labels):
    source = class_centroid_metrics(source_features, source_labels)["centroids"]
    target = class_centroid_metrics(target_features, target_labels)["centroids"]
    common = sorted(set(source) & set(target))
    return float(torch.stack([cosine_distance(source[key], target[key]) for key in common]).mean()) if common else float("nan")


def pse_temporal_metrics(features, positions, labels, bins=24, period=365.0):
    """Preserve time by aggregating observations within class/DOY bins."""
    features = F.normalize(torch.as_tensor(features).float(), dim=-1)
    positions, labels = torch.as_tensor(positions), torch.as_tensor(labels).long()
    bin_ids = torch.floor((positions.float().remainder(period) / period) * bins).long().clamp_max(bins - 1)
    centroids, distances = {}, []
    for domain_class in labels.unique(sorted=True):
        sample_mask = labels == domain_class
        for bin_id in range(bins):
            mask = sample_mask[:, None] & (bin_ids == bin_id)
            if mask.any():
                values = features[mask]
                centroid = F.normalize(values.mean(0), dim=0)
                centroids[(int(domain_class), bin_id)] = centroid
                distances.append(cosine_distance(values, centroid).mean())
    inter = []
    for bin_id in range(bins):
        keys = [key for key in centroids if key[1] == bin_id]
        for index, left in enumerate(keys):
            for right in keys[index + 1:]:
                inter.append(cosine_distance(centroids[left], centroids[right]))
    return {
        "centroids": centroids,
        "inter_sep": float(torch.stack(inter).mean()) if inter else float("nan"),
        "intra_disp": float(torch.stack(distances).mean()) if distances else float("nan"),
    }


def pse_domain_gap(source, target):
    common = sorted(set(source["centroids"]) & set(target["centroids"]))
    return float(torch.stack([
        cosine_distance(source["centroids"][key], target["centroids"][key])
        for key in common
    ]).mean()) if common else float("nan")
