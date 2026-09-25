"""Pure, read-only helpers for the final Shapelet structure audit."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.nn import functional as F


OUTPUT_KEYS = ("logits", "instance_feature", "shapelet_response", "shape_class_token")


@torch.no_grad()
def diagnostic_full(model, batch, temporal_shift=0):
    """Spell out the production FULL path without changing its operations."""
    spatial = model.spatial_encoder(
        batch["pixels"], batch["valid_pixels"], batch["extra"],
    )
    structure = model.structure_branch(spatial, batch["positions"])
    instance = model.temporal_encoder(
        spatial, batch["positions"] + temporal_shift,
        external_query=structure["shape_class_token"],
    )
    return {
        "logits": model.decoder(instance),
        "instance_feature": instance,
        **structure,
        "spatial": spatial,
    }


@torch.no_grad()
def compare_forward_paths(model, batch, temporal_shift=0):
    official = model.forward_with_temporal_shift(
        batch["pixels"], batch["valid_pixels"], batch["positions"], batch["extra"],
        temporal_shift=temporal_shift, return_dict=True,
    )
    full = diagnostic_full(model, batch, temporal_shift)
    spatial = model.spatial_encoder(
        batch["pixels"], batch["valid_pixels"], batch["extra"],
    )
    prepared = model.prepare_temporal_features(spatial, batch["positions"])
    structure = model.prepare_structure(prepared, batch["positions"])
    instance = model.temporal_encoder(
        prepared, batch["positions"] + temporal_shift,
        external_query=structure["shape_class_token"],
    )
    cached = {
        "logits": model.decoder(instance), "instance_feature": instance, **structure,
    }

    def difference(left, right):
        return {
            key: float((left[key] - right[key]).abs().max().detach().cpu())
            for key in OUTPUT_KEYS
        }

    return {
        "official_vs_full": difference(official, full),
        "official_vs_cached": difference(official, cached),
    }


def response_from_similarity(similarity, beta=5.0, hard_max=False, candidate_mask=None):
    """Re-pool frozen similarities without changing model parameters."""
    if candidate_mask is None:
        candidate_mask = torch.ones(
            similarity.shape[:2], dtype=torch.bool, device=similarity.device,
        )
    if candidate_mask.ndim == 1:
        candidate_mask = candidate_mask[None].expand(similarity.shape[0], -1)
    masked = similarity.masked_fill(~candidate_mask.unsqueeze(-1), -torch.inf)
    if hard_max:
        winners = masked.argmax(dim=1, keepdim=True)
        weights = torch.zeros_like(similarity).scatter_(1, winners, 1.0)
    else:
        weights = torch.softmax(float(beta) * masked, dim=1)
    return (weights * similarity).sum(dim=1), weights


@torch.no_grad()
def response_intervention_outputs(model, batch, temporal_shift=0, seed=1):
    spatial = model.spatial_encoder(
        batch["pixels"], batch["valid_pixels"], batch["extra"],
    )
    structure = model.structure_branch(spatial, batch["positions"])
    response = structure["shapelet_response"]
    generator = torch.Generator(device=response.device).manual_seed(int(seed))
    permutation = torch.randperm(response.shape[0], generator=generator, device=response.device)
    responses = {
        "FULL": response,
        "ZERO": response,
        "MEAN": response.mean(0, keepdim=True).expand_as(response),
        "SHUFFLE": response[permutation],
    }
    result = {}
    for name, current in responses.items():
        class_token = model.structure_branch.response_to_query(current)
        qshape = model.temporal_encoder.attention_heads.external_query_projection(class_token)
        external = class_token
        if name == "ZERO":
            qshape = torch.zeros_like(qshape)
            external = None
        instance = model.temporal_encoder(
            spatial, batch["positions"] + temporal_shift, external_query=external,
        )
        result[name] = {
            "logits": model.decoder(instance), "instance_feature": instance,
            "response": current, "shape_class_token": class_token,
            "qshape": qshape, "spatial": spatial, "positions": batch["positions"],
            "permutation": permutation,
        }
    return result


def _effective_rank(values):
    matrix = values.detach().float().reshape(-1, values.shape[-1]).cpu().numpy()
    singular = np.linalg.svd(matrix, compute_uv=False)
    total = singular.sum()
    if total <= 1e-12:
        return 0.0
    probability = singular / total
    return float(np.exp(-(probability * np.log(probability + 1e-12)).sum()))


def layer_statistics(values):
    values = values.detach().float()
    sample_view = values.reshape(values.shape[0], -1)
    result = {
        "sample_variance": float(sample_view.var(dim=0, unbiased=False).mean().cpu()),
        "candidate_variance": float("nan"),
        "feature_norm_mean": float(values.norm(dim=-1).mean().cpu()),
        "feature_norm_std": float(values.norm(dim=-1).std(unbiased=False).cpu()),
        "effective_rank": _effective_rank(values),
    }
    if values.ndim >= 3:
        result["candidate_variance"] = float(
            values.var(dim=1, unbiased=False).mean().cpu()
        )
    return result


def parameter_statistics(parameters):
    tensors = [value.detach().cpu().reshape(-1) for value in parameters]
    if not tensors:
        return {key: float("nan") for key in ("parameter_norm", "mean", "std", "min", "max")} | {"sha256": ""}
    values = torch.cat(tensors).float()
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.contiguous().numpy().tobytes())
    return {
        "parameter_norm": float(values.norm()), "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)), "min": float(values.min()),
        "max": float(values.max()), "sha256": digest.hexdigest(),
    }


@torch.no_grad()
def collect_structure_layers(model, spatial, positions, temporal_shift=0):
    """Collect exact existing layer outputs; no hooks or model state mutations."""
    branch = model.structure_branch
    exposed, _ = branch.exposer(spatial, positions)
    groups, _ = branch.window_extractor(exposed)
    layer_groups = {name: [] for name in (
        "normalized_morphology", "difference_morphology", "raw_encoder",
        "diff_encoder", "mean_encoder", "std_encoder", "fusion_linear",
        "fusion_layernorm", "shape_token",
    )}
    tokens = []
    for windows in groups:
        parts = branch.token_generator.components(windows)
        raw = branch.token_generator.raw_encoder(parts["normalized"])
        diff = branch.token_generator.diff_encoder(parts["difference"])
        mean = branch.token_generator.mean_encoder(parts["mean"])
        std = branch.token_generator.std_encoder(parts["std"])
        fused_input = torch.cat((raw, diff, mean, std), dim=-1)
        linear = branch.token_generator.fusion[0](fused_input)
        normalized = branch.token_generator.fusion[1](linear)
        token = branch.token_generator.fusion[2](normalized)
        values = (parts["normalized"], parts["difference"], raw, diff, mean, std, linear, normalized, token)
        for key, value in zip(layer_groups, values):
            layer_groups[key].append(value)
        tokens.append(token)
    combined = {key: torch.cat(value, dim=1) for key, value in layer_groups.items()}
    token = combined["shape_token"]
    response = branch.compute_rich_response(token)
    query = branch.response_to_query(response)
    qshape = model.temporal_encoder.attention_heads.external_query_projection(query)
    instance = model.temporal_encoder(
        spatial, positions + temporal_shift, external_query=query,
    )
    combined.update({
        "shapelet_response": response, "response_to_query": query,
        "qshape": qshape, "instance_feature": instance,
        "fourier_canonical": exposed,
    })
    return combined


def transfer_margin(source_features, source_labels, target_features, target_labels):
    """Return same-class and wrong-class coverage and their difference."""
    source = F.normalize(torch.as_tensor(source_features).float(), dim=-1).cpu().numpy()
    target = F.normalize(torch.as_tensor(target_features).float(), dim=-1).cpu().numpy()
    source_labels = np.asarray(source_labels); target_labels = np.asarray(target_labels)
    rows = {}
    for label in np.unique(target_labels):
        current = target[target_labels == label]
        same = source[source_labels == label]
        wrong = source[source_labels != label]
        if not len(current) or not len(same) or not len(wrong):
            continue
        same_coverage = (current @ same.T).max(1)
        wrong_coverage = (current @ wrong.T).max(1)
        rows[int(label)] = {
            "same_class_coverage": float(same_coverage.mean()),
            "wrong_class_coverage": float(wrong_coverage.mean()),
            "transfer_margin": float((same_coverage - wrong_coverage).mean()),
        }
    return rows
