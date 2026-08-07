"""Full-source prototype scanning and checkpoint-distance finalization.

The scanner runs a *deterministic* pass over the full labelled source training
split. It never uses the balanced/repeated training loader, never reads a
target batch, and keeps the model in eval mode with gradients disabled (the
functional geometry itself is unchanged; this is only a statistical scan).

Each training-epoch refresh computes the prototype bank in one pass. The
intra-class distance samples and quantiles are computed only once, during
checkpoint finalization, using a second deterministic pass.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .full_model import TSStructureModel
from .prototype_bank import (
    QUANTILE_LEVELS,
    SourcePrototypeBank,
    support_aware_q_distance,
)


def _model_forward(model: TSStructureModel, batch: dict) -> object:
    return model(
        batch["pixels"],
        batch["valid_pixels"],
        batch["positions"],
        batch.get("extra"),
        return_geometry=True,
    )


def _structure_geometry(output) -> tuple[Tensor, Tensor, Tensor]:
    geometry = output.geometry
    if geometry is None:
        raise RuntimeError("prototype scan requires functional geometry")
    return (
        geometry.structure_srvf,
        geometry.structure_support,
        geometry.structure_valid,
    )


def _trend_geometry(output) -> tuple[Tensor, Tensor, Tensor]:
    geometry = output.geometry
    if geometry is None:
        raise RuntimeError("prototype scan requires functional geometry")
    return (
        geometry.trend_srvf,
        geometry.trend_support,
        geometry.trend_valid,
    )


def _batch_labels(batch: dict, device: torch.device) -> Tensor:
    return batch["label"].to(device=device, dtype=torch.long)


def _canonical_grid(output) -> Tensor:
    return output.geometry.canonical_grid


def build_source_prototype_bank(
    model: TSStructureModel,
    source_scan_loader,
    num_classes: int,
    *,
    device: torch.device,
    eps: float = 1e-8,
    min_mean_support: float = 0.0,
) -> SourcePrototypeBank:
    """Build per-class source prototypes with a single full-source scan.

    Args:
        model: The Stage-1 model in eval mode during the scan.
        source_scan_loader: Deterministic full-source loader (no shuffle, no
            balancing, no repetition).
        num_classes: Number of source classes.
        device: Target device.
        eps: Denominator floor for support-weighted averages.
        min_mean_support: Minimum mean support for a sample's geometry to
            contribute to a prototype.

    Returns:
        A ``SourcePrototypeBank`` with ``version=0`` and empty distance samples.
    """
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            integration_weights: Tensor | None = None
            srvf_sum: list[torch.Tensor] = []
            support_sum: list[torch.Tensor] = []
            trend_sum: list[torch.Tensor] = []
            trend_support_sum: list[torch.Tensor] = []
            fused_sum: list[torch.Tensor] = []
            count: list[torch.Tensor] = []
            for _ in range(num_classes):
                srvf_sum.append(None)
                support_sum.append(None)
                trend_sum.append(None)
                trend_support_sum.append(None)
                fused_sum.append(None)
                count.append(torch.zeros((), dtype=torch.long, device=device))
            seen_grid = False

            for batch in source_scan_loader:
                output = _model_forward(model, batch)
                labels = _batch_labels(batch, device)
                trend_srvf, trend_support, _ = _trend_geometry(output)
                shape_srvf, shape_support, shape_valid = _structure_geometry(output)
                fused = output.fused_repr
                if not seen_grid:
                    grid = _canonical_grid(output)
                    if grid.ndim != 1:
                        raise ValueError("canonical_grid must have shape [K]")
                    integration_weights = torch.ones_like(grid)
                    integration_weights[[0, -1]] *= 0.5
                    integration_weights = integration_weights / integration_weights.sum()
                    seen_grid = True

                sample_valid = shape_valid & (
                    (shape_support * integration_weights.to(shape_support)).sum(dim=-1)
                    > min_mean_support
                )
                for class_id in range(num_classes):
                    mask = (labels == class_id) & sample_valid
                    if not torch.any(mask).item():
                        continue
                    q_c = shape_srvf[mask]
                    sup_c = shape_support[mask]
                    tr_c = trend_srvf[mask]
                    tr_sup_c = trend_support[mask]
                    fused_c = fused[mask]
                    weights_c = sup_c.unsqueeze(-1)
                    batch_shape_sum = (q_c * weights_c).sum(dim=0)
                    batch_support_sum = sup_c.sum(dim=0)
                    batch_trend_sum = (tr_c * tr_sup_c.unsqueeze(-1)).sum(dim=0)
                    batch_trend_support_sum = tr_sup_c.sum(dim=0)
                    batch_fused_sum = fused_c.sum(dim=0)
                    batch_count = int(mask.sum().item())
                    if srvf_sum[class_id] is None:
                        srvf_sum[class_id] = batch_shape_sum
                        support_sum[class_id] = batch_support_sum
                        trend_sum[class_id] = batch_trend_sum
                        trend_support_sum[class_id] = batch_trend_support_sum
                        fused_sum[class_id] = batch_fused_sum
                        count[class_id] = torch.tensor(
                            batch_count, device=device, dtype=torch.long
                        )
                    else:
                        srvf_sum[class_id] = srvf_sum[class_id] + batch_shape_sum
                        support_sum[class_id] = support_sum[class_id] + batch_support_sum
                        trend_sum[class_id] = trend_sum[class_id] + batch_trend_sum
                        trend_support_sum[class_id] = (
                            trend_support_sum[class_id] + batch_trend_support_sum
                        )
                        fused_sum[class_id] = fused_sum[class_id] + batch_fused_sum
                        count[class_id] = count[class_id] + batch_count

            if integration_weights is None:
                raise RuntimeError("source scan loader produced no batches")

            trend_srvf_out: list[Tensor] = []
            trend_support_out: list[Tensor] = []
            shape_srvf_out: list[Tensor] = []
            shape_support_out: list[Tensor] = []
            fused_out: list[Tensor] = []
            ready: list[bool] = []
            dtype = next(model.parameters()).dtype
            for class_id in range(num_classes):
                if count[class_id].item() == 0 or srvf_sum[class_id] is None:
                    trend_srvf_out.append(torch.zeros_like(shape_srvf[0][0:1]).squeeze(0))
                    trend_support_out.append(torch.zeros_like(shape_support[0][0:1]).squeeze(0))
                    shape_srvf_out.append(torch.zeros_like(shape_srvf[0][0:1]).squeeze(0))
                    shape_support_out.append(torch.zeros_like(shape_support[0][0:1]).squeeze(0))
                    fused_out.append(torch.zeros_like(fused[0]))
                    ready.append(False)
                    continue
                s = (srvf_sum[class_id] / (support_sum[class_id].unsqueeze(-1) + eps)).to(dtype=dtype)
                tr_s = (trend_sum[class_id] / (trend_support_sum[class_id].unsqueeze(-1) + eps)).to(dtype=dtype)
                fused_mu = (fused_sum[class_id] / count[class_id].float()).to(dtype=dtype)
                shape_srvf_out.append(s)
                shape_support_out.append((support_sum[class_id] / count[class_id].float()).to(dtype=dtype))
                trend_srvf_out.append(tr_s)
                trend_support_out.append((trend_support_sum[class_id] / count[class_id].float()).to(dtype=dtype))
                fused_out.append(fused_mu)
                ready.append(True)

            trend_srvf_t = torch.stack(trend_srvf_out)
            trend_support_t = torch.stack(trend_support_out)
            shape_srvf_t = torch.stack(shape_srvf_out)
            shape_support_t = torch.stack(shape_support_out)
            fused_t = torch.stack(fused_out)
            counts_t = torch.tensor([int(c.item()) for c in count], device=device, dtype=torch.long)
            ready_t = torch.tensor(ready, device=device, dtype=torch.bool)
            q_quantiles = torch.zeros(num_classes, len(QUANTILE_LEVELS), device=device, dtype=dtype)
            f_quantiles = torch.zeros_like(q_quantiles)
            return SourcePrototypeBank(
                trend_srvf=trend_srvf_t,
                shape_srvf=shape_srvf_t,
                trend_support=trend_support_t,
                shape_support=shape_support_t,
                fused=fused_t,
                class_counts=counts_t,
                ready=ready_t,
                q_distance_samples=tuple(torch.zeros(0, device=device, dtype=dtype) for _ in range(num_classes)),
                f_distance_samples=tuple(torch.zeros(0, device=device, dtype=dtype) for _ in range(num_classes)),
                q_quantiles=q_quantiles,
                f_quantiles=f_quantiles,
                version=0,
            )
    finally:
        model.train(was_training)


def finalize_distance_statistics(
    model: TSStructureModel,
    source_scan_loader,
    bank: SourcePrototypeBank,
    *,
    device: torch.device,
    num_shape_examples_per_class: int = 3,
    eps: float = 1e-8,
    min_mean_support: float = 0.0,
) -> tuple[SourcePrototypeBank, list[dict]]:
    """Compute intra-class distance samples, quantiles and Shape examples.

    Runs a second deterministic full-source pass. Returns an updated bank whose
    distance samples / quantiles are populated, plus a list of Shape examples.
    """
    num_classes = bank.trend_srvf.shape[0]
    q_samples: list[list[float]] = [[] for _ in range(num_classes)]
    f_samples: list[list[float]] = [[] for _ in range(num_classes)]
    # Per class, keep (distance, sample_id, q, support, positions)
    q_by_class: list[list[tuple]] = [[] for _ in range(num_classes)]
    integration_weights: Tensor | None = None

    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            for batch in source_scan_loader:
                output = _model_forward(model, batch)
                labels = _batch_labels(batch, device)
                shape_srvf, shape_support, shape_valid = _structure_geometry(output)
                fused = output.fused_repr
                if integration_weights is None:
                    grid = _canonical_grid(output)
                    integration_weights = torch.ones_like(grid)
                    integration_weights[[0, -1]] *= 0.5
                    integration_weights = integration_weights / integration_weights.sum()

                q_dist = support_aware_q_distance(
                    shape_srvf,
                    bank.shape_srvf,
                    shape_support,
                    bank.shape_support,
                    integration_weights.to(shape_srvf),
                    eps=eps,
                )
                fused_prototypes = torch.nn.functional.normalize(bank.fused, dim=-1)
                fused_norm = torch.nn.functional.normalize(fused, dim=-1)
                f_dist = 1.0 - (fused_norm @ fused_prototypes.T)

                for class_id in range(num_classes):
                    mask = (
                        (labels == class_id)
                        & shape_valid
                        & q_dist.valid[:, class_id]
                        & (
                            (shape_support * integration_weights.to(shape_support)).sum(dim=-1)
                            > min_mean_support
                        )
                    )
                    valid_indices = mask.nonzero(as_tuple=False).squeeze(-1)
                    if valid_indices.numel() == 0:
                        continue
                    q_d = q_dist.distance[valid_indices, class_id].tolist()
                    f_d = f_dist[valid_indices, class_id].tolist()
                    q_samples[class_id].extend(q_d)
                    f_samples[class_id].extend(f_d)
                    sample_ids = batch.get("parcel_index") if isinstance(batch, dict) else None
                    for offset, idx in enumerate(valid_indices.tolist()):
                        sample_id = (
                            int(sample_ids[idx].item())
                            if sample_ids is not None
                            else None
                        )
                        q_by_class[class_id].append(
                            (
                                float(q_d[offset]),
                                sample_id,
                                shape_srvf[idx].detach().cpu(),
                                shape_support[idx].detach().cpu(),
                                output.positions[idx].detach().cpu(),
                            )
                        )
        if integration_weights is None:
            raise RuntimeError("source scan loader produced no batches")

        q_samples_out: list[Tensor] = []
        f_samples_out: list[Tensor] = []
        q_quantiles = bank.q_quantiles.clone()
        f_quantiles = bank.f_quantiles.clone()
        examples: list[dict] = []
        for class_id in range(num_classes):
            q_sorted = torch.tensor(sorted(q_samples[class_id]), device=device)
            f_sorted = torch.tensor(sorted(f_samples[class_id]), device=device)
            q_samples_out.append(q_sorted)
            f_samples_out.append(f_sorted)
            if q_sorted.numel() > 0:
                for level_index, level in enumerate(QUANTILE_LEVELS):
                    q_quantiles[class_id, level_index] = torch.quantile(q_sorted, level)
            if f_sorted.numel() > 0:
                for level_index, level in enumerate(QUANTILE_LEVELS):
                    f_quantiles[class_id, level_index] = torch.quantile(f_sorted, level)
            examples.extend(_select_shape_examples(q_by_class[class_id], class_id))

        updated = SourcePrototypeBank(
            trend_srvf=bank.trend_srvf,
            shape_srvf=bank.shape_srvf,
            trend_support=bank.trend_support,
            shape_support=bank.shape_support,
            fused=bank.fused,
            class_counts=bank.class_counts,
            ready=bank.ready,
            q_distance_samples=tuple(q_samples_out),
            f_distance_samples=tuple(f_samples_out),
            q_quantiles=q_quantiles,
            f_quantiles=f_quantiles,
            version=bank.version,
        )
        return updated, examples
    finally:
        model.train(was_training)


def _select_shape_examples(
    entries: list[tuple], class_id: int
) -> list[dict]:
    """Pick up to three distinct true source samples per class."""
    if not entries:
        return []
    ordered = sorted(entries, key=lambda entry: entry[0])
    median_index = len(ordered) // 2
    nearest = ordered[0]
    median = ordered[min(median_index, len(ordered) - 1)]
    q95_index = min(int(0.95 * (len(ordered) - 1)), len(ordered) - 1)
    outer = ordered[max(q95_index, median_index + 1) if q95_index <= median_index else q95_index]

    chosen: list[tuple] = []
    for candidate in (nearest, median, outer):
        if candidate not in chosen:
            chosen.append(candidate)
    return [
        {
            "class_id": class_id,
            "sample_id": entry[1],
            "role": role,
            "q_shape": entry[2],
            "support": entry[3],
            "canonical_grid": torch.linspace(0.0, 1.0, entry[2].shape[0]),
            "distance_to_prototype": float(entry[0]),
            "original_positions": entry[4],
        }
        for entry, role in zip(
            chosen,
            ("prototype_nearest", "class_median", "outer_representative"),
        )
    ]

@torch.no_grad()
@torch.no_grad()
def refresh_source_fused_statistics(
    model: TSStructureModel,
    source_scan_loader,
    bank: SourcePrototypeBank,
    *,
    device: torch.device,
) -> SourcePrototypeBank:
    """Refresh only source fused-feature statistics with a frozen geometry bank.

    Stage 2 changes the raw Time2Vec/LTAE/classifier path but permanently freezes
    PSE, decomposition and functional geometry.  This helper therefore performs
    one deterministic EMA-teacher source scan, recomputes per-class fused means
    and empirical fused-distance distributions, and copies every geometry field
    from ``bank`` unchanged.  ``bank.version`` is intentionally preserved: it is
    the source-geometry version used to validate the cached target registration
    hypotheses.
    """
    if not isinstance(bank, SourcePrototypeBank):
        raise TypeError("bank must be a SourcePrototypeBank")
    num_classes = int(bank.ready.numel())
    fused_rows: list[list[Tensor]] = [[] for _ in range(num_classes)]
    was_training = model.training
    try:
        model.eval()
        for batch in source_scan_loader:
            extra = batch.get("extra")
            if isinstance(extra, Tensor):
                extra = extra.to(device=device)
            output = model(
                batch["pixels"].to(device=device),
                batch["valid_pixels"].to(device=device),
                batch["positions"].to(device=device),
                extra,
                return_geometry=False,
            )
            labels = batch["label"].to(device=device, dtype=torch.long)
            for class_id in range(num_classes):
                mask = labels == class_id
                if torch.any(mask).item():
                    fused_rows[class_id].append(output.fused_repr[mask].detach())
    finally:
        model.train(was_training)

    fused = bank.fused.detach().to(device=device).clone()
    f_samples: list[Tensor] = []
    f_quantiles = bank.f_quantiles.detach().to(device=device).clone()
    for class_id in range(num_classes):
        if not fused_rows[class_id]:
            f_samples.append(bank.f_distance_samples[class_id].detach().to(device=device))
            continue
        rows = torch.cat(fused_rows[class_id], dim=0)
        center = rows.mean(dim=0)
        fused[class_id] = center.to(dtype=fused.dtype)
        distances = 1.0 - torch.nn.functional.cosine_similarity(
            rows,
            center.unsqueeze(0).expand_as(rows),
            dim=-1,
        )
        distances = torch.sort(distances.detach()).values
        f_samples.append(distances)
        for level_index, level in enumerate(QUANTILE_LEVELS):
            f_quantiles[class_id, level_index] = torch.quantile(distances, level)

    return SourcePrototypeBank(
        trend_srvf=bank.trend_srvf.detach(),
        shape_srvf=bank.shape_srvf.detach(),
        trend_support=bank.trend_support.detach(),
        shape_support=bank.shape_support.detach(),
        fused=fused.detach(),
        class_counts=bank.class_counts.detach(),
        ready=bank.ready.detach(),
        q_distance_samples=tuple(item.detach() for item in bank.q_distance_samples),
        f_distance_samples=tuple(item.detach() for item in f_samples),
        q_quantiles=bank.q_quantiles.detach(),
        f_quantiles=f_quantiles.detach(),
        version=bank.version,
    )
