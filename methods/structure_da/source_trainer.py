"""Minimal source-only CE training loop for the two-stage structure model.

This round keeps the loop intentionally small: source CE, AMP, one optimizer,
and basic loss/accuracy logging. There is no prototype bank, no target
loader, no EMA and no geometry optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer

from .full_model import TSStructureModel


@dataclass(frozen=True)
class SourceTrainStepOutput:
    loss: Tensor
    classification_loss: Tensor
    logits: Tensor


def _resolve_time_mask(
    time_mask: Tensor | None,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> Tensor:
    if time_mask is None:
        return torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device)
    if not isinstance(time_mask, Tensor):
        raise ValueError("time_mask must be a torch.Tensor or None")
    if time_mask.ndim == 1:
        if time_mask.shape != (sequence_length,):
            raise ValueError("time_mask must have shape [L] or [B, L]")
        time_mask = time_mask.unsqueeze(0).expand(batch_size, -1)
    elif time_mask.ndim == 2:
        if time_mask.shape != (batch_size, sequence_length):
            raise ValueError("time_mask must have shape [L] or [B, L]")
    else:
        raise ValueError("time_mask must have shape [L] or [B, L]")
    return time_mask.to(device=device, dtype=torch.bool)


class SourceClassificationTrainer:
    """Run one source CE training step and basic accuracy logging."""

    def __init__(
        self,
        model: TSStructureModel,
        optimizer: Optimizer,
        *,
        device: torch.device,
        amp_enabled: bool,
        amp_dtype: str = "float16",
    ) -> None:
        if not isinstance(model, TSStructureModel):
            raise ValueError("model must be a TSStructureModel")
        if amp_dtype not in ("float16", "bfloat16"):
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.amp_enabled = bool(amp_enabled)
        self.amp_dtype = amp_dtype
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.amp_enabled
                and device.type == "cuda"
                and amp_dtype == "float16"
            ),
        )

    def train_step(self, batch: dict) -> dict[str, float]:
        """Run one source-only CE step and return scalar metrics."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pixels = batch["pixels"].to(device=self.device)
        valid_pixels = batch["valid_pixels"].to(device=self.device)
        positions = batch["positions"].to(device=self.device)
        labels = batch["label"].to(device=self.device, dtype=torch.long)
        amp_dtype = getattr(torch, self.amp_dtype)
        amp_on = self.amp_enabled and (
            self.device.type == "cuda" or amp_dtype == torch.bfloat16
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=amp_dtype,
            enabled=amp_on,
        ):
            output = self.model(
                pixels,
                valid_pixels,
                positions,
                batch.get("extra"),
                return_geometry=False,
            )
            loss = F.cross_entropy(output.logits, labels)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        with torch.no_grad():
            predictions = output.logits.detach().argmax(dim=-1)
            accuracy = (predictions == labels).float().mean().item()
        return {
            "loss": float(loss.detach().item()),
            "classification_loss": float(loss.detach().item()),
            "accuracy": float(accuracy),
        }
