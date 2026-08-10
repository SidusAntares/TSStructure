"""Stage-1 source trainer for the Phase-only TimeMatch-like classification path."""

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


class SourceClassificationTrainer:
    """Run one source-only CE step; functional geometry is not in the gradient path."""

    def __init__(
        self,
        model: TSStructureModel,
        optimizer: Optimizer,
        *,
        device: torch.device,
        amp_enabled: bool,
        amp_dtype: str = "float16",
        objective=None,
    ) -> None:
        del objective
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
            enabled=(self.amp_enabled and device.type == "cuda" and amp_dtype == "float16"),
        )

    def train_step(self, batch: dict, *, warmup: bool = False, bank=None) -> dict[str, float]:
        del warmup, bank
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pixels = batch["pixels"].to(device=self.device)
        valid_pixels = batch["valid_pixels"].to(device=self.device)
        positions = batch["positions"].to(device=self.device)
        labels = batch["label"].to(device=self.device, dtype=torch.long)
        extra = batch.get("extra")
        if isinstance(extra, Tensor):
            extra = extra.to(device=self.device)
        amp_dtype = getattr(torch, self.amp_dtype)
        amp_on = self.amp_enabled and (
            self.device.type == "cuda" or amp_dtype == torch.bfloat16
        )
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=amp_on):
            output = self.model(
                pixels,
                valid_pixels,
                positions,
                extra,
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
            "q_proto_loss": 0.0,
            "f_proto_loss": 0.0,
            "q_to_cls_loss": 0.0,
            "q_valid_count": 0.0,
            "f_valid_count": 0.0,
            "consistency_valid_count": 0.0,
            "accuracy": float(accuracy),
        }
