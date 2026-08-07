"""Stage-1 source trainer: CE warmup then full prototype-relative supervision.

The trainer owns a single task optimizer and never updates the prototype bank
inside a mini-batch. During warmup it runs the model with ``return_geometry=
False`` and applies only CE; afterwards it enables geometry and evaluates the
complete Stage-1 objective against the current (epoch-fixed) prototype bank.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer

from .full_model import TSStructureModel
from .prototype_bank import SourcePrototypeBank
from .stage1_objective import Stage1Objective


@dataclass(frozen=True)
class SourceTrainStepOutput:
    loss: Tensor
    classification_loss: Tensor
    logits: Tensor


class SourceClassificationTrainer:
    """Run one source-only Stage-1 training step."""

    def __init__(
        self,
        model: TSStructureModel,
        optimizer: Optimizer,
        *,
        device: torch.device,
        amp_enabled: bool,
        amp_dtype: str = "float16",
        objective: Stage1Objective | None = None,
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
        self.objective = objective
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.amp_enabled
                and device.type == "cuda"
                and amp_dtype == "float16"
            ),
        )

    def _integration_weights(self) -> Tensor:
        extractor = self.model.temporal_module.structure_geometry
        grid = extractor.functional_lift.canonical_grid.to(
            device=self.device, dtype=torch.float32
        )
        weights = torch.ones_like(grid)
        weights[[0, -1]] *= 0.5
        weights = weights / weights.sum()
        return weights

    def train_step(
        self,
        batch: dict,
        *,
        warmup: bool = False,
        bank: SourcePrototypeBank | None = None,
    ) -> dict[str, float]:
        """Run one Stage-1 source step and return scalar metrics.

        Args:
            batch: A source batch dict with ``pixels``, ``valid_pixels``,
                ``positions``, ``label`` and optionally ``extra``.
            warmup: When True only CE is applied and geometry is skipped.
            bank: The epoch-fixed prototype bank, used only when not warmup.
        """
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
                return_geometry=not warmup,
            )
            if warmup:
                loss = F.cross_entropy(output.logits, labels)
                q_proto = None
                f_proto = None
                q_to_cls = None
                q_count = 0
                f_count = 0
                consistency_count = 0
            else:
                if self.objective is None:
                    raise ValueError("objective is required outside warmup")
                if output.geometry is None:
                    raise RuntimeError("geometry must be computed outside warmup")
                geometry = output.geometry
                integration_weights = self._integration_weights()
                loss_output = self.objective(
                    logits=output.logits,
                    fused_repr=output.fused_repr,
                    labels=labels,
                    q=geometry.structure_srvf,
                    q_support=geometry.structure_support,
                    q_valid=geometry.structure_valid,
                    bank=bank,
                    integration_weights=integration_weights,
                    warmup=False,
                )
                loss = loss_output.total
                q_proto = loss_output.q_prototype
                f_proto = loss_output.fused_prototype
                q_to_cls = loss_output.q_to_classifier
                q_count = loss_output.q_valid_count
                f_count = loss_output.fused_valid_count
                consistency_count = loss_output.consistency_valid_count
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        with torch.no_grad():
            predictions = output.logits.detach().argmax(dim=-1)
            accuracy = (predictions == labels).float().mean().item()
        metrics: dict[str, float] = {
            "loss": float(loss.detach().item()),
            "classification_loss": float(F.cross_entropy(output.logits.detach(), labels).item()),
        }
        if warmup:
            metrics["q_proto_loss"] = 0.0
            metrics["f_proto_loss"] = 0.0
            metrics["q_to_cls_loss"] = 0.0
            metrics["q_valid_count"] = 0.0
            metrics["f_valid_count"] = 0.0
            metrics["consistency_valid_count"] = 0.0
        else:
            metrics["q_proto_loss"] = float(q_proto.detach().item())
            metrics["f_proto_loss"] = float(f_proto.detach().item())
            metrics["q_to_cls_loss"] = float(q_to_cls.detach().item())
            metrics["q_valid_count"] = float(q_count)
            metrics["f_valid_count"] = float(f_count)
            metrics["consistency_valid_count"] = float(consistency_count)
        metrics["accuracy"] = float(accuracy)
        return metrics
