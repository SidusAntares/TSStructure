"""EMA teacher restricted to the explicit Stage-2 trainable boundary."""

from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn

from .stage2_parameter_policy import Stage2ParameterPolicy


class Stage2EMATeacher:
    def __init__(
        self,
        teacher: nn.Module,
        *,
        ema_parameter_names: tuple[str, ...],
        copied_buffer_names: tuple[str, ...],
        decay: float,
    ) -> None:
        self._teacher = teacher
        self.ema_parameter_names = ema_parameter_names
        self._copied_buffer_names = copied_buffer_names
        self.decay = decay

    @staticmethod
    def _trainable_floating_buffer_names(
        student: nn.Module,
        trainable_parameter_names: set[str],
    ) -> tuple[str, ...]:
        names: list[str] = []
        for module_name, module in student.named_modules():
            prefix = f"{module_name}." if module_name else ""
            owns_trainable = any(
                f"{prefix}{local_name}" in trainable_parameter_names
                for local_name, _ in module.named_parameters(recurse=False)
            )
            if not owns_trainable:
                continue
            for local_name, buffer in module.named_buffers(recurse=False):
                if buffer.is_floating_point():
                    names.append(f"{prefix}{local_name}")
        return tuple(names)

    @classmethod
    def from_student(
        cls,
        student: nn.Module,
        policy: Stage2ParameterPolicy,
        decay: float,
    ) -> "Stage2EMATeacher":
        if not isinstance(student, nn.Module):
            raise TypeError("student must be a torch.nn.Module")
        if not isinstance(policy, Stage2ParameterPolicy):
            raise TypeError("policy must be a Stage2ParameterPolicy")
        try:
            decay_value = float(decay)
        except (TypeError, ValueError) as error:
            raise ValueError("decay must satisfy 0 <= decay < 1") from error
        if not math.isfinite(decay_value) or not 0.0 <= decay_value < 1.0:
            raise ValueError("decay must satisfy 0 <= decay < 1")
        parameter_names = {name for name, _ in student.named_parameters()}
        policy_names = set(policy.trainable_parameter_names) | set(
            policy.frozen_parameter_names
        )
        if parameter_names != policy_names:
            raise ValueError("policy must exactly partition the student parameters")

        teacher = deepcopy(student)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        copied_buffers = cls._trainable_floating_buffer_names(
            student, set(policy.trainable_parameter_names)
        )
        return cls(
            teacher,
            ema_parameter_names=policy.trainable_parameter_names,
            copied_buffer_names=copied_buffers,
            decay=decay_value,
        )

    @torch.no_grad()
    def update_after_optimizer_step(self, student: nn.Module) -> None:
        student_parameters = dict(student.named_parameters())
        teacher_parameters = dict(self._teacher.named_parameters())
        for name in self.ema_parameter_names:
            if name not in student_parameters or name not in teacher_parameters:
                raise ValueError(f"EMA parameter {name!r} is missing")
            teacher_parameters[name].mul_(self.decay).add_(
                student_parameters[name].detach(), alpha=1.0 - self.decay
            )
        student_buffers = dict(student.named_buffers())
        teacher_buffers = dict(self._teacher.named_buffers())
        for name in self._copied_buffer_names:
            if name not in student_buffers or name not in teacher_buffers:
                raise ValueError(f"EMA buffer {name!r} is missing")
            teacher_buffers[name].copy_(student_buffers[name].detach())
        self._teacher.eval()

    def model(self) -> nn.Module:
        return self._teacher
