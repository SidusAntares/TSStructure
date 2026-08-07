from __future__ import annotations

import pytest
import torch

from tests.structure_da.test_stage2_parameter_policy import _model


def test_ema_teacher_starts_equal_eval_and_frozen_without_mutating_student() -> None:
    from methods.structure_da.ema_teacher import Stage2EMATeacher
    from methods.structure_da.stage2_parameter_policy import configure_stage2_parameter_policy

    student = _model().train()
    policy = configure_stage2_parameter_policy(student)
    student_training = student.training
    student_requires_grad = {
        name: parameter.requires_grad for name, parameter in student.named_parameters()
    }
    ema = Stage2EMATeacher.from_student(student, policy, decay=0.75)
    teacher = ema.model()

    assert student.training is student_training
    assert student_requires_grad == {
        name: parameter.requires_grad for name, parameter in student.named_parameters()
    }
    assert teacher.training is False
    assert ema.ema_parameter_names == policy.trainable_parameter_names
    for (student_name, student_parameter), (teacher_name, teacher_parameter) in zip(
        student.named_parameters(), teacher.named_parameters()
    ):
        assert student_name == teacher_name
        torch.testing.assert_close(teacher_parameter, student_parameter, rtol=0, atol=0)
        assert teacher_parameter.requires_grad is False


def test_ema_updates_only_stage2_parameters_and_copies_trainable_floating_buffers() -> None:
    from methods.structure_da.ema_teacher import Stage2EMATeacher
    from methods.structure_da.stage2_parameter_policy import configure_stage2_parameter_policy

    student = _model()
    policy = configure_stage2_parameter_policy(student)
    ema = Stage2EMATeacher.from_student(student, policy, decay=0.75)
    teacher = ema.model()
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in teacher.named_parameters()
    }
    initial_buffers = {name: value.detach().clone() for name, value in teacher.named_buffers()}

    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(1.0)
        student.classifier[0].norm.running_mean.add_(3.0)
        student.backbone.pixel_set_encoder.mlp1[0].norm.running_mean.add_(5.0)
    ema.update_after_optimizer_step(student)

    trainable = set(policy.trainable_parameter_names)
    for name, parameter in teacher.named_parameters():
        expected = initial_parameters[name] + (0.25 if name in trainable else 0.0)
        torch.testing.assert_close(parameter, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(
        teacher.classifier[0].norm.running_mean,
        student.classifier[0].norm.running_mean,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        teacher.backbone.pixel_set_encoder.mlp1[0].norm.running_mean,
        initial_buffers["backbone.pixel_set_encoder.mlp1.0.norm.running_mean"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("decay", [-0.1, 1.0, float("nan")])
def test_ema_teacher_rejects_invalid_decay(decay: float) -> None:
    from methods.structure_da.ema_teacher import Stage2EMATeacher
    from methods.structure_da.stage2_parameter_policy import configure_stage2_parameter_policy

    student = _model()
    with pytest.raises(ValueError):
        Stage2EMATeacher.from_student(
            student, configure_stage2_parameter_policy(student), decay=decay
        )
