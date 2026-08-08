from __future__ import annotations

import json

import torch

from methods.structure_da import DomainShapeStatus
from tests.structure_da.test_stage1_training_helpers import _batch
from tests.structure_da.test_stage2_trainer import _real_trainer


def test_final_shape_synthesis_audit_is_read_only_and_exports_actual_examples(tmp_path) -> None:
    trainer = _real_trainer(DomainShapeStatus.CONFIRMED)
    trainer.output_dir = str(tmp_path)
    trainer.source_loader = [_batch()]
    before_steps = trainer.successful_optimizer_steps
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.student.named_parameters()
    }

    paths = trainer.write_final_shape_synthesis_audit(samples_per_class=2, max_batches=1)

    assert trainer.successful_optimizer_steps == before_steps == 0
    for name, parameter in trainer.student.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name])
    payload = json.loads((tmp_path / "stage2_shape_synthesis_audit.json").read_text())
    assert paths["json"] == str(tmp_path / "stage2_shape_synthesis_audit.json")
    assert paths["tensors"] == str(tmp_path / "stage2_shape_synthesis_audit.pt")
    assert payload["synthesis"]["available"] is True
    assert payload["synthesis"]["generated"] == 6
    assert payload["synthesis"]["sample_counts"] == {"0": 2, "1": 2, "2": 2}
    assert payload["synthesis"]["all_finite"] is True
    assert payload["synthesis"]["all_valid_support"] is True
    assert payload["synthesis"]["max_phase_leakage"] <= 1e-6
    assert payload["synthesis"]["max_q_shift_error"] <= 1e-6
    assert (tmp_path / "stage2_shape_synthesis_audit.pt").is_file()
