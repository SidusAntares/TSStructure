from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import methods.structure_da.phase_visualization_protocol as phasevis
from methods.structure_da import (
    DomainPhaseState,
    PhaseClassCenter,
    PhaseDecisionStatus,
    PhaseGroup,
    PhaseGroupStatus,
)
from methods.structure_da.stage2_trainer import (
    _phase_state_payload,
    run_stage2_statistics_diagnostic,
)


def _gamma(power: float = 1.0) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 8, dtype=torch.float64).pow(power)


def test_stage2_model_state_dict_is_accepted_by_visualization_loader() -> None:
    stage2 = {"model_state_dict": {"weight": torch.tensor([1.0])}}
    legacy = {"state_dict": {"weight": torch.tensor([2.0])}}
    assert phasevis.checkpoint_model_state_dict(stage2)["weight"].item() == 1.0
    assert phasevis.checkpoint_model_state_dict(legacy)["weight"].item() == 2.0


def test_phase_only_time_encoder_uses_single_ltae_contract() -> None:
    marker = object()
    model = SimpleNamespace(
        temporal_module=SimpleNamespace(
            raw_encoder=SimpleNamespace(time_encoder=marker)
        )
    )
    assert phasevis.phase_only_time_encoder(model) is marker

    stale_model = SimpleNamespace(
        temporal_module=SimpleNamespace(
            raw_encoder=SimpleNamespace(
                shared_ltae=SimpleNamespace(shared_time_encoder=marker)
            )
        )
    )
    try:
        phasevis.phase_only_time_encoder(stale_model)
    except RuntimeError as error:
        assert "raw_encoder.time_encoder" in str(error)
    else:
        raise AssertionError("removed dual-stream encoder path must not be accepted")


def test_fold_reconstruction_exposes_held_out_test_partition() -> None:
    source = np.arange(10, dtype=np.int64)
    target = np.arange(100, 110, dtype=np.int64)
    splits = phasevis.reconstruct_fold_splits(
        source,
        target,
        source="source",
        target="target",
        seed=7,
        val_ratio=0.1,
        test_ratio=0.2,
        fold=0,
    )
    for domain in ("source", "target"):
        assert len(splits[domain]["train"]) == 7
        assert len(splits[domain]["val"]) == 1
        assert len(splits[domain]["test"]) == 2
        assert splits[domain]["train"].isdisjoint(splits[domain]["test"])
        assert splits[domain]["val"].isdisjoint(splits[domain]["test"])


def test_visualization_uses_phase_usage_routes_not_only_founder_classes() -> None:
    group = {
        "group_id": 0,
        "member_classes": (0, 3),
        "status": "confirmed",
        "center_gamma": _gamma(),
    }
    mapping = phasevis.class_to_group(
        [group], phase_routes=(0, 0, 0, 0), num_classes=4
    )
    assert tuple(sorted(mapping)) == (0, 1, 2, 3)
    assert all(mapping[class_id]["group_id"] == 0 for class_id in mapping)


def test_pse_direct_metric_changes_only_time_coordinates() -> None:
    source_record = {
        "parcel_index": 1,
        "mask": torch.tensor([True, True, True]),
        "positions": torch.tensor([0.0, 0.5, 1.0]),
        "pse_tokens": torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
    }
    target_record = {
        "parcel_index": 2,
        "mask": torch.tensor([True, True, True]),
        "positions": torch.tensor([0.2, 0.6, 1.0]),
        "positions_after": torch.tensor([0.0, 0.5, 1.0]),
        "pse_tokens": source_record["pse_tokens"].clone(),
    }
    summary, rows, _ = phasevis.pse_class_metrics(
        [source_record], [target_record], grid_size=33
    )
    assert summary["pse_class_mean_l2_before"] > 0.0
    assert summary["pse_class_mean_l2_after"] < 1e-7
    assert rows[0]["pse_l2_after"] < rows[0]["pse_l2_before"]


def test_phase_checkpoint_payload_preserves_class_center_gammas() -> None:
    center = PhaseClassCenter(
        class_id=2,
        center_gamma=_gamma(1.1),
        candidate_count=7,
        effective_evidence_count=6.5,
        dispersion=0.01,
        diameter=0.02,
        median_distance=0.03,
        center_drift=0.04,
        valid=True,
        reject_reason=None,
    )
    group = PhaseGroup(
        group_id=0,
        member_classes=(2, 4),
        center_gamma=_gamma(1.05),
        within_dispersion=0.01,
        diameter=0.02,
        core_radius=0.03,
        sample_evidence_count=20.0,
        class_count=2,
        center_drift=0.01,
        status=PhaseGroupStatus.CONFIRMED,
        confirmation_age=2,
    )
    state = DomainPhaseState(
        scan_index=4,
        m=1,
        class_centers=(center,),
        valid_phase_classes=(2,),
        groups=(group,),
        rejected_classes=(),
        decision_status=PhaseDecisionStatus.NONIDENTITY_CONFIRMED,
        decision_stability_age=2,
    )
    payload = _phase_state_payload(state)
    assert payload["class_centers"][0]["class_id"] == 2
    torch.testing.assert_close(payload["class_centers"][0]["center_gamma"], center.center_gamma)
    assert payload["groups"][0]["sample_evidence_count"] == 20.0


def test_diagnostic_only_run_persists_zero_step_calibration_state() -> None:
    snapshot = SimpleNamespace(
        phase_state=SimpleNamespace(
            m=1,
            decision_status=PhaseDecisionStatus.NONIDENTITY_CONFIRMED,
        ),
        stable_labels=SimpleNamespace(num_stable_labels=10),
    )

    class FakeTrainer:
        successful_optimizer_steps = 0

        def __init__(self):
            self.saved = []

        def initialize_statistics(self):
            return snapshot

        def save_ema_checkpoint(self, filename, *, epoch, target_val):
            self.saved.append((filename, epoch, target_val))
            return f"/tmp/{filename}"

    trainer = FakeTrainer()
    result = run_stage2_statistics_diagnostic(trainer)
    assert result is snapshot
    assert trainer.saved == [("stage2_calibration_state.pt", 0, None)]


def test_visualization_scripts_follow_phase_only_forward_contract() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    for relative in (
        "scripts/visualize_stage2_phase_alignment.py",
        "scripts/compare_stage2_phase_vs_timematch_shift.py",
    ):
        tree = ast.parse((repository_root / relative).read_text(encoding="utf-8"))
        removed = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in {
                "trend_repr",
                "structure_repr",
                "shared_ltae",
                "shared_time_encoder",
            }
        }
        assert not removed, f"{relative} still uses removed Phase-only fields: {removed}"
