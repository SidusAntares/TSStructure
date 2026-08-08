from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from methods.structure_da import (
    DomainPhaseConfig,
    DomainPhaseState,
    DomainShapeConfig,
    DomainShapeState,
    DomainShapeStatus,
    PhaseGroup,
    PhaseGroupStatus,
    PhaseHypothesisScanConfig,
    SourcePrototypeBank,
    StableLabelConfig,
    StableTargetLabelScanResult,
    Stage2EMATeacher,
    Stage2ObjectiveConfig,
    Stage2StatisticsSnapshot,
    Stage2Trainer,
    Stage2TrainerConfig,
    TargetHypothesisScanResult,
    build_phase_only_synthetic_source_example,
    configure_stage2_parameter_policy,
    refresh_source_fused_statistics,
    run_stage2_statistics_diagnostic,
    run_stage2_training,
)

from tests.structure_da.test_stage1_training_helpers import _bank, _batch, _model


def _trainer_config(**objective_overrides) -> Stage2TrainerConfig:
    objective = dict(
        lambda_src_proto=0.1,
        lambda_src_cons=0.1,
        lambda_syn=1.0,
        lambda_syn_cons=0.1,
        tau_q=0.1,
        fused_margin=0.1,
    )
    objective.update(objective_overrides)
    return Stage2TrainerConfig(
        phase_scan=PhaseHypothesisScanConfig(
            registration_lambda=0.0,
            registration_gain_ratio_max=1.0,
            registration_min_common_support=0.0,
            registration_max_roughness=100.0,
            registration_min_increment=0.0,
            registration_max_local_speed=100.0,
            registration_max_deviation=1.0,
            class_hypothesis_margin=0.1,
        ),
        phase=DomainPhaseConfig(
            phase_min_samples_per_class=1.0,
            phase_class_dispersion_max=1.0,
            phase_class_diameter_max=1.0,
            phase_group_dispersion_max=1.0,
            phase_group_diameter_max=1.0,
            phase_group_core_separation=0.0,
            phase_global_radius=1.0,
            phase_confirmation_patience=2,
            phase_center_drift_max=1.0,
        ),
        stable_labels=StableLabelConfig(
            tau_f=0.1,
            tau_q=0.1,
            cls_confidence_min=0.0,
            cls_margin_min=None,
            fused_confidence_min=0.0,
            fused_margin_min=None,
            q_confidence_min=0.0,
            q_margin_min=None,
        ),
        shape=DomainShapeConfig(
            shape_min_valid_classes=2,
            shape_min_samples_per_class=1,
            shape_shared_ratio_min=0.0,
            shape_leave_one_out_drift_max=100.0,
            shape_center_drift_max=100.0,
            shape_effect_norm_max=100.0,
            shape_confirmation_patience=2,
        ),
        objective=Stage2ObjectiveConfig(**objective),
        ema_decay=0.9,
        lambda_delta=0.5,
        total_epochs=60,
        adaptation_block_epochs=20,
        amp_enabled=False,
    )


def _empty_stable() -> StableTargetLabelScanResult:
    return StableTargetLabelScanResult(
        candidates=(),
        stable_labels=(),
        num_samples=6,
        num_without_confirmed_phase=6,
        num_candidate_views=0,
        num_classifier_pass=0,
        num_fused_pass=0,
        num_q_pass=0,
        num_stable_labels=0,
        num_ambiguous_rejected=0,
        stable_class_counts=(0, 0, 0),
    )


def _shape_state(status: DomainShapeStatus, delta=None) -> DomainShapeState:
    return DomainShapeState(
        scan_index=0,
        status=status,
        class_centers=(),
        valid_classes=(),
        delta=delta,
        interactions=(),
        rho_shape=None,
        leave_one_out_drift=None,
        center_drift=None,
        confirmation_age=2 if status is DomainShapeStatus.CONFIRMED else 0,
    )


def _phase_state(*, confirmed: bool) -> DomainPhaseState:
    if not confirmed:
        return DomainPhaseState(
            scan_index=0,
            m=0,
            class_centers=(),
            valid_phase_classes=(),
            groups=(),
            rejected_classes=(),
        )
    gamma = torch.tensor([0.0, 0.18, 0.48, 0.78, 1.0])
    group = PhaseGroup(
        group_id=0,
        member_classes=(0, 1, 2),
        center_gamma=gamma,
        within_dispersion=0.0,
        diameter=0.0,
        core_radius=0.0,
        sample_evidence_count=6.0,
        class_count=3,
        center_drift=0.0,
        status=PhaseGroupStatus.CONFIRMED,
        confirmation_age=2,
    )
    return DomainPhaseState(
        scan_index=0,
        m=1,
        class_centers=(),
        valid_phase_classes=(0, 1, 2),
        groups=(group,),
        rejected_classes=(),
    )


class _FakeEMA:
    def __init__(self) -> None:
        self.teacher = object()

    def model(self):
        return self.teacher


class _FakeScheduleTrainer:
    def __init__(self, *, oracle_variant=0) -> None:
        self.config = SimpleNamespace(total_epochs=60, adaptation_block_epochs=20)
        self.ema_teacher = _FakeEMA()
        self.train_epochs = []
        self.saved = []
        self.refresh_epochs = []
        self.diagnostics = []
        self.oracle_writes = []
        self.initialize_calls = 0
        self.current_epoch = 0
        self.oracle_variant = oracle_variant

    def initialize_statistics(self):
        self.initialize_calls += 1

    def train_epoch(self, epoch):
        self.current_epoch = epoch
        self.train_epochs.append(epoch)
        return {"loss": float(epoch)}

    def save_ema_checkpoint(self, filename, *, epoch, target_val):
        self.saved.append((filename, epoch))

    def write_shape_diagnostics(self, epoch, *, suffix=""):
        self.diagnostics.append((epoch, suffix))

    def write_oracle_shape_snapshot(self, epoch):
        self.oracle_writes.append((epoch, self.oracle_variant))
        return {"ignored": self.oracle_variant}

    def refresh_source_features_and_statistics(self):
        self.refresh_epochs.append(self.current_epoch)


def test_exact_60_epoch_schedule_and_checkpoint_selection() -> None:
    trainer = _FakeScheduleTrainer()
    val_calls = []
    test_calls = []

    def val(_teacher, epoch):
        val_calls.append(epoch)
        # unique maximum at 17
        return {"accuracy": 0.5, "macro_f1": 1.0 - abs(epoch - 17) / 100.0}

    def test(_teacher, epoch):
        test_calls.append(epoch)
        # target test deliberately peaks at 40 and must not select the checkpoint
        return {"accuracy": 0.5, "macro_f1": 1.0 if epoch == 40 else 0.0}

    result = run_stage2_training(
        trainer, evaluate_target_val=val, evaluate_target_test=test
    )
    assert trainer.initialize_calls == 1
    assert trainer.train_epochs == list(range(1, 61))
    assert val_calls == list(range(1, 61))
    assert test_calls == [20, 40, 60]
    assert trainer.refresh_epochs == [20, 40, 60]
    assert [item for item in trainer.saved if item[0].startswith("stage2_ema_")] == [
        ("stage2_ema_020.pt", 20),
        ("stage2_ema_040.pt", 40),
        ("stage2_ema_060.pt", 60),
    ]
    assert trainer.saved[-1] == ("stage2_last_ema.pt", 60)
    best_saves = [item for item in trainer.saved if item[0] == "stage2_best_target_val_ema.pt"]
    assert best_saves[-1] == ("stage2_best_target_val_ema.pt", 17)
    assert result.best_target_val_epoch == 17
    assert result.final_diagnostic_target_test["macro_f1"] == 0.0


def test_target_test_and_oracle_outputs_cannot_change_training_trajectory() -> None:
    def run(test_values, oracle_variant):
        trainer = _FakeScheduleTrainer(oracle_variant=oracle_variant)
        val = lambda _teacher, epoch: {"accuracy": 0.0, "macro_f1": epoch / 100.0}
        values = iter(test_values)
        test = lambda _teacher, _epoch: {"accuracy": 0.0, "macro_f1": next(values)}
        result = run_stage2_training(
            trainer, evaluate_target_val=val, evaluate_target_test=test
        )
        return trainer, result

    first, first_result = run([0.1, 0.9, 0.2], 1)
    second, second_result = run([0.9, 0.1, 0.8], 999)
    assert first.train_epochs == second.train_epochs
    assert first.refresh_epochs == second.refresh_epochs
    assert first.saved == second.saved
    assert first_result.best_target_val_epoch == second_result.best_target_val_epoch == 60
    # Oracle target labels are no longer scanned automatically during training.
    assert first.oracle_writes == second.oracle_writes == []


def test_progressive_phase_evidence_budgets_are_nested_and_bounded() -> None:
    trainer = object.__new__(Stage2Trainer)
    trainer.config = SimpleNamespace(
        phase_evidence_initial_samples=64,
        phase_evidence_max_samples=512,
    )
    assert trainer._phase_evidence_budgets(1000) == (64, 128, 256, 512)
    assert trainer._phase_evidence_budgets(300) == (64, 128, 256, 300)
    assert trainer._phase_evidence_budgets(40) == (40,)


def test_phase_only_helper_changes_only_positions() -> None:
    phase = _phase_state(confirmed=True)
    trend = torch.randn(5, 4)
    structure = torch.randn(5, 4)
    q = torch.randn(5, 4)
    support = torch.ones(5)
    positions = torch.linspace(0.0, 1.0, 5)
    mask = torch.ones(5, dtype=torch.bool)
    example = build_phase_only_synthetic_source_example(
        source_sample_id=3,
        class_id=1,
        source_trend_tokens=trend,
        source_structure_tokens=structure,
        source_q_shape=q,
        source_q_support=support,
        source_positions=positions,
        mask=mask,
        phase_state=phase,
    )
    assert example is not None
    torch.testing.assert_close(example.trend_tokens, trend)
    torch.testing.assert_close(example.structure_tokens, structure)
    torch.testing.assert_close(example.q_shape, q)
    assert not torch.equal(example.target_style_positions, positions)


def _real_trainer(shape_status: DomainShapeStatus) -> Stage2Trainer:
    model = _model()
    policy = configure_stage2_parameter_policy(model)
    params = dict(model.named_parameters())
    optimizer = torch.optim.Adam(
        [params[name] for name in policy.trainable_parameter_names], lr=1e-3
    )
    ema = Stage2EMATeacher.from_student(model, policy, decay=0.9)
    bank: SourcePrototypeBank = _bank()
    trainer = Stage2Trainer(
        student=model,
        policy=policy,
        ema_teacher=ema,
        optimizer=optimizer,
        source_loader=[],
        source_scan_loader=[],
        target_statistics_loader=[],
        source_prototype_bank=bank,
        source_registration_bank=None,
        reg_extractor=None,
        config=_trainer_config(),
        device=torch.device("cpu"),
        output_dir="/tmp",
    )
    phase = _phase_state(confirmed=shape_status is not DomainShapeStatus.UNAVAILABLE)
    if shape_status is DomainShapeStatus.UNAVAILABLE:
        phase = _phase_state(confirmed=False)
    delta = torch.zeros(5, 4) if shape_status is DomainShapeStatus.CONFIRMED else None
    trainer.statistics = Stage2StatisticsSnapshot(
        phase_state=phase,
        stable_labels=_empty_stable(),
        shape_state=_shape_state(shape_status, delta),
    )
    return trainer


def test_m0_source_only_step_respects_gradient_boundary() -> None:
    trainer = _real_trainer(DomainShapeStatus.UNAVAILABLE)
    before = {name: p.detach().clone() for name, p in trainer.student.named_parameters()}
    frozen_buffers = {
        name: value.detach().clone()
        for name, value in trainer.student.backbone.named_buffers()
    }
    metrics = trainer.train_step(_batch())
    assert metrics["synthetic_count"] == 0.0
    assert metrics["optimizer_step_succeeded"] == 1.0
    trainable = set(trainer.policy.trainable_parameter_names)
    changed_trainable = []
    for name, parameter in trainer.student.named_parameters():
        changed = not torch.equal(before[name], parameter.detach())
        if name in trainable:
            changed_trainable.append(changed)
        else:
            assert not changed, name
            assert parameter.grad is None
    assert any(changed_trainable)
    for name, value in trainer.student.backbone.named_buffers():
        torch.testing.assert_close(value, frozen_buffers[name])
    assert all(parameter.grad is None for parameter in trainer.ema_teacher.model().parameters())


def test_confirmed_phase_without_shape_generates_phase_only_source() -> None:
    trainer = _real_trainer(DomainShapeStatus.REJECTED)
    trainer.statistics = Stage2StatisticsSnapshot(
        phase_state=_phase_state(confirmed=True),
        stable_labels=_empty_stable(),
        shape_state=_shape_state(DomainShapeStatus.REJECTED),
    )
    metrics = trainer.train_step(_batch())
    assert metrics["synthetic_count"] == pytest.approx(6.0)
    assert metrics["synthetic_cls"] >= 0.0


def test_confirmed_phase_and_shape_runs_round6_synthesis_path() -> None:
    trainer = _real_trainer(DomainShapeStatus.CONFIRMED)
    metrics = trainer.train_step(_batch())
    assert metrics["synthetic_count"] == pytest.approx(6.0)
    assert metrics["synthetic_cls"] >= 0.0
    assert metrics["synthetic_consistency"] >= 0.0


def test_source_fused_refresh_preserves_all_geometry_fields() -> None:
    from methods.structure_da import build_source_prototype_bank, finalize_distance_statistics
    from tests.structure_da.test_stage1_training_helpers import TinySourceDataset

    torch.manual_seed(5)
    model = _model().eval()
    loader = DataLoader(TinySourceDataset(n=24), batch_size=4, shuffle=False)
    bank = build_source_prototype_bank(model, loader, 3, device=torch.device("cpu"))
    bank, _ = finalize_distance_statistics(model, loader, bank, device=torch.device("cpu"))
    refreshed = refresh_source_fused_statistics(
        model, loader, bank, device=torch.device("cpu")
    )
    torch.testing.assert_close(refreshed.trend_srvf, bank.trend_srvf)
    torch.testing.assert_close(refreshed.shape_srvf, bank.shape_srvf)
    torch.testing.assert_close(refreshed.trend_support, bank.trend_support)
    torch.testing.assert_close(refreshed.shape_support, bank.shape_support)
    torch.testing.assert_close(refreshed.q_quantiles, bank.q_quantiles)
    assert refreshed.version == bank.version
    assert all(item.numel() > 0 for item in refreshed.f_distance_samples)


def test_statistics_object_is_not_replaced_inside_a_minibatch() -> None:
    trainer = _real_trainer(DomainShapeStatus.UNAVAILABLE)
    frozen = trainer.statistics
    trainer.train_step(_batch())
    assert trainer.statistics is frozen


def test_amp_skipped_optimizer_step_does_not_update_ema() -> None:
    trainer = _real_trainer(DomainShapeStatus.UNAVAILABLE)
    calls = []
    original_update = trainer.ema_teacher.update_after_optimizer_step

    def counted_update(student):
        calls.append(1)
        original_update(student)

    trainer.ema_teacher.update_after_optimizer_step = counted_update

    class SkippingScaler:
        def __init__(self):
            self.scale_value = 8.0

        def get_scale(self):
            return self.scale_value

        def scale(self, loss):
            return loss

        def step(self, _optimizer):
            return None

        def update(self):
            self.scale_value = 4.0

        def is_enabled(self):
            return True

    trainer.scaler = SkippingScaler()
    metrics = trainer.train_step(_batch())
    assert metrics["optimizer_step_succeeded"] == 0.0
    assert calls == []


def test_oracle_target_labels_are_write_only(tmp_path) -> None:
    trainer = _real_trainer(DomainShapeStatus.REJECTED)
    trainer.statistics = Stage2StatisticsSnapshot(
        phase_state=_phase_state(confirmed=True),
        stable_labels=_empty_stable(),
        shape_state=_shape_state(DomainShapeStatus.REJECTED),
    )
    batch_a = _batch()
    batch_b = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch_a.items()}
    batch_b["label"] = (batch_b["label"] + 1) % 3

    state_before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.student.named_parameters()
    }
    stats_before = trainer.statistics
    bank_before = trainer.source_prototype_bank

    first_dir = tmp_path / "first"
    first_dir.mkdir()
    trainer.output_dir = str(first_dir)
    trainer.target_statistics_loader = [batch_a]
    assert trainer.write_oracle_shape_snapshot(20) is None
    first = torch.load(first_dir / "oracle_target_shape_020.pt", weights_only=False)

    second_dir = tmp_path / "second"
    second_dir.mkdir()
    trainer.output_dir = str(second_dir)
    trainer.target_statistics_loader = [batch_b]
    assert trainer.write_oracle_shape_snapshot(20) is None
    second = torch.load(second_dir / "oracle_target_shape_020.pt", weights_only=False)

    assert trainer.statistics is stats_before
    assert trainer.source_prototype_bank is bank_before
    for name, parameter in trainer.student.named_parameters():
        torch.testing.assert_close(parameter, state_before[name])
    assert first["class_counts"] != second["class_counts"] or any(
        not torch.equal(first["class_centers"][class_id], second["class_centers"].get(class_id, torch.empty(0)))
        for class_id in first["class_centers"]
    )


def test_phase_dp_stops_when_domain_phase_confirms_and_shape_uses_direct_evidence(monkeypatch) -> None:
    import methods.structure_da.stage2_trainer as module

    class FakeScanner:
        total_cached_samples = 512

        def __init__(self):
            self.scan_budgets = []

        def scan_to_budget(self, budget):
            self.scan_budgets.append(budget)
            return TargetHypothesisScanResult(
                hypotheses=(),
                num_samples=budget,
                num_pairwise_attempted=budget,
                num_pre_support_rejected=0,
                num_solver_failed=0,
                num_gamma_rejected=0,
                num_gain_rejected=0,
                num_shape_support_rejected=0,
                num_outer_rejected=0,
                samples_with_zero_hypothesis=budget,
                samples_with_one_hypothesis=0,
                samples_with_two_hypotheses=0,
                num_solver_calls=budget,
                scanned_sample_ids=tuple(range(budget)),
            )

        def sample_ids_for_budget(self, budget):
            return tuple(range(budget))

    scanner = FakeScanner()
    trainer = object.__new__(Stage2Trainer)
    trainer.config = SimpleNamespace(
        phase_evidence_initial_samples=64,
        phase_evidence_max_samples=512,
        phase=object(),
    )
    trainer.source_geometry_version = 0
    trainer.phase_evidence_stages = 0
    trainer.hypothesis_scan_count = 1
    trainer.shape_evidence_stages = 0
    trainer.shape_evidence_sample_ids = ()
    trainer.hypothesis_cache = None
    trainer.statistics = None
    trainer._get_phase_scanner = lambda: scanner

    phase_calls = []

    def fake_phase(result, _config, previous_state=None):
        phase_calls.append(result.num_samples)
        return _phase_state(confirmed=result.num_samples >= 128)

    monkeypatch.setattr(module, "update_domain_phase_state", fake_phase)

    shape_budgets = []

    def fake_stable_shape(phase_state, previous_shape, *, sample_ids):
        assert _confirmed_phase_exists_for_test(phase_state)
        shape_budgets.append(len(sample_ids))
        stable = StableTargetLabelScanResult(
            candidates=(),
            stable_labels=(),
            num_samples=len(sample_ids),
            num_without_confirmed_phase=0,
            num_candidate_views=len(sample_ids),
            num_classifier_pass=0,
            num_fused_pass=0,
            num_q_pass=0,
            num_stable_labels=0,
            num_ambiguous_rejected=0,
            stable_class_counts=(0, 0, 0),
        )
        status = (
            DomainShapeStatus.CONFIRMED
            if len(shape_budgets) >= 2
            else DomainShapeStatus.PROVISIONAL
        )
        return stable, _shape_state(status)

    trainer._stable_and_shape_from_fixed_phase = fake_stable_shape
    snapshot = Stage2Trainer.initialize_statistics(trainer)

    # Exact-DP acquisition stops as soon as Phase is confirmed at 128.
    assert scanner.scan_budgets == [64, 128]
    assert phase_calls == [64, 128]
    # Shape confirmation consumes nested direct confirmed-phase evidence and
    # does not trigger any additional DP stages.
    assert shape_budgets == [64, 128]
    assert snapshot.shape_state.status is DomainShapeStatus.CONFIRMED
    assert trainer.shape_evidence_sample_ids == tuple(range(128))


def _confirmed_phase_exists_for_test(state: DomainPhaseState) -> bool:
    return any(group.status is PhaseGroupStatus.CONFIRMED for group in state.groups)


def test_stage2_checkpoint_contains_full_runtime_statistics_without_feature_snapshots(tmp_path) -> None:
    trainer = _real_trainer(DomainShapeStatus.CONFIRMED)
    trainer.output_dir = str(tmp_path)
    trainer.hypothesis_cache = SimpleNamespace(
        result=TargetHypothesisScanResult(
            hypotheses=(),
            num_samples=3,
            num_pairwise_attempted=3,
            num_pre_support_rejected=0,
            num_solver_failed=0,
            num_gamma_rejected=0,
            num_gain_rejected=0,
            num_shape_support_rejected=0,
            num_outer_rejected=0,
            samples_with_zero_hypothesis=3,
            samples_with_one_hypothesis=0,
            samples_with_two_hypotheses=0,
            scanned_sample_ids=(4, 8, 12),
        )
    )
    trainer.shape_evidence_sample_ids = (4, 8)
    path = trainer.save_ema_checkpoint(
        "stage2_test.pt", epoch=20, target_val={"accuracy": 0.5, "macro_f1": 0.4}
    )
    state = torch.load(path, weights_only=False)

    assert state["phase_state"]["groups"][0]["center_gamma"].device.type == "cpu"
    assert state["domain_shape_state"]["delta"].device.type == "cpu"
    assert state["phase_evidence_sample_ids"] == (4, 8, 12)
    assert state["shape_evidence_sample_ids"] == (4, 8)
    assert "stable_label_state" in state
    assert "source_prototype_bank" in state


def test_statistics_diagnostic_never_trains_or_updates_optimizer() -> None:
    phase = _phase_state(confirmed=False)
    snapshot = SimpleNamespace(
        phase_state=phase,
        stable_labels=SimpleNamespace(num_stable_labels=0),
        shape_state=SimpleNamespace(status=DomainShapeStatus.UNAVAILABLE),
    )

    class DiagnosticTrainer:
        def __init__(self):
            self.successful_optimizer_steps = 0
            self.train_calls = 0
            self.diagnostics = []

        def initialize_statistics(self):
            return snapshot

        def write_shape_diagnostics(self, epoch, *, suffix=""):
            self.diagnostics.append((epoch, suffix))

        def train_epoch(self, _epoch):
            self.train_calls += 1
            raise AssertionError("diagnostic-only mode must not train")

    trainer = DiagnosticTrainer()
    result = run_stage2_statistics_diagnostic(trainer)
    assert result is snapshot
    assert trainer.successful_optimizer_steps == 0
    assert trainer.train_calls == 0
    assert trainer.diagnostics == [(0, "initial")]
