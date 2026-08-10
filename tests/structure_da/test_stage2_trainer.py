from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from methods.structure_da import (
    DomainPhaseConfig,
    DomainPhaseState,
    PhaseDecisionStatus,
    DomainShapeConfig,
    DomainShapeState,
    DomainShapeStatus,
    PhaseGroup,
    PhaseGroupStatus,
    PhaseHypothesisScanConfig,
    SourcePrototypeBank,
    StableLabelConfig,
    StableTargetLabel,
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
        lambda_target=1.0,
        focal_gamma=1.0,
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
            decision_status=PhaseDecisionStatus.UNCONFIRMED,
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
        decision_status=PhaseDecisionStatus.NONIDENTITY_CONFIRMED,
        decision_stability_age=2,
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
        self.stable_refresh_epochs = []
        self.shape_refresh_epochs = []
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

    def refresh_source_features(self):
        self.refresh_epochs.append(self.current_epoch)

    def refresh_stable_labels(self):
        self.stable_refresh_epochs.append(self.current_epoch)

    def refresh_domain_shape(self):
        self.shape_refresh_epochs.append(self.current_epoch)


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
    assert trainer.refresh_epochs == [20, 40]
    assert trainer.shape_refresh_epochs == [20, 40]
    assert trainer.stable_refresh_epochs == list(range(1, 60))
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
    assert first.stable_refresh_epochs == second.stable_refresh_epochs
    assert first.shape_refresh_epochs == second.shape_refresh_epochs
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


def _real_trainer(
    shape_status: DomainShapeStatus,
    *,
    target_time_keep_ratio: float = 0.8,
    with_scheduler: bool = False,
) -> Stage2Trainer:
    model = _model()
    policy = configure_stage2_parameter_policy(model)
    params = dict(model.named_parameters())
    optimizer = torch.optim.Adam(
        [params[name] for name in policy.trainable_parameter_names], lr=1e-3
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0)
        if with_scheduler
        else None
    )
    ema = Stage2EMATeacher.from_student(model, policy, decay=0.9)
    bank: SourcePrototypeBank = _bank()
    trainer = Stage2Trainer(
        student=model,
        policy=policy,
        ema_teacher=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        source_loader=[],
        source_scan_loader=[],
        target_statistics_loader=[],
        source_prototype_bank=bank,
        source_registration_bank=None,
        reg_extractor=None,
        config=replace(
            _trainer_config(),
            target_time_keep_ratio=target_time_keep_ratio,
        ),
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


def test_unconfirmed_phase_forbids_source_only_stage2_optimizer_step() -> None:
    trainer = _real_trainer(DomainShapeStatus.UNAVAILABLE)
    before = {name: p.detach().clone() for name, p in trainer.student.named_parameters()}
    with pytest.raises(RuntimeError, match="forbidden without confirmed adaptation evidence"):
        trainer.train_step(_batch())
    for name, parameter in trainer.student.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name])
    assert trainer.successful_optimizer_steps == 0


def test_confirmed_phase_without_shape_generates_phase_only_source() -> None:
    trainer = _real_trainer(DomainShapeStatus.REJECTED)
    trainer.statistics = Stage2StatisticsSnapshot(
        phase_state=_phase_state(confirmed=True),
        stable_labels=_empty_stable(),
        shape_state=_shape_state(DomainShapeStatus.REJECTED),
    )
    metrics = trainer.train_step(_batch())
    assert metrics["source_count"] == pytest.approx(6.0)
    assert metrics["source_to_target"] >= 0.0


def test_confirmed_phase_and_shape_runs_round6_synthesis_path() -> None:
    trainer = _real_trainer(DomainShapeStatus.CONFIRMED)
    metrics = trainer.train_step(_batch())
    assert metrics["source_count"] == pytest.approx(6.0)
    assert metrics["source_to_target"] >= 0.0


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
    trainer = _real_trainer(DomainShapeStatus.REJECTED)
    frozen = trainer.statistics
    trainer.train_step(_batch())
    assert trainer.statistics is frozen


def test_amp_skipped_optimizer_step_does_not_update_ema() -> None:
    trainer = _real_trainer(DomainShapeStatus.REJECTED)
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


def test_stable_label_refresh_uses_full_target_loader_not_phase_dp_subset(monkeypatch) -> None:
    import methods.structure_da.stage2_trainer as module

    trainer = object.__new__(Stage2Trainer)
    trainer.ema_teacher = SimpleNamespace(model=lambda: object())
    trainer.target_statistics_loader = object()
    trainer.target_stable_label_loader = object()
    trainer.source_prototype_bank = _bank()
    trainer.config = SimpleNamespace(stable_labels=object())
    trainer.stable_label_refresh_count = 0
    observed = []

    def fake_scan(**kwargs):
        observed.append(kwargs["target_loader"])
        return _empty_stable()

    monkeypatch.setattr(module, "scan_stable_target_labels_from_confirmed_phase", fake_scan)
    result = Stage2Trainer._stable_from_fixed_phase(trainer, _phase_state(confirmed=True))

    assert result.num_stable_labels == 0
    assert observed == [trainer.target_stable_label_loader]
    assert observed[0] is not trainer.target_statistics_loader
    assert trainer.stable_label_refresh_count == 1


def test_phase_calibration_finishes_before_single_initial_label_and_shape_refresh(monkeypatch) -> None:
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
    trainer.source_prototype_bank = _bank()
    trainer.phase_evidence_stages = 0
    trainer.hypothesis_scan_count = 1
    trainer.shape_evidence_stages = 0
    trainer.stable_label_refresh_count = 0
    trainer.shape_evidence_sample_ids = ()
    trainer.hypothesis_cache = None
    trainer.statistics = None
    trainer._get_phase_scanner = lambda: scanner

    phase_calls = []

    def fake_phase(result, _config, previous_state=None):
        phase_calls.append(result.num_samples)
        return _phase_state(confirmed=result.num_samples >= 128)

    monkeypatch.setattr(module, "update_domain_phase_state", fake_phase)

    label_refreshes = []

    def fake_stable(phase_state):
        assert _confirmed_phase_exists_for_test(phase_state)
        label_refreshes.append(phase_state.scan_index)
        trainer.stable_label_refresh_count += 1
        return StableTargetLabelScanResult(
            candidates=(),
            stable_labels=(),
            num_samples=2048,
            num_without_confirmed_phase=0,
            num_candidate_views=2048,
            num_classifier_pass=0,
            num_fused_pass=0,
            num_q_pass=0,
            num_stable_labels=0,
            num_ambiguous_rejected=0,
            stable_class_counts=(0, 0, 0),
        )

    def fake_shape(stable_result, previous_shape):
        trainer.shape_evidence_stages += 1
        return _shape_state(DomainShapeStatus.PROVISIONAL)

    trainer._stable_from_fixed_phase = fake_stable
    trainer._shape_from_stable = fake_shape
    snapshot = Stage2Trainer.initialize_statistics(trainer)

    # Domain Phase model order is settled from all configured nested evidence
    # budgets before adaptation begins; it is not refreshed during training.
    assert scanner.scan_budgets == [64, 128, 256, 512]
    assert phase_calls == [64, 128, 256, 512]
    # Stable Label and Domain Shape initialize once from the final fixed Phase.
    assert len(label_refreshes) == 1
    assert snapshot.shape_state.status is DomainShapeStatus.PROVISIONAL
    # Stable-Label/Shape evidence is no longer tied to the 512-sample DP budget.
    assert trainer.shape_evidence_sample_ids == ()

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


def _identity_phase_state_for_trainer() -> DomainPhaseState:
    return DomainPhaseState(
        scan_index=1,
        m=0,
        class_centers=(),
        valid_phase_classes=(0, 1, 2),
        groups=(),
        rejected_classes=(),
        decision_status=PhaseDecisionStatus.IDENTITY_CONFIRMED,
        decision_stability_age=2,
        identity_evidence_classes=(0, 1, 2),
        identity_evidence_count=6.0,
    )


def test_identity_confirmed_and_shape_confirmed_runs_shape_only_synthesis() -> None:
    trainer = _real_trainer(DomainShapeStatus.CONFIRMED)
    trainer.statistics = Stage2StatisticsSnapshot(
        phase_state=_identity_phase_state_for_trainer(),
        stable_labels=_empty_stable(),
        shape_state=_shape_state(DomainShapeStatus.CONFIRMED, torch.zeros(5, 4)),
    )
    metrics = trainer.train_step(_batch())
    assert metrics["source_count"] == pytest.approx(6.0)
    assert metrics["optimizer_step_succeeded"] == 1.0


class _AbstainScheduleTrainer:
    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self.config = SimpleNamespace(total_epochs=60, adaptation_block_epochs=20)
        self.ema_teacher = _FakeEMA()
        self.successful_optimizer_steps = 0
        self.train_epochs = []
        self.saved = []
        self.diagnostics = []

    def initialize_statistics(self):
        return self.snapshot

    def train_epoch(self, epoch):
        self.train_epochs.append(epoch)
        raise AssertionError("abstain path must not execute any Stage-2 optimizer epoch")

    def write_shape_diagnostics(self, epoch, *, suffix=""):
        self.diagnostics.append((epoch, suffix))

    def save_ema_checkpoint(self, filename, *, epoch, target_val):
        self.saved.append((filename, epoch, target_val))


def test_unconfirmed_phase_returns_stage1_model_without_stage2_optimizer_steps() -> None:
    snapshot = Stage2StatisticsSnapshot(
        phase_state=_phase_state(confirmed=False),
        stable_labels=_empty_stable(),
        shape_state=_shape_state(DomainShapeStatus.UNAVAILABLE),
    )
    trainer = _AbstainScheduleTrainer(snapshot)
    val_calls = []
    test_calls = []
    result = run_stage2_training(
        trainer,
        evaluate_target_val=lambda _teacher, epoch: val_calls.append(epoch) or {},
        evaluate_target_test=lambda _teacher, epoch: test_calls.append(epoch) or {
            "accuracy": 0.5,
            "macro_f1": 0.4,
        },
    )
    assert trainer.train_epochs == []
    assert trainer.successful_optimizer_steps == 0
    assert val_calls == []
    assert test_calls == [0]
    assert trainer.saved == [("stage2_last_ema.pt", 0, None)]
    assert trainer.diagnostics == [(0, "no_adaptation")]
    assert result.adaptation_performed is False
    assert result.abstain_reason == "phase_unconfirmed"
    assert result.best_target_val_epoch is None


def test_identity_confirmed_without_shape_also_abstains() -> None:
    snapshot = Stage2StatisticsSnapshot(
        phase_state=_identity_phase_state_for_trainer(),
        stable_labels=_empty_stable(),
        shape_state=_shape_state(DomainShapeStatus.REJECTED),
    )
    trainer = _AbstainScheduleTrainer(snapshot)
    result = run_stage2_training(
        trainer,
        evaluate_target_val=lambda _teacher, _epoch: (_ for _ in ()).throw(
            AssertionError("target val must not select a no-adaptation checkpoint")
        ),
        evaluate_target_test=lambda _teacher, epoch: {
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "epoch": epoch,
        },
    )
    assert trainer.train_epochs == []
    assert result.adaptation_performed is False
    assert result.abstain_reason == "confirmed_phase_without_actionable_target_or_shape_evidence"
    assert result.final_diagnostic_target_test["epoch"] == 0



def _stable_result_with_labels(labels: list[tuple[int, int, int, float]]) -> StableTargetLabelScanResult:
    items = tuple(
        StableTargetLabel(
            sample_id=sample_id,
            class_id=class_id,
            group_id=group_id,
            aligned_q_shape=torch.zeros(5, 4),
            aligned_q_support=torch.ones(5),
            fused_repr=torch.zeros(8),
            confidence_summary=confidence,
        )
        for sample_id, class_id, group_id, confidence in labels
    )
    counts = [0, 0, 0]
    for item in items:
        counts[item.class_id] += 1
    return StableTargetLabelScanResult(
        candidates=(),
        stable_labels=items,
        num_samples=6,
        num_without_confirmed_phase=0,
        num_candidate_views=len(items),
        num_classifier_pass=len(items),
        num_fused_pass=len(items),
        num_q_pass=len(items),
        num_stable_labels=len(items),
        num_ambiguous_rejected=0,
        stable_class_counts=tuple(counts),
    )


def test_student_native_target_uses_stable_labels_not_dataset_truth() -> None:
    trainer = _real_trainer(DomainShapeStatus.REJECTED)
    stable = _stable_result_with_labels([(0, 2, 0, 0.9), (3, 1, 0, 0.8)])
    trainer.statistics = Stage2StatisticsSnapshot(
        phase_state=_phase_state(confirmed=True),
        stable_labels=stable,
        shape_state=_shape_state(DomainShapeStatus.REJECTED),
        phase_routes=(0, 0, 0),
    )
    batch = _batch()
    batch["index"] = torch.arange(6)
    batch["label"] = torch.zeros(6, dtype=torch.long)  # oracle labels must be ignored
    logits, labels = trainer._target_forward_native(batch)
    assert logits is not None and logits.shape[0] == 2
    assert labels is not None
    assert labels.tolist() == [2, 1]


def test_student_strong_target_mask_preserves_native_axis_and_masks_time_steps() -> None:
    trainer = _real_trainer(
        DomainShapeStatus.REJECTED,
        target_time_keep_ratio=0.6,
    )
    base = torch.ones(3, 10, dtype=torch.bool)
    torch.manual_seed(7)
    strong = trainer._strong_native_target_time_mask(base)
    assert strong.dtype is torch.bool
    assert strong.shape == base.shape
    assert strong.sum(dim=1).tolist() == [6, 6, 6]
    assert torch.all(strong <= base)


def test_stage2_scheduler_steps_after_successful_optimizer_step() -> None:
    trainer = _real_trainer(
        DomainShapeStatus.REJECTED,
        with_scheduler=True,
    )
    initial_lr = float(trainer.optimizer.param_groups[0]["lr"])
    trainer.train_step(_batch())
    assert float(trainer.optimizer.param_groups[0]["lr"]) < initial_lr


def _m2_phase_state() -> DomainPhaseState:
    grid = torch.linspace(0.0, 1.0, 5)
    group0 = PhaseGroup(
        group_id=0,
        member_classes=(0,),
        center_gamma=grid.square(),
        within_dispersion=0.0,
        diameter=0.0,
        core_radius=0.0,
        sample_evidence_count=4.0,
        class_count=1,
        center_drift=0.0,
        status=PhaseGroupStatus.CONFIRMED,
        confirmation_age=2,
    )
    group1 = PhaseGroup(
        group_id=1,
        member_classes=(1,),
        center_gamma=torch.sqrt(grid),
        within_dispersion=0.0,
        diameter=0.0,
        core_radius=0.0,
        sample_evidence_count=4.0,
        class_count=1,
        center_drift=0.0,
        status=PhaseGroupStatus.CONFIRMED,
        confirmation_age=2,
    )
    return DomainPhaseState(
        scan_index=1,
        m=2,
        class_centers=(),
        valid_phase_classes=(0, 1),
        groups=(group0, group1),
        rejected_classes=(2,),
        decision_status=PhaseDecisionStatus.NONIDENTITY_CONFIRMED,
        decision_stability_age=2,
    )


def test_m1_confirmed_phase_routes_all_classes_not_only_founding_members() -> None:
    from methods.structure_da.stage2_trainer import _derive_phase_routes

    phase = _phase_state(confirmed=True)
    # Pretend only class 0 founded the group; M=1 must still serve every class.
    group = phase.groups[0]
    phase = DomainPhaseState(
        **{
            **phase.__dict__,
            "groups": (PhaseGroup(**{**group.__dict__, "member_classes": (0,)}),),
        }
    )
    assert _derive_phase_routes(phase, _empty_stable(), 3) == (0, 0, 0)


def test_m2_nonfounding_class_routes_from_stable_group_evidence() -> None:
    from methods.structure_da.stage2_trainer import _derive_phase_routes

    stable = _stable_result_with_labels([
        (0, 2, 1, 0.9),
        (1, 2, 1, 0.8),
        (2, 2, 0, 0.2),
    ])
    assert _derive_phase_routes(_m2_phase_state(), stable, 3) == (0, 1, 1)


def test_m2_ambiguous_nonfounding_class_remains_unrouted() -> None:
    from methods.structure_da.stage2_trainer import _derive_phase_routes

    stable = _stable_result_with_labels([
        (0, 2, 0, 0.8),
        (1, 2, 1, 0.8),
    ])
    assert _derive_phase_routes(_m2_phase_state(), stable, 3) == (0, 1, None)
