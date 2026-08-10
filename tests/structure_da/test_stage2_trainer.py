from __future__ import annotations

from types import SimpleNamespace

import torch

from methods.structure_da import (
    DomainPhaseState,
    PhaseDecisionStatus,
    PhaseGroup,
    PhaseGroupStatus,
    StableTargetLabel,
    StableTargetLabelScanResult,
    Stage2StatisticsSnapshot,
    Stage2Trainer,
    Stage2TrainerConfig,
    Stage2ObjectiveConfig,
    DomainPhaseConfig,
    StableLabelConfig,
    PhaseHypothesisScanConfig,
    run_stage2_training,
)
from methods.structure_da.stage2_trainer import _derive_phase_routes


def _config(**overrides) -> Stage2TrainerConfig:
    values=dict(
        phase_scan=PhaseHypothesisScanConfig(
            registration_lambda=0.0,registration_gain_ratio_max=1.0,
            registration_min_common_support=0.0,registration_max_roughness=1e9,
            registration_min_increment=0.0,registration_max_local_speed=100.0,
            registration_max_deviation=1.0,class_hypothesis_margin=0.0,
        ),
        phase=DomainPhaseConfig(
            phase_min_samples_per_class=1.0,phase_class_dispersion_max=1.0,
            phase_class_diameter_max=1.0,phase_group_dispersion_max=1.0,
            phase_group_diameter_max=1.0,phase_group_core_separation=0.1,
            phase_global_radius=1.0,phase_confirmation_patience=2,
            phase_center_drift_max=1.0,
        ),
        stable_labels=StableLabelConfig(
            tau_f=0.1,tau_q=0.1,cls_confidence_min=0.0,cls_margin_min=None,
            fused_confidence_min=0.0,fused_margin_min=None,q_confidence_min=0.0,q_margin_min=None,
        ),
        objective=Stage2ObjectiveConfig(lambda_target=1.0,focal_gamma=1.0),
        ema_decay=0.9999,total_epochs=3,
        amp_enabled=False,target_time_keep_ratio=0.8,
    )
    values.update(overrides)
    return Stage2TrainerConfig(**values)


def _group(group_id:int, classes:tuple[int,...], gamma=None) -> PhaseGroup:
    if gamma is None:
        gamma=torch.tensor([0.0,0.2,0.5,0.8,1.0],dtype=torch.float64)
    return PhaseGroup(
        group_id=group_id,member_classes=classes,center_gamma=gamma,
        within_dispersion=0.0,diameter=0.0,core_radius=0.0,
        sample_evidence_count=10.0,class_count=len(classes),center_drift=0.0,
        status=PhaseGroupStatus.CONFIRMED,confirmation_age=2,
    )


def _phase(groups:tuple[PhaseGroup,...]) -> DomainPhaseState:
    return DomainPhaseState(
        scan_index=3,m=len(groups),class_centers=(),
        valid_phase_classes=tuple(sorted({c for g in groups for c in g.member_classes})),
        groups=groups,rejected_classes=(),
        decision_status=PhaseDecisionStatus.NONIDENTITY_CONFIRMED,
        decision_stability_age=2,
    )


def _stable(labels:tuple[StableTargetLabel,...]=(), num_classes:int=4) -> StableTargetLabelScanResult:
    counts=[0]*num_classes
    for item in labels: counts[item.class_id]+=1
    return StableTargetLabelScanResult(
        candidates=(),stable_labels=labels,num_samples=10,num_without_confirmed_phase=0,
        num_candidate_views=len(labels),num_classifier_pass=len(labels),
        num_fused_pass=len(labels),num_q_pass=len(labels),num_stable_labels=len(labels),
        num_ambiguous_rejected=0,stable_class_counts=tuple(counts),
    )


def _label(sample:int, cls:int, group:int, confidence:float=1.0) -> StableTargetLabel:
    return StableTargetLabel(
        sample_id=sample,class_id=cls,group_id=group,
        aligned_q_shape=torch.zeros(5,2),aligned_q_support=torch.ones(5),
        fused_repr=torch.zeros(4),confidence_summary=confidence,
    )


def test_stage2_config_has_no_domain_shape_state() -> None:
    cfg=_config()
    assert not hasattr(cfg,'shape')
    assert not hasattr(cfg,'lambda_delta')


def test_single_confirmed_phase_is_available_to_all_classes() -> None:
    phase=_phase((_group(0,(0,1)),))
    routes=_derive_phase_routes(phase,_stable(num_classes=4),4)
    assert routes==(0,0,0,0)


def test_m2_routes_founders_then_expands_from_stable_labels() -> None:
    g0=_group(0,(0,1),torch.tensor([0.,.15,.45,.75,1.],dtype=torch.float64))
    g1=_group(1,(2,),torch.tensor([0.,.25,.60,.85,1.],dtype=torch.float64))
    stable=_stable((_label(1,3,1,2.0),_label(2,3,0,0.5)),4)
    routes=_derive_phase_routes(_phase((g0,g1)),stable,4)
    assert routes==(0,0,1,1)


def test_unresolved_source_route_keeps_native_positions() -> None:
    trainer=Stage2Trainer.__new__(Stage2Trainer)
    phase=_phase((_group(0,(0,)),_group(1,(1,),torch.tensor([0.,.3,.6,.9,1.],dtype=torch.float64))))
    trainer.statistics=Stage2StatisticsSnapshot(phase,_stable(num_classes=3),(0,1,None))
    positions=torch.tensor([[0.,.25,.5,.75,1.],[0.,.25,.5,.75,1.]])
    mask=torch.ones_like(positions,dtype=torch.bool)
    labels=torch.tensor([0,2])
    mapped=trainer._source_phase_positions(positions,mask,labels)
    assert not torch.equal(mapped[0],positions[0])
    torch.testing.assert_close(mapped[1],positions[1])


def test_strong_target_mask_only_removes_native_acquisitions() -> None:
    trainer=Stage2Trainer.__new__(Stage2Trainer)
    trainer.device=torch.device('cpu'); trainer.config=SimpleNamespace(target_time_keep_ratio=0.5)
    torch.manual_seed(0)
    base=torch.tensor([[True,True,True,True,False],[True,False,True,False,True]])
    strong=trainer._strong_native_target_time_mask(base)
    assert torch.all(~strong | base)
    assert strong.sum(dim=1).tolist()==[2,2]


class _EMA:
    def model(self): return object()


class _ScheduleTrainer:
    def __init__(self):
        self.config=SimpleNamespace(total_epochs=3)
        self.ema_teacher=_EMA(); self.saved=[]; self.stable=[]; self.trained=[]
        phase=_phase((_group(0,(0,1)),))
        self.snapshot=Stage2StatisticsSnapshot(phase,_stable((_label(1,0,0),),2),(0,0))
    def initialize_statistics(self): return self.snapshot
    def refresh_stable_labels(self): self.stable.append(len(self.trained)); return self.snapshot
    def train_epoch(self,epoch): self.trained.append(epoch); return {'loss':0.0}
    def save_ema_checkpoint(self,name,*,epoch,target_val): self.saved.append((name,epoch)); return name


def test_training_schedule_has_no_domain_shape_refresh() -> None:
    trainer=_ScheduleTrainer()
    vals=[]
    result=run_stage2_training(
        trainer,
        evaluate_target_val=lambda _m,e: vals.append(e) or {'accuracy':0.5,'macro_f1':float(e)},
        evaluate_target_test=lambda _m,e: {'accuracy':0.5,'macro_f1':0.5},
    )
    assert trainer.trained==[1,2,3]
    assert trainer.stable==[1,2]
    assert not hasattr(trainer,'refresh_domain_shape')
    assert not hasattr(trainer,'refresh_source_features')
    assert result.best_target_val_epoch==3
