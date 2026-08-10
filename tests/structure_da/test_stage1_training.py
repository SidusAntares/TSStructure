from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from methods.structure_da import SourceClassificationTrainer, build_source_prototype_bank
from tests.structure_da.test_stage1_training_helpers import TinySourceDataset, _batch, _model


def _trainer(model):
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    return SourceClassificationTrainer(model,optimizer,device=torch.device('cpu'),amp_enabled=False)


def test_stage1_step_is_source_ce_only() -> None:
    model=_model(); trainer=_trainer(model)
    metrics=trainer.train_step(_batch())
    assert metrics['loss']==metrics['classification_loss']
    assert metrics['q_proto_loss']==metrics['f_proto_loss']==metrics['q_to_cls_loss']==0.0


def test_stage1_ce_does_not_execute_decomposition_or_geometry(monkeypatch) -> None:
    model=_model(); trainer=_trainer(model)
    decomp_calls=geometry_calls=0
    original_decomp=model.backbone.decomposition.forward
    original_geo=model.temporal_module.trend_geometry.forward
    def decomp(*args,**kwargs):
        nonlocal decomp_calls; decomp_calls+=1; return original_decomp(*args,**kwargs)
    def geo(*args,**kwargs):
        nonlocal geometry_calls; geometry_calls+=1; return original_geo(*args,**kwargs)
    monkeypatch.setattr(model.backbone.decomposition,'forward',decomp)
    monkeypatch.setattr(model.temporal_module.trend_geometry,'forward',geo)
    trainer.train_step(_batch())
    assert decomp_calls==0 and geometry_calls==0


def test_full_source_geometry_scan_is_separate_from_training() -> None:
    model=_model().eval(); dataset=TinySourceDataset(n=24)
    loader=DataLoader(dataset,batch_size=4,shuffle=False,drop_last=False)
    bank=build_source_prototype_bank(model,loader,3,device=torch.device('cpu'))
    assert bank.class_counts.tolist()==[8,8,8]
    assert bank.fused.shape==(3,4)
    assert bank.ready.tolist()==[True,True,True]


def test_stage1_trainer_requires_no_target_or_prototype_inputs() -> None:
    import inspect
    sig=inspect.signature(SourceClassificationTrainer.train_step)
    assert 'target_batch' not in sig.parameters
    model=_model(); trainer=_trainer(model)
    # compatibility kwargs are ignored and cannot alter CE semantics
    metrics=trainer.train_step(_batch(),warmup=False,bank=None)
    assert metrics['loss']==metrics['classification_loss']


def test_train_source_classification_declares_phase_only_protocol() -> None:
    pytest.importorskip('zarr')
    from pathlib import Path
    text=Path('train.py').read_text(encoding='utf-8')
    assert 'classification_path=pse_single_ltae' in text
    assert 'objective=source_ce' in text
    assert 'geometry_scan=checkpoint_finalize_only' in text
    assert 'Stage1Objective' not in text
