from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from methods.structure_da import SourceClassificationTrainer, build_source_prototype_bank, finalize_distance_statistics
from tests.structure_da.test_stage1_training_helpers import TinySourceDataset, _model


def test_checkpoint_geometry_bank_matches_single_stream_task_dimension() -> None:
    model=_model().eval(); dataset=TinySourceDataset(n=24)
    loader=DataLoader(dataset,batch_size=4,shuffle=False,drop_last=False)
    bank=build_source_prototype_bank(model,loader,3,device=torch.device('cpu'))
    final,examples=finalize_distance_statistics(model,loader,bank,device=torch.device('cpu'))
    assert final.shape_srvf.shape==(3,5,4)
    assert final.fused.shape==(3,4)
    assert final.ready.tolist()==[True,True,True]
    assert all(len([e for e in examples if e['class_id']==c])<=3 for c in range(3))


def test_prototype_scan_is_deterministic_for_selected_stage1_state() -> None:
    model=_model().eval(); dataset=TinySourceDataset(n=24)
    loader=DataLoader(dataset,batch_size=4,shuffle=False,drop_last=False)
    a=build_source_prototype_bank(model,loader,3,device=torch.device('cpu'))
    b=build_source_prototype_bank(model,loader,3,device=torch.device('cpu'))
    torch.testing.assert_close(a.shape_srvf,b.shape_srvf,rtol=0,atol=0)
    torch.testing.assert_close(a.fused,b.fused,rtol=0,atol=0)


def test_stage1_ce_then_geometry_finalize_smoke() -> None:
    model=_model(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    trainer=SourceClassificationTrainer(model,optimizer,device=torch.device('cpu'),amp_enabled=False)
    dataset=TinySourceDataset(n=12); loader=DataLoader(dataset,batch_size=4,shuffle=False)
    for batch in loader:
        metrics=trainer.train_step(batch)
        assert torch.isfinite(torch.tensor(metrics['loss'])).item()
    bank=build_source_prototype_bank(model.eval(),loader,3,device=torch.device('cpu'))
    assert bank.ready.tolist()==[True,True,True]
