import numpy as np
import pytest
import torch

from analysis.local_structure_query_audit import (
    attention_similarity,
    audit_checkpoint_path,
    build_audit_rows,
)


def test_attention_similarity_uses_real_local_and_base_maps():
    local = torch.tensor([[[[1., 0.], [0., 1.]]]])
    base = torch.tensor([[[[1., 0.]]]])
    result = attention_similarity(local, base)
    assert result["local_local_attention_similarity"] == pytest.approx(0.)
    assert result["local_base_attention_similarity"] == pytest.approx(.5)


def test_audit_rows_contain_only_three_representations_and_fixed_metrics():
    metrics = {
        name: {
            "source_val_macro_f1": .1,
            "target_oracle_macro_f1": .2,
            "source_to_target_macro_f1": .3,
            "source_to_target_per_class_f1": np.array([.4, .5]),
        }
        for name in ("E_flat", "z_Q", "z_T")
    }
    rows = build_audit_rows(
        "FR2_DK1", ["corn", "horsebeans"], metrics,
        {"source_val": {"local_local_attention_similarity": .7,
                        "local_base_attention_similarity": .8},
         "target_val": {"local_local_attention_similarity": .6,
                        "local_base_attention_similarity": .5}},
    )
    macro = [row for row in rows if row["scope"] == "macro"]
    hard = [row for row in rows if row["scope"] == "per_class"]
    assert [row["representation"] for row in macro] == ["E_flat", "z_Q", "z_T"]
    assert {row["class"] for row in hard} == {"horsebeans"}
    assert macro[0]["source_val_macro_f1"] == .1
    assert macro[0]["target_oracle_macro_f1"] == .2
    assert macro[0]["source_to_target_macro_f1"] == .3


def test_uq_checkpoint_path_is_best_model_and_task_specific(tmp_path):
    path = audit_checkpoint_path(tmp_path, "AT1_DK1")
    assert path == tmp_path / "uq_AT1_DK1_seed1" / "fold_0" / "model.pt"
