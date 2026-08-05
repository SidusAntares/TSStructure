import pytest
import torch

from train import load_structure_da_state_dict


def test_legacy_global_alignment_checkpoint_has_clear_incompatibility_error() -> None:
    model = torch.nn.Linear(2, 2)
    state = model.state_dict()
    state["alignment.discriminator.network.0.weight"] = torch.zeros(2, 2)

    with pytest.raises(RuntimeError, match="removed global fused-feature domain alignment"):
        load_structure_da_state_dict(model, state)


def test_current_checkpoint_loads_strictly() -> None:
    model = torch.nn.Linear(2, 2)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    load_structure_da_state_dict(model, state)
