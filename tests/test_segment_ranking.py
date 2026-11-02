from types import SimpleNamespace

import pytest
import torch


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_encoder = SimpleNamespace(mlp=torch.nn.Identity(), dropout=torch.nn.Identity())
        self.decoder = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.decoder.weight.copy_(torch.tensor([[1.], [-1.]]))


@pytest.mark.parametrize('correct_order', [True, False])
def test_ranking_compares_nontrivial_subsets_and_deletion_importance(correct_order):
    from utils.segment_ranking import batch_ranking_diagnostic
    model = Classifier().eval()
    probabilities = torch.tensor([[[.9, .2, .1] if correct_order else [.1, .2, .9]]])
    info = dict(probabilities=probabilities, candidate_count=torch.tensor([3]),
        segments=torch.tensor([[0, 0, 1, 1, 2, 2]]),
        attention=torch.ones(1, 1, 6)/6,
        values=torch.tensor([[[[2.], [2.], [0.], [0.], [0.], [0.]]]]))
    torch.manual_seed(7)
    rows, deletions = batch_ranking_diagnostic(model, info, torch.tensor([0]), 1.)
    assert len(rows) == 2 and len(deletions) == 3
    assert all(0 < row['kept_segments'] < 3 for row in rows)
    assert deletions[0]['deletion_loss_delta'] > deletions[1]['deletion_loss_delta']
    assert all(row['loss_gain_vs_random'] > 0 if correct_order else row['loss_gain_vs_random'] < 0 for row in rows)
    assert all(row['deletion_spearman'] > .8 if correct_order else row['deletion_spearman'] < -.8 for row in rows)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_single_candidate_is_marked_unidentifiable_by_exclusion():
    from utils.segment_ranking import batch_ranking_diagnostic
    info = dict(probabilities=torch.ones(1, 1, 1), candidate_count=torch.tensor([1]),
        segments=torch.zeros(1, 6, dtype=torch.long), attention=torch.ones(1, 1, 6)/6,
        values=torch.ones(1, 1, 6, 1))
    assert batch_ranking_diagnostic(Classifier().eval(), info, torch.tensor([0]), 1.) == ([], [])
