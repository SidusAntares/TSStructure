import torch

from models.ltae import MultiHeadAttention
from utils.head_evidence import (
    SourceEvidenceTracker,
    build_counterfactual_head_embeddings,
    relative_logit_contributions,
    select_attention_intervals,
)


def test_shortest_interval_uses_actual_dates_and_top_weight_mass():
    attention = torch.tensor([[[0.34, 0.32, 0.18, 0.10, 0.04, 0.02]]])
    positions = torch.tensor([[10, 12, 100, 150, 180, 200]])
    inside, valid = select_attention_intervals(
        attention, positions, top_fraction=0.5, coverage=0.75,
        min_observations=2, min_outside=2,
    )
    assert valid.tolist() == [[True]]
    assert inside[0, 0].tolist() == [True, True, False, False, False, False]


def test_ltae_head_parts_preserve_original_pooling_in_eval_mode():
    torch.manual_seed(3)
    module = MultiHeadAttention(n_head=2, d_k=2, d_in=4).eval()
    inputs = torch.randn(3, 5, 4)
    pooled, dropped, attention, values = module(inputs, return_parts=True)
    expected = (attention.unsqueeze(-2) @ values).transpose(1, 2).reshape(3, 4)
    assert torch.allclose(pooled, expected, atol=1e-6)
    assert torch.allclose(dropped.squeeze(-2), attention)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(3, 2))


def test_insufficient_observed_points_skips_head_instead_of_widening_interval():
    attention = torch.tensor([[[0.7, 0.1, 0.1, 0.05, 0.05]]])
    positions = torch.tensor([[10, 20, 30, 40, 50]])
    inside, valid = select_attention_intervals(
        attention, positions, top_fraction=0.2, coverage=0.75,
        min_observations=3, min_outside=2,
    )
    assert valid.tolist() == [[False]]
    assert not inside.any()


def test_counterfactual_replaces_only_the_selected_head_and_backpropagates():
    attention = torch.tensor([[[0.5, 0.3, 0.1, 0.1],
                               [0.1, 0.1, 0.3, 0.5]]])
    values = torch.arange(1, 9, dtype=torch.float32).view(1, 2, 4, 1)
    values.requires_grad_()
    inside = torch.tensor([[[True, True, False, False],
                            [False, False, True, True]]])
    candidates = build_counterfactual_head_embeddings(attention, values, inside)
    assert candidates.shape == (1, 3, 2)
    assert torch.allclose(candidates[0, 1, 1], candidates[0, 0, 1])
    assert torch.allclose(candidates[0, 2, 0], candidates[0, 0, 0])
    assert not torch.allclose(candidates[0, 1, 0], candidates[0, 0, 0])
    candidates.sum().backward()
    assert torch.isfinite(values.grad).all()


def test_relative_logit_contribution_ignores_common_class_logit_shift():
    # Replacing a head may decrease every class logit equally without
    # changing classification. This must contribute zero.
    logits = torch.tensor([[[3.0, 2.0], [2.0, 1.0], [2.5, 2.0]]])
    scores = relative_logit_contributions(logits, torch.tensor([0]))
    assert torch.allclose(scores, torch.tensor([[0.0, 0.5]]))


def test_source_evidence_gates_positive_and_ranking_without_target_labels():
    tracker = SourceEvidenceTracker(
        num_classes=2, num_heads=3, device="cpu",
        min_count=2, min_positive_rate=0.5,
        min_source_contribution=0.05, rank_gap=0.2,
    )
    tracker.update(
        torch.tensor([0, 0]),
        torch.tensor([[1.5, 0.6, -0.1], [1.4, 0.5, -0.2]]),
        torch.ones(2, 3, dtype=torch.bool),
    )
    target = torch.tensor([[0.0, 0.3, 2.0]], requires_grad=True)
    positive, ranking, active = tracker.losses(
        target, torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([0]), torch.tensor([1.0]),
        positive_margin=0.1, ranking_margin=0.1,
    )
    assert active == 2
    assert positive.item() > 0
    assert ranking.item() > 0
    (positive + ranking).backward()
    assert torch.isfinite(target.grad).all()


def test_source_gate_skips_unseen_classes_and_invalid_target_heads():
    tracker = SourceEvidenceTracker(
        num_classes=2, num_heads=2, device="cpu", min_count=1,
        min_positive_rate=0.5, min_source_contribution=0.05,
        rank_gap=0.1,
    )
    tracker.update(torch.tensor([0]), torch.tensor([[1.0, 0.5]]),
                   torch.tensor([[True, True]]))
    target = torch.tensor([[0.0, 0.0]], requires_grad=True)
    positive, ranking, active = tracker.losses(
        target, torch.tensor([[False, False]]),
        torch.tensor([0]), torch.tensor([1.0]),
        positive_margin=0.1, ranking_margin=0.1,
    )
    assert active == 0
    assert positive.item() == ranking.item() == 0
