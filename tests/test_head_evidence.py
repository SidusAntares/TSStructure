import torch
import pytest

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


def test_top30_keeps_scattered_points_without_interval_or_mass_filter():
    from utils.head_evidence import select_attention_top_fraction
    attention = torch.tensor([[[.25, .01, .23, .01, .01, .01, .01, .01, .22, .24]]])
    mask, valid = select_attention_top_fraction(attention, torch.arange(10)[None])
    assert torch.where(mask[0, 0])[0].tolist() == [0, 2, 9]
    assert valid.tolist() == [[True]]


def test_training_gate05_matches_diagnostic_gate_and_preserves_other_heads():
    from utils.timematch_diagnostic import counterfactual_embeddings
    attention = torch.tensor([[[.5, .3, .2], [.2, .3, .5]]], requires_grad=True)
    values = torch.arange(1., 7.).reshape(1, 2, 3, 1).requires_grad_()
    mask = torch.tensor([[[True, False, False], [False, False, True]]])
    actual = build_counterfactual_head_embeddings(attention, values, mask, intervention='gate', gate=.5)
    expected = counterfactual_embeddings(attention, values, mask, 'gate05')
    assert torch.equal(actual, expected)
    assert actual[0, 1, 1] == actual[0, 0, 1]
    assert actual[0, 2, 0] == actual[0, 0, 0]
    actual.sum().backward()
    assert torch.isfinite(attention.grad).all() and torch.isfinite(values.grad).all()


@pytest.mark.parametrize('mode', ['positive', 'positive_rank'])
def test_real_pse_top30_gate05_loss_backpropagates(mode):
    from models.stclassifier import PseLTae
    model = PseLTae(num_classes=3, with_extra=False).train()
    pixels = torch.rand(4, 30, 10, 8)
    mask = torch.ones(4, 30, 8)
    dates = torch.arange(30)[None].repeat(4, 1)*10
    logits, evidence, valid = model.forward_with_head_evidence(
        pixels, mask, dates, torch.zeros(4, 4), structure='top30', intervention='gate', gate=.5)
    labels = torch.tensor([0, 1, 2, 0])
    contributions = relative_logit_contributions(evidence, labels)
    tracker = SourceEvidenceTracker(3, 16, 'cpu', min_count=1,
                                    min_positive_rate=0., min_source_contribution=-100., rank_gap=0.)
    tracker.update(labels, contributions, valid)
    positive, rank, _ = tracker.losses(contributions, valid, labels, torch.ones(4))
    loss = torch.nn.functional.cross_entropy(logits, labels) + .1*positive
    if mode == 'positive_rank':
        loss = loss + .1*rank
    loss.backward()
    assert valid.all() and evidence.shape == (4, 17, 3)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


@pytest.mark.parametrize('mode', ['positive', 'positive_rank'])
def test_timematch_teacher_student_passes_top30_gate05_through_training(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace
    import numpy as np
    import timematch
    from models.stclassifier import PseLTae
    torch.manual_seed(12)
    student = PseLTae(num_classes=3, with_extra=False)
    source = tmp_path / 'source/fold_0'
    source.mkdir(parents=True)
    torch.save({'state_dict': student.state_dict()}, source / 'model.pt')
    sample = dict(pixels=torch.rand(4, 30, 10, 8), valid_pixels=torch.ones(4, 30, 8),
                  positions=torch.arange(30)[None].repeat(4, 1)*10,
                  extra=torch.zeros(4, 4), label=torch.tensor([0, 1, 2, 0]))
    target = SimpleNamespace(dataset=SimpleNamespace(get_labels=lambda: np.array([0, 1, 2, 0])))
    monkeypatch.setattr(timematch, 'get_data_loaders', lambda *a: ([sample], target, [(sample, sample)]))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    config = SimpleNamespace(
        weights=str(tmp_path / 'source'), balance_source=True, num_classes=3,
        use_focal_loss=True, focal_loss_gamma=1., steps_per_epoch=2, lr=.0001, weight_decay=.0001,
        epochs=1, max_temporal_shift=60, estimate_shift=False, shift_source=True,
        progress_bar='off', pseudo_threshold=0., domain_specific_bn=True,
        trade_off=2., ema_decay=.9999, log_step=1, run_validation=False, output_student=True,
        evidence_mode=mode, evidence_structure='top30', evidence_intervention='gate', evidence_gate=.5,
        evidence_top_fraction=.3, evidence_coverage=.75, evidence_min_observations=3, evidence_min_outside=2,
        evidence_source_min_count=1, evidence_source_positive_rate=0., evidence_source_min_contribution=-100.,
        evidence_source_rank_gap=0., evidence_warmup_steps=1, evidence_positive_margin=10.,
        evidence_ranking_margin=.1, evidence_positive_weight=.1, evidence_rank_weight=.1,
    )
    logs = []
    writer = SimpleNamespace(add_scalar=lambda *args: logs.append(args))
    path = tmp_path / 'final.pt'
    timematch.train_timematch(student, config, writer, None, torch.device('cpu'), path, 0, {})
    assert any(row[0] == 'evidence/active_target_heads' and row[1] > 0 for row in logs)
    assert any(row[0] == 'evidence/positive_loss' and row[1] > 0 for row in logs)
    assert all(torch.isfinite(value).all() for value in student.state_dict().values())
    assert torch.load(path, weights_only=False)['completed_epochs'] == 1
