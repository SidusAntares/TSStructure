from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from methods.structure_da import SourcePrototypeBank, Stage1LossOutput, Stage1Objective


def _weights(grid: int = 6) -> torch.Tensor:
    w = torch.full((grid,), 1.0 / (grid - 1))
    w[[0, -1]] *= 0.5
    return w / w.sum()


def _bank(shape_srvf=None, fused=None, ready=None) -> SourcePrototypeBank:
    return SourcePrototypeBank(
        trend_srvf=torch.zeros(3, 6, 4),
        shape_srvf=torch.zeros(3, 6, 4) if shape_srvf is None else shape_srvf,
        trend_support=torch.ones(3, 6),
        shape_support=torch.ones(3, 6),
        fused=torch.zeros(3, 8) if fused is None else fused,
        class_counts=torch.tensor([5, 5, 5]),
        ready=torch.ones(3, dtype=torch.bool) if ready is None else ready,
        q_distance_samples=(torch.zeros(0), torch.zeros(0), torch.zeros(0)),
        f_distance_samples=(torch.zeros(0), torch.zeros(0), torch.zeros(0)),
        q_quantiles=torch.zeros(3, 3),
        f_quantiles=torch.zeros(3, 3),
        version=0,
    )


def _run(obj, logits, fused, labels, q, sup, valid, bank, warmup=False):
    return obj(
        logits=logits,
        fused_repr=fused,
        labels=labels,
        q=q,
        q_support=sup,
        q_valid=valid,
        bank=bank,
        integration_weights=_weights(),
        warmup=warmup,
    )


def test_warmup_reduces_to_classification_only() -> None:
    obj = Stage1Objective(num_classes=3)
    logits = torch.randn(6, 3)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])

    out = _run(obj, logits, torch.randn(6, 8), labels, None, None, None, None, warmup=True)

    assert isinstance(out, Stage1LossOutput)
    torch.testing.assert_close(out.total, out.classification)
    assert out.q_prototype.item() == 0.0
    assert out.fused_prototype.item() == 0.0
    assert out.q_to_classifier.item() == 0.0
    assert out.q_valid_count == 0
    assert out.fused_valid_count == 0
    assert out.consistency_valid_count == 0


def test_q_margin_satisfied_gives_zero_loss() -> None:
    # d_plus + margin <= d_minus must give exactly zero.
    q = torch.zeros(2, 6, 4)
    q[1, :, :] = 10.0  # far from every prototype -> large positive, large negative gap
    sup = torch.ones(2, 6)
    valid = torch.ones(2, dtype=torch.bool)
    bank = _bank()
    obj = Stage1Objective(num_classes=3, margin_q=0.1)
    out = _run(
        obj,
        torch.randn(2, 3),
        torch.randn(2, 8),
        torch.tensor([0, 1]),
        q,
        sup,
        valid,
        bank,
    )
    # q[0] is exactly the class-0 prototype -> d_plus=0, d_minus>=0 -> zero.
    # q[1] is far from everything but true class is class 1 whose prototype is 0,
    # so d_plus is large and d_minus is also large; relative margin may still be
    # satisfied because all prototypes coincide. With identical prototypes the
    # positive and negative distances are equal, so the margin is violated.
    # Use a query that is exactly the true prototype for a guaranteed zero.
    qz = torch.zeros(2, 6, 4)
    bankz = _bank()
    outz = _run(
        obj,
        torch.randn(2, 3),
        torch.randn(2, 8),
        torch.tensor([0, 1]),
        qz,
        sup,
        valid,
        bankz,
    )
    # q == class prototypes -> d_plus = 0 and d_minus = 0 -> relu(0.1 + 0 - 0) = 0.1
    # This is *not* the satisfied case. Instead build prototypes so d_plus < d_minus - margin.
    # Construct: class 0 prototype at origin, class 1 and 2 far away.
    shape = torch.zeros(3, 6, 4)
    shape[1, :, 0] = 10.0
    shape[2, :, 1] = 10.0
    b = _bank(shape_srvf=shape)
    qs = torch.zeros(2, 6, 4)  # query near class 0
    outs = _run(
        obj,
        torch.randn(2, 3),
        torch.randn(2, 8),
        torch.tensor([0, 0]),
        qs,
        sup,
        valid,
        b,
    )
    # d_plus (to class 0) = 0; d_minus (to 10 or 10) ~ 10 -> margin satisfied -> 0
    assert outs.q_prototype.item() == 0.0
    del out, outz


def test_q_margin_violated_gives_positive_loss() -> None:
    # All prototypes identical: d_plus == d_minus -> violation = margin > 0.
    shape = torch.zeros(3, 6, 4)
    bank = _bank(shape_srvf=shape)
    obj = Stage1Objective(num_classes=3, margin_q=0.1)
    q = torch.zeros(2, 6, 4)
    sup = torch.ones(2, 6)
    valid = torch.ones(2, dtype=torch.bool)

    out = _run(
        obj,
        torch.randn(2, 3),
        torch.randn(2, 8),
        torch.tensor([0, 1]),
        q,
        sup,
        valid,
        bank,
    )
    assert out.q_prototype.item() > 0.0


def test_margin_satisfied_no_extra_compression() -> None:
    # Once d_plus + margin <= d_minus, reducing d_plus further must not change loss.
    shape = torch.zeros(3, 6, 4)
    shape[1, :, 0] = 10.0
    shape[2, :, 1] = 10.0
    bank = _bank(shape_srvf=shape)
    obj = Stage1Objective(num_classes=3, margin_q=0.1)
    sup = torch.ones(1, 6)
    valid = torch.ones(1, dtype=torch.bool)

    q_a = torch.zeros(1, 6, 4)         # d_plus=0
    q_b = torch.zeros(1, 6, 4) * 0.001  # closer still, but margin already satisfied
    out_a = _run(obj, torch.randn(1, 3), torch.randn(1, 8), torch.tensor([0]), q_a, sup, valid, bank)
    out_b = _run(obj, torch.randn(1, 3), torch.randn(1, 8), torch.tensor([0]), q_b, sup, valid, bank)
    assert out_a.q_prototype.item() == 0.0
    assert out_b.q_prototype.item() == 0.0


def test_fused_margin_behavior() -> None:
    fused_proto = torch.zeros(3, 8)
    fused_proto[1, 0] = 1.0
    fused_proto[2, 1] = 1.0
    bank = _bank(fused=fused_proto)
    obj = Stage1Objective(num_classes=3, margin_f=0.1)
    f = torch.zeros(2, 8)
    labels = torch.tensor([0, 0])
    logits = torch.randn(2, 3)

    # Query at origin: cosine to class-0 prototype (origin) undefined (zero norm).
    # Give the true prototype nonzero so cosine is defined.
    fused_proto2 = torch.zeros(3, 8)
    fused_proto2[0, 0] = 1.0
    fused_proto2[1, 2] = 1.0
    fused_proto2[2, 3] = 1.0
    bank2 = _bank(fused=fused_proto2)
    f2 = torch.zeros(2, 8)
    f2[:, 0] = 1.0  # aligned with class-0 prototype
    out = _run(obj, logits, f2, labels, torch.randn(2, 6, 4), torch.ones(2, 6), torch.ones(2, dtype=torch.bool), bank2)
    # d_plus to class0 = 1 - cos(f, mu0) = 1 - 1 = 0; d_minus = 1 - 0 = 1 -> satisfied
    assert out.fused_prototype.item() == 0.0

    # Violate: make f far from its true class and near a wrong class.
    f3 = torch.zeros(2, 8)
    f3[:, 1] = 1.0  # aligned with class-1 but labeled class 0
    out3 = _run(obj, logits, f3, labels, torch.randn(2, 6, 4), torch.ones(2, 6), torch.ones(2, dtype=torch.bool), bank2)
    assert out3.fused_prototype.item() > 0.0


def test_q_to_classifier_stop_gradient() -> None:
    # KL must update logits but not the q path, and must not touch fused features.
    obj = Stage1Objective(num_classes=3, tau_q=0.1)
    shape = torch.zeros(3, 6, 4)
    shape[1, :, 0] = 10.0
    shape[2, :, 1] = 10.0
    bank = _bank(shape_srvf=shape)
    q = torch.zeros(2, 6, 4, requires_grad=True)
    sup = torch.ones(2, 6)
    valid = torch.ones(2, dtype=torch.bool)
    logits = torch.randn(2, 3, requires_grad=True)
    labels = torch.tensor([0, 1])
    fused = torch.randn(2, 8, requires_grad=True)

    out = _run(obj, logits, fused, labels, q, sup, valid, bank)
    out.q_to_classifier.backward()

    assert logits.grad is not None and logits.grad.abs().sum().item() > 0
    # The q tensor must receive no gradient from the KL term.
    assert q.grad is None or q.grad.abs().sum().item() == 0
    # The fused features are not part of the q-to-cls term at all.
    assert fused.grad is None


def test_no_valid_consistency_samples_returns_zero() -> None:
    obj = Stage1Objective(num_classes=3)
    bank = _bank(ready=torch.tensor([True, True, False]))
    q = torch.randn(2, 6, 4)
    sup = torch.ones(2, 6)
    valid = torch.ones(2, dtype=torch.bool)
    logits = torch.randn(2, 3, requires_grad=True)

    out = _run(obj, logits, torch.randn(2, 8), torch.tensor([0, 1]), q, sup, valid, bank)
    # Not all classes ready -> q-to-cls is a typed zero.
    assert out.q_to_classifier.item() == 0.0
    assert out.consistency_valid_count == 0


def test_prototype_not_trainable_and_not_in_optimizer() -> None:
    bank = _bank()
    assert not isinstance(bank.fused, torch.nn.Parameter)
    assert not isinstance(bank.shape_srvf, torch.nn.Parameter)
    # A Stage1Objective holds no parameters.
    obj = Stage1Objective(num_classes=3)
    assert not any(True for _ in obj.parameters())
