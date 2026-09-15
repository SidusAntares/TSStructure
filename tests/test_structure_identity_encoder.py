"""07B source-only structure identity encoder contracts."""
from __future__ import annotations

import importlib
import inspect
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch


def model_api():
    return importlib.import_module("models.structure_identity.encoder")


def experiment_api():
    return importlib.import_module("analysis.structure_identity_encoder_experiment")


def synthetic_inputs(batch=3, points=32, channels=6):
    torch.manual_seed(4)
    shape = torch.randn(batch, points, channels)
    amplitude = torch.randn(batch, points, channels)
    event_types = torch.tensor([[1, 2, 0], [2, 1, 2], [0, 0, 0]])[:batch]
    event_numeric = torch.randn(batch, 3, 4)
    event_mask = event_types.ne(0)
    fine_types = torch.tensor([[1, 2, 1, 0], [2, 1, 0, 0], [0, 0, 0, 0]])[:batch]
    fine_numeric = torch.randn(batch, 4, 6)
    fine_mask = fine_types.ne(0)
    return dict(
        shape_waveform=shape,
        amplitude_waveform=amplitude,
        event_types=event_types,
        event_numeric=event_numeric,
        event_mask=event_mask,
        fine_types=fine_types,
        fine_numeric=fine_numeric,
        fine_mask=fine_mask,
    )


@pytest.mark.parametrize("variant", ["Waveform", "Event", "Fusion"])
def test_encoder_returns_finite_l2_normalized_128d_token(variant):
    m = model_api()
    model = m.StructureIdentityEncoder(waveform_dim=6, variant=variant, dropout=0.0).eval()
    output = model(**synthetic_inputs())
    assert output.shape == (3, 128)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)


def test_model_forward_signature_cannot_receive_forbidden_metadata():
    m = model_api()
    signature = inspect.signature(m.StructureIdentityEncoder.forward)
    forbidden = {"class_id", "class_name", "domain_id", "source", "target", "sample_id",
                 "start_day", "end_day", "center_day", "duration_days", "absolute_time"}
    assert forbidden.isdisjoint(signature.parameters)


def test_waveform_preprocessing_has_frozen_shape_and_amplitude_semantics():
    m = model_api()
    values = torch.tensor([[[2.0, 4.0], [5.0, 4.0], [8.0, 8.0]]])
    shape = m.shape_waveform(values)
    amplitude = m.amplitude_waveform(values)
    torch.testing.assert_close(amplitude, values - values[:, :1])
    torch.testing.assert_close(shape.norm(dim=(1, 2)), torch.ones(1))
    torch.testing.assert_close(shape[:, 0], torch.zeros(1, 2))
    assert not torch.allclose(amplitude.norm(dim=(1, 2)), torch.ones(1))
    with pytest.raises(ValueError, match="32"):
        m.StructureIdentityEncoder(2)(**synthetic_inputs(points=31, channels=2))


def test_waveform_branches_are_independent_multiscale_encoders():
    m = model_api()
    model = m.StructureIdentityEncoder(6, variant="Fusion")
    assert model.shape_encoder is not model.amplitude_encoder
    assert {layer.kernel_size[0] for layer in model.shape_encoder.multiscale} == {3, 5, 7}
    assert {layer.kernel_size[0] for layer in model.amplitude_encoder.multiscale} == {3, 5, 7}


def test_event_and_fine_sequences_support_padding_and_empty_rows():
    m = model_api()
    encoder = m.SequenceTokenEncoder(numeric_dim=4, num_types=3, dropout=0.0).eval()
    types = torch.tensor([[1, 2, 0], [0, 0, 0]])
    numeric = torch.randn(2, 3, 4)
    mask = types.ne(0)
    result = encoder(types, numeric, mask)
    assert result.shape == (2, 64) and torch.isfinite(result).all()
    changed = numeric.clone(); changed[0, 2] = 1e6
    torch.testing.assert_close(result[0], encoder(types, changed, mask)[0])


def test_shape_token_fuses_only_active_branch_tokens():
    m = model_api()
    assert m.StructureIdentityEncoder(6, variant="Waveform").active_branches == ("shape", "amplitude")
    assert m.StructureIdentityEncoder(6, variant="Event").active_branches == ("event", "fine")
    assert m.StructureIdentityEncoder(6, variant="Fusion").active_branches == (
        "shape", "amplitude", "event", "fine"
    )


def test_fusion_backward_reaches_all_four_structure_branches():
    m = model_api()
    model = m.StructureIdentityEncoder(6, variant="Fusion", dropout=0.0)
    output = model(**synthetic_inputs())
    loss = (output * torch.linspace(0.1, 1.0, output.shape[-1])).sum()
    loss.backward()
    for module in (model.shape_encoder, model.amplitude_encoder,
                   model.event_encoder, model.fine_encoder, model.fusion):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in module.parameters())


def test_relative_queries_are_32_points_and_unwrap_cross_year():
    e = experiment_api()
    normal = e.relative_queries({"start_day": 10, "end_day": 41})
    cross = e.relative_queries({"start_day": 350, "end_day": 16})
    assert len(normal) == len(cross) == 32
    assert normal[[0, -1]].tolist() == [10, 41]
    assert cross[0] == 350 and cross[-1] == 381


def test_sample_level_stratified_split_is_reproducible_and_disjoint():
    e = experiment_api()
    rows = [dict(sample_id=class_id * 100 + index, class_id=class_id)
            for class_id in range(2) for index in range(20)]
    first = e.stratified_sample_split(rows, seed=1)
    second = e.stratified_sample_split(rows, seed=1)
    assert first == second
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert sorted(first["train"] + first["validation"] + first["test"]) == sorted(r["sample_id"] for r in rows)


def test_group_assignment_keeps_every_pair_from_one_sample_together():
    e = experiment_api()
    split = {"train": [1], "validation": [2], "test": [3]}
    groups = [dict(sample_id=i, reference_structure_id=ref) for i in (1, 2, 3) for ref in ("A", "B")]
    assigned = e.assign_groups_to_split(groups, split)
    assert {row["sample_id"] for row in assigned["train"]} == {1}
    assert {row["sample_id"] for row in assigned["validation"]} == {2}
    assert {row["sample_id"] for row in assigned["test"]} == {3}


def test_normalization_statistics_fit_train_only():
    e = experiment_api()
    train = [dict(amplitude=np.ones((32, 2)), events=np.ones((2, 4)), fine=np.ones((2, 6)))]
    held_out = [dict(amplitude=np.full((32, 2), 1000.), events=np.full((2, 4), 1000.), fine=np.full((2, 6), 1000.))]
    stats = e.fit_normalization_statistics(train)
    np.testing.assert_allclose(stats["amplitude_mean"], [1, 1])
    np.testing.assert_allclose(stats["event_mean"], [1] * 4)
    np.testing.assert_allclose(stats["fine_mean"], [1] * 6)
    assert not np.isclose(stats["amplitude_mean"], np.mean([r["amplitude"] for r in train + held_out])).all()


def test_candidate_group_requires_gplus_and_only_same_direction_accepted_negatives():
    e = experiment_api()
    candidates = [
        dict(segment_id="positive", direction="RISE", accepted=True),
        dict(segment_id="negative", direction="RISE", accepted=True),
        dict(segment_id="rejected", direction="RISE", accepted=False),
        dict(segment_id="fall", direction="FALL", accepted=True),
    ]
    group = e.build_candidate_group("R", "positive", "RISE", candidates, sample_id=9, class_id=2)
    assert [row["segment_id"] for row in group["candidates"]] == ["positive", "negative"]
    assert group["positive_index"] == 0
    with pytest.raises(ValueError, match="positive"):
        e.build_candidate_group("R", "missing", "RISE", candidates, 9, 2)


def test_padding_candidate_groups_builds_correct_mask_and_positive_indices():
    e = experiment_api()
    batch = e.pad_candidate_embeddings([
        torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([[1., 1.]])
    ], [1, 0])
    assert batch.embeddings.shape == (2, 2, 2)
    assert batch.mask.tolist() == [[True, True], [True, False]]
    assert batch.positive_index.tolist() == [1, 0]


def test_listwise_and_positive_losses_follow_frozen_definition():
    e = experiment_api()
    scores = torch.tensor([[2., 0.], [1., -99.]])
    mask = torch.tensor([[True, True], [True, False]])
    positive = torch.tensor([0, 0])
    rank, used = e.listwise_ranking_loss(scores, mask, positive, temperature=1.0)
    assert used == 1
    assert rank == pytest.approx(torch.log1p(torch.exp(torch.tensor(-2.))).item())
    refs = torch.tensor([[1., 0.], [1., 0.]])
    positives = torch.tensor([[1., 0.], [0., 1.]])
    consistency = e.positive_consistency_loss(refs, positives)
    assert consistency.item() == pytest.approx(.5)
    total = e.structure_identity_loss(scores, mask, positive, refs, positives, temperature=1.0, lambda_pos=.1)
    assert total["loss"].item() == pytest.approx(rank.item() + .05)


def test_cosine_candidate_scores_are_dot_products_of_normalized_tokens():
    e = experiment_api()
    reference = torch.tensor([[1., 0.]])
    candidates = torch.tensor([[[1., 0.], [0., 1.]]])
    scores = e.cosine_candidate_scores(reference, candidates)
    torch.testing.assert_close(scores, torch.tensor([[1., 0.]]))


def test_retrieval_metrics_are_multi_candidate_and_include_margin():
    e = experiment_api()
    rows = e.retrieval_rows([dict(group_id="g", candidate_ids=["p", "n"], positive_id="p")],
                            [np.array([.8, .2])])
    assert rows[0]["exact"] and rows[0]["top2"] and rows[0]["mrr"] == 1
    assert rows[0]["pairwise_win_rate"] == 1 and rows[0]["margin"] == pytest.approx(.6)
    assert rows[0]["unique_best"]


def test_precision_calibration_maximizes_coverage_at_required_precision():
    e = experiment_api()
    rows = [
        dict(top1_similarity=.95, margin=.4, exact=True),
        dict(top1_similarity=.90, margin=.3, exact=True),
        dict(top1_similarity=.85, margin=.2, exact=False),
        dict(top1_similarity=.80, margin=.1, exact=True),
    ]
    calibrated = e.calibrate_rejection_thresholds(rows, minimum_precision=.95)
    assert calibrated["accepted"] == 2
    assert calibrated["precision"] == 1
    assert calibrated["coverage"] == pytest.approx(.5)
    frozen = e.apply_rejection_thresholds(rows, calibrated)
    assert sum(row["accepted"] for row in frozen) == 2


def test_test_evaluation_uses_frozen_threshold_without_search(monkeypatch):
    e = experiment_api()
    monkeypatch.setattr(e, "calibrate_rejection_thresholds", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))
    result = e.evaluate_with_frozen_thresholds(
        [dict(top1_similarity=.8, margin=.2, exact=True)],
        dict(similarity_threshold=.7, margin_threshold=.1),
    )
    assert result["accepted_correct"] == 1 and result["accepted_wrong"] == 0


def test_variants_share_identical_candidate_pool_and_split():
    e = experiment_api()
    groups = [dict(group_id="a", candidate_ids=["p", "n"], sample_id=1)]
    packets = e.variant_packets(groups, ("Waveform", "Event", "Fusion"))
    assert {tuple(packet[0]["candidate_ids"]) for packet in packets.values()} == {("p", "n")}
    assert {packet[0]["sample_id"] for packet in packets.values()} == {1}


def test_train_only_reference_rejects_empty_event_or_fine_representation():
    e = experiment_api()
    valid = e.validate_train_reference(dict(events=[1], fine=[1]))
    assert valid["valid_reference"]
    assert not e.validate_train_reference(dict(events=[], fine=[1]))["valid_reference"]
    assert not e.validate_train_reference(dict(events=[1], fine=[]))["valid_reference"]


def test_model_seed_is_reproducible():
    e = experiment_api(); m = model_api()
    e.seed_everything(7); first = m.StructureIdentityEncoder(6, dropout=0.0)
    e.seed_everything(7); second = m.StructureIdentityEncoder(6, dropout=0.0)
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))


def test_atomic_source_publication_preserves_previous_output_on_failure():
    e = experiment_api()
    root = Path(__file__).parent / f"_tmp_07b_atomic_{uuid.uuid4().hex}"
    try:
        final = root / "AT1"; final.mkdir(parents=True); (final / "old.txt").write_text("old")
        with pytest.raises(RuntimeError, match="incomplete"):
            with e.staged_source_output(final, required=("three_seed_summary.csv",)) as staging:
                (staging / "partial.txt").write_text("partial")
        assert (final / "old.txt").read_text() == "old"
        with e.staged_source_output(final, required=("three_seed_summary.csv",)) as staging:
            (staging / "three_seed_summary.csv").write_text("header\n")
        assert not (final / "old.txt").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_static_source_only_and_launcher_contracts():
    root = Path(__file__).parents[1]
    model = (root / "models/structure_identity/encoder.py").read_text(encoding="utf-8")
    experiment = (root / "analysis/structure_identity_encoder_experiment.py").read_text(encoding="utf-8")
    runner = (root / "scripts/train_structure_identity_encoder.py").read_text(encoding="utf-8")
    launcher = (root / "scripts/run_structure_identity_encoder_4sources.sh").read_text(encoding="utf-8")
    combined = model + experiment + runner
    forbidden_inputs = ("target_loader", "target_checkpoint", "domain_discriminator", "warp_loss", "class_loss")
    assert not any(token in combined for token in forbidden_inputs)
    assert "target_data_used" in runner and "False" in runner
    assert all(token in launcher for token in ("GPU0", "GPU1", "GPU2", "GPU3", "AT1", "DK1", "FR1", "FR2"))
    assert not any(token in launcher for token in ("git ", "curl ", "wget ", "pip install", "conda ", "nohup"))
    assert "staged_source_output" in experiment
