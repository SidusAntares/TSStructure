import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import analysis.structure_semantic_features as features
import analysis.structure_semantic_abc as abc
import scripts.run_structure_semantic_abc_seed1 as scheduler
from models.stclassifier import PseStructureProtoLTae
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def test_response96_uses_six_blocks_not_contiguous_anchor_records():
    response = np.arange(96, dtype=np.float32)[None]
    events = features.response_to_anchor_events(response)
    assert events.shape == (1, 16, 6)
    np.testing.assert_array_equal(events[0, 3], [3, 19, 35, 51, 67, 83])
    assert features.EVENT_COMPONENTS == (
        "strength", "concentration", "phase_k1_cos", "phase_k1_sin",
        "phase_k2_cos", "phase_k2_sin",
    )


def test_sample_keys_reject_duplicate_domain_parcel_pairs():
    assert features.sample_keys("france/31TCJ/2017", [7, 8]) == (
        ("france/31TCJ/2017", 7), ("france/31TCJ/2017", 8),
    )
    with pytest.raises(ValueError, match="duplicate sample key"):
        features.sample_keys("france/31TCJ/2017", [7, 7])


def test_build_abc_datasets_never_instantiates_target_test(monkeypatch):
    config = SimpleNamespace(
        source="source-domain", target="target-domain", num_folds=1,
        val_ratio=.2, test_ratio=.2, classes=["a", "b"], seed=1,
        with_extra=False, combine_spring_and_winter=False,
    )
    split = {
        "source-domain": {"train": [1, 2], "val": [3], "test": [4]},
        "target-domain": {"train": [5, 6], "val": [7], "test": [8]},
    }
    calls = []

    monkeypatch.setattr(
        features, "prepare_official_protocol", lambda cfg: ({}, {"protocol": "official"}),
    )
    monkeypatch.setattr(
        features, "create_official_splits",
        lambda domains, folds, eligible, val_ratio, test_ratio: [split],
    )

    def fake_dataset(data_root, domain, cfg, indices):
        calls.append((domain, tuple(indices)))
        return {"domain": domain, "indices": tuple(indices)}

    monkeypatch.setattr(features, "make_dataset", fake_dataset)
    _, protocol, returned_split, datasets = features.build_abc_datasets(
        config, "/dataset", 1,
    )
    assert protocol == {"protocol": "official"}
    assert returned_split == split
    assert set(datasets) == {
        "source_train", "source_val", "target_train", "target_val",
    }
    assert ("target-domain", (8,)) not in calls
    assert calls == [
        ("source-domain", (1, 2)), ("source-domain", (3,)),
        ("target-domain", (5, 6)), ("target-domain", (7,)),
    ]


def test_build_abc_datasets_reseeds_python_random_for_exact_replay(monkeypatch):
    config = SimpleNamespace(
        source="source", target="target", num_folds=1,
        val_ratio=.2, test_ratio=.2, classes=["a", "b"], seed=1,
        with_extra=False, combine_spring_and_winter=False,
    )
    monkeypatch.setattr(
        features, "prepare_official_protocol",
        lambda cfg: ({"source": list(range(20)), "target": list(range(20))}, {}),
    )

    def shuffled_splits(domains, folds, eligible, val_ratio, test_ratio):
        result = {}
        for domain in domains:
            indices = list(eligible[domain])
            random.shuffle(indices)
            result[domain] = {
                "train": indices[:12], "val": indices[12:16], "test": indices[16:],
            }
        return [result]

    monkeypatch.setattr(features, "create_official_splits", shuffled_splits)
    monkeypatch.setattr(
        features, "make_dataset",
        lambda data_root, domain, cfg, indices: tuple(indices),
    )
    random.seed(999)
    first = features.build_abc_datasets(config, "/dataset", 7)[2]
    random.seed(12345)
    second = features.build_abc_datasets(config, "/dataset", 7)[2]
    assert first == second


def test_manifest_replay_rebuilds_exact_cached_parcel_sets_without_target_test(monkeypatch):
    config = SimpleNamespace(source="source", target="target", classes=["a", "b"])
    manifest = {
        "source_train": [{"domain": "source", "parcel_index": 9}],
        "source_val": [{"domain": "source", "parcel_index": 4}],
        "target_train": [{"domain": "target", "parcel_index": 17}],
        "target_val": [{"domain": "target", "parcel_index": 23}],
    }
    calls = []

    class Dataset:
        def __init__(self, indices):
            self.indices = np.asarray(indices, dtype=np.int64)

        def get_parcel_indices(self):
            return self.indices

    def fake_dataset(data_root, domain, cfg, indices):
        calls.append((domain, tuple(indices)))
        return Dataset(indices)

    monkeypatch.setattr(features, "make_dataset", fake_dataset)
    datasets = features.replay_abc_datasets(config, "/dataset", 1, manifest)
    assert set(datasets) == set(features.ALLOWED_SPLITS)
    assert calls == [
        ("source", (9,)), ("source", (4,)),
        ("target", (17,)), ("target", (23,)),
    ]


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))


def test_frozen_models_require_strict_teacher_and_record_separate_shifts(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "checkpoint_last.pt"
    checkpoint.touch()
    packet = {
        "epoch": 19,
        "config": {"epochs": 20, "classes": ["a", "b"]},
        "state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "teacher_state_dict": {"weight": torch.tensor([3.0, 4.0])},
        "global_temporal_shift": -6,
    }
    monkeypatch.setattr(features, "require_complete_checkpoint", lambda path: packet)
    monkeypatch.setattr(features, "validate_uda_checkpoint", lambda *args: {})
    monkeypatch.setattr(features, "create_model_from_config", lambda config: _TinyModel())

    frozen = features.load_frozen_models(checkpoint, "FR2_DK1", "E", "cpu")
    torch.testing.assert_close(frozen.student.weight, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(frozen.teacher.weight, torch.tensor([3.0, 4.0]))
    assert frozen.shifts == {"student": 0, "teacher": -6}
    assert frozen.student is not frozen.teacher

    bad = dict(packet)
    bad["teacher_state_dict"] = {}
    monkeypatch.setattr(features, "require_complete_checkpoint", lambda path: bad)
    with pytest.raises(RuntimeError, match="strict EMA Teacher"):
        features.load_frozen_models(checkpoint, "FR2_DK1", "E", "cpu")


def test_global_shift_must_be_one_integer_day(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint_last.pt"
    checkpoint.touch()
    packet = {
        "epoch": 19,
        "config": {"epochs": 20, "classes": ["a", "b"]},
        "state_dict": {"weight": torch.zeros(2)},
        "teacher_state_dict": {"weight": torch.zeros(2)},
        "global_temporal_shift": 1.5,
    }
    monkeypatch.setattr(features, "require_complete_checkpoint", lambda path: packet)
    monkeypatch.setattr(features, "validate_uda_checkpoint", lambda *args: {})
    monkeypatch.setattr(features, "create_model_from_config", lambda config: _TinyModel())
    with pytest.raises(ValueError, match="integer day"):
        features.load_frozen_models(checkpoint, "FR2_DK1", "G", "cpu")


class _TinyParcelDataset(torch.utils.data.Dataset):
    def __init__(self, domain, count=3):
        self.domain = domain
        self.count = count

    def __len__(self):
        return self.count

    def get_shapes(self):
        return [(10, 3, 3 + index) for index in range(self.count)]

    def __getitem__(self, index):
        pixels = 3 + index
        generator = torch.Generator().manual_seed(100 + index)
        return {
            "pixels": torch.randn(10, 3, pixels, generator=generator),
            "valid_pixels": torch.ones(10, pixels),
            "positions": torch.arange(10) * 30,
            "extra": torch.zeros(4),
            "label": torch.tensor(index % 3),
            "parcel_index": 1000 + index,
        }


def _phase_model():
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation="phase_moment", shape_injection="current_query",
    ).eval()
    model.temporal_encoder.attention_heads.external_query_projection.weight.data.normal_(0, .1)
    return model


def test_one_pass_role_separated_cache_has_correct_shift_and_no_target_train_labels(
    tmp_path,
):
    torch.manual_seed(7)
    student = _phase_model()
    teacher = _phase_model()
    frozen = features.FrozenModels(
        student=student, teacher=teacher,
        config=SimpleNamespace(classes=["a", "b", "c"]), packet={},
        shifts={"student": 0, "teacher": -5},
    )
    datasets = {
        "source_train": _TinyParcelDataset("source"),
        "source_val": _TinyParcelDataset("source"),
        "target_train": _TinyParcelDataset("target"),
        "target_val": _TinyParcelDataset("target"),
    }
    manifest = features.extract_checkpoint_features(
        frozen, datasets, {"source": "source", "target": "target"},
        tmp_path, device="cpu", batch_size=3, pixel_budget=8,
        representative_count=1,
    )
    assert manifest.extraction_passes == {
        role: {split: 1 for split in features.ALLOWED_SPLITS}
        for role in ("student", "teacher")
    }
    assert manifest.applied_shifts["student"] == {
        split: 0 for split in features.ALLOWED_SPLITS
    }
    assert manifest.applied_shifts["teacher"] == {
        split: -5 for split in features.ALLOWED_SPLITS
    }

    target_train = features.load_feature_split(tmp_path, "teacher", "target_train")
    assert "label" not in target_train
    assert target_train["events"].shape == (3, 16, 6)
    assert target_train["representative_curves"].shape[0] <= 16 * 2
    assert target_train["representative_curves"].shape[-1] == 8

    source = features.load_feature_split(tmp_path, "student", "source_train")
    assert source["label"].tolist() == [0, 1, 2]
    assert list(zip(source["domain"].tolist(), source["parcel_index"].tolist())) == [
        ("source", 1000), ("source", 1001), ("source", 1002),
    ]
    first = datasets["source_train"][0]
    with torch.no_grad():
        expected = student.forward_with_temporal_shift(
            first["pixels"][None], first["valid_pixels"][None],
            first["positions"][None], first["extra"][None],
            temporal_shift=0, return_dict=True,
        )["logits"][0].numpy()
    np.testing.assert_allclose(source["logits"][0], expected, atol=1e-5)
    features.validate_feature_cache(tmp_path, manifest)


def _event_fixture():
    rng = np.random.default_rng(12)
    labels = np.repeat([0, 1], 24)
    events = rng.normal(0, .15, size=(48, 16, 6))
    events[labels == 0, 2, 0] += 1.0
    events[labels == 0, 2, 2] += .6
    events[labels == 1, 2, 0] -= 1.0
    events[labels == 1, 2, 2] -= .6
    return events, labels


def test_pairwise_effects_and_source_only_candidates_are_role_local():
    train, labels = _event_fixture()
    val = train + np.random.default_rng(2).normal(0, .02, train.shape)
    target_unlabeled = train[:30] + .01
    student_rows = abc.pairwise_event_statistics(
        train, labels, 0, 1, "source_train", "student", seed=1,
    )
    teacher_rows = abc.pairwise_event_statistics(
        -train, labels, 0, 1, "source_train", "teacher", seed=1,
    )
    assert student_rows[2]["effect_strength"] > 0
    assert teacher_rows[2]["effect_strength"] < 0

    val_rows = abc.pairwise_event_statistics(
        val, labels, 0, 1, "source_val", "student", seed=1,
    )
    candidates = abc.select_source_only_candidates(
        student_rows, val_rows, train, target_unlabeled,
        min_support=10, min_effect=.5, min_direction_cosine=.5,
        min_target_range=.5, role="student", pair=(0, 1), seed=1,
    )
    assert any(row["anchor_id"] == 2 for row in candidates)
    assert all(row["role"] == "student" for row in candidates)
    assert all("target_label" not in row for row in candidates)


def test_missing_pair_class_returns_na_rows():
    events = np.zeros((5, 16, 6), dtype=np.float32)
    rows = abc.pairwise_event_statistics(
        events, np.zeros(5, dtype=np.int64), 0, 2,
        "target_val", "teacher", seed=1,
    )
    assert len(rows) == 16
    assert all(row["support_y"] == 5 and row["support_c"] == 0 for row in rows)
    assert all(row["status"] == "N/A" and np.isnan(row["effect_norm"]) for row in rows)


def test_joint_support_rejects_correlated_off_manifold_event():
    rng = np.random.default_rng(4)
    x = np.linspace(-2, 2, 80)
    source = np.column_stack((x, x + rng.normal(0, .03, len(x))))
    support = abc.fit_joint_support(source, quantile=.95, seed=1)
    accepted = abc.joint_support_mask(support, np.asarray([[1.0, 1.0], [1.0, -1.0]]))
    assert accepted.tolist() == [True, False]


def test_pairwise_probe_margin_reconstructs_all_anchor_components():
    rng = np.random.default_rng(5)
    response = rng.normal(size=(60, 96))
    labels = np.repeat([0, 1, 2], 20)
    response[labels == 0, :16] += .8
    response[labels == 1, 32:48] -= .7
    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RidgeClassifier(alpha=1.0)),
    ]).fit(response, labels)
    result = abc.decompose_pairwise_anchor_margin(probe, response[:10], 0, 1)
    reconstructed = result["anchor_component_contribution"].sum((1, 2)) + result["bias"]
    np.testing.assert_allclose(reconstructed, result["margin"], atol=1e-8)
    assert result["anchor_component_contribution"].shape == (10, 16, 6)


def test_teacher_structure_support_sign_and_source_only_calibration():
    train, labels = _event_fixture()
    train_rows = abc.pairwise_event_statistics(
        train, labels, 0, 1, "source_train", "teacher", seed=1,
    )
    val_rows = abc.pairwise_event_statistics(
        train + .01, labels, 0, 1, "source_val", "teacher", seed=1,
    )
    candidates = abc.select_source_only_candidates(
        train_rows, val_rows, train, train[:30], min_support=10,
        min_effect=.5, min_direction_cosine=.5, min_target_range=.5,
        role="teacher", pair=(0, 1), seed=1,
    )
    calibration = abc.fit_source_structure_support(candidates)
    predictions = np.asarray([0, 1])
    samples = np.stack((train[0], train[-1]))
    scored = abc.score_structure_support(calibration, samples, predictions)
    assert scored["available"].tolist() == [True, True]
    assert scored["margin"][0] > 0
    assert scored["margin"][1] > 0
    assert scored["conflict"].tolist() == [False, False]

    val_scores = abc.score_structure_support(calibration, train, labels)
    threshold = abc.calibrate_source_threshold(
        val_scores, np.ones(len(labels), dtype=bool), minimum_coverage=.5,
    )
    assert np.isfinite(threshold["structure_threshold"])
    assert threshold["fit_split"] == "source_val"


def test_matched_coverage_comparison_uses_identical_keys_and_counts():
    keys = tuple(("target", index) for index in range(8))
    labels = np.asarray([0, 0, 0, 1, 1, 1, 1, 1])
    predictions = np.asarray([0, 1, 0, 1, 0, 1, 1, 0])
    confidence = np.asarray([.99, .98, .97, .96, .95, .94, .93, .92])
    structure = np.asarray([.9, -.5, .8, .7, -.2, .6, .5, -.1])
    anchor = np.asarray([.4, .3, .8, .7, .2, .6, .5, .1])
    rows = abc.coverage_error_curves(
        keys, labels, predictions, confidence, anchor, structure,
        confidence_threshold=.9, structure_threshold=.55,
    )
    matched = [row for row in rows if row["curve_point"] == "matched"]
    counts = {row["method"]: row["accepted"] for row in matched}
    assert counts["confidence_matched"] == counts["anchor_reliability_matched"]
    assert counts["confidence_matched"] == counts["structure_support"]
    assert all(row["eligible"] == 8 for row in matched)
    assert all(row["sample_key_digest"] == matched[0]["sample_key_digest"] for row in matched)


def test_no_candidates_produces_explicit_baseline_status():
    calibration = abc.fit_source_structure_support([])
    scores = abc.score_structure_support(
        calibration, np.zeros((3, 16, 6)), np.asarray([0, 1, 0]),
    )
    assert calibration["status"] == "no_reliable_candidates"
    assert not scores["available"].any()
    assert np.isnan(scores["margin"]).all()


def _tiny_batch(count=2):
    dataset = _TinyParcelDataset("target", count=count)
    return features._semantic_collate([dataset[index] for index in range(count)])


def test_normal_student_sensitivity_matches_forward_and_native_block_mapping():
    torch.manual_seed(18)
    model = _phase_model()
    batch = _tiny_batch(2)
    result = abc.normal_pair_margin_sensitivity(
        model, batch, np.asarray([0, 1]), np.asarray([1, 0]), temporal_shift=0,
    )
    with torch.no_grad():
        expected = model.forward_with_temporal_shift(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch["extra"], temporal_shift=0, return_dict=True,
        )["logits"]
    torch.testing.assert_close(torch.from_numpy(result["logits"]), expected)
    assert result["response_gradient"].shape == (2, 16, 6)
    np.testing.assert_array_equal(
        result["response_gradient_flat"][:, 3],
        result["response_gradient"][:, 3, 0],
    )
    np.testing.assert_array_equal(
        result["response_gradient_flat"][:, 19],
        result["response_gradient"][:, 3, 1],
    )


def test_local_perturbation_requires_joint_event_feasibility():
    torch.manual_seed(19)
    model = _phase_model()
    batch = _tiny_batch(1)
    base = abc.normal_pair_margin_sensitivity(
        model, batch, np.asarray([0]), np.asarray([1]), temporal_shift=0,
    )
    event = base["response_events"][0, 0]
    rng = np.random.default_rng(3)
    source = event + rng.normal(0, .04, size=(80, 6))
    source[:, 1] = source[:, 0] + rng.normal(0, .005, 80)
    support = abc.fit_joint_support(source, quantile=.99, seed=1)
    result = abc.apply_in_range_event_perturbation(
        model, batch, support, anchor_id=0, component_id=0,
        requested_delta=.2, y_id=0, c_id=1,
    )
    assert result["accepted"]
    assert result["joint_feasible"]
    assert abs(result["applied_delta"]) <= .2
    assert np.isfinite(result["actual_margin_delta"])


def test_query_parameter_gradients_do_not_update_or_accumulate_parameters():
    torch.manual_seed(20)
    model = _phase_model()
    batch = _tiny_batch(2)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    packet = abc.sample_query_gradients(
        model, batch, torch.tensor([0, 1]), temporal_shift=0,
    )
    assert set(packet["vectors"]) == {
        "response_to_query", "external_query_projection",
    }
    assert packet["vectors"]["response_to_query"].shape[0] == 2
    assert packet["vectors"]["external_query_projection"].shape[0] == 2
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_target_gradient_groups_use_only_unlabeled_scores_and_are_class_conditioned():
    scores = {
        "available": np.asarray([True, True, True, False]),
        "conflict": np.asarray([False, True, False, False]),
        "out_of_range": np.asarray([False, False, True, False]),
        "margin": np.asarray([.8, -.2, .4, np.nan]),
    }
    prediction = np.asarray([0, 1, 0, 1])
    confidence = np.asarray([.95, .99, .92, .98])
    groups = abc.build_target_gradient_groups(
        scores, prediction, confidence, threshold=.5,
    )
    assert groups["supported"].tolist() == [True, False, False, False]
    assert groups["conflict_or_unsupported"].tolist() == [False, True, True, True]

    packets = {
        "source": {"vectors": {"response_to_query": np.ones((2, 3))}},
        "target_supported": {"vectors": {"response_to_query": np.ones((2, 3))}},
        "target_conflict": {"vectors": {"response_to_query": -np.ones((2, 3))}},
    }
    rows = abc.summarize_gradient_groups(
        packets,
        {"source": np.asarray([0, 1]), "target_supported": np.asarray([0, 1]),
         "target_conflict": np.asarray([0, 1])},
    )
    assert any(row["scope"] == "overall" for row in rows)
    assert any(row["scope"] == "class" and row["class_id"] == 0 for row in rows)


def test_stage_artifact_contract_and_resume_preserve_completed_peer(tmp_path):
    assert set(abc.required_stage_artifacts("A")) == {
        "A_anchor_event_semantics.csv", "A_pairwise_discrimination.csv",
        "A_transferability_validation.csv", "A_anchor_candidates.json",
        "A_representative_windows.png", "A_findings.md",
    }
    assert set(abc.required_stage_artifacts("B")) == {
        "B_source_calibration.json", "B_sample_scores.csv",
        "B_coverage_error_curves.csv", "B_per_class_reliability.csv",
        "B_conflict_cases.csv", "B_findings.md",
    }
    assert set(abc.required_stage_artifacts("C")) == {
        "C_pairwise_model_sensitivity.csv", "C_local_perturbations.csv",
        "C_gradient_groups.csv", "C_gradient_samples.csv", "C_findings.md",
    }
    state_path = tmp_path / "stage_status.json"
    abc.update_stage_status(state_path, "extract", "complete")
    abc.update_stage_status(state_path, "A", "complete")
    abc.update_stage_status(state_path, "B", "failed", error="synthetic")
    state = json.loads(state_path.read_text())
    assert state["stages"]["A"]["status"] == "complete"
    assert state["stages"]["B"]["error"] == "synthetic"
    assert abc.pending_checkpoint_stages(state) == ["B", "C", "report"]


@pytest.mark.parametrize(
    "evidence,expected",
    [
        ({"stable_candidates": 0}, "insufficient_evidence"),
        ({"stable_candidates": 2, "b_incremental": True,
          "conflict_gradient_anomaly": True}, "target_query_gradient_control"),
        ({"stable_candidates": 2, "uses_unstable_evidence": True},
         "query_evidence_consistency"),
        ({"stable_candidates": 1, "b_incremental": False},
         "structure_representation_alignment"),
    ],
)
def test_summary_selects_exactly_one_permitted_direction(evidence, expected):
    assert abc.choose_next_method(evidence) == expected


def test_checkpoint_report_is_composed_from_completed_stage_findings(tmp_path):
    for stage in "ABC":
        (tmp_path / f"{stage}_findings.md").write_text(
            f"# {stage}\n\nFinding {stage}.\n", encoding="utf-8",
        )
    path = abc.write_checkpoint_report(tmp_path, "E", "FR2_DK1")
    text = path.read_text(encoding="utf-8")
    assert "E FR2_DK1" in text
    assert all(f"Finding {stage}." in text for stage in "ABC")


def test_abc_scheduler_graph_has_no_barrier_and_never_builds_training_command(tmp_path):
    jobs = scheduler.build_job_graph()
    keys = {job.key for job in jobs}
    assert len(keys) == 23
    assert scheduler.INITIAL_GPU_ASSIGNMENT == {
        "extract:E:FR1_FR2": "0", "extract:E:FR2_DK1": "1",
        "extract:G:FR1_FR2": "2", "extract:G:FR2_DK1": "3",
    }
    completed = {"extract:E:FR2_DK1"}
    running = {
        "extract:E:FR1_FR2", "extract:G:FR1_FR2", "extract:G:FR2_DK1",
    }
    ready = {job.key for job in scheduler.ready_jobs(jobs, completed, running)}
    assert "A:E:FR2_DK1" in ready
    assert "A:E:FR1_FR2" not in ready

    pair_ready = {job.key for job in scheduler.ready_jobs(
        jobs, {"report:E:FR2_DK1", "report:G:FR2_DK1"}, set(),
    )}
    assert "pair:FR2_DK1" in pair_ready
    args = scheduler.build_parser().parse_args([
        "dry-run", "--experiment-root", str(tmp_path),
    ])
    for job in jobs:
        assert "train.py" not in " ".join(scheduler.job_command(job, args))


def test_scheduler_dynamic_gpu_choice_and_resume_state(tmp_path):
    job = next(
        row for row in scheduler.build_job_graph()
        if row.key == "C:E:FR1_FR2"
    )
    assert scheduler.select_gpu(
        ("0", "2"), {"0": 10.0, "2": 4.0}, job,
    ) == "2"
    state_path = tmp_path / "scheduler_state.json"
    scheduler.save_state(state_path, {
        "completed": ["extract:E:FR1_FR2"], "failed": {},
        "gpu_seconds": {"0": 3.0, "1": 0.0, "2": 0.0, "3": 0.0},
    })
    state = scheduler.load_state(state_path, ("0", "1", "2", "3"))
    assert state["completed"] == ["extract:E:FR1_FR2"]
    assert state["gpu_seconds"]["0"] == 3.0


def test_preflight_reports_missing_checkpoints_data_and_four_gpus(tmp_path):
    args = scheduler.build_parser().parse_args([
        "dry-run", "--experiment-root", str(tmp_path / "experiment"),
        "--data-root", str(tmp_path / "missing-data"),
    ])
    result = scheduler.preflight(args, cuda_available=True, cuda_count=1)
    codes = {row["code"] for row in result["blockers"]}
    assert "missing_data_root" in codes
    assert "missing_checkpoint" in codes
    assert "insufficient_cuda_devices" in codes
    assert result["training_commands_present"] is False


def test_checkpoint_json_accepts_paths_numpy_and_config_namespaces(tmp_path):
    path = tmp_path / "manifest.json"
    abc._write_json(path, {
        "path": Path("checkpoint.pt"),
        "scalar": np.int64(7),
        "array": np.asarray([1.0, 2.0]),
        "config": SimpleNamespace(source="FR1"),
    })
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "array": [1.0, 2.0],
        "config": {"source": "FR1"},
        "path": "checkpoint.pt",
        "scalar": 7,
    }


def test_cli_entrypoint_runs_only_after_all_stage_definitions():
    source = Path(abc.__file__).read_text(encoding="utf-8")
    assert source.rindex('if __name__ == "__main__":') > source.index("def run_a_stage(")


def test_analysis_cli_can_be_launched_as_a_direct_script():
    script = Path(abc.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--mode" in result.stdout
