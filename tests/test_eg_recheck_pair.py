from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import analysis.diagnose_eg_pair as eg_diagnostic
from analysis.diagnose_eg_pair import (
    analyze_d2,
    decompose_linear_margin,
    neutral_query_intervention,
    require_complete_checkpoint,
    run_independent_diagnostic_stages,
    sha256_file,
    validate_uda_checkpoint,
)
from models.stclassifier import PseStructureProtoLTae
import scripts.run_eg_recheck_pair_seed1 as scheduler
from scripts.run_eg_recheck_pair_seed1 import (
    INITIAL_GPU_ASSIGNMENT,
    _quarantine_incomplete,
    build_parser,
    build_job_graph,
    diagnostic_command,
    paired_uda_config,
    ready_jobs,
    select_gpu,
    source_checkpoint_path,
)


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


def test_e_g_config_diff_is_only_detach_epochs():
    e = paired_uda_config("E")
    g = paired_uda_config("G")
    differences = {key for key in e if e[key] != g[key]}
    assert differences == {"target_structure_detach_epochs"}
    assert e["target_structure_detach_epochs"] == 0
    assert g["target_structure_detach_epochs"] == 5
    assert e["detach_target_structure"] is False


def test_missing_git_metadata_does_not_block_preflight(monkeypatch):
    def unavailable(*args, **kwargs):
        raise scheduler.subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(scheduler.subprocess, "check_output", unavailable)
    assert scheduler.git_commit_or_unavailable() == "unavailable"


def test_actual_checkpoint_configuration_is_validated_not_assumed():
    config = paired_uda_config("G")
    config.update({
        "source": "france/31TCJ/2017",
        "target": "denmark/32VNH/2017",
    })
    packet = {"epoch": 19, "config": config}
    assert validate_uda_checkpoint(packet, "FR2_DK1", "G")[
        "target_structure_detach_epochs"
    ] == 5
    packet["config"] = {**config, "lr": .001}
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_uda_checkpoint(packet, "FR2_DK1", "G")


def test_sha256_is_stable_and_same_source_can_be_bound_to_pair():
    checkpoint = Path("tests/.eg_recheck_sha_model.pt")
    try:
        checkpoint.write_bytes(b"paired-source")
        first = sha256_file(checkpoint)
        second = sha256_file(checkpoint)
        assert first == second
        assert len(first) == 64
        assert source_checkpoint_path("source-root", "FR1_FR2") == Path(
            "source-root/source_FR1_seed1/fold_0/model.pt"
        )
    finally:
        checkpoint.unlink(missing_ok=True)


def test_final_checkpoint_requires_epoch_19():
    path = Path("tests/checkpoint_last.pt")
    try:
        torch.save({"epoch": 19, "config": {"epochs": 20}}, path)
        assert require_complete_checkpoint(path)["epoch"] == 19
        torch.save({"epoch": 18, "config": {"epochs": 20}}, path)
        with pytest.raises(ValueError, match="epoch=19"):
            require_complete_checkpoint(path)
    finally:
        path.unlink(missing_ok=True)


def test_d1_block_contributions_reconstruct_full_margin():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(48, 96))
    y = np.repeat(np.arange(3), 16)
    x[y == 1, :16] += 1.5
    x[y == 2, 32:] -= .8
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RidgeClassifier(alpha=1.0)),
    ]).fit(x, y)
    rows = decompose_linear_margin(estimator, x[:12], y[:12])
    reconstructed = (
        rows["strength_contribution"] + rows["concentration_contribution"]
        + rows["phase_contribution"] + rows["bias_contribution"]
    )
    np.testing.assert_allclose(reconstructed, rows["margin"], atol=1e-8)
    assert np.all(rows["competitor"] != y[:12])


def test_d1_reconstruction_uses_roundoff_bound_not_fixed_tolerance():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(1, 96))
    true_weight = rng.normal(size=96) * 1e8
    true_weight[-1] = (
        1.0 - np.sum(features[0, :-1] * true_weight[:-1])
    ) / features[0, -1]

    class IdentityScaler:
        @staticmethod
        def transform(values):
            return values

    class LinearClassifier:
        classes_ = np.arange(3)
        coef_ = np.stack((true_weight, np.zeros(96), np.zeros(96)))
        intercept_ = np.asarray((0.0, 0.0, -1e20))

        def decision_function(self, values):
            return values @ self.coef_.T + self.intercept_

    class Estimator:
        named_steps = {
            "scaler": IdentityScaler(), "classifier": LinearClassifier(),
        }

        def decision_function(self, values):
            return self.named_steps["classifier"].decision_function(values)

    result = decompose_linear_margin(Estimator(), features, np.asarray([0]))
    assert result["margin_reconstruction_error"][0] > 1e-7
    assert result["margin_reconstruction_error"][0] <= result[
        "margin_reconstruction_bound"
    ][0]


def test_d1_reconstruction_uses_actual_float32_decision_precision():
    rng = np.random.default_rng(19)
    features = rng.normal(size=(8, 96)).astype(np.float32)
    weights = rng.normal(size=(3, 96)).astype(np.float32)
    intercept = rng.normal(size=3).astype(np.float32)

    class IdentityScaler:
        @staticmethod
        def transform(values):
            return values

    class Float32LinearClassifier:
        classes_ = np.arange(3)
        coef_ = weights
        intercept_ = intercept

        def decision_function(self, values):
            values = np.asarray(values, dtype=np.float32)
            return values @ self.coef_.T + self.intercept_

    class Estimator:
        named_steps = {
            "scaler": IdentityScaler(), "classifier": Float32LinearClassifier(),
        }

        def decision_function(self, values):
            return self.named_steps["classifier"].decision_function(values)

    result = decompose_linear_margin(
        Estimator(), features, np.arange(len(features)) % 3,
    )
    assert result["margin_reconstruction_error"].max() > np.finfo(np.float64).eps
    assert np.all(
        result["margin_reconstruction_error"]
        <= result["margin_reconstruction_bound"]
    )


def test_d2_trusted_wrong_semantic_rate_uses_same_class_window_mask(monkeypatch):
    rng = np.random.default_rng(23)

    def split(labels, domain, offset):
        labels = np.asarray(labels, dtype=np.int64)
        samples, windows = len(labels), 3
        similarity = rng.normal(size=(samples, windows, 16)).astype(np.float32)
        similarity[..., 0] += offset
        weights = np.exp(similarity)
        weights /= weights.sum(axis=1, keepdims=True)
        distribution = np.exp(similarity)
        distribution /= distribution.sum(axis=-1, keepdims=True)
        return {
            "label": labels,
            "domain": domain,
            "parcel_id": np.arange(samples),
            "similarity": similarity,
            "weights": weights,
            "anchor_distribution": distribution,
            "response": rng.normal(size=(samples, 96)).astype(np.float32),
            "window_centers_days": np.arange(windows, dtype=np.float32),
            "window_curve_norm": rng.normal(size=(samples, windows, 5)).astype(np.float32),
        }

    arrays = {
        "source_train": split([0, 0, 1, 1], "source", 0.0),
        "source_val": split([0, 1, 1], "source", 1.0),
        "target_val": split([0, 0, 1, 1, 1], "target", 1.0),
    }
    output = Path("tests/.eg_d2_output")
    output.mkdir(exist_ok=True)
    monkeypatch.setattr(eg_diagnostic, "_plot_d2_examples", lambda *_: None)
    try:
        _, _, associations = analyze_d2(
            "FR1_FR2", "E", arrays, ["class0", "class1"], output,
        )
        assert len(associations) == 4
        assert all(
            np.isnan(row["wrong_source_semantic_rate_among_trusted"])
            or 0.0 <= row["wrong_source_semantic_rate_among_trusted"] <= 1.0
            for row in associations
        )
    finally:
        for path in output.iterdir():
            path.unlink()
        output.rmdir()


def test_diagnostic_stages_continue_after_one_stage_fails():
    calls = []

    def fail():
        calls.append("D1")
        raise RuntimeError("broken D1")

    def pass_d2():
        calls.append("D2")
        return "d2"

    def pass_d3():
        calls.append("D3")
        return "d3"

    results, statuses = run_independent_diagnostic_stages({
        "D1": fail, "D2": pass_d2, "D3": pass_d3,
    })
    assert calls == ["D1", "D2", "D3"]
    assert statuses["D1"]["status"] == "failed"
    assert statuses["D2"]["status"] == "complete"
    assert statuses["D3"]["status"] == "complete"
    assert results == {"D2": "d2", "D3": "d3"}


def test_d3_neutral_removes_query_and_normal_matches_model():
    torch.manual_seed(5)
    model = _phase_model()
    pixels = torch.randn(2, 10, 3, 4)
    valid = torch.ones(2, 10, 4)
    positions = torch.arange(10).repeat(2, 1) * 30
    extra = torch.zeros(2, 4)
    with torch.no_grad():
        expected = model.forward_with_temporal_shift(
            pixels, valid, positions, extra, temporal_shift=0, return_dict=True,
        )["logits"]
        spatial = model.spatial_encoder(pixels, valid, extra)
        structure = model.prepare_structure(spatial, positions, 0)
        result = neutral_query_intervention(model, spatial, positions, structure)
    torch.testing.assert_close(result["normal_logits"], expected)
    assert result["query_correction_norm"].min() > 0
    assert not torch.allclose(result["normal_logits"], result["neutral_logits"])
    assert result["normal_attention"].shape == result["neutral_attention"].shape


def test_diagnostic_rejects_unfinished_checkpoint():
    missing = Path("tests/missing_checkpoint_last.pt")
    missing.unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        require_complete_checkpoint(missing)


def test_fr2_diagnostic_ready_without_waiting_for_fr1():
    jobs = build_job_graph()
    completed = {"uda:E:FR2_DK1"}
    running = {"uda:E:FR1_FR2", "uda:G:FR1_FR2", "uda:G:FR2_DK1"}
    ready = {job.key for job in ready_jobs(jobs, completed, running)}
    assert "diagnostic:E:FR2_DK1" in ready
    assert "diagnostic:E:FR1_FR2" not in ready


def test_resume_does_not_repeat_completed_training_and_balances_fr1():
    jobs = build_job_graph()
    completed = {"uda:E:FR2_DK1", "diagnostic:E:FR2_DK1"}
    ready = {job.key for job in ready_jobs(jobs, completed, set())}
    assert "uda:E:FR2_DK1" not in ready
    assert "diagnostic:E:FR2_DK1" not in ready
    assert INITIAL_GPU_ASSIGNMENT["uda:E:FR1_FR2"] == "0"
    gpu = select_gpu(
        free_gpus=("1", "3"), gpu_seconds={"1": 500., "3": 100.},
        job=next(job for job in jobs if job.key == "diagnostic:E:FR1_FR2"),
    )
    assert gpu == "3"
    parsed = build_parser().parse_args([
        "resume", "--only-task", "FR2_DK1", "--retry-failed",
    ])
    assert parsed.retry_failed is True
    assert parsed.only_task == "FR2_DK1"
    retry_all = build_parser().parse_args(["resume", "--retry-failed"])
    assert retry_all.retry_failed is True
    assert retry_all.only_task is None


def test_partial_diagnostic_resume_only_retries_failed_stages():
    root = Path("tests/.eg_partial_resume")
    diagnostic = root / "diagnostics/E_FR1_FR2"
    log = root / "logs/DIAG_E_FR1_FR2_seed1.log"
    diagnostic.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    (diagnostic / "diagnostic_manifest.json").write_text(
        '{"status":"partial","stage_status":{'
        '"D1":{"status":"failed"},"D2":{"status":"failed"},'
        '"D3":{"status":"complete"}}}',
        encoding="utf-8",
    )
    log.write_text("old failure", encoding="utf-8")
    args = SimpleNamespace(
        experiment_root=str(root), source_root="source-root",
        python="python", data_root="data-root", audit_batch_size=64,
    )
    job = next(
        item for item in build_job_graph()
        if item.key == "diagnostic:E:FR1_FR2"
    )
    try:
        command = diagnostic_command(job, args)
        stages = command[command.index("--stages") + 1:]
        assert stages == ["D1", "D2"]
        _quarantine_incomplete(job, args)
        assert diagnostic.is_dir()
        assert (diagnostic / "diagnostic_manifest.json").is_file()
    finally:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        root.rmdir()
