import hashlib
import inspect
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import pytest
import torch


def _manual_reference(mu, tau, weight, support):
    from methods.structure_da.class_organization_loss import ClassOrganizationReference

    return ClassOrganizationReference(
        mu=torch.as_tensor(mu, dtype=torch.float32),
        tau=torch.as_tensor(tau, dtype=torch.float32),
        weight=torch.as_tensor(weight, dtype=torch.float32),
        support=torch.as_tensor(support, dtype=torch.long),
        metadata={
            "class_ids": list(range(len(support))),
            "anchor_count": 16,
            "response_dim": 96,
            "source_checkpoint_sha256": "test-sha",
            "source_org_effective": True,
        },
    )


def test_response96_decodes_block_major_magnitudes():
    from methods.structure_da.class_organization_loss import response96_to_organization

    response = torch.zeros(1, 96)
    response[:, 0:16] = torch.arange(16)
    response[:, 16:32] = 100 + torch.arange(16)
    response[:, 32:48] = 3
    response[:, 48:64] = 4
    response[:, 64:80] = 5
    response[:, 80:96] = 12

    organization = response96_to_organization(response)

    assert organization.shape == (1, 16, 4)
    torch.testing.assert_close(organization[0, :, 0], torch.arange(16).float())
    torch.testing.assert_close(organization[0, :, 1], 100 + torch.arange(16).float())
    torch.testing.assert_close(organization[0, :, 2], torch.full((16,), 5.0))
    torch.testing.assert_close(organization[0, :, 3], torch.full((16,), 13.0))


def test_phase_magnitudes_are_invariant_to_global_rotation():
    from methods.structure_da.class_organization_loss import response96_to_organization

    generator = torch.Generator().manual_seed(7)
    response = torch.randn(5, 96, generator=generator)
    angle = torch.tensor(0.73)
    rotated = response.clone()
    for cosine, sine in ((32, 48), (64, 80)):
        x = response[:, cosine:cosine + 16]
        y = response[:, sine:sine + 16]
        rotated[:, cosine:cosine + 16] = x * angle.cos() - y * angle.sin()
        rotated[:, sine:sine + 16] = x * angle.sin() + y * angle.cos()

    torch.testing.assert_close(
        response96_to_organization(response),
        response96_to_organization(rotated),
        atol=1e-6,
        rtol=1e-6,
    )


def test_reference_builder_zeroes_unsupported_and_nondiscriminative_statistics():
    from methods.structure_da.class_organization_loss import build_organization_reference

    response = torch.zeros(13, 96)
    labels = torch.tensor([0] * 6 + [1] * 6 + [2])
    response[:6, 0] = torch.linspace(0.0, 0.1, 6)
    response[6:12, 0] = torch.linspace(2.0, 2.1, 6)
    reference = build_organization_reference(
        response, labels, num_classes=3, min_support=4,
    )

    assert reference.support.tolist() == [6, 6, 1]
    assert reference.weight[0, 0, 0] > 0
    assert reference.weight[1, 0, 0] > 0
    assert torch.count_nonzero(reference.weight[2]) == 0
    assert torch.count_nonzero(reference.weight[:, 1:, :]) == 0
    assert list(reference.parameters()) == []
    assert not reference.mu.requires_grad


def test_organization_loss_is_zero_inside_tolerance_and_has_outward_gradient():
    from methods.structure_da.class_organization_loss import class_organization_loss

    shape = (1, 16, 4)
    reference = _manual_reference(
        mu=torch.zeros(shape), tau=torch.ones(shape),
        weight=torch.ones(shape), support=[10],
    )
    inside = torch.zeros(2, 96, requires_grad=True)
    inside_result = class_organization_loss(
        inside, torch.zeros(2, dtype=torch.long), reference,
    )
    assert inside_result["loss"].item() == 0.0

    outside = torch.zeros(2, 96, requires_grad=True)
    outside.data[:, 0] = 3.0
    result = class_organization_loss(
        outside, torch.zeros(2, dtype=torch.long), reference,
    )
    assert result["loss"].item() > 0
    result["loss"].backward()
    assert outside.grad[:, 0].abs().sum() > 0
    assert reference.mu.grad is None
    assert reference.weight.grad is None


def test_empty_trusted_target_batch_returns_differentiable_zero():
    from methods.structure_da.class_organization_loss import target_organization_objective

    reference = _manual_reference(
        mu=torch.zeros(2, 16, 4), tau=torch.ones(2, 16, 4),
        weight=torch.ones(2, 16, 4), support=[10, 10],
    )
    response = torch.randn(3, 96, requires_grad=True)
    objective = target_organization_objective(
        response, pseudo_labels=torch.tensor([0, 1, 0]),
        trusted_mask=torch.zeros(3, dtype=torch.bool), reference=reference,
        epoch=8, configured_weight=0.02, start_epoch=5,
    )

    assert objective["loss"].item() == 0.0
    assert objective["weighted_loss"].item() == 0.0
    objective["weighted_loss"].backward()
    assert response.grad is not None
    assert torch.count_nonzero(response.grad) == 0


def test_target_organization_is_strictly_off_for_first_five_epochs():
    from methods.structure_da.class_organization_loss import target_organization_objective

    reference = _manual_reference(
        mu=torch.zeros(1, 16, 4), tau=torch.ones(1, 16, 4),
        weight=torch.ones(1, 16, 4), support=[10],
    )
    response = torch.zeros(2, 96, requires_grad=True)
    response.data[:, 0] = 4.0
    labels = torch.zeros(2, dtype=torch.long)
    trusted = torch.ones(2, dtype=torch.bool)

    for epoch in range(5):
        objective = target_organization_objective(
            response, labels, trusted, reference, epoch, 0.02, start_epoch=5,
        )
        assert objective["loss"].item() == 0.0
        assert objective["effective_weight"] == 0.0
    enabled = target_organization_objective(
        response, labels, trusted, reference, 5, 0.02, start_epoch=5,
    )
    assert enabled["loss"].item() > 0
    assert enabled["effective_weight"] == pytest.approx(0.02)


def test_reference_roundtrip_validates_checkpoint_sha_and_dimensions():
    from methods.structure_da.class_organization_loss import (
        load_organization_reference,
        save_organization_reference,
        sha256_file,
    )

    work = Path.cwd() / f".class_org_test_{uuid.uuid4().hex}"
    work.mkdir()
    try:
        checkpoint = work / "model.pt"
        checkpoint.write_bytes(b"source-best")
        digest = hashlib.sha256(b"source-best").hexdigest()
        reference = _manual_reference(
            mu=torch.zeros(2, 16, 4), tau=torch.ones(2, 16, 4),
            weight=torch.ones(2, 16, 4), support=[10, 10],
        )
        reference.metadata["source_checkpoint_sha256"] = digest
        path = work / "class_organization_reference.pt"
        save_organization_reference(reference, path)

        restored = load_organization_reference(
            path, source_checkpoint=checkpoint, expected_class_ids=[0, 1],
            expected_anchor_count=16, expected_response_dim=96,
        )
        assert sha256_file(checkpoint) == digest
        torch.testing.assert_close(restored.mu, reference.mu)
        with pytest.raises(ValueError, match="class order"):
            load_organization_reference(
                path, source_checkpoint=checkpoint, expected_class_ids=[1, 0],
                expected_anchor_count=16, expected_response_dim=96,
            )
        checkpoint.write_bytes(b"changed")
        with pytest.raises(ValueError, match="SHA256"):
            load_organization_reference(
                path, source_checkpoint=checkpoint, expected_class_ids=[0, 1],
                expected_anchor_count=16, expected_response_dim=96,
            )
    finally:
        shutil.rmtree(work)


def test_default_disabled_objective_is_exact_zero_without_reference():
    from methods.structure_da.class_organization_loss import target_organization_objective

    response = torch.randn(2, 96, requires_grad=True)
    result = target_organization_objective(
        response, torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool),
        reference=None, epoch=12, configured_weight=0.0, start_epoch=5,
    )
    assert result["loss"].item() == 0.0
    assert result["weighted_loss"].item() == 0.0


def test_ab_commands_share_exact_source_checkpoint_and_reference():
    from scripts.run_structure_g_class_org_seed1 import pipelines, uda_command

    args = SimpleNamespace(
        python="python", data_root="/data", root=Path("experiments/org"),
    )
    pipeline = pipelines()[0]
    command_a = uda_command(args, pipeline, "A")
    command_b = uda_command(args, pipeline, "B")

    def value(command, option):
        return command[command.index(option) + 1]

    assert value(command_a, "--weights") == value(command_b, "--weights")
    assert value(command_a, "--class-org-reference") == value(
        command_b, "--class-org-reference",
    )
    assert value(command_a, "--class-org-target-weight") == "0.0"
    assert value(command_b, "--class-org-target-weight") == "0.02"
    assert value(command_a, "--target-structure-detach-epochs") == "5"
    assert value(command_b, "--target-structure-detach-epochs") == "5"


def test_scheduler_has_two_dependency_pipelines_and_resume_modes():
    from scripts.run_structure_g_class_org_seed1 import (
        build_parser,
        pipelines,
        source_assignments,
        uda_assignments,
    )

    planned = pipelines()
    assert [item.task for item in planned] == ["FR1_FR2", "FR2_DK1"]
    assert len({item.source_alias for item in planned}) == 2
    parser = build_parser()
    assert parser.parse_args(["dry-run"]).mode == "dry-run"
    assert parser.parse_args(["run"]).mode == "run"
    assert parser.parse_args(["resume"]).mode == "resume"
    assert parser.parse_args(["dry-run"]).gpus == ["0", "1", "2", "3"]

    gpus = ["0", "1", "2", "3"]
    source_plan = source_assignments(gpus)
    assert [(gpu, pipeline.source_alias) for gpu, pipeline in source_plan] == [
        ("0", "FR1"), ("1", "FR2"),
    ]
    uda_plan = uda_assignments(gpus)
    assert [(gpu, variant, pipeline.task) for gpu, variant, pipeline in uda_plan] == [
        ("0", "A", "FR2_DK1"),
        ("1", "A", "FR1_FR2"),
        ("2", "B", "FR1_FR2"),
        ("3", "B", "FR2_DK1"),
    ]


def test_organization_loss_reaches_pse_token_encoder_and_anchors():
    from methods.structure_da.class_organization_loss import (
        ClassOrganizationReference,
        class_organization_loss,
    )
    from models.stclassifier import PseStructureProtoLTae

    torch.manual_seed(19)
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation="phase_moment", shape_injection="current_query",
        structure_shift_mode="none",
    ).train()
    pixels = torch.randn(3, 10, 3, 4)
    mask = torch.ones(3, 10, 4)
    positions = torch.arange(10).repeat(3, 1) * 30
    output = model(pixels, mask, positions, torch.zeros(3, 4), return_dict=True)
    reference = ClassOrganizationReference(
        torch.zeros(3, 16, 4), torch.full((3, 16, 4), 1e-3),
        torch.ones(3, 16, 4), torch.full((3,), 10),
        {"class_ids": [0, 1, 2], "anchor_count": 16, "response_dim": 96},
    )
    loss = class_organization_loss(
        output["shapelet_response"], torch.tensor([0, 1, 2]), reference,
    )["loss"]
    loss.backward()

    assert model.structure_branch.shapelet_dictionary.anchors.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in model.structure_branch.token_generator.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.spatial_encoder.parameters())


def test_class_organization_cli_defaults_are_disabled():
    import argparse
    from train import add_model_arguments

    parser = add_model_arguments(argparse.ArgumentParser())
    config = parser.parse_args([])
    assert config.class_org_source_weight == 0.0
    assert config.class_org_target_weight == 0.0
    assert config.class_org_warmup_epochs == 10
    assert config.class_org_target_start_epoch == 5


def test_class_organization_reference_loader_is_single_process(monkeypatch):
    import train

    dataset = torch.utils.data.TensorDataset(torch.arange(6))
    monkeypatch.setattr(
        train,
        "GroupByShapesBatchSampler",
        lambda data_source, batch_size: torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(data_source),
            batch_size=batch_size,
            drop_last=False,
        ),
    )

    loader = train.create_class_organization_reference_loader(
        dataset,
        batch_size=2,
    )

    assert loader.num_workers == 0
    assert loader.persistent_workers is False
    assert [batch[0].tolist() for batch in loader] == [[0, 1], [2, 3], [4, 5]]


def test_source_and_uda_training_are_wired_to_the_frozen_reference_without_gt():
    import timematch
    import train

    supervised = inspect.getsource(train.train_supervised)
    uda = inspect.getsource(timematch._train_structure_proto_timematch)
    assert "build_source_organization_reference" in supervised
    assert "loss_class_organization_source" in supervised
    assert "load_organization_reference" in uda
    assert "target_organization_objective" in uda
    assert "target_output[\"shapelet_response\"], pseudo, trusted_mask" in uda
    assert "target_output[\"shapelet_response\"], target_gt" not in uda


def test_ab_diagnostics_reuse_response_organization_audit_with_distinct_scopes():
    from scripts.run_structure_g_class_org_seed1 import diagnostics_commands, pipelines

    args = SimpleNamespace(
        python="python", data_root="/data", root=Path("experiments/org"),
        diagnostic_batch_size=32, diagnostic_pixel_budget=4096,
    )
    pipeline = pipelines()[0]
    commands = diagnostics_commands(args, pipeline, "B")

    assert [item[0] for item in commands] == ["extract", "audit"]
    extract = commands[0][1]
    audit = commands[1][1]
    assert "analysis.structure_g_response_organization_audit" in extract
    assert extract[extract.index("--variant") + 1] == "B"
    assert extract[extract.index("--scope") + 1] == "B_FR1_FR2"
    assert audit[audit.index("--scope") + 1] == "B_FR1_FR2"
    assert "target_test" not in " ".join(extract)


def test_summary_contract_contains_all_requested_artifacts():
    from scripts.run_structure_g_class_org_seed1 import summary_paths

    root = Path("experiments/org")
    names = {path.name for path in summary_paths(root)}
    assert names == {
        "Source_Org_FR1_metrics.csv",
        "Source_Org_FR2_metrics.csv",
        "A_B_2tasks_seed1.csv",
        "per_class_comparison.csv",
        "organization_diagnostics.csv",
        "FINAL_report.md",
    }


def test_target_objective_never_mutates_frozen_source_reference():
    from methods.structure_da.class_organization_loss import target_organization_objective

    reference = _manual_reference(
        mu=torch.zeros(2, 16, 4), tau=torch.ones(2, 16, 4),
        weight=torch.ones(2, 16, 4), support=[10, 10],
    )
    before = {name: value.clone() for name, value in reference.state_dict().items()}
    response = torch.randn(4, 96, requires_grad=True)
    result = target_organization_objective(
        response, torch.tensor([0, 1, 0, 1]), torch.ones(4, dtype=torch.bool),
        reference, epoch=5, configured_weight=.02,
    )
    result["weighted_loss"].backward()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_summary_generation_reads_completed_artifacts_without_model_inference():
    from scripts.run_structure_g_class_org_seed1 import generate_summary, pipelines

    work = Path.cwd() / f".class_org_summary_{uuid.uuid4().hex}"
    baseline = work / "g_baseline"
    baseline_logs = work / "g_logs"
    try:
        for pipeline in pipelines():
            source_fold = work / f"outputs/source/SRC_ORG_{pipeline.source_alias}_seed1/fold_0"
            source_fold.mkdir(parents=True)
            source_config = {
                "classes": ["a", "b"], "shape_representation": "phase_moment",
            }
            torch.save({"epoch": 99, "config": source_config}, source_fold / "checkpoint_last.pt")
            torch.save({"epoch": 15, "config": source_config}, source_fold / "model.pt")
            state = {
                "mu": torch.zeros(2, 16, 4), "tau": torch.ones(2, 16, 4),
                "weight": torch.ones(2, 16, 4), "support": torch.tensor([8, 9]),
            }
            metadata = {"source_org_effective": True, "warmup_epochs": 10}
            packet = {"schema_version": 1, "state_dict": state, "metadata": metadata}
            torch.save(packet, source_fold / "class_organization_reference.pt")
            torch.save(packet, source_fold / "class_organization_reference_training.pt")
            (source_fold / f"test_metrics_{pipeline.source.replace('/', '_')}.json").write_text(
                json.dumps({
                    "macro_f1": .5, "accuracy": .6, "per_class_f1": [.4, .6],
                    "per_class_precision": [.4, .6], "per_class_recall": [.4, .6],
                    "support": [4, 5],
                }), encoding="utf-8",
            )
            source_log = work / f"logs/SRC_ORG_{pipeline.source_alias}_seed1.log"
            source_log.parent.mkdir(parents=True, exist_ok=True)
            source_log.write_text(
                "CLASS_ORG_SOURCE_EPOCH|epoch=10|enabled=true|loss_raw=1|loss_weighted=.02|classification_ratio=.1|valid_statistics=5\n"
                "SHAPE_AUX_EPOCH|epoch=10|domain=source|loss=.2|accuracy=.8|samples=9\n"
                "Validation result: loss=1, acc=.5, f1=.4\n",
                encoding="utf-8",
            )
            for method, target_weight in (("A", 0.), ("B", .02)):
                fold = work / f"outputs/uda/{method}_{pipeline.task}_seed1/fold_0"
                fold.mkdir(parents=True)
                config = {
                    "classes": ["a", "b"], "target_structure_detach_epochs": 5,
                    "class_org_target_weight": target_weight,
                }
                torch.save({"epoch": 19, "config": config}, fold / "checkpoint_last.pt")
                (fold / "test_metrics_final_target.json").write_text(json.dumps({
                    "macro_f1": .7, "accuracy": .8, "per_class_f1": [.6, .8],
                    "per_class_precision": [.6, .8], "per_class_recall": [.6, .8],
                    "support": [4, 5],
                }), encoding="utf-8")
                log = work / f"logs/{method}_{pipeline.task}_seed1.log"
                log.write_text(
                    "SHAPE_V2CLEAN_EPOCH|epoch=0|pseudo_coverage=.2|accepted_pseudo_accuracy=.9|loss_class_organization_target=0\n"
                    "Validation result: loss=1, acc=.5, f1=.6\n",
                    encoding="utf-8",
                )
            fold = baseline / f"G_{pipeline.task}_seed1/fold_0"
            fold.mkdir(parents=True)
            torch.save({"epoch": 19, "config": {"classes": ["a", "b"]}}, fold / "checkpoint_last.pt")
            (fold / "test_metrics_final_target.json").write_text(json.dumps({
                "macro_f1": .65, "accuracy": .7, "per_class_f1": [.6, .7],
                "per_class_precision": [.6, .7], "per_class_recall": [.6, .7],
                "support": [4, 5],
            }), encoding="utf-8")
        args = SimpleNamespace(
            root=work, g_baseline_root=baseline,
            g_baseline_log_root=baseline_logs,
        )
        generate_summary(args)
        expected = {
            "Source_Org_FR1_metrics.csv", "Source_Org_FR2_metrics.csv",
            "A_B_2tasks_seed1.csv", "per_class_comparison.csv",
            "organization_diagnostics.csv", "FINAL_report.md",
        }
        assert expected <= {path.name for path in work.iterdir()}
        assert "FR1_FR2" in (work / "FINAL_report.md").read_text(encoding="utf-8")
    finally:
        shutil.rmtree(work)


def test_resume_status_requires_complete_compatible_source_and_uda_artifacts():
    from methods.structure_da.class_organization_loss import (
        ClassOrganizationReference, save_organization_reference, sha256_file,
    )
    from scripts.run_structure_g_class_org_seed1 import artifact_status

    work = Path.cwd() / f".class_org_status_{uuid.uuid4().hex}"
    try:
        source = work / "source"
        source.mkdir(parents=True)
        source_packet = {
            "epoch": 99,
            "config": {"shape_representation": "phase_moment", "classes": ["a", "b"]},
        }
        torch.save(source_packet, source / "checkpoint_last.pt")
        torch.save(source_packet, source / "model.pt")
        reference = ClassOrganizationReference(
            torch.zeros(2, 16, 4), torch.ones(2, 16, 4),
            torch.ones(2, 16, 4), torch.tensor([8, 8]),
            {
                "class_ids": [0, 1], "anchor_count": 16, "response_dim": 96,
                "source_checkpoint_sha256": sha256_file(source / "model.pt"),
                "source_org_effective": True,
            },
        )
        save_organization_reference(reference, source / "class_organization_reference.pt")
        save_organization_reference(reference, source / "class_organization_reference_training.pt")
        assert artifact_status(source, "source") == "complete"

        uda = work / "uda"
        uda.mkdir()
        uda_packet = {
            "epoch": 19,
            "config": {
                "shape_representation": "phase_moment",
                "class_org_target_weight": .02,
                "target_structure_detach_epochs": 5,
            },
        }
        torch.save(uda_packet, uda / "checkpoint_last.pt")
        torch.save(uda_packet, uda / "model.pt")
        assert artifact_status(uda, "uda", "B") == "complete"
        assert artifact_status(uda, "uda", "A") == "incompatible"
        (uda / "model.pt").unlink()
        assert artifact_status(uda, "uda", "B") == "partial"
    finally:
        shutil.rmtree(work)
