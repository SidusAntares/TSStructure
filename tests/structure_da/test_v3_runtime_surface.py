from __future__ import annotations

import inspect
from pathlib import Path

import methods.structure_da as structure_da
from methods.structure_da.joint_trainer import PerClassPhaseDiagnosticsAccumulator


LEGACY_HIGH_LEVEL_SYMBOLS = (
    "Shared" + "TemporalStructureOperator",
    "TemporalStructure" + "Extractor",
    "TemporalStructure" + "PairOutput",
    "TemporalGeometry" + "PairOutput",
    "PhaseCoordinate" + "Encoder",
    "TemporalStructure" + "Encoder",
    "QualityAwareComponent" + "Classifier",
    "HierarchicalQuality" + "Fusion",
    "ComponentAwareShared" + "LTAE",
)


def test_package_does_not_export_legacy_high_level_symbols() -> None:
    assert not (set(LEGACY_HIGH_LEVEL_SYMBOLS) & set(structure_da.__all__))
    assert all(not hasattr(structure_da, name) for name in LEGACY_HIGH_LEVEL_SYMBOLS)


def test_runtime_tree_does_not_reference_legacy_high_level_symbols() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    runtime_source = "\n".join(
        path.read_text(encoding="utf-8")
        for root in ("methods", "models")
        for path in (repository_root / root).rglob("*.py")
    )
    assert all(name not in runtime_source for name in LEGACY_HIGH_LEVEL_SYMBOLS)


def test_per_class_diagnostics_batch_update_has_no_sample_list_conversion() -> None:
    source = inspect.getsource(PerClassPhaseDiagnosticsAccumulator.update)
    assert ".tolist()" not in source
