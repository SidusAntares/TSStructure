import importlib
import sys


LEGACY_SYMBOLS = {
    "Structure" + "DAModel",
    "Structure" + "DAForwardOutput",
    "Structure" + "DATrainingConfig",
    "Structure" + "DATrainStepOutput",
    "ResolvedStructure" + "DATraining",
    "ComponentStructure" + "Classifier",
    "ComponentStructureOutput",
    "ComponentQualityPerception",
    "StructuralQualityPerception",
    "Diversity" + "Scorer",
    "StructuralAdversarial" + "Adapter",
    "SDADiscriminator",
    "apply_quality_" + "warmup",
    "quality_gate_" + "progress",
    "component_diversity_" + "loss",
    "create_structure_da_train_loaders",
    "train_structure_da",
}

LEGACY_MODULES = {
    "adaptation",
    "losses",
    "method",
    "model",
    "quality",
    "schedules",
    "structure_ops",
    "trainer",
}


def test_public_api_exports_only_real_unique_symbols() -> None:
    module = importlib.import_module("methods.structure_da")

    assert len(module.__all__) == len(set(module.__all__))
    assert all(hasattr(module, name) for name in module.__all__)
    assert LEGACY_SYMBOLS.isdisjoint(module.__all__)


def test_importing_public_api_does_not_import_legacy_modules() -> None:
    importlib.import_module("methods.structure_da")

    imported = {
        name.rsplit(".", 1)[-1]
        for name in sys.modules
        if name.startswith("methods.structure_da.")
    }
    assert LEGACY_MODULES.isdisjoint(imported)
