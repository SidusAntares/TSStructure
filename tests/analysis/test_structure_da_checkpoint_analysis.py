"""Regression tests for removing the obsolete checkpoint diagnosis surface."""

from argparse import _SubParsersAction
from pathlib import Path

import pytest

import analysis.structure_da as analysis_package
from scripts import analyze_structure_da


REPO_ROOT = Path(__file__).resolve().parents[2]


def _commands() -> set[str]:
    parser = analyze_structure_da.build_parser()
    subparsers = next(
        action for action in parser._actions
        if isinstance(action, _SubParsersAction)
    )
    return set(subparsers.choices)


def test_analysis_cli_exposes_supported_commands() -> None:
    assert _commands() == {
        "logs", "raw", "ndvi-decomposition", "ndvi-ts-diagnostic"
    }


def test_analysis_cli_rejects_removed_checkpoint_command() -> None:
    with pytest.raises(SystemExit):
        analyze_structure_da.build_parser().parse_args(["checkpoint"])


def test_analysis_help_omits_checkpoint_command() -> None:
    assert "checkpoint" not in analyze_structure_da.build_parser().format_help()


def test_checkpoint_analysis_module_is_deleted() -> None:
    assert not (
        REPO_ROOT / "analysis" / "structure_da" / "checkpoint_analysis.py"
    ).exists()


def test_analysis_package_has_no_checkpoint_api() -> None:
    assert not hasattr(analysis_package, "checkpoint_analysis")
    assert all("checkpoint" not in name for name in analysis_package.__all__)


def test_analysis_cli_source_has_no_checkpoint_runtime_import() -> None:
    source = (REPO_ROOT / "scripts" / "analyze_structure_da.py").read_text(
        encoding="utf-8"
    )
    assert "checkpoint_analysis" not in source
    assert "run_checkpoint_analysis" not in source
