from pathlib import Path


def test_reimts_closedset_smoke_script_exists():
    script = Path("closedset_scripts/smoke_reimts_mtan_closedset.sh").resolve()

    assert script.is_file()


def test_reimts_closedset_smoke_enables_round_two_diagnostics():
    script = Path("closedset_scripts/smoke_reimts_mtan_closedset.sh").read_text()

    assert script.count("--reimts_loss_mode patch") == 2
    assert script.count("--reimts_patch_diagnostics true") == 2
