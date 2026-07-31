from pathlib import Path

import numpy as np
import pandas as pd

from analysis.structure_da.log_analysis import build_analysis_tables
from analysis.structure_da.log_parser import parse_task_log
from analysis.structure_da.plotting import (
    class_metric_plot_data,
    plot_log_diagnostics,
)
from scripts.analyze_structure_da import build_parser
from tests.analysis.test_structure_da_log_parser import _mini_log


def test_class_plot_keeps_absent_seed_but_excludes_it_from_summary():
    task = pd.DataFrame(
        {
            "class_name": ["wheat", "wheat", "wheat"],
            "seed": [1, 2, 3],
            "support": [10, 0, 5],
            "f1": [0.8, 0.0, 0.6],
        }
    )

    points, mean, std, absent_seeds = class_metric_plot_data(task, "wheat", "f1")

    assert [point[0] for point in points] == [1, 2, 3]
    assert absent_seeds == (2,)
    assert np.isclose(mean, 0.7)
    assert np.isclose(std, 0.1)


def test_cli_uses_documented_default_output_root():
    args = build_parser().parse_args(["logs", "--experiment-dir", "records"])

    assert args.output_dir == Path("analysis_outputs/structure_da_full_3seeds_v1")


def test_training_diagnostic_plots_cover_new_epoch_signals(tmp_path: Path):
    log_path = tmp_path / "DK1_to_FR2_seed1.log"
    log_path.write_text(_mini_log(10), encoding="utf-8")
    run = parse_task_log(log_path)
    tables = build_analysis_tables([run])

    plot_log_diagnostics([run], tables, tmp_path / "figures")

    training = tmp_path / "figures" / "training" / run.run_name
    for filename in (
        "losses.png",
        "source_train_accuracy_and_validation.png",
        "energy_fractions.png",
        "fusion_norms.png",
        "quality_alpha.png",
        "quality_beta.png",
        "structure_valid_rates.png",
        "geometry_losses.png",
        "domain_accuracy_and_grl.png",
    ):
        assert (training / filename).is_file()
