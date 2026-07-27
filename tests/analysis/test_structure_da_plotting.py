from pathlib import Path

import numpy as np
import pandas as pd

from analysis.structure_da.plotting import class_metric_plot_data
from scripts.analyze_structure_da import build_parser


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
