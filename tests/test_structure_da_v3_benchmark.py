from __future__ import annotations

import json
import subprocess
import sys


def test_v3_benchmark_runs_on_synthetic_cpu_data_and_outputs_json() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_structure_da_v3.py",
            "--device",
            "cpu",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["device"] == "cpu"
    assert payload["iterations"] == 1
    assert payload["whole_train_iteration_ms"]["median"] > 0
    assert payload["peak_memory_mib"] is None
