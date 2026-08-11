from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_train_torch_load_calls_use_cpu_map_location() -> None:
    """Training checkpoints must not depend on the GPU index used when saving."""

    tree = ast.parse((REPO_ROOT / "train.py").read_text(encoding="utf-8"))
    torch_load_calls = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
            and func.attr == "load"
        ):
            continue
        torch_load_calls.append(node)

    assert torch_load_calls, "expected at least one torch.load call in train.py"
    for call in torch_load_calls:
        kwargs = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
        assert "map_location" in kwargs, (
            f"torch.load at train.py:{call.lineno} is missing map_location"
        )
        map_location = kwargs["map_location"]
        assert isinstance(map_location, ast.Constant)
        assert map_location.value == "cpu", (
            f"torch.load at train.py:{call.lineno} must deserialize via CPU"
        )
