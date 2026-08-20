import ast
from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_timematch_phase_4tasks_4gpu_seed1.sh"
)


def test_launcher_skips_tasks_with_complete_stage2_metrics():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'if [[ -f "$stage2_root/test_metrics.json" ]]' in text
    assert "TIMEMATCH_PHASE_TASK_SKIP" in text


def test_fullbank_selected_loader_uses_current_runtime_contract():
    trainer = LAUNCHER.parent / "train_timematch_phase.py"
    tree = ast.parse(trainer.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "selected_loader"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert len(call.args) == 5
    assert isinstance(call.args[4], ast.Name) and call.args[4].id == "runtime"
    assert {keyword.arg for keyword in call.keywords} == {
        "batch_size",
        "num_workers",
    }


def test_trainer_calls_match_all_current_timematch_runtime_signatures():
    trainer_tree = ast.parse(
        (LAUNCHER.parent / "train_timematch_phase.py").read_text(encoding="utf-8")
    )
    runtime_tree = ast.parse(
        (LAUNCHER.parent / "timematch_runtime.py").read_text(encoding="utf-8")
    )
    definitions = {
        node.name: node
        for node in runtime_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    checked = 0
    for call in (node for node in ast.walk(trainer_tree) if isinstance(node, ast.Call)):
        if not (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "tm"
            and call.func.attr in definitions
        ):
            continue
        definition = definitions[call.func.attr]
        positional = definition.args.posonlyargs + definition.args.args
        keyword_names = {
            argument.arg
            for argument in positional + definition.args.kwonlyargs
        }
        assert definition.args.vararg is not None or len(call.args) <= len(positional), call.func.attr
        if definition.args.kwarg is None:
            assert all(keyword.arg in keyword_names for keyword in call.keywords), call.func.attr
        checked += 1
    assert checked >= 10
