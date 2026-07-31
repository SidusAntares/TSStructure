"""Parse immutable Structure DA task logs into typed records."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Optional

import numpy as np


@dataclass
class EpochRecord:
    epoch: int
    steps: int
    total: float
    task: float
    quality: float
    geometry: float
    alignment: float
    domain_accuracy: float
    alpha_T: float
    alpha_D: float
    alpha_R: float
    beta_T_temp: float
    beta_D_temp: float
    beta_T_channel: float
    beta_D_channel: float
    grl: float
    lr: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_macro_f1: Optional[float] = None


@dataclass(frozen=True)
class ClassMetric:
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ParsedRun:
    path: Path
    run_name: str
    source: str
    target: str
    seed: int
    status: str
    config: dict[str, Any] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    epochs: list[EpochRecord] = field(default_factory=list)
    class_metrics: list[ClassMetric] = field(default_factory=list)
    confusion: Optional[np.ndarray] = None
    target_accuracy: Optional[float] = None
    target_macro_f1: Optional[float] = None
    git_head: Optional[str] = None
    date_start: Optional[str] = None
    failure_exit_code: Optional[int] = None
    reported_improvement_epochs: tuple[int, ...] = ()

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self.protocol.get("classes", ()))

    @property
    def num_classes(self) -> int:
        return int(self.protocol.get("num_classes", len(self.classes)))

    @property
    def best_epoch_record(self) -> Optional[EpochRecord]:
        candidates = [epoch for epoch in self.epochs if epoch.val_macro_f1 is not None]
        return max(candidates, key=lambda epoch: epoch.val_macro_f1) if candidates else None

    @property
    def best_source_val_epoch(self) -> Optional[int]:
        best = self.best_epoch_record
        return best.epoch if best else None

    @property
    def best_source_val_f1(self) -> Optional[float]:
        best = self.best_epoch_record
        return best.val_macro_f1 if best else None


def _convert_value(value: str) -> Any:
    value = value.strip()
    if value in {"None", "null"}:
        return None
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _pipe_fields(line: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in line.strip().split("|")[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = _convert_value(value)
    return result


def _namespace_fields(line: str) -> dict[str, Any]:
    try:
        expression = ast.parse(line.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(expression, ast.Call):
        return {}
    result: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            continue
        try:
            result[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError):
            continue
    return result


def _find_class_metric(lines: list[str], class_name: str) -> Optional[ClassMetric]:
    pattern = re.compile(
        rf"^\s*{re.escape(class_name)}\s+"
        r"(?P<precision>\d+(?:\.\d+)?)\s+"
        r"(?P<recall>\d+(?:\.\d+)?)\s+"
        r"(?P<f1>\d+(?:\.\d+)?)\s+(?P<support>\d+)\s*$"
    )
    for line in lines:
        match = pattern.match(line)
        if match:
            return ClassMetric(
                class_name=class_name,
                precision=float(match.group("precision")),
                recall=float(match.group("recall")),
                f1=float(match.group("f1")),
                support=int(match.group("support")),
            )
    return None


def _parse_confusion(lines: list[str], classes: tuple[str, ...]) -> Optional[np.ndarray]:
    rows: list[list[int]] = []
    for class_name in classes:
        start = next(
            (index for index, line in enumerate(lines) if line.startswith(f"{class_name} [")),
            None,
        )
        if start is None:
            return None
        text = lines[start]
        index = start + 1
        while "]" not in text and index < len(lines):
            text += " " + lines[index]
            index += 1
        bracket = text[text.find("[") + 1:text.rfind("]")]
        row = [int(value) for value in re.findall(r"-?\d+", bracket)]
        if len(row) != len(classes):
            return None
        rows.append(row)
    return np.asarray(rows, dtype=np.int64)


def parse_task_log(path: Path | str) -> ParsedRun:
    """Parse one task log without modifying it."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((_pipe_fields(line) for line in lines if line.startswith("TASK_START|")), {})
    match = re.match(r"^(?P<source>[^_]+)_to_(?P<target>[^_]+)_seed(?P<seed>\d+)$", path.stem)
    run_name = str(start.get("run", path.stem))
    source = str(start.get("source", match.group("source") if match else ""))
    target = str(start.get("target", match.group("target") if match else ""))
    seed = int(start.get("seed", match.group("seed") if match else 0))
    config = next((_namespace_fields(line) for line in lines if line.startswith("Namespace(")), {})
    protocol = next((_pipe_fields(line) for line in lines if line.startswith("CLOSED_SET_PROTOCOL|")), {})
    if "classes" in protocol:
        protocol["classes"] = tuple(str(protocol["classes"]).split(","))
    counts = next((_pipe_fields(line) for line in lines if line.startswith("CLOSED_SET_COUNTS|")), {})

    epochs: list[EpochRecord] = []
    reported_improvement_epochs: list[int] = []
    pending: Optional[EpochRecord] = None
    validation_pattern = re.compile(
        r"^Validation result: loss=(?P<loss>[-+\d.eE]+), "
        r"acc=(?P<accuracy>[-+\d.eE]+), f1=(?P<f1>[-+\d.eE]+)"
    )
    for line in lines:
        if line.startswith("TRAIN_EPOCH|"):
            values = _pipe_fields(line)
            pending = EpochRecord(
                epoch=int(str(values["epoch"]).split("/")[0]),
                steps=int(values["steps"]),
                total=float(values["total"]), task=float(values["task"]),
                quality=float(values["quality"]),
                geometry=float(values["geometry"]),
                alignment=float(values["alignment"]),
                domain_accuracy=float(values["domain_accuracy"]),
                alpha_T=float(values["alpha_T"]),
                alpha_D=float(values["alpha_D"]),
                alpha_R=float(values["alpha_R"]),
                beta_T_temp=float(values["beta_T_temp"]),
                beta_D_temp=float(values["beta_D_temp"]),
                beta_T_channel=float(values["beta_T_channel"]),
                beta_D_channel=float(values["beta_D_channel"]),
                grl=float(values["grl"]),
                lr=float(values["lr"]),
            )
            epochs.append(pending)
        else:
            validation = validation_pattern.match(line)
            if validation and pending is not None and pending.val_macro_f1 is None:
                pending.val_loss = float(validation.group("loss"))
                pending.val_accuracy = float(validation.group("accuracy"))
                pending.val_macro_f1 = float(validation.group("f1"))
            elif line.startswith("Validation F1 improved") and pending is not None:
                reported_improvement_epochs.append(pending.epoch)

    test_pattern = re.compile(
        r"^Test result for .*: accuracy=(?P<accuracy>[-+\d.eE]+), "
        r"f1=(?P<f1>[-+\d.eE]+)"
    )
    target_accuracy = target_macro_f1 = None
    for line in lines:
        test = test_pattern.match(line)
        if test:
            target_accuracy = float(test.group("accuracy"))
            target_macro_f1 = float(test.group("f1"))

    classes = tuple(protocol.get("classes", ()))
    class_metrics = [metric for name in classes if (metric := _find_class_metric(lines, name))]
    confusion = _parse_confusion(lines, classes) if classes else None
    failed_line = next((line for line in lines if line.startswith("TASK_FAILED|")), None)
    if any(line.startswith("TASK_DONE|") for line in lines):
        status = "completed"
    elif failed_line is not None:
        status = "failed"
    else:
        status = "incomplete"
    failure_exit_code = None
    if failed_line:
        failure_exit_code = int(_pipe_fields(failed_line).get("exit_code", 0))

    return ParsedRun(
        path=path, run_name=run_name, source=source, target=target, seed=seed,
        status=status, config=config, protocol=protocol, counts=counts,
        epochs=epochs, class_metrics=class_metrics, confusion=confusion,
        target_accuracy=target_accuracy, target_macro_f1=target_macro_f1,
        git_head=next((line.split("|", 1)[1] for line in lines if line.startswith("GIT_HEAD|")), None),
        date_start=next((line.split("|", 1)[1] for line in lines if line.startswith("DATE_START|")), None),
        failure_exit_code=failure_exit_code,
        reported_improvement_epochs=tuple(reported_improvement_epochs),
    )
