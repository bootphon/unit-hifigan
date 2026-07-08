import json
from pathlib import Path

import pytest
import torch
from torch.profiler import ProfilerAction

from unit_hifigan.utils import AverageMeter, AverageMeters, MetricsLogger, latest_checkpoint, scheduler_fn


def test_average_meter() -> None:
    meter = AverageMeter(torch.device("cpu"))
    meter.update(torch.tensor(1.0))
    meter.update(torch.tensor(3.0))
    assert meter.pop() == pytest.approx(2.0)
    assert meter.avg.item() == pytest.approx(0.0)  # pop resets the meter
    meter.update(torch.tensor(5.0), n=3)
    assert meter.pop() == pytest.approx(5.0)


def test_average_meter_state_dict() -> None:
    meter = AverageMeter(torch.device("cpu"))
    meter.update(torch.tensor(1.0))
    meter.update(torch.tensor(2.0))
    restored = AverageMeter(torch.device("cpu"))
    restored.load_state_dict(meter.state_dict())
    assert restored.pop() == pytest.approx(1.5)


def test_average_meters() -> None:
    meters = AverageMeters()
    meters.update(loss=torch.tensor(1.0), other=torch.tensor(10.0))
    meters.update(loss=torch.tensor(3.0))
    assert meters.pop() == {"loss": pytest.approx(2.0), "other": pytest.approx(10.0)}
    meters.update(loss=torch.tensor(4.0))
    restored = AverageMeters()
    restored.load_state_dict(meters.state_dict())
    assert restored.pop() == {"loss": pytest.approx(4.0), "other": pytest.approx(0.0)}


def test_metrics_logger(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    with MetricsLogger(path) as logger:
        logger.log({"step": 1, "loss": 0.5})
        logger.log({"step": 2, "loss": 0.25})
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.25}]


def test_metrics_logger_without_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(path)
    logger.log({"step": 1})  # Must open the file instead of printing to stdout
    logger.close()
    assert json.loads(path.read_text()) == {"step": 1}


def test_latest_checkpoint(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path) is None
    for step in (2, 10, 9):  # Numeric ordering, not lexicographic
        (tmp_path / f"checkpoint-{step}.pt").touch()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-10.pt"


def test_scheduler_fn() -> None:
    actions = [scheduler_fn(step, skip_first=2, warmup=1, active=2) for step in range(6)]
    assert actions == [
        ProfilerAction.NONE,
        ProfilerAction.NONE,
        ProfilerAction.WARMUP,
        ProfilerAction.RECORD,
        ProfilerAction.RECORD_AND_SAVE,
        ProfilerAction.NONE,
    ]
