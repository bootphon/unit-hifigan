import importlib.resources
import json
import os
import random
import string
import subprocess
import tempfile
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import torch
from torch import distributed as dist
from torch.profiler import ProfilerAction, profile

if TYPE_CHECKING:
    from io import TextIOWrapper


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)


def set_pytorch_flags() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.autograd.set_detect_anomaly(mode=False, check_nan=True)


def init_distributed() -> None:
    if os.getenv("LOCAL_RANK") is not None:  # torchrun
        global_rank, local_rank, world_size = os.environ["RANK"], os.environ["LOCAL_RANK"], os.environ["WORLD_SIZE"]
        master_port, master_addr = os.environ["MASTER_PORT"], os.environ["MASTER_ADDR"]
    elif os.getenv("SLURM_JOB_ID") is not None:  # Slurm
        scontrol_cmd = ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        master_addr = subprocess.check_output(scontrol_cmd, text=True).split()[0]
        master_port = str(random.Random(int(os.environ["SLURM_JOB_ID"])).randint(20_000, 60_000))
        global_rank, local_rank = os.environ["SLURM_PROCID"], os.environ["SLURM_LOCALID"]
        world_size = os.environ["SLURM_NTASKS"]
    else:  # One local GPU
        global_rank, local_rank, world_size = "0", "0", "1"
        master_port, master_addr = str(random.Random(-1).randint(20_000, 60_000)), "127.0.0.1"
    os.environ["RANK"], os.environ["LOCAL_RANK"], os.environ["WORLD_SIZE"] = global_rank, local_rank, world_size
    os.environ["MASTER_PORT"], os.environ["MASTER_ADDR"] = master_port, master_addr
    dist.init_process_group(backend="nccl")


def match_correct_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]
    assert dtype != torch.bfloat16 or torch.cuda.get_device_properties(device).major >= 8, "bfloat16 not supported"
    return dtype


class AverageMeter:
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        self.sum = torch.zeros((), device=self.device, dtype=self.dtype)
        self.count = torch.zeros((), device=self.device, dtype=torch.long)
        self.avg = torch.zeros((), device=self.device, dtype=self.dtype)

    def update(self, val: torch.Tensor, n: int = 1) -> None:
        self.sum += val.detach() * n
        self.count += n
        self.avg = self.sum / self.count

    def pop(self) -> float:
        val = self.avg
        self.reset()
        return val.item()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"sum": self.sum, "count": self.count, "avg": self.avg}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.sum = state_dict["sum"].to(device=self.device, dtype=self.dtype)
        self.count = state_dict["count"].to(device=self.device, dtype=torch.long)
        self.avg = state_dict["avg"].to(device=self.device, dtype=self.dtype)


class AverageMeters:
    def __init__(self) -> None:
        self.meters = {}

    def __getitem__(self, name: str) -> AverageMeter:
        return self.meters[name]

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def update(self, **kwargs: torch.Tensor) -> None:
        for name, value in kwargs.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter(device=value.device, dtype=value.dtype)
            self.meters[name].update(value)

    def pop(self) -> dict[str, float]:
        return {name: meter.pop() for name, meter in self.meters.items()}

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {name: meter.state_dict() for name, meter in self.meters.items()}

    def load_state_dict(self, state_dict: dict[str, dict[str, torch.Tensor]]) -> None:
        self.meters = {}
        for name, meter_state in state_dict.items():
            meter = AverageMeter(device=meter_state["sum"].device, dtype=meter_state["sum"].dtype)
            meter.load_state_dict(meter_state)
            self.meters[name] = meter


class MetricsLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.jsonl_writer: TextIOWrapper | None = None

    def log(self, metrics: dict[str, Any]) -> None:
        print(json.dumps(metrics), file=self.jsonl_writer, flush=True)

    def open(self) -> None:
        if self.jsonl_writer is None:
            self.jsonl_writer = self.path.open("a")

    def close(self, _: int = 0) -> None:
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(0 if exc_type is None else 1)

    def __del__(self) -> None:
        self.close()


def html_trace_handler(p: profile, path: str | Path) -> None:
    """Adapted from https://github.com/facebookresearch/lingua/blob/main/lingua/profiling.py."""
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".json") as fp:
        p.export_chrome_trace(fp.name)
        content = Path(fp.name).read_text(encoding="utf-8")
    viztracer = importlib.resources.files("viztracer")
    template = (viztracer / "html/trace_viewer_embedder.html").read_text(encoding="utf-8")
    sub = {"trace_viewer_full": (viztracer / "html/trace_viewer_full.html").read_text(encoding="utf-8")}
    sub["json_data"] = content.replace("</script>", "<\\/script>")
    Path(path).write_text(string.Template(template).substitute(sub), encoding="utf-8")


def scheduler_fn(step: int, skip_first: int, warmup: int, active: int) -> ProfilerAction:
    if step < skip_first or step >= skip_first + warmup + active:
        return ProfilerAction.NONE
    if step < skip_first + warmup:
        return ProfilerAction.WARMUP
    if step < skip_first + warmup + active - 1:
        return ProfilerAction.RECORD
    return ProfilerAction.RECORD_AND_SAVE


def torch_profiler(path: str | Path, skip_first: int = 100, warmup: int = 5, active: int = 2) -> profile:
    return profile(
        schedule=partial(scheduler_fn, skip_first=skip_first, warmup=warmup, active=active),
        on_trace_ready=partial(html_trace_handler, path=path),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    )


def latest_checkpoint(folder: Path) -> Path | None:
    paths = sorted(folder.glob("checkpoint-*.pt"), key=lambda x: int(x.stem.removeprefix("checkpoint-")))
    return paths[-1] if paths else None
