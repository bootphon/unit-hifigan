import json
import os
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from unit_hifigan.data import build_dataloader
from unit_hifigan.loss import discriminator_loss, generator_loss
from unit_hifigan.model import UnitDiscriminator, UnitVocoder
from unit_hifigan.utils import (
    AverageMeters,
    MetricsLogger,
    init_distributed,
    latest_checkpoint,
    match_correct_dtype,
    set_pytorch_flags,
    set_seed,
    torch_profiler,
)


@dataclass(frozen=True)
class TrainConfig:
    workdir: str
    train_manifest: str
    val_manifest: str
    n_units: int
    max_steps: int = 500_000
    max_epochs: int = 2_000
    seed: int = 1234
    lr: float = 2e-4
    betas: tuple[float, float] = (0.8, 0.99)
    gamma: float = 0.999
    batch_size: int = 64
    segment_size: int = 8_960
    units_hop_size: int = 320  # Downsampling ratio of the encoder; SAMPLE_RATE / units frequency
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    log_interval: int = 100
    val_interval: int = 5_000

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Invalid dtype: {self.dtype}")
        assert Path(self.train_manifest).is_file(), f"Train manifest not found: {self.train_manifest}"
        assert Path(self.val_manifest).is_file(), f"Validation manifest not found: {self.val_manifest}"
        assert self.n_units > 0, "n_units must be positive"


@torch.no_grad()
def validate(
    vocoder: DistributedDataParallel,
    discriminator: DistributedDataParallel,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    *,
    with_autocast: bool,
) -> dict[str, float]:
    vocoder.eval()
    discriminator.eval()
    loss_mel_total, loss_d_total, loss_g_total = (torch.zeros((), device=device) for _ in range(3))
    for batch in loader:
        batch = batch.to(device)  # noqa: PLW2901
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=with_autocast):
            real = discriminator(batch.audio)
            generated = discriminator(vocoder(batch.units, speaker=batch.speaker, style=batch.style, f0=batch.f0))
            loss_g, losses_g = generator_loss(real, generated)
            loss_d = discriminator_loss(real, generated)[0]
        loss_mel_total += losses_g["mel"]
        loss_d_total += loss_d
        loss_g_total += loss_g
    vocoder.train()
    discriminator.train()
    return {
        "val/loss_mel": (loss_mel_total / len(loader)).item(),
        "val/loss_g": (loss_g_total / len(loader)).item(),
        "val/loss_d": (loss_d_total / len(loader)).item(),
    }


def train(cfg: TrainConfig) -> None:  # noqa: PLR0915
    with ExitStack() as stack:
        # Setup training and utilities
        workdir = Path(cfg.workdir)
        set_seed(cfg.seed)
        set_pytorch_flags()
        init_distributed()
        stack.callback(dist.destroy_process_group)
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        dtype = match_correct_dtype(cfg.dtype, device)
        global_rank = dist.get_rank()
        is_main, with_autocast = global_rank == 0, dtype != torch.float32
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "config.json").write_text(json.dumps(asdict(cfg), indent=4))
        logger = stack.enter_context(MetricsLogger(workdir / f"metrics-{global_rank}.jsonl"))
        profiler = stack.enter_context(torch_profiler(workdir / f"profiler-{global_rank}.html"))
        meters = AverageMeters()

        # Dataloaders, models, and optimizers
        loader = build_dataloader(cfg.train_manifest, cfg.batch_size, cfg.segment_size, cfg.units_hop_size, cfg.seed)
        val_loader = build_dataloader(
            cfg.val_manifest, cfg.batch_size, cfg.segment_size, cfg.units_hop_size, cfg.seed, is_train=False
        )
        val_loader.set_metadata(speakers=loader.speakers, styles=loader.styles, f0_bins=loader.f0_bins)
        vocoder = UnitVocoder(cfg.n_units, speakers=loader.speakers, styles=loader.styles, f0_bins=loader.f0_bins)
        vocoder = vocoder.train().to(device)
        discriminator = UnitDiscriminator().train().to(device)
        optim_g = AdamW(vocoder.parameters(), cfg.lr, cfg.betas, fused=True)
        optim_d = AdamW(discriminator.parameters(), cfg.lr, cfg.betas, fused=True)
        scheduler_g, scheduler_d = ExponentialLR(optim_g, cfg.gamma), ExponentialLR(optim_d, cfg.gamma)
        scaler = torch.GradScaler()

        # Checkpointing
        def save_checkpoint() -> None:
            if not is_main:
                return
            checkpoint = {
                "vocoder": vocoder.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                "loader": loader.state_dict(),
                "scaler": scaler.state_dict(),
                "step": torch.tensor(step),
                "epoch": torch.tensor(epoch),
                "meters": meters.state_dict(),
            }
            torch.save(checkpoint, workdir / f"checkpoint-{step}.pt")
            vocoder.save_pretrained(workdir / f"vocoder-{step}")

        stack.callback(save_checkpoint)  # Save on exit

        if (checkpoint_path := latest_checkpoint(workdir)) is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            vocoder.load_state_dict(checkpoint["vocoder"])
            discriminator.load_state_dict(checkpoint["discriminator"])
            optim_g.load_state_dict(checkpoint["optim_g"])
            optim_d.load_state_dict(checkpoint["optim_d"])
            scheduler_g.load_state_dict(checkpoint["scheduler_g"])
            scheduler_d.load_state_dict(checkpoint["scheduler_d"])
            loader.load_state_dict(checkpoint["loader"])
            scaler.load_state_dict(checkpoint["scaler"])
            meters.load_state_dict(checkpoint["meters"])
            step = checkpoint["step"].item()
            epoch = checkpoint["epoch"].item()
        else:
            step, epoch = 1, 1
        pbar = stack.enter_context(tqdm(initial=step, disable=not is_main, total=cfg.max_steps))

        # DDP and torch compile
        ddp_vocoder = DistributedDataParallel(torch.compile(vocoder, fullgraph=True, dynamic=False), [device.index])
        ddp_disc = DistributedDataParallel(torch.compile(discriminator, dynamic=False), [device.index])

        # Training loop
        while step < cfg.max_steps and epoch < cfg.max_epochs:
            loader.sampler.set_epoch(epoch)
            for batch in loader:
                if step >= cfg.max_steps:
                    break
                # Training step
                batch = batch.to(device)  # noqa: PLW2901
                optim_d.zero_grad()
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=with_autocast):
                    generated = ddp_vocoder(batch.units, speaker=batch.speaker, style=batch.style, f0=batch.f0)
                    loss_d, _ = discriminator_loss(ddp_disc(batch.audio), ddp_disc(generated.detach()))
                scaler.scale(loss_d).backward()
                scaler.step(optim_d)
                optim_g.zero_grad()
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=with_autocast):
                    loss_g, losses_g = generator_loss(ddp_disc(batch.audio), ddp_disc(generated))
                scaler.scale(loss_g).backward()
                scaler.step(optim_g)
                scaler.update()
                pbar.update()
                meters.update(loss_mel=losses_g["mel"], loss_g=loss_g, loss_d=loss_d)

                # Logging
                if step % cfg.log_interval == 0:
                    to_log = {"step": step, "epoch": epoch, "train/lr": scheduler_d.get_last_lr()[0]}
                    to_log |= {f"train/{key}": value for key, value in meters.pop().items()}
                    logger.log(to_log)
                    pbar.set_postfix(
                        loss_mel=to_log["train/loss_mel"],
                        loss_g=to_log["train/loss_g"],
                        loss_d=to_log["train/loss_d"],
                        epoch=to_log["epoch"],
                    )

                # Validation
                if step % cfg.val_interval == 0:
                    save_checkpoint()
                    dist.barrier()
                    val = validate(ddp_vocoder, ddp_disc, val_loader, device, dtype, with_autocast=with_autocast)
                    logger.log({"step": step, "epoch": epoch} | val)
                profiler.step()
                step += 1
            scheduler_d.step()
            scheduler_g.step()
            epoch += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a UnitHiFiGAN model")
    parser.add_argument("workdir", type=Path, help="Directory to save logs and checkpoints")
    parser.add_argument("--config", type=Path, help="Path to a JSON config file")
    parser.add_argument("--train", type=Path, help="Path to the training manifest file (if config not provided)")
    parser.add_argument("--val", type=Path, help="Path to the validation manifest file (if config not provided)")
    parser.add_argument("--units", type=int, help="Number of units in the speech tokenizer (if config not provided)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (if config not provided)")
    args = parser.parse_args()

    if (args.workdir / "config.json").is_file():
        cfg = TrainConfig(**json.loads((args.workdir / "config.json").read_text()))
        print("Resuming training with existing configuration")
    elif args.config is not None:
        cfg = TrainConfig(**json.loads(args.config.read_text()))
        print(f"Starting training with configuration from {args.config}")
    else:
        cfg = TrainConfig(
            str(args.workdir.expanduser().resolve()),
            str(args.train.expanduser().resolve()),
            str(args.val.expanduser().resolve()),
            int(args.units),
            seed=args.seed,
        )
    train(cfg)
