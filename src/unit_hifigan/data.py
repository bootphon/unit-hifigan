import os
from functools import cached_property, partial
from pathlib import Path
from types import NoneType
from typing import NamedTuple

import polars as pl
import torch
import torch.distributed as dist
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from torchcodec.decoders import AudioDecoder, WavDecoder
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from unit_hifigan.model import SAMPLE_RATE


def load_audio(source: str | Path) -> Tensor:
    decoder = WavDecoder if Path(source).suffix == ".wav" else AudioDecoder
    samples = decoder(source).get_all_samples()
    assert samples.sample_rate == SAMPLE_RATE
    data = samples.data
    assert data.size(0) == 1
    assert data.ndim == 2
    return 0.95 * (data / data.abs().max().clamp(min=1e-8))


def read_manifest(path: str | Path) -> pl.DataFrame:
    match Path(path).suffixes[0]:
        case ".jsonl":
            manifest = pl.read_ndjson(path)
        case ".csv":
            manifest = pl.read_csv(path)
        case _:
            raise ValueError(path)
    return manifest


class AudioItem(NamedTuple):
    units: Tensor
    audio: Tensor
    speaker: Tensor | None
    style: Tensor | None
    f0: Tensor | None

    def to(self, device: DeviceLikeType) -> "AudioItem":
        return AudioItem(
            self.units.to(device),
            self.audio.to(device),
            None if self.speaker is None else self.speaker.to(device),
            None if self.style is None else self.style.to(device),
            None if self.f0 is None else self.f0.to(device),
        )


def crop_segment(
    units: Tensor,
    audio: Tensor,
    units_hop_size: int,
    audio_segment_size: int,
    *,
    random_crop: bool = True,
) -> tuple[Tensor, Tensor]:
    assert audio_segment_size % units_hop_size == 0
    units_segment_size = audio_segment_size // units_hop_size
    units_length = min(audio.size(-1) // units_hop_size, units.size(-1))
    units = units[..., :units_length]
    audio = audio[..., : units_length * units_hop_size]
    while audio.size(-1) < audio_segment_size:
        audio, units = torch.hstack([audio, audio]), torch.hstack([units, units])
    n_positions = audio.size(-1) // units_hop_size - units_segment_size + 1
    start_units = torch.randint(0, n_positions, (1,)).item() if random_crop else 0
    start_audio = start_units * units_hop_size
    units = units[..., start_units : start_units + units_segment_size]
    audio = audio[..., start_audio : start_audio + audio_segment_size]
    return units, audio


class AudioDataset(Dataset[AudioItem]):
    def __init__(
        self,
        manifest: str | Path,
        segment_size: int,
        units_hop_size: int,
        *,
        random_crop: bool = True,
    ) -> None:
        assert segment_size % units_hop_size == 0, "segment_size must be multiple of SAMPLE_RATE / unit_frequency"
        self.segment_size, self.units_hop_size, self.random_crop = segment_size, units_hop_size, random_crop
        self.manifest = read_manifest(manifest)
        assert {"units", "audio"}.issubset(set(self.manifest.columns))
        if "speaker" in self.manifest.columns:
            self.speakers = self.manifest["speaker"].value_counts(sort=True)["speaker"].to_list()
        else:
            self.speakers = None
        if "style" in self.manifest.columns:
            self.styles = self.manifest["style"].value_counts(sort=True)["style"].to_list()
        else:
            self.styles = None
        if any("f0" in col for col in self.manifest.columns):
            raise NotImplementedError("f0 conditioning is not supported yet")
        self.f0_bins = None

    @cached_property
    def speaker_to_index(self) -> dict[str, int] | None:
        return {s: i for i, s in enumerate(self.speakers)} if self.speakers else None

    @cached_property
    def style_to_index(self) -> dict[str, int] | None:
        return {s: i for i, s in enumerate(self.styles)} if self.styles else None

    @cached_property
    def f0_bins_to_index(self) -> dict[int, int] | None:
        return None

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> AudioItem:
        entry = self.manifest[index].to_dicts()[0]
        units, audio = torch.tensor(entry["units"], dtype=torch.long), load_audio(entry["audio"])
        units, audio = crop_segment(units, audio, self.units_hop_size, self.segment_size, random_crop=self.random_crop)
        speaker = (
            torch.tensor([self.speaker_to_index[entry["speaker"]]], dtype=torch.long)
            if self.speaker_to_index
            else None
        )
        style = torch.tensor([self.style_to_index[entry["style"]]], dtype=torch.long) if self.style_to_index else None
        return AudioItem(units, audio, speaker, style, f0=None)


class AudioDataLoader(StatefulDataLoader[AudioItem]):
    dataset: AudioDataset

    def set_metadata(self, *, speakers: list[str] | None, styles: list[str] | None, f0_bins: list[int] | None) -> None:
        for name, cached, values in (
            ("speakers", "speaker_to_index", speakers),
            ("styles", "style_to_index", styles),
            ("f0_bins", "f0_bins_to_index", f0_bins),
        ):
            current = getattr(self.dataset, name)
            if current is not None and not set(current) <= set(values or []):
                raise ValueError(f"The {name} of this manifest are not a subset of the provided {name}")
            setattr(self.dataset, name, values)
            self.dataset.__dict__.pop(cached, None)  # Invalidate the cached_property

    @property
    def speakers(self) -> list[str] | None:
        return self.dataset.speakers

    @property
    def styles(self) -> list[str] | None:
        return self.dataset.styles

    @property
    def f0_bins(self) -> list[int] | None:
        return self.dataset.f0_bins


def default_num_workers() -> int:
    n_local_ranks = max(torch.cuda.device_count(), 1) if dist.is_initialized() else 1
    return max((os.process_cpu_count() or 1) // n_local_ranks, 1)


def collate_none_fn(batch, *, collate_fn_map):
    return None


def build_dataloader(
    manifest: str | Path,
    batch_size: int,
    segment_size: int,
    units_hop_size: int,
    seed: int,
    *,
    is_train: bool = True,
) -> AudioDataLoader:
    dataset = AudioDataset(manifest, segment_size, units_hop_size, random_crop=is_train)
    num_workers = default_num_workers()
    return AudioDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=None if dist.is_initialized() else is_train,
        sampler=StatefulDistributedSampler(dataset, shuffle=is_train, seed=seed, drop_last=is_train)
        if dist.is_initialized()
        else None,
        num_workers=num_workers,
        collate_fn=partial(collate, collate_fn_map=default_collate_fn_map | {NoneType: collate_none_fn}),
        drop_last=is_train,
        generator=torch.Generator().manual_seed(seed + (dist.get_rank() if dist.is_initialized() else 0)),
        persistent_workers=is_train and num_workers > 0,
    )
