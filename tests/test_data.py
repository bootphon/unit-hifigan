import json
from pathlib import Path

import pytest
import torch
from torchcodec.encoders import AudioEncoder

from unit_hifigan.data import AudioDataset, build_dataloader, crop_segment, load_audio, read_manifest
from unit_hifigan.model import SAMPLE_RATE

UNITS_HOP_SIZE = 320
SEGMENT_SIZE = 1_600  # 5 units

pytestmark = pytest.mark.filterwarnings("ignore:'set_vital' is deprecated, please do not call:UserWarning")


def write_audio(path: Path, n_samples: int, sample_rate: int = SAMPLE_RATE, *, silent: bool = False) -> None:
    samples = torch.zeros(n_samples) if silent else torch.sin(torch.linspace(0, 100, n_samples))
    AudioEncoder(samples.unsqueeze(0), sample_rate=sample_rate).to_file(path)


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    entries = []
    for index, (speaker, style) in enumerate([("alice", "read"), ("bob", "read"), ("bob", "laugh")]):
        audio = tmp_path / f"audio-{index}.wav"
        write_audio(audio, SAMPLE_RATE)  # 1 second, 50 units
        units = torch.randint(0, 100, (SAMPLE_RATE // UNITS_HOP_SIZE,)).tolist()
        entries.append({"audio": str(audio), "units": units, "speaker": speaker, "style": style})
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(entry) for entry in entries))
    return path


def test_load_audio(tmp_path: Path) -> None:
    write_audio(tmp_path / "audio.wav", SAMPLE_RATE)
    audio = load_audio(tmp_path / "audio.wav")
    assert audio.shape == (1, SAMPLE_RATE)
    assert audio.abs().max() == pytest.approx(0.95, abs=1e-3)


def test_load_audio_silence(tmp_path: Path) -> None:
    write_audio(tmp_path / "silence.wav", SAMPLE_RATE, silent=True)
    audio = load_audio(tmp_path / "silence.wav")
    assert not audio.isnan().any()
    torch.testing.assert_close(audio, torch.zeros(1, SAMPLE_RATE))


def test_load_audio_wrong_sample_rate(tmp_path: Path) -> None:
    write_audio(tmp_path / "audio.wav", SAMPLE_RATE, sample_rate=22_050)
    with pytest.raises(AssertionError):
        load_audio(tmp_path / "audio.wav")


def test_read_manifest(manifest: Path) -> None:
    frame = read_manifest(manifest)
    assert len(frame) == 3
    assert {"audio", "units", "speaker", "style"} <= set(frame.columns)
    with pytest.raises(ValueError, match="manifest"):
        read_manifest(manifest.with_suffix(".parquet"))


@pytest.mark.parametrize("n_units", [3, 5, 50])
def test_crop_segment(n_units: int) -> None:
    units = torch.arange(n_units)
    audio = torch.arange(n_units * UNITS_HOP_SIZE, dtype=torch.float32).unsqueeze(0)
    cropped_units, cropped_audio = crop_segment(units, audio, UNITS_HOP_SIZE, SEGMENT_SIZE)
    assert cropped_units.shape == (SEGMENT_SIZE // UNITS_HOP_SIZE,)
    assert cropped_audio.shape == (1, SEGMENT_SIZE)
    if n_units >= SEGMENT_SIZE // UNITS_HOP_SIZE:  # Audio must stay aligned with the units
        torch.testing.assert_close(cropped_audio[0, 0], cropped_units[0].float() * UNITS_HOP_SIZE)


def test_crop_segment_deterministic() -> None:
    units, audio = torch.arange(50), torch.arange(50 * UNITS_HOP_SIZE, dtype=torch.float32).unsqueeze(0)
    cropped_units, cropped_audio = crop_segment(units, audio, UNITS_HOP_SIZE, SEGMENT_SIZE, random_crop=False)
    torch.testing.assert_close(cropped_units, units[: SEGMENT_SIZE // UNITS_HOP_SIZE])
    torch.testing.assert_close(cropped_audio, audio[:, :SEGMENT_SIZE])


def test_audio_dataset(manifest: Path) -> None:
    dataset = AudioDataset(manifest, SEGMENT_SIZE, UNITS_HOP_SIZE)
    assert len(dataset) == 3
    assert dataset.speakers == ["bob", "alice"]  # Sorted by frequency
    assert dataset.styles == ["read", "laugh"]
    item = dataset[0]
    assert item.units.shape == (SEGMENT_SIZE // UNITS_HOP_SIZE,)
    assert item.audio.shape == (1, SEGMENT_SIZE)
    assert item.speaker is not None
    assert dataset.speaker_to_index is not None
    assert item.speaker.item() == dataset.speaker_to_index["alice"]
    assert item.f0 is None


def test_audio_dataset_deterministic(manifest: Path) -> None:
    dataset = AudioDataset(manifest, SEGMENT_SIZE, UNITS_HOP_SIZE, random_crop=False)
    first, second = dataset[1], dataset[1]
    torch.testing.assert_close(first.units, second.units)
    torch.testing.assert_close(first.audio, second.audio)


def test_build_dataloader(manifest: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("unit_hifigan.data.default_num_workers", lambda: 0)
    loader = build_dataloader(manifest, batch_size=2, segment_size=SEGMENT_SIZE, units_hop_size=UNITS_HOP_SIZE, seed=0)
    batch = next(iter(loader))
    assert batch.units.shape == (2, SEGMENT_SIZE // UNITS_HOP_SIZE)
    assert batch.audio.shape == (2, 1, SEGMENT_SIZE)
    assert batch.speaker is not None
    assert batch.speaker.shape == (2, 1)
    assert batch.f0 is None  # The NoneType collate function must keep missing fields as None


def test_set_metadata(manifest: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("unit_hifigan.data.default_num_workers", lambda: 0)
    loader = build_dataloader(manifest, batch_size=2, segment_size=SEGMENT_SIZE, units_hop_size=UNITS_HOP_SIZE, seed=0)
    assert loader.dataset.speaker_to_index == {"bob": 0, "alice": 1}  # Populate the cache before set_metadata
    loader.set_metadata(speakers=["carol", "alice", "bob"], styles=["read", "laugh"], f0_bins=None)
    assert loader.speakers == ["carol", "alice", "bob"]
    assert loader.dataset.speaker_to_index == {"carol": 0, "alice": 1, "bob": 2}  # The cache must be invalidated
    with pytest.raises(ValueError, match="not a subset"):
        loader.set_metadata(speakers=["alice"], styles=["read", "laugh"], f0_bins=None)
    with pytest.raises(ValueError, match="not a subset"):
        loader.set_metadata(speakers=["carol", "alice", "bob"], styles=None, f0_bins=None)
