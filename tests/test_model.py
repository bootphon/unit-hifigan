from pathlib import Path

import pytest
import torch

from unit_hifigan.model import (
    UnitDiscriminator,
    UnitVocoder,
    one_dim_to_two_dim,
    tensor_from_name,
    upsample_embedding,
)

UPSAMPLE_FACTOR = 5 * 4 * 4 * 2 * 2  # Product of the generator upsampling strides


def test_one_dim_to_two_dim() -> None:
    x = torch.randn(2, 1, 12)
    assert one_dim_to_two_dim(x, torch.tensor(3)).shape == (2, 1, 4, 3)
    padded = one_dim_to_two_dim(torch.randn(2, 1, 10), torch.tensor(3))
    assert padded.shape == (2, 1, 4, 3)


def test_upsample_embedding() -> None:
    embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2D: (batch, channels)
    upsampled = upsample_embedding(embedding, 4)
    assert upsampled.shape == (2, 2, 4)
    torch.testing.assert_close(upsampled[0, 0], torch.ones(4))
    upsampled = upsample_embedding(torch.tensor([1.0, 2.0]), 3)  # 1D: (batch,)
    assert upsampled.shape == (2, 1, 3)
    embedding = torch.tensor([[[1.0, 2.0]]])  # 3D: (batch, channels, length)
    upsampled = upsample_embedding(embedding, 6)
    torch.testing.assert_close(upsampled, torch.tensor([[[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]]]))
    with pytest.raises(AssertionError):
        upsample_embedding(embedding, 5)  # n_frames not a multiple of the embedding length


def test_tensor_from_name() -> None:
    mapping = {"a": 0, "b": 1}
    torch.testing.assert_close(
        tensor_from_name(["b", "a"], mapping, torch.device("cpu")),
        torch.tensor([[1], [0]]),
    )


def test_vocoder_forward_shapes() -> None:
    vocoder = UnitVocoder(10)
    units = torch.randint(0, 10, (2, 7))
    assert vocoder(units).shape == (2, 1, 7 * UPSAMPLE_FACTOR)
    assert vocoder.n_units == 10
    assert vocoder.speakers == []


def test_vocoder_conditioning() -> None:
    vocoder = UnitVocoder(10, speakers=["alice", "bob"], styles=["read"])
    units = torch.randint(0, 10, (2, 7))
    speaker, style = torch.tensor([[0], [1]]), torch.tensor([[0], [0]])
    assert vocoder(units, speaker=speaker, style=style).shape == (2, 1, 7 * UPSAMPLE_FACTOR)
    with pytest.raises(AssertionError, match="speaker must be provided"):
        vocoder(units, style=style)
    with pytest.raises(AssertionError, match="style must be provided"):
        vocoder(units, speaker=speaker)


def test_vocoder_generate() -> None:
    vocoder = UnitVocoder(10, speakers=["alice", "bob"], styles=["read"])
    units = torch.randint(0, 10, (2, 7))
    audio = vocoder.generate(units, speaker=["bob", "alice"], style=["read", "read"])
    assert audio.shape == (2, 7 * UPSAMPLE_FACTOR)
    torch.testing.assert_close(audio.abs().amax(dim=-1), torch.ones(2))  # Peak-normalized per sample


def test_vocoder_setters() -> None:
    vocoder = UnitVocoder(10, speakers=["spkr0", "spkr1"])
    with pytest.raises(ValueError, match="Cannot change the number of speakers"):
        vocoder.speakers = ["alice"]
    vocoder.speakers = ["alice", "bob"]
    assert vocoder._speaker_to_index == {"alice": 0, "bob": 1}  # noqa: SLF001
    with pytest.raises(ValueError, match="Cannot change the number of styles"):
        vocoder.styles = ["read"]


def test_vocoder_save_load_pretrained(tmp_path: Path) -> None:
    vocoder = UnitVocoder(10, speakers=["alice", "bob"], styles=["read"]).eval()
    vocoder.save_pretrained(tmp_path / "vocoder")
    loaded = UnitVocoder.from_pretrained(tmp_path / "vocoder").eval()
    assert loaded.n_units == vocoder.n_units
    assert loaded.speakers == vocoder.speakers
    assert loaded.styles == vocoder.styles
    units = torch.randint(0, 10, (1, 5))
    speaker, style = torch.tensor([[1]]), torch.tensor([[0]])
    with torch.inference_mode():
        torch.testing.assert_close(
            loaded(units, speaker=speaker, style=style), vocoder(units, speaker=speaker, style=style)
        )


def test_discriminator_forward_shapes() -> None:
    discriminator = UnitDiscriminator().eval()
    length = 2_560
    with torch.inference_mode():
        output = discriminator(torch.rand(1, 1, length) * 2 - 1)
    assert output.log_mel_spectrogram.shape == (1, 80, length // 256)
    assert len(output.mpd) == 5
    assert len(output.msd) == 3
    assert len(output.mpd_features) == 5 * 6  # 5 discriminators, 5 convs + conv_post
    assert len(output.msd_features) == 3 * 8  # 3 discriminators, 7 convs + conv_post


def test_log_mel_spectrogram_matches_original_padding() -> None:
    """Reflect padding by (n_fft - hop_length) / 2, followed by an uncentered spectrogram."""
    discriminator = UnitDiscriminator()
    waveform = torch.rand(1, 2_560) * 2 - 1
    padded = torch.nn.functional.pad(waveform.unsqueeze(1), (384, 384), "reflect").squeeze(1)
    expected = torch.clamp(discriminator.mel_spectrogram(padded), min=1e-5).log()
    torch.testing.assert_close(discriminator.log_mel_spectrogram(waveform), expected)
