from pathlib import Path

import numpy as np
import pysptk
import pytest
import torch
from torchcodec.encoders import AudioEncoder
from torchdtw import dtw_path

from unit_hifigan.mcd import MelCepstrum, collect_audios, distortion, frequency_warping, mel_cepstral_distortion
from unit_hifigan.model import SAMPLE_RATE


def test_frequency_warping_identity_without_warping() -> None:
    torch.testing.assert_close(frequency_warping(8, 4, alpha=0.0), torch.eye(4, 8))


def test_frequency_warping_matches_sptk() -> None:
    cepstrum = torch.randn(512, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    warped = frequency_warping(512, 24, alpha=0.42).double() @ cepstrum
    expected = pysptk.freqt(cepstrum.numpy(), order=23, alpha=0.42)
    torch.testing.assert_close(warped, torch.from_numpy(expected), atol=1e-6, rtol=1e-6)


def test_mel_cepstrum_matches_sptk() -> None:
    audio = torch.rand((1, SAMPLE_RATE), generator=torch.Generator().manual_seed(0)) - 0.5
    cepstrum = MelCepstrum()(audio).squeeze(0)
    window = np.hamming(512)
    frames = [audio[0, start : start + 512].numpy() * window for start in range(0, SAMPLE_RATE - 512 + 1, 256)]
    periodograms = [np.maximum(np.abs(np.fft.rfft(frame)) ** 2, 1e-10) for frame in frames]
    expected = np.stack([pysptk.sp2mc(periodogram, order=23, alpha=0.42) for periodogram in periodograms])
    assert cepstrum.shape == (len(frames), 23)
    torch.testing.assert_close(cepstrum.double(), torch.from_numpy(expected[:, 1:]), atol=1e-5, rtol=1e-5)


def test_mel_cepstrum_shape() -> None:
    cepstrum = MelCepstrum()(torch.randn(2, SAMPLE_RATE))
    assert cepstrum.ndim == 3
    assert cepstrum.size(0) == 2
    assert cepstrum.size(-1) == 23


def test_mel_cepstrum_invariant_to_scaling() -> None:
    audio = torch.rand((1, SAMPLE_RATE), generator=torch.Generator().manual_seed(0)) - 0.5
    torch.testing.assert_close(MelCepstrum()(audio), MelCepstrum()(0.5 * audio), atol=1e-4, rtol=1e-4)


def test_dtw_path_diagonal() -> None:
    cost = torch.ones(4, 4) - torch.eye(4)
    path = dtw_path(cost)
    assert path.tolist() == [[0, 0], [1, 1], [2, 2], [3, 3]]


def test_dtw_path_known_alignment() -> None:
    features = torch.randn(10, 4)
    repeated = features.repeat_interleave(2, dim=0)
    cost = torch.cdist(features, repeated)
    path = dtw_path(cost)
    assert float(cost[path[:, 0], path[:, 1]].sum()) == pytest.approx(0)
    assert path[0].tolist() == [0, 0]
    assert path[-1].tolist() == [9, 19]


def test_distortion_identical_is_zero() -> None:
    cepstrum = torch.randn(50, 13)
    mcd, frames = distortion(cepstrum, cepstrum)
    assert mcd == pytest.approx(0)
    assert frames == 50


def test_distortion_positive() -> None:
    mcd, _ = distortion(torch.randn(50, 13), torch.randn(60, 13))
    assert mcd > 0


def test_collect_audios_rejects_duplicates(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "utt1.wav").touch()
    (tmp_path / "nested/utt1.flac").touch()
    with pytest.raises(ValueError, match="Duplicate"):
        collect_audios(tmp_path)


def write_audio(path: Path, audio: torch.Tensor) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    AudioEncoder(audio, sample_rate=SAMPLE_RATE).to_file(path)


def test_mel_cepstral_distortion(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    for name in ("utt1", "utt2"):
        audio = torch.rand((1, SAMPLE_RATE), generator=generator) - 0.5
        write_audio(tmp_path / f"reference/{name}.wav", audio)
        write_audio(tmp_path / f"generated/nested/{name}.wav", audio)
    output = mel_cepstral_distortion(tmp_path / "reference", tmp_path / "generated")
    assert output.columns == ["audio", "mcd", "frames"]
    assert output["audio"].to_list() == ["utt1", "utt2"]
    assert all(mcd == pytest.approx(0, abs=1e-2) for mcd in output["mcd"])


def test_mel_cepstral_distortion_missing_reference(tmp_path: Path) -> None:
    audio = torch.zeros(1, SAMPLE_RATE)
    write_audio(tmp_path / "reference/utt1.wav", audio)
    write_audio(tmp_path / "generated/utt2.wav", audio)
    with pytest.raises(ValueError, match="utt2"):
        mel_cepstral_distortion(tmp_path / "reference", tmp_path / "generated")
