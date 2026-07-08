import math
from pathlib import Path

import polars as pl
import torch
from torch import Tensor, nn
from torchaudio.transforms import Spectrogram
from torchdtw import dtw_path
from tqdm import tqdm

from unit_hifigan.data import load_audio

AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".ogg", ".opus"})
DB_CONSTANT = 10 * math.sqrt(2) / math.log(10)  # Natural log cepstra to decibels


def frequency_warping(n_input: int, n_output: int, alpha: float) -> Tensor:
    """Matrix applying Oppenheim's frequency transform, mapping a cepstrum to an all-pass warped mel cepstrum."""
    matrix = torch.zeros(n_output, n_input, dtype=torch.float64)
    matrix[0] = alpha ** torch.arange(n_input, dtype=torch.float64)
    matrix[1, 1:] = (1 - alpha**2) * torch.arange(1, n_input, dtype=torch.float64) * matrix[0, :-1]
    for i in range(2, n_output):
        for j in range(1, n_input):
            matrix[i, j] = matrix[i - 1, j - 1] + alpha * (matrix[i, j - 1] - matrix[i - 1, j])
    return matrix.float()


class MelCepstrum(nn.Module):
    """SPTK-style mel cepstra for the classic mel cepstral distortion recipe, in pure PyTorch.

    Follows the settings of ESPnet's `evaluate_mcd.py` at 16kHz: periodograms from Hamming windows
    of 512 samples with hop 256, and mel cepstra of order 23 with all-pass warping factor 0.42.
    The mel cepstra are computed like `pysptk.sp2mc`: log periodogram, to real cepstrum by inverse
    FFT (with c(0) halved), to mel cepstrum by Oppenheim's frequency transform. This chain is
    linear, so it is applied as a single precomputed matrix on the log periodogram.

    Notes on the differences with the reference implementations:
    - `sp2mc` is the standard non-iterative cepstral estimate. ESPnet instead runs the iterative
      mel cepstral analysis of `pysptk.mcep` (Newton refinement of the same estimate), and other
      recipes start from a WORLD CheapTrick envelope: both are non-linear (the latter F0-adaptive)
      with no pure PyTorch equivalent. The two estimators differ by roughly 1 dB in cepstral
      distance on identical audio, so absolute MCDs are not directly comparable with numbers from
      those recipes; comparisons within this implementation are consistent.
    - The energy coefficient c(0) is dropped, making the metric invariant to the audio level
      (as in Kominek & Black; ESPnet keeps c(0)).
    - The warping factor 0.42 approximates the mel scale at 16kHz only; see
      `pysptk.util.mcepalpha` for other sample rates.
    """

    warping: Tensor

    def __init__(self, n_mcep: int = 23, alpha: float = 0.42) -> None:
        super().__init__()
        self.spectrogram = Spectrogram(
            n_fft=512,
            win_length=512,
            hop_length=256,
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},  # SPTK windows are symmetric
            power=2,
            normalized=False,
            center=False,
        )
        warping = frequency_warping(n_input=512, n_output=n_mcep + 1, alpha=alpha)
        warping[:, 0] /= 2  # Halving of c(0) as in `sp2mc`, folded into the matrix
        self.register_buffer("warping", warping)

    def forward(self, waveform: Tensor) -> Tensor:
        assert waveform.size(-1) >= self.spectrogram.n_fft, "Audio shorter than one analysis window"
        periodogram = self.spectrogram(waveform).transpose(-1, -2)  # (..., frames, n_fft // 2 + 1)
        cepstrum = torch.fft.irfft(torch.clamp(periodogram, min=1e-10).log())  # (..., frames, n_fft)
        return (cepstrum @ self.warping.T)[..., 1:]  # (..., frames, n_mcep), without the energy c(0)


def distortion(reference: Tensor, generated: Tensor) -> tuple[float, int]:
    """MCD in dB between two mel cepstra of shape (frames, n_mcep), and the length of the DTW alignment path.

    Same formula as ESPnet: 10 * sqrt(2) / ln(10) times the mean Euclidean distance over aligned
    frame pairs, but with an exact DTW instead of the approximate `fastdtw`.
    """
    cost = torch.cdist(reference, generated, compute_mode="donot_use_mm_for_euclid_dist")
    path = dtw_path(cost.cpu())
    return DB_CONSTANT * float(cost[path[:, 0], path[:, 1]].mean()), len(path)


def collect_audios(root: str | Path) -> dict[str, Path]:
    files = sorted(path for path in Path(root).rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES)
    audios = {path.stem: path for path in files}
    if len(audios) != len(files):
        raise ValueError(f"Duplicate audio file names in {root}")
    return audios


def mel_cepstral_distortion(
    root_reference: str | Path,
    root_generated: str | Path,
    *,
    n_mcep: int = 23,
    alpha: float = 0.42,
) -> pl.DataFrame:
    references, generations = collect_audios(root_reference), collect_audios(root_generated)
    if missing := sorted(set(generations) - set(references)):
        raise ValueError(f"Generated audio files without a reference counterpart: {missing}")
    mel_cepstrum = MelCepstrum(n_mcep, alpha).eval()
    outputs = []
    for name, generated in tqdm(sorted(generations.items())):
        reference = references[name]
        with torch.inference_mode():
            cepstrum_reference = mel_cepstrum(load_audio(reference)).squeeze(0)
            cepstrum_generated = mel_cepstrum(load_audio(generated)).squeeze(0)
            mcd, frames = distortion(cepstrum_reference, cepstrum_generated)
        outputs.append({"audio": name, "mcd": mcd, "frames": frames})
    return pl.DataFrame(outputs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mel cepstral distortion.")
    parser.add_argument("reference", type=Path, help="Directory with the reference audio files.")
    parser.add_argument("generated", type=Path, help="Directory with generated audio files of matching names.")
    parser.add_argument("output", type=Path, help="Path to the output file with per-file MCDs")
    parser.add_argument("--n-mcep", type=int, default=23, help="Order of the mel cepstra (excluding c0)")
    parser.add_argument("--alpha", type=float, default=0.42, help="All-pass frequency warping factor")
    args = parser.parse_args()
    output = mel_cepstral_distortion(args.reference, args.generated, n_mcep=args.n_mcep, alpha=args.alpha)
    output.write_ndjson(args.output)
    print(f"MCD={output['mcd'].mean():.3f} ± {output['mcd'].std():.3f}")
