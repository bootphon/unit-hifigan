import argparse
from pathlib import Path

import polars as pl
import torch
import whisper
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import edit_distance
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm
from whisper.audio import SAMPLE_RATE
from whisper.normalizers import EnglishTextNormalizer

from unit_hifigan.data import read_manifest


class ASRDataset(Dataset):
    def __init__(self, root: str | Path, path_manifest: str | Path, n_mels: int) -> None:
        self.n_mels = n_mels
        self.root = Path(root)
        self.manifest = read_manifest(path_manifest)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, str, str]:
        if item < 0 or item >= len(self.manifest):
            raise IndexError("Index out of range")
        entry = self.manifest[item]
        samples = AudioDecoder(self.root / entry["audio"][0]).get_all_samples()
        assert samples.sample_rate == SAMPLE_RATE
        audio = whisper.pad_or_trim(samples.data.flatten())
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)
        return mel, entry["transcript"][0], entry["audio"][0]


def whisper_word_error_rate(
    root: str | Path,
    path_manifest: str | Path,
    *,
    model_name: str = "large-v3",
    batch_size: int = 16,
    device: str = "cuda",
) -> pl.DataFrame:
    model = whisper.load_model(model_name, device=device).eval()
    dataset = ASRDataset(root, path_manifest, model.dims.n_mels)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)

    audios, hypotheses, references = [], [], []
    for mels, texts, paths in tqdm(loader):
        hypotheses += [result.text for result in model.decode(mels.to(device), options)]
        references += texts
        audios += paths
    normalizer = EnglishTextNormalizer()
    hypotheses_clean = [normalizer(text) for text in hypotheses]
    references_clean = [normalizer(text) for text in references]

    outputs = []
    for hypothesis, transcript, audio in zip(hypotheses_clean, references_clean, audios, strict=True):
        word_ed, word_length = edit_distance(hypothesis.split(), transcript.split()), len(transcript.split())
        char_ed, char_length = edit_distance(hypothesis, transcript), len(transcript)
        output = {
            "audio": audio,
            "transcript": transcript,
            "hypothesis": hypothesis,
            "wer": word_ed / word_length,
            "cer": char_ed / char_length,
            "word_ed": word_ed,
            "word_length": word_length,
            "char_ed": char_ed,
            "char_length": char_length,
        }
        outputs.append(output)
    return pl.DataFrame(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper ASR.")
    parser.add_argument("root", type=Path, help="Root directory for the audio files.")
    parser.add_argument("manifest", type=Path, help="Path to the manifest file, with columns 'audio', 'transcript'.")
    parser.add_argument("output", type=Path, help="Path to the output file with predictions and WERs")
    parser.add_argument(
        "--model",
        type=str,
        choices=whisper.available_models(),
        default="large-v3",
        help="Name of the Whisper model to use. Default: large-v3",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    output = whisper_word_error_rate(args.root, args.manifest, model_name=args.model, batch_size=args.batch_size)
    output.write_ndjson(args.output)
    wer = output["word_ed"].sum() / output["word_length"].sum()
    cer = output["char_ed"].sum() / output["char_length"].sum()
    print(f"WER={wer:.2%}, CER={cer:.2%}")
