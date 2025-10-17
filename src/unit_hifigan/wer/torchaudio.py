import re
from pathlib import Path

import polars as pl
import torch
from torch.nn import functional as F
from torchaudio import pipelines
from torchaudio.functional import edit_distance
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from unit_hifigan.data import read_manifest


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z' ]", " ", text.lower()).strip()


def greedy_decode(emission: torch.Tensor, id2token: dict, blank_idx: int = 0) -> str:
    hypothesis = emission.argmax(-1).unique_consecutive()
    hypothesis = hypothesis[hypothesis != blank_idx]
    return "".join(id2token[int(i)].lower().replace("|", " ") for i in hypothesis).strip()


def torchaudio_word_error_rate(
    root: str | Path,
    path_manifest: str | Path,
    *,
    pipeline: pipelines.Wav2Vec2ASRBundle = pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H,
    device: str = "cuda",
) -> pl.DataFrame:
    manifest = read_manifest(path_manifest)
    wav2vec2, id2token = pipeline.get_model().eval().to(device), dict(enumerate(pipeline.get_labels()))
    outputs = []
    for source, raw_transcript in tqdm(manifest[["audio", "transcript"]].iter_rows(), total=len(manifest)):
        transcript = normalize_text(raw_transcript)
        samples = AudioDecoder(Path(root) / source).get_all_samples()
        assert samples.sample_rate == pipeline.sample_rate
        with torch.inference_mode():
            waveform = F.layer_norm(samples.data.to(device), samples.data.shape)
            emission, _ = wav2vec2(waveform)
            hypothesis = greedy_decode(emission.squeeze(), id2token)
        word_ed, word_length = edit_distance(hypothesis.split(), transcript.split()), len(transcript.split())
        char_ed, char_length = edit_distance(hypothesis, transcript), len(transcript)
        output = {
            "audio": source,
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
    import argparse

    parser = argparse.ArgumentParser(description="Torchaudio ASR.")
    parser.add_argument("root", type=Path, help="Root directory for the audio files.")
    parser.add_argument("manifest", type=Path, help="Path to the manifest file, with columns 'audio', 'transcript'.")
    parser.add_argument("output", type=Path, help="Path to the output file with predictions and WERs")
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the pretrained ASR pipeline in Torchaudio. Default is: WAV2VEC2_ASR_LARGE_LV60K_960H.",
        default="WAV2VEC2_ASR_LARGE_LV60K_960H",
    )
    args = parser.parse_args()

    output = torchaudio_word_error_rate(args.root, args.manifest, pipeline=getattr(pipelines, args.pipeline))
    output.write_ndjson(args.output)
    wer = output["word_ed"].sum() / output["word_length"].sum()
    cer = output["char_ed"].sum() / output["char_length"].sum()
    print(f"WER={wer:.2%}, CER={cer:.2%}")
