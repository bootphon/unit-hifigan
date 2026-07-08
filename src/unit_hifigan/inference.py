import os
from os.path import commonpath
from pathlib import Path

import polars as pl
import torch
from torchcodec.encoders import AudioEncoder
from tqdm import tqdm

from unit_hifigan.data import read_manifest
from unit_hifigan.model import SAMPLE_RATE, UnitVocoder


def inference(root: str | Path, path_model: str | Path, path_manifest: str | Path, device_type: str = "cuda") -> None:
    root = Path(root)
    device = torch.device(device_type)
    model = UnitVocoder.from_pretrained(path_model).eval().to(device)
    model.compile(fullgraph=True, dynamic=True)
    manifest = read_manifest(path_manifest)
    common = commonpath([str(Path(audio).parent) for audio in manifest["audio"]])
    manifest = manifest.with_columns(pl.col("audio").str.strip_prefix(f"{common}{os.sep}"))
    with torch.inference_mode():
        for entry in tqdm(manifest.iter_rows(named=True), total=len(manifest)):
            dest = root / entry["audio"]
            dest.parent.mkdir(exist_ok=True, parents=True)
            units = torch.tensor(entry["units"], dtype=torch.long, device=device).unsqueeze(0)
            speaker = [entry["speaker"]] if "speaker" in entry else None
            style = [entry["style"]] if "style" in entry else None
            audio = model.generate(units, speaker=speaker, style=style).squeeze().cpu()
            AudioEncoder(audio, sample_rate=SAMPLE_RATE).to_file(dest)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Root directory to generated files")
    parser.add_argument("model", type=Path, help="Path to trained model")
    parser.add_argument("manifest", type=Path, help="Manifest file (with columns 'audio' and 'units')")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    args = parser.parse_args()
    inference(args.root, args.model, args.manifest, args.device)
