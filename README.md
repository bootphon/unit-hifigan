# Unit HiFi-GAN

Minimal re-implementation of HiFi-GAN training on discrete speech units.

The package is available on PyPI:

```bash
pip install unit-hifigan
```

It is based on the ["Speech Resynthesis from Discrete Disentangled Self-Supervised Representations"](https://github.com/facebookresearch/speech-resynthesis)
repository, with a clean and minimal re-implementation for both training and inference.
We follow their default hyperparameters and model configurations.

## Usage

### Vocoder

The vocoder is available via the `UnitVocoder` class. You can condition it on `speakers` and `styles` (support for f0 will come later).

Load a pretrained model from a local directory or a distant HuggingFace repository like this:

```python
from unit_hifigan import UnitVocoder

vocoder = UnitVocoder.from_pretrained("coml/hubert-phoneme-classification", revision="vocoder-base-l11")
```

You can also load models from the legacy implementation or from [textlesslib](https://github.com/facebookresearch/textlesslib) with `UnitVocoder.from_legacy_pretrained`:

```python
import requests
from unit_hifigan import UnitVocoder

url = "https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500/"
vocoder = UnitVocoder.from_legacy_pretrained(url + "generator.pt")
vocoder.speakers = requests.get(url + "speakers.txt").text.splitlines()
vocoder.styles = requests.get(url + "styles.txt").text.splitlines()
```

You can then generate audio at 16kHz with `UnitVocoder.generate` as below:

```python
import torch
from unit_hifigan import UnitVocoder

units, speaker, style = torch.randint(0, 500, (1, 100)), ["speaker-0"], ["reading"]
vocoder = UnitVocoder(500, speakers=["speaker-0", "speaker-1"], styles=["reading", "crying", "laughing"])
audio = vocoder.generate(units, speaker=speaker, style=style)
```

### Training a Unit HiFi-GAN

#### Data preparation

For training, you need to have manifest files for the training and validation datasets.
The manifests are JSONL files with fields `audio` (string), `units` (list of integers), and optionally `style` (string) or `speaker` (string):

```jsonl
{"audio":"audio-1.wav","units":[24,24,173,289,289,441,487,370,370],"speaker":"spkr02"}
```

- `audio`: full path to the audio file
- `units`: discrete units from the speech encoder (for example, HuBERT layer 11 and K-means K=500)
- `speaker`: name of the speaker (for speaker conditioning)
- `style`: name of the style (for style conditioning)

#### Training

Via the CLI:

```bash
# Minimal command with default configuration
python -m unit_hifigan.train --train $TRAIN_MANIFEST --val $VAL_MANIFEST --units $N_UNITS

# If you have a JSON config file
python -m unit_hifigan.train --config $CONFIG
```
You can also use the `unit_hifigan.train.train` function in your Python code if you prefer. Check out `unit_hifigan.train.TrainConfig` for the list of configuration options.

The pipeline supports DDP by default, when run with either torchrun or Slurm. Have a look at `unit_hifigan.utils.init_distributed` for how distributed training is initialized.


### Synthesis

Via the CLI:

```bash
python -m unit_hifigan.inference $GENERATIONS $PRETRAINED_MODEL $INFERENCE_MANIFEST
```

where the inference manifest has the same format as the training and validation ones.

### Evaluation

#### Whisper ASR

```bash
python -m unit_hifigan.wer.whisper $GENERATIONS $ASR_MANIFEST $JSONL_OUTPUT --model $MODEL
```

where $MODEL is the name of the Whisper variant (`large-v3` if not provided).
The ASR manifest has the followings fields:

- `audio`: **relative** path to the audio files from the $GENERATIONS directory.
- `transcript`: ground truth transcription

#### Wav2vec 2.0 ASR

```bash
python -m unit_hifigan.wer.torchaudio $GENERATIONS $ASR_MANIFEST $JSONL_OUTPUT --model $MODEL
```

where $MODEL is the name of the torchaudio ASR pipeline to use (by default `WAV2VEC2_ASR_LARGE_LV60K_960H`).


## TODO

- Add MCD evaluation
- Add support for F0

## Acknowledgements

- [Speech Resynthesis from Discrete Disentangled Self-Supervised Representations](https://www.isca-archive.org/interspeech_2021/polyak21_interspeech.html): https://github.com/facebookresearch/speech-resynthesis
- [HiFi-GAN](https://proceedings.neurips.cc/paper/2020/hash/c5d736809766d46260d816d8dbc9eb44-Abstract.html): https://github.com/jik876/hifi-gan
- [EXPRESSO](https://www.isca-archive.org/interspeech_2023/nguyen23_interspeech.html): https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso
