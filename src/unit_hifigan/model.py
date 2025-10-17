from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.init import normal_
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from torchaudio.transforms import MelSpectrogram

from unit_hifigan.compatibility import convert_discriminators_state_dict, convert_vocoder_state_dict, legacy_metadata

SAMPLE_RATE = 16_000
LRELU_SLOPE = 0.1


def one_dim_to_two_dim(x: Tensor, period: Tensor) -> Tensor:
    batch, channels, length = x.size()
    if length % period != 0:  # period must be a 0-dim tensor
        pad_length = period - (length % period)
        x = F.pad(x, (0, pad_length), "reflect")  # ty: ignore[invalid-argument-type]
        length += pad_length
    return x.view(batch, channels, length // period, period)  # ty: ignore[invalid-argument-type]


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = torch.tensor(period)
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(512, 1_024, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(1_024, 1_024, (5, 1), 1, (2, 0))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(1_024, 1, (3, 1), 1, (1, 0)))

    @torch.compiler.disable  # To remove when https://github.com/pytorch/pytorch/issues/165749 is fixed
    def forward_conv(self, conv: nn.Conv2d, x: Tensor) -> Tensor:
        return conv(x)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        feature_maps = []
        x = one_dim_to_two_dim(x, self.period)
        for conv in self.convs:
            x = F.leaky_relu(self.forward_conv(conv, x), LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        return x.view(x.size(0), -1), feature_maps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                PeriodDiscriminator(2),
                PeriodDiscriminator(3),
                PeriodDiscriminator(5),
                PeriodDiscriminator(7),
                PeriodDiscriminator(11),
            ]
        )

    def forward(self, x: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        y, feature_maps = zip(*[discr(x) for discr in self.discriminators], strict=True)
        return y, tuple(fm for fms in feature_maps for fm in fms)


class ScaleDiscriminator(nn.Module):
    def __init__(self, *, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm(nn.Conv1d(1, 128, 15, 1, 7)),
                norm(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
                norm(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
                norm(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
                norm(nn.Conv1d(512, 1_024, 41, 4, 20, groups=16)),
                norm(nn.Conv1d(1_024, 1_024, 41, 1, 20, groups=16)),
                norm(nn.Conv1d(1_024, 1_024, 5, 1, 2)),
            ]
        )
        self.conv_post = norm(nn.Conv1d(1_024, 1, 3, 1, 1))

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        feature_maps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        return x.view(x.size(0), -1), feature_maps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.poolings = nn.ModuleList([nn.Identity(), nn.AvgPool1d(4, 2, 2), nn.AvgPool1d(4, 2, 2)])
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(use_spectral_norm=True),
                ScaleDiscriminator(),
                ScaleDiscriminator(),
            ]
        )

    def forward(self, x: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        y, feature_maps = [], []
        for discriminator, pooling in zip(self.discriminators, self.poolings, strict=True):
            x = pooling(x)
            yi, fmi = discriminator(x)
            y.append(yi)
            feature_maps.extend(fmi)
        return tuple(y), tuple(feature_maps)


class UnitDiscriminatorOutput(NamedTuple):
    log_mel_spectrogram: Tensor
    mpd: tuple[Tensor, ...]
    mpd_features: tuple[Tensor, ...]
    msd: tuple[Tensor, ...]
    msd_features: tuple[Tensor, ...]


class UnitDiscriminator(nn.Module, PyTorchModelHubMixin):
    def __init__(self) -> None:
        super().__init__()
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1_024,
            win_length=1_024,
            hop_length=256,
            f_min=0,
            f_max=8_000,
            pad=384,
            n_mels=80,
            window_fn=torch.hann_window,
            power=1,
            normalized=False,
            center=False,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )

    @torch.compiler.disable
    def log_mel_spectrogram(self, waveform: Tensor) -> Tensor:
        return torch.clamp(self.mel_spectrogram(waveform), min=1e-5).log()

    def forward(self, audio: Tensor) -> UnitDiscriminatorOutput:
        log_mel_spectrogram = self.log_mel_spectrogram(audio.squeeze(1))
        mpd, mpd_features = self.mpd(audio)
        msd, msd_features = self.msd(audio)
        return UnitDiscriminatorOutput(log_mel_spectrogram, mpd, mpd_features, msd, msd_features)

    @classmethod
    def from_legacy_pretrained(cls, path: str | Path) -> "UnitDiscriminator":
        model = UnitDiscriminator().eval()
        try:
            state_dict = torch.load(path, map_location="cpu")
        except FileNotFoundError:
            state_dict = torch.hub.load_state_dict_from_url(str(path), map_location="cpu")
        state_dict = convert_discriminators_state_dict(state_dict["mpd"], state_dict["msd"])
        model.load_state_dict(state_dict)
        return model


class ConvLayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding_conv1 = (kernel_size * dilation - dilation) // 2
        padding_conv2 = (kernel_size - 1) // 2
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=padding_conv1, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=padding_conv2))

    def forward(self, x: Tensor) -> Tensor:
        xt = F.leaky_relu(x, LRELU_SLOPE)
        xt = self.conv1(xt)
        xt = F.leaky_relu(xt, LRELU_SLOPE)
        xt = self.conv2(xt)
        return x + xt


class MultiReceptiveFieldFusion(nn.Module):
    def __init__(self, channels: int, kernel_sizes: Sequence[int], dilations: Sequence[int]) -> None:
        super().__init__()
        self.channels = channels
        self.residual_blocks = nn.ModuleList(
            nn.Sequential(*[ConvLayer(channels, kernel_size, dilation) for dilation in dilations])
            for kernel_size in kernel_sizes
        )

    def forward(self, x: Tensor) -> Tensor:
        xs = torch.zeros_like(x)
        for block in self.residual_blocks:
            xs += block(x)
        return xs / len(self.residual_blocks)


def upsamplers(input_channels: int, kernel_sizes: Sequence[int], strides: Sequence[int]) -> nn.ModuleList:
    upsamplers = nn.ModuleList()
    for index, (kernel_size, stride) in enumerate(zip(kernel_sizes, strides, strict=True)):
        in_channels, out_channels = input_channels // 2**index, input_channels // 2 ** (index + 1)
        padding = (kernel_size - stride) // 2
        upsamplers.append(weight_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)))
    return upsamplers


def mrfs(input_channels: int, length: int, kernel_sizes: Sequence[int], dilations: Sequence[int]) -> nn.ModuleList:
    return nn.ModuleList(
        MultiReceptiveFieldFusion(input_channels // 2 ** (index + 1), kernel_sizes, dilations)
        for index in range(length)
    )


class Generator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        upsample_input_channels: int = 512,
        upsample_strides: Sequence[int] = (5, 4, 4, 2, 2),
        upsample_kernel_sizes: Sequence[int] = (11, 8, 8, 4, 4),
        mrf_kernel_sizes: Sequence[int] = (3, 7, 11),
        mrf_dilations: Sequence[int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        assert len(upsample_strides) == len(upsample_kernel_sizes)
        self.n_upsamples = len(upsample_strides)
        self.conv_pre = weight_norm(nn.Conv1d(input_dim, upsample_input_channels, 7, padding=3))
        self.upsamplers = upsamplers(upsample_input_channels, upsample_kernel_sizes, upsample_strides)
        self.mrfs = mrfs(upsample_input_channels, self.n_upsamples, mrf_kernel_sizes, mrf_dilations)
        self.conv_post = weight_norm(nn.Conv1d(self.mrfs[-1].channels, 1, 7, padding=3))
        self.apply(lambda m: normal_(m.weight, 0.0, 0.01) if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)) else None)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_pre(x)
        for i in range(self.n_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.upsamplers[i](x)
            x = self.mrfs[i](x)
        x = F.leaky_relu(x)  # Missing LRELU_SLOPE argument in the original code
        x = self.conv_post(x)
        return x.tanh()


def upsample_embedding(embedding: Tensor, n_frames: int) -> Tensor:
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(2)
    elif embedding.ndim == 1:
        embedding = embedding.view(-1, 1, 1)
    batch_size, channels, length = embedding.size()
    assert n_frames % length == 0
    return embedding.unsqueeze(3).repeat(1, 1, 1, n_frames // length).view(batch_size, channels, n_frames)


def tensor_from_name[T](items: Iterable[T], mapping: dict[T, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([[mapping[item]] for item in items], dtype=torch.long, device=device)


class UnitVocoder(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        n_units: int,
        *,
        speakers: list[str] | None = None,
        styles: list[str] | None = None,
        f0_bins: list[int] | None = None,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self._speakers, self._styles, self._f0_bins = speakers or [], styles or [], f0_bins or []
        self.unit_embedding = nn.Embedding(n_units, embedding_dim)
        self.speaker_embedding = nn.Embedding(len(self.speakers), embedding_dim) if self.speakers else None
        self.style_embedding = nn.Embedding(len(self.styles), embedding_dim) if self.styles else None
        self.f0_embedding = nn.Embedding(len(self.f0_bins), embedding_dim) if self.f0_bins else None
        input_dim = embedding_dim * (1 + bool(self.speakers) + bool(self.styles) + bool(self.f0_bins))
        self.generator = Generator(input_dim)
        self._speaker_to_index = {s: i for i, s in enumerate(self.speakers)}
        self._style_to_index = {s: i for i, s in enumerate(self.styles)}
        self._f0_bins_to_index = {b: i for i, b in enumerate(self.f0_bins)}

    def forward(
        self,
        units: Tensor,
        *,
        speaker: Tensor | None = None,
        style: Tensor | None = None,
        f0: Tensor | None = None,
    ) -> Tensor:
        x = self.unit_embedding(units).transpose(1, 2)
        if self.f0_embedding:
            assert f0 is not None, "f0 must be provided when f0_embedding is used"
            f0_embed = upsample_embedding(self.f0_embedding(f0).transpose(1, 2), x.size(-1))
            x = torch.cat([x, f0_embed], dim=1)
        if self.speaker_embedding:
            assert speaker is not None, "speaker must be provided when speaker_embedding is used"
            speaker_embed = upsample_embedding(self.speaker_embedding(speaker).transpose(1, 2), x.size(-1))
            x = torch.cat([x, speaker_embed], dim=1)
        if self.style_embedding:
            assert style is not None, "style must be provided when style_embedding is used"
            style_embed = upsample_embedding(self.style_embedding(style).transpose(1, 2), x.size(-1))
            x = torch.cat([x, style_embed], dim=1)
        return self.generator(x)

    @torch.inference_mode()
    def generate(
        self,
        units: Tensor,
        *,
        speaker: Iterable[str] | None = None,
        style: Iterable[str] | None = None,
        f0: Iterable[int] | None = None,
    ) -> Tensor:
        audio = self(
            units,
            speaker=tensor_from_name(speaker, self._speaker_to_index, units.device) if speaker else None,
            style=tensor_from_name(style, self._style_to_index, units.device) if style else None,
            f0=tensor_from_name(f0, self._f0_bins_to_index, units.device) if f0 else None,
        ).squeeze(1)
        return audio / audio.abs().max(dim=-1, keepdim=True).values

    @property
    def n_units(self) -> int:
        return self.unit_embedding.weight.size(0)

    @property
    def speakers(self) -> list[str]:
        return self._speakers

    @speakers.setter
    def speakers(self, value: list[str]) -> None:
        if self._speakers is None or len(value) != len(self._speakers):
            raise ValueError("Cannot change the number of speakers")
        self._speakers = value
        self._speaker_to_index = {s: i for i, s in enumerate(self._speakers)}
        self._hub_mixin_config["speakers"] = self._speakers

    @property
    def styles(self) -> list[str]:
        return self._styles

    @styles.setter
    def styles(self, value: list[str]) -> None:
        if self._styles is None or len(value) != len(self._styles):
            raise ValueError("Cannot change the number of styles")
        self._styles = value
        self._style_to_index = {s: i for i, s in enumerate(self._styles)}
        self._hub_mixin_config["styles"] = self._styles

    @property
    def f0_bins(self) -> list[int]:
        return self._f0_bins

    @f0_bins.setter
    def f0_bins(self, value: list[int]) -> None:
        if self._f0_bins is None or len(value) != len(self._f0_bins):
            raise ValueError("Cannot change the number of f0 bins")
        self._f0_bins = value
        self._f0_bins_to_index = {b: i for i, b in enumerate(self._f0_bins)}
        self._hub_mixin_config["f0_bins"] = self._f0_bins

    @classmethod
    def from_legacy_pretrained(cls, path: str | Path) -> "UnitVocoder":
        try:
            state_dict = torch.load(path, map_location="cpu")["generator"]
        except FileNotFoundError:
            state_dict = torch.hub.load_state_dict_from_url(str(path), map_location="cpu")["generator"]
        n_units, speakers, styles, f0_bins = legacy_metadata(state_dict)
        model = UnitVocoder(n_units, speakers=speakers, styles=styles, f0_bins=f0_bins).eval()
        model.load_state_dict(convert_vocoder_state_dict(state_dict))
        return model
