from collections.abc import Iterator

import pytest
import torch

from unit_hifigan.loss import (
    LAMBDA_FEATURE_MATCHING,
    LAMBDA_MEL,
    discriminator_loss,
    feature_matching_loss,
    gan_discriminator_loss,
    gan_generator_loss,
    generator_loss,
)
from unit_hifigan.model import UnitDiscriminatorOutput


@pytest.fixture(autouse=True)
def force_eager() -> Iterator[None]:
    with torch.compiler.set_stance("force_eager"):
        yield


def fake_discriminator_output(generator: torch.Generator) -> UnitDiscriminatorOutput:
    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    return UnitDiscriminatorOutput(
        log_mel_spectrogram=randn(2, 80, 35),
        mpd=tuple(randn(2, 16) for _ in range(5)),
        mpd_features=tuple(randn(2, 4, 8) for _ in range(30)),
        msd=tuple(randn(2, 16) for _ in range(3)),
        msd_features=tuple(randn(2, 4, 8) for _ in range(24)),
    )


def test_feature_matching_loss() -> None:
    generator = torch.Generator().manual_seed(0)
    fm_real = tuple(torch.randn(2, 4, 8, generator=generator) for _ in range(3))
    fm_gen = tuple(torch.randn(2, 4, 8, generator=generator) for _ in range(3))
    expected = sum((gen - real).abs().mean() for real, gen in zip(fm_real, fm_gen, strict=True))
    torch.testing.assert_close(feature_matching_loss(fm_real, fm_gen), expected)


def test_gan_generator_loss() -> None:
    generator = torch.Generator().manual_seed(0)
    y_gen = tuple(torch.randn(2, 16, generator=generator) for _ in range(3))
    expected = sum((y - 1).square().mean() for y in y_gen)
    torch.testing.assert_close(gan_generator_loss(y_gen), expected)
    torch.testing.assert_close(gan_generator_loss((torch.ones(2, 16),)), torch.tensor(0.0))


def test_gan_discriminator_loss() -> None:
    generator = torch.Generator().manual_seed(0)
    y_real = tuple(torch.randn(2, 16, generator=generator) for _ in range(3))
    y_gen = tuple(torch.randn(2, 16, generator=generator) for _ in range(3))
    expected = sum((r - 1).square().mean() + g.square().mean() for r, g in zip(y_real, y_gen, strict=True))
    torch.testing.assert_close(gan_discriminator_loss(y_real, y_gen), expected)
    perfect = gan_discriminator_loss((torch.ones(2, 16),), (torch.zeros(2, 16),))
    torch.testing.assert_close(perfect, torch.tensor(0.0))


def test_generator_loss_weighting() -> None:
    """The total must match the original: 45 * mel + 2 * (fm_mpd + fm_msd) + gen_mpd + gen_msd."""
    generator = torch.Generator().manual_seed(0)
    real, generated = fake_discriminator_output(generator), fake_discriminator_output(generator)
    loss, losses = generator_loss(real, generated)
    expected = (
        LAMBDA_MEL * losses["mel"]
        + LAMBDA_FEATURE_MATCHING * (losses["fm_mpd"] + losses["fm_msd"])
        + losses["mpd"]
        + losses["msd"]
    )
    torch.testing.assert_close(loss, expected)
    expected_mel = (real.log_mel_spectrogram - generated.log_mel_spectrogram).abs().mean()
    torch.testing.assert_close(losses["mel"], expected_mel)
    torch.testing.assert_close(losses["fm_mpd"], feature_matching_loss(real.mpd_features, generated.mpd_features))
    torch.testing.assert_close(losses["fm_msd"], feature_matching_loss(real.msd_features, generated.msd_features))
    torch.testing.assert_close(losses["mpd"], gan_generator_loss(generated.mpd))
    torch.testing.assert_close(losses["msd"], gan_generator_loss(generated.msd))


def test_discriminator_loss() -> None:
    generator = torch.Generator().manual_seed(0)
    real, generated = fake_discriminator_output(generator), fake_discriminator_output(generator)
    loss, losses = discriminator_loss(real, generated)
    torch.testing.assert_close(loss, losses["mpd"] + losses["msd"])
    torch.testing.assert_close(losses["mpd"], gan_discriminator_loss(real.mpd, generated.mpd))
    torch.testing.assert_close(losses["msd"], gan_discriminator_loss(real.msd, generated.msd))
