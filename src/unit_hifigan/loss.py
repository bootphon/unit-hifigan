import torch
from torch import Tensor

from unit_hifigan.model import UnitDiscriminatorOutput

type DictTensor = dict[str, Tensor]

LAMBDA_MEL = 45.0
LAMBDA_FEATURE_MATCHING = 2.0


def feature_matching_loss(fm_real: tuple[Tensor, ...], fm_gen: tuple[Tensor, ...]) -> Tensor:
    return sum([loss.mean() for loss in torch._foreach_abs(torch._foreach_add(fm_gen, fm_real, alpha=-1.0))])


def gan_generator_loss(y_gen: tuple[Tensor, ...]) -> Tensor:
    return sum([loss.mean() for loss in torch._foreach_pow(torch._foreach_add(y_gen, -1), 2)])


def gan_discriminator_loss(y_real: tuple[Tensor, ...], y_gen: tuple[Tensor, ...]) -> Tensor:
    losses_real = torch._foreach_pow(torch._foreach_add(y_real, -1), 2)
    loss_gen = torch._foreach_pow(y_gen, 2)
    return sum([loss.mean() for loss in losses_real + loss_gen])


@torch.compile(fullgraph=True, dynamic=False)
def discriminator_loss(real: UnitDiscriminatorOutput, generated: UnitDiscriminatorOutput) -> tuple[Tensor, DictTensor]:
    loss_discr_mpd = gan_discriminator_loss(real.mpd, generated.mpd)
    loss_discr_msd = gan_discriminator_loss(real.msd, generated.msd)
    return loss_discr_mpd + loss_discr_msd, {"mpd": loss_discr_mpd, "msd": loss_discr_msd}


@torch.compile(fullgraph=True, dynamic=False)
def generator_loss(real: UnitDiscriminatorOutput, generated: UnitDiscriminatorOutput) -> tuple[Tensor, DictTensor]:
    loss_mel = (real.log_mel_spectrogram - generated.log_mel_spectrogram).abs().mean()
    loss_fm_mpd = feature_matching_loss(real.mpd_features, generated.mpd_features)
    loss_fm_msd = feature_matching_loss(real.msd_features, generated.msd_features)
    loss_gen_mpd, loss_gen_msd = gan_generator_loss(generated.mpd), gan_generator_loss(generated.msd)
    loss = (
        LAMBDA_MEL * loss_mel
        + LAMBDA_FEATURE_MATCHING * loss_fm_mpd
        + LAMBDA_FEATURE_MATCHING * loss_gen_mpd
        + loss_fm_msd
        + loss_gen_msd
    )
    losses = {"mel": loss_mel, "fm_mpd": loss_fm_mpd, "fm_msd": loss_fm_msd, "mpd": loss_gen_mpd, "msd": loss_gen_msd}
    return loss, losses
