"""Tests for the discriminator model.

We cannot test parity in training mode for the MultiScaleDiscriminator because
of the spectral_norm. The legacy and the current implementations differ in training
mode: the old one first updates the "v" tensor and then "u", while the new one
does the opposite. See:
- https://github.com/pytorch/pytorch/blob/69b05913fb0332f9a938c74e26b106e2bd24d82e/torch/nn/utils/spectral_norm.py#L99-L106
- https://github.com/pytorch/pytorch/blob/69b05913fb0332f9a938c74e26b106e2bd24d82e/torch/nn/utils/parametrizations.py#L488-L503
"""

import warnings

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from models import MultiPeriodDiscriminator, MultiScaleDiscriminator  # ty: ignore[unresolved-import]

from unit_hifigan.compatibility import convert_discriminators_state_dict
from unit_hifigan.model import UnitDiscriminator

type Discriminators = tuple[UnitDiscriminator, tuple[MultiPeriodDiscriminator, MultiScaleDiscriminator]]


@pytest.fixture(scope="session")
def discriminators(device: torch.device) -> Discriminators:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ref_mpd = MultiPeriodDiscriminator().to(device)
        ref_msd = MultiScaleDiscriminator().eval().to(device)
    ours = UnitDiscriminator().to(device)
    ours.msd.eval()
    ours.load_state_dict(convert_discriminators_state_dict(ref_mpd.state_dict(), ref_msd.state_dict()))
    return ours, (ref_mpd, ref_msd)


@given(batch_size=st.integers(1, 32), length=st.integers(256, 64_000))
@settings(deadline=None)
def test_discriminator(discriminators: Discriminators, device: torch.device, batch_size: int, length: int) -> None:
    x = torch.testing.make_tensor((batch_size, 1, length), dtype=torch.float32, device=device)
    ours, (ref_mpd, ref_msd) = discriminators
    (y1, _, fmap1, _), (y2, _, fmap2, _) = ref_mpd(x, x), ref_msd(x, x)
    output = ours(x)
    torch.testing.assert_close(output.mpd, y1)
    torch.testing.assert_close(output.msd, y2)
    k = 0
    for fmap in fmap1:
        for layer in fmap:
            torch.testing.assert_close(output.mpd_features[k], layer)
            k += 1
    k = 0
    for fmap in fmap2:
        for layer in fmap:
            torch.testing.assert_close(output.msd_features[k], layer)
            k += 1
