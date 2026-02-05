import warnings

import pytest
import torch
from examples.expresso.models import MultiSpkrMultiStyleCodeGenerator  # ty: ignore[unresolved-import]
from hypothesis import given, settings
from hypothesis import strategies as st
from utils import AttrDict  # ty: ignore[unresolved-import]

from unit_hifigan import UnitVocoder
from unit_hifigan.compatibility import convert_vocoder_state_dict

type Vocoders = tuple[UnitVocoder, MultiSpkrMultiStyleCodeGenerator]


@pytest.fixture(scope="session")
def vocoders(config: dict, device: torch.device) -> Vocoders:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ref = MultiSpkrMultiStyleCodeGenerator(AttrDict(config))
    speakers = [f"speaker-{k}" for k in range(ref.spkr.weight.size(0))] if ref.spkr is not None else None
    styles = [f"style-{k}" for k in range(ref.style.weight.size(0))] if ref.style is not None else None
    f0_bins = list(range(ref.f0.weight.size(0))) if ref.f0 is not None else None
    ours = UnitVocoder(n_units=config["num_embeddings"], speakers=speakers, styles=styles, f0_bins=f0_bins)
    ours.load_state_dict(convert_vocoder_state_dict(ref.state_dict()))
    return ours.to(device), ref.to(device)


@given(batch_size=st.integers(1, 32), length=st.integers(1, 200))
@settings(deadline=None)
def test_vocoder(vocoders: Vocoders, device: torch.device, batch_size: int, length: int) -> None:
    ours, ref = vocoders
    n_speakers, n_styles = len(ours.speakers), len(ours.styles)
    units = torch.testing.make_tensor((batch_size, length), dtype=torch.long, device=device, low=0, high=ours.n_units)
    speaker = torch.testing.make_tensor((batch_size, 1), dtype=torch.long, device=device, low=0, high=n_speakers)
    style = torch.testing.make_tensor((batch_size, 1), dtype=torch.long, device=device, low=0, high=n_styles)
    x1 = ref(code=units, spkr=speaker, style=style)[0]
    x2 = ours(units, speaker=speaker, style=style)
    torch.testing.assert_close(x1, x2)
