import torch

from unit_hifigan.compatibility import convert_from_legacy_norm, find_weight_tuples, legacy_metadata


def test_find_weight_tuples() -> None:
    keys = ["conv.weight_g", "conv.weight_v", "other.weight", "post.weight_g"]
    assert find_weight_tuples(keys, ["weight_g", "weight_v"]) == [("conv.weight_g", "conv.weight_v")]
    assert find_weight_tuples(["weight_g", "weight_v"], ["weight_g", "weight_v"]) == [("weight_g", "weight_v")]


def test_convert_from_legacy_weight_norm() -> None:
    state_dict = {
        "conv.weight_g": torch.randn(4, 1, 1),
        "conv.weight_v": torch.randn(4, 2, 3),
        "conv.bias": torch.randn(4),
    }
    converted = convert_from_legacy_norm(state_dict)
    assert set(converted) == {
        "conv.parametrizations.weight.original0",
        "conv.parametrizations.weight.original1",
        "conv.bias",
    }
    torch.testing.assert_close(converted["conv.parametrizations.weight.original0"], state_dict["conv.weight_g"])
    torch.testing.assert_close(converted["conv.parametrizations.weight.original1"], state_dict["conv.weight_v"])


def test_convert_from_legacy_spectral_norm() -> None:
    state_dict = {
        "conv.weight_orig": torch.randn(4, 2, 3),
        "conv.weight_u": torch.randn(4),
        "conv.weight_v": torch.randn(6),
    }
    converted = convert_from_legacy_norm(state_dict)
    assert set(converted) == {
        "conv.parametrizations.weight.original",
        "conv.parametrizations.weight.0._u",
        "conv.parametrizations.weight.0._v",
    }


def test_legacy_metadata() -> None:
    state_dict = {"dict.weight": torch.randn(500, 128), "spkr.weight": torch.randn(3, 128)}
    n_units, speakers, styles, f0_bins = legacy_metadata(state_dict)
    assert n_units == 500
    assert speakers == ["spkr0", "spkr1", "spkr2"]
    assert styles is None
    assert f0_bins is None
