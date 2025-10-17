from collections import defaultdict
from collections.abc import Iterable

import torch
from torch import Tensor
from torchaudio.functional import melscale_fbanks


def find_weight_tuples(keys: Iterable[str], required: list[str]) -> list[tuple[str, ...]]:
    prefix_map = defaultdict(dict)
    for key in keys:
        if "." in key:
            prefix, suffix = key.rsplit(".", 1)
            prefix_map[prefix][suffix] = key
        else:
            prefix_map[None][key] = key
    return [
        tuple(suffix_dict[s] for s in required)
        for suffix_dict in prefix_map.values()
        if all(s in suffix_dict for s in required)
    ]


def convert_from_legacy_norm(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    new_state_dict, norm_keys = {}, set()
    # First convert spectral_norm keys
    for weight_orig, u, v in find_weight_tuples(state_dict.keys(), ["weight_orig", "weight_u", "weight_v"]):
        new_weight_orig_key = weight_orig.removesuffix("weight_orig") + "parametrizations.weight.original"
        new_state_dict[new_weight_orig_key] = state_dict[weight_orig]
        new_state_dict[u.removesuffix("weight_u") + "parametrizations.weight.0._u"] = state_dict[u]
        new_state_dict[v.removesuffix("weight_v") + "parametrizations.weight.0._v"] = state_dict[v]
        norm_keys.update({weight_orig, u, v})

    # Then convert weight_norm keys
    for g, v in find_weight_tuples(state_dict.keys(), ["weight_g", "weight_v"]):
        new_state_dict[g.removesuffix("weight_g") + "parametrizations.weight.original0"] = state_dict[g]
        new_state_dict[v.removesuffix("weight_v") + "parametrizations.weight.original1"] = state_dict[v]
        norm_keys.update({g, v})

    # Finally copy other keys
    new_state_dict |= {k: v for k, v in state_dict.items() if k not in norm_keys}
    return new_state_dict


def convert_vocoder_state_dict(old_state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    new_state_dict = {}
    for key, tensor in old_state_dict.items():
        if key.startswith("ups."):
            new_state_dict[key.replace("ups.", "generator.upsamplers.")] = tensor
        elif key.startswith("conv_"):
            new_state_dict[f"generator.{key}"] = tensor
        elif key == "dict.weight":
            new_state_dict["unit_embedding.weight"] = tensor
        elif key == "spkr.weight":
            new_state_dict["speaker_embedding.weight"] = tensor
        elif key == "style.weight":
            new_state_dict["style_embedding.weight"] = tensor
        elif key.startswith("resblocks"):
            old_layer = key.split(".")[1]
            new_layer, new_block = divmod(int(old_layer), 3)
            conv, sub_layer, weight = key.removeprefix(f"resblocks.{old_layer}.").split(".")
            conv = {"convs1": "conv1", "convs2": "conv2"}[conv]
            new_k = f"generator.mrfs.{new_layer}.residual_blocks.{new_block}.{sub_layer}.{conv}.{weight}"
            new_state_dict[new_k] = tensor
        else:
            raise ValueError(f"Unexpected key {key} in state_dict")
    return convert_from_legacy_norm(new_state_dict)


def convert_discriminators_state_dict(
    old_mpd_state_dict: dict[str, Tensor],
    old_msd_state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    mpd = {f"mpd.{key}": val for key, val in convert_from_legacy_norm(old_mpd_state_dict).items()}
    msd = {f"msd.{key}": val for key, val in convert_from_legacy_norm(old_msd_state_dict).items()}
    mel_spectrogram = {
        "mel_spectrogram.spectrogram.window": torch.hann_window(1_024),
        "mel_spectrogram.mel_scale.fb": melscale_fbanks(1_024 // 2 + 1, 0, 8_000, 80, 16_000, "slaney", "slaney"),
    }
    return mpd | msd | mel_spectrogram


def legacy_metadata(state_dict: dict[str, Tensor]) -> tuple[int, list[str] | None, list[str] | None, list[int] | None]:
    n_units = state_dict["dict.weight"].size(0)
    speakers = [f"spkr{i}" for i in range(state_dict["spkr.weight"].size(0))] if "spkr.weight" in state_dict else None
    styles = [f"style{i}" for i in range(state_dict["style.weight"].size(0))] if "style.weight" in state_dict else None
    f0_bins = list(range(state_dict["f0_dict.weight"].size(0))) if "f0_dict.weight" in state_dict else None
    return n_units, speakers, styles, f0_bins
