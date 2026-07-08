import torch

from unit_hifigan.wer.torchaudio import greedy_decode, normalize_text


def test_normalize_text() -> None:
    assert normalize_text("Hello, World!") == "hello  world"
    assert normalize_text("it's a test") == "it's a test"
    assert normalize_text("123 numbers") == "numbers"


def test_greedy_decode() -> None:
    id2token = {0: "-", 1: "A", 2: "B", 3: "|"}
    tokens = [1, 1, 0, 2, 3, 3, 2]  # Collapse repeats, drop blanks, "|" is the word separator
    emission = torch.nn.functional.one_hot(torch.tensor(tokens), num_classes=4).float()
    assert greedy_decode(emission, id2token) == "ab b"
