from pathlib import Path

import pytest

from unit_hifigan.train import TrainConfig


@pytest.fixture
def manifests(tmp_path: Path) -> tuple[Path, Path]:
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.touch()
    val.touch()
    return train, val


def test_train_config(tmp_path: Path, manifests: tuple[Path, Path]) -> None:
    train, val = manifests
    cfg = TrainConfig(str(tmp_path), str(train), str(val), n_units=100)
    assert cfg.n_units == 100
    assert cfg.segment_size % cfg.units_hop_size == 0


def test_train_config_betas_from_json(tmp_path: Path, manifests: tuple[Path, Path]) -> None:
    train, val = manifests
    cfg = TrainConfig(str(tmp_path), str(train), str(val), n_units=100, betas=[0.5, 0.9])  # ty: ignore[invalid-argument-type]
    assert cfg.betas == (0.5, 0.9)  # A JSON config provides a list, AdamW expects a tuple


def test_train_config_validation(tmp_path: Path, manifests: tuple[Path, Path]) -> None:
    train, val = manifests
    with pytest.raises(ValueError, match="Invalid dtype"):
        TrainConfig(str(tmp_path), str(train), str(val), n_units=100, dtype="float64")  # ty: ignore[invalid-argument-type]
    with pytest.raises(AssertionError, match="n_units must be positive"):
        TrainConfig(str(tmp_path), str(train), str(val), n_units=0)
    with pytest.raises(AssertionError, match="Train manifest not found"):
        TrainConfig(str(tmp_path), str(tmp_path / "missing.jsonl"), str(val), n_units=100)
    with pytest.raises(AssertionError, match="Validation manifest not found"):
        TrainConfig(str(tmp_path), str(train), str(tmp_path / "missing.jsonl"), n_units=100)
