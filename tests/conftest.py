import json
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def config(reference_path: Path) -> dict:
    with (Path(__file__).parent / "speech-resynthesis/examples/expresso/config/expresso_config.json").open() as f:
        return json.load(f)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


sys.path.append(f"{Path(__file__).parent}/speech-resynthesis")
