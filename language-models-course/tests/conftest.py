# tests/conftest.py
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
