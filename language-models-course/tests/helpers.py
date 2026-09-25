# tests/helpers.py
"""Shared helpers for the tests (not fixtures)."""

from collections.abc import Callable
from typing import TypeVar

import pytest

T = TypeVar("T")


def download_or_skip(fn: Callable[[], T]) -> T:
    """Run a download; skip the test (instead of failing) when there is no network."""
    try:
        return fn()
    except OSError as error:
        pytest.skip(f"network unavailable: {error}")
