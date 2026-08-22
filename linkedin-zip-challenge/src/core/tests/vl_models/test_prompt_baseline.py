# src/core/tests/vl_models/test_prompt_baseline.py
"""Pins the frozen prompts so they cannot drift without a failing test.

Every number in ``ai-collab/reports/2026-08-15_vl-p0-p1-baseline.html`` was measured
against these exact strings. Editing them silently would make the published baseline
incomparable with anything measured afterwards, and nothing else in the suite would
notice -- which is why the guarantee is a hash rather than a comment.

If a change here is intentional, re-measure the baseline and update both the hash and
the report; do not just paste in the new digest.
"""

import hashlib

from src.core.vl_models.prompt_baseline import build_puzzle_prompt
from src.core.vl_models.prompt_variants import build_sized_puzzle_prompt

BASELINE_PROMPT_SHA256 = (
    "b8e75a8c8c66b1581d4aeb3bed26fcdc0d4b5654119f72bb397eb2bb2a63cfaa"
)
SIZED_PROMPT_SHA256 = "ef7298f344c3fe3ec87a2b0a1c9a2536ac237b3707d708ac4a4c2034c6404228"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_baseline_prompt_is_unchanged():
    assert _sha256(build_puzzle_prompt()) == BASELINE_PROMPT_SHA256


def test_sized_prompt_is_unchanged():
    assert _sha256(build_sized_puzzle_prompt()) == SIZED_PROMPT_SHA256


def test_scratchpad_reexports_the_same_prompt():
    """``final_puzzle_parser`` is superseded but still imported by name elsewhere."""
    from src.core.vl_models.final_puzzle_parser import (
        build_puzzle_prompt as reexported,
    )

    assert reexported() == build_puzzle_prompt()
