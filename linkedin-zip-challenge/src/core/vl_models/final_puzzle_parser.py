# src/core/vl_models/final_puzzle_parser.py
"""SCRATCHPAD -- superseded by ``puzzle_parser.py``. Do not build on this file.

This is the 2025-10-24 proof of concept that settled the "hybrid strategy" question
(prompt engineering plus local JSON parsing, instead of tool calling). It is kept
because the handover and the reports reference it by name, and because its
``__main__`` block is a handy one-shot demo.

What was wrong with it, and why ``puzzle_parser.py`` exists:

*   it hard-coded ``openbmb/minicpm-o2.6`` and ``http://localhost:11434/v1`` -- the
    2025-10 model and the **wrong port** (this project publishes Ollama on 11435),
    bypassing ``src.settings`` entirely;
*   it printed instead of logging, against ``rules.md``;
*   it returned ``None`` on failure, which makes a caller fail silently;
*   it appended to ``sys.path`` at import time.

The frozen few-shot assets it used to define now live in ``prompt_baseline.py`` so
they no longer depend on a scratchpad; they are re-exported here unchanged for the
modules that still import them by this path.
"""

import json
from pathlib import Path

from loguru import logger

from src.core.vl_models.prompt_baseline import (
    PUZZLE_01_JSON_STR,
    PUZZLE_02_JSON_STR,
    PUZZLE_03_JSON_STR,
    build_puzzle_prompt,
)
from src.core.vl_models.puzzle_parser import (
    ParseResult,
    PuzzleParseError,
    extract_json_block,
    parse_puzzle_image,
)

__all__ = [
    "PUZZLE_01_JSON_STR",
    "PUZZLE_02_JSON_STR",
    "PUZZLE_03_JSON_STR",
    "build_puzzle_prompt",
    "extract_json_block",
    "parse_puzzle_image",
]


def _demo(image_path: Path) -> ParseResult | None:
    """One-shot demo against the configured backend."""
    logger.info("--- Parsing {} ---", image_path.name)
    try:
        result = parse_puzzle_image(image_path)
    except (PuzzleParseError, FileNotFoundError) as error:
        logger.error("{}: {}", type(error).__name__, error)
        return None

    logger.success("Parsed {}", image_path.name)
    logger.info(
        json.dumps(
            result.puzzle,
            default=lambda o: sorted(o) if isinstance(o, set) else str(o),
            indent=2,
        )
    )
    for warning in result.warnings:
        logger.warning(warning)
    return result


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    illustrations_dir = project_root / "illustrations"
    # puzzle_01 is in the few-shot prompt (a reproduction check, not generalisation);
    # puzzle_04 is not, so it is the one that actually tests the model.
    for name in ("puzzle_01.png", "puzzle_04.png"):
        _demo(illustrations_dir / name)
