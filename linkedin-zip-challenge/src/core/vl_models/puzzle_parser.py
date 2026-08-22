# src/core/vl_models/puzzle_parser.py
"""Turns a screenshot of a Zip puzzle into a solver-ready ``Puzzle``.

This is the supported entry point. ``final_puzzle_parser.py`` is the 2025-10
scratchpad it replaces: that one hard-codes a model tag and the wrong Ollama port,
bypasses ``src.settings`` and prints instead of logging.

Design notes:

*   The model is asked for a JSON string rather than a tool call. Tool calling was
    tried on 2025-10-24 and rejected: the models that supported it had broken vision.
*   Transport lives in ``backends.py`` so the benchmark and the app exercise exactly
    the same code path, including the thinking switch.
*   Failures raise. A parser that returns ``None`` makes an API endpoint answer 200
    with an empty body, which is the silent breakage the plan calls out.
*   Walls the model invents outside the grid, or between cells that are not
    neighbours, are dropped and reported in ``ParseResult.warnings`` -- dropping them
    silently would hide exactly the failure mode the baseline flagged as fatal.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from src.core.utils import Puzzle, parse_puzzle_layout
from src.core.vl_models.backends import (
    BACKEND_OPENAI_COMPAT,
    VisionBackend,
    build_backend,
)
from src.core.vl_models.prompt_baseline import build_puzzle_prompt
from src.core.vl_models.schema import SimplePuzzleOutput, WallPair
from src.settings import get_settings

JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*\})\s*```", re.DOTALL)

# The baseline setting: reasoning off. Measured 2026-08-22 on qwen3.5:4b-q8_0 as
# 4.1s/JSON 2/2 with it off against 66s/JSON 0/2 with it on.
DEFAULT_THINK = False


class PuzzleParseError(Exception):
    """Base class for every way parsing an image can fail."""


class VisionBackendError(PuzzleParseError):
    """The model could not be reached, or refused the request."""


class ModelOutputError(PuzzleParseError):
    """The model answered, but not with a puzzle we can use."""


@dataclass(frozen=True)
class ParseResult:
    """The puzzle, plus anything the caller should show the user before solving."""

    puzzle: Puzzle
    warnings: tuple[str, ...] = ()


def extract_json_block(text: str) -> str | None:
    """Pulls the JSON object out of a markdown block, or out of bare text."""
    match = JSON_BLOCK_PATTERN.search(text)
    if match:
        return match.group(1)
    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index != -1 and end_index > start_index:
        return text[start_index : end_index + 1]
    return None


def default_backend() -> VisionBackend:
    """Builds the backend the app ships with, from ``.env`` -- never hard-coded."""
    settings = get_settings()
    if not settings.ollama_model_name or not settings.ollama_provider_url:
        raise VisionBackendError(
            "ollama_model_name and ollama_provider_url must be set (see .env.example). "
            "Without them there is no model to call."
        )
    return build_backend(
        name=BACKEND_OPENAI_COMPAT,
        model=settings.ollama_model_name,
        base_url=settings.ollama_provider_url,
        think=DEFAULT_THINK,
    )


def _validate_layout(layout: list[list[str]]) -> None:
    if not layout or not layout[0]:
        raise ModelOutputError("The model returned an empty grid.")
    widths = {len(row) for row in layout}
    if len(widths) != 1:
        raise ModelOutputError(
            f"The model returned a ragged grid: rows have widths {sorted(widths)}."
        )


def _collect_walls(
    wall_pairs: list[WallPair], height: int, width: int
) -> tuple[set[tuple[tuple[int, int], tuple[int, int]]], list[str]]:
    walls: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    warnings: list[str] = []

    for pair in wall_pairs:
        if len(pair.cell1) != 2 or len(pair.cell2) != 2:
            warnings.append(f"Dropped a wall with malformed coordinates: {pair}.")
            continue
        cell1 = (pair.cell1[0], pair.cell1[1])
        cell2 = (pair.cell2[0], pair.cell2[1])
        if not _in_bounds(cell1, height, width) or not _in_bounds(cell2, height, width):
            warnings.append(f"Dropped an out-of-bounds wall: {cell1}-{cell2}.")
            continue
        if abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1]) != 1:
            warnings.append(
                f"Dropped a wall between non-adjacent cells: {cell1}-{cell2}."
            )
            continue
        walls.add(tuple(sorted((cell1, cell2))))

    return walls, warnings


def _in_bounds(cell: tuple[int, int], height: int, width: int) -> bool:
    return 0 <= cell[0] < height and 0 <= cell[1] < width


def to_puzzle(model_output: SimplePuzzleOutput) -> ParseResult:
    """Converts validated model output into the solver's ``Puzzle`` format."""
    _validate_layout(model_output.layout)
    puzzle = parse_puzzle_layout(model_output.layout)
    height, width = puzzle["grid_size"]
    walls, warnings = _collect_walls(model_output.walls, height, width)
    puzzle["walls"] = walls

    if not puzzle["num_map"]:
        warnings.append("The model found no numbered waypoints in the image.")
    for message in warnings:
        logger.warning(message)

    return ParseResult(puzzle=puzzle, warnings=tuple(warnings))


def parse_model_output(text: str) -> ParseResult:
    """The pure half: model text in, puzzle out. No network, so it is unit-testable."""
    json_str = extract_json_block(text)
    if not json_str:
        raise ModelOutputError(
            "No JSON object found in the model output. This usually means reasoning "
            "was left on and the answer was buried in it."
        )
    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as error:
        raise ModelOutputError(f"The model emitted invalid JSON: {error}") from error
    try:
        validated = SimplePuzzleOutput(**payload)
    except ValidationError as error:
        raise ModelOutputError(
            f"The model's JSON does not match the expected schema: {error}"
        ) from error
    return to_puzzle(validated)


def parse_puzzle_image(
    image_path: Path,
    backend: VisionBackend | None = None,
    prompt: str | None = None,
) -> ParseResult:
    """Reads a puzzle screenshot and returns a solver-ready puzzle.

    Raises ``VisionBackendError`` if the model cannot be reached and
    ``ModelOutputError`` if it answers with something unusable.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"No such image: {image_path}")

    backend = backend or default_backend()
    prompt = prompt if prompt is not None else build_puzzle_prompt()

    logger.info(
        "Parsing {} with {} via {}", image_path.name, backend.model, backend.name
    )
    try:
        response = backend.generate(image_path, prompt)
    except Exception as error:
        raise VisionBackendError(
            f"The vision backend '{backend.name}' failed for model "
            f"'{backend.model}': {type(error).__name__}: {error}"
        ) from error

    if response.thinking_characters:
        logger.debug(
            "Model spent {} characters on reasoning", response.thinking_characters
        )
    return parse_model_output(response.text)
