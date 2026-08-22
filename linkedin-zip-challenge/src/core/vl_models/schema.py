# src/core/vl_models/schema.py
"""The one contract shared by the training labels and the model's answer.

This is the root fix for a mismatch the 2025-10 dataset had: it wrote a
Chain-of-Draft label (``chain_of_draft`` + ``final_ans``) while the parser expected
``layout`` + ``walls``, so anything trained on it needed a conversion step and the
training target never looked like what the model is asked to emit at inference.

Here the same Pydantic model defines both ends:

*   ``dataset_builder`` renders a puzzle and writes ``to_prompt_json(from_puzzle(p))``
    as the label -- byte-for-byte the shape the few-shot examples use;
*   ``puzzle_parser`` validates the model's reply with ``SimplePuzzleOutput``.

So the fine-tuning target *is* the inference format. No adapter, and no way for the
two to drift apart without a test failing.

Nothing here depends on ``pydantic-ai``. That library is only the transport on the
inference side (``backends.py``); the schema is plain ``pydantic``, which is what
lets the offline dataset builder share it.
"""

import json

from pydantic import BaseModel, Field

from src.core.utils import Puzzle

EMPTY_CELL = "  "
BLOCKED_CELL = "xx"
CELL_WIDTH = 2

# The few-shot examples in prompt_baseline.py are rendered at this indent; labels use
# the same one so the training target matches the demonstrated format exactly.
PROMPT_JSON_INDENT = 2


class WallPair(BaseModel):
    """A wall between two adjacent cells."""

    cell1: list[int] = Field(description="Coordinates [row, col] of the first cell.")
    cell2: list[int] = Field(
        description="Coordinates [row, col] of the second, adjacent cell."
    )


class SimplePuzzleOutput(BaseModel):
    """The JSON structure the model is instructed to generate."""

    layout: list[list[str]] = Field(
        description="2D array representing the grid. Use '  ' for empty cells, and two-digit strings like '01' for numbers."
    )
    walls: list[WallPair] = Field(description="A list of wall objects.")


def from_puzzle(puzzle: Puzzle) -> SimplePuzzleOutput:
    """Renders a solver-side ``Puzzle`` into the model-facing schema.

    Built from ``grid`` and ``blocked_cells`` rather than the stored
    ``puzzle_layout`` so it works for puzzles that came from the generator as well
    as those parsed from text.
    """
    height, width = puzzle["grid_size"]
    blocked = puzzle["blocked_cells"]
    grid = puzzle["grid"]

    layout: list[list[str]] = []
    for r in range(height):
        row: list[str] = []
        for c in range(width):
            if (r, c) in blocked:
                row.append(BLOCKED_CELL)
            elif grid[r][c] > 0:
                row.append(f"{grid[r][c]:0{CELL_WIDTH}d}")
            else:
                row.append(EMPTY_CELL)
        layout.append(row)

    walls = [
        WallPair(cell1=list(cell1), cell2=list(cell2))
        for cell1, cell2 in sorted(puzzle["walls"])
    ]
    return SimplePuzzleOutput(layout=layout, walls=walls)


def to_prompt_json(output: SimplePuzzleOutput) -> str:
    """Serialises a label exactly the way the few-shot examples present one.

    Plain ``json.dumps(..., indent=2)`` is wrong here: it puts every string and every
    integer on its own line, so a 6x6 board becomes ~40 lines instead of 6. That is
    both a format the prompt never demonstrates and roughly ten times the tokens, paid
    on every training example and every generated answer. Rows and wall objects
    therefore stay on one line, matching ``prompt_baseline.PUZZLE_01_JSON_STR``.
    """
    indent = " " * PROMPT_JSON_INDENT
    rows = ",\n".join(f"{indent * 2}{json.dumps(row)}" for row in output.layout)
    walls = ",\n".join(
        f"{indent * 2}{json.dumps(wall.model_dump())}" for wall in output.walls
    )
    walls_block = f"[\n{walls}\n{indent}]" if output.walls else "[]"
    return (
        "{\n"
        f'{indent}"layout": [\n{rows}\n{indent}],\n'
        f'{indent}"walls": {walls_block}\n'
        "}"
    )
