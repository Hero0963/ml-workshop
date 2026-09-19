# src/app/schemas/puzzle.py
"""Request and response shapes for generating a puzzle.

The board comes back in the same shape the vision endpoint uses -- a layout of
two-character cells and a list of walls -- so a client that can show a board read from a
screenshot can show a generated one with the same code.
"""

from pydantic import BaseModel, Field

from src.app.schemas.vision import WallOut

#: The sizes the UIs offer. The generator itself takes any size, but past 6x6 its
#: time-limited search starts to fail, and 4-6 is also what the RL solver was trained on.
MIN_GENERATED_SIZE = 4
MAX_GENERATED_SIZE = 6
DEFAULT_GENERATED_SIZE = 6


class GenerateRequest(BaseModel):
    """Schema for the generate request body."""

    size: int = Field(
        DEFAULT_GENERATED_SIZE,
        ge=MIN_GENERATED_SIZE,
        le=MAX_GENERATED_SIZE,
        description="Rows and columns of the square board.",
    )


class GenerateResponse(BaseModel):
    """A new board. It always has a solution: the path is drawn before the numbers."""

    grid_size: tuple[int, int]
    layout: list[list[str]] = Field(
        ...,
        description='Two-character cells: "  " for empty, a zero-padded number such as "01" for a waypoint.',
    )
    walls: list[WallOut] = Field(default_factory=list)
