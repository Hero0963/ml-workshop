# src/app/schemas/vision.py
"""Response shape for reading a puzzle out of a screenshot and solving it.

Three fields exist because the endpoint has three separable failure modes, and
collapsing them would hide the one that matters:

*   ``warnings`` -- walls the model invented outside the grid, or between cells that are
    not neighbours. The parser drops them, and dropping them silently is exactly the
    failure the baseline measurement flagged as fatal, so they are reported.
*   ``solvable`` -- the confidence flag. Every real board is built from a Hamiltonian
    path before the puzzle is carved out of it, so a real board always has a solution.
    An unsolvable reading is therefore *known* to be a misread, with no ground truth
    needed. The dangerous case is the opposite: a board that solves cleanly but was
    never in the picture, which nothing can flag.
*   ``solution_path`` -- absent when ``solvable`` is false, so a client cannot mistake
    "could not read it" for "there is no answer".
"""

from pydantic import BaseModel, ConfigDict, Field


class WallOut(BaseModel):
    """One wall, as the pair of cells it sits between."""

    cell1: tuple[int, int]
    cell2: tuple[int, int]


class VisionSolveResponse(BaseModel):
    """Schema for the vision solve response body."""

    # `model_name` collides with pydantic's reserved `model_` namespace, which would
    # otherwise emit a warning on import. The field is named for what it holds.
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(..., description="The vision model that read the image.")
    prompt_variant: str = Field(..., description="Prompt paired with that model.")
    grid_size: tuple[int, int] = Field(..., description="Rows and columns read.")
    layout: list[list[str]] = Field(..., description="The grid as the model read it.")
    walls: list[WallOut] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description="Walls that were dropped, and anything else worth showing a user.",
    )
    solvable: bool = Field(
        ...,
        description=(
            "Whether the board as read has a solution. False means the reading is "
            "definitely wrong, since every real board is generated from a path."
        ),
    )
    solver_name: str
    solution_path: str | None = None
    solution_gif_b64: str | None = None
    solution_final_image_b64: str | None = None
