# src/app/schemas/solver.py
from pydantic import BaseModel, Field


class SolverRequest(BaseModel):
    """Schema for the puzzle solver request body."""

    puzzle_layout_str: str = Field(
        ..., description="The string representation of the puzzle layout list."
    )
    walls_str: str = Field(
        "set()", description="The string representation of the walls set."
    )
    solver_name: str = Field(..., description="The name of the solver to use.")


class SolverInfo(BaseModel):
    """One solver `/solve` accepts, as `src.core.solvers.registry` describes it."""

    name: str = Field(..., description="The value to send as `solver_name`.")
    kind: str = Field(
        ...,
        description=(
            "'exact' always answers or proves there is no solution; 'learned' and "
            "'heuristic' may give up, which says nothing about the board."
        ),
    )
    note: str = Field(..., description="The trade-off, in a sentence or two.")


class SolverResponse(BaseModel):
    """Schema for the puzzle solver response body."""

    solution_path: str
    solution_gif_b64: str | None = None
    solution_final_image_b64: str | None = None
