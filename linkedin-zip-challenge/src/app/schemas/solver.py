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


class SolverResponse(BaseModel):
    """Schema for the puzzle solver response body."""

    solution_path: str
    solution_gif_b64: str | None = None
    solution_final_image_b64: str | None = None
