# src/app/routers/solver.py
import ast
import base64
import os
import tempfile
import pprint

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from src.app.schemas.solver import SolverRequest, SolverResponse
from src.core.solvers.a_star import solve_puzzle_a_star
from src.core.solvers.cp import solve_puzzle_cp
from src.core.solvers.dfs import solve_puzzle as solve_puzzle_dfs
from src.core.utils import (
    parse_puzzle_layout,
    save_detailed_animation_as_gif,
    save_solution_as_image,
)

# --- Router and Solver Mapping ---
router = APIRouter()

SOLVERS = {
    "DFS": solve_puzzle_dfs,
    "A* (heapq)": solve_puzzle_a_star,
    "CP-SAT": solve_puzzle_cp,
}


@router.post("/solve", response_model=SolverResponse)
def solve_puzzle_api(request: SolverRequest) -> SolverResponse:
    """API endpoint to solve a puzzle.

    Receives a puzzle layout and solver name, returns the solution path
    and a Base64-encoded GIF of the solution animation.
    """
    # 1. Parse and validate the puzzle layout and walls from the request string
    try:
        puzzle_layout = ast.literal_eval(request.puzzle_layout_str)
        if not isinstance(puzzle_layout, list):
            raise TypeError("Layout input is not a list.")

        walls = ast.literal_eval(request.walls_str)
        if not isinstance(walls, set):
            raise TypeError("Walls input is not a set.")

        puzzle_data = parse_puzzle_layout(puzzle_layout)
        puzzle_data["walls"] = walls  # Add the walls to the puzzle data

    except (ValueError, SyntaxError, TypeError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error parsing input: {e}. Ensure layout and walls are valid Python literals.",
        )

    # 2. Log the parsed input and get the selected solver function
    logger.debug("--- Parsed Puzzle Input for Solver ---")
    logger.debug(f"Solver: {request.solver_name}")
    logger.debug(f"Puzzle Data:\n{pprint.pformat(puzzle_data)}")
    logger.debug("------------------------------------")

    solver_func = SOLVERS.get(request.solver_name)
    if not solver_func:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Solver '{request.solver_name}' not found.",
        )

    # 3. Run the solver
    try:
        solution_path = solver_func(puzzle_data)
    except Exception as e:
        # Log the exception for more detailed server-side debugging
        logger.exception("Exception during puzzle solving")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during solving: {e}",
        )

    if not solution_path:
        return SolverResponse(
            solution_path=f"Solver '{request.solver_name}' could not find a solution.",
            solution_gif_b64=None,
            solution_final_image_b64=None,
        )

    # 4. Generate the GIF and encode it in Base64
    fp_gif = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    gif_path = fp_gif.name
    fp_gif.close()

    # 5. Generate the final static image and encode it
    fp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img_path = fp_img.name
    fp_img.close()

    try:
        # Generate GIF
        save_detailed_animation_as_gif(puzzle_data, solution_path, gif_path)
        with open(gif_path, "rb") as f:
            gif_binary_data = f.read()

        # Generate static image
        save_solution_as_image(puzzle_data, solution_path, img_path)
        with open(img_path, "rb") as f:
            img_binary_data = f.read()

    except Exception as e:
        logger.exception("Failed to generate solution images")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate solution image: {e}",
        )
    finally:
        # Ensure the temporary files are always deleted
        if os.path.exists(gif_path):
            os.remove(gif_path)
            logger.debug(f"Deleted temporary file: {gif_path}")
        if os.path.exists(img_path):
            os.remove(img_path)
            logger.debug(f"Deleted temporary file: {img_path}")

    gif_b64 = base64.b64encode(gif_binary_data).decode("utf-8")
    img_b64 = base64.b64encode(img_binary_data).decode("utf-8")
    solution_str = " -> ".join(map(str, solution_path))

    return SolverResponse(
        solution_path=solution_str,
        solution_gif_b64=gif_b64,
        solution_final_image_b64=img_b64,
    )
