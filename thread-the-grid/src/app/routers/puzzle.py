# src/app/routers/puzzle.py
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from src.app.schemas.puzzle import GenerateRequest, GenerateResponse
from src.app.schemas.vision import WallOut
from src.core.puzzle_generation.puzzle_generator import generate_puzzle

#: The generator's backtracking search gives up on a wall clock, so a single call can
#: come back empty -- about one 5x5 board in eight, measured 2026-09-19. Retrying is
#: cheaper than making every client handle that.
GENERATION_ATTEMPTS = 3

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
def generate_puzzle_api(request: GenerateRequest) -> GenerateResponse:
    """A new square board with walls and no blocked cells, guaranteed to be solvable."""
    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        result = generate_puzzle(
            m=request.size, n=request.size, has_walls=True, num_blocked_cells=0
        )
        if result:
            break
        logger.warning(
            f"Generator gave up on a {request.size}x{request.size} board "
            f"(attempt {attempt}/{GENERATION_ATTEMPTS})"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Could not generate a {request.size}x{request.size} puzzle in "
                f"{GENERATION_ATTEMPTS} attempts. Please try again."
            ),
        )

    puzzle, _ = result
    return GenerateResponse(
        grid_size=puzzle["grid_size"],
        layout=puzzle["puzzle_layout"],
        walls=[
            WallOut(cell1=first, cell2=second)
            for first, second in sorted(tuple(sorted(wall)) for wall in puzzle["walls"])
        ],
    )
