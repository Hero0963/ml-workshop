# src/app/routers/vision.py
"""Reads a Zip puzzle out of an uploaded screenshot and solves it.

The whole pipeline in one call: image -> JSON -> ``Puzzle`` -> solver -> picture.

Failures are reported, never swallowed. ``puzzle_parser`` raises rather than returning
``None`` precisely so this endpoint cannot answer 200 with an empty body, and the two
exception types map onto two different HTTP answers because they mean different things
to a caller:

*   ``VisionBackendError`` -> **503**. The model is not reachable; retrying later is
    reasonable and nothing is wrong with the request.
*   ``ModelOutputError`` -> **422**. The model answered but not usably, which usually
    means the image is not a Zip puzzle -- retrying the same image will not help.

A successful read is still not necessarily a correct one, which is why ``solvable`` is
part of the response rather than a log line. See ``schemas/vision.py``.
"""

import base64
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from loguru import logger

from src.app.schemas.vision import VisionSolveResponse, WallOut
from src.core.solvers.registry import SOLVERS
from src.core.utils import save_detailed_animation_as_gif, save_solution_as_image
from src.core.vl_models.prompt_variants import build_prompt
from src.core.vl_models.puzzle_parser import (
    ModelOutputError,
    ParseResult,
    VisionBackendError,
    default_backend,
    parse_puzzle_image,
)
from src.core.vl_models.request_log import log_request
from src.core.vl_models.schema import from_puzzle
from src.settings import get_settings

router = APIRouter()

# `SOLVERS` comes from the shared registry, so this endpoint offers exactly what
# /api/solver/solve offers. CP-SAT stays the default because a board read out of an image
# can be nonsense, and it decides "no solution" quickly instead of exploring.
# ⚠ The RL solver is in the list but is out of distribution here: it trained on boards
# with 0 or 2-5 walls, and a real screenshot can carry ten or more. It will usually
# answer "no solution found" rather than anything wrong, which is the honest failure.
DEFAULT_SOLVER = "CP-SAT"

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _log_dir() -> Path | None:
    """Resolves the configured log folder, or ``None`` when logging is switched off."""
    configured = get_settings().vision_log_dir.strip()
    if not configured:
        return None
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _record(
    image_bytes: bytes, suffix: str, raw_output: str, **context: object
) -> None:
    """Keeps the image and the model's answer, if a log folder is configured."""
    log_dir = _log_dir()
    if log_dir is None:
        return
    settings = get_settings()
    log_request(
        log_dir,
        image_bytes,
        suffix,
        raw_output,
        model_name=settings.ollama_model_name,
        prompt_variant=settings.vision_prompt_variant,
        **context,
    )


def _suffix_for(upload: UploadFile) -> str:
    """The backend picks its media type from the suffix, so it has to be a real one."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported image type {suffix or '(none)'}. "
                f"Expected one of {sorted(SUPPORTED_SUFFIXES)}."
            ),
        )
    return suffix


def _parse_upload(image_bytes: bytes, suffix: str) -> ParseResult:
    """Writes the upload to a temp file, because the backends read from a path."""
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(handle.name)
    handle.write(image_bytes)
    handle.close()

    settings = get_settings()
    try:
        return parse_puzzle_image(
            temp_path,
            backend=default_backend(),
            prompt=build_prompt(settings.vision_prompt_variant),
        )
    finally:
        temp_path.unlink(missing_ok=True)


def _render(puzzle: dict, path: list, want_gif: bool) -> tuple[str, str | None]:
    """Returns the final PNG, and the animation only when it was asked for."""
    png_handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    png_path = Path(png_handle.name)
    png_handle.close()
    gif_path: Path | None = None

    try:
        save_solution_as_image(puzzle, path, str(png_path))
        png_b64 = base64.b64encode(png_path.read_bytes()).decode("utf-8")

        gif_b64 = None
        if want_gif:
            gif_handle = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
            gif_path = Path(gif_handle.name)
            gif_handle.close()
            save_detailed_animation_as_gif(puzzle, path, str(gif_path))
            gif_b64 = base64.b64encode(gif_path.read_bytes()).decode("utf-8")
        return png_b64, gif_b64
    finally:
        png_path.unlink(missing_ok=True)
        if gif_path is not None:
            gif_path.unlink(missing_ok=True)


# Deliberately a sync `def`, not `async def`. Every step here blocks -- the model call,
# CP-SAT, PIL rendering -- and the transport underneath is `pydantic-ai`'s `run_sync`,
# which raises "This event loop is already running" if it is called from inside one. A
# sync handler is run in FastAPI's threadpool, which both fixes that and keeps the event
# loop free while a 4B model spends seconds on an image.
@router.post("/solve", response_model=VisionSolveResponse)
def solve_from_image(
    image: UploadFile = File(..., description="A screenshot of a Zip puzzle."),
    solver_name: str = Form(DEFAULT_SOLVER),
    include_gif: bool = Form(False),
) -> VisionSolveResponse:
    """Reads the puzzle in the uploaded image and solves it."""
    solver = SOLVERS.get(solver_name)
    if solver is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Solver '{solver_name}' not found. Expected one of {sorted(SOLVERS)}.",
        )

    suffix = _suffix_for(image)
    image_bytes = image.file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image is empty.",
        )

    try:
        result = _parse_upload(image_bytes, suffix)
    except VisionBackendError as error:
        logger.exception("The vision backend could not be reached")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    except ModelOutputError as error:
        logger.warning("The model answered with something unusable: {}", error)
        # Logged on purpose: an unusable answer is the case no evaluation set holds and
        # no retry fixes, so it is the most worth keeping.
        _record(
            image_bytes,
            suffix,
            error.raw_output,
            usable=False,
            parse_error=str(error),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error

    puzzle = result.puzzle
    # Rendered back through the shared schema rather than kept from the model text, so
    # what the response shows is what the solver actually ran on.
    layout = from_puzzle(puzzle).layout
    try:
        solution_path = solver(puzzle)
    except Exception as error:
        logger.exception("Solving the board read from the image failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during solving: {error}",
        ) from error

    settings = get_settings()
    walls = [
        WallOut(cell1=cell1, cell2=cell2) for cell1, cell2 in sorted(puzzle["walls"])
    ]
    warnings = list(result.warnings)
    _record(
        image_bytes,
        suffix,
        result.raw_output,
        usable=True,
        generation_seconds=result.generation_seconds,
        grid_size=list(puzzle["grid_size"]),
        wall_count=len(walls),
        solvable=bool(solution_path),
        parser_warnings=list(result.warnings),
    )

    if not solution_path:
        warnings.append(
            "The board as read has no solution, so at least one wall or waypoint was "
            "misread -- every real Zip board is generated from a complete path."
        )
        logger.warning("Unsolvable reading from {}", image.filename)
        return VisionSolveResponse(
            model_name=settings.ollama_model_name,
            prompt_variant=settings.vision_prompt_variant,
            grid_size=puzzle["grid_size"],
            layout=layout,
            walls=walls,
            warnings=warnings,
            solvable=False,
            solver_name=solver_name,
        )

    png_b64, gif_b64 = _render(puzzle, solution_path, include_gif)
    return VisionSolveResponse(
        model_name=settings.ollama_model_name,
        prompt_variant=settings.vision_prompt_variant,
        grid_size=puzzle["grid_size"],
        layout=layout,
        walls=walls,
        warnings=warnings,
        solvable=True,
        solver_name=solver_name,
        solution_path=" -> ".join(map(str, solution_path)),
        solution_gif_b64=gif_b64,
        solution_final_image_b64=png_b64,
    )
