# src/core/vl_models/dataset_builder.py
"""Builds the synthetic training set: generate -> render -> label -> verify.

Usage:
    uv run python -m src.core.vl_models.dataset_builder --count 8000 --name main_6x6
    uv run python -m src.core.vl_models.dataset_builder --count 20 --name preview --no-verify

Four things this does differently from the 2025-10 builder
(``src/core/puzzle_generation/generate_cod_dataset.py``), each because the old dataset
turned out to be unusable for training:

*   **The label is the inference schema.** It writes ``SimplePuzzleOutput`` JSON, the
    exact text the model is asked to produce, so fine-tuning has no adapter step. The
    old one wrote a Chain-of-Draft structure that had to be converted.
*   **Wall counts are ours, not the generator's default.** The shared generator hard-codes
    2-5 walls (``puzzle_generator.py:10-11,123``) with no override. Measured on the six
    real screenshots, 6x6 boards carry 0, 4, 4 and 10 -- and **two of the six have none at
    all**. So this module asks for a puzzle *without* walls, recomputes the safe edges
    itself and samples its own count. ``src/core/puzzle_generation/`` is shared with the
    RL track and is not modified.
*   **Every random choice is seeded and recorded.** Each item draws its recipe from
    ``random.Random(seed + index)``, and the recipe is written into ``metadata.jsonl``,
    so any individual sample can be audited or re-rendered.
*   **Labels are verified by CP-SAT.** Adding only safe walls cannot make a puzzle
    unsolvable in theory, but the plan's done condition asks for it to be checked rather
    than argued.

⚠ **Re-running this command does not reproduce the dataset bit-for-bit, and the manifest
does not claim it does.** ``generate_puzzle`` aborts its randomized backtracking on
*wall-clock* time, so whether a given search is cut short depends on machine load, and a
clipped attempt consumes a different amount of the shared ``random`` stream than a
completed one. Measured 2026-08-22: two runs of the same seed differed on **8 of 30**
samples at a 0.5s budget and still **3 of 30** at 5s, because 6x6 search times are
heavy-tailed (median 0.07s, max 5.4s). Fixing it properly means bounding the search on
*work* instead of time, which lives in ``src/core/puzzle_generation/`` -- shared with the
RL track and read-only here.

So the dataset, not the command, is the unit of reproducibility: the manifest carries
SHA-256 digests of ``metadata.jsonl`` and of the image bytes, which is what lets you
confirm that the copy on Colab is the copy you inspected locally. **Build once, upload
the result; never regenerate it on another machine and assume it matches.**
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger
from PIL import Image

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.solvers.cp import solve_puzzle_cp
from src.core.utils import Puzzle
from src.core.vl_models.render_puzzle import (
    DEFAULT_CELL_SIZE,
    THEMES,
    RenderTheme,
    render_puzzle,
)
from src.core.vl_models.schema import from_puzzle, to_prompt_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "vl"
IMAGES_DIRNAME = "images"
METADATA_FILENAME = "metadata.jsonl"
MANIFEST_FILENAME = "manifest.json"

DEFAULT_SEED = 20260822
DEFAULT_COUNT = 8000
DEFAULT_SIZES = (6,)

# The generator's randomized backtracking has a heavy tail. Measured 2026-08-22 over 30
# seeded 6x6 searches: median 0.07s, but 8 of 30 over 0.5s and a maximum of 5.4s. A short
# budget is therefore a large speed win -- abandoning a slow search and retrying with a
# fresh seed beats waiting for it -- and it is why raising this to 5s made a 30-sample
# build roughly 3x slower.
#
# It is also why this builder does not promise bit-identical regeneration; see the
# module docstring.
GENERATOR_TIMEOUT_SECONDS = 0.5
MAX_GENERATION_ATTEMPTS = 12

# Real 6x6 screenshots carry 0-10 walls. The upper bound has headroom; the lower bound is
# zero on purpose, because a hallucinated wall breaks a puzzle exactly as badly as a
# missed one and a model that never sees an empty board has no reason to learn restraint.
DEFAULT_MIN_WALLS = 0
DEFAULT_MAX_WALLS = 12

CELL_SIZE_CHOICES = (72, 86, 100, 116, 132)
BUTTON_PROBABILITY = 0.65
CURSOR_PROBABILITY = 0.2
DARK_THEME_PROBABILITY = 0.25
ROTATION_PROBABILITY = 0.3
ROTATION_DEGREES = 2.0
JPEG_PROBABILITY = 0.35
JPEG_QUALITY_RANGE = (60, 95)

PNG_SUFFIX = ".png"
JPEG_SUFFIX = ".jpg"


@dataclass(frozen=True)
class ItemRecipe:
    """Every random choice for one sample, so a record can be reproduced or audited."""

    index: int
    seed: int
    size: int
    wall_count: int
    theme: str
    cell_size: int
    show_buttons: bool
    show_cursor: bool
    rotation_degrees: float
    jpeg_quality: int | None


def _all_possible_walls(
    height: int, width: int
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    walls: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for row in range(height):
        for column in range(width - 1):
            walls.add(tuple(sorted(((row, column), (row, column + 1)))))
    for row in range(height - 1):
        for column in range(width):
            walls.add(tuple(sorted(((row, column), (row + 1, column)))))
    return walls


def sample_walls(
    puzzle: Puzzle,
    solution_path: list[tuple[int, int]],
    wall_count: int,
    rng: random.Random,
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    """Picks ``wall_count`` walls that the known solution never crosses.

    Mirrors ``puzzle_generator.py:107-125`` deliberately rather than importing it: that
    logic is welded to the generator's own 2-5 count, and the shared module is read-only
    for this track.
    """
    height, width = puzzle["grid_size"]
    solution_edges = {
        tuple(sorted((solution_path[i], solution_path[i + 1])))
        for i in range(len(solution_path) - 1)
    }
    safe_walls = sorted(_all_possible_walls(height, width) - solution_edges)
    if wall_count > len(safe_walls):
        wall_count = len(safe_walls)
    return set(rng.sample(safe_walls, wall_count))


def draw_recipe(
    index: int, seed: int, sizes: tuple[int, ...], min_walls: int, max_walls: int
) -> ItemRecipe:
    """All randomness for one item, drawn from its own stream."""
    rng = random.Random(seed + index)
    return ItemRecipe(
        index=index,
        seed=seed + index,
        size=rng.choice(sizes),
        wall_count=rng.randint(min_walls, max_walls),
        theme="dark" if rng.random() < DARK_THEME_PROBABILITY else "light",
        cell_size=rng.choice(CELL_SIZE_CHOICES),
        show_buttons=rng.random() < BUTTON_PROBABILITY,
        show_cursor=rng.random() < CURSOR_PROBABILITY,
        rotation_degrees=(
            rng.uniform(-ROTATION_DEGREES, ROTATION_DEGREES)
            if rng.random() < ROTATION_PROBABILITY
            else 0.0
        ),
        jpeg_quality=(
            rng.randint(*JPEG_QUALITY_RANGE)
            if rng.random() < JPEG_PROBABILITY
            else None
        ),
    )


def build_puzzle(recipe: ItemRecipe) -> tuple[Puzzle, list[tuple[int, int]]] | None:
    """Generates one puzzle and gives it our own wall count.

    ``generate_puzzle`` draws from the global ``random`` module, so it is seeded here.
    It also returns ``None`` on the odd-board parity problem documented in the RL
    handover, hence the retry loop.
    """
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        random.seed(recipe.seed + attempt * MAX_GENERATION_ATTEMPTS)
        result = generate_puzzle(
            m=recipe.size,
            n=recipe.size,
            has_walls=False,
            timeout_per_attempt=GENERATOR_TIMEOUT_SECONDS,
        )
        if result is None:
            continue
        puzzle, solution_path = result
        wall_rng = random.Random(recipe.seed)
        puzzle["walls"] = sample_walls(
            puzzle, solution_path, recipe.wall_count, wall_rng
        )
        return puzzle, solution_path
    return None


def render_with_recipe(puzzle: Puzzle, recipe: ItemRecipe) -> Image.Image:
    theme: RenderTheme = THEMES[recipe.theme]
    image = render_puzzle(
        puzzle,
        cell_size=recipe.cell_size,
        theme=theme,
        show_buttons=recipe.show_buttons,
        show_cursor=recipe.show_cursor,
    )
    if recipe.rotation_degrees:
        image = image.rotate(
            recipe.rotation_degrees,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=theme.background,
        )
    return image


def _save(image: Image.Image, images_dir: Path, recipe: ItemRecipe) -> str:
    stem = f"{recipe.index:06d}"
    if recipe.jpeg_quality is None:
        path = images_dir / f"{stem}{PNG_SUFFIX}"
        image.save(path, "PNG")
    else:
        path = images_dir / f"{stem}{JPEG_SUFFIX}"
        image.save(path, "JPEG", quality=recipe.jpeg_quality)
    return f"{IMAGES_DIRNAME}/{path.name}"


PROGRESS_EVERY = 500


def _digest_images(dataset_dir: Path, records: list[dict[str, object]]) -> str:
    """One digest over every image, in metadata order.

    This is what makes a copy verifiable: run ``verify_dataset`` on Colab after the
    upload and compare, instead of regenerating and hoping.
    """
    digest = hashlib.sha256()
    for record in records:
        digest.update((dataset_dir / str(record["file_name"])).read_bytes())
    return digest.hexdigest()


def verify_dataset(dataset_dir: Path) -> bool:
    """Recomputes both digests and checks them against the manifest."""
    manifest = json.loads((dataset_dir / MANIFEST_FILENAME).read_text("utf-8"))
    metadata_path = dataset_dir / METADATA_FILENAME
    records = [
        json.loads(line) for line in metadata_path.read_text("utf-8").splitlines()
    ]
    metadata_ok = (
        hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        == manifest["metadata_sha256"]
    )
    images_ok = _digest_images(dataset_dir, records) == manifest["images_sha256"]
    logger.info("metadata digest {}", "ok" if metadata_ok else "MISMATCH")
    logger.info("images digest   {}", "ok" if images_ok else "MISMATCH")
    return metadata_ok and images_ok


def _drain(
    results: Iterable[dict[str, object] | None], count: int
) -> Iterator[dict[str, object] | None]:
    """Yields results in order, logging progress as they land."""
    for position, record in enumerate(results, start=1):
        if position % PROGRESS_EVERY == 0:
            logger.info("{}/{} done", position, count)
        yield record


def build_item(task: tuple[ItemRecipe, Path, bool]) -> dict[str, object] | None:
    """One sample end to end: generate, verify, render, label.

    Returns ``None`` when the generator gave up (odd-board parity) and a record with
    ``"unsolvable": True`` when CP-SAT rejected the labels, so the caller can count both
    rather than have them disappear silently.
    """
    recipe, images_dir, verify = task
    # Every worker is a fresh process; generate_puzzle logs a line per attempt.
    logger.disable("src.core.puzzle_generation.puzzle_generator")

    built = build_puzzle(recipe)
    if built is None:
        return None
    puzzle, _ = built

    if verify and not solve_puzzle_cp(puzzle):
        return {"index": recipe.index, "unsolvable": True}

    image = render_with_recipe(puzzle, recipe)
    file_name = _save(image, images_dir, recipe)
    return {
        "file_name": file_name,
        "label": to_prompt_json(from_puzzle(puzzle)),
        "grid_size": recipe.size,
        "wall_count": len(puzzle["walls"]),
        "waypoint_count": len(puzzle["num_map"]),
        "theme": recipe.theme,
        "cell_size": recipe.cell_size,
        "show_buttons": recipe.show_buttons,
        "show_cursor": recipe.show_cursor,
        "rotation_degrees": round(recipe.rotation_degrees, 3),
        "jpeg_quality": recipe.jpeg_quality,
        "seed": recipe.seed,
    }


def build_dataset(
    count: int,
    name: str,
    seed: int = DEFAULT_SEED,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    min_walls: int = DEFAULT_MIN_WALLS,
    max_walls: int = DEFAULT_MAX_WALLS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    verify: bool = True,
) -> Path:
    """Writes an HF ``imagefolder``-style dataset and returns its directory.

    Deliberately single-process. A ``multiprocessing.Pool`` was tried on 2026-08-22 and
    **hung on Windows** -- no output at all after 10 minutes for 32 items, with or
    without the CP-SAT step, because ``spawn`` re-imports this module in every child.
    The sequential path costs a one-off wait for a build that happens rarely, which is
    cheaper than shipping a flag that deadlocks.
    """
    logger.disable("src.core.puzzle_generation.puzzle_generator")

    dataset_dir = output_root / name
    images_dir = dataset_dir / IMAGES_DIRNAME
    images_dir.mkdir(parents=True, exist_ok=True)

    wall_histogram: Counter[int] = Counter()
    size_histogram: Counter[int] = Counter()
    theme_histogram: Counter[str] = Counter()
    skipped = 0
    unsolvable: list[int] = []
    records: list[dict[str, object]] = []

    tasks = (
        (draw_recipe(index, seed, sizes, min_walls, max_walls), images_dir, verify)
        for index in range(count)
    )
    logger.info("building {} samples", count)

    for record in _drain((build_item(task) for task in tasks), count):
        if record is None:
            skipped += 1
        elif record.get("unsolvable"):
            unsolvable.append(int(record["index"]))
        else:
            records.append(record)
            wall_histogram[int(record["wall_count"])] += 1
            size_histogram[int(record["grid_size"])] += 1
            theme_histogram[str(record["theme"])] += 1

    metadata_path = dataset_dir / METADATA_FILENAME
    metadata_bytes = (
        "\n".join(json.dumps(record) for record in records) + "\n"
    ).encode("utf-8")
    metadata_path.write_bytes(metadata_bytes)

    manifest = {
        "name": name,
        "requested": count,
        "written": len(records),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "images_sha256": _digest_images(dataset_dir, records),
        "skipped_generation_failures": skipped,
        "unsolvable_rejected": unsolvable,
        "seed": seed,
        "sizes": list(sizes),
        "wall_range": [min_walls, max_walls],
        "verified_with_cp_sat": verify,
        "wall_histogram": dict(sorted(wall_histogram.items())),
        "size_histogram": dict(sorted(size_histogram.items())),
        "theme_histogram": dict(theme_histogram),
        "renderer_defaults": {
            "cell_size_choices": list(CELL_SIZE_CHOICES),
            "base_cell_size": DEFAULT_CELL_SIZE,
        },
        "example_recipe": asdict(draw_recipe(0, seed, sizes, min_walls, max_walls)),
    }
    (dataset_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.success(
        "{} samples in {} (skipped {}, unsolvable {})",
        len(records),
        dataset_dir,
        skipped,
        len(unsolvable),
    )
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--name", help="Dataset directory name")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--sizes",
        default=",".join(str(size) for size in DEFAULT_SIZES),
        help="Comma-separated square grid sizes, e.g. '6' or '5,6,7'",
    )
    parser.add_argument("--min-walls", type=int, default=DEFAULT_MIN_WALLS)
    parser.add_argument("--max-walls", type=int, default=DEFAULT_MAX_WALLS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="Verify an existing dataset directory against its manifest digests, then exit",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip the CP-SAT solvability check (faster, but the labels are unproven)",
    )
    args = parser.parse_args()

    if args.check is not None:
        raise SystemExit(0 if verify_dataset(args.check) else 1)

    build_dataset(
        count=args.count,
        name=args.name,
        seed=args.seed,
        sizes=tuple(int(size) for size in args.sizes.split(",")),
        min_walls=args.min_walls,
        max_walls=args.max_walls,
        output_root=args.out_root,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
