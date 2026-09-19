# ai-collab/reports/artifacts/vlm-walkthrough/vlm_demo.py
"""One screenshot through the app's own reading path, with every step printed.

Calls the same functions as POST /api/vision/solve (build_prompt, default_backend,
parse_puzzle_image, CP-SAT, is_solution). The only addition is a hook on httpx that
prints the request actually sent to Ollama, with the image's base64 shortened.

Run it inside the app container, so the settings and the route to Ollama are the app's:

    docker exec -w /app -e PYTHONPATH=/app <app container> \
        uv run python /tmp/vlm_demo.py <image> <solution.png> [reference.json]
"""

import base64
import hashlib
import json
import sys
from pathlib import Path

import httpx
from PIL import Image

from src.core.solvers.registry import SOLVERS
from src.core.solvers.verify import is_solution
from src.core.utils import save_solution_as_image
from src.core.vl_models.prompt_variants import build_prompt
from src.core.vl_models.puzzle_parser import default_backend, parse_puzzle_image
from src.core.vl_models.schema import from_puzzle
from src.settings import get_settings

SOLVER = "CP-SAT"
BASE64_SHOWN_CHARS = 60
RULE_WIDTH = 24


def rule(title: str) -> None:
    print(f"\n{'=' * RULE_WIDTH} {title} {'=' * RULE_WIDTH}", flush=True)


def shorten_images(node: object) -> object:
    """Keeps the request readable: data URLs are cut to their first characters."""
    if isinstance(node, dict):
        return {key: shorten_images(value) for key, value in node.items()}
    if isinstance(node, list):
        return [shorten_images(value) for value in node]
    if isinstance(node, str) and node.startswith("data:image/"):
        return f"{node[:BASE64_SHOWN_CHARS]}... ({len(node):,} chars in total)"
    return node


def install_request_printer() -> None:
    """Prints the HTTP request that leaves this process, whichever httpx client sends it."""

    def show(request: httpx.Request) -> None:
        if not request.url.path.endswith("/chat/completions"):
            return
        rule("3. REQUEST actually sent to Ollama")
        print(f"{request.method} {request.url}")
        body = json.loads(request.content)
        print(json.dumps(shorten_images(body), indent=2, ensure_ascii=False))

    original_async_send = httpx.AsyncClient.send
    original_send = httpx.Client.send

    async def async_send(
        self: httpx.AsyncClient, request: httpx.Request, *args: object, **kwargs: object
    ) -> httpx.Response:
        show(request)
        return await original_async_send(self, request, *args, **kwargs)

    def send(
        self: httpx.Client, request: httpx.Request, *args: object, **kwargs: object
    ) -> httpx.Response:
        show(request)
        return original_send(self, request, *args, **kwargs)

    httpx.AsyncClient.send = async_send
    httpx.Client.send = send


def as_wall_set(walls: list) -> set[frozenset[tuple[int, int]]]:
    return {frozenset((tuple(cell1), tuple(cell2))) for cell1, cell2 in walls}


def show_grid(layout: list[list[str]]) -> None:
    for row in layout:
        print(
            "  | " + " | ".join(cell if cell.strip() else " ." for cell in row) + " |"
        )


def main() -> None:
    image = Path(sys.argv[1])
    solution_png = Path(sys.argv[2])
    reference = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else None

    settings = get_settings()
    prompt = build_prompt(settings.vision_prompt_variant)
    backend = default_backend()
    image_bytes = image.read_bytes()
    with Image.open(image) as picture:
        pixels, colour_mode = picture.size, picture.mode

    rule("1. INPUT: image")
    print(f"file        {image.name}")
    print(f"bytes       {len(image_bytes):,}")
    print(f"sha256      {hashlib.sha256(image_bytes).hexdigest()}")
    print(f"pixels      {pixels[0]} x {pixels[1]} ({colour_mode})")
    print(
        f"base64      {len(base64.b64encode(image_bytes)):,} chars, sent as-is (no resizing)"
    )

    rule(f"2. INPUT: prompt (variant '{settings.vision_prompt_variant}', verbatim)")
    print(prompt)

    print(f"\nmodel       {backend.model}")
    print(
        f"transport   {backend.name} (pydantic-ai -> Ollama /v1), think={backend.think}"
    )

    install_request_printer()
    result = parse_puzzle_image(image, backend=backend, prompt=prompt)

    rule("4. OUTPUT: the model's reply (verbatim)")
    print(result.raw_output)
    print(f"\n(generation {result.generation_seconds} s)")

    rule("5. PARSED into a board")
    layout = from_puzzle(result.puzzle).layout
    show_grid(layout)
    walls = sorted(result.puzzle["walls"])
    print(f"walls ({len(walls)}):")
    for cell1, cell2 in walls:
        print(f"  {tuple(cell1)} | {tuple(cell2)}")
    print(f"parser warnings: {list(result.warnings) or 'none'}")

    if reference is not None:
        rule("5b. COMPARED with the reference reading")
        layout_ok = layout == reference["layout"]
        read_walls = as_wall_set(walls)
        expected_walls = as_wall_set(reference["walls"])
        print(f"layout cell for cell: {'MATCH' if layout_ok else 'DIFFERENT'}")
        print(
            f"walls as a set:       {'MATCH' if read_walls == expected_walls else 'DIFFERENT'}"
        )
        for wall in sorted(expected_walls - read_walls, key=sorted):
            print(f"  missed   {sorted(wall)}")
        for wall in sorted(read_walls - expected_walls, key=sorted):
            print(f"  invented {sorted(wall)}")

    rule(f"6. SOLVED with {SOLVER} and checked by the independent verifier")
    path = SOLVERS[SOLVER](result.puzzle)
    print(
        f"path found: {bool(path)}  |  is_solution: {is_solution(result.puzzle, path)}"
    )
    if path:
        print(f"{len(path)} cells: " + " -> ".join(f"{tuple(cell)}" for cell in path))
        save_solution_as_image(result.puzzle, path, str(solution_png))
        print(f"drawn to {solution_png}")


if __name__ == "__main__":
    main()
