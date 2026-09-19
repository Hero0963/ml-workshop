# ai-collab/reports/artifacts/wrap-up-acceptance/vision_check.py
"""Screenshot -> board -> solution, judged against the label, over HTTP.

For every image of a dataset built by `src.core.vl_models.dataset_builder` (images/ plus
metadata.jsonl), posts the image to `/api/vision/solve` and checks three things apart:
the layout read cell for cell, the walls read as a set, and whether the returned path
solves the *labelled* board -- the only end-to-end claim that matters, since a misread
board can still be solvable.

    docker exec -w /app -e PYTHONPATH=/app <app container> \
        uv run python /tmp/vision_check.py /tmp/<dataset dir> --out /tmp/vision.json
"""

import argparse
import ast
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from src.core.solvers.verify import is_solution
from src.core.utils import parse_puzzle_layout

DEFAULT_BASE_URL = "http://127.0.0.1:7440"
REQUEST_TIMEOUT_SECONDS = 600
SOLVER = "CP-SAT"


def _post_image(base_url: str, image: Path) -> tuple[int, dict, float]:
    boundary = uuid.uuid4().hex
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{image.name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + image.read_bytes()
        + (
            f"\r\n--{boundary}\r\n"
            'Content-Disposition: form-data; name="solver_name"\r\n\r\n'
            f"{SOLVER}\r\n--{boundary}--\r\n"
        ).encode()
    )
    request = urllib.request.Request(
        base_url + "/api/vision/solve",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as r:
            status, payload = r.status, json.loads(r.read())
    except urllib.error.HTTPError as error:
        status, payload = error.code, json.loads(error.read() or b"{}")
    return status, payload, time.perf_counter() - started


def _wall_set(walls: list[dict]) -> set[tuple]:
    return {tuple(sorted((tuple(w["cell1"]), tuple(w["cell2"])))) for w in walls}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for line in (
        (args.dataset / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        record = json.loads(line)
        label = json.loads(record["label"])
        status, payload, seconds = _post_image(
            args.base_url, args.dataset / record["file_name"]
        )
        truth = parse_puzzle_layout(label["layout"])
        truth["walls"] = _wall_set(label["walls"])
        path_text = payload.get("solution_path") or ""
        path = (
            [ast.literal_eval(step) for step in path_text.split(" -> ")]
            if status == 200 and "->" in path_text
            else None
        )
        rows.append(
            {
                "image": record["file_name"],
                "label_walls": len(label["walls"]),
                "status": status,
                "seconds": round(seconds, 1),
                "layout_exact": payload.get("layout") == label["layout"],
                "walls_exact": _wall_set(payload.get("walls", [])) == truth["walls"],
                "solvable": payload.get("solvable"),
                "warnings": payload.get("warnings"),
                "solves_labelled_board": is_solution(truth, path),
                "model_name": payload.get("model_name"),
                "detail": payload.get("detail"),
            }
        )
        row = rows[-1]
        print(
            f"{row['status']} {row['seconds']:6.1f}s {row['image']:20} walls={row['label_walls']:2} "
            f"layout={row['layout_exact']} walls_ok={row['walls_exact']} "
            f"solves_label={row['solves_labelled_board']} {row['detail'] or ''}"
        )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
