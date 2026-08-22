# src/core/vl_models/benchmark.py
"""Measurement harness for the vision-language puzzle parser.

Produces the metric layers the VLM track plan asks for (see
``ai-collab/plans/2026-08-15_track-vlm-parser.md``, stages P0 and P1):

1. JSON parse rate -- did the model emit a block that validates against
   ``SimplePuzzleOutput``?
2. Structural accuracy -- per-cell layout accuracy plus wall precision/recall/F1
   against the ground truth in ``src/core/tests/conftest.py``.
3. End-to-end -- feed the predicted puzzle to CP-SAT and compare the resulting
   path with the ground-truth path.
4. Cost -- wall-clock latency, Ollama's own timing breakdown, and peak GPU
   memory sampled from ``nvidia-smi`` while the request is in flight.

The few-shot prompt is imported from ``prompt_baseline`` on purpose: the baseline
has to measure the prompt that already exists, not a copy that can drift away from
it. Transports come from ``backends`` for the same reason -- this harness must
measure the code the app runs.

Usage:
    uv run python -m src.core.vl_models.benchmark --model qwen3.5:4b
    uv run python -m src.core.vl_models.benchmark --model gemma4:e4b --images puzzle_01 puzzle_04
"""

import argparse
import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image
from pydantic import ValidationError

from src.core.solvers.cp import solve_puzzle_cp
from src.core.tests.conftest import puzzles_to_test
from src.core.utils import parse_puzzle_layout
from src.core.vl_models.backends import (
    BACKEND_NATIVE,
    BACKEND_OPENAI_COMPAT,
    build_backend,
)
from src.core.vl_models.prompt_baseline import build_puzzle_prompt
from src.core.vl_models.prompt_variants import (
    PROMPT_BASELINE,
    PROMPT_CHOICES,
    PROMPT_FINETUNE,
    PROMPT_SIZED,
    build_finetune_prompt,
    build_sized_puzzle_prompt,
)
from src.core.vl_models.puzzle_parser import SimplePuzzleOutput, extract_json_block

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ILLUSTRATIONS_DIR = PROJECT_ROOT / "illustrations"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "ai-collab" / "reports" / "artifacts" / "vl-benchmark"
)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11435"
DEFAULT_NUM_CTX = 8192
DEFAULT_SEED = 20260815
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 900.0

GPU_POLL_INTERVAL_SECONDS = 0.25
NVIDIA_SMI_COMMAND = (
    "nvidia-smi",
    "--query-gpu=memory.used",
    "--format=csv,noheader,nounits",
)
SUBPROCESS_TIMEOUT_SECONDS = 15.0

# CP-SAT has no time limit inside solve_puzzle_cp, so refuse oversized
# hallucinated grids rather than risk an unbounded solve.
END_TO_END_MAX_CELLS = 100

EMPTY_CELL = ""
BLOCKED_CELL = "xx"

CLIENT_NATIVE = BACKEND_NATIVE
CLIENT_OPENAI_COMPAT = BACKEND_OPENAI_COMPAT
# `pydantic-ai` was the name this transport shipped under before the transports moved
# into backends.py. Artifacts in ai-collab/reports/artifacts/ still carry it, so the
# flag keeps accepting it and normalises to the canonical name on the way in.
CLIENT_LEGACY_PYDANTIC_AI = "pydantic-ai"
CLIENT_ALIASES = {CLIENT_LEGACY_PYDANTIC_AI: CLIENT_OPENAI_COMPAT}
CLIENT_CHOICES = (CLIENT_NATIVE, CLIENT_OPENAI_COMPAT, CLIENT_LEGACY_PYDANTIC_AI)

# One entry per --prompt choice. A fine-tuned checkpoint has to be measured with the very
# instruction it was trained on, so that variant lives here too rather than being a
# special case at the call site.
PROMPT_BUILDERS = {
    PROMPT_BASELINE: build_puzzle_prompt,
    PROMPT_SIZED: build_sized_puzzle_prompt,
    PROMPT_FINETUNE: build_finetune_prompt,
}


# --- Ground truth ---------------------------------------------------------


def _ground_truth_by_id() -> dict[str, tuple[dict[str, Any], list[tuple[int, int]]]]:
    """Maps ``puzzle_01`` .. ``puzzle_06`` to their (puzzle, solution) pair."""
    return {
        test_id: (puzzle, solution) for puzzle, solution, test_id in puzzles_to_test
    }


# --- Normalisation --------------------------------------------------------


def normalize_cell(raw: str) -> str:
    """Collapses the model's cell spelling into ``''``, ``'xx'`` or ``'NN'``."""
    token = str(raw).strip().lower()
    if token.isdigit():
        return f"{int(token):02d}"
    if token == BLOCKED_CELL:
        return BLOCKED_CELL
    return EMPTY_CELL


def normalize_layout(layout: list[list[str]]) -> list[list[str]]:
    return [[normalize_cell(cell) for cell in row] for row in layout]


def standardize_walls(
    wall_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    return {tuple(sorted(pair)) for pair in wall_pairs}


def _is_adjacent(cell1: tuple[int, int], cell2: tuple[int, int]) -> bool:
    return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1]) == 1


# --- Metrics --------------------------------------------------------------


def score_layout(predicted: list[list[str]], truth: list[list[str]]) -> dict[str, Any]:
    """Per-cell accuracy, plus a separate recall for the numbered waypoints.

    Waypoints are what the solver actually needs, and they are a tiny minority
    of the cells -- a model that predicts an all-empty grid would still score
    high on raw cell accuracy, so the two are reported side by side.
    """
    predicted_norm = normalize_layout(predicted)
    truth_norm = normalize_layout(truth)
    truth_shape = (len(truth_norm), len(truth_norm[0]))
    predicted_shape = (
        len(predicted_norm),
        len(predicted_norm[0]) if predicted_norm else 0,
    )
    shape_match = predicted_shape == truth_shape

    if not shape_match:
        return {
            "shape_match": False,
            "predicted_shape": list(predicted_shape),
            "truth_shape": list(truth_shape),
            "cell_accuracy": 0.0,
            "waypoint_recall": 0.0,
            "waypoint_total": sum(
                1 for row in truth_norm for cell in row if cell.isdigit()
            ),
        }

    total_cells = truth_shape[0] * truth_shape[1]
    correct_cells = 0
    waypoint_total = 0
    waypoint_correct = 0
    for row_index, truth_row in enumerate(truth_norm):
        for col_index, truth_cell in enumerate(truth_row):
            predicted_cell = predicted_norm[row_index][col_index]
            if predicted_cell == truth_cell:
                correct_cells += 1
            if truth_cell.isdigit():
                waypoint_total += 1
                if predicted_cell == truth_cell:
                    waypoint_correct += 1

    return {
        "shape_match": True,
        "predicted_shape": list(predicted_shape),
        "truth_shape": list(truth_shape),
        "cell_accuracy": correct_cells / total_cells,
        "waypoint_recall": (waypoint_correct / waypoint_total)
        if waypoint_total
        else 1.0,
        "waypoint_total": waypoint_total,
    }


def score_walls(
    predicted: set[tuple[tuple[int, int], tuple[int, int]]],
    truth: set[tuple[tuple[int, int], tuple[int, int]]],
) -> dict[str, Any]:
    """Precision/recall/F1 over the wall set, the metric most likely to be bad."""
    true_positives = len(predicted & truth)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(truth) if truth else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    if not truth and not predicted:
        precision, recall, f1 = 1.0, 1.0, 1.0
    return {
        "predicted_count": len(predicted),
        "truth_count": len(truth),
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score_end_to_end(
    predicted_puzzle: dict[str, Any], truth_solution: list[tuple[int, int]]
) -> dict[str, Any]:
    """Runs CP-SAT on the predicted puzzle and compares with the truth path."""
    height, width = predicted_puzzle["grid_size"]
    if height * width > END_TO_END_MAX_CELLS:
        return {"attempted": False, "reason": "grid_too_large", "solved": False}

    started = time.perf_counter()
    try:
        path = solve_puzzle_cp(predicted_puzzle)
    # A malformed prediction can violate assumptions solve_puzzle_cp does not check.
    except Exception as error:
        logger.warning(f"CP-SAT raised on predicted puzzle: {error}")
        return {"attempted": True, "solved": False, "error": str(error)}
    elapsed = time.perf_counter() - started

    return {
        "attempted": True,
        "solved": path is not None,
        "path_matches_truth": path == truth_solution,
        "solve_seconds": elapsed,
    }


# --- GPU sampling ---------------------------------------------------------


def query_gpu_memory_mib() -> int | None:
    """Total GPU memory in use right now, or None when nvidia-smi is unavailable."""
    try:
        completed = subprocess.run(
            NVIDIA_SMI_COMMAND,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = (
        completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    )
    return int(first_line) if first_line.isdigit() else None


class GpuMemoryMonitor:
    """Samples total GPU memory in a background thread and keeps the peak.

    The number is whole-device usage, not per-process: it is only meaningful as
    ``peak - idle_baseline`` on an otherwise quiet GPU.
    """

    def __init__(
        self, poll_interval_seconds: float = GPU_POLL_INTERVAL_SECONDS
    ) -> None:
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_mib: int | None = None
        self.peak_mib: int | None = None

    def __enter__(self) -> "GpuMemoryMonitor":
        self.baseline_mib = query_gpu_memory_mib()
        self.peak_mib = self.baseline_mib
        if self.baseline_mib is None:
            logger.warning(
                "nvidia-smi unavailable; GPU memory will be reported as null"
            )
            return self
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=SUBPROCESS_TIMEOUT_SECONDS)

    def _poll(self) -> None:
        while not self._stop_event.wait(self._poll_interval_seconds):
            sample = query_gpu_memory_mib()
            if sample is not None and (self.peak_mib is None or sample > self.peak_mib):
                self.peak_mib = sample

    @property
    def delta_mib(self) -> int | None:
        if self.peak_mib is None or self.baseline_mib is None:
            return None
        return self.peak_mib - self.baseline_mib


def ollama_ps(container_name: str) -> str:
    """``ollama ps`` output, which attributes VRAM to the loaded model."""
    try:
        completed = subprocess.run(
            ("docker", "exec", container_name, "ollama", "ps"),
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"
    return completed.stdout.strip()


# --- Model call -----------------------------------------------------------


@dataclass
class ModelCall:
    """One request to Ollama, with everything needed to reproduce it."""

    model: str
    image: str
    wall_seconds: float
    client: str = CLIENT_NATIVE
    raw_output: str = ""
    error: str | None = None
    load_seconds: float | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_seconds: float | None = None
    gpu_baseline_mib: int | None = None
    gpu_peak_mib: int | None = None
    gpu_delta_mib: int | None = None
    thinking_characters: int = 0
    ollama_ps: str = ""


def call_model(
    model: str,
    image_path: Path,
    prompt: str,
    base_url: str,
    num_ctx: int,
    seed: int,
    temperature: float,
    timeout_seconds: float,
    container_name: str,
    think: bool | None = None,
    client: str = CLIENT_NATIVE,
) -> ModelCall:
    """One request to the model, timed and instrumented.

    ``client`` picks the transport, built by ``backends.build_backend`` so this
    harness and the shipped parser exercise the same code: ``native`` talks to
    Ollama's own ``/api/chat`` (the only one that reports load/eval timings, token
    counts and a separate ``thinking`` field), ``openai-compat`` goes through the
    ``/v1`` endpoint -- fewer counters, but it is the path the app ships.

    ``think`` is left unset by default so each model keeps its own default;
    reasoning models spend both latency and context on it, so whether it
    happened is recorded rather than assumed. Both transports now honour it --
    before 2026-08-22 only ``native`` did, which made ``--no-think`` a silent
    no-op on the other one and cost it 16x in latency.
    """
    backend = build_backend(
        name=CLIENT_ALIASES.get(client, client),
        model=model,
        base_url=base_url,
        think=think,
        num_ctx=num_ctx,
        seed=seed,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )

    with GpuMemoryMonitor() as monitor:
        started = time.perf_counter()
        try:
            response = backend.generate(image_path, prompt)
            fields = {
                "raw_output": response.text,
                "thinking_characters": response.thinking_characters,
                "load_seconds": response.load_seconds,
                "prompt_eval_count": response.prompt_tokens,
                "eval_count": response.output_tokens,
                "eval_seconds": response.eval_seconds,
            }
        except Exception as error:
            return ModelCall(
                model=model,
                image=image_path.name,
                client=client,
                wall_seconds=time.perf_counter() - started,
                error=f"{type(error).__name__}: {error}",
                gpu_baseline_mib=monitor.baseline_mib,
                gpu_peak_mib=monitor.peak_mib,
                gpu_delta_mib=monitor.delta_mib,
            )
        wall_seconds = time.perf_counter() - started
        loaded_ps = ollama_ps(container_name)

    return ModelCall(
        model=model,
        image=image_path.name,
        client=client,
        wall_seconds=wall_seconds,
        gpu_baseline_mib=monitor.baseline_mib,
        gpu_peak_mib=monitor.peak_mib,
        gpu_delta_mib=monitor.delta_mib,
        ollama_ps=loaded_ps,
        **fields,
    )


# --- Evaluation -----------------------------------------------------------


def evaluate_call(call: ModelCall, puzzle_id: str) -> dict[str, Any]:
    """Turns one raw model response into the four metric layers."""
    result: dict[str, Any] = {"puzzle_id": puzzle_id, "call": asdict(call)}
    if call.error is not None:
        result["json_parsed"] = False
        result["parse_error"] = call.error
        return result

    json_block = extract_json_block(call.raw_output)
    if json_block is None:
        result["json_parsed"] = False
        result["parse_error"] = "no JSON block in response"
        return result

    try:
        validated = SimplePuzzleOutput(**json.loads(json_block))
    except (json.JSONDecodeError, ValidationError, TypeError) as error:
        result["json_parsed"] = False
        result["parse_error"] = f"{type(error).__name__}: {error}"
        return result

    result["json_parsed"] = True

    truth_puzzle, truth_solution = _ground_truth_by_id()[puzzle_id]
    result["layout"] = score_layout(validated.layout, truth_puzzle["puzzle_layout"])

    raw_wall_pairs = [
        (tuple(pair.cell1), tuple(pair.cell2)) for pair in validated.walls or []
    ]
    valid_wall_pairs = [pair for pair in raw_wall_pairs if _is_adjacent(*pair)]
    result["walls"] = score_walls(
        standardize_walls(valid_wall_pairs), truth_puzzle["walls"]
    )
    result["walls"]["non_adjacent_dropped"] = len(raw_wall_pairs) - len(
        valid_wall_pairs
    )

    predicted_puzzle = parse_puzzle_layout(validated.layout)
    predicted_puzzle["walls"] = standardize_walls(valid_wall_pairs)
    result["end_to_end"] = score_end_to_end(predicted_puzzle, truth_solution)
    return result


@dataclass
class BenchmarkRun:
    """Everything needed to reproduce and read one benchmark invocation."""

    model: str
    seed: int
    temperature: float
    num_ctx: int
    base_url: str
    started_at: str
    prompt_characters: int
    client: str = CLIENT_NATIVE
    prompt_variant: str = PROMPT_BASELINE
    think: bool | None = None
    images: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)


def run_benchmark(
    model: str,
    puzzle_ids: list[str],
    base_url: str,
    num_ctx: int,
    seed: int,
    temperature: float,
    timeout_seconds: float,
    container_name: str,
    think: bool | None = None,
    client: str = CLIENT_NATIVE,
    prompt_variant: str = PROMPT_BASELINE,
) -> BenchmarkRun:
    client = CLIENT_ALIASES.get(client, client)
    prompt = PROMPT_BUILDERS[prompt_variant]()
    run = BenchmarkRun(
        model=model,
        seed=seed,
        temperature=temperature,
        num_ctx=num_ctx,
        base_url=base_url,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        prompt_characters=len(prompt),
        client=client,
        prompt_variant=prompt_variant,
        think=think,
    )

    for puzzle_id in puzzle_ids:
        image_path = ILLUSTRATIONS_DIR / f"{puzzle_id}.png"
        with Image.open(image_path) as image:
            run.images.append(
                {
                    "puzzle_id": puzzle_id,
                    "path": image_path.name,
                    "size": list(image.size),
                }
            )
        logger.info(f"[{model}] {puzzle_id}: requesting ...")
        call = call_model(
            model=model,
            image_path=image_path,
            prompt=prompt,
            base_url=base_url,
            num_ctx=num_ctx,
            seed=seed,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            container_name=container_name,
            think=think,
            client=client,
        )
        result = evaluate_call(call, puzzle_id)
        run.results.append(result)
        logger.info(
            f"[{model}] {puzzle_id}: {call.wall_seconds:.1f}s "
            f"json={result['json_parsed']} "
            f"peak_gpu={call.gpu_peak_mib} MiB"
        )

    return run


def summarize(run: BenchmarkRun) -> dict[str, Any]:
    """Aggregates the per-image results into the numbers the report needs."""
    total = len(run.results)
    parsed = [result for result in run.results if result["json_parsed"]]
    latencies = [result["call"]["wall_seconds"] for result in run.results]
    peaks = [
        result["call"]["gpu_peak_mib"]
        for result in run.results
        if result["call"]["gpu_peak_mib"] is not None
    ]
    return {
        "model": run.model,
        "client": run.client,
        "prompt_variant": run.prompt_variant,
        "images": total,
        "json_parse_rate": len(parsed) / total if total else 0.0,
        "mean_latency_seconds": sum(latencies) / total if total else 0.0,
        "max_latency_seconds": max(latencies) if latencies else 0.0,
        "peak_gpu_mib": max(peaks) if peaks else None,
        "mean_cell_accuracy": _mean(
            [result["layout"]["cell_accuracy"] for result in parsed]
        ),
        "mean_waypoint_recall": _mean(
            [result["layout"]["waypoint_recall"] for result in parsed]
        ),
        "mean_wall_f1": _mean([result["walls"]["f1"] for result in parsed]),
        # Wall-free puzzles score a free 1.0, which flatters the mean; the
        # walled subset is the number that actually tracks the hard part.
        "mean_wall_f1_walled_only": _mean(
            [
                result["walls"]["f1"]
                for result in parsed
                if result["walls"]["truth_count"] > 0
            ]
        ),
        "walled_puzzles": sum(
            1 for result in parsed if result["walls"]["truth_count"] > 0
        ),
        "shape_correct": sum(1 for result in parsed if result["layout"]["shape_match"]),
        "end_to_end_solved": sum(
            1 for result in parsed if result["end_to_end"].get("solved")
        ),
        "end_to_end_matches_truth": sum(
            1 for result in parsed if result["end_to_end"].get("path_matches_truth")
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Ollama model tag, e.g. qwen3.5:4b"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=[f"puzzle_0{index}" for index in range(1, 7)],
        help="Puzzle ids to run, matching illustrations/<id>.png",
    )
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--container", default="zip_ollama_server")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--client",
        choices=CLIENT_CHOICES,
        default=CLIENT_NATIVE,
        help="native = Ollama /api/chat (full timing counters); "
        "openai-compat = the /v1 path the shipped parser uses "
        "('pydantic-ai' is accepted as the legacy name for it)",
    )
    parser.add_argument(
        "--prompt",
        choices=PROMPT_CHOICES,
        default=PROMPT_BASELINE,
        help="baseline = the frozen prompt the published baseline was measured on; "
        "sized = baseline plus a grid-sizing step and a 7x7 example",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--think",
        dest="think",
        action="store_true",
        default=None,
        help="force reasoning on (models that support it)",
    )
    thinking.add_argument(
        "--no-think",
        dest="think",
        action="store_false",
        help="force reasoning off; unset leaves the model default",
    )
    args = parser.parse_args()

    run = run_benchmark(
        model=args.model,
        puzzle_ids=args.images,
        base_url=args.base_url,
        num_ctx=args.num_ctx,
        seed=args.seed,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        container_name=args.container,
        think=args.think,
        client=args.client,
        prompt_variant=args.prompt,
    )
    summary = summarize(run)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace(":", "_").replace("/", "_")
    out_path = (
        args.out_dir
        / f"{time.strftime('%Y%m%d-%H%M%S')}_{model_slug}_{run.client}.json"
    )
    out_path.write_text(
        json.dumps({"run": asdict(run), "summary": summary}, indent=2),
        encoding="utf-8",
    )

    logger.info(f"summary: {json.dumps(summary, indent=2)}")
    logger.info(f"artifact written to {out_path}")


if __name__ == "__main__":
    main()
