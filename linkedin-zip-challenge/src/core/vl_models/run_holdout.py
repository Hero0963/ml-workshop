# src/core/vl_models/run_holdout.py
"""Runs a model over a slice of a synthetic dataset and writes raw predictions.

Usage:
    uv run python -m src.core.vl_models.run_holdout \
        --dataset datasets/vl/main_6x6 --model zip-qwen35-4b-p4c:f16 --count 200

This is the local counterpart of section 11 of ``notebooks/p4c_finetune_8000.ipynb``.
That cell ran on Colab against the LoRA adapter in memory; this one runs against a
served model, which is the only way to check that the export in P4d preserved what the
training produced. Both write the same JSONL, so
``src.core.vl_models.score_predictions`` scores either without knowing the difference.

Two deliberate constraints:

*   **It computes no metrics.** Scoring lives in ``score_predictions`` alone, for the
    same reason the notebook does not score: a second implementation is how the
    benchmark numbers and the shipped parser drifted apart once already.
*   **The default slice is the tail.** P4c trained on ``records[:-200]`` and held out
    ``records[-200:]``. Scoring the head instead would be scoring the training set,
    which is precisely the mistake section 4 of that notebook exists to prevent.

Output is written line by line and flushed, so an interrupted run keeps what finished.
"""

import argparse
import json
import time
from pathlib import Path

from loguru import logger

from src.core.vl_models.backends import (
    BACKEND_CHOICES,
    BACKEND_OPENAI_COMPAT,
    VisionBackend,
    build_backend,
)
from src.core.vl_models.prompt_variants import (
    PROMPT_CHOICES,
    PROMPT_FINETUNE,
    build_prompt,
)
from src.settings import get_settings

METADATA_FILENAME = "metadata.jsonl"
DEFAULT_HOLDOUT_SIZE = 200
PROGRESS_EVERY = 10

SLICE_TAIL = "tail"
SLICE_HEAD = "head"
SLICE_CHOICES = (SLICE_TAIL, SLICE_HEAD)


def read_records(dataset_dir: Path) -> list[dict]:
    metadata_path = dataset_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No {METADATA_FILENAME} in {dataset_dir}.")
    return [
        json.loads(line)
        for line in metadata_path.read_text("utf-8").splitlines()
        if line.strip()
    ]


def select_slice(records: list[dict], count: int, which: str) -> list[dict]:
    """Takes the held-out tail by default; the head is training data."""
    if count > len(records):
        raise ValueError(
            f"Asked for {count} records but the dataset holds {len(records)}."
        )
    return records[-count:] if which == SLICE_TAIL else records[:count]


def run(
    records: list[dict],
    dataset_dir: Path,
    backend: VisionBackend,
    prompt: str,
    out_path: Path,
) -> float:
    """Generates one answer per record into a JSONL. Returns seconds per sample."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # The explicit newline is on purpose: Windows text mode would translate to CRLF, and
    # this file is a versioned artifact meant to be byte-comparable with the one the
    # Colab notebook writes on Linux.
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for position, record in enumerate(records, start=1):
            image_path = dataset_dir / record["file_name"]
            call_started = time.perf_counter()
            response = backend.generate(image_path, prompt)
            elapsed = time.perf_counter() - call_started
            handle.write(
                json.dumps(
                    {
                        **record,
                        "raw_output": response.text,
                        "generation_seconds": round(elapsed, 3),
                    }
                )
                + "\n"
            )
            handle.flush()
            if position % PROGRESS_EVERY == 0:
                logger.info(
                    "{}/{} ({:.1f}s each so far)",
                    position,
                    len(records),
                    (time.perf_counter() - started) / position,
                )

    return (time.perf_counter() - started) / len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Dataset directory."
    )
    parser.add_argument("--model", default=None, help="Model tag; defaults to .env.")
    parser.add_argument("--count", type=int, default=DEFAULT_HOLDOUT_SIZE)
    parser.add_argument("--slice", choices=SLICE_CHOICES, default=SLICE_TAIL)
    parser.add_argument("--prompt", choices=PROMPT_CHOICES, default=PROMPT_FINETUNE)
    parser.add_argument(
        "--client", choices=BACKEND_CHOICES, default=BACKEND_OPENAI_COMPAT
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Where to write the JSONL."
    )
    args = parser.parse_args()

    settings = get_settings()
    records = select_slice(read_records(args.dataset), args.count, args.slice)
    backend = build_backend(
        name=args.client,
        model=args.model or settings.ollama_model_name,
        base_url=settings.ollama_provider_url,
        think=False,
    )

    logger.info(
        "{} records from the {} of {} through {} on {}",
        len(records),
        args.slice,
        args.dataset,
        backend.name,
        backend.model,
    )
    seconds = run(records, args.dataset, backend, build_prompt(args.prompt), args.out)
    logger.info("{:.1f}s/sample, wrote {} to {}", seconds, len(records), args.out)


if __name__ == "__main__":
    main()
