# src/core/vl_models/request_log.py
"""Keeps a copy of every image the vision endpoint reads, and what the model said.

The folder it writes is **deliberately the same shape as a training dataset** --
``images/`` plus a ``metadata.jsonl`` whose fields are named exactly as
``dataset_builder`` names them:

    logs/vision/
        images/20260829-041500_a1b2c3d4.png
        metadata.jsonl

That is not decoration. ``score_predictions`` needs ``label`` and ``raw_output`` and
carries ``file_name`` and ``generation_seconds``; every one of those is written here
except ``label``, which nobody can know for a picture a user just uploaded. So adding a
hand-written ``label`` to a line turns real usage into a scoreable evaluation sample at
zero extra tooling -- which is the cheapest route to the real-screenshot evaluation set
that has otherwise been deferred.

Two rules this module holds itself to:

*   **It never breaks a request.** Logging is a side effect of answering; a full disk or
    a read-only folder must not turn a good answer into a 500. Every failure is caught
    and logged.
*   **It never writes when disabled.** An empty ``vision_log_dir`` means off, because a
    user solving puzzles is not obliged to accumulate a dataset on their disk.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

IMAGES_DIRNAME = "images"
METADATA_FILENAME = "metadata.jsonl"
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"

# Enough of the digest to keep two uploads in the same second apart, short enough to
# read. The full digest goes in the record, so exact duplicates stay detectable.
SHORT_DIGEST_CHARACTERS = 8


def _record_name(image_bytes: bytes, suffix: str) -> str:
    digest = hashlib.sha256(image_bytes).hexdigest()
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    return f"{stamp}_{digest[:SHORT_DIGEST_CHARACTERS]}{suffix}"


def log_request(
    log_dir: Path,
    image_bytes: bytes,
    suffix: str,
    raw_output: str,
    **context: Any,
) -> Path | None:
    """Appends one record and returns the stored image path, or ``None`` on failure.

    ``context`` is written verbatim alongside -- model name, prompt variant, timings,
    warnings, whatever the caller thinks is worth keeping.
    """
    try:
        images_dir = log_dir / IMAGES_DIRNAME
        images_dir.mkdir(parents=True, exist_ok=True)

        name = _record_name(image_bytes, suffix)
        image_path = images_dir / name
        image_path.write_bytes(image_bytes)

        record = {
            "file_name": f"{IMAGES_DIRNAME}/{name}",
            "logged_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            **context,
            "raw_output": raw_output,
        }
        # newline="\n" so the artifact is byte-comparable with the ones produced on
        # Linux; Windows text mode would otherwise write CRLF.
        with (log_dir / METADATA_FILENAME).open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.debug("logged vision request to {}", image_path)
        return image_path
    except Exception:
        # Answering the request matters more than keeping the record of it.
        logger.exception("Could not write the vision request log to {}", log_dir)
        return None
