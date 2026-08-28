# src/core/tests/vl_models/test_request_log.py
"""The log is only worth having if it survives bad days and stays scoreable.

Two properties carry the whole design, and each has a test that fails loudly if it
breaks: a failed write must never propagate to the caller (the answer matters more than
the record of it), and the record must keep the field names ``score_predictions`` reads,
so that adding a hand-written ``label`` is all it takes to turn real usage into an
evaluation sample.
"""

import json

import pytest

from src.core.vl_models.request_log import (
    IMAGES_DIRNAME,
    METADATA_FILENAME,
    log_request,
)
from src.core.vl_models.score_predictions import CARRIED_FIELDS, score_record

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"payload"
RAW_OUTPUT = '{"layout": [["01", "02"]], "walls": []}'


def _lines(log_dir):
    return [
        json.loads(line)
        for line in (log_dir / METADATA_FILENAME).read_text("utf-8").splitlines()
    ]


class TestLogRequest:
    def test_writes_the_image_and_one_metadata_line(self, tmp_path):
        image_path = log_request(tmp_path, PNG_BYTES, ".png", RAW_OUTPUT, usable=True)

        assert image_path is not None
        assert image_path.read_bytes() == PNG_BYTES
        (record,) = _lines(tmp_path)
        assert record["raw_output"] == RAW_OUTPUT
        assert record["usable"] is True

    def test_file_name_points_at_the_image_the_way_a_dataset_does(self, tmp_path):
        """`score_predictions` and `run_holdout` both resolve `file_name` this way."""
        image_path = log_request(tmp_path, PNG_BYTES, ".png", RAW_OUTPUT)

        (record,) = _lines(tmp_path)
        assert record["file_name"].startswith(f"{IMAGES_DIRNAME}/")
        assert (tmp_path / record["file_name"]) == image_path

    def test_appends_rather_than_overwriting(self, tmp_path):
        log_request(tmp_path, PNG_BYTES, ".png", RAW_OUTPUT)
        log_request(tmp_path, PNG_BYTES + b"x", ".png", RAW_OUTPUT)

        assert len(_lines(tmp_path)) == 2

    def test_keeps_two_uploads_in_the_same_second_apart(self, tmp_path):
        """The timestamp alone collides; the digest is what separates them."""
        first = log_request(tmp_path, PNG_BYTES, ".png", RAW_OUTPUT)
        second = log_request(tmp_path, PNG_BYTES + b"different", ".png", RAW_OUTPUT)

        assert first != second

    def test_records_the_full_digest_so_duplicates_stay_detectable(self, tmp_path):
        log_request(tmp_path, PNG_BYTES, ".png", RAW_OUTPUT)
        log_request(tmp_path, PNG_BYTES, ".png", RAW_OUTPUT)

        first, second = _lines(tmp_path)
        assert first["image_sha256"] == second["image_sha256"]
        assert len(first["image_sha256"]) == 64

    def test_writes_lf_so_the_artifact_does_not_depend_on_the_platform(self, tmp_path):
        log_request(tmp_path, PNG_BYTES, ".png", RAW_OUTPUT)

        assert b"\r\n" not in (tmp_path / METADATA_FILENAME).read_bytes()

    def test_a_failed_write_returns_none_instead_of_raising(
        self, tmp_path, monkeypatch
    ):
        """A full disk must not turn a correct answer into a 500."""

        def explode(*args, **kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr("pathlib.Path.write_bytes", explode)

        assert log_request(tmp_path, PNG_BYTES, ".png", RAW_OUTPUT) is None


class TestStaysScoreable:
    def test_a_logged_line_plus_a_hand_written_label_scores(self, tmp_path):
        """This is the whole point of matching the dataset's field names."""
        log_request(
            tmp_path,
            PNG_BYTES,
            ".png",
            RAW_OUTPUT,
            generation_seconds=1.5,
        )
        (record,) = _lines(tmp_path)
        record["label"] = RAW_OUTPUT  # what a human would fill in

        scored = score_record(record, solve=False)

        assert scored["json_parsed"] is True
        assert scored["exact_match"] is True

    @pytest.mark.parametrize("field", CARRIED_FIELDS)
    def test_writes_every_field_the_scorer_carries_through(self, tmp_path, field):
        log_request(tmp_path, PNG_BYTES, ".png", RAW_OUTPUT, generation_seconds=1.5)

        (record,) = _lines(tmp_path)
        assert field in record
