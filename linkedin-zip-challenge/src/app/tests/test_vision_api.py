# src/app/tests/test_vision_api.py
"""Endpoint tests for /api/vision/solve, with the model stubbed out.

No Ollama and no GPU: every case replaces the vision backend with one that returns a
fixed string, which is the whole point -- what is under test is the endpoint's handling
of what a model *can* return, and the interesting cases (unreachable model, unparseable
answer, hallucinated walls, an unsolvable reading) are the ones a live model will not
produce on demand.
"""

import asyncio
import json

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.app.main import app
from src.core.tests.conftest import puzzle_01_layout
from src.core.vl_models.backends import VisionResponse
from src.core.vl_models.puzzle_parser import VisionBackendError
from src.core.vl_models.schema import from_puzzle, to_prompt_json
from src.core.utils import parse_puzzle_layout

client = TestClient(app)

ENDPOINT = "/api/vision/solve"
LOG_METADATA = "metadata.jsonl"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def _truth_json() -> str:
    """The label for puzzle_01, in exactly the shape the model is trained to emit."""
    return to_prompt_json(from_puzzle(parse_puzzle_layout(puzzle_01_layout)))


class _StubBackend:
    """Stands in for the transport; `name` and `model` are read on the error path."""

    name = "stub"
    model = "stub-model"

    def __init__(self, text: str):
        self._text = text

    def generate(self, image_path, prompt) -> VisionResponse:
        # The real transport is pydantic-ai's `run_sync`, which raises "This event loop
        # is already running" when called from inside one. A stub that ignores that
        # passes happily against a handler the live model cannot use -- which is exactly
        # what happened on 2026-08-29, caught only by a real request. So the stub holds
        # the handler to the same rule.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return VisionResponse(text=self._text)
        raise AssertionError(
            "The vision backend was called from inside a running event loop. The "
            "endpoint must be a sync `def` so FastAPI runs it in a threadpool."
        )


@pytest.fixture(autouse=True)
def log_dir(tmp_path, monkeypatch):
    """Every test writes its request log into a temp folder, never the real one.

    Without this the suite quietly accumulates junk in `logs/vision/`, which is exactly
    the folder meant to become an evaluation set.
    """
    target = tmp_path / "vision-log"
    monkeypatch.setattr("src.app.routers.vision._log_dir", lambda: target)
    return target


def _logged(log_dir) -> list[dict]:
    path = log_dir / LOG_METADATA
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def _stub_model(monkeypatch, text: str) -> None:
    monkeypatch.setattr(
        "src.app.routers.vision.default_backend", lambda: _StubBackend(text)
    )


def _upload(filename: str = "board.png", content: bytes = PNG_BYTES, **form):
    return client.post(
        ENDPOINT, files={"image": (filename, content, "image/png")}, data=form
    )


class TestHappyPath:
    def test_reads_the_board_and_solves_it(self, monkeypatch):
        _stub_model(monkeypatch, _truth_json())

        response = _upload()

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["solvable"] is True
        assert data["layout"] == puzzle_01_layout
        assert data["solution_path"]
        assert data["solution_final_image_b64"]
        assert data["warnings"] == []

    def test_skips_the_animation_unless_asked(self, monkeypatch):
        """The GIF costs real time per upload, so it is opt-in."""
        _stub_model(monkeypatch, _truth_json())

        assert _upload().json()["solution_gif_b64"] is None
        assert _upload(include_gif="true").json()["solution_gif_b64"] is not None

    def test_reports_which_model_answered(self, monkeypatch):
        """A screenshot read by the un-finetuned model must not look like the tuned one."""
        _stub_model(monkeypatch, _truth_json())

        data = _upload().json()

        assert data["model_name"]
        assert data["prompt_variant"]

    def test_accepts_a_json_fenced_answer(self, monkeypatch):
        """Un-finetuned models wrap the object in a markdown block."""
        _stub_model(monkeypatch, f"Here you go:\n```json\n{_truth_json()}\n```")

        assert _upload().json()["solvable"] is True


class TestReadingsThatAreWrong:
    def test_reports_a_dropped_hallucinated_wall_instead_of_hiding_it(
        self, monkeypatch
    ):
        """Silently dropping invented walls is the failure the baseline called fatal."""
        payload = json.loads(_truth_json())
        payload["walls"].append({"cell1": [0, 0], "cell2": [5, 5]})
        _stub_model(monkeypatch, json.dumps(payload))

        data = _upload().json()

        assert any("non-adjacent" in warning for warning in data["warnings"])
        assert data["solvable"] is True

    def test_flags_an_unsolvable_reading_and_withholds_a_path(self, monkeypatch):
        """Every real board comes from a path, so unsolvable means definitely misread."""
        payload = json.loads(_truth_json())
        payload["walls"] = [
            {"cell1": [0, 0], "cell2": [0, 1]},
            {"cell1": [0, 0], "cell2": [1, 0]},
        ]
        _stub_model(monkeypatch, json.dumps(payload))

        data = _upload().json()

        assert data["solvable"] is False
        assert data["solution_path"] is None
        assert any("no solution" in warning for warning in data["warnings"])

    def test_unparseable_output_is_422_not_a_200_with_nothing_in_it(self, monkeypatch):
        _stub_model(monkeypatch, "I am sorry, I cannot see a puzzle in this image.")

        assert _upload().status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestRefusals:
    def test_an_unreachable_model_is_503(self, monkeypatch):
        """A missing model is a server-side condition, not a bad request."""

        def explode():
            raise VisionBackendError("ollama is not running")

        monkeypatch.setattr("src.app.routers.vision.default_backend", explode)

        response = _upload()

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "ollama" in response.json()["detail"]

    def test_rejects_a_file_type_the_backend_cannot_label(self, monkeypatch):
        """The media type is derived from the suffix, so an unknown one must not pass."""
        _stub_model(monkeypatch, _truth_json())

        response = _upload(filename="board.bmp")

        assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

    def test_rejects_an_empty_upload(self, monkeypatch):
        _stub_model(monkeypatch, _truth_json())

        assert _upload(content=b"").status_code == status.HTTP_400_BAD_REQUEST

    def test_rejects_an_unknown_solver(self, monkeypatch):
        _stub_model(monkeypatch, _truth_json())

        response = _upload(solver_name="Simulated Annealing")

        assert response.status_code == status.HTTP_404_NOT_FOUND


def test_the_backend_is_never_called_from_inside_the_event_loop(monkeypatch):
    """Named separately so the failure reads as the cause, not as a broken puzzle."""
    _stub_model(monkeypatch, _truth_json())

    assert _upload().status_code == status.HTTP_200_OK


class TestRequestLogging:
    def test_keeps_the_image_and_the_answer(self, monkeypatch, log_dir):
        _stub_model(monkeypatch, _truth_json())

        _upload()

        (record,) = _logged(log_dir)
        assert record["raw_output"] == _truth_json()
        assert record["usable"] is True
        assert record["solvable"] is True
        assert (log_dir / record["file_name"]).read_bytes() == PNG_BYTES

    def test_keeps_an_unusable_answer_too(self, monkeypatch, log_dir):
        """The 422 case is the one no evaluation set contains, so it is worth most."""
        _stub_model(monkeypatch, "I cannot see a puzzle in this image.")

        assert _upload().status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

        (record,) = _logged(log_dir)
        assert record["usable"] is False
        assert record["raw_output"] == "I cannot see a puzzle in this image."
        assert "parse_error" in record

    def test_records_which_model_and_prompt_produced_it(self, monkeypatch, log_dir):
        """A log mixing two models with no way to tell them apart is not evidence."""
        _stub_model(monkeypatch, _truth_json())

        _upload()

        (record,) = _logged(log_dir)
        assert record["model_name"]
        assert record["prompt_variant"]

    def test_writes_nothing_when_logging_is_switched_off(self, monkeypatch, log_dir):
        monkeypatch.setattr("src.app.routers.vision._log_dir", lambda: None)
        _stub_model(monkeypatch, _truth_json())

        assert _upload().status_code == status.HTTP_200_OK
        assert not log_dir.exists()

    def test_a_broken_log_does_not_break_the_answer(self, monkeypatch, log_dir):
        """Answering correctly matters more than recording that we did."""

        def explode(*args, **kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr("pathlib.Path.write_bytes", explode)
        _stub_model(monkeypatch, _truth_json())

        assert _upload().status_code == status.HTTP_200_OK
