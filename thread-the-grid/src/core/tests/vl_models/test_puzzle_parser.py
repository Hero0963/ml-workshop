# src/core/tests/vl_models/test_puzzle_parser.py
"""Covers the production parser with a fake backend -- no Ollama, no GPU.

The plan's done condition for P5 is explicit about this: the VL call has to be
mockable so the suite runs anywhere. Everything here drives ``parse_model_output``
(pure) or a stub ``VisionBackend``.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.core.vl_models.backends import BACKEND_OPENAI_COMPAT, VisionResponse
from src.core.vl_models.puzzle_parser import (
    ModelOutputError,
    VisionBackendError,
    default_backend,
    extract_json_block,
    parse_model_output,
    parse_puzzle_image,
)


@dataclass(frozen=True)
class SimpleSettings:
    """The two settings ``default_backend`` reads, without pydantic-settings' .env."""

    ollama_model_name: str
    ollama_provider_url: str


VALID_PAYLOAD = """```json
{
  "layout": [
    ["  ", "01", "  "],
    ["  ", "  ", "  "],
    ["03", "  ", "02"]
  ],
  "walls": [{"cell1": [0, 0], "cell2": [0, 1]}]
}
```"""


@dataclass
class FakeBackend:
    """Stands in for a real transport; records what it was asked for."""

    text: str
    name: str = "fake"
    model: str = "fake-model"
    error: Exception | None = None
    seen_prompt: str | None = None

    def generate(self, image_path: Path, prompt: str) -> VisionResponse:
        self.seen_prompt = prompt
        if self.error is not None:
            raise self.error
        return VisionResponse(text=self.text)


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "puzzle.png"
    path.write_bytes(b"not-a-real-png")
    return path


class TestExtractJsonBlock:
    def test_reads_a_fenced_block(self):
        assert extract_json_block('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_falls_back_to_bare_braces(self):
        assert extract_json_block('chatter {"a": 1} more') == '{"a": 1}'

    def test_returns_none_when_there_is_no_object(self):
        assert extract_json_block("I am thinking about it") is None


class TestParseModelOutput:
    def test_builds_a_solver_ready_puzzle(self):
        result = parse_model_output(VALID_PAYLOAD)

        assert result.puzzle["grid_size"] == (3, 3)
        assert result.puzzle["num_map"] == {1: (0, 1), 2: (2, 2), 3: (2, 0)}
        assert result.puzzle["walls"] == {((0, 0), (0, 1))}
        assert result.warnings == ()

    def test_walls_are_stored_in_canonical_order(self):
        """The pair is sorted, so the same wall never appears twice."""
        payload = """{"layout": [["01", "02"]],
                      "walls": [{"cell1": [0, 1], "cell2": [0, 0]}]}"""
        assert parse_model_output(payload).puzzle["walls"] == {((0, 0), (0, 1))}

    def test_rejects_output_with_no_json(self):
        with pytest.raises(ModelOutputError, match="No JSON object"):
            parse_model_output("Let me think about this puzzle...")

    def test_rejects_malformed_json(self):
        with pytest.raises(ModelOutputError, match="invalid JSON"):
            parse_model_output('{"layout": [["01"]], "walls": [},}')

    def test_rejects_output_that_misses_the_schema(self):
        with pytest.raises(ModelOutputError, match="does not match"):
            parse_model_output('{"layout": [["01"]]}')

    def test_rejects_an_empty_grid(self):
        with pytest.raises(ModelOutputError, match="empty grid"):
            parse_model_output('{"layout": [], "walls": []}')

    def test_rejects_a_ragged_grid(self):
        """A hallucinated row length would crash the solver much further downstream."""
        payload = '{"layout": [["01", "  "], ["  "]], "walls": []}'
        with pytest.raises(ModelOutputError, match="ragged grid"):
            parse_model_output(payload)


class TestHallucinatedWalls:
    """Invented walls are as fatal as missed ones, so they are dropped and reported."""

    def test_out_of_bounds_walls_are_dropped_with_a_warning(self):
        payload = """{"layout": [["01", "02"]],
                      "walls": [{"cell1": [0, 0], "cell2": [0, 9]}]}"""
        result = parse_model_output(payload)

        assert result.puzzle["walls"] == set()
        assert any("out-of-bounds" in warning for warning in result.warnings)

    def test_non_adjacent_walls_are_dropped_with_a_warning(self):
        payload = """{"layout": [["01", "  ", "02"]],
                      "walls": [{"cell1": [0, 0], "cell2": [0, 2]}]}"""
        result = parse_model_output(payload)

        assert result.puzzle["walls"] == set()
        assert any("non-adjacent" in warning for warning in result.warnings)

    def test_a_grid_with_no_waypoints_is_reported(self):
        result = parse_model_output('{"layout": [["  ", "  "]], "walls": []}')
        assert any("no numbered waypoints" in w for w in result.warnings)


class TestParsePuzzleImage:
    def test_uses_the_injected_backend_and_prompt(self, image: Path):
        backend = FakeBackend(text=VALID_PAYLOAD)
        result = parse_puzzle_image(image, backend=backend, prompt="my prompt")

        assert backend.seen_prompt == "my prompt"
        assert result.puzzle["grid_size"] == (3, 3)

    def test_missing_image_is_reported_before_any_model_call(self, tmp_path: Path):
        backend = FakeBackend(text=VALID_PAYLOAD)
        with pytest.raises(FileNotFoundError):
            parse_puzzle_image(tmp_path / "nope.png", backend=backend)
        assert backend.seen_prompt is None

    def test_a_backend_failure_surfaces_as_vision_backend_error(self, image: Path):
        """A missing model must not look like a parsing problem."""
        backend = FakeBackend(
            text="", error=ConnectionError("model 'qwen3.5:4b-q8_0' not found")
        )
        with pytest.raises(VisionBackendError, match="not found"):
            parse_puzzle_image(image, backend=backend)

    def test_reasoning_leftovers_surface_as_a_model_output_error(self, image: Path):
        """The exact failure that made the un-plumbed think flag so expensive."""
        backend = FakeBackend(text="Okay, so the grid looks like a 6x6 board and ...")
        with pytest.raises(ModelOutputError, match="reasoning"):
            parse_puzzle_image(image, backend=backend)


class TestDefaultBackend:
    def test_unconfigured_settings_fail_loudly(self, monkeypatch):
        """No model configured must say so, not fail somewhere deep in the transport."""
        monkeypatch.setattr(
            "src.core.vl_models.puzzle_parser.get_settings",
            lambda: SimpleSettings(ollama_model_name="", ollama_provider_url=""),
        )
        with pytest.raises(VisionBackendError, match="must be set"):
            default_backend()

    def test_built_from_settings_with_reasoning_off(self, monkeypatch):
        monkeypatch.setattr(
            "src.core.vl_models.puzzle_parser.get_settings",
            lambda: SimpleSettings(
                ollama_model_name="qwen3.5:4b-q8_0",
                ollama_provider_url="http://host:11435",
            ),
        )
        backend = default_backend()

        assert backend.name == BACKEND_OPENAI_COMPAT
        assert backend.model == "qwen3.5:4b-q8_0"
        assert backend.think is False
