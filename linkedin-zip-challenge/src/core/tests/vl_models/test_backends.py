# src/core/tests/vl_models/test_backends.py
"""Covers the transport layer without talking to Ollama.

The regression these tests exist for: before 2026-08-22 the thinking switch was only
wired into the native transport, so ``--no-think`` was a silent no-op on the
OpenAI-compatible path. Measured cost of that silence on ``qwen3.5:4b-q8_0``:
66s/JSON 0/2 against 4.1s/JSON 2/2. The knob is different on each surface --
``/api/chat`` takes ``think``, ``/v1`` ignores it and takes ``reasoning_effort`` --
so each translation is asserted separately.
"""

from pathlib import Path

import pytest

from src.core.vl_models.backends import (
    BACKEND_NATIVE,
    BACKEND_OPENAI_COMPAT,
    REASONING_EFFORT_OFF,
    OllamaNativeBackend,
    OllamaOpenAICompatBackend,
    build_backend,
    media_type_for,
    native_base_url,
    openai_compat_base_url,
)


class TestBaseUrls:
    @pytest.mark.parametrize(
        "given",
        ["http://host:11435", "http://host:11435/", "http://host:11435/v1"],
    )
    def test_native_url_never_keeps_the_v1_suffix(self, given: str):
        assert native_base_url(given) == "http://host:11435"

    @pytest.mark.parametrize(
        "given",
        ["http://host:11435", "http://host:11435/", "http://host:11435/v1"],
    )
    def test_openai_compat_url_always_ends_in_v1(self, given: str):
        assert openai_compat_base_url(given) == "http://host:11435/v1"


class TestMediaType:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("a.png", "image/png"),
            ("a.PNG", "image/png"),
            ("a.jpg", "image/jpeg"),
            ("a.jpeg", "image/jpeg"),
        ],
    )
    def test_jpg_is_normalised_to_jpeg(self, filename: str, expected: str):
        assert media_type_for(Path(filename)) == expected


class TestThinkingSwitch:
    """The whole point of the module: one flag, two very different wire formats."""

    def test_openai_compat_disables_reasoning_via_reasoning_effort(self):
        backend = OllamaOpenAICompatBackend(
            model="m", base_url="http://host:11435", think=False
        )
        assert (
            backend._model_settings()["openai_reasoning_effort"] == REASONING_EFFORT_OFF
        )

    def test_openai_compat_leaves_the_default_alone_when_think_is_unset(self):
        backend = OllamaOpenAICompatBackend(model="m", base_url="http://host:11435")
        assert "openai_reasoning_effort" not in backend._model_settings()

    def test_openai_compat_cannot_force_reasoning_on(self):
        """There is no verified value for this, so it must not silently pretend."""
        backend = OllamaOpenAICompatBackend(
            model="m", base_url="http://host:11435", think=True
        )
        assert "openai_reasoning_effort" not in backend._model_settings()

    def test_native_sends_think_on_the_wire(self, tmp_path: Path, monkeypatch):
        captured: dict[str, object] = {}

        class FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None: ...

            def json(self) -> dict[str, object]:
                return {"message": {"content": "ok", "thinking": "abc"}}

        def fake_post(url: str, json: dict, timeout: float):
            captured["url"] = url
            captured["payload"] = json
            return FakeResponse()

        monkeypatch.setattr("src.core.vl_models.backends.httpx.post", fake_post)
        image = tmp_path / "puzzle.png"
        image.write_bytes(b"not-a-real-png")

        backend = OllamaNativeBackend(
            model="m", base_url="http://host:11435/v1", think=False
        )
        response = backend.generate(image, "prompt")

        assert captured["url"] == "http://host:11435/api/chat"
        assert captured["payload"]["think"] is False
        assert response.text == "ok"
        assert response.thinking_characters == 3

    def test_native_omits_think_when_unset(self, tmp_path: Path, monkeypatch):
        captured: dict[str, object] = {}

        class FakeResponse:
            def raise_for_status(self) -> None: ...

            def json(self) -> dict[str, object]:
                return {"message": {"content": "ok"}}

        monkeypatch.setattr(
            "src.core.vl_models.backends.httpx.post",
            lambda url, json, timeout: (captured.update(payload=json), FakeResponse())[
                1
            ],
        )
        image = tmp_path / "puzzle.png"
        image.write_bytes(b"x")

        OllamaNativeBackend(model="m", base_url="http://host:11435").generate(
            image, "prompt"
        )
        assert "think" not in captured["payload"]


class TestBuildBackend:
    def test_dispatches_by_name(self):
        assert isinstance(
            build_backend(BACKEND_NATIVE, "m", "http://host"), OllamaNativeBackend
        )
        assert isinstance(
            build_backend(BACKEND_OPENAI_COMPAT, "m", "http://host"),
            OllamaOpenAICompatBackend,
        )

    def test_rejects_an_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            build_backend("telepathy", "m", "http://host")
