# src/core/vl_models/backends.py
"""Transports that carry one image plus one prompt to a local vision model.

Two backends exist because Ollama exposes two different HTTP surfaces and they do
**not** accept the same knobs:

``native``
    Ollama's own ``/api/chat``. Reports load/eval timings, token counts and the
    reasoning text in a separate ``thinking`` field, and takes a boolean ``think``.

``openai-compat``
    Ollama's ``/v1`` OpenAI-compatible endpoint, reached through ``pydantic-ai``.
    Fewer counters, but it is the transport the shipped parser is built on.

The distinction is not cosmetic. Measured 2026-08-22 against ``qwen3.5:4b-q8_0``:
``/v1`` silently **ignores** a top-level ``think`` field, while
``reasoning_effort="none"`` does disable reasoning (9.7s/1392 reasoning chars ->
0.9s/0). An earlier version of the benchmark passed ``think`` only to the native
path, so ``--no-think`` was a no-op on ``openai-compat`` and that transport measured
66s per image with a 0/2 JSON parse rate against 4.1s and 2/2 on native.

This module is the single place that knows the translation, so the two callers --
``benchmark.py`` and ``puzzle_parser.py`` -- cannot drift apart again.
"""

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from loguru import logger

BACKEND_NATIVE = "native"
BACKEND_OPENAI_COMPAT = "openai-compat"
BACKEND_CHOICES = (BACKEND_NATIVE, BACKEND_OPENAI_COMPAT)

# Ollama maps OpenAI's `reasoning_effort` onto its own thinking switch; "none" is
# the value that turns reasoning off. Verified 2026-08-22, see module docstring.
REASONING_EFFORT_OFF = "none"

DEFAULT_NUM_CTX = 8192
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42

NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class VisionResponse:
    """What every backend returns, whether or not the transport can measure it."""

    text: str
    thinking_characters: int = 0
    load_seconds: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    eval_seconds: float | None = None


class VisionBackend(Protocol):
    """One image plus one prompt in, one block of text out."""

    name: str
    model: str

    def generate(self, image_path: Path, prompt: str) -> VisionResponse: ...


def _nanoseconds_to_seconds(value: int | None) -> float | None:
    return value / NANOSECONDS_PER_SECOND if value is not None else None


def native_base_url(base_url: str) -> str:
    """Ollama's own API lives at the root, never under ``/v1``."""
    trimmed = base_url.rstrip("/")
    return trimmed[: -len("/v1")] if trimmed.endswith("/v1") else trimmed


def openai_compat_base_url(base_url: str) -> str:
    """pydantic-ai's ``OllamaProvider`` expects the ``/v1`` OpenAI-compatible root."""
    trimmed = base_url.rstrip("/")
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


def media_type_for(image_path: Path) -> str:
    suffix = image_path.suffix.lstrip(".").lower()
    return f"image/{'jpeg' if suffix == 'jpg' else suffix}"


@dataclass(frozen=True)
class OllamaNativeBackend:
    """Ollama's ``/api/chat``: the only transport that reports timing counters."""

    model: str
    base_url: str
    think: bool | None = None
    num_ctx: int = DEFAULT_NUM_CTX
    seed: int = DEFAULT_SEED
    temperature: float = DEFAULT_TEMPERATURE
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    name: str = BACKEND_NATIVE

    def generate(self, image_path: Path, prompt: str) -> VisionResponse:
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "user", "content": prompt, "images": [encoded_image]}
            ],
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
            },
        }
        if self.think is not None:
            payload["think"] = self.think

        response = httpx.post(
            f"{native_base_url(self.base_url)}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        message = body.get("message", {})
        return VisionResponse(
            text=message.get("content", ""),
            thinking_characters=len(message.get("thinking") or ""),
            load_seconds=_nanoseconds_to_seconds(body.get("load_duration")),
            prompt_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
            eval_seconds=_nanoseconds_to_seconds(body.get("eval_duration")),
        )


@dataclass(frozen=True)
class OllamaOpenAICompatBackend:
    """The ``/v1`` path via pydantic-ai, i.e. the transport the app will ship.

    ``output_type`` stays ``str`` on purpose: the 2025-10-24 experiments settled on
    prompt engineering plus local parsing because the models that supported tool
    calling had broken vision.
    """

    model: str
    base_url: str
    think: bool | None = None
    seed: int = DEFAULT_SEED
    temperature: float = DEFAULT_TEMPERATURE
    name: str = BACKEND_OPENAI_COMPAT

    def _model_settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if self.think is False:
            settings["openai_reasoning_effort"] = REASONING_EFFORT_OFF
        elif self.think is True:
            # No verified value forces reasoning back on over /v1; the model's own
            # default already is on for reasoning models, so leaving it unset is
            # the honest behaviour rather than guessing an effort level.
            logger.warning(
                "think=True is not expressible on the {} transport; "
                "falling back to the model default",
                BACKEND_OPENAI_COMPAT,
            )
        return settings

    def generate(self, image_path: Path, prompt: str) -> VisionResponse:
        from pydantic_ai import Agent
        from pydantic_ai.messages import BinaryContent
        from pydantic_ai.models.ollama import OllamaModel
        from pydantic_ai.providers.ollama import OllamaProvider

        agent = Agent(
            model=OllamaModel(
                self.model,
                provider=OllamaProvider(base_url=openai_compat_base_url(self.base_url)),
            ),
            output_type=str,
        )
        image_content = BinaryContent(
            data=image_path.read_bytes(), media_type=media_type_for(image_path)
        )
        result = agent.run_sync(
            [prompt, image_content], model_settings=self._model_settings()
        )
        usage = result.usage
        return VisionResponse(
            text=result.output,
            prompt_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )


def build_backend(
    name: str,
    model: str,
    base_url: str,
    think: bool | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    seed: int = DEFAULT_SEED,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> VisionBackend:
    """Picks a backend by name, so callers can expose it as one CLI flag."""
    if name == BACKEND_OPENAI_COMPAT:
        return OllamaOpenAICompatBackend(
            model=model,
            base_url=base_url,
            think=think,
            seed=seed,
            temperature=temperature,
        )
    if name == BACKEND_NATIVE:
        return OllamaNativeBackend(
            model=model,
            base_url=base_url,
            think=think,
            num_ctx=num_ctx,
            seed=seed,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unknown backend '{name}'. Expected one of {BACKEND_CHOICES}.")
