"""Translate Spanish and code-switched questions to English before retrieval (DECISIONS.md D26)."""

import json
import re
import time
import urllib.request
from typing import Protocol
from urllib.error import URLError

from atr.cache import JsonCache, cache_key
from atr.evalset.models import QuestionLanguage

PROMPT_TEMPLATE = (
    "Translate the text below into English. It is a question about soccer and may mix "
    "Spanish and English. Translate every sentence in full: do not answer it, do not "
    "summarize it, and do not drop any sentence. Reply with the English translation only.\n\n"
    "Text:\n{text}"
)
# A translation much shorter or longer than its source lost or invented content.
MIN_LENGTH_RATIO = 0.5
MAX_LENGTH_RATIO = 2.0
# Reasoning models that take Ollama's `think` level; low keeps a translation fast.
LOW_EFFORT_REASONERS = frozenset({"gpt-oss:20b"})
_LATIN_OR_COMMON = re.compile(r"^[\u0000-\u024F\u2000-\u206F\u20A0-\u20BF]*$")


def translation_problem(source: str, translation: str) -> str | None:
    """Why a model reply cannot be trusted as a translation, or None when it looks sound."""
    if not _LATIN_OR_COMMON.match(translation):
        return "contains non-Latin script"
    if "English:" in translation or "Text:" in translation:
        return "repeats the prompt"
    ratio = len(translation.split()) / max(len(source.split()), 1)
    if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
        return f"length ratio {ratio:.2f} is outside [{MIN_LENGTH_RATIO}, {MAX_LENGTH_RATIO}]"
    return None


class QueryRewriter(Protocol):
    name: str

    def rewrite(self, text: str, language: QuestionLanguage) -> str: ...

    def cost_ms(self, text: str, language: QuestionLanguage) -> float:
        """What rewriting this text cost when it was computed, so cached reruns stay honest."""
        ...


class Identity:
    name = "none"

    def rewrite(self, text: str, language: QuestionLanguage) -> str:
        return text

    def cost_ms(self, text: str, language: QuestionLanguage) -> float:
        return 0.0


def _post(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body: dict[str, object] = json.load(response)
    return body


class OllamaTranslator:
    def __init__(
        self,
        *,
        model: str = "qwen2.5:14b",
        fallback_model: str | None = "gpt-oss:20b",
        host: str = "http://localhost:11434",
        cache: JsonCache,
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._fallback_model = fallback_model
        self._host = host.rstrip("/")
        self._cache = cache
        self._timeout = timeout
        self.name = f"translate:{model}"

    def rewrite(self, text: str, language: QuestionLanguage) -> str:
        if language == "en":
            return text
        cached = self._entry(text)
        if cached is not None:
            return cached[0]
        started = time.perf_counter()
        translation = self._ask(text)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._cache.put(self._key(text), {"text": translation, "ms": elapsed_ms})
        return translation

    def cost_ms(self, text: str, language: QuestionLanguage) -> float:
        if language == "en":
            return 0.0
        cached = self._entry(text)
        return 0.0 if cached is None else cached[1]

    def _key(self, text: str) -> str:
        fallback = self._fallback_model or ""
        return cache_key("translate", self._model, fallback, PROMPT_TEMPLATE, text)

    def _entry(self, text: str) -> tuple[str, float] | None:
        cached = self._cache.get(self._key(text))
        if not isinstance(cached, dict):
            return None
        translation, elapsed_ms = cached.get("text"), cached.get("ms")
        if not isinstance(translation, str) or not isinstance(elapsed_ms, int | float):
            return None
        return translation, float(elapsed_ms)

    def _ask(self, text: str) -> str:
        """The first usable reply, trying the fallback model when the main one is rejected.

        Temperature is 0, so asking the same model again would return the same reply.
        """
        models = [self._model] + ([self._fallback_model] if self._fallback_model else [])
        problems: list[str] = []
        for model in models:
            translation = self._generate(model, text)
            problem = translation_problem(text, translation)
            if problem is None:
                return translation
            problems.append(f"{model}: {problem}")
        raise RuntimeError(
            f"no usable translation from {self._model} ({'; '.join(problems)}) for {text[:60]!r}"
        )

    def _generate(self, model: str, text: str) -> str:
        payload: dict[str, object] = {
            "model": model,
            "prompt": PROMPT_TEMPLATE.format(text=text),
            "stream": False,
            "options": {"temperature": 0, "seed": 0},
        }
        if model in LOW_EFFORT_REASONERS:
            payload["think"] = "low"
        try:
            reply = _post(f"{self._host}/api/generate", payload, self._timeout)
        except (URLError, OSError, ValueError) as error:
            raise RuntimeError(f"Ollama request failed for model {model}: {error}") from error
        response = reply.get("response")
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(f"Ollama model {model} returned no translation")
        return response.strip()
