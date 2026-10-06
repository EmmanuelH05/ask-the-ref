"""Translate Spanish and code-switched questions to English before retrieval (DECISIONS.md D26)."""

import json
import time
import urllib.request
from typing import Protocol
from urllib.error import URLError

from atr.cache import JsonCache, cache_key
from atr.evalset.models import QuestionLanguage

PROMPT_TEMPLATE = (
    "Translate the following soccer question into English. "
    "Reply with the English translation only, with no explanation and no quotation marks.\n\n"
    "Question: {text}\n\nEnglish:"
)


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
        host: str = "http://localhost:11434",
        cache: JsonCache,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
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
        return cache_key("translate", self._model, PROMPT_TEMPLATE, text)

    def _entry(self, text: str) -> tuple[str, float] | None:
        cached = self._cache.get(self._key(text))
        if not isinstance(cached, dict):
            return None
        translation, elapsed_ms = cached.get("text"), cached.get("ms")
        if not isinstance(translation, str) or not isinstance(elapsed_ms, int | float):
            return None
        return translation, float(elapsed_ms)

    def _ask(self, text: str) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "prompt": PROMPT_TEMPLATE.format(text=text),
            "stream": False,
            "options": {"temperature": 0, "seed": 0},
        }
        try:
            reply = _post(f"{self._host}/api/generate", payload, self._timeout)
        except (URLError, OSError, ValueError) as error:
            raise RuntimeError(f"Ollama request failed for model {self._model}: {error}") from error
        response = reply.get("response")
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(f"Ollama model {self._model} returned no translation")
        return response.strip()
