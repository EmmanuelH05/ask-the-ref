"""Cached, deterministic completions from a local Ollama server (DECISIONS.md D34)."""

import json
import time
import urllib.request
from typing import NamedTuple
from urllib.error import URLError

from atr.cache import JsonCache, cache_key


class Completion(NamedTuple):
    text: str
    ms: float


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


class OllamaLLM:
    def __init__(
        self,
        *,
        cache: JsonCache,
        host: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self._cache = cache
        self._host = host.rstrip("/")
        self._timeout = timeout

    def complete(self, model: str, prompt: str, *, think: str | None = None) -> Completion:
        key = cache_key("llm", model, think or "", prompt)
        cached = self._stored(key)
        if cached is not None:
            return cached
        started = time.perf_counter()
        text = self._generate(model, prompt, think)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._cache.put(key, {"text": text, "ms": elapsed_ms})
        return Completion(text, elapsed_ms)

    def _stored(self, key: str) -> Completion | None:
        entry = self._cache.get(key)
        if not isinstance(entry, dict):
            return None
        text, elapsed_ms = entry.get("text"), entry.get("ms")
        if not isinstance(text, str) or not isinstance(elapsed_ms, int | float):
            return None
        return Completion(text, float(elapsed_ms))

    def _generate(self, model: str, prompt: str, think: str | None) -> str:
        payload: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": 0},
        }
        if think is not None:
            payload["think"] = think
        try:
            reply = _post(f"{self._host}/api/generate", payload, self._timeout)
        except (URLError, OSError, ValueError) as error:
            raise RuntimeError(f"Ollama request failed for model {model}: {error}") from error
        response = reply.get("response")
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(f"Ollama model {model} returned no response")
        return response.strip()
