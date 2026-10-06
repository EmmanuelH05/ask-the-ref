from pathlib import Path
from urllib.error import URLError

import pytest

from atr import llm
from atr.cache import JsonCache, cache_key
from atr.llm import Completion, OllamaLLM


class FakePost:
    def __init__(self, reply: dict[str, object] | Exception | None = None) -> None:
        self.reply: dict[str, object] | Exception = (
            {"response": "  hello  "} if reply is None else reply
        )
        self.calls: list[tuple[str, dict[str, object], float]] = []

    def __call__(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        self.calls.append((url, payload, timeout))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def post(monkeypatch: pytest.MonkeyPatch) -> FakePost:
    fake = FakePost()
    monkeypatch.setattr(llm, "_post", fake)
    return fake


def make(tmp_path: Path) -> OllamaLLM:
    return OllamaLLM(cache=JsonCache(tmp_path))


def test_posts_generate_payload(post: FakePost, tmp_path: Path) -> None:
    make(tmp_path).complete("m1", "why?")

    url, payload, _ = post.calls[0]
    assert url.endswith("/api/generate")
    assert payload == {
        "model": "m1",
        "prompt": "why?",
        "stream": False,
        "options": {"temperature": 0, "seed": 0},
    }


def test_think_in_payload_only_when_given(post: FakePost, tmp_path: Path) -> None:
    client = make(tmp_path)
    client.complete("m1", "a")
    client.complete("m1", "a", think="low")

    assert "think" not in post.calls[0][1]
    assert post.calls[1][1]["think"] == "low"


def test_response_text_is_stripped(post: FakePost, tmp_path: Path) -> None:
    assert make(tmp_path).complete("m1", "a").text == "hello"


def test_ms_is_measured_wall_clock(
    post: FakePost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    times = iter([5.0, 5.5])
    monkeypatch.setattr(llm.time, "perf_counter", lambda: next(times))

    assert make(tmp_path).complete("m1", "a") == Completion("hello", 500.0)


def test_cache_hit_returns_stored_ms_without_post(
    post: FakePost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    times = iter([5.0, 5.5, 9.0, 9.0001])
    monkeypatch.setattr(llm.time, "perf_counter", lambda: next(times))
    client = make(tmp_path)

    first = client.complete("m1", "a")
    second = client.complete("m1", "a")

    assert len(post.calls) == 1
    assert second == first == Completion("hello", 500.0)


def test_cache_hit_from_fresh_client_on_same_directory(post: FakePost, tmp_path: Path) -> None:
    first = make(tmp_path).complete("m1", "a")
    second = make(tmp_path).complete("m1", "a")

    assert len(post.calls) == 1
    assert second == first


def test_model_prompt_and_think_are_separate_entries(post: FakePost, tmp_path: Path) -> None:
    client = make(tmp_path)
    client.complete("m1", "a")
    client.complete("m2", "a")
    client.complete("m1", "b")
    client.complete("m1", "a", think="low")

    assert len(post.calls) == 4


@pytest.mark.parametrize(
    "reply",
    [URLError("down"), {"other": "x"}, {"response": "   "}, {"response": 3}],
)
def test_failures_raise_naming_model_and_cache_nothing(
    post: FakePost, tmp_path: Path, reply: dict[str, object] | Exception
) -> None:
    post.reply = reply
    client = make(tmp_path)

    with pytest.raises(RuntimeError, match="model-x"):
        client.complete("model-x", "a")
    with pytest.raises(RuntimeError, match="model-x"):
        client.complete("model-x", "a")

    assert len(post.calls) == 2


def test_wrong_shape_cache_entry_is_a_miss(post: FakePost, tmp_path: Path) -> None:
    cache = JsonCache(tmp_path)
    cache.put(cache_key("llm", "m1", "", "a"), "bare string")

    result = OllamaLLM(cache=cache).complete("m1", "a")

    assert result.text == "hello"
    assert len(post.calls) == 1
