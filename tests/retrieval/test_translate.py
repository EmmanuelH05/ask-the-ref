import inspect
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

from atr.cache import JsonCache
from atr.evalset.models import QuestionLanguage
from atr.retrieval import translate
from atr.retrieval.translate import Identity, OllamaTranslator, QueryRewriter

SPANISH = "¿Puede el portero recoger el balón con la mano tras un pase de un compañero?"
MIXED = "Can the keeper agarrar la pelota after a throw-in?"
ANSWER = "Can the keeper pick it up?"


class FakePost:
    def __init__(self, reply: dict[str, object] | Exception) -> None:
        self.reply = reply
        self.calls: list[tuple[str, dict[str, object], float]] = []

    def __call__(
        self, url: str, payload: dict[str, object], timeout: float
    ) -> dict[str, object]:
        self.calls.append((url, payload, timeout))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def install(monkeypatch: pytest.MonkeyPatch, reply: dict[str, object] | Exception) -> FakePost:
    fake = FakePost(reply)
    monkeypatch.setattr(translate, "_post", fake)
    return fake


def make(tmp_path: Path, model: str = "m1") -> OllamaTranslator:
    return OllamaTranslator(model=model, cache=JsonCache(tmp_path / "cache"))


@pytest.mark.parametrize("language", ["en", "es", "cs"])
def test_identity_returns_text_for_every_language(language: QuestionLanguage) -> None:
    assert Identity().rewrite("some text", language) == "some text"


def test_identity_name_is_none() -> None:
    assert Identity().name == "none"


def test_both_rewriters_satisfy_the_protocol(tmp_path: Path) -> None:
    first: QueryRewriter = Identity()
    second: QueryRewriter = make(tmp_path)
    assert first.name == "none"
    assert second.name == "translate:m1"


def test_english_is_unchanged_and_makes_no_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, {"response": "x"})
    text = "Is a throw-in a free kick?"
    assert make(tmp_path).rewrite(text, "en") == text
    assert fake.calls == []


def test_spanish_makes_one_call_with_expected_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, {"response": ANSWER})
    make(tmp_path).rewrite(SPANISH, "es")
    assert len(fake.calls) == 1
    url, payload, _ = fake.calls[0]
    assert url.endswith("/api/generate")
    assert payload["model"] == "m1"
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0, "seed": 0}
    prompt: Any = payload["prompt"]
    assert SPANISH in prompt


def test_response_is_stripped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {"response": "  \n" + ANSWER + "\n "})
    assert make(tmp_path).rewrite(SPANISH, "es") == ANSWER


def test_second_rewrite_is_cached_even_from_a_fresh_translator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, {"response": ANSWER})
    first = make(tmp_path).rewrite(SPANISH, "es")
    second = make(tmp_path).rewrite(SPANISH, "es")
    third = make(tmp_path).rewrite(SPANISH, "es")
    assert first == second == third == ANSWER
    assert len(fake.calls) == 1


def test_same_translator_instance_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, {"response": ANSWER})
    translator = make(tmp_path)
    translator.rewrite(SPANISH, "es")
    translator.rewrite(SPANISH, "es")
    assert len(fake.calls) == 1


def test_different_model_is_a_different_cache_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, {"response": ANSWER})
    make(tmp_path, "m1").rewrite(SPANISH, "es")
    make(tmp_path, "m2").rewrite(SPANISH, "es")
    assert len(fake.calls) == 2


def test_code_switched_question_is_translated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, {"response": "Can the keeper grab the ball after a throw-in?"})
    result = make(tmp_path).rewrite(MIXED, "cs")
    assert result == "Can the keeper grab the ball after a throw-in?"
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "reply",
    [{"other": "x"}, {"response": ""}, {"response": "  \n "}, URLError("refused")],
)
def test_failures_raise_naming_model_and_cache_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: dict[str, object] | Exception
) -> None:
    fake = install(monkeypatch, reply)
    translator = make(tmp_path, "model-xyz")
    with pytest.raises(RuntimeError, match="model-xyz"):
        translator.rewrite(SPANISH, "es")
    with pytest.raises(RuntimeError, match="model-xyz"):
        translator.rewrite(SPANISH, "es")
    assert len(fake.calls) == 2


def test_name_includes_model(tmp_path: Path) -> None:
    assert make(tmp_path, "qwen").name == "translate:qwen"


def test_prompt_template_has_text_placeholder() -> None:
    assert "{text}" in translate.PROMPT_TEMPLATE


def test_post_signature() -> None:
    assert inspect.isfunction(translate._post)
    assert list(inspect.signature(translate._post).parameters) == ["url", "payload", "timeout"]
