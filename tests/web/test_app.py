import json

from atr.answer.generate import REFUSAL_TEXT, Answer
from atr.corpus.models import Passage, count_words
from atr.pipeline import PipelineResult
from atr.retrieval.base import Hit
from atr.web.app import (
    DISCLAIMER,
    Service,
    cited_sentence,
    detect_language,
    matched_words,
    route,
    why_found,
)


def passage(passage_id: str, language: str, text: str) -> Passage:
    return Passage.model_validate(
        {
            "passage_id": passage_id,
            "language": language,
            "law_number": 11,
            "law_title": "Offside",
            "section_index": 1,
            "section_title": "Offside position",
            "subheading": None,
            "source_slug": "offside-position",
            "text": text,
            "word_count": count_words(text),
        }
    )


CORPUS = {
    ("en", "L11.1"): passage(
        "L11.1",
        "en",
        "A player is in an offside position if nearer to the goal line. It is not an offence.",
    ),
    ("es", "L11.1"): passage("L11.1", "es", "Un jugador está en posición de fuera de juego."),
    ("en", "L1.1"): passage("L1.1", "en", "The field of play must be rectangular."),
}


def test_detect_language_separates_english_spanish_and_a_mix() -> None:
    assert detect_language("Is the ball out when it touches the line?") == "en"
    assert detect_language("¿Cuánto dura el descanso del medio tiempo?") == "es"
    assert detect_language("es offside si the ball comes off a defender?") == "cs"
    assert detect_language("offside?") == "en"


def test_matched_words_returns_query_words_whose_stems_appear_in_the_passage() -> None:
    assert matched_words("offside positions", CORPUS[("en", "L11.1")].text) == [
        "offside",
        "positions",
    ]
    assert matched_words("the keeper", CORPUS[("en", "L11.1")].text) == []


def test_why_found_names_matched_words_or_calls_it_semantic() -> None:
    assert why_found(["offside", "goal"]) == "matched words: offside, goal"
    assert why_found([]) == "semantic match (no shared keywords)"


def test_cited_sentence_picks_the_passage_sentence_closest_to_the_answer() -> None:
    text = CORPUS[("en", "L11.1")].text
    assert cited_sentence("Being in an offside position is not an offence.", text) == (
        "It is not an offence."
    )
    assert cited_sentence("Totally unrelated words here.", text) is None


class FakeSearch:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, str, int]] = []

    def search(self, text: str, language: str, k: int) -> PipelineResult:
        self.calls.append((text, language, k))
        return PipelineResult(self.hits[:k], [text], {"dense": 5.0})


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []

    def answer(self, question: str, language: str, passages: list[Passage]) -> Answer:
        self.calls.append((question, language, [p.passage_id for p in passages]))
        if not passages:
            return Answer(text=REFUSAL_TEXT[language], citations=[], refused=True, ms=0.0)  # type: ignore[index]
        return Answer(text="Not an offence [L11.1].", citations=["L11.1"], refused=False, ms=40.0)


def service(top: float) -> tuple[Service, FakeSearch, FakeGenerator]:
    full = FakeSearch([Hit("L11.1", top), Hit("L1.1", top - 0.2)])
    keyword = FakeSearch([Hit("L1.1", 3.0)])
    generator = FakeGenerator()
    return (
        Service(full, keyword, generator, CORPUS, threshold=0.5, track_label="T"),
        full,
        generator,
    )  # type: ignore[arg-type]


def test_ask_answers_with_citations_both_editions_and_a_keyword_comparison() -> None:
    svc, full, generator = service(0.8)
    response = svc.ask("es offside si the ball comes off a defender?")
    assert response.language == "cs" and not response.refused
    assert full.calls[0][1] == "cs"
    assert generator.calls[0][2] == ["L11.1", "L1.1"]
    cited = response.passages[0]
    assert cited.passage_id == "L11.1" and cited.cited and cited.text_es.startswith("Un jugador")
    assert cited.score == 0.8 and "matched words" in cited.why
    assert cited.marked_sentence == "It is not an offence."
    assert response.passages[1].cited is False
    assert [p.passage_id for p in response.keyword_only] == ["L1.1"]
    assert response.disclaimer == DISCLAIMER


def test_ask_refuses_under_the_threshold_without_generating() -> None:
    svc, _, generator = service(0.3)
    response = svc.ask("What is the capital of France?")
    assert response.refused and response.answer == REFUSAL_TEXT["en"]
    assert generator.calls == [("What is the capital of France?", "en", [])]
    assert response.passages == [] and response.top_score == 0.3


def test_route_serves_the_page_the_api_and_404() -> None:
    svc, _, _ = service(0.8)
    status, content_type, body = route(svc, "GET", "/", b"", results={})
    assert status == 200 and content_type.startswith("text/html") and b"<form" in body
    status, content_type, body = route(
        svc, "POST", "/api/ask", json.dumps({"question": "offside?"}).encode(), results={}
    )
    assert status == 200 and content_type.startswith("application/json")
    assert json.loads(body)["passages"][0]["passage_id"] == "L11.1"
    status, _, body = route(svc, "POST", "/api/ask", b"{}", results={})
    assert status == 400
    status, _, _ = route(svc, "GET", "/nope", b"", results={})
    assert status == 404
    status, _, body = route(svc, "GET", "/results", b"", results={"ablation": "table here"})
    assert status == 200 and b"table here" in body
    status, _, body = route(svc, "GET", "/api/results", b"", results={"ablation": "x"})
    assert json.loads(body) == {"ablation": "x"}


def test_detect_language_keeps_plain_spanish_out_of_the_code_switched_path() -> None:
    assert detect_language("¿Puede un jugador volver a entrar al campo?") == "es"
    assert detect_language("¿Cuándo se concede un tiro libre directo a un equipo?") == "es"


def test_route_rejects_an_overlong_question_before_touching_the_service() -> None:
    from typing import cast

    body = json.dumps({"question": "x" * 501}).encode("utf-8")
    status, _, out = route(cast(Service, None), "POST", "/api/ask", body, results={})
    assert status == 413 and b"500" in out
    status, _, out = route(cast(Service, None), "POST", "/api/ask", b"{not json", results={})
    assert status == 400
