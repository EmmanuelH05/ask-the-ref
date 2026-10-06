import pytest

from atr.retrieval.base import Hit
from atr.retrieval.rerank import RERANKERS, CrossEncoderReranker, Reranker, rerank


class FakeReranker:
    """Scores a text by how many words it shares with the query, and records each call."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        words = set(query.split())
        return [float(len(words & set(text.split()))) for text in texts]


def test_rerankers_hold_exactly_the_two_keys() -> None:
    assert RERANKERS == {
        "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "bge-m3": "BAAI/bge-reranker-v2-m3",
    }


def test_cross_encoder_satisfies_protocol() -> None:
    reranker: Reranker = CrossEncoderReranker("minilm")
    assert reranker.name == "minilm"


def test_cross_encoder_name_is_the_key() -> None:
    assert CrossEncoderReranker("bge-m3").name == "bge-m3"


def test_cross_encoder_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="nope"):
        CrossEncoderReranker("nope")


def test_cross_encoder_construction_loads_nothing() -> None:
    assert not CrossEncoderReranker("minilm").is_loaded


def test_cross_encoder_empty_score_returns_empty_without_loading() -> None:
    reranker = CrossEncoderReranker("minilm")
    assert reranker.score("any query", []) == []
    assert not reranker.is_loaded


def test_rerank_orders_by_reranker_score_not_input_order() -> None:
    candidates = [("a", "green kettle"), ("b", "red blue kettle"), ("c", "red kettle")]
    hits = rerank("red blue kettle", candidates, FakeReranker(), 3)
    assert hits == [Hit("b", 3.0), Hit("c", 2.0), Hit("a", 1.0)]


def test_rerank_scores_all_candidates_in_one_call() -> None:
    fake = FakeReranker()
    candidates = [("a", "one"), ("b", "two"), ("c", "three")]
    rerank("one two", candidates, fake, 1)
    assert fake.calls == [("one two", ["one", "two", "three"])]


def test_rerank_merges_chunks_of_one_passage_keeping_best_score() -> None:
    candidates = [("a", "red"), ("a", "red blue"), ("b", "red")]
    hits = rerank("red blue", candidates, FakeReranker(), 5)
    assert hits == [Hit("a", 2.0), Hit("b", 1.0)]


def test_rerank_k_limits_result() -> None:
    candidates = [("a", "x"), ("b", "x y"), ("c", "x y z")]
    assert [h.passage_id for h in rerank("x y z", candidates, FakeReranker(), 2)] == ["c", "b"]


def test_rerank_ties_ordered_by_passage_id() -> None:
    candidates = [("b", "x"), ("c", "x"), ("a", "x")]
    hits = rerank("x", candidates, FakeReranker(), 3)
    assert [h.passage_id for h in hits] == ["a", "b", "c"]


def test_rerank_no_candidates_returns_empty_and_makes_no_call() -> None:
    fake = FakeReranker()
    assert rerank("x", [], fake, 3) == []
    assert fake.calls == []
