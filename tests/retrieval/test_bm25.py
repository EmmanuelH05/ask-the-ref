import math

import pytest

from atr.retrieval.base import Doc, Retriever
from atr.retrieval.bm25 import BM25Retriever

# The game is invented: nothing here is real Laws of the Game wording.


def doc(passage_id: str, text: str, language: str = "en") -> Doc:
    return Doc(passage_id, language, text)  # type: ignore[arg-type]


def filler(count: int) -> list[Doc]:
    return [doc(f"f{i}", f"quix plim vank dorp {i}x") for i in range(count)]


def test_hand_computed_score_on_three_docs() -> None:
    # N = 3, query "zorb": df = 2 (docs a and b).
    # idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6)
    # dl: a = 3, b = 4, c = 4, so avgdl = 11 / 3. k1 = 1.5, b = 0.75.
    docs = [
        doc("a", "zorb quix plim"),
        doc("b", "zorb zorb vank dorp"),
        doc("c", "quix vank dorp plim"),
    ]
    idf = math.log(1.6)
    # a: tf = 1, dl = 3
    norm_a = 1.5 * (0.25 + 0.75 * 3 / (11 / 3))
    expected_a = idf * 1 * 2.5 / (1 + norm_a)
    # b: tf = 2, dl = 4
    norm_b = 1.5 * (0.25 + 0.75 * 4 / (11 / 3))
    expected_b = idf * 2 * 2.5 / (2 + norm_b)

    hits = BM25Retriever(docs).search("zorb", 10)

    assert [h.passage_id for h in hits] == ["b", "a"]
    assert hits[0].score == pytest.approx(expected_b)
    assert hits[1].score == pytest.approx(expected_a)
    assert hits[1].score == pytest.approx(0.5118, abs=1e-3)


def test_more_occurrences_outrank_fewer() -> None:
    docs = [doc("once", "zorb quix plim"), doc("twice", "zorb zorb plim"), *filler(3)]
    hits = BM25Retriever(docs).search("zorb", 5)
    assert [h.passage_id for h in hits] == ["twice", "once"]


def test_rare_term_outweighs_common_term() -> None:
    docs = [doc("rare", "zorb quix"), doc("common", "vank quix"), *filler(5)]
    hits = BM25Retriever(docs).search("zorb vank", 3)
    assert hits[0].passage_id == "rare"


def test_shorter_doc_outranks_longer_with_same_tf() -> None:
    docs = [doc("short", "zorb quix"), doc("long", "zorb quix plim vank dorp"), *filler(3)]
    hits = BM25Retriever(docs).search("zorb", 5)
    assert [h.passage_id for h in hits] == ["short", "long"]


def test_repeated_query_term_counts_once() -> None:
    docs = [doc("a", "zorb quix"), doc("b", "plim")]
    retriever = BM25Retriever(docs)
    assert retriever.search("zorb zorb", 5) == retriever.search("zorb", 5)


def test_stemming_connects_query_and_doc() -> None:
    hits = BM25Retriever([doc("a", "the zorb"), doc("b", "quix")]).search("zorbs", 5)
    assert [h.passage_id for h in hits] == ["a"]


def test_accent_folding_connects_query_and_doc() -> None:
    docs = [doc("a", "el árbitro", "es"), doc("b", "quix", "es")]
    assert [h.passage_id for h in BM25Retriever(docs).search("arbitro", 5)] == ["a"]


def test_docs_without_query_term_not_returned() -> None:
    docs = [doc("a", "zorb"), doc("b", "quix"), doc("c", "plim")]
    assert [h.passage_id for h in BM25Retriever(docs).search("zorb", 5)] == ["a"]


def test_punctuation_only_query_returns_empty() -> None:
    assert BM25Retriever([doc("a", "zorb")]).search("?! ... --", 5) == []


def test_k_limits_and_oversized_k_is_fine() -> None:
    docs = [doc(f"d{i}", "zorb quix") for i in range(4)] + filler(2)
    retriever = BM25Retriever(docs)
    assert len(retriever.search("zorb", 2)) == 2
    assert len(retriever.search("zorb", 100)) == 4


def test_shared_passage_id_gives_one_hit_with_best_score() -> None:
    docs = [
        doc("p1", "zorb quix plim vank dorp", "en"),
        doc("p1", "zorb quix", "es"),
        doc("p2", "plim"),
    ]
    retriever = BM25Retriever(docs)
    hits = retriever.search("zorb", 5)
    separate = [doc("p1-en", docs[0].text), doc("p1-es", docs[1].text), docs[2]]
    by_id = {h.passage_id: h.score for h in BM25Retriever(separate).search("zorb", 5)}
    assert [h.passage_id for h in hits] == ["p1"]
    assert hits[0].score == pytest.approx(max(by_id["p1-en"], by_id["p1-es"]))


def test_ties_ordered_by_passage_id() -> None:
    docs = [doc("b", "zorb quix"), doc("a", "zorb quix"), doc("c", "zorb quix")]
    assert [h.passage_id for h in BM25Retriever(docs).search("zorb", 5)] == ["a", "b", "c"]


def test_empty_doc_list_raises() -> None:
    with pytest.raises(ValueError):
        BM25Retriever([])


def test_name_defaults_and_can_be_set() -> None:
    docs = [doc("a", "zorb")]
    assert BM25Retriever(docs).name == "bm25"
    assert BM25Retriever(docs, name="custom").name == "custom"


def test_satisfies_retriever_protocol() -> None:
    retriever: Retriever = BM25Retriever([doc("a", "zorb")])
    assert retriever.search("zorb", 1)[0].passage_id == "a"


def test_scores_do_not_depend_on_the_order_query_terms_are_summed_in() -> None:
    docs = [Doc("L1.1", "en", "zorb quix plim wobble gate zorb"), Doc("L1.2", "en", "plim gate")]
    retriever = BM25Retriever(docs)
    forward = retriever.search("zorb quix plim wobble gate", 2)
    backward = retriever.search("gate wobble plim quix zorb", 2)
    assert forward == backward
