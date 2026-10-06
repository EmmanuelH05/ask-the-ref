import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from atr.cache import cache_key
from atr.retrieval.base import Doc, Retriever
from atr.retrieval.dense import (
    ENCODERS,
    CachedEncoder,
    DenseRetriever,
    Encoder,
    SentenceTransformerEncoder,
    apply_prefix,
)

_LETTERS = "abcd"


def _vectorize(text: str) -> NDArray[np.float32]:
    counts = np.array([text.count(letter) for letter in _LETTERS], dtype=np.float32)
    if not counts.any():
        counts[0] = 1.0
    normalized: NDArray[np.float32] = counts / np.linalg.norm(counts)
    return normalized


def _stack(texts: list[str], suffix: str = "") -> NDArray[np.float32]:
    return np.stack([_vectorize(t + suffix) for t in texts]).astype(np.float32)


class FakeEncoder:
    name = "fake"

    def __init__(self) -> None:
        self.query_batches: list[list[str]] = []
        self.passage_batches: list[list[str]] = []

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        self.query_batches.append(list(texts))
        return _stack(texts)

    def encode_passages(self, texts: list[str]) -> NDArray[np.float32]:
        self.passage_batches.append(list(texts))
        return _stack(texts)


class OffsetPassageEncoder(FakeEncoder):
    """Passage vectors differ from query vectors for the same text."""

    def encode_passages(self, texts: list[str]) -> NDArray[np.float32]:
        self.passage_batches.append(list(texts))
        return _stack(texts, "d")


def _docs(*pairs: tuple[str, str]) -> list[Doc]:
    return [Doc(pid, "en", text) for pid, text in pairs]


def test_ranks_closest_doc_first() -> None:
    docs = _docs(("p1", "aaab"), ("p2", "bbbc"), ("p3", "cccc"))
    retriever = DenseRetriever(docs, FakeEncoder(), name="dense")
    assert retriever.search("bbc", 3)[0].passage_id == "p2"


def test_score_is_dot_product_of_normalized_vectors() -> None:
    retriever = DenseRetriever(_docs(("p1", "aab")), FakeEncoder(), name="dense")
    (hit,) = retriever.search("a", 1)
    # doc [2,1,0,0]/sqrt(5) . query [1,0,0,0]
    assert hit.score == pytest.approx(2 / math.sqrt(5), abs=1e-6)


def test_k_limits_results() -> None:
    docs = _docs(("p1", "a"), ("p2", "b"), ("p3", "c"))
    assert len(DenseRetriever(docs, FakeEncoder(), name="d").search("a", 2)) == 2


def test_shared_passage_id_returns_one_hit_with_best_score() -> None:
    docs = _docs(("p1", "bbbb"), ("p1", "aaaa"), ("p2", "cccc"))
    hits = DenseRetriever(docs, FakeEncoder(), name="d").search("a", 5)
    assert [h.passage_id for h in hits] == ["p1", "p2"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)


def test_empty_docs_raise() -> None:
    with pytest.raises(ValueError):
        DenseRetriever([], FakeEncoder(), name="d")


def test_docs_encoded_once_and_search_encodes_only_query() -> None:
    encoder = FakeEncoder()
    retriever = DenseRetriever(_docs(("p1", "aa"), ("p2", "bb")), encoder, name="d")
    assert encoder.passage_batches == [["aa", "bb"]]
    retriever.search("a", 1)
    retriever.search("b", 1)
    assert encoder.passage_batches == [["aa", "bb"]]
    assert encoder.query_batches == [["a"], ["b"]]


def test_satisfies_retriever_protocol() -> None:
    retriever: Retriever = DenseRetriever(_docs(("p1", "a")), FakeEncoder(), name="dense")
    assert retriever.name == "dense"


def test_fake_encoder_satisfies_encoder_protocol() -> None:
    encoder: Encoder = FakeEncoder()
    assert encoder.name == "fake"


def test_cached_returns_same_vectors_and_writes_one_file_per_text(tmp_path: Path) -> None:
    cached = CachedEncoder(FakeEncoder(), tmp_path)
    texts = ["ab", "bc", "ab"]
    out = cached.encode_passages(texts)
    np.testing.assert_allclose(out, FakeEncoder().encode_passages(texts))
    assert out.dtype == np.float32
    assert len(list(tmp_path.rglob("*.npy"))) == 2
    key = cache_key("fake", "passage", "ab")
    assert (tmp_path / key[:2] / f"{key}.npy").exists()


def test_second_call_sends_nothing_to_inner(tmp_path: Path) -> None:
    inner = FakeEncoder()
    cached = CachedEncoder(inner, tmp_path)
    cached.encode_queries(["ab", "bc"])
    cached.encode_queries(["ab", "bc"])
    assert inner.query_batches == [["ab", "bc"]]


def test_mixed_call_sends_only_missing_in_one_batch_and_keeps_order(tmp_path: Path) -> None:
    inner = FakeEncoder()
    cached = CachedEncoder(inner, tmp_path)
    cached.encode_queries(["ab"])
    out = cached.encode_queries(["cc", "ab", "dd"])
    assert inner.query_batches == [["ab"], ["cc", "dd"]]
    np.testing.assert_allclose(out, FakeEncoder().encode_queries(["cc", "ab", "dd"]))


def test_queries_and_passages_are_separate_entries(tmp_path: Path) -> None:
    inner = OffsetPassageEncoder()
    cached = CachedEncoder(inner, tmp_path)
    q = cached.encode_queries(["ab"])
    p = cached.encode_passages(["ab"])
    assert len(list(tmp_path.rglob("*.npy"))) == 2
    assert not np.allclose(q, p)
    assert inner.passage_batches == [["ab"]]


def test_empty_input_returns_no_rows_and_calls_nothing(tmp_path: Path) -> None:
    inner = FakeEncoder()
    cached = CachedEncoder(inner, tmp_path)
    assert cached.encode_queries([]).shape[0] == 0
    assert cached.encode_passages([]).shape[0] == 0
    assert inner.query_batches == [] and inner.passage_batches == []


def test_cached_encoder_name_matches_inner(tmp_path: Path) -> None:
    assert CachedEncoder(FakeEncoder(), tmp_path).name == "fake"


def test_encoders_registry_is_exact() -> None:
    assert set(ENCODERS) == {"bge-small-en", "me5-small"}
    bge, me5 = ENCODERS["bge-small-en"], ENCODERS["me5-small"]
    assert bge.key == "bge-small-en"
    assert bge.model_name == "BAAI/bge-small-en-v1.5"
    assert bge.query_prefix == "Represent this sentence for searching relevant passages: "
    assert bge.passage_prefix == ""
    assert me5.key == "me5-small"
    assert me5.model_name == "intfloat/multilingual-e5-small"
    assert me5.query_prefix == "query: "
    assert me5.passage_prefix == "passage: "


def test_apply_prefix_prepends_to_each_text() -> None:
    assert apply_prefix("query: ", ["a", "b"]) == ["query: a", "query: b"]
    assert apply_prefix("", ["a"]) == ["a"]


def test_sentence_transformer_encoder_construction_loads_nothing() -> None:
    encoder = SentenceTransformerEncoder(ENCODERS["me5-small"], device="cpu")
    assert encoder.name == "me5-small"
    assert encoder.is_loaded is False
