import math
from collections import Counter

from atr.retrieval.base import Doc, Hit, top_hits
from atr.text import search_tokens


class BM25Retriever:
    """Okapi BM25 over stemmed tokens; the postings are built once in the constructor."""

    def __init__(
        self, docs: list[Doc], *, k1: float = 1.5, b: float = 0.75, name: str = "bm25"
    ) -> None:
        if not docs:
            raise ValueError("BM25Retriever needs at least one doc")
        self.name = name
        self._k1 = k1
        self._b = b
        self._passage_ids = [d.passage_id for d in docs]
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for index, d in enumerate(docs):
            tokens = search_tokens(d.text)
            self._lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self._postings.setdefault(term, []).append((index, tf))
        self._avgdl = sum(self._lengths) / len(docs)

    def _idf(self, df: int) -> float:
        n = len(self._passage_ids)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[Hit]:
        scores: dict[int, float] = {}
        for term in sorted(set(search_tokens(query))):
            postings = self._postings.get(term, [])
            idf = self._idf(len(postings))
            for index, tf in postings:
                length_norm = 1 - self._b + self._b * self._lengths[index] / self._avgdl
                gain = idf * tf * (self._k1 + 1) / (tf + self._k1 * length_norm)
                scores[index] = scores.get(index, 0.0) + gain
        hits = [Hit(self._passage_ids[i], s) for i, s in scores.items() if s > 0]
        return top_hits(hits, k)
