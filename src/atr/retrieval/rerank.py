"""Cross-encoder reranking: rescore (query, text) pairs jointly, then keep the best per passage."""

from typing import Any, Protocol

from atr.retrieval.base import Hit, top_hits


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


RERANKERS: dict[str, str] = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-m3": "BAAI/bge-reranker-v2-m3",
}


class CrossEncoderReranker:
    def __init__(
        self,
        key: str,
        *,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        if key not in RERANKERS:
            raise ValueError(f"unknown reranker {key!r}; choose from {sorted(RERANKERS)}")
        self._key = key
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        # Any: the model is imported lazily, so its type cannot be named at module level.
        self._model: Any = None
        self.name = key

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                RERANKERS[self._key], device=self._device, max_length=self._max_length
            )
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        model = self._load()
        scores = model.predict([(query, text) for text in texts], batch_size=self._batch_size)
        return [float(s) for s in scores]


def rerank(query: str, candidates: list[tuple[str, str]], reranker: Reranker, k: int) -> list[Hit]:
    """Score every (passage id, text) candidate in one call; keep the best score per passage."""
    if not candidates:
        return []
    scores = reranker.score(query, [text for _, text in candidates])
    hits = [Hit(pid, s) for (pid, _), s in zip(candidates, scores, strict=True)]
    return top_hits(hits, k)
