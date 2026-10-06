"""Dense retrieval: encode texts to normalized vectors and rank by dot product."""

from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from atr.cache import cache_key
from atr.retrieval.base import Doc, Hit, top_hits

Kind = Literal["query", "passage"]


class Encoder(Protocol):
    # Identifies the vector space: the model and how texts are prefixed. Cache keys use it.
    fingerprint: str
    name: str

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]: ...

    def encode_passages(self, texts: list[str]) -> NDArray[np.float32]: ...


class EncoderSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    model_name: str
    query_prefix: str
    passage_prefix: str


ENCODERS: dict[str, EncoderSpec] = {
    "bge-small-en": EncoderSpec(
        key="bge-small-en",
        model_name="BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        passage_prefix="",
    ),
    "me5-small": EncoderSpec(
        key="me5-small",
        model_name="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
}


def apply_prefix(prefix: str, texts: list[str]) -> list[str]:
    return [prefix + text for text in texts]


class SentenceTransformerEncoder:
    def __init__(self, spec: EncoderSpec, *, device: str | None = None) -> None:
        self._spec = spec
        self._device = device
        # Any: the model is imported lazily, so its type cannot be named at module level.
        self._model: Any = None
        self.name = spec.key
        self.fingerprint = f"{spec.model_name}|{spec.query_prefix}|{spec.passage_prefix}"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._spec.model_name, device=self._device)
        return self._model

    def _encode(self, prefix: str, texts: list[str]) -> NDArray[np.float32]:
        model = self._load()
        vectors = model.encode(apply_prefix(prefix, texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        return self._encode(self._spec.query_prefix, texts)

    def encode_passages(self, texts: list[str]) -> NDArray[np.float32]:
        return self._encode(self._spec.passage_prefix, texts)


class CachedEncoder:
    def __init__(self, inner: Encoder, cache_dir: Path) -> None:
        self._inner = inner
        self._dir = cache_dir
        self.name = inner.name
        self.fingerprint = inner.fingerprint

    def _path(self, kind: Kind, text: str) -> Path:
        key = cache_key(self._inner.fingerprint, kind, text)
        return self._dir / key[:2] / f"{key}.npy"

    def _encode(self, kind: Kind, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        found: dict[str, NDArray[np.float32]] = {}
        missing = [t for t in dict.fromkeys(texts) if not self._path(kind, t).exists()]
        if missing:
            encode = self._inner.encode_queries if kind == "query" else self._inner.encode_passages
            for text, vector in zip(missing, encode(missing), strict=True):
                path = self._path(kind, text)
                path.parent.mkdir(parents=True, exist_ok=True)
                # Write then rename, so an interrupted run never leaves a half-written entry.
                temporary = path.with_suffix(".tmp.npy")
                np.save(temporary, vector)
                temporary.replace(path)
                found[text] = vector
        for text in texts:
            if text not in found:
                found[text] = np.load(self._path(kind, text))
        return np.stack([found[text] for text in texts]).astype(np.float32)

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        return self._encode("query", texts)

    def encode_passages(self, texts: list[str]) -> NDArray[np.float32]:
        return self._encode("passage", texts)


class DenseRetriever:
    def __init__(self, docs: list[Doc], encoder: Encoder, *, name: str) -> None:
        if not docs:
            raise ValueError("DenseRetriever needs at least one doc")
        self.name = name
        self._docs = docs
        self._encoder = encoder
        self._matrix = encoder.encode_passages([doc.text for doc in docs])

    def search(self, query: str, k: int) -> list[Hit]:
        vector = self._encoder.encode_queries([query])[0]
        scores = self._matrix @ vector
        hits = [Hit(doc.passage_id, float(s)) for doc, s in zip(self._docs, scores, strict=True)]
        return top_hits(hits, k)
