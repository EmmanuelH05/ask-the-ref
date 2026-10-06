from pathlib import Path
from typing import NamedTuple, Protocol

from atr.corpus.models import Language, Passage


class Hit(NamedTuple):
    passage_id: str
    score: float


class Doc(NamedTuple):
    """One indexed unit. Several docs may share a passage id (two editions, or chunks)."""

    passage_id: str
    language: Language
    text: str


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...


def top_hits(hits: list[Hit], k: int) -> list[Hit]:
    """Best score per passage id, highest first, ties broken by id so runs are repeatable."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    best: dict[str, float] = {}
    for passage_id, score in hits:
        if passage_id not in best or score > best[passage_id]:
            best[passage_id] = score
    ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    return [Hit(passage_id, score) for passage_id, score in ranked[:k]]


def load_docs(corpus_dir: Path, languages: tuple[Language, ...]) -> list[Doc]:
    docs: list[Doc] = []
    for language in languages:
        lines = (corpus_dir / f"{language}.jsonl").read_text(encoding="utf-8").splitlines()
        for passage in map(Passage.model_validate_json, lines):
            docs.append(Doc(passage.passage_id, passage.language, passage.text))
    return docs
