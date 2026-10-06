import hashlib
from collections.abc import Mapping
from pathlib import Path

from atr.corpus.models import Language, Passage
from atr.retrieval.base import Doc
from atr.retrieval.chunking import ChunkStrategy, chunk_text


def load_passages(corpus_dir: Path, languages: tuple[Language, ...]) -> list[Passage]:
    passages: list[Passage] = []
    for language in languages:
        lines = (corpus_dir / f"{language}.jsonl").read_text(encoding="utf-8").splitlines()
        passages.extend(Passage.model_validate_json(line) for line in lines)
    return passages


def verify_passages(passages: list[Passage], pins: Mapping[tuple[str, str], str]) -> None:
    """Refuse passages whose text is not the text the corpus manifest pinned (D28)."""
    for passage in passages:
        expected = pins.get((passage.language, passage.passage_id))
        actual = hashlib.sha256(passage.text.encode("utf-8")).hexdigest()
        if expected != actual:
            state = "is not in the corpus manifest" if expected is None else "has changed"
            raise ValueError(
                f"{passage.language} passage {passage.passage_id} {state}; rebuild the corpus"
            )


def title_line(passage: Passage) -> str:
    if passage.law_number is None:
        return f"{passage.law_title} > {passage.section_title}"
    line = f"Law {passage.law_number}: {passage.law_title} > {passage.section_title}"
    return f"{line} > {passage.subheading}" if passage.subheading else line


def build_docs(
    passages: list[Passage], *, title_prefix: bool, chunking: ChunkStrategy
) -> list[Doc]:
    docs: list[Doc] = []
    for passage in passages:
        prefix = title_line(passage) + "\n" if title_prefix else ""
        for chunk in chunk_text(passage.text, chunking):
            docs.append(Doc(passage.passage_id, passage.language, prefix + chunk))
    return docs
