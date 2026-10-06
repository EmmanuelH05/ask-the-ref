from pathlib import Path

import pytest

from atr.corpus.models import Passage, count_words
from atr.retrieval.base import Doc, Hit, load_docs, top_hits, verify_docs


def test_top_hits_keeps_the_best_score_per_passage_and_orders_by_score_then_id() -> None:
    hits = [Hit("L1.2", 0.5), Hit("L1.1", 0.9), Hit("L1.2", 0.7), Hit("L1.0", 0.7)]
    assert top_hits(hits, 2) == [Hit("L1.1", 0.9), Hit("L1.0", 0.7)]
    assert top_hits(hits, 10)[-1] == Hit("L1.2", 0.7)


def test_top_hits_rejects_a_non_positive_k() -> None:
    with pytest.raises(ValueError):
        top_hits([], 0)


def test_load_docs_reads_the_requested_editions_in_order(tmp_path: Path) -> None:
    for language, text in (("en", "blue wombats"), ("es", "wombats azules")):
        passage = Passage(
            passage_id="L1.1",
            language=language,  # type: ignore[arg-type]
            law_number=1,
            law_title="Invented",
            section_index=1,
            section_title="Invented",
            subheading=None,
            source_slug="invented",
            text=text,
            word_count=count_words(text),
        )
        (tmp_path / f"{language}.jsonl").write_text(passage.model_dump_json() + "\n")
    assert load_docs(tmp_path, ("en",)) == [Doc("L1.1", "en", "blue wombats")]
    assert [d.language for d in load_docs(tmp_path, ("en", "es"))] == ["en", "es"]


def test_verify_docs_accepts_matching_text_and_names_a_changed_or_unknown_passage() -> None:
    import hashlib

    good = Doc("L1.1", "en", "blue wombats")
    pins = {("en", "L1.1"): hashlib.sha256(b"blue wombats").hexdigest()}
    verify_docs([good], pins)
    with pytest.raises(ValueError, match=r"L1\.1"):
        verify_docs([Doc("L1.1", "en", "red wombats")], pins)
    with pytest.raises(ValueError, match=r"L9\.9"):
        verify_docs([Doc("L9.9", "en", "blue wombats")], pins)
