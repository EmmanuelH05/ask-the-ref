import hashlib

import pytest

from atr.corpus.models import Passage, count_words
from atr.retrieval.docs import build_docs, title_line, verify_passages


def passage(passage_id: str, text: str, *, law: int | None = 4, sub: str | None = None) -> Passage:
    return Passage(
        passage_id=passage_id,
        language="en",
        law_number=law,
        law_title="Zorb Gear" if law else "Zorb Terms",
        section_index=2 if law else None,
        section_title="Helmets" if law else "Wobble",
        subheading=sub,
        source_slug="helmets",
        text=text,
        word_count=count_words(text),
    )


def test_title_line_for_a_law_passage_with_and_without_subheading() -> None:
    assert title_line(passage("L4.2", "x")) == "Law 4: Zorb Gear > Helmets"
    assert title_line(passage("L4.2.1", "x", sub="Straps")) == "Law 4: Zorb Gear > Helmets > Straps"


def test_title_line_for_a_glossary_passage() -> None:
    assert title_line(passage("G.t.wobble", "x", law=None)) == "Zorb Terms > Wobble"


def test_build_docs_plain_is_one_doc_per_passage() -> None:
    docs = build_docs(
        [passage("L4.2", "Helmets are round.")], title_prefix=False, chunking="passage"
    )
    assert [(d.passage_id, d.language, d.text) for d in docs] == [
        ("L4.2", "en", "Helmets are round.")
    ]


def test_build_docs_prefixes_every_chunk_with_the_title() -> None:
    text = "Helmets must:\n\n- be round\n- be blue"
    docs = build_docs([passage("L4.2", text)], title_prefix=True, chunking="items")
    assert [d.text for d in docs] == [
        "Law 4: Zorb Gear > Helmets\nHelmets must:\n- be round",
        "Law 4: Zorb Gear > Helmets\nHelmets must:\n- be blue",
    ]
    assert {d.passage_id for d in docs} == {"L4.2"}


def test_verify_passages_names_a_changed_passage() -> None:
    good = passage("L4.2", "Helmets are round.")
    pins = {("en", "L4.2"): hashlib.sha256(b"Helmets are round.").hexdigest()}
    verify_passages([good], pins)
    with pytest.raises(ValueError, match=r"L4\.2"):
        verify_passages([passage("L4.2", "Helmets are square.")], pins)
