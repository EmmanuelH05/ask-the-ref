import re
from typing import Literal

import pytest

from atr.corpus.align import align
from atr.corpus.models import Passage


def make(
    pid: str,
    lang: Literal["en", "es"],
    *,
    slug: str = "zorp",
    law: int | None = 1,
    section: int | None = 1,
    text: str = "blue wombats juggle quietly",
) -> Passage:
    return Passage(
        passage_id=pid,
        language=lang,
        law_number=law,
        law_title="Invented Title",
        section_index=section,
        section_title="Invented Section",
        subheading=None,
        source_slug=slug,
        text=text,
        word_count=len(re.findall(r"\w+", text)),
    )


def test_matching_ids_are_paired_in_en_order() -> None:
    en = [make("L1.2", "en"), make("L1.1", "en")]
    es = [make("L1.1", "es"), make("L1.2", "es")]
    result = align(en, es)
    assert result.pairs == [("L1.2", "L1.2"), ("L1.1", "L1.1")]
    assert result.unaligned_en == []
    assert result.unaligned_es == []
    assert result.mismatches == []


def test_id_only_in_one_edition_has_no_mismatch_line() -> None:
    en = [make("L1.1", "en"), make("L1.2", "en")]
    es = [make("L1.1", "es"), make("L1.3", "es")]
    result = align(en, es)
    assert result.pairs == [("L1.1", "L1.1")]
    assert result.unaligned_en == ["L1.2"]
    assert result.unaligned_es == ["L1.3"]
    assert result.mismatches == []


def test_source_slug_mismatch_is_not_paired() -> None:
    result = align([make("L1.1", "en", slug="aaa")], [make("L1.1", "es", slug="bbb")])
    assert result.pairs == []
    assert result.unaligned_en == ["L1.1"]
    assert result.unaligned_es == ["L1.1"]
    assert len(result.mismatches) == 1
    line = result.mismatches[0]
    assert "L1.1" in line
    assert "source_slug" in line
    assert "aaa" in line
    assert "bbb" in line


def test_law_number_and_section_index_mismatches_are_not_paired() -> None:
    en = [make("L1.1", "en", law=1), make("L2.1", "en", section=1)]
    es = [make("L1.1", "es", law=2), make("L2.1", "es", section=None)]
    result = align(en, es)
    assert result.pairs == []
    assert result.unaligned_en == ["L1.1", "L2.1"]
    assert result.unaligned_es == ["L1.1", "L2.1"]
    assert len(result.mismatches) == 2
    assert "law_number" in result.mismatches[0]
    assert "1" in result.mismatches[0]
    assert "2" in result.mismatches[0]
    assert "section_index" in result.mismatches[1]
    assert "None" in result.mismatches[1]


def test_duplicate_in_en_is_never_paired() -> None:
    en = [make("L1.1", "en"), make("L1.1", "en"), make("L1.2", "en")]
    es = [make("L1.1", "es"), make("L1.2", "es")]
    result = align(en, es)
    assert result.pairs == [("L1.2", "L1.2")]
    assert result.unaligned_en == ["L1.1"]
    assert result.unaligned_es == ["L1.1"]
    assert len(result.mismatches) == 1
    assert "L1.1" in result.mismatches[0]
    assert "duplicated in en" in result.mismatches[0]


def test_duplicate_in_es_only_lists_once_and_names_language() -> None:
    en = [make("L1.1", "en")]
    es = [make("L1.1", "es"), make("L1.1", "es")]
    result = align(en, es)
    assert result.pairs == []
    assert result.unaligned_en == ["L1.1"]
    assert result.unaligned_es == ["L1.1"]
    assert len(result.mismatches) == 1
    assert "duplicated in es" in result.mismatches[0]


def test_duplicate_without_twin_is_listed_once_in_its_edition() -> None:
    es = [make("L1.1", "es"), make("L1.1", "es")]
    result = align([], es)
    assert result.unaligned_en == []
    assert result.unaligned_es == ["L1.1"]
    assert len(result.mismatches) == 1
    assert "duplicated in es" in result.mismatches[0]


def test_duplicate_in_both_editions_gets_a_line_per_language() -> None:
    en = [make("L1.1", "en"), make("L1.1", "en")]
    es = [make("L1.1", "es"), make("L1.1", "es")]
    result = align(en, es)
    assert result.unaligned_en == ["L1.1"]
    assert result.unaligned_es == ["L1.1"]
    assert len(result.mismatches) == 2
    assert any("duplicated in en" in m for m in result.mismatches)
    assert any("duplicated in es" in m for m in result.mismatches)


def test_unaligned_orders_follow_each_edition() -> None:
    en = [make("L3.1", "en"), make("L1.1", "en"), make("L2.1", "en")]
    es = [make("L2.1", "es", law=9), make("L5.1", "es"), make("L0.1", "es")]
    result = align(en, es)
    assert result.unaligned_en == ["L3.1", "L1.1", "L2.1"]
    assert result.unaligned_es == ["L2.1", "L5.1", "L0.1"]


def test_wrong_language_raises_value_error() -> None:
    with pytest.raises(ValueError):
        align([make("L1.1", "es")], [])
    with pytest.raises(ValueError):
        align([], [make("L1.1", "en")])


def test_empty_inputs() -> None:
    result = align([], [])
    assert result.pairs == []
    assert result.unaligned_en == []
    assert result.unaligned_es == []
    assert result.mismatches == []
