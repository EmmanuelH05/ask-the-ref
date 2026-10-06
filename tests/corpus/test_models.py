import pytest
from pydantic import ValidationError

from atr.corpus.models import Alignment, BuildReport, Passage, count_words


def make_passage(**overrides: object) -> Passage:
    fields: dict[str, object] = {
        "passage_id": "L11.1",
        "language": "en",
        "law_number": 11,
        "law_title": "Invented Law",
        "section_index": 1,
        "section_title": "Invented section",
        "subheading": None,
        "source_slug": "invented-section",
        "text": "One two three.",
        "word_count": 3,
    }
    return Passage.model_validate({**fields, **overrides})


def test_count_words_counts_word_runs_not_punctuation() -> None:
    assert count_words("- one, two\n\nthree (4 m)") == 5


def test_count_words_counts_accented_spanish_words_once() -> None:
    assert count_words("árbitro señaló") == 2


def test_passage_is_frozen() -> None:
    passage = make_passage()
    with pytest.raises(ValidationError):
        passage.text = "changed"  # type: ignore[misc]


def test_passage_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        make_passage(extra_field="x")


def test_passage_rejects_word_count_that_disagrees_with_text() -> None:
    with pytest.raises(ValidationError):
        make_passage(word_count=99)


def test_passage_rejects_unknown_language() -> None:
    with pytest.raises(ValidationError):
        make_passage(language="fr")


def test_alignment_rate_is_pairs_over_all_ids() -> None:
    alignment = Alignment(
        pairs=[("L1.1", "L1.1"), ("L1.2", "L1.2")],
        unaligned_en=["L1.3"],
        unaligned_es=[],
        mismatches=[],
    )
    assert alignment.rate == pytest.approx(2 / 3)


def test_alignment_rate_is_zero_when_empty() -> None:
    assert Alignment(pairs=[], unaligned_en=[], unaligned_es=[], mismatches=[]).rate == 0.0


def test_build_report_adds_counts_and_keeps_skipped_pages() -> None:
    first = BuildReport(
        deleted_fragments=1,
        explanation_attrs=2,
        images=3,
        faq_entries_skipped=4,
        pages_skipped=["a"],
    )
    second = BuildReport(
        deleted_fragments=10,
        explanation_attrs=20,
        images=30,
        faq_entries_skipped=40,
        pages_skipped=["b"],
    )
    assert first + second == BuildReport(
        deleted_fragments=11,
        explanation_attrs=22,
        images=33,
        faq_entries_skipped=44,
        pages_skipped=["a", "b"],
    )
