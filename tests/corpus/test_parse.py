import pytest

from atr.corpus.parse import parse_law


def section(
    content: str,
    index: str | None = "1",
    title: str = "Scoring",
    slug: str = "scoring",
) -> dict[str, object]:
    return {
        "sectionIndex": index,
        "lawSectionCategoryName": title,
        "slug": slug,
        "content": content,
    }


def page(
    sections: list[dict[str, object]],
    faq: object = None,
    law: int = 4,
    include_faq: bool = True,
) -> dict[str, object]:
    context: dict[str, object] = {
        "chapterIndex": law,
        "chapter": {"title": "The Glorp Ball", "sections": sections},
    }
    if include_faq:
        context["questionsAndAnswers"] = faq
    return {"result": {"pageContext": context}}


def test_section_without_h3_gives_one_passage() -> None:
    passages, _ = parse_law(page([section("<p>Zorbs hop twice.</p>", "3")]), "en")

    assert len(passages) == 1
    only = passages[0]
    assert only.passage_id == "L4.3"
    assert only.section_index == 3
    assert only.subheading is None
    assert only.text == "Zorbs hop twice."


def test_null_section_index_is_the_introduction() -> None:
    passages, _ = parse_law(page([section("<p>Welcome.</p>", None)]), "en")

    assert passages[0].passage_id == "L4.0"
    assert passages[0].section_index == 0


def test_lead_and_two_h3_blocks_are_numbered_from_zero() -> None:
    html = "<p>Lead.</p><h3>Alpha</h3><p>One.</p><h3>Beta</h3><p>Two.</p>"

    passages, _ = parse_law(page([section(html, "2")]), "en")

    assert [p.passage_id for p in passages] == ["L4.2.0", "L4.2.1", "L4.2.2"]
    assert [p.subheading for p in passages] == [None, "Alpha", "Beta"]
    assert [p.text for p in passages] == ["Lead.", "One.", "Two."]


def test_h3_without_lead_starts_at_one() -> None:
    passages, _ = parse_law(page([section("<h3>Alpha</h3><p>One.</p>", "2")]), "en")

    assert [p.passage_id for p in passages] == ["L4.2.1"]
    assert passages[0].subheading == "Alpha"


def test_image_only_section_is_empty_but_images_counted() -> None:
    passages, report = parse_law(page([section("<img src='x.png'>", "5")]), "en")

    assert passages == []
    assert report.empty_passages == ["L4.5"]
    assert report.images == 1


def test_empty_h3_block_still_consumes_its_number() -> None:
    html = "<h3>Alpha</h3><h3>Beta</h3><p>Two.</p>"

    passages, report = parse_law(page([section(html, "2")]), "en")

    assert [p.passage_id for p in passages] == ["L4.2.2"]
    assert report.empty_passages == ["L4.2.1"]


def test_report_sums_across_sections() -> None:
    first = section("<p>Keep <deleted>gone</deleted> this.</p>", "1")
    second = section(
        "<p data-explanation='a'>Hi</p><p data-explanation='b'>Yo <deleted>x</deleted></p>",
        "2",
    )

    _, report = parse_law(page([first, second]), "en")

    assert report.deleted_fragments == 2
    assert report.explanation_attrs == 2


@pytest.mark.parametrize(
    ("faq", "include", "expected"),
    [([{"q": 1}, {"q": 2}, {"q": 3}], True, 3), (None, True, 0), (None, False, 0)],
)
def test_faq_entries_skipped(faq: object, include: bool, expected: int) -> None:
    _, report = parse_law(page([section("<p>Hi.</p>")], faq=faq, include_faq=include), "en")

    assert report.faq_entries_skipped == expected


def test_metadata_is_carried_onto_every_passage() -> None:
    html = "<p>Lead.</p><h3>Alpha</h3><p>One.</p>"

    passages, _ = parse_law(page([section(html, "2", "Kickoff", "kickoff")], law=7), "es")

    assert len(passages) == 2
    for passage in passages:
        assert passage.law_number == 7
        assert passage.law_title == "The Glorp Ball"
        assert passage.section_title == "Kickoff"
        assert passage.source_slug == "kickoff"
        assert passage.language == "es"


def test_passages_come_out_in_document_order() -> None:
    sections = [
        section("<p>B.</p>", "2"),
        section("<p>A.</p>", None),
        section("<p>C.</p>", "1"),
    ]

    passages, _ = parse_law(page(sections), "en")

    assert [p.passage_id for p in passages] == ["L4.2", "L4.0", "L4.1"]


def test_word_count_matches_text() -> None:
    passages, _ = parse_law(page([section("<p>Three small words.</p>")]), "en")

    assert passages[0].word_count == 3


def test_missing_chapter_raises() -> None:
    bad: dict[str, object] = {"result": {"pageContext": {"chapterIndex": 4}}}

    with pytest.raises(ValueError, match="chapter"):
        parse_law(bad, "en")


def test_non_numeric_section_index_raises() -> None:
    with pytest.raises(ValueError, match="Law 4"):
        parse_law(page([section("<p>Hi.</p>", "abc")]), "en")


def test_duplicate_passage_id_raises() -> None:
    sections = [section("<p>A.</p>", "1"), section("<p>B.</p>", "1")]

    with pytest.raises(ValueError, match=r"L4\.1"):
        parse_law(page(sections), "en")
