import pytest

from atr.corpus.glossary import glossary_entries, parse_glossary, slugify
from atr.corpus.models import count_words

PAGE = "zorb-terms"


def page_data(*blocks: dict[str, object], title: str = "Zorb glossary") -> dict[str, object]:
    return {"result": {"pageContext": {"title": title, "contentBlocks": list(blocks)}}}


def block(html: str) -> dict[str, object]:
    return {"htmlContent": html}


def entry(term: str, definition: str) -> str:
    return f"<h4>{term}</h4><p>{definition}</p>"


def test_slugify_mixed_case_and_separators() -> None:
    assert slugify("Quick  Zorb -- Pass") == "quick-zorb-pass"


def test_slugify_leading_trailing_punctuation() -> None:
    assert slugify("  ...Zorb!?  ") == "zorb"


def test_slugify_curly_quotes_and_brackets() -> None:
    assert slugify("“Zorb” [x] (y)") == "zorb-x-y"


def test_slugify_bracket_and_parenthesis_variants_match() -> None:
    assert slugify("Alpha beta [AB]") == slugify("Alpha beta (AB)")


def test_slugify_accented_letters_become_separators() -> None:
    assert slugify("Pelota") == "pelota"
    assert slugify("aéb") == "a-b"


def test_entries_skip_blocks_without_html_content() -> None:
    data = page_data({"title": "A"}, block(entry("Wib", "d1")), {"other": 1})
    assert glossary_entries(data) == [("Wib", "<p>d1</p>")]


def test_entries_join_blocks_in_order() -> None:
    data = page_data(block(entry("Wib", "d1")), {"x": 1}, block(entry("Wob", "d2")))
    assert [term for term, _ in glossary_entries(data)] == ["Wib", "Wob"]


def test_entries_clean_the_term() -> None:
    data = page_data(block("<h4><strong>Wib</strong>&nbsp;</h4><p>d</p>"))
    assert glossary_entries(data)[0][0] == "Wib"


def test_english_ids_use_page_and_slug() -> None:
    data = page_data(block(entry("Quick Wib [QW]", "Moves fast.")))
    passages, report = parse_glossary(data, "en", PAGE)
    p = passages[0]
    assert p.passage_id == "G.zorb-terms.quick-wib-qw"
    assert p.source_slug == "quick-wib-qw"
    assert p.section_title == "Quick Wib [QW]"
    assert p.language == "en"
    assert p.text == "Moves fast."
    assert p.word_count == count_words(p.text)
    assert report.empty_passages == []


def test_passage_fixed_fields() -> None:
    data = page_data(block(entry("Wib", "d")), title="Zorb glossary")
    p = parse_glossary(data, "en", PAGE)[0][0]
    assert p.law_number is None
    assert p.section_index is None
    assert p.subheading is None
    assert p.law_title == "Zorb glossary"


@pytest.mark.parametrize("arrow", ["→", "&rarr;", "&rarr;&nbsp;"])
def test_spanish_arrow_gives_english_slug_and_strips_annotation(arrow: str) -> None:
    data = page_data(block(entry(f"Wibo ({arrow} Quick Wib)", "d")))
    passages, _ = parse_glossary(data, "es", PAGE)
    assert passages[0].passage_id == "G.zorb-terms.quick-wib"
    assert passages[0].section_title == "Wibo"
    assert passages[0].source_slug == "quick-wib"


def test_spanish_term_without_arrow_on_arrow_page_uses_own_slug() -> None:
    html = entry("Wibo (→ Wib)", "a") + entry("Wobo", "b")
    passages, _ = parse_glossary(page_data(block(html)), "es", PAGE, ["Wib", "Wob"])
    assert [p.passage_id for p in passages] == ["G.zorb-terms.wib", "G.zorb-terms.wobo"]


def test_positional_matching_when_no_arrows() -> None:
    html = entry("Wibo", "a") + entry("Wobo", "b")
    passages, _ = parse_glossary(page_data(block(html)), "es", PAGE, ["Wib", "Wob"])
    assert [p.passage_id for p in passages] == ["G.zorb-terms.wib", "G.zorb-terms.wob"]
    assert [p.section_title for p in passages] == ["Wibo", "Wobo"]


def test_length_mismatch_raises_naming_page_and_counts() -> None:
    html = entry("Wibo", "a") + entry("Wobo", "b")
    with pytest.raises(ValueError, match=r"zorb-terms.*2.*3"):
        parse_glossary(page_data(block(html)), "es", PAGE, ["Wib", "Wob", "Wub"])


def test_empty_definition_goes_to_empty_passages() -> None:
    html = "<h4>Wib</h4>" + entry("Wob", "d")
    passages, report = parse_glossary(page_data(block(html)), "en", PAGE)
    assert [p.passage_id for p in passages] == ["G.zorb-terms.wob"]
    assert report.empty_passages == ["G.zorb-terms.wib"]


def test_two_spanish_terms_one_english_term_share_id() -> None:
    html = entry("Wibo (→ Wib)", "a") + entry("Wiba (→ Wib)", "b")
    passages, _ = parse_glossary(page_data(block(html)), "es", PAGE)
    assert [p.passage_id for p in passages] == ["G.zorb-terms.wib"] * 2


def test_report_sums_drop_counts() -> None:
    html = (
        "<h4>Wib</h4><p>a<deleted>x</deleted> <em data-explanation='r'>b</em></p>"
        "<h4>Wob</h4><p>c<deleted>y</deleted><img src='i'></p>"
    )
    _, report = parse_glossary(page_data(block(html)), "en", PAGE)
    assert report.deleted_fragments == 2
    assert report.explanation_attrs == 1
    assert report.images == 1
    assert report.faq_entries_skipped == 0


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"result": {}},
        {"result": {"pageContext": {"title": "t"}}},
        {"result": {"pageContext": {"title": "t", "contentBlocks": "no"}}},
        {"result": {"pageContext": {"title": "t", "contentBlocks": [{"htmlContent": 3}]}}},
    ],
)
def test_malformed_input_raises_naming_page(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=PAGE):
        parse_glossary(bad, "en", PAGE)


def test_cross_reference_stubs_are_dropped_and_listed() -> None:
    page = {
        "result": {
            "pageContext": {
                "title": "Términos",
                "contentBlocks": [
                    {
                        "htmlContent": "<h4>Zorbo (&rarr; zorb)</h4><p>Una cosa redonda.</p>"
                        "<h4>Zorbito (&rarr; zorb)</h4><p>V. zorbo.</p>"
                    }
                ],
            }
        }
    }
    passages, report = parse_glossary(page, "es", "football-terms")
    assert [p.section_title for p in passages] == ["Zorbo"]
    assert report.cross_references_dropped == ["G.football-terms.zorb (Zorbito)"]


def test_english_see_stub_is_dropped_but_a_long_definition_starting_with_see_is_kept() -> None:
    page = {
        "result": {
            "pageContext": {
                "title": "Terms",
                "contentBlocks": [
                    {
                        "htmlContent": "<h4>Spin-off</h4><p>See &lsquo;Zorb (spin-off)&rsquo;</p>"
                        "<h4>Watch</h4><p>See the zorb and track it until it stops.</p>"
                    }
                ],
            }
        }
    }
    passages, report = parse_glossary(page, "en", "football-terms")
    assert [p.section_title for p in passages] == ["Watch"]
    assert report.cross_references_dropped == ["G.football-terms.spin-off (Spin-off)"]


def test_bold_line_with_break_is_an_entry_boundary_when_the_cms_forgot_the_h4() -> None:
    page = {
        "result": {
            "pageContext": {
                "title": "Terms",
                "contentBlocks": [
                    {
                        "htmlContent": "<h4>Zorb</h4><p>A round thing.</p>"
                        "<p><code><em><strong>Quib (QB)</strong></em></code><br>"
                        "<code><em>A square thing.</em></code></p>"
                    }
                ],
            }
        }
    }
    passages, _ = parse_glossary(page, "en", "football-terms")
    assert [(p.passage_id, p.text) for p in passages] == [
        ("G.football-terms.zorb", "A round thing."),
        ("G.football-terms.quib-qb", "A square thing."),
    ]


def test_spanish_term_without_an_arrow_can_be_joined_by_a_recorded_override() -> None:
    page = {
        "result": {
            "pageContext": {
                "title": "Términos",
                "contentBlocks": [
                    {
                        "htmlContent": "<h4>Zorbo (&rarr; zorb)</h4><p>Una cosa redonda.</p>"
                        "<h4>Tecnología semiautomatizada para la detección del fuera de juego</h4>"
                        "<p>Una cosa nueva.</p>"
                    }
                ],
            }
        }
    }
    passages, _ = parse_glossary(page, "es", "football-terms")
    assert passages[1].passage_id == "G.football-terms.semi-automated-offside-technology-saot"
    assert passages[1].section_title.startswith("Tecnología")
