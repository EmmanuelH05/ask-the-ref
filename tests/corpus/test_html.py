import pytest
from pydantic import ValidationError

from atr.corpus.html import Cleaned, clean_html, split_subsections


def test_paragraphs_separated_by_one_blank_line() -> None:
    assert clean_html("<p>Zorp one.</p><p>Zorp two.</p>\n<p>Zorp three.</p>").text == (
        "Zorp one.\n\nZorp two.\n\nZorp three."
    )


def test_list_items_become_dash_lines() -> None:
    assert clean_html("<ul><li>Gleb</li><li>Wump</li></ul>").text == "- Gleb\n- Wump"


def test_nested_list_indents_two_spaces_per_level() -> None:
    html = "<ul><li>Gleb<ul><li>Wump<ul><li>Frin</li></ul></li></ul></li><li>Dax</li></ul>"
    assert clean_html(html).text == "- Gleb\n  - Wump\n    - Frin\n- Dax"


def test_list_after_paragraph_gets_blank_line() -> None:
    assert clean_html("<p>Intro</p><ul><li>A</li></ul><p>Outro</p>").text == (
        "Intro\n\n- A\n\nOutro"
    )


def test_br_is_newline() -> None:
    assert clean_html("<p>Line one<br>Line two<br/>Line three</p>").text == (
        "Line one\nLine two\nLine three"
    )


def test_entities_unescaped_and_nbsp_to_space() -> None:
    assert clean_html("<p>The gleb&rsquo;s&nbsp;turn &rarr; next\xa0round</p>").text == (
        "The gleb" + chr(0x2019) + "s turn " + chr(0x2192) + " next round"
    )


def test_space_runs_collapse_and_ends_stripped() -> None:
    assert clean_html("  <p>  a   b \n\n  c  </p>  ").text == "a b c"


@pytest.mark.parametrize("tag", ["deleted", "deletedexplanation"])
def test_deleted_content_dropped_and_counted(tag: str) -> None:
    result = clean_html(f"<p>Keep <{tag}>gone <strong>deep</strong></{tag}>this.</p>")
    assert result.text == "Keep this."
    assert result.deleted_fragments == 1


def test_nested_deleted_counted_once() -> None:
    html = (
        "<p>a<deleted>x<deletedexplanation>y<deleted>z</deleted>"
        "</deletedexplanation></deleted>b</p>"
    )
    result = clean_html(html)
    assert result.text == "ab"
    assert result.deleted_fragments == 1


def test_separate_deleted_elements_each_counted() -> None:
    result = clean_html("<p><deleted>a</deleted>k<deletedexplanation>b</deletedexplanation></p>")
    assert result.deleted_fragments == 2
    assert result.text == "k"


@pytest.mark.parametrize("tag", ["added", "addedexplanation", "explanation", "strong", "em"])
def test_kept_inline_tags(tag: str) -> None:
    result = clean_html(f"<p>Pre <{tag}>kept words</{tag}> post</p>")
    assert result.text == "Pre kept words post"
    assert result.deleted_fragments == 0


def test_data_explanation_counted_and_never_in_text() -> None:
    html = (
        '<p data-explanation="x1">A <em data-explanation="x2">b</em> '
        '<deleted data-explanation="x3">c</deleted></p>'
    )
    result = clean_html(html)
    assert result.explanation_attrs == 3
    assert "x1" not in result.text
    assert "data-explanation" not in result.text
    assert result.text == "A b"


@pytest.mark.parametrize(
    "image",
    [
        '<ResponsiveImage src="a.png"/>',
        "<ResponsiveImage></ResponsiveImage>",
        '<TheFieldInteractivePicture src="a"/>',
        "<TheFieldInteractivePicture></TheFieldInteractivePicture>",
        '<img src="a.png">',
        '<img src="a.png"/>',
    ],
)
def test_each_image_kind_counted_once_and_dropped(image: str) -> None:
    result = clean_html(f"<p>Before {image} after</p>")
    assert result.images == 1
    assert result.text == "Before after"


def test_h3_rendered_on_own_paragraph() -> None:
    assert clean_html("<p>Intro</p><h3>Head</h3><p>Body</p>").text == "Intro\n\nHead\n\nBody"


def test_cleaned_is_frozen_and_forbids_extra() -> None:
    result = clean_html("<p>a</p>")
    with pytest.raises(ValidationError):
        result.text = "b"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Cleaned(text="a", deleted_fragments=0, images=0, explanation_attrs=0, extra=1)  # type: ignore[call-arg]


def test_split_with_lead_text() -> None:
    html = "<p>Lead</p><h3>One</h3><p>A</p><h3><em>Two</em> </h3><p>B</p>"
    assert split_subsections(html) == [
        (None, "<p>Lead</p>"),
        ("One", "<p>A</p>"),
        ("Two", "<p>B</p>"),
    ]


def test_split_without_lead_text() -> None:
    html = "<div></div><h3>One</h3><p>A</p>"
    assert split_subsections(html) == [("One", "<p>A</p>")]


def test_split_with_no_h3_returns_whole_html() -> None:
    html = "<p>Only</p>"
    assert split_subsections(html) == [(None, html)]


def test_split_whitespace_only_lead_is_skipped() -> None:
    html = " \n<p>&nbsp;\xa0</p> <h3>One</h3><p>A</p>"
    assert split_subsections(html) == [("One", "<p>A</p>")]


def test_split_heading_text_is_cleaned() -> None:
    expected_title = "The gleb" + chr(0x2019) + "s"
    assert split_subsections("<h3>The&nbsp;gleb&rsquo;s</h3>x") == [(expected_title, "x")]


def test_paragraph_inside_list_item_keeps_the_bullet() -> None:
    html = "<ul>\n <li> <p>red zorbs</p> </li>\n <li> <p>blue zorbs</p> </li>\n</ul>"
    assert clean_html(html).text == "- red zorbs\n- blue zorbs"


def test_list_with_paragraph_items_sits_between_paragraphs() -> None:
    html = "<p>Counts if:</p><ul><li><p>it spins</p></li><li><p>it hums</p></li></ul><p>End.</p>"
    assert clean_html(html).text == "Counts if:\n\n- it spins\n- it hums\n\nEnd."


def test_nested_list_with_paragraph_items_is_indented() -> None:
    html = "<ul><li><p>outer</p><ul><li><p>inner</p></li></ul></li></ul>"
    assert clean_html(html).text == "- outer\n  - inner"


def test_second_paragraph_in_a_list_item_is_indented_under_its_bullet() -> None:
    assert clean_html("<ul><li><p>one</p><p>two</p></li></ul>").text == "- one\n  two"


def test_text_after_a_nested_list_returns_to_the_outer_item() -> None:
    html = "<ul><li><p>a</p><ul><li><p>b</p></li></ul><p>c</p></li><li><p>d</p></li></ul>"
    assert clean_html(html).text == "- a\n  - b\n  c\n- d"
