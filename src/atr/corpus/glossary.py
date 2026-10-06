import re

from atr.corpus.html import clean_html
from atr.corpus.models import BuildReport, Language, Passage, count_words

_NON_SLUG = re.compile(r"[^a-z0-9]+")
_H4 = re.compile(r"<h4\b[^>]*>(.*?)</h4\s*>", re.IGNORECASE | re.DOTALL)
# The CMS sometimes types a new entry as a bold line and a <br> inside a paragraph
# instead of giving it an <h4>; without this it is swallowed by the entry above it.
_BOLD_TERM = re.compile(
    r"<p>\s*(?:<(?:code|em)>\s*)*<strong>(.*?)</strong>\s*(?:</(?:code|em)>\s*)*<br\s*/?>",
    re.IGNORECASE | re.DOTALL,
)
_ARROW = re.compile(r"\(\s*→\s*(.+?)\s*\)\s*$")
# "V." is the Spanish edition's "see"; a stub this short only points at another entry.
_CROSS_REFERENCE = re.compile(r"^(V\.|See)\s")
CROSS_REFERENCE_MAX_WORDS = 6
# Spanish entries IFAB published without the English cross-reference, joined by hand.
# Key: slug of the Spanish term. Value: slug of its English twin.
ES_SLUG_OVERRIDES = {
    "tecnolog-a-semiautomatizada-para-la-detecci-n-del-fuera-de-juego": (
        "semi-automated-offside-technology-saot"
    ),
}


def _is_cross_reference(text: str) -> bool:
    return bool(_CROSS_REFERENCE.match(text)) and count_words(text) <= CROSS_REFERENCE_MAX_WORDS


def slugify(term: str) -> str:
    return _NON_SLUG.sub("-", term.lower()).strip("-")


def _page_context(page_data: dict[str, object]) -> dict[str, object]:
    result = page_data.get("result")
    context = result.get("pageContext") if isinstance(result, dict) else None
    if not isinstance(context, dict):
        raise ValueError("missing result.pageContext")
    return context


def _html_contents(context: dict[str, object]) -> list[str]:
    blocks = context.get("contentBlocks")
    if not isinstance(blocks, list):
        raise ValueError("contentBlocks missing or not a list")
    contents: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("contentBlocks holds a non-object")
        html = block.get("htmlContent")
        if html is None:
            continue
        if not isinstance(html, str):
            raise ValueError("htmlContent is not a string")
        contents.append(html)
    return contents


def glossary_entries(page_data: dict[str, object]) -> list[tuple[str, str]]:
    html = _BOLD_TERM.sub(r"<h4>\1</h4><p>", "".join(_html_contents(_page_context(page_data))))
    headings = list(_H4.finditer(html))
    entries: list[tuple[str, str]] = []
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(html)
        entries.append((clean_html(match.group(1)).text, html[match.end() : end]))
    return entries


def _slugs(
    terms: list[str], language: Language, page: str, english_terms: list[str] | None
) -> list[tuple[str, str]]:
    """Return (displayed term, slug) per entry, following the D7 rules."""
    if language == "en":
        return [(term, slugify(term)) for term in terms]
    arrows = [_ARROW.search(term) for term in terms]
    if english_terms is not None and not any(arrows):
        if len(english_terms) != len(terms):
            raise ValueError(
                f"{page}: {len(terms)} Spanish entries but {len(english_terms)} English terms"
            )
        return [(term, slugify(en)) for term, en in zip(terms, english_terms, strict=True)]
    return [
        (term[: m.start()].rstrip(), slugify(m.group(1)))
        if m
        else (term, ES_SLUG_OVERRIDES.get(slugify(term), slugify(term)))
        for term, m in zip(terms, arrows, strict=True)
    ]


def _passage(
    passage_id: str, language: Language, title: str, term: str, slug: str, text: str
) -> Passage:
    return Passage(
        passage_id=passage_id,
        language=language,
        law_number=None,
        law_title=title,
        section_index=None,
        section_title=term,
        subheading=None,
        source_slug=slug,
        text=text,
        word_count=count_words(text),
    )


def parse_glossary(
    page_data: dict[str, object],
    language: Language,
    page: str,
    english_terms: list[str] | None = None,
) -> tuple[list[Passage], BuildReport]:
    try:
        context = _page_context(page_data)
        entries = glossary_entries(page_data)
    except ValueError as error:
        raise ValueError(f"glossary page {page}: {error}") from error
    title = context.get("title")
    if not isinstance(title, str):
        raise ValueError(f"glossary page {page}: title missing or not a string")
    named = _slugs([term for term, _ in entries], language, page, english_terms)
    passages: list[Passage] = []
    empty: list[str] = []
    stubs: list[str] = []
    totals = [0, 0, 0]
    for (_, definition), (term, slug) in zip(entries, named, strict=True):
        cleaned = clean_html(definition)
        totals[0] += cleaned.deleted_fragments
        totals[1] += cleaned.images
        totals[2] += cleaned.explanation_attrs
        passage_id = f"G.{page}.{slug}"
        if not cleaned.text:
            empty.append(passage_id)
            continue
        if _is_cross_reference(cleaned.text):
            stubs.append(f"{passage_id} ({term})")
            continue
        passages.append(_passage(passage_id, language, title, term, slug, cleaned.text))
    report = BuildReport(
        deleted_fragments=totals[0],
        images=totals[1],
        explanation_attrs=totals[2],
        empty_passages=empty,
        cross_references_dropped=stubs,
    )
    return passages, report
