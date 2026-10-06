# ruff: noqa: N815
# Field names mirror the CMS JSON keys, so they stay camelCase.
from pydantic import BaseModel, ConfigDict, ValidationError

from atr.corpus.html import clean_html, split_subsections
from atr.corpus.models import BuildReport, Language, Passage, count_words


# extra="ignore" is deliberate: the real JSON carries many fields we do not read.
class _Input(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _Section(_Input):
    sectionIndex: str | None
    lawSectionCategoryName: str
    slug: str
    content: str


class _Chapter(_Input):
    title: str
    sections: list[_Section]


class _PageContext(_Input):
    chapterIndex: int
    chapter: _Chapter
    questionsAndAnswers: list[object] | None = None


class _Result(_Input):
    pageContext: _PageContext


class _Page(_Input):
    result: _Result


def _law_label(page_data: dict[str, object]) -> str:
    result = page_data.get("result")
    context = result.get("pageContext") if isinstance(result, dict) else None
    number = context.get("chapterIndex") if isinstance(context, dict) else None
    return f"Law {number}" if number is not None else "Law (unknown number)"


def _section_index(section: _Section, label: str) -> int:
    if section.sectionIndex is None:
        return 0
    try:
        return int(section.sectionIndex)
    except ValueError:
        raise ValueError(
            f"{label}: sectionIndex {section.sectionIndex!r} is not a number"
        ) from None


def _block_ids(base: str, blocks: list[tuple[str | None, str]]) -> list[str]:
    if len(blocks) == 1 and blocks[0][0] is None:
        return [base]
    has_lead = blocks[0][0] is None
    # The lead is .0; h3 blocks always count from 1, with or without a lead.
    return [
        f"{base}.{position if has_lead else position + 1}" for position in range(len(blocks))
    ]


def _parse_section(
    section: _Section, context: _PageContext, language: Language, label: str
) -> tuple[list[Passage], list[str], list[str], BuildReport]:
    law = context.chapterIndex
    index = _section_index(section, label)
    blocks = split_subsections(section.content)
    ids = _block_ids(f"L{law}.{index}", blocks)
    passages: list[Passage] = []
    empty: list[str] = []
    counts = BuildReport()
    for passage_id, (heading, raw) in zip(ids, blocks, strict=True):
        cleaned = clean_html(raw)
        counts = counts + BuildReport(
            deleted_fragments=cleaned.deleted_fragments,
            images=cleaned.images,
            explanation_attrs=cleaned.explanation_attrs,
        )
        if not cleaned.text:
            empty.append(passage_id)
            continue
        passages.append(
            Passage(
                passage_id=passage_id,
                language=language,
                law_number=law,
                law_title=context.chapter.title,
                section_index=index,
                section_title=section.lawSectionCategoryName,
                subheading=heading,
                source_slug=section.slug,
                text=cleaned.text,
                word_count=count_words(cleaned.text),
            )
        )
    return passages, empty, ids, counts


def parse_law(
    page_data: dict[str, object], language: Language
) -> tuple[list[Passage], BuildReport]:
    label = _law_label(page_data)
    try:
        context = _Page.model_validate(page_data).result.pageContext
    except ValidationError as error:
        raise ValueError(f"{label}: malformed page data: {error}") from error

    passages: list[Passage] = []
    seen: set[str] = set()
    report = BuildReport(faq_entries_skipped=len(context.questionsAndAnswers or []))
    for section in context.chapter.sections:
        found, empty, ids, counts = _parse_section(section, context, language, label)
        for passage_id in ids:
            if passage_id in seen:
                raise ValueError(f"{label}: duplicate passage id {passage_id}")
            seen.add(passage_id)
        passages.extend(found)
        report = report + counts + BuildReport(empty_passages=empty)
    return passages, report
