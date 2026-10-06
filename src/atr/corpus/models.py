import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

Language = Literal["en", "es"]

_WORD = re.compile(r"\w+")


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Passage(_Frozen):
    passage_id: str
    language: Language
    law_number: int | None
    law_title: str
    section_index: int | None
    section_title: str
    subheading: str | None
    source_slug: str
    text: str
    word_count: int

    @model_validator(mode="after")
    def _word_count_matches_text(self) -> Self:
        actual = count_words(self.text)
        if self.word_count != actual:
            raise ValueError(
                f"{self.passage_id}: word_count {self.word_count} but text has {actual}"
            )
        return self


class Alignment(_Frozen):
    pairs: list[tuple[str, str]]
    unaligned_en: list[str]
    unaligned_es: list[str]
    mismatches: list[str]

    @property
    def rate(self) -> float:
        """Share of distinct passage IDs, across both editions, that found a twin."""
        total = len(self.pairs) + len(self.unaligned_en) + len(self.unaligned_es)
        return len(self.pairs) / total if total else 0.0


class BuildReport(_Frozen):
    deleted_fragments: int = 0
    explanation_attrs: int = 0
    images: int = 0
    faq_entries_skipped: int = 0
    pages_skipped: list[str] = []
    empty_passages: list[str] = []

    def __add__(self, other: "BuildReport") -> "BuildReport":
        return BuildReport(
            deleted_fragments=self.deleted_fragments + other.deleted_fragments,
            explanation_attrs=self.explanation_attrs + other.explanation_attrs,
            images=self.images + other.images,
            faq_entries_skipped=self.faq_entries_skipped + other.faq_entries_skipped,
            pages_skipped=[*self.pages_skipped, *other.pages_skipped],
            empty_passages=[*self.empty_passages, *other.empty_passages],
        )
