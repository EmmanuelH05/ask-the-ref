import re
from typing import Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, model_validator

QuestionType = Literal[
    "plain", "paraphrase", "scenario", "multi_rule", "false_premise", "out_of_scope"
]
# "cs" is intrasentential Spanish-English code-switching.
QuestionLanguage = Literal["en", "es", "cs"]
Split = Literal["dev", "test", "heldout"]
Expected = Literal["answer", "refuse", "correct_premise"]

QUESTION_TYPES: tuple[QuestionType, ...] = get_args(QuestionType)
QUESTION_LANGUAGES: tuple[QuestionLanguage, ...] = get_args(QuestionLanguage)
# Whole laws kept out of every training and tuning step (DECISIONS.md D11).
HELD_OUT_LAWS = (3, 5, 16)

_PASSAGE_ID = re.compile(r"^(L\d+\.\d+(\.\d+)?|G\.[a-z-]+\.[a-z0-9-]+)$")
_LAW = re.compile(r"^L(\d+)\.")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class QuestionGroup(_Frozen):
    """One question in three languages. The versions share gold passages and a split."""

    group_id: str
    type: QuestionType
    gold: list[str]
    en: str
    es: str
    cs: str
    answer: str
    note: str
    spanish_reviewed: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        for language in QUESTION_LANGUAGES:
            if not getattr(self, language).strip():
                raise ValueError(f"{self.group_id}: {language} text is blank")
        if len(set(self.gold)) != len(self.gold):
            raise ValueError(f"{self.group_id}: duplicate gold passage")
        bad = [passage_id for passage_id in self.gold if not _PASSAGE_ID.match(passage_id)]
        if bad:
            raise ValueError(f"{self.group_id}: not a passage id: {bad}")
        if self.type == "out_of_scope" and self.gold:
            raise ValueError(f"{self.group_id}: out_of_scope must have no gold passage")
        if self.type == "multi_rule" and len(self.gold) < 2:
            raise ValueError(f"{self.group_id}: multi_rule needs at least two gold passages")
        if self.type in ("plain", "paraphrase", "scenario") and not self.gold:
            raise ValueError(f"{self.group_id}: {self.type} needs a gold passage")
        return self

    @property
    def expected(self) -> Expected:
        if self.type == "out_of_scope":
            return "refuse"
        if self.type == "false_premise":
            return "correct_premise"
        return "answer"


class Question(_Frozen):
    question_id: str
    group_id: str
    language: QuestionLanguage
    type: QuestionType
    text: str
    gold: list[str]


def gold_laws(gold: list[str]) -> set[int]:
    return {int(match.group(1)) for passage_id in gold if (match := _LAW.match(passage_id))}


def is_held_out(group: QuestionGroup) -> bool:
    return bool(gold_laws(group.gold) & set(HELD_OUT_LAWS))


def expand(group: QuestionGroup) -> list[Question]:
    return [
        Question(
            question_id=f"{group.group_id}.{language}",
            group_id=group.group_id,
            language=language,
            type=group.type,
            text=getattr(group, language),
            gold=list(group.gold),
        )
        for language in QUESTION_LANGUAGES
    ]
