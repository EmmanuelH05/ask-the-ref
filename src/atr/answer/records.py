"""One answered question per record; every phase 7 table is rebuilt from these."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from atr.answer.generate import Claim
from atr.evalset.models import QuestionLanguage, QuestionType, Split

System = Literal["closed_book", "full", "oracle"]
SYSTEMS: tuple[System, ...] = ("closed_book", "full", "oracle")


class AnswerRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    group_id: str
    language: QuestionLanguage
    type: QuestionType
    split: Split
    system: System
    question: str
    gold: list[str]
    retrieved: list[str]
    top_score: float | None
    threshold: float | None
    refused: bool
    answer: str
    citations: list[str]
    claims: list[Claim]
    latency_ms: dict[str, float]


def load_answers(run_dir: Path) -> list[AnswerRecord]:
    path = run_dir / "records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [AnswerRecord.model_validate_json(line) for line in lines if line.strip()]


def records_sha256(run_dir: Path) -> str:
    """Hash of the records file, which stamps every file derived from it (judgments, sheet)."""
    return hashlib.sha256((run_dir / "records.jsonl").read_bytes()).hexdigest()


def write_answers(records: list[AnswerRecord], run_dir: Path) -> None:
    """Write the records; verdicts left from an earlier run of the same system are removed."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "records.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8"
    )
    stale = run_dir / "judgments.json"
    if stale.is_file():
        stale.unlink()
