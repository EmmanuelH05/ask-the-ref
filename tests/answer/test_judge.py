from pathlib import Path

import pytest

from atr.answer.generate import GENERATOR_MODEL, Claim
from atr.answer.judge import (
    JUDGE_MODEL,
    Judge,
    Verdict,
    count_undecided,
    labels_from_judgments,
    load_judgments,
    parse_verdict,
    write_judgments,
)
from atr.answer.records import AnswerRecord, write_answers
from atr.llm import Completion


class ScriptedLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, str | None]] = []

    def complete(self, model: str, prompt: str, *, think: str | None = None) -> Completion:
        self.calls.append((model, prompt, think))
        return Completion(self.reply, 80.0)


def test_parse_verdict_reads_the_first_word_in_either_language() -> None:
    assert parse_verdict("Yes.") is True
    assert parse_verdict("yes, the passage states it") is True
    assert parse_verdict("No") is False
    assert parse_verdict("NO - the passage says 11 m") is False
    assert parse_verdict("Sí") is True
    assert parse_verdict("Maybe") is None
    assert parse_verdict("") is None


def test_judge_uses_a_different_model_from_the_generator() -> None:
    assert JUDGE_MODEL != GENERATOR_MODEL


def test_judge_asks_once_per_claim_with_the_cited_passages() -> None:
    llm = ScriptedLLM("yes")
    judge = Judge(llm)
    verdict = judge.judge(Claim(text="The ball is round", citations=["L2.1"]), {"L2.1": "Round."})
    assert verdict == Verdict(supported=True, ms=80.0)
    model, prompt, think = llm.calls[0]
    assert model == JUDGE_MODEL and think == "low"
    assert "The ball is round" in prompt and "Round." in prompt


def test_judge_marks_an_uncited_claim_unsupported_without_a_call() -> None:
    llm = ScriptedLLM("yes")
    verdict = Judge(llm).judge(Claim(text="Bare", citations=[]), {"L2.1": "x"})
    assert verdict == Verdict(supported=False, ms=0.0)
    assert llm.calls == []


def test_judge_returns_none_for_an_unreadable_reply() -> None:
    llm = ScriptedLLM("It depends.")
    verdict = Judge(llm).judge(Claim(text="c", citations=["L2.1"]), {"L2.1": "x"})
    assert verdict.supported is None


def _answer(question_id: str) -> AnswerRecord:
    return AnswerRecord(
        question_id=question_id,
        group_id=question_id.split(".")[0],
        language="en",
        type="plain",
        split="dev",
        system="full",
        question="q?",
        gold=["L2.1"],
        retrieved=["L2.1"],
        top_score=0.9,
        threshold=0.5,
        refused=False,
        answer="The ball is round [L2.1].",
        citations=["L2.1"],
        claims=[Claim(text="The ball is round", citations=["L2.1"])],
        latency_ms={},
    )


def test_parse_verdict_reads_through_markdown_and_labels() -> None:
    assert parse_verdict("**Yes**") is True
    assert parse_verdict("Yes:") is True
    assert parse_verdict("Answer: no") is False
    assert parse_verdict("Supported: yes.") is True
    assert parse_verdict("Not supported") is None


def test_judge_refuses_a_cited_id_missing_from_the_corpus() -> None:
    with pytest.raises(KeyError):
        Judge(ScriptedLLM("yes")).judge(Claim(text="x", citations=["L9.9"]), {"L2.1": "Round."})


def test_labels_from_judgments_counts_undecided_as_unsupported() -> None:
    judgments = {"A": [True, None, False]}
    assert labels_from_judgments(judgments) == {("A", 0): True, ("A", 1): False, ("A", 2): False}
    assert count_undecided(judgments) == 1


def test_judgments_round_trip_and_refuse_records_they_did_not_judge(tmp_path: Path) -> None:
    assert load_judgments(tmp_path) is None
    write_answers([_answer("A.en")], tmp_path)
    write_judgments(tmp_path, {"A.en": [True, None]})
    assert load_judgments(tmp_path) == {"A.en": [True, None]}
    write_answers([_answer("B.en")], tmp_path)  # new answers remove the stale verdicts
    assert load_judgments(tmp_path) is None
    write_judgments(tmp_path, {"B.en": [False]})
    (tmp_path / "records.jsonl").write_text("edited\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rerun the judge"):
        load_judgments(tmp_path)
