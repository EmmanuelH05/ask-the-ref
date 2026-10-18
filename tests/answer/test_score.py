import pytest

from atr.answer.generate import Claim
from atr.answer.records import AnswerRecord
from atr.answer.score import (
    all_gold_cited,
    citation_correct,
    error_split,
    refusal_correct,
    summarize_answers,
    supported_rate,
)


def rec(
    question_id: str,
    kind: str,
    gold: list[str],
    citations: list[str],
    *,
    refused: bool = False,
    lang: str = "en",
    claims: int = 2,
) -> AnswerRecord:
    return AnswerRecord(
        question_id=question_id,
        group_id=question_id.split(".")[0],
        language=lang,  # type: ignore[arg-type]
        type=kind,  # type: ignore[arg-type]
        split="dev",
        system="full",
        question="q?",
        gold=gold,
        retrieved=["L1.1", "L2.1"],
        top_score=0.7,
        threshold=0.5,
        refused=refused,
        answer="a [L1.1]. b [L1.1].",
        citations=citations,
        claims=[Claim(text=f"c{i}", citations=citations) for i in range(claims)],
        latency_ms={"generate": 100.0},
    )


def test_citation_correct_needs_an_answer_that_cites_a_gold_passage() -> None:
    assert citation_correct(rec("A.en", "plain", ["L1.1"], ["L1.1"])) is True
    assert citation_correct(rec("A.en", "plain", ["L1.1"], ["L2.1"])) is False
    assert citation_correct(rec("A.en", "plain", ["L1.1"], [], refused=True)) is False
    assert citation_correct(rec("A.en", "out_of_scope", [], [])) is None


def test_all_gold_cited_requires_every_gold_passage() -> None:
    assert all_gold_cited(rec("A.en", "multi_rule", ["L1.1", "L2.1"], ["L1.1"])) is False
    assert all_gold_cited(rec("A.en", "multi_rule", ["L1.1", "L2.1"], ["L2.1", "L1.1"])) is True
    assert all_gold_cited(rec("A.en", "out_of_scope", [], [])) is None


def test_refusal_correct_means_refusing_exactly_the_out_of_scope_questions() -> None:
    assert refusal_correct(rec("A.en", "out_of_scope", [], [], refused=True))
    assert not refusal_correct(rec("A.en", "out_of_scope", [], ["L1.1"]))
    assert refusal_correct(rec("A.en", "plain", ["L1.1"], ["L1.1"]))
    assert not refusal_correct(rec("A.en", "false_premise", ["L1.1"], [], refused=True))


def test_supported_rate_uses_judge_labels_per_claim() -> None:
    records = [rec("A.en", "plain", ["L1.1"], ["L1.1"], claims=2)]
    labels = {("A.en", 0): True, ("A.en", 1): False}
    assert supported_rate(records, labels) == pytest.approx(0.5)
    assert supported_rate(records, {}) is None
    assert supported_rate([rec("B.en", "plain", ["L1.1"], [], refused=True)], labels) is None


def test_summarize_answers_gives_rows_per_slice_with_counts() -> None:
    records = [
        rec("A.en", "plain", ["L1.1"], ["L1.1"]),
        rec("A.es", "plain", ["L1.1"], ["L2.1"], lang="es"),
        rec("B.en", "out_of_scope", [], [], refused=True),
    ]
    rows = {row.slice: row for row in summarize_answers(records, None, resamples=200, seed=0)}
    assert rows["all"].n == 3 and rows["all"].n_with_gold == 2
    assert rows["all"].citation_accuracy.mean == pytest.approx(0.5)
    assert rows["all"].refusal_accuracy.mean == pytest.approx(1.0)
    assert rows["all"].supported_claims is None
    assert rows["lang=es"].citation_accuracy.mean == pytest.approx(0.0)
    assert rows["type=out_of_scope"].citation_accuracy is None


def test_error_split_attributes_loss_between_retrieval_and_generation() -> None:
    closed = [rec("A.en", "plain", ["L1.1"], ["L2.1"]), rec("B.en", "plain", ["L2.1"], ["L2.1"])]
    full = [rec("A.en", "plain", ["L1.1"], ["L1.1"]), rec("B.en", "plain", ["L2.1"], [])]
    oracle = [rec("A.en", "plain", ["L1.1"], ["L1.1"]), rec("B.en", "plain", ["L2.1"], ["L2.1"])]
    split = error_split(closed, full, oracle, resamples=200, seed=0)
    assert split.closed_book == pytest.approx(0.5)
    assert split.full == pytest.approx(0.5)
    assert split.oracle == pytest.approx(1.0)
    assert split.retrieval_gap.mean == pytest.approx(0.5)
    assert split.generation_gap == pytest.approx(0.0)


def test_error_split_rejects_runs_over_different_questions() -> None:
    a = [rec("A.en", "plain", ["L1.1"], ["L1.1"])]
    b = [rec("B.en", "plain", ["L1.1"], ["L1.1"])]
    with pytest.raises(ValueError):
        error_split(a, b, a, resamples=10, seed=0)


def test_error_split_retrieval_value_is_full_minus_closed_book() -> None:
    closed = [rec("A.en", "plain", ["L1.1"], ["L2.1"]), rec("B.en", "plain", ["L2.1"], ["L1.1"])]
    full = [rec("A.en", "plain", ["L1.1"], ["L1.1"]), rec("B.en", "plain", ["L2.1"], ["L2.1"])]
    split = error_split(closed, full, full, resamples=200, seed=0)
    assert split.retrieval_value.mean == pytest.approx(1.0)
    assert split.retrieval_gap.mean == pytest.approx(0.0)
