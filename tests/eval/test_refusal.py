import math

import pytest

from atr.eval.metrics import QuestionScores
from atr.eval.refusal import (
    Candidate,
    auroc,
    balanced_accuracy,
    candidates,
    curve,
    pick_threshold,
    refusal_rows,
)
from atr.eval.report import Record


def record(
    question_id: str, kind: str, scores: list[float], recall_at_5: float | None, lang: str = "en"
) -> Record:
    metrics = (
        None
        if recall_at_5 is None
        else QuestionScores(recall_at_1=0.0, recall_at_5=recall_at_5, mrr=0.5, ndcg_at_10=0.5)
    )
    return Record.model_validate(
        {
            "question_id": question_id,
            "group_id": question_id.split(".")[0],
            "language": lang,
            "type": kind,
            "split": "dev",
            "query": "q",
            "gold": [] if kind == "out_of_scope" else ["L1.1"],
            "ranked": ["L1.1", "L1.2"],
            "scores": scores,
            "latency_ms": {},
            "metrics": metrics,
        }
    )


def cand(score: float, answerable: bool, hit: bool, kind: str = "plain") -> Candidate:
    return Candidate(f"q{score}", f"q{score}", kind, "en", score, answerable, hit)


def test_auroc_is_one_when_every_positive_beats_every_negative() -> None:
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1, 0.2], [0.9, 0.8]) == 0.0


def test_auroc_counts_ties_as_half() -> None:
    assert auroc([0.5], [0.5]) == 0.5
    assert auroc([0.5, 0.9], [0.5, 0.1]) == pytest.approx(0.875)


def test_auroc_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        auroc([], [0.1])


def test_candidates_read_top_score_answerability_and_hit() -> None:
    found = candidates(
        [
            record("A.en", "plain", [0.8, 0.7], 1.0),
            record("B.en", "out_of_scope", [0.6], None),
            record("C.en", "paraphrase", [0.7], 0.0),
            record("D.en", "plain", [], 0.0),
        ]
    )
    assert [(c.score, c.answerable, c.hit) for c in found] == [
        (0.8, True, True),
        (0.6, False, False),
        (0.7, True, False),
        (-math.inf, True, False),
    ]


def test_curve_sweeps_every_distinct_score_from_answer_all_to_refuse_all() -> None:
    points = curve([cand(0.9, True, True), cand(0.5, False, False), cand(0.3, True, False)])
    assert [p.threshold for p in points] == [-math.inf, 0.3, 0.5, 0.9, math.inf]
    answer_all = points[0]
    assert answer_all.answered_share == 1.0
    assert answer_all.accuracy_when_answered == pytest.approx(1 / 3)
    assert answer_all.decision_accuracy == pytest.approx(2 / 3)
    at_half = points[2]
    assert at_half.answered_share == pytest.approx(2 / 3)
    assert at_half.accuracy_when_answered == pytest.approx(1 / 2)
    assert at_half.decision_accuracy == pytest.approx(1 / 3)
    refuse_all = points[-1]
    assert refuse_all.answered_share == 0.0 and refuse_all.accuracy_when_answered is None


def test_balanced_accuracy_averages_answer_rate_and_refusal_rate() -> None:
    members = [cand(0.9, True, True), cand(0.8, True, False), cand(0.5, False, False)]
    assert balanced_accuracy(members, 0.6) == 1.0
    assert balanced_accuracy(members, 0.85) == pytest.approx((0.5 + 1.0) / 2)
    assert balanced_accuracy(members, -math.inf) == pytest.approx(0.5)


def test_pick_threshold_maximizes_balanced_accuracy_and_breaks_ties_low() -> None:
    members = [cand(0.9, True, True), cand(0.8, True, False), cand(0.5, False, False)]
    # Every threshold in (0.5, 0.8] scores 1.0; the lowest candidate, 0.8, is chosen.
    assert pick_threshold(members) == 0.8


def test_pick_threshold_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        pick_threshold([cand(0.9, True, True)])


def test_refusal_rows_report_refused_share_and_decision_accuracy_per_slice() -> None:
    members = [
        cand(0.9, True, True, "plain"),
        cand(0.2, True, False, "false_premise"),
        cand(0.1, False, False, "out_of_scope"),
    ]
    rows = {row.slice: row for row in refusal_rows(members, 0.5)}
    assert rows["all"].n == 3 and rows["all"].refused_share == pytest.approx(2 / 3)
    assert rows["all"].decision_accuracy == pytest.approx(2 / 3)
    assert rows["type=false_premise"].refused_share == 1.0
    assert rows["type=false_premise"].decision_accuracy == 0.0
    assert rows["type=out_of_scope"].decision_accuracy == 1.0
    assert rows["lang=en"].n == 3


def test_pick_threshold_breaks_a_real_tie_toward_the_lower_threshold() -> None:
    members = [
        cand(0.9, True, True),
        cand(0.6, False, False),
        cand(0.5, True, True),
        cand(0.2, False, False),
    ]
    # Balanced accuracy is 0.75 at both 0.5 and 0.9; the lower one answers more.
    assert balanced_accuracy(members, 0.5) == pytest.approx(0.75)
    assert balanced_accuracy(members, 0.9) == pytest.approx(0.75)
    assert pick_threshold(members) == 0.5
