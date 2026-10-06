import math

import pytest
from pydantic import ValidationError

from atr.eval.metrics import (
    METRICS,
    QuestionScores,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_question,
)

FILLER = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]


def ranked_with(positions: dict[int, str], length: int = 12) -> list[str]:
    """A ranked list of `length` filler ids with the given ids at 1-based positions."""
    out = FILLER[:length]
    for position, passage_id in positions.items():
        out[position - 1] = passage_id
    return out


# recall_at_k


def test_recall_single_gold_at_rank_1() -> None:
    assert recall_at_k(ranked_with({1: "g"}), {"g"}, 5) == pytest.approx(1.0)


def test_recall_single_gold_at_rank_5() -> None:
    assert recall_at_k(ranked_with({5: "g"}), {"g"}, 5) == pytest.approx(1.0)


def test_recall_single_gold_at_rank_6_is_zero_at_5() -> None:
    assert recall_at_k(ranked_with({6: "g"}), {"g"}, 5) == pytest.approx(0.0)


def test_recall_single_gold_at_rank_6_found_at_6() -> None:
    assert recall_at_k(ranked_with({6: "g"}), {"g"}, 6) == pytest.approx(1.0)


def test_recall_gold_not_found() -> None:
    assert recall_at_k(FILLER, {"g"}, 5) == pytest.approx(0.0)


def test_recall_two_gold_one_in_top_k() -> None:
    assert recall_at_k(ranked_with({2: "a", 7: "b"}), {"a", "b"}, 5) == pytest.approx(0.5)


def test_recall_two_gold_both_in_top_k() -> None:
    assert recall_at_k(ranked_with({2: "a", 4: "b"}), {"a", "b"}, 5) == pytest.approx(1.0)


# reciprocal_rank


def test_rr_first_gold_at_rank_1() -> None:
    assert reciprocal_rank(ranked_with({1: "g"}), {"g"}) == pytest.approx(1.0)


def test_rr_first_gold_at_rank_3() -> None:
    assert reciprocal_rank(ranked_with({3: "g"}), {"g"}) == pytest.approx(1 / 3)


def test_rr_gold_absent() -> None:
    assert reciprocal_rank(FILLER, {"g"}) == pytest.approx(0.0)


def test_rr_two_gold_earlier_one_counts() -> None:
    assert reciprocal_rank(ranked_with({2: "a", 4: "b"}), {"a", "b"}) == pytest.approx(1 / 2)


def test_rr_rank_beyond_any_cutoff_still_counts() -> None:
    assert reciprocal_rank(ranked_with({12: "g"}), {"g"}) == pytest.approx(1 / 12)


# ndcg_at_k


def test_ndcg_single_gold_at_rank_1() -> None:
    assert ndcg_at_k(ranked_with({1: "g"}), {"g"}, 10) == pytest.approx(1.0)


def test_ndcg_single_gold_at_rank_2() -> None:
    assert ndcg_at_k(ranked_with({2: "g"}), {"g"}, 10) == pytest.approx(1 / math.log2(3))


def test_ndcg_single_gold_at_rank_11_with_k_10_is_zero() -> None:
    assert ndcg_at_k(ranked_with({11: "g"}), {"g"}, 10) == pytest.approx(0.0)


def test_ndcg_two_gold_at_ranks_1_and_3() -> None:
    # DCG = 1 + 1/log2(4) = 1.5; IDCG = 1 + 1/log2(3)
    expected = (1 + 1 / 2) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked_with({1: "a", 3: "b"}), {"a", "b"}, 10) == pytest.approx(expected)


def test_ndcg_two_gold_at_ranks_2_and_3() -> None:
    # DCG = 1/log2(3) + 1/log2(4); IDCG = 1 + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / 2) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked_with({2: "a", 3: "b"}), {"a", "b"}, 10) == pytest.approx(expected)


def test_ndcg_two_gold_perfect_order_is_one() -> None:
    assert ndcg_at_k(ranked_with({1: "a", 2: "b"}), {"a", "b"}, 10) == pytest.approx(1.0)


def test_ndcg_three_gold_k_2_idcg_uses_two_positions() -> None:
    # Only "a" is in the top 2: DCG = 1; IDCG = 1 + 1/log2(3), not three terms.
    expected = 1 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked_with({1: "a", 5: "b"}), {"a", "b", "c"}, 2) == pytest.approx(expected)


def test_ndcg_three_gold_k_2_both_slots_gold_is_one() -> None:
    assert ndcg_at_k(ranked_with({1: "a", 2: "b"}), {"a", "b", "c"}, 2) == pytest.approx(1.0)


def test_ndcg_ranked_shorter_than_k() -> None:
    assert ndcg_at_k(["x1", "g"], {"g"}, 10) == pytest.approx(1 / math.log2(3))


# short and empty ranked lists


def test_recall_and_rr_with_short_list() -> None:
    assert recall_at_k(["g"], {"g"}, 5) == pytest.approx(1.0)
    assert reciprocal_rank(["x1", "g"], {"g"}) == pytest.approx(1 / 2)


def test_empty_ranked_gives_zero_everywhere() -> None:
    gold = {"g"}
    assert recall_at_k([], gold, 5) == pytest.approx(0.0)
    assert reciprocal_rank([], gold) == pytest.approx(0.0)
    assert ndcg_at_k([], gold, 10) == pytest.approx(0.0)
    scores = score_question([], gold)
    assert scores == QuestionScores(recall_at_1=0.0, recall_at_5=0.0, mrr=0.0, ndcg_at_10=0.0)


# score_question and METRICS


def test_score_question_returns_four_values_together() -> None:
    scores = score_question(ranked_with({3: "g"}), {"g"})
    assert scores.recall_at_1 == pytest.approx(0.0)
    assert scores.recall_at_5 == pytest.approx(1.0)
    assert scores.mrr == pytest.approx(1 / 3)
    assert scores.ndcg_at_10 == pytest.approx(1 / 2)  # 1 / log2(3 + 1)


def test_score_question_two_gold() -> None:
    scores = score_question(ranked_with({1: "a", 7: "b"}), {"a", "b"})
    assert scores.recall_at_1 == pytest.approx(0.5)
    assert scores.recall_at_5 == pytest.approx(0.5)
    assert scores.mrr == pytest.approx(1.0)
    # DCG = 1 + 1/log2(8) = 1 + 1/3; IDCG = 1 + 1/log2(3)
    assert scores.ndcg_at_10 == pytest.approx((1 + 1 / 3) / (1 + 1 / math.log2(3)))


def test_question_scores_is_frozen_and_forbids_extra() -> None:
    scores = score_question(["g"], {"g"})
    with pytest.raises(ValidationError):
        scores.mrr = 0.0
    with pytest.raises(ValidationError):
        QuestionScores.model_validate(
            {"recall_at_1": 1.0, "recall_at_5": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0, "extra": 1.0}
        )


def test_metrics_constant_names_the_four_fields() -> None:
    assert METRICS == ("recall_at_1", "recall_at_5", "mrr", "ndcg_at_10")
    assert set(METRICS) == set(QuestionScores.model_fields)


# errors


def test_empty_gold_raises_everywhere() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 5)
    with pytest.raises(ValueError):
        reciprocal_rank(["a"], set())
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set(), 10)
    with pytest.raises(ValueError):
        score_question(["a"], set())


def test_duplicate_ranked_ids_raise_everywhere() -> None:
    ranked = ["a", "b", "a"]
    with pytest.raises(ValueError):
        recall_at_k(ranked, {"a"}, 5)
    with pytest.raises(ValueError):
        reciprocal_rank(ranked, {"a"})
    with pytest.raises(ValueError):
        ndcg_at_k(ranked, {"a"}, 10)
    with pytest.raises(ValueError):
        score_question(ranked, {"a"})


def test_k_below_one_raises() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, 0)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a"}, 0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, -1)
