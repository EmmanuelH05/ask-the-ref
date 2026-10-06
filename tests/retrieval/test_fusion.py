import pytest

from atr.retrieval.base import Hit
from atr.retrieval.fusion import rrf


def ranking(*ids: str, scores: tuple[float, ...] | None = None) -> list[Hit]:
    given = scores if scores is not None else tuple(float(len(ids) - i) for i in range(len(ids)))
    return [Hit(passage_id, score) for passage_id, score in zip(ids, given, strict=True)]


def test_single_ranking_keeps_order_with_reciprocal_scores() -> None:
    fused = rrf([ranking("a", "b", "c")])

    assert [h.passage_id for h in fused] == ["a", "b", "c"]
    assert [h.score for h in fused] == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_id_ranked_first_in_two_rankings_scores_double() -> None:
    fused = rrf([ranking("a", "b"), ranking("a", "c")])

    assert fused[0].passage_id == "a"
    assert fused[0].score == pytest.approx(2 / 61)


def test_id_in_only_one_ranking_still_appears() -> None:
    fused = rrf([ranking("a"), ranking("b")])

    assert {h.passage_id for h in fused} == {"a", "b"}


def test_three_ids_two_rankings_fused_order_differs_from_both_inputs() -> None:
    # a: ranks 1 and 3; b: ranks 2 and 1; c: ranks 3 and 2. Fused order is b, a, c.
    fused = rrf([ranking("a", "b", "c"), ranking("b", "c", "a")])

    assert [h.passage_id for h in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61 + 1 / 63)
    assert fused[2].score == pytest.approx(1 / 63 + 1 / 62)


def test_input_scores_are_ignored() -> None:
    small = rrf([ranking("a", "b", scores=(0.002, 0.001)), ranking("b", "a", scores=(0.9, 0.1))])
    large = rrf([ranking("a", "b", scores=(9000.0, 1.0)), ranking("b", "a", scores=(5.0, -5.0))])

    assert small == large
    assert [h.score for h in small] == pytest.approx([1 / 61 + 1 / 62] * 2)


def test_ties_in_fused_score_are_ordered_by_passage_id() -> None:
    fused = rrf([ranking("b"), ranking("a")])

    assert [h.passage_id for h in fused] == ["a", "b"]
    assert fused[0].score == fused[1].score == pytest.approx(1 / 61)


def test_custom_c_changes_scores() -> None:
    fused = rrf([ranking("a", "b")], c=10)

    assert [h.score for h in fused] == pytest.approx([1 / 11, 1 / 12])


def test_repeated_id_in_a_ranking_raises() -> None:
    with pytest.raises(ValueError):
        rrf([ranking("a", "b", "a")])


def test_c_of_zero_raises() -> None:
    with pytest.raises(ValueError):
        rrf([ranking("a")], c=0)


def test_no_rankings_returns_empty() -> None:
    assert rrf([]) == []


def test_all_empty_rankings_return_empty() -> None:
    assert rrf([[], []]) == []
