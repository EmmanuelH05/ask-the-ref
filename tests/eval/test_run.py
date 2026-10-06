import pytest

from atr.eval.run import check_split_allowed, evaluate, run_name
from atr.evalset.models import Question
from atr.retrieval.base import Hit
from atr.retrieval.translate import Identity


class FixedRetriever:
    name = "fixed"

    def search(self, query: str, k: int) -> list[Hit]:
        return [Hit("L1.1", 2.0), Hit("L1.2", 1.0)][:k]


def question(question_id: str, gold: list[str], kind: str = "plain") -> Question:
    return Question.model_validate(
        {
            "question_id": question_id,
            "group_id": question_id.split(".")[0],
            "language": "en",
            "type": kind,
            "text": "Invented question?",
            "gold": gold,
        }
    )


def test_check_split_allowed_bars_test_while_groups_are_unreviewed() -> None:
    check_split_allowed("dev", unreviewed_groups=5)
    check_split_allowed("heldout", unreviewed_groups=5)
    check_split_allowed("test", unreviewed_groups=0)
    with pytest.raises(PermissionError, match="D18"):
        check_split_allowed("test", unreviewed_groups=5)


def test_run_name_encodes_retriever_rewriter_index_and_split() -> None:
    assert run_name("bm25", translate=False, index="en", split="dev") == "bm25__en__dev"
    assert (
        run_name("dense-en", translate=True, index="en+es", split="heldout")
        == "dense-en+translate__en+es__heldout"
    )


def test_evaluate_scores_questions_with_gold_and_leaves_the_rest_unscored() -> None:
    questions = [
        question("A-1.en", ["L1.2"]),
        question("A-2.en", [], kind="out_of_scope"),
    ]
    records = evaluate(questions, FixedRetriever(), Identity(), split="dev", k=10)
    assert [r.question_id for r in records] == ["A-1.en", "A-2.en"]
    scored, unscored = records
    assert scored.ranked == ["L1.1", "L1.2"] and scored.scores == [2.0, 1.0]
    assert scored.metrics is not None and scored.metrics.mrr == pytest.approx(0.5)
    assert scored.metrics.recall_at_1 == 0.0 and scored.metrics.recall_at_5 == 1.0
    assert unscored.metrics is None and unscored.ranked == ["L1.1", "L1.2"]
    assert set(scored.latency_ms) == {"rewrite", "search"}
    assert all(value >= 0 for value in scored.latency_ms.values())
