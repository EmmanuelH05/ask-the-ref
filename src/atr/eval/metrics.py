import math

from pydantic import BaseModel, ConfigDict


class QuestionScores(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recall_at_1: float
    recall_at_5: float
    mrr: float
    ndcg_at_10: float


METRICS: tuple[str, ...] = ("recall_at_1", "recall_at_5", "mrr", "ndcg_at_10")


def _check(ranked: list[str], gold: set[str], k: int | None = None) -> None:
    if not gold:
        raise ValueError("gold must not be empty")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked contains duplicate passage ids")
    if k is not None and k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    _check(ranked, gold, k)
    return len(gold.intersection(ranked[:k])) / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    _check(ranked, gold)
    for position, passage_id in enumerate(ranked, start=1):
        if passage_id in gold:
            return 1 / position
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    _check(ranked, gold, k)
    dcg = sum(
        1 / math.log2(position + 1)
        for position, passage_id in enumerate(ranked[:k], start=1)
        if passage_id in gold
    )
    # The ideal list cannot hold more gold ids than there are slots or gold ids.
    idcg = sum(1 / math.log2(position + 1) for position in range(1, min(len(gold), k) + 1))
    return dcg / idcg


def score_question(ranked: list[str], gold: set[str]) -> QuestionScores:
    return QuestionScores(
        recall_at_1=recall_at_k(ranked, gold, 1),
        recall_at_5=recall_at_k(ranked, gold, 5),
        mrr=reciprocal_rank(ranked, gold),
        ndcg_at_10=ndcg_at_k(ranked, gold, 10),
    )
