import pytest

from atr.eval.ablation import STARTS, STEPS, Step, Variant, choose, run_chain, total_ms
from atr.eval.metrics import QuestionScores
from atr.eval.report import Record
from atr.pipeline import PipelineConfig


def records(hits: list[float], *, ms: float = 10.0) -> list[Record]:
    out = []
    for n, hit in enumerate(hits):
        out.append(
            Record.model_validate(
                {
                    "question_id": f"A-{n}.en",
                    "group_id": f"A-{n}",
                    "language": "en",
                    "type": "plain",
                    "split": "dev",
                    "query": "invented",
                    "gold": ["L1.1"],
                    "ranked": ["L1.1"],
                    "scores": [1.0],
                    "latency_ms": {"dense": ms, "rerank": ms},
                    "metrics": QuestionScores(
                        recall_at_1=hit, recall_at_5=hit, mrr=hit, ndcg_at_10=hit
                    ),
                }
            )
        )
    return out


BASE = records([0.0] * 12)
BETTER = records([1.0] * 10 + [0.0] * 2)
BEST = records([1.0] * 12)
SAME = records([0.0] * 12)
WORSE_NOISE = records([1.0] + [0.0] * 11)


def test_total_ms_sums_the_stages_of_one_question() -> None:
    assert total_ms(records([1.0], ms=7.5)[0]) == 15.0


def test_choose_keeps_nothing_when_no_interval_clears_zero() -> None:
    winner, intervals = choose(BASE, {"same": SAME, "noise": WORSE_NOISE}, 500, 0)
    assert winner is None
    assert intervals["same"].mean == 0.0 and intervals["noise"].low <= 0.0


def test_choose_keeps_the_passing_variant_with_the_higher_recall() -> None:
    winner, intervals = choose(BASE, {"better": BETTER, "best": BEST, "same": SAME}, 500, 0)
    assert winner == "best"
    assert intervals["better"].low > 0


def test_choose_breaks_a_recall_tie_by_lower_latency() -> None:
    slow, fast = records([1.0] * 12, ms=500.0), records([1.0] * 12, ms=5.0)
    assert choose(BASE, {"slow": slow, "fast": fast}, 500, 0)[0] == "fast"


def test_run_chain_applies_kept_changes_and_carries_them_forward() -> None:
    start = PipelineConfig(label="start", dense="me5-small")
    steps = [
        Step("hybrid", ("M",), (Variant("rrf", {"bm25": True}),)),
        Step("titles", ("M",), (Variant("titles", {"title_prefix": True}),)),
        Step("only-t", ("T",), (Variant("skip", {"rewrite": True}),)),
    ]

    def evaluate(config: PipelineConfig) -> list[Record]:
        return BEST if config.bm25 and not config.title_prefix else BASE

    trials, final = run_chain("M", start, steps, evaluate, resamples=500, seed=0)
    assert [(t.step, t.variant, t.kept) for t in trials] == [
        ("hybrid", "rrf", True),
        ("titles", "titles", False),
    ]
    assert final.bm25 and not final.title_prefix
    assert trials[1].base_config_id == trials[0].config.config_id
    assert trials[0].recall_at_5 == pytest.approx(1.0)
    assert trials[0].base_recall_at_5 == pytest.approx(0.0)


def test_registered_plan_matches_the_pre_registered_order() -> None:
    assert [step.name for step in STEPS] == [
        "both-editions",
        "hybrid",
        "title-prefix",
        "reranker",
        "chunking",
        "rewrite",
        "decompose",
    ]
    assert STARTS["T"].translate and STARTS["T"].dense == "bge-small-en"
    assert not STARTS["M"].translate and STARTS["M"].dense == "me5-small"
    reranker = next(step for step in STEPS if step.name == "reranker")
    assert [(v.name, v.tracks) for v in reranker.variants] == [
        ("minilm", ("T",)),
        ("bge-m3", ("T", "M")),
    ]
