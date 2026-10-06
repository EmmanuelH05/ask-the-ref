import pytest

from atr.eval.metrics import QuestionScores
from atr.eval.report import Record, compare_runs, format_rows, latency_summary, summarize


def record(
    question_id: str,
    *,
    language: str = "en",
    kind: str = "plain",
    hit: float | None = 1.0,
    search_ms: float = 1.0,
) -> Record:
    metrics = (
        None
        if hit is None
        else QuestionScores(recall_at_1=hit, recall_at_5=hit, mrr=hit, ndcg_at_10=hit)
    )
    return Record.model_validate(
        {
            "question_id": question_id,
            "group_id": question_id.split(".")[0],
            "language": language,
            "type": kind,
            "split": "dev",
            "query": "invented",
            "gold": [] if hit is None else ["L1.1"],
            "ranked": ["L1.1"],
            "scores": [1.0],
            "latency_ms": {"search": search_ms},
            "metrics": metrics,
        }
    )


def test_summarize_has_all_then_types_then_languages_and_skips_unscored() -> None:
    records = [
        record("A-1.en", hit=1.0),
        record("A-1.es", language="es", hit=0.0),
        record("A-2.en", kind="paraphrase", hit=0.0),
        record("A-3.en", kind="out_of_scope", hit=None),
    ]
    rows = summarize(records, resamples=200, seed=0)
    assert [row.slice for row in rows] == [
        "all",
        "type=plain",
        "type=paraphrase",
        "lang=en",
        "lang=es",
    ]
    overall = rows[0]
    assert overall.n == 3 and overall.n_groups == 2
    assert overall.metrics["recall_at_1"].mean == pytest.approx(1 / 3)
    assert rows[1].metrics["mrr"].mean == pytest.approx(0.5)
    assert rows[4].n == 1


def test_summarize_resamples_groups_for_pooled_slices() -> None:
    records = [record("A-1.en"), record("A-1.es", language="es"), record("A-2.en", hit=0.0)]
    rows = {row.slice: row for row in summarize(records, resamples=200, seed=0)}
    assert rows["all"].metrics["mrr"].n_clusters == 2
    assert rows["lang=en"].metrics["mrr"].n_clusters == 2


def test_summarize_of_no_scored_records_raises() -> None:
    with pytest.raises(ValueError):
        summarize([record("A-1.en", hit=None)], resamples=10, seed=0)


def test_format_rows_prints_mean_interval_and_counts() -> None:
    text = format_rows(summarize([record("A-1.en"), record("A-2.en", hit=0.0)], 200, 0))
    assert "slice" in text and "recall@1" in text and "nDCG@10" in text
    assert "0.50 [0.00, 1.00]" in text
    assert text.splitlines()[1].split()[:2] == ["all", "2"]


def test_latency_summary_gives_nearest_rank_percentiles_per_stage() -> None:
    records = [record(f"A-{i}.en", search_ms=float(i)) for i in range(1, 101)]
    assert latency_summary(records) == {"search": {"p50": 50.0, "p95": 95.0, "n": 100}}


def test_compare_runs_reports_b_minus_a_and_whether_zero_is_excluded() -> None:
    a = [record(f"A-{i}.en", hit=0.0) for i in range(6)]
    b = [record(f"A-{i}.en", hit=1.0) for i in range(6)]
    rows = compare_runs(a, b, resamples=200, seed=0)
    assert rows[0].slice == "all"
    assert rows[0].metrics["recall_at_1"].mean == pytest.approx(1.0)
    assert rows[0].decided["recall_at_1"] is True
    same = compare_runs(a, a, resamples=200, seed=0)
    assert same[0].decided["mrr"] is False


def test_compare_runs_refuses_runs_over_different_questions() -> None:
    with pytest.raises(ValueError):
        compare_runs([record("A-1.en")], [record("A-2.en")], resamples=10, seed=0)
