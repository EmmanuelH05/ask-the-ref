import pytest

from atr.answer.generate import Claim
from atr.answer.records import AnswerRecord
from atr.eval.errors import (
    CAUSES,
    Failure,
    counts_by_language,
    format_counts,
    is_list_dominated,
    label_failures,
    pick_examples,
)
from atr.eval.metrics import QuestionScores
from atr.eval.report import Record


def record(
    qid: str, kind: str, gold: list[str], ranked: list[str], lang: str = "en", query: str = "q"
) -> Record:
    metrics = None
    if gold:
        hit = len(set(gold) & set(ranked[:5])) / len(gold)
        metrics = QuestionScores(recall_at_1=0.0, recall_at_5=hit, mrr=0.0, ndcg_at_10=0.0)
    return Record.model_validate(
        {
            "question_id": qid,
            "group_id": qid.split(".")[0],
            "language": lang,
            "type": kind,
            "split": "dev",
            "query": query,
            "gold": gold,
            "ranked": ranked,
            "scores": [0.9] * len(ranked),
            "latency_ms": {},
            "metrics": metrics,
        }
    )


def answer(
    qid: str, kind: str, gold: list[str], *, refused: bool, citations: list[str]
) -> AnswerRecord:
    return AnswerRecord(
        question_id=qid,
        group_id=qid.split(".")[0],
        language=qid.split(".")[1],  # type: ignore[arg-type]
        type=kind,  # type: ignore[arg-type]
        split="dev",
        system="full",
        question="q",
        gold=gold,
        retrieved=["L1.1"],
        top_score=0.9,
        threshold=0.5,
        refused=refused,
        answer="a",
        citations=citations,
        claims=[Claim(text="a", citations=citations)],
        latency_ms={},
    )


TEXTS = {
    "L1.1": "The field must be rectangular. The touchline is longer than the goal line.",
    "L4.1": "Equipment:\n- a shirt\n- shorts\n- socks\n- shinguards",
    "L11.1": "A player is not offside unless the ball is played by a team-mate.",
    "G.football-terms.offside": "Offside: a position.",
    "L12.1": "A direct free kick is awarded for a foul.",
    "L12.2": "An indirect free kick is awarded.",
}


def test_cause_list_is_the_fixed_eight() -> None:
    assert CAUSES == (
        "vocabulary mismatch",
        "needs two passages",
        "table or list lookup",
        "glossary term",
        "negation or exception",
        "language mismatch",
        "wrong refusal",
        "unsupported claim",
    )


def test_is_list_dominated_needs_half_the_lines_to_be_items() -> None:
    assert is_list_dominated(TEXTS["L4.1"])
    assert not is_list_dominated(TEXTS["L1.1"])
    assert not is_list_dominated("")


def test_wrong_refusal_in_both_directions_comes_first() -> None:
    records = [
        record("A.en", "plain", ["L1.1"], ["L1.1"]),
        record("B.en", "out_of_scope", [], ["L1.1"]),
    ]
    answers = {
        "A.en": answer("A.en", "plain", ["L1.1"], refused=True, citations=[]),
        "B.en": answer("B.en", "out_of_scope", [], refused=False, citations=["L1.1"]),
    }
    failures = label_failures(records, answers, {}, TEXTS)
    assert [(f.question_id, f.cause) for f in failures] == [
        ("A.en", "wrong refusal"),
        ("B.en", "wrong refusal"),
    ]


def test_unsupported_claim_when_retrieval_hit_but_the_answer_did_not_cite_gold() -> None:
    records = [record("A.en", "plain", ["L1.1"], ["L1.1"])]
    answers = {"A.en": answer("A.en", "plain", ["L1.1"], refused=False, citations=["L4.1"])}
    assert label_failures(records, answers, {}, TEXTS)[0].cause == "unsupported claim"
    good = {"A.en": answer("A.en", "plain", ["L1.1"], refused=False, citations=["L1.1"])}
    assert (
        label_failures(records, good, {("A.en", 0): False}, TEXTS)[0].cause == "unsupported claim"
    )
    assert label_failures(records, good, {("A.en", 0): True}, TEXTS) == []


def test_language_mismatch_when_the_english_sibling_succeeded() -> None:
    records = [
        record("A.en", "plain", ["L1.1"], ["L1.1"]),
        record("A.es", "plain", ["L1.1"], ["L4.1"], lang="es", query="¿Qué forma tiene el campo?"),
    ]
    failures = label_failures(records, None, None, TEXTS)
    assert [(f.question_id, f.cause) for f in failures] == [("A.es", "language mismatch")]


def test_needs_two_passages_for_a_partial_multi_rule_hit() -> None:
    records = [record("A.en", "multi_rule", ["L12.1", "L12.2"], ["L12.1", "L4.1"])]
    assert label_failures(records, None, None, TEXTS)[0].cause == "needs two passages"


def test_glossary_list_negation_and_vocabulary_in_order() -> None:
    records = [
        record("G.en", "plain", ["G.football-terms.offside"], ["L1.1"], query="what is offside"),
        record("L.en", "plain", ["L4.1"], ["L1.1"], query="what equipment do players wear"),
        record("N.en", "plain", ["L11.1"], ["L1.1"], query="when is a player offside"),
        record("V.en", "paraphrase", ["L1.1"], ["L4.1"], query="shape of the pitch"),
    ]
    causes = {f.question_id: f.cause for f in label_failures(records, None, None, TEXTS)}
    assert causes == {
        "G.en": "glossary term",
        "L.en": "table or list lookup",
        "N.en": "negation or exception",
        "V.en": "vocabulary mismatch",
    }


def test_overrides_replace_the_rule_label() -> None:
    records = [record("V.en", "paraphrase", ["L1.1"], ["L4.1"])]
    failures = label_failures(records, None, None, TEXTS, overrides={"V.en": "glossary term"})
    assert failures[0].cause == "glossary term" and failures[0].evidence.startswith("override")


def test_counts_by_language_and_their_formatting() -> None:
    failures = [
        Failure(
            question_id="A.en",
            group_id="A",
            language="en",
            type="plain",
            question="q",
            gold=["L1.1"],
            top5=[],
            overlap=0.5,
            cause="vocabulary mismatch",
            evidence="e",
        ),
        Failure(
            question_id="A.es",
            group_id="A",
            language="es",
            type="plain",
            question="q",
            gold=["L1.1"],
            top5=[],
            overlap=0.5,
            cause="vocabulary mismatch",
            evidence="e",
        ),
        Failure(
            question_id="B.es",
            group_id="B",
            language="es",
            type="plain",
            question="q",
            gold=["L1.1"],
            top5=[],
            overlap=0.5,
            cause="glossary term",
            evidence="e",
        ),
    ]
    counts = counts_by_language(failures)
    assert counts["vocabulary mismatch"] == {"en": 1, "es": 1, "cs": 0}
    assert counts["glossary term"] == {"en": 0, "es": 1, "cs": 0}
    table = format_counts(counts)
    assert "vocabulary mismatch" in table and "total" in table


def test_pick_examples_rotates_over_causes_preferring_high_overlap() -> None:
    def failure(qid: str, cause: str, overlap: float) -> Failure:
        return Failure(
            question_id=qid,
            group_id=qid,
            language="en",
            type="plain",
            question="q",
            gold=[],
            top5=[],
            overlap=overlap,
            cause=cause,
            evidence="e",
        )  # type: ignore[arg-type]

    failures = [
        failure("a", "vocabulary mismatch", 0.2),
        failure("b", "vocabulary mismatch", 0.8),
        failure("c", "glossary term", 0.5),
        failure("d", "wrong refusal", 0.1),
    ]
    picked = pick_examples(failures, n=3)
    assert [f.question_id for f in picked] == ["b", "c", "d"]
    assert len(pick_examples(failures, n=10)) == 4


def test_failure_rejects_an_unknown_cause() -> None:
    with pytest.raises(ValueError):
        Failure(
            question_id="a",
            group_id="a",
            language="en",
            type="plain",
            question="q",
            gold=[],
            top5=[],
            overlap=0.0,
            cause="bad luck",
            evidence="e",
        )  # type: ignore[arg-type]


def test_has_negation_ignores_the_article_a_but_reads_a_menos_as_a_phrase() -> None:
    from atr.eval.errors import has_negation

    assert not has_negation("What is a goal kick?")
    assert not has_negation("When does the referee award a penalty?")
    assert has_negation("a menos que el árbitro lo indique")
    assert has_negation("not unless the ball is in play")


def test_label_failures_refuses_answers_for_another_question_set() -> None:
    records = [record("g1.en", "plain", ["L1.1"], ["L2.1"])]
    with pytest.raises(ValueError, match="no answer record"):
        label_failures(records, {}, None, {"L1.1": "Size rules."})


def test_label_failures_rejects_overrides_for_unknown_questions() -> None:
    records = [record("g1.en", "plain", ["L1.1"], ["L2.1"])]
    with pytest.raises(ValueError, match="unknown question ids"):
        label_failures(records, None, None, {"L1.1": "x"}, overrides={"nope": "glossary term"})


def test_errors_main_refuses_the_barred_test_split() -> None:
    from atr.eval.errors import main

    with pytest.raises(PermissionError, match="barred"):
        main(["--split", "test"])
