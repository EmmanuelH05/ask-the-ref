import pytest

from atr.answer.generate import Claim
from atr.answer.grading import (
    Agreement,
    agreement,
    cohen_kappa,
    parse_grades,
    render_sheet,
    sample_for_grading,
)
from atr.answer.records import AnswerRecord


def rec(question_id: str, kind: str, lang: str, *, refused: bool = False) -> AnswerRecord:
    return AnswerRecord(
        question_id=question_id,
        group_id=question_id.split(".")[0],
        language=lang,  # type: ignore[arg-type]
        type=kind,  # type: ignore[arg-type]
        split="dev",
        system="full",
        question=f"question {question_id}?",
        gold=["L1.1"],
        retrieved=["L1.1"],
        top_score=0.8,
        threshold=0.5,
        refused=refused,
        answer="Claim one [L1.1]. Claim two [L1.1].",
        citations=["L1.1"],
        claims=[]
        if refused
        else [Claim(text="Claim one", citations=["L1.1"]), Claim(text="Claim two", citations=[])],
        latency_ms={},
    )


def test_sample_for_grading_spreads_over_type_and_language_cells() -> None:
    records = [
        rec(f"{kind[0].upper()}{i}.{lang}", kind, lang)
        for kind in ("plain", "scenario")
        for lang in ("en", "es", "cs")
        for i in range(5)
    ]
    chosen = sample_for_grading(records, n=12, seed=0)
    assert len(chosen) == 12
    cells = {(r.type, r.language) for r in chosen}
    assert len(cells) == 6
    assert sample_for_grading(records, n=12, seed=0) == chosen
    assert sample_for_grading(records, n=12, seed=1) != chosen


def test_sample_for_grading_skips_refusals_and_caps_at_the_available_answers() -> None:
    records = [rec("A.en", "plain", "en", refused=True), rec("B.en", "plain", "en")]
    assert [r.question_id for r in sample_for_grading(records, n=60, seed=0)] == ["B.en"]


def test_render_sheet_and_parse_grades_round_trip() -> None:
    records = [rec("A.en", "plain", "en"), rec("B.es", "scenario", "es")]
    sheet = render_sheet(records, {"L1.1": "The passage text."})
    assert "### 1. A.en" in sheet and "The passage text." in sheet
    marked = sheet.replace("- [ ] (A.en#0)", "- [y] (A.en#0)").replace(
        "- [ ] (A.en#1)", "- [n] (A.en#1)"
    )
    grades = parse_grades(marked)
    assert grades == {("A.en", 0): True, ("A.en", 1): False}


def test_parse_grades_rejects_an_unknown_mark() -> None:
    with pytest.raises(ValueError, match=r"A\.en#0"):
        parse_grades("- [maybe] (A.en#0) claim")


def test_cohen_kappa_is_one_for_identical_labels_and_zero_for_chance() -> None:
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)


def test_cohen_kappa_handles_a_single_class() -> None:
    assert cohen_kappa([True, True], [True, True]) == 1.0
    assert cohen_kappa([True, True], [False, False]) == pytest.approx(0.0)


def test_agreement_compares_only_claims_both_sides_labeled() -> None:
    human = {("A", 0): True, ("A", 1): False, ("B", 0): True}
    judge = {("A", 0): True, ("A", 1): True, ("C", 0): False}
    result = agreement(human, judge)
    assert result == Agreement(
        n=2, percent_agreement=0.5, kappa=0.0, human_supported=0.5, judge_supported=1.0
    )


def test_agreement_needs_at_least_one_shared_claim() -> None:
    with pytest.raises(ValueError):
        agreement({("A", 0): True}, {("B", 0): True})


def test_render_sheet_carries_the_answers_stamp_and_parse_stamp_reads_it() -> None:
    from atr.answer.grading import parse_stamp, unmarked_claims

    sheet = render_sheet([rec("A.en", "plain", "en")], {"L1.1": "Size."}, stamp="a" * 64)
    assert parse_stamp(sheet) == "a" * 64
    assert parse_stamp("# Hand-grading sheet\n") is None
    assert unmarked_claims(sheet) == 2
    assert unmarked_claims(sheet.replace("- [ ] (A.en#0)", "- [y] (A.en#0)")) == 1


def test_parse_grades_raises_on_a_claim_line_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="cannot be read"):
        parse_grades("-[y] (A.en#0) Claim one")
    with pytest.raises(ValueError, match="not y or n"):
        parse_grades("- [maybe] (A.en#0) Claim one")


def test_cohen_kappa_matches_the_textbook_value() -> None:
    # Observed agreement 0.75; expected by chance 0.5; kappa (0.75 - 0.5) / (1 - 0.5).
    assert cohen_kappa([True, True, True, False], [True, True, False, False]) == pytest.approx(0.5)
