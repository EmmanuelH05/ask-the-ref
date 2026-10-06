import pytest
from pydantic import ValidationError

from atr.evalset.models import HELD_OUT_LAWS, QuestionGroup, expand, gold_laws, is_held_out


def group(**overrides: object) -> QuestionGroup:
    fields: dict[str, object] = {
        "group_id": "A-01",
        "type": "plain",
        "gold": ["L11.1"],
        "en": "Can a zorb spin twice?",
        "es": "¿Puede un zorb girar dos veces?",
        "cs": "¿Puede un zorb spin twice?",
        "answer": "Yes.",
        "note": "Invented.",
    }
    return QuestionGroup.model_validate({**fields, **overrides})


def test_group_is_frozen_and_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        group(extra="x")
    with pytest.raises(ValidationError):
        group().en = "changed"  # type: ignore[misc]


def test_out_of_scope_must_have_no_gold() -> None:
    assert group(type="out_of_scope", gold=[]).expected == "refuse"
    with pytest.raises(ValidationError):
        group(type="out_of_scope", gold=["L1.1"])


def test_multi_rule_needs_two_gold_passages() -> None:
    assert group(type="multi_rule", gold=["L1.1", "L2.1"]).expected == "answer"
    with pytest.raises(ValidationError):
        group(type="multi_rule", gold=["L1.1"])


@pytest.mark.parametrize("kind", ["plain", "paraphrase", "scenario"])
def test_answerable_types_need_gold(kind: str) -> None:
    with pytest.raises(ValidationError):
        group(type=kind, gold=[])


def test_false_premise_may_have_gold_or_none() -> None:
    assert group(type="false_premise", gold=[]).expected == "correct_premise"
    assert group(type="false_premise", gold=["L10.1"]).expected == "correct_premise"


def test_gold_must_be_unique_and_well_formed() -> None:
    with pytest.raises(ValidationError):
        group(gold=["L1.1", "L1.1"])
    with pytest.raises(ValidationError):
        group(gold=["Law 11"])


def test_blank_text_is_rejected() -> None:
    with pytest.raises(ValidationError):
        group(cs="   ")


def test_gold_laws_reads_law_numbers_and_ignores_glossary() -> None:
    assert gold_laws(["L12.4.3", "L3.1", "G.football-terms.advantage"]) == {12, 3}


def test_is_held_out_when_any_gold_passage_is_in_a_held_out_law() -> None:
    assert HELD_OUT_LAWS == (3, 5, 16)
    assert is_held_out(group(type="multi_rule", gold=["L11.1", "L5.3"]))
    assert not is_held_out(group(gold=["L11.1"]))
    assert not is_held_out(group(type="out_of_scope", gold=[]))


def test_expand_gives_one_question_per_language() -> None:
    questions = expand(group())
    assert [(q.question_id, q.language) for q in questions] == [
        ("A-01.en", "en"),
        ("A-01.es", "es"),
        ("A-01.cs", "cs"),
    ]
    assert questions[2].text == "¿Puede un zorb spin twice?"
    assert all(q.gold == ["L11.1"] and q.type == "plain" for q in questions)
