import pytest

from atr.evalset.models import QuestionGroup
from atr.evalset.stats import count_table, summarize


def group(group_id: str, kind: str, gold: list[str]) -> QuestionGroup:
    return QuestionGroup.model_validate(
        {
            "group_id": group_id,
            "type": kind,
            "gold": gold,
            "en": f"Invented question {group_id}?",
            "es": f"¿Pregunta inventada {group_id}?",
            "cs": f"¿Pregunta invented {group_id}?",
            "answer": "Invented.",
            "note": "Invented.",
        }
    )


def test_summarize_known_values() -> None:
    summary = summarize([0.0, 0.5, 1.0, 0.5])
    assert (summary.n, summary.minimum, summary.median, summary.maximum) == (4, 0.0, 0.5, 1.0)
    assert summary.mean == pytest.approx(0.5)


def test_summarize_of_nothing_is_none() -> None:
    assert summarize([]) is None


def test_count_table_counts_questions_as_three_per_group() -> None:
    groups = [group("A-1", "plain", ["L1.1"]), group("A-2", "plain", ["L1.2"])]
    groups.append(group("A-3", "out_of_scope", []))
    splits = {"A-1": "dev", "A-2": "test", "A-3": "dev"}
    table = count_table(groups, splits)  # type: ignore[arg-type]
    assert table[("plain", "dev")] == 1
    assert table[("plain", "test")] == 1
    assert table[("out_of_scope", "dev")] == 1
    assert table[("out_of_scope", "heldout")] == 0
    assert sum(table.values()) == 3
